import torch.multiprocessing as mp
import os
import glob
import argparse
import yaml
import queue
import torch
import torch.nn as nn
import tqdm
import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

import accelerate
import traceback
from transformers import LlamaForCausalLM as OrigLlama
from transformers import AutoConfig
from typing import Dict, Any, List, Tuple, Optional, Callable

from src.model.llama import LlamaForCausalLM
from src.utils.utils import *
from src.data import get_loaders
from src.eval.main_fn import eval
from src.quantize_compress import LinearVQ
from src.sparse_compress import SparseLinear
from src.permute_compress import PermutedSparseLinear
from accelerate import infer_auto_device_map, dispatch_model
import numpy as np

#set to be deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def compression_worker(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    gpu_queue: mp.Queue,
    lock: "mp.synchronize.Lock",
    stop_event: "mp.synchronize.Event",
):
    """Quantizes a layer of the model

    Args:
        task_queue (mp.Queue): task queue, each element consits of a tuple of (layer_name[str],
        config[DictConfig])
        result_queue (mp.Queue): the results queue, each element is a tuple of
        (
                    layer_name[str],
                    save_path[str],
                    (n_bits[int], n_params[int]) if quantizing, (n_nonzero[int], n_params[int]) if pruning
        )
        gpu_queue (mp.Queue): a queue of the available GPU ids
        lock (mp.synchronize.Lock): a lock for accessing the GPU queue
        stop_event (mp.Event): event to stop the worker if a exception is raised
    """

    while True:
        try:
            # Get next task (non-blocking)
            layer_name, cfg = task_queue.get_nowait()
            # layer name is expected to be of the form layer_{i}/{self_attn|mlp}.{q|k|v|o|up|down|gate}_proj
        except queue.Empty:
            # No more tasks, exit
            return

        # Get an available GPU
        with lock:
            gpu_id = gpu_queue.get()

        try:
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(device)
            seed(cfg.seed, seed_all=True)

            # load the original weight
            weight = torch.load(
                os.path.join(cfg.weight_path, layer_name + ".pt"), map_location=device
            )["weight"]
            original_dtype = weight.dtype

            weight = weight.to(torch.float32)

            # create the compression module
            if cfg.compress.method == "LinearVQ":
                compression_module = LinearVQ(weight=weight, add_bias=cfg.add_bias)
            elif cfg.compress.method == "Sparse":
                compression_module = SparseLinear(weight=weight, add_bias=cfg.add_bias)
            elif cfg.compress.method == "Permute":
                compression_module = PermutedSparseLinear(
                    weight=weight, add_bias=cfg.add_bias,
                    verbose = cfg.verbose
                )
            else:
                raise ValueError(f"Unknown compression type {cfg.compress.method}")

            # load the hessian
            if hasattr(cfg, "hessian_path"):
                hessian = torch.load(
                    os.path.join(cfg.hessian_path, layer_name + ".pt"),
                    map_location=device,
                )["hessian"]
                compression_module.hessian = hessian.to(torch.float32)
            if hasattr(cfg, "hessianDiag_path"):
                hessianDiag = torch.load(
                    os.path.join(cfg.hessianDiag_path, layer_name + ".pt"),
                    map_location=device,
                )["hessianDiag"]
                compression_module.hessianDiag = hessianDiag.to(torch.float32)

            else:
                raise ValueError("hessian not found in the hessian file")
            # print("here")
            #if the state dict exists, load it
            # print(os.path.join(cfg.temp_path, layer_name + ".pt"))
            if cfg.resume_layerwise and os.path.exists(
                os.path.join(cfg.temp_path, layer_name + ".pt")
            ):
                state_dict = torch.load(
                    os.path.join(cfg.temp_path, layer_name + ".pt"),
                    map_location=device,
                )
                compression_module.blank_recreate(**cfg.compress.kwargs)
                compression_module.load_state_dict(state_dict)
                if cfg.verbose:
                    print(
                        f"Loaded compression module {layer_name} from {cfg.temp_path}",
                        flush=True,
                    )
            else:
                # raise ValueError("stop here")
                # compress the layer
                compression_module.compress(**cfg.compress.kwargs)
            compression_module.to(original_dtype)

            if cfg.verbose:
                print(
                    f"Compression module {layer_name} created, average unweighted l2 distortion: {compression_module.get_reconstruction_error()}, average weighted l2 distortion: {compression_module.get_reconstruction_error(error_weight = hessianDiag)}",
                    flush=True,
                )
            # save the state dict
            state_dict = compression_module.state_dict()

            os.makedirs(
                os.path.dirname(os.path.join(cfg.temp_path, layer_name + ".pt")),
                exist_ok=True,
            )
            torch.save(state_dict, os.path.join(cfg.temp_path, layer_name + ".pt"))

            # calculate the number of parameters
            compression_measure = ()
            if cfg.compress.method == "LinearVQ":
                compression_measure = (
                    compression_module.get_n_bits(),
                    compression_module.get_n_original_parameters(),
                )
            elif cfg.compress.method == "Sparse" or cfg.compress.method == "Permute":
                compression_measure = (
                    compression_module.get_n_nonzero(),
                    compression_module.get_n_original_parameters(),
                )
        except Exception as e:
            print(
                f"========================= Error in quantization of {layer_name} ========================="
            )
            # print the error and the traceback
            print(e)
            traceback.print_exc()
            # Set the stop event to signal the main process
            stop_event.set()
            raise e
        finally:
            # Put the GPU back in the queue
            with lock:
                gpu_queue.put(gpu_id)

            # Put the result in the result queue
            result_queue.put(
                (
                    layer_name,
                    os.path.join(cfg.temp_path, layer_name + ".pt"),
                    compression_measure,
                )
            )


@hydra.main(version_base=None, config_path="./config", config_name="compress")
def main(cfg: DictConfig):
    #log to wandb if needed
    if cfg.log_wandb:
        wandb.init(
            project="NoWag-1shot",
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.name,
        )
        
    #print the config
    print(OmegaConf.to_yaml(cfg))
    if cfg.resume and os.path.exists(cfg.save_path):
        #try to load the model
        try:
            compressed_model = LlamaForCausalLM.from_pretrained(
                os.path.join(cfg.save_path, "model"),
                torch_dtype="auto",
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            print("Loaded model from", cfg.save_path)
        except Exception as e:
            print("Error loading model from", cfg.save_path)
            # print the error and traceback
            print(traceback.format_exc())
            print(e)
            compressed_model = None
            raise e
        # raise NotImplementedError("This is a test")
    else:
        compressed_model = None
    if compressed_model is None:
        print(f"compressing model {cfg.base_model}")
        devices = torch.cuda.device_count()
        gpu_ids = list(range(devices))
        print(f"Available GPUs: {gpu_ids}")

        # Create queues for tasks and results
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        gpu_queue = mp.Queue()
        lock = mp.Lock()
        for gpu_id in gpu_ids:
            gpu_queue.put(gpu_id)

        # create our list of tasks
        weight_paths = glob.glob(os.path.join(cfg.weight_path, "*/*.pt"))
        print("n weights found", len(weight_paths))

        # create the task queue
        for weight_path in weight_paths:
            layer_name = weight_path.replace(cfg.weight_path, "").replace(".pt", "")[1:]
            task_queue.put((layer_name, cfg))

        # Create a stop event
        stop_event = mp.Event()

        # Create a pool of workers
        num_workers = min(len(gpu_ids), mp.cpu_count())
        processes = []
        print(f"Starting {num_workers} workers")
        for _ in range(num_workers):
            p = mp.Process(
                target=compression_worker,
                args=(task_queue, result_queue, gpu_queue, lock, stop_event),
            )
            p.start()
            processes.append(p)

        # Create a progress bar
        pbar = tqdm.tqdm(total=len(weight_paths), desc="Compressing layers")

        checkpoints_dict: Dict[str, str] = {}
        running_first = 0
        running_params = 0

        # Process results
        tasks_done = 0
        while tasks_done < len(weight_paths):
            try:
                layer_name, save_path, (first, n_params) = result_queue.get(timeout=1)
                # print(f"Layer {layer_name} quantized and saved to {save_path}")
                checkpoints_dict[layer_name] = save_path
                running_first += first
                running_params += n_params
                tasks_done += 1
                pbar.update(1)
            except queue.Empty:
                pass

            if stop_event.is_set():
                print("Stopping all workers due to error")
                break
        pbar.close()

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Check if any process raised an exception
        if stop_event.is_set():
            print("One or more processes raised an exception. Exiting.")
            for p in processes:
                p.terminate()
            return

        # print out the results
        print("=" * 10, "Compression Level", "=" * 10)
        print(f"Total number of parameters: {human_format(running_params)}")
        if cfg.compress.method == "LinearVQ":
            print(f"Total number of bytes: {human_format(running_first)}")
            print(f"Average bits per parameter: {round(running_first/running_params, 4)}")
        elif cfg.compress.method == "Sparse" or cfg.compress.method == "Permute":
            print(f"Total number of non-zero parameters: {human_format(running_first)}")
            print(f"Actual Pruning fraction: {round(running_first/running_params, 4)}")
        print(f"=" * 25)

        # Reload and Save in hf format

        orig_config = AutoConfig.from_pretrained(
            cfg.base_model, dtype="auto", device_map="cpu", attn_implementation="sdpa"
        )
        orig_model = OrigLlama.from_pretrained(
            cfg.base_model,
            config=orig_config,
            torch_dtype="auto",
            device_map="cpu",
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )

        compression_config = {
            "compression_kwargs": OmegaConf.to_container(cfg.compress.kwargs, resolve=True),
            "compression_type": cfg.compress.method,
            "add_bias": cfg.add_bias,
            "skip_list": None,
        }

        orig_config.compress_config = compression_config

        compressed_model = LlamaForCausalLM(orig_config)
        compressed_model.to(orig_config.torch_dtype)
        compressed_model.load_state_dict(orig_model.state_dict(), strict=False)

        del orig_model

        for layer_name, save_path in tqdm.tqdm(
            checkpoints_dict.items(), desc="Loading quantized layers"
        ):
            # now split by /
            layer_name = layer_name.split("/")[-2:]
            # from the first part, we can get which layer it is
            i_layer = int(layer_name[0].replace("layer_", ""))
            # from the second part we can get which module (self_attn, mlp, etc) and which layer it is
            submodule_name, linear_name = layer_name[1].split(".")

            # now we get the right module
            layer = getattr(
                getattr(compressed_model.model.layers[i_layer], submodule_name), linear_name
            )
            # record the original dtype
            orig_dtype = next(layer.parameters()).dtype
            orig_device = next(layer.parameters()).device
            # load the state dict
            state_dict = torch.load(save_path, map_location=orig_device)
            layer.load_state_dict(state_dict)
            layer.to(orig_dtype)
            # delete the state dict to save memory
            del state_dict
        clean()
        # save the model
        compressed_model.save_pretrained(os.path.join(cfg.save_path, "model"))

        # Move the model to an auto device map using accelerate

        device_map = infer_auto_device_map(compressed_model)
        compressed_model = dispatch_model(compressed_model, device_map=device_map)
        
        #delete all the temporary files
        # for save_path in checkpoints_dict.values():
        #     os.remove(save_path)
        

    # evaluate the models
    # first have them cache the quantized weights
    with torch.no_grad():
        recursive_apply(compressed_model, "cache_reconstruct", {"denormalize": True})

        # get the name of the model
        compressed_model.to(torch.float16)
        compressed_model.seqlen = (
            cfg.seqlen if cfg.seqlen > 0 else compressed_model.config.max_position_embeddings
        )
        print("using seqlen", compressed_model.seqlen)

        compressed_model.eval()
        if hasattr(cfg, "eval"):
            eval(compressed_model, cfg)


if __name__ == "__main__":
    main()
