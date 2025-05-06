import torch
import transformers
import yaml
import numpy as np
import sys
import os


from transformers import LlamaForCausalLM as OrigLlama
import os
from src import data
import tqdm
import torch
import argparse
import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict
import torch.nn as nn
import torch.amp as amp
from torch import autocast
from dataclasses import dataclass, field
import wandb
from src.eval.main_fn import eval
from accelerate import dispatch_model, infer_auto_device_map


from src.utils import utils
from src.model import llama
from src.utils import model_utils


@dataclass
class Config:
    batch_size: int = 32
    batch_size_val: int = (32,)
    num_epochs: int = 10
    grad_accum_steps: int = 1
    optimizer_config: Optional[OmegaConf] = field(
        default_factory=lambda: {"_target_": "torch.optim.AdamW", "lr": 1e-5}
    )
    scheduler_config: Optional[OmegaConf] = None
    loss_l: int = 2  # l1 or l2 loss
    early_stop_patience: int = -1
    clip_grad: float = 0.0
    temp_dir: str = "/tmp"


def get_loss(
    layer: llama.LlamaDecoderLayer,
    inputs: torch.FloatTensor,
    labels: torch.FloatTensor,
    train_config: Config,
    layer_kwargs: Optional[Dict] = None,
) -> torch.FloatTensor:

    with autocast(device_type="cuda", dtype=torch.float16):
        predict = layer(inputs, **layer_kwargs)[0]
        loss = torch.norm(predict - labels, p=train_config.loss_l, dim=-1).mean()
    return loss


def train_layer(
    layer: llama.LlamaDecoderLayer,
    train_inps: torch.FloatTensor,
    train_outputs: torch.FloatTensor,
    valid_inps: torch.FloatTensor,
    valid_outputs: torch.FloatTensor,
    train_config: Config,
    layer_kwargs: Optional[Dict] = None,
    prefix: str = "layer_0",
    log_wandb: bool = False,
) -> None:
    """
    Train a single layer of the Llama model
    """

    # move the layer to cuda
    layer = layer.cuda()
    # count the number of parameters to train
    n_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    tqdm.tqdm.write(f"Layer {prefix} has {n_params} parameters")
    # create the optimizer
    optimizer = instantiate(train_config.optimizer_config, params=layer.parameters())

    # create the scheduler if needed
    if train_config.scheduler_config is not None:
        scheduler = instantiate(train_config.scheduler_config, optimizer=optimizer)
    else:
        scheduler = None

    scaler = amp.GradScaler()
    patience_counter = 0

    # pre evaluate the layer
    layer.eval()
    with torch.no_grad():
        val_loss = 0
        n_iter = 0
        for i in tqdm.tqdm(
            range(0, len(valid_inps), train_config.batch_size_val),
            desc=f"Validating Epoch -1",
            leave=False,
            disable=True,
        ):
            inputs = valid_inps[i : i + train_config.batch_size_val].cuda()
            labels = valid_outputs[i : i + train_config.batch_size_val].cuda()

            # get the loss
            loss = get_loss(layer, inputs, labels, train_config, layer_kwargs)
            val_loss += loss.item()
            n_iter += 1
        tqdm.tqdm.write(f"Epoch {-1} Validation loss {val_loss / n_iter}")
        if log_wandb:
            wandb.log({f"{prefix}_valid_loss": val_loss / n_iter})

    best_loss = val_loss
    if scheduler is not None and isinstance(
        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
    ):
        scheduler.step(val_loss / n_iter)
    torch.save(
        layer.state_dict(), os.path.join(train_config.temp_dir, f"{prefix}_best.pt")
    )

    for epoch in tqdm.tqdm(
        range(train_config.num_epochs), desc=f"Training {prefix}", leave=False
    ):
        # train the layer
        layer.train()
        n_iter = 0
        overall_loss = 0
        for i in tqdm.tqdm(
            range(0, len(train_inps), train_config.batch_size),
            desc=f"Epoch {epoch}",
            leave=False,
            disable=True,
        ):
            # get the inputs and labels
            inputs = train_inps[i : i + train_config.batch_size].cuda()
            labels = train_outputs[i : i + train_config.batch_size].cuda()

            # zero the gradients
            optimizer.zero_grad()

            # get the loss
            loss = get_loss(layer, inputs, labels, train_config, layer_kwargs)

            # scale and backprop
            scaler.scale(loss / train_config.grad_accum_steps).backward()

            # step the optimizer
            if (n_iter + 1) % train_config.grad_accum_steps == 0:
                # clip the graidents if needed
                if train_config.clip_grad > 0:
                    norm = nn.utils.clip_grad_norm_(
                        layer.parameters(), train_config.clip_grad
                    )
                # otherwise just record the norm of the gradients
                else:
                    norm = nn.utils.get_total_norm(
                        [p.grad for p in layer.parameters() if p.grad is not None]
                    )
                scaler.step(optimizer)
                scaler.update()

                # log the g
                if scheduler is not None and not isinstance(
                    scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    scheduler.step()
                optimizer.zero_grad()
            # tqdm.tqdm.desc(f"Epoch {epoch} Iter {n_iter} loss {loss.item()}")
            n_iter += 1
            overall_loss += loss.item()
            if log_wandb:
                wandb.log(
                    {f"{prefix}_train_loss": loss.item(), f"{prefix}_grad_norm": norm}
                )
        tqdm.tqdm.write(f"Epoch {epoch} Overall loss {overall_loss / n_iter}")

        # validate the layer
        layer.eval()
        with torch.no_grad():
            val_loss = 0
            n_iter = 0
            for i in tqdm.tqdm(
                range(0, len(valid_inps), train_config.batch_size_val),
                desc=f"Validating Epoch {epoch}",
                leave=False,
                disable=True,
            ):
                inputs = valid_inps[i : i + train_config.batch_size_val].cuda()
                labels = valid_outputs[i : i + train_config.batch_size_val].cuda()

                # get the loss
                loss = get_loss(layer, inputs, labels, train_config, layer_kwargs)
                val_loss += loss.item()
                n_iter += 1
            tqdm.tqdm.write(f"Epoch {epoch} Validation loss {val_loss / n_iter}")
            if log_wandb:
                wandb.log({f"{prefix}_valid_loss": val_loss / n_iter})
            if scheduler is not None and isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step(val_loss / n_iter)

        # check if the loss is best
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # save the model
            torch.save(
                layer.state_dict(),
                os.path.join(train_config.temp_dir, f"{prefix}_best.pt"),
            )
            tqdm.tqdm.write(f"Saved best model at epoch {epoch}")
        else:
            patience_counter += 1
            tqdm.tqdm.write(
                f"Patience counter {patience_counter}, allowed {train_config.early_stop_patience}"
            )
            if (
                patience_counter == train_config.early_stop_patience
            ):  # this allows for negative patience to disable early stopping
                tqdm.tqdm.write(f"Early stopping at epoch {epoch}")
                break

    # load the best model
    if best_loss != val_loss:
        layer.load_state_dict(
            torch.load(os.path.join(train_config.temp_dir, f"{prefix}_best.pt"))
        )
    # delete the temp file
    os.remove(os.path.join(train_config.temp_dir, f"{prefix}_best.pt"))
    return layer.cpu()


def process_kwargs(
    kwargs: Dict,
    move_to_cuda: bool = True,
) -> Dict:

    if move_to_cuda:
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.cuda()
    else:  # assume that we want to move to cpu
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.cpu()
    return kwargs


@hydra.main(
    version_base=None, config_path="config", config_name="layer_by_layer_ft_config"
)
def main(cfg: DictConfig):
    # initiate wandb
    assert cfg.run_name!="_EMPTY_", "run_name must be set"
    if cfg.resume and os.path.exists(cfg.save_path):
        #try to load the model
        try:
            compressed_model = llama.LlamaForCausalLM.from_pretrained(
                os.path.join(cfg.save_path, "model"),
                torch_dtype="auto",
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            print("Loaded model from", cfg.save_path)
            
            seqlen = cfg.seqlen # this can be -1, in which case we switch to using the model's full sequence length


            if seqlen <= 0:
                seqlen = compressed_model.config.max_position_embeddings
                print(f"Using sequence length: {seqlen}")
                
            
        except Exception as e:
            print("Error loading model from", cfg.save_path)
            print(e)
            compressed_model = None
        # raise NotImplementedError("This is a test")
    else:
        compressed_model = None
        
    # print("Compressed model: ", compressed_model)
    # raise NotImplementedError("This is a test")
        
    if compressed_model is None:
        if cfg.log_wandb:
            wandb.init(
                project="llama_layerwise_ft",
                config=OmegaConf.to_container(cfg, resolve=True),
                name=f"{base_model.config.name_or_path}-{cfg.run_name}",
            )
        print("config:")
        print(OmegaConf.to_yaml(cfg))
        print("N_visible GPUs: ", torch.cuda.device_count())

        utils.seed(cfg.seed)
        # we expect the config to have a sub section called model with the name of the base model and the path to the quantized model
        base_model = cfg.base_model
        compressed_model_path = os.path.join(cfg.compressed_model_path,"model")
        seqlen = cfg.seqlen # this can be -1, in which case we switch to using the model's full sequence length

        base_model = OrigLlama.from_pretrained(
            base_model, device_map="cpu", torch_dtype=torch.float32
        )
        compressed_model = llama.LlamaForCausalLM.from_pretrained(
            compressed_model_path, device_map="cpu", torch_dtype=torch.float32
        )

        if seqlen <= 0:
            seqlen = base_model.config.max_position_embeddings
            print(f"Using sequence length: {seqlen}")

        # load the overall data, we expect the config to have a dataset section with the name of the dataset
        dataset_cfg = cfg.dataset
        overall_data: list[torch.FloatTensor] = data.get_loaders(
            dataset_cfg.name,
            nsamples=dataset_cfg.ft_n_train + dataset_cfg.ft_n_val,
            model=base_model.config.name_or_path,
            train_test="train",
            seqlen=seqlen,
        )

        inps = torch.zeros(
            (len(overall_data), seqlen, base_model.config.hidden_size),
            dtype=torch.float32,
            device="cpu",
        )
        cache = {"i": 0, "kwargs": None}

        use_cache = compressed_model.config.use_cache
        base_model.config.use_cache = False
        compressed_model.config.use_cache = False

        base_model.model.embed_tokens = base_model.model.embed_tokens.cuda()
        base_model.model.norm = base_model.model.norm.cuda()
        base_model.model.rotary_emb = base_model.model.rotary_emb.cuda()
        base_model.model.layers[0] = base_model.model.layers[0].cuda()

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                # print(kwargs)
                # raise Exception("stop")
                inps[cache["i"]] = inp.cpu()
                cache["i"] += 1
                cache["kwargs"] = process_kwargs(kwargs, move_to_cuda=False)
                raise ValueError

        base_model.model.layers[0] = Catcher(base_model.model.layers[0])
        with torch.no_grad():
            for batch in tqdm.tqdm(
                overall_data, desc="getting inputs", miniters=len(overall_data) // 100
            ):
                try:
                    base_model(batch[0].cuda())
                except ValueError:
                    pass
        base_model.model.layers[0] = base_model.model.layers[0].module

        base_model.model.layers[0] = base_model.model.layers[0].cpu()
        base_model.model.embed_tokens = base_model.model.embed_tokens.cpu()
        base_model.model.norm = base_model.model.norm.cpu()
        base_model.model.rotary_emb = base_model.model.rotary_emb.cpu()

        # we just need two things, the inputs and the kwargs
        del overall_data
        utils.clean()
        kwargs = process_kwargs(cache["kwargs"], move_to_cuda=True)

        train_args = instantiate({"_target_": Config}, _recursive_=False, **cfg.ft_args)
        os.makedirs(train_args.temp_dir, exist_ok=True)
        outputs = torch.zeros_like(inps)
        idxs = torch.randperm(len(inps))
        inps = inps[idxs]
        outputs = outputs[idxs]

        for i_layer in tqdm.tqdm(range(len(base_model.model.layers))):
            print(f"Training layer {i_layer}")
            # get the outputs
            with torch.no_grad():
                original_layer = base_model.model.layers[i_layer].cuda()
                outputs = model_utils.inference_layer(
                    layer=original_layer,
                    inps=inps,
                    outs=outputs,
                    layer_kwargs=kwargs,
                    dev="cuda",
                    batch_size=cfg.ft_args.batch_size_val,
                    inplace=False,
                    offload_activations=True,
                )
                original_layer.cpu()
            utils.clean()
            # train the layer
            compressed_model.model.layers[i_layer] = train_layer(
                compressed_model.model.layers[i_layer],
                inps[: dataset_cfg.ft_n_train],
                outputs[: dataset_cfg.ft_n_train],
                inps[dataset_cfg.ft_n_train :],
                outputs[dataset_cfg.ft_n_train :],
                train_args,
                kwargs,
                prefix=f"layer_{i_layer}",
                log_wandb=cfg.log_wandb,
            )
            if cfg.sequential:
                quantized_layer = compressed_model.model.layers[i_layer].cuda()
                outputs = model_utils.inference_layer(
                    layer=quantized_layer,
                    inps=inps,
                    outs=outputs,
                    layer_kwargs=kwargs,
                    dev="cuda",
                    batch_size=cfg.ft_args.batch_size_val,
                    inplace=False,
                    offload_activations=True,
                )
                quantized_layer.cpu()
            utils.clean()

            # swap the inps and outputs
            inps, outputs = outputs, inps

        # save the model
        os.makedirs(cfg.save_path, exist_ok=True)
        compressed_model.save_pretrained(os.path.join(cfg.save_path, "model"))
        
        utils.clean()
        
        #evaluate the model
        
        # Move the model to an auto device map using accelerate

        device_map = infer_auto_device_map(compressed_model)
        compressed_model = dispatch_model(compressed_model, device_map=device_map)
        # evaluate the models
        # first have them cache the quantized weights
    with torch.no_grad():

        utils.recursive_apply(compressed_model, "cache_reconstruct", {"denormalize": True})

        compressed_model.to(torch.float16)
        compressed_model.seqlen = seqlen

        compressed_model.eval()
        if hasattr(cfg, "eval"):
            eval(compressed_model, cfg)
        
        wandb.finish()


if __name__ == "__main__":
    main()
