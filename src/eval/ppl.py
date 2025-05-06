import os

print("pid", os.getpid())

CUDA_LAUNCH_BLOCKING = 1
import time

import torch
import random
import torch.nn as nn
from typing import List, Optional, Tuple, Union, Literal

# from vector_quantizer import *
import tqdm

# from quant import *
import random
import numpy as np


from src.utils.model_utils import find_layers, get_llama, inference_layer
from src.utils.quantized_model import load_model_from_checkpoints
import src.data as data
import src.utils.utils as utils
import yaml
import transformers.models.llama.modeling_llama as llama

try:
    import wandb

    has_wandb = True
except:
    has_wandb = False


# two ways to do the evaluation, layer by layer
# or all at once


@torch.no_grad()
def ppl_llama_eval_layer_by_layer(
    model,
    testenc,
    dev,
    dataset: str,
    log_wandb: bool = False,
    offload_activations: bool = False,
    batch_size: int = 8,
    base_model: str = "llama",
    results_log_path: str = None,
    disable_tqdm: bool = False,
) -> float:
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    print("nsamples", nsamples)
    print("testenc.numel()", testenc.numel())
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    model.model.norm = model.model.norm.to(dev)

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=dev if not offload_activations else "cpu",
    )
    cache = {"i": 0, "kwargs": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp if not offload_activations else inp.cpu()
            cache["i"] += 1
            cache["kwargs"] = kwargs
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass

    kwargs = cache["kwargs"]
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    # print("inps", inps)
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    for name in kwargs:
        if isinstance(kwargs[name], torch.Tensor):
            kwargs[name] = kwargs[name].to(dev)

    for i in tqdm.tqdm(range(len(layers)), desc="Inference", disable=disable_tqdm):
        # print(i)
        layer = layers[i].to(dev)
        outs = inference_layer(
            layer,
            inps,
            outs,
            kwargs,
            dev,
            offload_activations,
            batch_size,
            disable_tqdm=True,
        )
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps
        # break

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = (
            inps[i].unsqueeze(0)
            if not offload_activations
            else inps[i].unsqueeze(0).to(dev)
        )
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        # print(lm_logits)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        # print(shift_labels)
        # raise Exception("stop")
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        # print("loss", loss)
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
        print({f"/perplexity/{base_model}/{dataset}": ppl.item()})
        wandb.log({f"/perplexity/{base_model}/{dataset}": ppl.item()})
    if results_log_path is not None:
        # assume that it is a yaml file
        # if it exits
        if os.path.exists(results_log_path):
            results = yaml.load(open(results_log_path, "r"), Loader=yaml.FullLoader)
        else:
            results = {}

        # if we don't have a ppl key, we create it
        if "ppl" not in results:
            results["ppl"] = {}
        results["ppl"][dataset] = ppl.item()

        with open(results_log_path, "w") as f:
            yaml.dump(results, f)

    model.config.use_cache = use_cache
    return ppl.item()


@torch.no_grad()
def ppl_eval_basic(
    model,
    testenc,
    dataset_name: str,
    log_wandb: bool = False,
    results_log_yaml: str = None,
) -> float:
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    nlls = []
    for i in tqdm.tqdm(range(nsamples)):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].cuda()
        outs = model(batch)
        lm_logits = outs["logits"]
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][
            :, 1:
        ].cuda()
        # print(shift_labels)
        # raise Exception("stop")
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        # print("loss", loss)
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"{dataset_name} Perplexity: {ppl.item():3f}")
    if log_wandb:
        wandb.log({f"perplexity/{dataset_name}": ppl.item()})
    if results_log_yaml is not None:
        # assume that it is a yaml file
        # if it exits
        if os.path.exists(results_log_yaml):
            results = yaml.load(open(results_log_yaml, "r"), Loader=yaml.FullLoader)
        else:
            results = {}

        # if we don't have a ppl key, we create it
        if "ppl" not in results:
            results["ppl"] = {}
        results["ppl"][dataset_name] = ppl.item()

        with open(results_log_yaml, "w") as f:
            yaml.dump(results, f)

    return ppl.item()
