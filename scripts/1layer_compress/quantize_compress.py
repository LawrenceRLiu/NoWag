import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import numpy as np
import os
print("pid", os.getpid())
import sys
from typing import Tuple, Optional, Union, List, Callable

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))


import src.linear_compress as lc
import src.quantizers.vector_quantizer as vq

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--load_path", type=str, default="/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians/pajama/128/layer_0/self_attn.q_proj.pt")
parser.add_argument("--save_path", type=str, default="/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/tensorized/pajama/128/layer_0/self_attn.q_proj.pt")
parser.add_argument("--device", type = str, default = "cuda",
                    help = "device to use for training")

parser.add_argument("--d", type = int, default = 4,
                    help = "subvector dimension")
parser.add_argument("--n_bits", type = float, default = 2)
parser.add_argument("--norm_order", type = int, nargs = "+", default = [0,1],)

#alignment parameters
parser.add_argument("--lr", type = float, default = 1e-3)
parser.add_argument("--lr_multiplier", type = float, default = 1,
                    help = "optional learning rate multiplier to decay for learning rate scheduler")
parser.add_argument("--n_iter", type = int, default = 100)
parser.add_argument("--clip_grad", type = float, default = 1e-1,
                    help = "optional gradient clipping if -1 no clipping is applied")
parser.add_argument("--low_bound", type = float, default = 1e-5,
                    help = "optional lower bound for the loss, stopping after reaching this value")
parser.add_argument("--patience", type = int, default = 250,
                    help = "optional patience for early stopping")
parser.add_argument("--patience_scheduler", type = int, default = 50,
                    help = "reduce learning rate on plateau scheduler patience")
parser.add_argument("--verbose", type = int, default = 1,
                    help = "print after this many iterations")
parser.add_argument("--seed", type = int, default = 1,)


args = parser.parse_args()
torch.manual_seed(args.seed)    
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
data = torch.load(args.load_path)
original_dtype = data["weight"].dtype
print("original dtype", original_dtype)
print(data.keys())

compression_module = lc.LinearQuantized(
    data["weight"].to(args.device).to(torch.float32),
)

compression_module.hessian = data["hessian"].to(args.device).to(torch.float32)

compression_module.quantize(
    vq.VectorQuantizer,
    d = args.d,
    n_bits = args.n_bits,
    n_iter = args.n_iter,
    norm_order = args.norm_order,
)
compression_module.set_additional_attributes_as_trainable()
best_loss = compression_module.align(
    None,
    lr = args.lr,
    lr_multiplier = args.lr_multiplier,
    n_iters = args.n_iter,
    clip_grad = args.clip_grad,
    low_bound = args.low_bound,
    patience = args.patience,
    patience_scheduler = args.patience_scheduler,
    verbose = args.verbose,
)

print("best_loss", best_loss)
print("n_params", compression_module.get_n_original_parameters())
print("n_bits", compression_module.get_n_bits())

# compression_module.to(original_dtype)
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
torch.save(compression_module.state_dict(), args.save_path)
#save the args


import yaml
current_filetype = args.save_path[args.save_path.rfind("."):]
args_save_path = args.save_path[:args.save_path.rfind(".")] + "_args.yaml"
# print("args_save_path", args_save_path)

#save the args as a yaml file
with open(args_save_path, "w") as f:
    args_dict = vars(args)
    #add a arg that these are quantized weights
    args_dict["compression_type"] = "quantized"
    yaml.dump(args_dict, f)
    


