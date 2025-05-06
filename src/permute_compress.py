import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import numpy as np
import os
import sys
import copy
import matplotlib.pyplot as plt

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Tuple, Optional, Union, List, Literal
from dataclasses import dataclass, field

from src.compression_parent import CompressedLinear
from src.utils import normalizer as normalize
from src.utils import beta_schedulers
from src.utils import sparse
from src.utils import utils

class PermutatedSoftSparse(nn.Module):
    
    def __init__(self, original_weight: torch.FloatTensor,
                 normalizer: normalize.Normalizer,
                 N: int,
                 M: int,
                 n_permutations0:int = 10,
                 n_permutations1:int = 10,  
                 gamma: float = -0.1,
                 xi:float = 1.1
                 ) -> None:
        """
        A permutation based sparse layer with a soft sparse mask

        Args:
            original_weight (torch.FloatTensor): the original weight of the layer
            N (int): N of N:M semi-structured sparsity
            M (int): M of N:M semi-structured sparsity
            n_permutations0 (int, optional): the number of permutations to do of 0th dimension. Defaults to 10.
            n_permutations1 (int, optional): the number of permutations to do on the . Defaults to 10.
            gamma (float, optional): softmax scaling from gamma tro xi. Defaults to -0.1.
            xi (float, optional) Defaults to 1.1.
        """
        super().__init__()

        self.weight = nn.Parameter(torch.zeros_like(original_weight), requires_grad=True)
        self.sparse_mask = nn.Parameter(torch.zeros_like(original_weight), requires_grad=True)
        self.sparse_mask.data.uniform_(-1, 1)
        
        
        self.register_buffer("permutations0", torch.stack([torch.arange(original_weight.shape[0])] + [torch.randperm(original_weight.shape[0]) for _ in range(n_permutations0-1)])) 
        self.register_buffer("permutations1", torch.stack([torch.arange(original_weight.shape[1])] + [torch.randperm(original_weight.shape[1]) for _ in range(n_permutations1-1)]))

        
        self.permutation_scales0 = nn.Parameter(torch.randn(n_permutations0, original_weight.shape[0])/n_permutations0, requires_grad=True)
        self.permutation_scales1 = nn.Parameter(torch.randn(n_permutations1, original_weight.shape[1])/n_permutations1, requires_grad=True)
        
        
        self.N = N
        self.M = M
        self.gamma = gamma
        self.xi = xi
        
        self.original_weight = original_weight
        self.normalizer = normalizer
        self.to(original_weight.device)
        
    def sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        #scaled and clipped sigmoid function
        return torch.clip((self.xi-self.gamma) * torch.sigmoid(x) + self.gamma, 0, 1)
    
    def reconstruct_weight(self, sparse_weight: torch.Tensor,
                           denormalize:bool = True) -> torch.Tensor:
        weight = torch.einsum("ijk,jk->ik",
                              sparse_weight[:, self.permutations1], #shape of (d_out, n_permutations, d_in)
                                self.permutation_scales1) #shape of (d_out, d_in)
        # print("Weight shape:", weight.shape)
        weight = torch.einsum("ijk,ij->jk",
                              weight[self.permutations0,:], #shape of (n_permutations, d_out, d_in)
                                self.permutation_scales0)
        
        if denormalize:
            weight = self.normalizer.denormalize(weight)
        return weight
    
    def forward(self, denormalize:bool = False) -> torch.Tensor:
        # Apply the sparse mask to the weight
        sparse_weight = self.weight * self.sigmoid(self.sparse_mask)
        
        return self.reconstruct_weight(sparse_weight, denormalize=denormalize)
    
    
    def binary_penalty(self, beta: float = 1.0) -> torch.Tensor:
        return torch.mean(1-torch.abs(2*self.sigmoid(self.sparse_mask)-1)**beta)
    
    def sparsity_penalty(self, l2: bool = True
                         ) -> torch.Tensor:
        # Sparsity penalty for the sparse mask
        if l2:
            return torch.sqrt(torch.mean(torch.square(torch.sum(self.sigmoid(self.sparse_mask).view(-1, self.M), dim=1) - self.N)))
        else:
            return torch.mean(torch.abs(torch.sum(self.sigmoid(self.sparse_mask).view(-1, self.M), dim=1) - self.N))
    
    @torch.no_grad() 
    def get_hard_sparse_mask(self) -> torch.Tensor:
        sparse_mask_soft = self.sigmoid(self.sparse_mask)
        
        sparse_mask_hard = torch.zeros_like(sparse_mask_soft)
        
        #for each M elements in the sparse mask, set the top N elements to 1 and the rest to 0
        idxs = torch.argsort(sparse_mask_soft.view(-1, self.M), dim=1, descending=True)[:, :self.N]
        sparse_mask_hard = sparse_mask_hard.view(-1, self.M).scatter_(1, idxs, 1)
        return sparse_mask_hard.view_as(sparse_mask_soft)
    

    def get_hard_weight(self, denormalize:bool)->torch.Tensor:
        #get the real weight matrix after applying the sparse mask and permutation matrices
        sparse_mask = self.get_hard_sparse_mask()
        sparse_weight = self.weight * sparse_mask
        return self.reconstruct_weight(sparse_weight, denormalize=denormalize)
    

@dataclass
class loss_fn_class:
    soft_compression_module: PermutatedSoftSparse 
    # beta_scheduler: beta_schedulers.LinearScheduler
    name: Literal["l1", "l2"] = "l2"
    wrt_to_normalized: bool = True,
    trainable_normalizer: bool = False
    loss_weighting: Union[torch.FloatTensor, None] = None
    lambda_sparse: float = 1.0
    lambda_binary: float = 1.0
    sparsity_reg_type: Literal["l1", "l2"] = "l1"
    hard_frac: float = 0.75
    """
    loss function for each linear layer, the following are the parameters:
    - compression_module: the compression module to be used
    - name: the name of the loss function, either 'l1' or 'l2'
    - wrt_to_normalized: if True, the loss is computed with respect to the normalized weight
    - trainable_normalizer: if True, the normalizer is trained
    - loss_weighting: the loss weighting, either 'hessianDiag', 'hessian' or a torch.FloatTensor
    """
    
    def __post_init__(self):
        # print("self.beta_scheduler:", self.beta_scheduler)
        
        self.loss_weighting = self.loss_weighting
        
        
        if self.name not in ["l1", "l2"]:
            raise ValueError("loss_fn must be either 'l1' or 'l2'")
        
        if self.sparsity_reg_type not in ["l1", "l2"]:
            raise ValueError("sparsity_reg_type must be either 'l1' or 'l2'")
        
        #get the ground truth weight we want to minimize the loss to
        with torch.no_grad():
            if self.wrt_to_normalized:
                self.ground_truth = self.soft_compression_module.normalizer.normalize(self.soft_compression_module.original_weight)
            else:
                self.ground_truth = self.soft_compression_module.original_weight
        self.ground_truth = self.ground_truth.detach()
        self.ground_truth.requires_grad = False
        
        self.recon_loss_scale = 1.0
        with torch.no_grad():
            self.recon_loss_scale = self.reconstruction_loss(recon_weight = torch.zeros_like(self.ground_truth)).detach()
        
        #if we are training wrt to the non normalized weight
        #if we do not need to train the normalizer, disable the gradient

        if not self.trainable_normalizer and not self.wrt_to_normalized:
            #for each parameter in the normalizer, disable the gradient
            for param in self.soft_compression_module.normalizer.parameters():
                param.requires_grad = False
            
        # print("self.beta_scheduler:", self.beta_scheduler)
        #otherwise, the gradient should automatically be enabled
        
        
    def reconstruction_loss(self, hard: bool = False, recon_weight:Optional[torch.FloatTensor] = None) -> torch.Tensor:
        #get the reconstruction loss
        
        #get the reconstructed weight
        if recon_weight is None:
            if hard:
                recon_weight = self.soft_compression_module.get_hard_weight(denormalize=self.wrt_to_normalized)
            else:
                recon_weight = self.soft_compression_module.forward(denormalize=self.wrt_to_normalized)
        
        #get the loss
        if self.name == "l1":
            loss = F.l1_loss(recon_weight, self.ground_truth, reduction = "none")
        elif self.name == "l2":
            loss = F.mse_loss(recon_weight, self.ground_truth, reduction = "none")
            
        #this is the element wise loss, now we need to handle the loss weighting
        if self.loss_weighting is None:
            #then just take the average and return
            loss = loss.mean()
        #if our loss weighting is a 1d tensor
        elif self.loss_weighting.ndim == 1:
            #then we need to multiply the loss by the loss weighting
            #and then take the mean
            loss = (loss * self.loss_weighting.unsqueeze(0)).mean() 
        #if our loss weighting is a 2d tensor
        elif self.loss_weighting.ndim == 2:
            #then we need to multiply the loss by the loss weighting
            #and then take the mean
            loss = torch.einsum("ij,jk,ik->", loss, self.loss_weighting, loss)/loss.numel()
        
        #otherwise raise an error
        else:
            raise ValueError("loss_weighting must be either 'hessianDiag', 'hessian' or a torch.FloatTensor")
        return loss/self.recon_loss_scale   
    
    def __call__(self,beta) -> Tuple[torch.Tensor, dict[torch.Tensor]]:
        """the overall loss function, which
        is the sum of the reconstruction loss, the binary penalty and the sparsity penalty
        """
        
        reconstruction_loss = self.reconstruction_loss(hard = np.random.rand() < self.hard_frac)
        
        #get the binary penalty
        binary_penalty = self.soft_compression_module.binary_penalty(beta=beta)   
        
        #get the sparsity penalty
        sparsity_penalty = self.soft_compression_module.sparsity_penalty(l2=self.sparsity_reg_type == "l2")
        
        #get the total loss
        total_loss = reconstruction_loss + self.lambda_sparse * sparsity_penalty + self.lambda_binary * binary_penalty
        
        return total_loss, {
            "reconstruction_loss": reconstruction_loss,
            "sparsity_penalty": sparsity_penalty,
            "binary_penalty": binary_penalty
        }
        
    def finish(self)->None:
        #if we have set the normalizer to not trainable, we need to set it back to trainable
        if not self.trainable_normalizer and not self.wrt_to_normalized:
            #for each parameter in the normalizer, enable the gradient
            for param in self.soft_compression_module.normalizer.parameters():
                param.requires_grad = True
         

class PermutedSparseLinear(CompressedLinear):
    name = "PermutedSparseLinear"
    permutations_idx0: torch.LongTensor
    permutations_idx1: torch.LongTensor
    permutations_idx1_inv: torch.LongTensor
    permutation_scales0: torch.FloatTensor
    permutation_scales1: torch.FloatTensor
    
    def permuted_sparse_(
        self,
        permutation_config: DictConfig,
        loss_fn_config: DictConfig,
        optimizer_config: DictConfig,
        beta_scheduler_config: DictConfig,
        n_iters: int = 1000,
        normalizer: Optional[normalize.Normalizer] = None,
        normalizer_kwargs: Optional[dict] = None,
        logging_path: Optional[str] = None,
        log_freq = 10,
        plotting_path: Optional[str] = None,
        
    ):
        if logging_path is not None:
            os.makedirs(os.path.dirname(logging_path), exist_ok=True)
        if plotting_path is not None:
            os.makedirs(os.path.dirname(plotting_path), exist_ok=True)
        
        normalized_weight = self.initialize_normalizer(
            normalizer=normalizer, normalizer_kwargs=normalizer_kwargs
        )
        
        soft_sparse = instantiate(
            {"_target_": PermutatedSoftSparse},
            original_weight=self.original_weight,
            normalizer=self.normalizer,
            **permutation_config,
        )
        
        #get the correct loss weighting
        if loss_fn_config.loss_weighting == "hessianDiag":
            loss_weighting = self.get_hessianDiag()
            # del loss_fn_config.loss_weighting
        elif loss_fn_config.loss_weighting == "hessian":
            loss_weighting = self.get_hessian()
        else:
            loss_weighting = None
            
        #remove loss_weighting from the config
        del loss_fn_config.loss_weighting
        beta_scheduler = instantiate(
            beta_scheduler_config,
            n_iters=n_iters,
        )
        # print("beta_scheduler:", beta_scheduler)
        
        loss_fn = instantiate(
            {"_target_": loss_fn_class},
            soft_compression_module=soft_sparse,
            # beta_scheduler=beta_scheduler,
            loss_weighting=loss_weighting,
            _recursive_=False,
            **loss_fn_config,
        )
        
        # loss_fn_class(
        #     soft_compression_module=soft_sparse,
        #     beta_scheduler=beta_scheduler,
        #     **loss_fn_config,
        # )
        
        # 
        optimizer = instantiate(
            optimizer_config,
            params=soft_sparse.parameters(),
        )
        
        
        #train the soft sparse layer
        
        #create a simple logger
        logger = {
            "iter": [],
            "loss": [],
            "reconstruction_loss": [],
            "sparsity_penalty": [],
            "binary_penalty": [],
            "hard_reconstruction_loss": [],
        }
        
        for i in range(n_iters):
            #get the loss
            loss, loss_dict = loss_fn(beta_scheduler.get())
            
            #zero the gradients
            optimizer.zero_grad()
            
            #backpropagate the loss
            loss.backward()
            
            #step the optimizer
            optimizer.step()
            
            #step the beta scheduler
            beta_scheduler.step()
            
            #log the loss
            logger["iter"].append(i)
            logger["loss"].append(loss.item())
            logger["reconstruction_loss"].append(loss_dict["reconstruction_loss"].item())
            logger["sparsity_penalty"].append(loss_dict["sparsity_penalty"].item())
            logger["binary_penalty"].append(loss_dict["binary_penalty"].item())
            
            #if we are logging
            if i%log_freq == log_freq-1 or i==0:
                with torch.no_grad():
                    hard_reconstruction_loss = loss_fn.reconstruction_loss(hard=True)
                
                if hard_reconstruction_loss <= min(logger["hard_reconstruction_loss"], default=1e10):
                    best_state_dict = copy.deepcopy(soft_sparse.state_dict()) 
                
                logger["hard_reconstruction_loss"].append(hard_reconstruction_loss.item())
                log_str = f"Iter: {i}, Loss: {loss.item()}, Reconstruction Loss: {loss_dict['reconstruction_loss'].item()}, Hard Reconstruction Loss: {hard_reconstruction_loss.item()}, Binary Penalty: {loss_dict['binary_penalty'].item()}, Sparsity Penalty: {loss_dict['sparsity_penalty'].item()}"
                if logging_path is not None:
                    with open(logging_path, "a") as f:
                        f.write(log_str + "\n")
                if self.verbose:
                    print(log_str)
                    
            #otherwise, just repeat the last value
            else:
                logger["hard_reconstruction_loss"].append(logger["hard_reconstruction_loss"][-1])
                
        # if we are plotting, we will plot out 4 things and 2x2 subplots
        if plotting_path is not None:
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            
            axs[0, 0].plot(logger["iter"], logger["loss"])
            axs[0, 0].set_title("Loss")
            axs[0, 0].set_xlabel("Iteration")
            axs[0, 0].set_ylabel("Loss")
            
            axs[0, 1].plot(logger["iter"], logger["reconstruction_loss"], label="Reconstruction Loss")
            axs[0, 1].plot(logger["iter"], logger["hard_reconstruction_loss"], label="Hard Reconstruction Loss")
            axs[0, 1].set_title("Reconstruction Loss")
            axs[0, 1].set_xlabel("Iteration")
            axs[0, 1].set_ylabel("Loss")
            axs[0, 1].legend()
            
            axs[1, 0].plot(logger["iter"], logger["sparsity_penalty"])
            axs[1, 0].set_title("Sparsity Penalty")
            axs[1, 0].set_xlabel("Iteration")
            axs[1, 0].set_ylabel("Sparsity Penalty")
            axs[1, 1].plot(logger["iter"], logger["binary_penalty"])
            axs[1, 1].set_title("Binary Penalty")
            axs[1, 1].set_xlabel("Iteration")
            axs[1, 1].set_ylabel("Binary Penalty")
            
            #set all y axis to be log
            for ax in axs.flat:
                ax.set_yscale("log")
            plt.tight_layout()

            plt.savefig(plotting_path)
        
        #load the best state dict
        soft_sparse.load_state_dict(best_state_dict)
        
        actual_mask = soft_sparse.get_hard_sparse_mask()
        
        weight = soft_sparse.weight.detach().clone()
        
        #create the sparse module
        self.sparse_module = sparse.UnstructuredSparse(
            self.out_features,
            self.in_features,
            permutation_config.N/permutation_config.M,
            self.original_weight.device,
            pattern=(permutation_config.N, permutation_config.M),
        )
        self.sparse_module.sparse(actual_mask, weight)
        
        #save the permutation indicies and the permutation scales
        self.register_buffer("permutations_idx0", soft_sparse.permutations0)
        self.register_buffer("permutations_idx1", soft_sparse.permutations1)
        self.register_buffer("permutations_idx1_inv", torch.argsort(soft_sparse.permutations1, dim=1))
        
        self.permutation_scales0 = nn.Parameter(soft_sparse.permutation_scales0.clone().detach(), requires_grad=True)
        self.permutation_scales1 = nn.Parameter(soft_sparse.permutation_scales1.clone().detach(), requires_grad=True)
        
        
    def compress(self, 
                permutation_config: DictConfig,
                loss_fn_config: DictConfig,
                optimizer_config: DictConfig,
                beta_scheduler_config: DictConfig,
                n_iters: int = 1000,
                normalizer: Optional[normalize.Normalizer] = None,
                normalizer_kwargs: Optional[dict] = None,
                logging_path: Optional[str] = None,
                log_freq = 10,
                plotting_path: Optional[str] = None,
    ):
        self.compressed = True
        return self.permuted_sparse_(
            permutation_config=permutation_config,
            loss_fn_config=loss_fn_config,
            optimizer_config=optimizer_config,
            beta_scheduler_config=beta_scheduler_config,
            n_iters=n_iters,
            normalizer=normalizer,
            normalizer_kwargs=normalizer_kwargs,
            logging_path=logging_path,
            log_freq=log_freq,
            plotting_path=plotting_path
        )
        
    
    def _no_checkpoint_forward(self, x: torch.FloatTensor):
        if self.forward_method == "reconstruct":
            if self.denormalization_method == "otf":
                y = F.linear(
                    self.normalizer.denormalize_otf_in(x),
                    self.reconstruct(denormalize=False),
                )
                y = self.normalizer.denormalize_otf_out(y) + (
                    self.bias if self.bias is not None else 0
                )
            else:
                # tqdm.tqdm.write(f"x dtype {x.dtype}, denormalize dtype {self.reconstruct(denormalize = self.denormalization_method == 'reconstruct').dtype}")
                y = F.linear(
                    x,
                    self.reconstruct(
                        denormalize=self.denormalization_method == "reconstruct"
                    ),
                    self.bias,
                )
        else:
            assert (
                self.denormalization_method == "otf"
            ), "on the fly denormalization is only supported for on the fly sparsity"
            
            #permutate, sum and then pass through the sparse module
            # print("x pre permutation:", x)
            # print("x[..., self.permutations_idx1]:", x[..., self.permutations_idx1_inv].shape)
            x = self.normalizer.denormalize_otf_in(x)
            x = torch.einsum("...ij,ij->...j", x[..., self.permutations_idx1_inv], 
                             self.permutation_scales1[torch.arange(self.permutation_scales1.shape[0]).unsqueeze(1),
                                                      self.permutations_idx1_inv])
            # print("")
            # print("x pre normalization:", x)
            # print("x post normalization:", x)
            #pass through the sparse module
            y = self.sparse_module(x)
            # print("y pre normalization:", y)
            # print("y post normalization:", y)
            # print("y post normalization:", y)
            #permutate again
            y = torch.einsum("...ij,ij->...j", y[..., self.permutations_idx0], self.permutation_scales0)
            y = self.normalizer.denormalize_otf_out(y)
            # print("y post permutation:", y)
        return y
    
    def reconstruct_(self, denormalize: bool = True) -> torch.FloatTensor:
        #reconstruct the weight
        weight = self.sparse_module.reconstruct()
        
        #permutate the weight
        weight = torch.einsum("ijk,jk->ik",
                              weight[:, self.permutations_idx1], #shape of (d_out, n_permutations, d_in)
                                self.permutation_scales1) #shape of (d_out, d_in)
        # print("Weight shape:", weight.shape)
        weight = torch.einsum("ijk,ij->jk",
                              weight[self.permutations_idx0,:], #shape of (n_permutations, d_out, d_in)
                                self.permutation_scales0)
        
        if denormalize:
            weight = self.normalizer.denormalize(weight)
            
        return weight
    
    def blank_recreate(self,
                       permutation_config: DictConfig,
                        normalizer: Optional[normalize.Normalizer] = None,
                        normalizer_kwargs: Optional[dict] = None,
                        **kwargs
                            ) -> None:
        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = normalize.Normalizer.blank_recreate(
                self.original_weight, **normalizer_kwargs
            )
            
        self.sparse_module = sparse.UnstructuredSparse(
            self.out_features,
            self.in_features,
            permutation_config.N/permutation_config.M,
            self.original_weight.device,
            pattern=(permutation_config.N, permutation_config.M),
        )

        
        #save the permutation indicies and the permutation scales
        self.register_buffer("permutations_idx0", torch.zeros(
            (permutation_config.n_permutations0, self.original_weight.shape[0]
            ), dtype=torch.long
        )
        )
        self.register_buffer("permutations_idx1", torch.zeros(
            (permutation_config.n_permutations1, self.original_weight.shape[1]
            ), dtype=torch.long
        )
        )
        self.register_buffer("permutations_idx1_inv", torch.zeros(
            (permutation_config.n_permutations1, self.original_weight.shape[1]
            ), dtype=torch.long
        )
        )
        
        self.permutation_scales0 = nn.Parameter(torch.zeros(permutation_config.n_permutations0, self.original_weight.shape[0]), requires_grad=True)
        self.permutation_scales1 = nn.Parameter(torch.zeros(permutation_config.n_permutations1, self.original_weight.shape[1]), requires_grad=True)
        self.to(self.original_weight.device)
        
    def get_additional_bits(self):
        
        additional_bits = sum([a.numel() for a in [
            self.permutation_scales0,
            self.permutation_scales1,
            self.permutations_idx0,
            self.permutations_idx1,
        ]]) * 16 # 16 bits per float or int
        return additional_bits
    
    def get_n_bits(self):
        n_bits = 0
        if self.compressed:
            n_bits = self.normalizer.get_n_bits() + self.get_additional_bits() + self.sparse_module.get_n_bits()
        return n_bits
    
    def get_n_nonzero(self):
        # if self.compressed:
        recon = self.reconstruct(denormalize=False)
        n_nonzero = torch.sum(recon != 0).item() + (self.normalizer.get_n_bits() + self.get_additional_bits())//16
        return n_nonzero
        # else:
        #     return self.original_weight.numel()
        
        
        
#testing main fn 
@hydra.main(config_path="../config/compress", config_name="permute")
def testing_main(cfg: DictConfig) -> None:
    utils.seed(0)
    device = "cuda:7"
    print("current_directory:", os.getcwd())
    weight_path = "/data/lliu/NoWAG/models/meta-llama/Llama-2-7b-hf/original_weights/layer_0/mlp.down_proj.pt"
    hessian_diag = weight_path.replace("original_weights", "hessianDiags/seed_0/pajama/128")

    
    weight = torch.load(weight_path, map_location=device)["weight"].to(torch.float32).detach()
    hessian_diag = torch.load(hessian_diag, map_location=device)["hessianDiag"].to(torch.float32 )
    
    compression_module = PermutedSparseLinear(weight, verbose=True)
    compression_module.hessianDiag = hessian_diag
    print("cfg:")
    #print out the cfg
    print(OmegaConf.to_yaml(cfg))
    # raise ValueError("stop here")
    compression_module.compress(
        **cfg
    )
    
    
    torch.set_printoptions(sci_mode=False)
    
    #run some checks:
    print("reconstructued_weight:", compression_module.reconstruct(denormalize=True))
    print("original_weight:", weight)
    
    #create a random input 
    x = torch.randn(1, weight.shape[1]).to(device)
    
    #try several different forward pass methods
    y_naive = compression_module(x)
    
    
    compression_module.forward_method = "otf"
    compression_module.denormalization_method = "otf"
    y_otf = compression_module(x)
    print("y_naive:", y_naive)
    print("y_otf:", y_otf)
    print("maximum difference:", torch.max(torch.abs(y_naive - y_otf)))
    assert torch.allclose(y_naive, y_otf, atol=1e-5), "Naive and otf forward pass do not match"
    
    #try the blank recreate method
    
    state_dict = compression_module.state_dict()    
    
    new_compression_module = PermutedSparseLinear(weight)
    
    new_compression_module.blank_recreate(
        **cfg)
    
    new_compression_module.load_state_dict(state_dict)
    
    assert torch.allclose(
        new_compression_module.reconstruct(), compression_module.reconstruct(), atol=1e-5
    ), "Weight does not match"
    
    

if __name__ == "__main__":
    testing_main()