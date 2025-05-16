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
import itertools
import hydra
import tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Tuple, Optional, Union, List, Literal
from dataclasses import dataclass, field

from src.compression_parent import CompressedLinear
from src.utils import normalizer as normalize
from src.utils import schedulers
from src.utils import sparse
from src.utils import utils

# import tqdm
from torch_linear_assignment import batch_linear_assignment, assignment_to_indices




class Mask(nn.Module):
    """A mask class that allows for both fixed and trainable mask"""
    temp_scheduler: Optional[schedulers.LinearScheduler] = None
    random_mask: bool = False
    
    def __init__(self, importances:torch.FloatTensor,
                 N:int,
                 M:int, 
                 trainable_mask:bool = False,
                 init_range: float = 1.0,
                 temp_scheduler_config: Optional[DictConfig] = None,
                 random_mask: bool = False,
                 add_noise: bool = False, #default values
                 return_hard: bool = False,
    ):
        super(Mask, self).__init__()
        self.trainable_mask = trainable_mask
        self.M, self.N = M, N
        if random_mask:
            importances = torch.rand_like(importances)
            self.random_mask = True
            
        largest_idxs = torch.sort(importances.view(-1, M), dim=-1).indices[..., -N:]
        
        if trainable_mask:
            mask = torch.ones_like(importances).view(-1, M)*(-init_range)
            self.temp_scheduler = instantiate(temp_scheduler_config)
            with torch.no_grad():
                mask[torch.arange(mask.shape[0]).view(-1, 1), largest_idxs] = init_range
            
            self.mask = nn.Parameter(mask, requires_grad=True)
            self.reshape_shape = importances.shape
            self.add_noise = add_noise
            self.return_hard = return_hard
            
        else:
            self.register_buffer("mask", torch.zeros_like(importances, dtype=torch.bool))
        
            self.mask = self.mask.view(-1, M)
            self.mask[torch.arange(self.mask.shape[0]).view(-1, 1), largest_idxs] = True

            self.mask = self.mask.view(importances.shape)
        
    def soft_forward(self, add_noise:Optional[bool] = None, return_hard:Optional[bool] = None)->torch.FloatTensor:
        if add_noise is None:
            add_noise = self.add_noise
        if return_hard is None:
            return_hard = self.return_hard
            
        if add_noise:
            logits = self.mask +  utils.gumble_noise(self.mask)
        else:
            logits = self.mask
        
        #scale by the temperature and subtract the max to prevent overflow
        logits = logits / self.temp_scheduler.get()
        logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
        
        p = torch.zeros_like(logits)
        hard_mask = torch.zeros_like(logits)
        for i in range(self.N):
            p_ = torch.exp(logits)
            p_ = (1-hard_mask) * p_
            p_ = p_ / torch.sum(p_, dim=-1, keepdim=True) 
            
            p += p_
            
            hard_mask[torch.arange(logits.shape[0]), torch.argmax(p_, dim=-1)] = 1  
        
        if return_hard:
            return ((hard_mask - p).detach() + p).view(self.reshape_shape)
        else:
            return p.view(self.reshape_shape)
        
            
        
    
    def structured_sparsity_regularization(self):
        return 0.0
        p = self.soft_forward(add_noise=False, return_hard=False)
        sums = torch.sum(p.view(-1, self.M), dim=1)
        return torch.mean((self.N - sums)**2)
    

    def forward(self,**kwargs):
        if self.trainable_mask:
            return self.soft_forward(**kwargs)
        else:
            return self.mask
        
    def step_temp(self):
        if self.temp_scheduler is not None:
            self.temp_scheduler.step()
    
    def set_mask(self, mask:torch.FloatTensor):
        if self.trainable_mask:
            raise ValueError("Mask is trainable, cannot set mask")
        else:
            self.mask = mask.view_as(self.mask)
        
            
            
            

class PermutedSparseWeight(nn.Module):
    original_weight: torch.FloatTensor
    importance_weight: Union[None, torch.FloatTensor] #shape of (d_in) if not None
    permutations_0: torch.LongTensor #shape of (n_permutations_0, d_out)
    permutations_1: torch.LongTensor #shape of (n_permutations_1, d_in)
    c_0: nn.Parameter #shape of (n_permutations_0, d_out)
    c_1: nn.Parameter #shape of (n_permutations_1, d_in)
    mask: Mask #shape of (d_out, d_in) with N of every M elements set to 1
    X: nn.Parameter #shape of (d_out, d_in)
    block_size: int
    eps: float = 1e-8
    prev_dim: int = -1
    
    
    
    def __init__(self, original_weight: torch.FloatTensor, 
                 n_permutations_0: int, 
                 n_permutations_1: int,
                 N: int,
                 M: int,
                 block_size: int,
                 mask_cfg: Optional[DictConfig] = None,
                 importance_weight: Optional[torch.FloatTensor] = None,
    ):
        
        super(PermutedSparseWeight, self).__init__()
        
        
        d_out, d_in = original_weight.shape
        
        assert d_in % block_size == 0, f"d_in = {d_in} must be divisible by block_size = {block_size}"
        assert d_out % block_size == 0, f"d_out = {d_out} must be divisible by block_size = {block_size}"
        
        self.d_out = d_out
        self.d_in = d_in
        
        if not mask_cfg.random_mask:
            #initalize each of the permutation matricies
            self.register_buffer("permutations_0",
                torch.stack(
                    [torch.arange(d_out)] +
                [
                (torch.stack([torch.randperm(block_size) for _ in range(d_out // block_size)]) + torch.arange(0, d_out, block_size).view(-1, 1)).view(-1) for _ in range(n_permutations_0 - 1)])
            )
            #shape of (n_permutations_0, d_out)
            
            self.register_buffer("permutations_1",
                torch.stack(
                    [torch.arange(d_in)] +
                [
                (torch.stack([torch.randperm(block_size) for _ in range(d_in // block_size)]) + torch.arange(0, d_in, block_size).view(-1, 1)).view(-1) for _ in range(n_permutations_1 - 1)])
            )
            #shape of (n_permutations_1, d_in)
            
            #initalize the scaling factors
            # self.c_0 = nn.Parameter(torch.ones(n_permutations_0, d_out)/n_permutations_0, requires_grad=True)
            # self.c_1 = nn.Parameter(torch.ones(n_permutations_1, d_in)/n_permutations_1, requires_grad=True)
            self.c_0 = nn.Parameter(torch.concat(
                [torch.ones(1, d_out), torch.zeros(n_permutations_0 - 1, d_out)], dim=0
            ), requires_grad=True)
            self.c_1 = nn.Parameter(torch.concat(
                [torch.ones(1, d_in), torch.zeros(n_permutations_1 - 1, d_in)], dim=0
            ), requires_grad=True)
            
            #don't change the first permutation
            self.permutations_0_greedy_order = torch.randperm(n_permutations_0 - 1) + 1
            self.permutations_1_greedy_order = torch.randperm(n_permutations_1 - 1) + 1
            
            
        else:
            #initalize each of the permutation matricies
            self.register_buffer("permutations_0",
                torch.stack(
                [
                (torch.stack([torch.randperm(block_size) for _ in range(d_out // block_size)]) + torch.arange(0, d_out, block_size).view(-1, 1)).view(-1) for _ in range(n_permutations_0 )])
            )
            #shape of (n_permutations_0, d_out)
            
            self.register_buffer("permutations_1",
                torch.stack(
                [
                (torch.stack([torch.randperm(block_size) for _ in range(d_in // block_size)]) + torch.arange(0, d_in, block_size).view(-1, 1)).view(-1) for _ in range(n_permutations_1 )])
            )
            self.c_0 = nn.Parameter(torch.concat(
                [torch.randn(n_permutations_0 , d_out)], dim=0
            ), requires_grad=True)
            self.c_1 = nn.Parameter(torch.concat(
                [torch.randn(n_permutations_1 , d_in)], dim=0
            ), requires_grad=True)
            
            self.permutations_0_greedy_order = torch.randperm(n_permutations_0)
            self.permutations_1_greedy_order = torch.randperm(n_permutations_1)
        
        # #set the first scaling factor to 1
        # self.c_0[0] = 1
        # self.c_1[0] = 1
        
        #initalize the mask
        self.mask = Mask(
            torch.abs(original_weight) if importance_weight is None else (original_weight**2 * importance_weight.unsqueeze(0)),
            N=N,
            M=M,
            **mask_cfg,
        )
        
        self.original_weight = original_weight.detach().clone()
        self.original_weight.requires_grad = False
        
        self.importance_weight = importance_weight
        if mask_cfg.random_mask:
            self.X = nn.Parameter(torch.zeros_like(original_weight.detach().clone()))
        else:
            self.X = nn.Parameter(original_weight.detach().clone())
        # self.X = nn.Parameter(torch.randn(N * (d_out * d_in)//M) * torch.std(original_weight).item())
        
        with torch.no_grad():
            self.loss_scaling = {"mean":1.0, "sum": 1.0}    
            self.loss_scaling = {"mean":self.recon_loss(reduction = "mean", zero_sub = True).item(),
                                    "sum": self.recon_loss(reduction = "sum", zero_sub = True).item()}
        
        self.to(original_weight.device)
        self.block_size = block_size

        
            
        
        
        
        
    def forward(self, permutations_to_ignore: Tuple[set[int],set[int]] = [{},{}], 
                mask_kwargs: Optional[dict]= {}):
        
        W_return = self.mask(**mask_kwargs) * self.X #shape of (d_out, d_in)
        
        # #get the masked weight
        # W_return[self.mask] =  self.X
        # print(W_return)
        # print(self.original_weight)
        #apply the permutations
        
        #first apply along the 1st dimension
        perms_1 = [i for i in range(self.permutations_1.shape[0]) if i not in permutations_to_ignore[1]]
        #if all of the permutations are to be ignored
        if len(perms_1) > 0:
            # print("here")
            # print("self.c_1[perms_1]",self.c_1[perms_1])
            W_return = torch.einsum("ijk, jk-> ik", W_return[:, self.permutations_1[perms_1]], self.c_1[perms_1])
        # print(W_return)
        #then apply along the 0th dimension
        perms_0 = [i for i in range(self.permutations_0.shape[0]) if i not in permutations_to_ignore[0]]
        
        #if all of the permutations are to be ignored
        if len(perms_0) > 0:
            W_return = torch.einsum("ijk, ij-> jk", W_return[self.permutations_0[perms_0],:], self.c_0[perms_0])
        # print(W_return)
        # raise ValueError("No permutations to apply")
        #return the weight
        return W_return
        
    @torch.no_grad()
    def greedy_optimize_permutation(self, dim: Literal[0, 1],
                                    mask_kwargs: Optional[dict] = {}):
        
        # #select a permutation based on the gradients of the scaled permutation matricies
        
        # #randomly select a permutation by the gradients
        # A = torch.zeros((self.permutations_0.shape[0],self.d_out, self.d_out), device = self.original_weight.device, requires_grad=True)
        # B = torch.zeros((self.permutations_1.shape[0], self.d_in, self.d_in), device = self.original_weight.device, requires_grad=True)
            
        # idx = torch.arange(self.d_out)
        # for i in range(self.permutations_0.shape[0]):
        #     A[i,idx, self.permutations_0[i]] = self.c_0[i]
        
        # idx = torch.arange(self.d_in)
        # for i in range(self.permutations_1.shape[0]):
        #     B[i,self.permutations_1[i],idx] = self.c_1[i]
        # with torch.set_grad_enabled(True):
                
        #     W_recon = torch.einsum("nik,kl,mlj->ij", A, self.X * self.mask(**mask_kwargs), B)
            
        #     loss = self.recon_loss(reduction = "mean", recon_weight = W_recon)
        #     loss.backward()
            
            
        #     A_grad = A.grad.clone().detach()
        #     B_grad = B.grad.clone().detach()
            
        # #use importance scores to select the permutation to optimize
        # A_importances = 1/(torch.norm(A_grad, p=1, dim = (1,2)) + self.eps)#/(torch.norm(A * A_grad,p=1,dim=(1,2))+self.eps) #shape of (n_permutations_0)
        # A_importances[0] = 0
        # # torch.norm(A_grad, p=1, dim = (1,2))/(torch.norm(A,p=1,dim=(1,2))+self.eps) #shape of (n_permutations_0)
        # B_importances = 1/(torch.norm(B_grad, p=1, dim = (1,2)) + self.eps)#/(torch.norm(B * B_grad, p=1,dim = (1,2)) + self.eps) #shape of (n_permutations_1)
        # B_importances[0] = 0
        # # torch.norm(B_grad, p=1, dim = (1,2))/(torch.norm(B,p=1,dim=(1,2))+self.eps) #shape of (n_permutations_1)
        # # print("A_importances", A_importances)
        # # print("B_importances", B_importances)
        # if self.prev_dim == 1:
        #     B_importances[self.prev_selected_permutation] = 0
        # elif self.prev_dim == 0:
        #     A_importances[self.prev_selected_permutation] = 0
        
        # importances = torch.cat([A_importances, B_importances], dim = 0) #shape of (n_permutations_0 + n_permutations_1)
        # #randomly select a permutation weighted by the importances
        # selected_permutation = torch.multinomial(importances, 1).item()
        # dim = 0 if selected_permutation < self.permutations_0.shape[0] else 1
        # selected_permutation = selected_permutation if dim == 0 else selected_permutation - self.permutations_0.shape[0]
        # print("dim", dim, "selected_permutation", selected_permutation)
        
        # self.prev_dim = dim
        # self.prev_selected_permutation = selected_permutation
        if self.mask.random_mask:
            selected_permutation = torch.randint(self.permutations_0.shape[0] + self.permutations_1.shape[0], (1,)).item()
            dim = 0 if selected_permutation < self.permutations_0.shape[0] else 1
            selected_permutation = selected_permutation if dim == 0 else selected_permutation - self.permutations_0.shape[0]
        else:
            selected_permutation = torch.randint(self.permutations_0.shape[0] + self.permutations_1.shape[0] - 2, (1,)).item()
            dim = 0 if selected_permutation < self.permutations_0.shape[0]-1 else 1
            selected_permutation = selected_permutation + 1 if dim == 0 else selected_permutation - self.permutations_0.shape[0] + 2
        print("dim", dim, "selected_permutation", selected_permutation)
        
        # selected_permutation = torch.randint(0, self.permutations_0.shape[0], (1,)).item() if dim == 0 else torch.randint(0, self.permutations_1.shape[0], (1,)).item()
        
        if dim == 0:
            # selected_permutation = self.permutations_0_greedy_order[0]
            # #circularly shift the order
            # self.permutations_0_greedy_order = torch.roll(self.permutations_0_greedy_order, -1)
            #we ignore this permutation and calculate the elementwise loss when we are only consider the other permutations
            recon_loss = self.original_weight - self.forward(permutations_to_ignore=({selected_permutation}, {}),
                                                             mask_kwargs=mask_kwargs) #this is of shape (d_out, d_in)
            

            #get the pre_permutation weight
            pre_permutation_weight = self.forward(permutations_to_ignore=({range(self.permutations_0.shape[0])}, {}),
                            mask_kwargs=mask_kwargs) #this is of shape (d_out, d_in)
            
            if self.importance_weight is not None:
                recon_loss = recon_loss * torch.sqrt(self.importance_weight.unsqueeze(0))
                pre_permutation_weight = pre_permutation_weight * torch.sqrt(self.importance_weight.unsqueeze(0))

            orig_scales = self.c_0[selected_permutation].clone().detach() #shape of (d_out)
            
        else:
            # selected_permutation = self.permutations_1_greedy_order[0]
            # #circularly shift the order
            # self.permutations_1_greedy_order = torch.roll(self.permutations_1_greedy_order, -1)
            #we ignore this permutation and calculate the elementwise loss when we are only consider the other permutations
            # recon_loss = self.recon_loss(permutations_to_ignore=({}, {selected_permutation}), reduction="none")
            recon_loss = self.original_weight -  self.forward(permutations_to_ignore=({}, {selected_permutation}),
                            mask_kwargs=mask_kwargs) #this is of shape (d_out, d_in)
            #get the pre_permutation weight
            pre_permutation_weight = self.forward(permutations_to_ignore=({}, {range(self.permutations_1.shape[0])}),
                            mask_kwargs=mask_kwargs)
            
            #transpose both of these to allow us to use the same code 
                
            recon_loss = recon_loss.transpose(0,1) #shape of (d_in, d_out)
            pre_permutation_weight = pre_permutation_weight.transpose(0,1)
            
            orig_scales = self.c_1[selected_permutation].clone().detach() #shape of (d_in)
            
        
        # print("recon_loss", recon_loss.shape)
        #from now lets denote the shapes of recon_loss and pre_permutation_weight as (d_1, d_2) where we want to find a permutation of d_1
        d_1, d_2 = recon_loss.shape
        # print("d_1, d_2", d_1, d_2)
        #for both, break it into batches
        recon_loss = recon_loss.view((-1, self.block_size, d_2)) #shape of (d_1/m, b, d_2)
        pre_permutation_weight = pre_permutation_weight.view((-1, self.block_size, d_2)) #shape of (d_1/m, b, d_2)
        
        #calculate the optimal scaling and the resulting loss 
        
        #the l2 loss is given by ||x x^Ty/(x^T x) - y||^2
        # where x is the pre_permutation_weight and y is the recon_loss
        #the second order term is  (x^T y)^2
        #the first order term is -2 (x^T y)^2
        #we will ignore the constant term
        
        #calculate the optimal scaling by first precomputing the inner product
        recon_pre_perm_inner = torch.bmm(recon_loss, pre_permutation_weight.transpose(1,2)) #shape of (d_1/m, b, b)
    
        perm_perm_inner = torch.sum(pre_permutation_weight**2, dim = -1).unsqueeze(1) #shape of (d_1/m, 1, b)
        recon_loss_inner = torch.sum(recon_loss**2, dim = -1).unsqueeze(2) #shape of (d_1/m, b, 1)
        cost = (- 1/(perm_perm_inner+self.eps)) * recon_pre_perm_inner**2 + recon_loss_inner#shape of (d_1/m, b, b)
        # print("cost finite", torch.all(torch.isfinite(cost)))
        if self.importance_weight is not None and dim == 1:
            cost = cost *  self.importance_weight.view(-1, self.block_size, 1)
        # cost = perm_perm_inner.unsqueeze(1) - 2*recon_pre_perm_inner + recon_loss_inner.unsqueeze(2) #shape of (d_1/m, b, b)  
        # print( torch.sum(((pre_permutation_weight.unsqueeze(1) - recon_loss.unsqueeze(2))**2)[i], dim = -1))
        # print(cost[i])
        #use the linear assignment to get the optimal assignments
        assignments = batch_linear_assignment(cost)
        row_ind, col_ind = assignment_to_indices(assignments)
        
        #get the optimal
        scales = (recon_pre_perm_inner/(perm_perm_inner+self.eps))[torch.arange(recon_loss.shape[0]).unsqueeze(1), row_ind, col_ind] #shape of (d_1/m, b)
        scale_mask = (col_ind == torch.arange(self.block_size, device = col_ind.device)).all(dim = 1) #shape of (d_1/m)
        
        scales[scale_mask] = orig_scales.view_as(scales)[scale_mask] #shape of (d_1/m, b)
        overall_indicies = col_ind + torch.arange(0, recon_loss.shape[0], device = col_ind.device).view(-1, 1) * self.block_size #shape of (d_1/m, b)
        # print("expected_loss", cost[i][row_ind[i], col_ind[i]])
        # print((pre_permutation_weight[i,col_ind[i]]).shape)
        # print("loss_computed", torch.sum(
        #     (pre_permutation_weight[i,col_ind[i]]* scales[i].unsqueeze(1)
        #      - recon_loss[i,:])**2, dim = 1))
        # print((recon_pre_perm_inner/(recon_recon_inner + self.eps)).shape)
        # scales = scales[torch.arange(recon_loss.shape[0]).unsqueeze(1), row_ind, col_ind] #shape of (d_1/m, b)
        # print("scales",scales[i])
        #put these back into the corresponding places
        if dim == 0:
            # print(overall_indicies)
            self.permutations_0[selected_permutation] = overall_indicies.view(-1) #shape of (d_out)
            
            self.c_0[selected_permutation] = scales.view(-1)
            
            # loss = torch.sum(self.recon_loss(reduction = "none").view(-1, self.block_size, self.d_in), dim = 2) #shape of (d_1/m, b)
            # print("loss gotten", torch.sum(loss[i], dim = 1))
            # self.permutations_0[selected_permutation] = overall_indicies.view(-1) #shape of (d_out)
            # self.c_0[selected_permutation] = scales.view(-1) #shape of (d_out)
        else:
            
            # s = self.forward(permutations_to_ignore=({}, {selected_permutation})).T.view_as(recon_loss) #shape of (d_1/m, b, d_2)
            # # print(pre_permutation_weight[torch.arange(recon_loss.shape[0]).unsqueeze(1), col_ind].shape)
            # # print(scales.unsqueeze(1).shape)
            # expected_weight = (s  + pre_permutation_weight[torch.arange(recon_loss.shape[0]).unsqueeze(1), col_ind] * scales.unsqueeze(2)) #shape of (d_1/m, b, d_2)
            # expected_weight = expected_weight.view(self.d_in,self.d_out).T
            # print("expected_weight", expected_weight[:, i*self.block_size:(i+1)*self.block_size])
            # print("expected_l2  loss", torch.sum((expected_weight - self.original_weight)**2, dim = 0)[i*self.block_size:(i+1)*self.block_size])
            # print("expected overall loss", torch.mean((expected_weight - self.original_weight)**2))
            self.permutations_1[selected_permutation] = overall_indicies.view(-1) #shape of (d_in)
            
            self.c_1[selected_permutation] = scales.view(-1) #shape of (d_in)
            
            # print("new_weight_block", self()[:, i*self.block_size:(i+1)*self.block_size])
            
            # self.c_1[selected_permutation] = scales.view(-1)
        
            # loss = torch.sum(self.recon_loss(reduction = "none").T.view(-1, self.block_size, self.d_in), dim = 2) #shape of (d_1/m, b, d_2)
            # print("loss", loss.shape)
            # print("loss gotten", loss[i])
        # assert torch.allclose(loss, optimal_costs, atol = 1e-4, rtol = 1e-4), f"losses are not close {loss} {optimal_costs}"
        
    def recon_loss(self, reduction: Literal["mean", "sum", "none"] = "mean",
                   zero_sub:bool = False, recon_weight: Optional[torch.FloatTensor] = None,
                   **kwargs):
        if recon_weight is None:
            if zero_sub:
                recon_weight = torch.zeros_like(self.original_weight) #used to get the loss scaling
            else:
                recon_weight = self(**kwargs)
        
        # print(((recon_weight - self.original_weight) ** 2).shape)
        #reconstruction loss
        recon_loss_elementwise = (recon_weight - self.original_weight) ** 2 if self.importance_weight is None else (recon_weight - self.original_weight) ** 2 * self.importance_weight.unsqueeze(0)
        # print("recon_loss_elementwise", recon_loss_elementwise.shape)
        if reduction == "mean":
            return recon_loss_elementwise.mean()/self.loss_scaling["mean"] #scale the loss by the scaling factor
        elif reduction == "sum":
            return recon_loss_elementwise.sum()/self.loss_scaling["sum"] #scale the loss by the scaling factor
        elif reduction == "none":
            return recon_loss_elementwise
    
    def regularization(self):
        if self.mask.trainable_mask:
            return self.mask.structured_sparsity_regularization()
        return 0.0
    
    # @torch.no_grad()
    def greedy_optimize_mask(self, 
                            n_to_optimize: int = 1
    ):
        #assert that the mask is not trainable
        assert self.mask.trainable_mask == False, "Mask is trainable, cannot optimize"
        
        #create the A and B matricies
        with torch.no_grad():
            A = torch.zeros((self.d_out, self.d_out), device = self.original_weight.device)
            B = torch.zeros((self.d_in, self.d_in), device = self.original_weight.device)
            
            idx = torch.arange(self.d_out)
            for i in range(self.permutations_0.shape[0]):
                A[idx, self.permutations_0[i]] += self.c_0[i]
            
            idx = torch.arange(self.d_in)
            for i in range(self.permutations_1.shape[0]):
                B[self.permutations_1[i],idx] += self.c_1[i]
            
        sparse_masked_values = (self.X * self.mask()).detach().clone()
        #get the gradient of the loss wrt to the sparse masked values
        with torch.set_grad_enabled(True):
            sparse_masked_values.requires_grad = True
            W_recon = A @ sparse_masked_values @ B #shape of (d_out, d_in)
            assert torch.allclose(W_recon, self(), atol = 1e-4, rtol = 1e-4), f"W_recon {W_recon} {self()}"
            # print("W_recon", W_recon)
            loss = self.recon_loss(reduction = "mean", recon_weight = W_recon)
            loss.backward()
        with torch.no_grad():
            grads = sparse_masked_values.grad #shape of (d_out, d_in)
            grads_unmasked = grads * self.mask() #shape of (d_out, d_in)
            grads_masked = grads * (~self.mask()) #shape of (d_out, d_in)
        
            non_zero_importances = torch.abs(grads_unmasked  * sparse_masked_values.detach())
            
            relative_importances = torch.norm(grads_masked.view(-1, self.mask.M), p=1, dim = 1) / (torch.norm(non_zero_importances.view(-1, self.mask.M), p=1, dim = 1) + self.eps) #shape of (d_out)
            # relative_importances = torch.norm(sparse_masked_values.view(-1, self.mask.M), p=1, dim = 1)
            group_idxs = torch.argsort(relative_importances, dim = 0, descending = True) #shape of (d_out)
            # print("relavtive_importances", relative_importances[group_idxs])
            # A_norms = torch.norm(A, p=2, dim = 1) #shape of (d_out)
            # A_norm_idxs 
            n_groups = self.d_out * self.d_in//self.mask.M
            possible_unmasked_ids = torch.tensor([list(p) for p in itertools.combinations(range(self.mask.M), self.mask.N)],dtype = torch.long, device = self.original_weight.device)
            # print("possible_unmasked_ids", possible_unmasked_ids)
            # n_optimized = 0
            for g_i in range(n_to_optimize):
                #randomly select a group
                group_idx = group_idxs[g_i]
                #get the group indicies
                i = group_idx // (self.d_in//self.mask.M)
                j = group_idx % (self.d_in//self.mask.M) * self.mask.M
                # print("group_idx", group_idx, "i", i, "j", j)   
                a = A[:,i].clone()
                a_squared = torch.norm(a, p=2)**2
                # print("a_squared", a_squared)
                # if a_squared <= self.eps:
                #     #pick a different group
                #     continue
                
                B_prime = B[j:j+self.mask.M,:].clone() #shape of (M, d_in)
                mask = self.mask().clone().detach().view(-1, self.mask.M)
                current_mask = mask[group_idx].clone()
                # print("torch.where(current_mask)", torch.where(current_mask))
                # print(possible_unmasked_ids == torch.where(current_mask)[0])
                current_unmasked_ids = torch.where(torch.all(possible_unmasked_ids == torch.where(current_mask)[0], dim = 1))
                # raise ValueError("stop here")
                # current_mask_possible_id = torch.where(torch.all(possible_unmasked_ids == torch.where(current_mask), dim = 1))[0]
                # print(torch.where(current_mask))
                # raise ValueError("stop here")
                old_values = self.X[i, j:j+self.mask.M][current_mask].clone()   
                # print("current_mask", mask[group_idx])
                # print("dense values", self.X[i, j:j+self.mask.M][mask[group_idx]])
                #zero out the entire group
                mask[group_idx] = False 
                # mask = mask.view_as(self.X)
                assert torch.all(mask.view_as(self.X)[i, j:j+self.mask.M] == False)
                W_recon_rest = A @ (self.X * mask.view_as(self.X))
                
                # print(((self.X[i, j:j+self.mask.M] * current_mask).unsqueeze(0) @ B_prime).shape)
                # t_0 = (self.X[i, j:j+self.mask.M] * current_mask).unsqueeze(0) @ B_prime
                # t = W_recon_rest + a.unsqueeze(1) @ (self.X[i, j:j+self.mask.M] * current_mask).unsqueeze(0) @ B_prime
                # print("max diff", torch.max(torch.abs(t - self())))
                # print("max relative diff", torch.max(torch.abs(t - self())/self())) 
                # assert torch.allclose(self(), t, atol = 1e-4, rtol = 1e-4), f"recon {self()} other recon {t}"
                W_residue = self.original_weight - W_recon_rest
                
                #if we are scaling by importance weight, multiply both the B and residue by the importance weight
                if self.importance_weight is not None:
                    # print("scaling")
                    B_prime *= torch.sqrt(self.importance_weight.unsqueeze(0))
                    W_residue *= torch.sqrt(self.importance_weight.unsqueeze(0))
                    
                #multply residue by a
                target = a @ W_residue #shape of (d_in)
                # print("target", target)
                #multiply B by a_squared
                # B_prime *= a_squared
                #use least squares to find the optimal values and residue for each combination of weights
                # print("torch.tile(target, (len(possible_unmasked_ids), 1)).unsqueeze(-1).shape", torch.tile(target, (len(possible_unmasked_ids), 1)).unsqueeze(-1).shape)
                sols, residues,*_ = torch.linalg.lstsq(B_prime[possible_unmasked_ids].permute(0,2,1), #shape of (M choose N, d_in, N),
                                                    torch.tile(target, (len(possible_unmasked_ids), 1)).unsqueeze(-1) #shape of (M choose N, d_in, 1)
                )
                # print("sols", sols.shape)
                # print("residues", residues)
                #squeeze the sols 
                sols = sols.squeeze(-1)/a_squared #shape of (M choose N, d_in)
                # print(B_prime.shape)
                # sols[current_unmasked_ids] = self.X[i, j:j+self.mask.M][current_mask].clone() #shape of (M choose N, d_in)
                # W_recon_group = torch.einsum("i, bd,bdj->bij", a, sols, B_prime[possible_unmasked_ids]) #shape of (M choose N, d_out, d_in)
                # # print("W_recon_group", W_recon_group.shape)
                # errors = torch.mean( (W_residue.unsqueeze(0)- W_recon_group)**2, (1,2))/self.loss_scaling["mean"] #shape of (M choose N)
                # print("errors", errors) 
                #get the index of the minimum residue
                min_idx = torch.argmin(residues)
                
                best_unmasked = possible_unmasked_ids[min_idx] #shape of (N)
                mask[group_idx, best_unmasked] = True 
                #update the mask
                self.mask.set_mask(mask.view_as(self.X))
                # assert torch.all(self.mask() == mask.view_as(self.X))
                #update the values
                # print("best_unmasked", best_unmasked)
                # print("j+best_unmasked", j+best_unmasked)
                # print("sols[min_idx]", sols[min_idx])   
                self.X[i, j + best_unmasked] = sols[min_idx]
                if torch.any(mask[group_idx] != current_mask):
                    print("previous mask", current_mask)
                    print("new mask", mask[group_idx])  
                    print("old values", old_values)
                    print("new values", self.X[i, j:j+self.mask.M][mask[group_idx]])
                    print("new_recon_loss", self.recon_loss(reduction = "mean", mask_kwargs={"return_hard": True}).item())

                # print(self.X[i, j:j+self.mask.M] * mask[group_idx])
                # t_0_ = (self.X[i, j:j+self.mask.M] * mask[group_idx]).unsqueeze(0)
                # assert torch.allclose(t_0, t_0_, atol = 1e-4, rtol = 1e-4), f"max diff {torch.max(torch.abs(t_0-t_0_))} {t_0} {t_0_}"
                # t = W_recon_rest + a.unsqueeze(1) @ (self.X[i, j:j+self.mask.M] * mask[group_idx]).unsqueeze(0) @ B_prime
                # print("max diff", torch.max(torch.abs(t - self())))
                # print("max relative diff", torch.max(torch.abs(t - self())/self())) 
                # assert torch.allclose(self(), t, atol = 1e-4, rtol = 1e-4), f"recon {self()} other recon {t}"

                # n_optimized += 1
                # break
                # raise ValueError("stop here")
                # print("="*10)
                # raise ValueError("stop here")
            
        
            
            
            
        
        
        
        
        
        
        
        


        
@dataclass
class TrainingConfig:
    n_iters: int = 100
    n_warmup_optimizer_steps: int = -1
    n_optimizer_steps_per_iter: int = 10
    n_permutation_changes_per_iter: int = 1
    greedy_optimize_steps_per_iter: int = 10
    temp_step_every: Literal["overall", "optimizer"] = "overall"
    regularization_weight: float = 0.01
    permutation_acceptance_rtol: float = 1e-3
    early_stop_atol: float = 1e-6
    early_stop_rtol: float = 1e-5
    loss_atol: float = 1e-6
    loss_rtol: float = 1e-5
    overall_patience: int = 10
    optimizer_patience: int = 5
    logging_path: Optional[str] = None
    plotting_path: Optional[str] = None
    log_freq: int = 1
    mask_kwargs: Optional[dict] = None
        
         

class PermutedSparseLinear(CompressedLinear):
    name = "PermutedSparseLinear"
    permutations_idx0: torch.LongTensor
    permutations_idx1: torch.LongTensor
    permutations_idx1_inv: torch.LongTensor
    permutation_scales0: torch.FloatTensor
    permutation_scales1: torch.FloatTensor
    low_rank: bool = False
    
    def permuted_sparse_(
        self,
        permutation_config: DictConfig,
        optimizer_config: DictConfig,
        training_config: DictConfig,
        normalizer: Optional[normalize.Normalizer] = None,
        normalizer_kwargs: Optional[dict] = None,
        
    ):
        
        training_config = TrainingConfig(**training_config)
        
        if training_config.logging_path is not None:
            os.makedirs(os.path.dirname(training_config.logging_path), exist_ok=True)
        if training_config.plotting_path is not None:
            os.makedirs(os.path.dirname(training_config.plotting_path), exist_ok=True)
        
        normalized_weight = self.initialize_normalizer(
            normalizer=normalizer, normalizer_kwargs=normalizer_kwargs
        )
        
    
        
        #get the correct loss weighting
        if permutation_config.importance_weight == "hessianDiag":
            importance_weight = self.get_hessianDiag()
            # del loss_fn_config.loss_weighting
        elif permutation_config.importance_weight == "hessian":
            raise ValueError("hessian weighting not implemented yet")
        else:
            importance_weight = None
            
        #remove loss_weighting from the config
        del permutation_config.importance_weight
        
        trainable_sparse = PermutedSparseWeight(
            normalized_weight,
            importance_weight=importance_weight,
            **permutation_config,
        )
        
        optimizer = instantiate(
            optimizer_config,
            params=trainable_sparse.parameters(),
        )
        
        
        #train the soft sparse layer
        
        #create a simple logger
        initial_loss = trainable_sparse.recon_loss(reduction="mean", mask_kwargs={"add_noise": False, "return_hard": True})
        if self.verbose:
            print(f"Initial loss: {initial_loss.item()}")
        l = [initial_loss.item()]
        c_0_norms = []
        c_1_norms = []
        remaining_patience = training_config.overall_patience
        for i in range(training_config.n_iters):
            
            c_0_norms.append(torch.norm(trainable_sparse.c_0, p=1, dim = 1).detach().cpu().numpy()/ trainable_sparse.d_out)
            c_1_norms.append(torch.norm(trainable_sparse.c_1, p=1, dim = 1).detach().cpu().numpy()/ trainable_sparse.d_in)
            remaining_optimizer_patience = training_config.optimizer_patience
            for j in tqdm.tqdm(range(training_config.n_warmup_optimizer_steps if i==0 and training_config.n_warmup_optimizer_steps > 0 else training_config.n_optimizer_steps_per_iter), disable = not self.verbose): 
                optimizer.zero_grad()
                recon_loss = trainable_sparse.recon_loss(reduction="mean") 
                loss = recon_loss + training_config.regularization_weight * trainable_sparse.regularization()
                loss.backward()
                optimizer.step()
                l.append(recon_loss.item())
                if len(l) > 2:
                    if np.allclose(
                        l[-1], l[-2], rtol=training_config.loss_rtol, atol=training_config.loss_atol):
                        remaining_optimizer_patience -= 1
                        if remaining_optimizer_patience == 0:
                            if self.verbose:
                                print("Optimizer converged, stopping early")
                            break
                    else:
                        remaining_optimizer_patience = training_config.optimizer_patience
                
                if training_config.temp_step_every == "optimizer":  
                    trainable_sparse.mask.step_temp()
                        # if self.verbose:
                        #     print("Loss converged, stopping early")
                        # break
            # if n_adam_steps_per_iter > 0:
            #     print(f"loss after {n_adam_steps_per_iter} steps = {loss.item()}")
            if training_config.temp_step_every == "overall":
                trainable_sparse.mask.step_temp()
            if training_config.n_optimizer_steps_per_iter == 0:
                loss = trainable_sparse.recon_loss(reduction="mean", mask_kwargs=training_config.mask_kwargs)
                
            with torch.no_grad():
                 for j in range(training_config.n_permutation_changes_per_iter):
                    prev_state_dict = copy.deepcopy(trainable_sparse.state_dict()) 
                    #randomly select a dimension to optimize
                    permutation_dim = (j+(training_config.n_permutation_changes_per_iter)*i) % 2
                    # permutation_dim = 0|
                    # print(f"Greedy optimizing permutation {permutation_dim}")

                    prev_loss = trainable_sparse.recon_loss(reduction="mean", mask_kwargs=training_config.mask_kwargs)
                    # print("new_loss", new_loss.item())        
                    trainable_sparse.greedy_optimize_permutation(permutation_dim,
                                                                 training_config.mask_kwargs)
                    new_loss = trainable_sparse.recon_loss(reduction="mean", mask_kwargs=training_config.mask_kwargs)
                    # print("new_loss", new_loss.item())        
                    # assert new_loss <= prev_loss + 1e-8, f"Loss should decrease but is {new_loss} > {prev_loss}"
                    # print(f"prev_loss = {prev_loss.item()}, new_loss = {new_loss.item()}")
                    if new_loss/prev_loss -1 > training_config.permutation_acceptance_rtol:
                        # print("Loss increased, reverting to previous state")
                        trainable_sparse.load_state_dict(prev_state_dict)
                        assert trainable_sparse.recon_loss(reduction="mean", mask_kwargs=training_config.mask_kwargs) == prev_loss, f"Loss should be the same but is {trainable_sparse.recon_loss()} != {prev_loss}"
                        # P.load_state_dict(prev_state_dict)
                        # print("prev_loss", prev_loss.item())
                        # print("new_loss", new_loss.item())
                    else:
                        l.append(new_loss.item())
                        # l_real.append(trainable_sparse.recon_loss(mask_kwargs={"return_hard": True}).item())
                        
            if training_config.greedy_optimize_steps_per_iter > 0:
                with torch.no_grad():   
                    prev_state_dict = copy.deepcopy(trainable_sparse.state_dict()) 
                    #randomly select a dimension to optimize
                    permutation_dim = 0 if torch.randint(0, 2, (1,)).item() == 0 else 1
                    # permutation_dim = 0|
                    # print(f"Greedy optimizing permutation {permutation_dim}")

                    prev_loss = trainable_sparse.recon_loss(reduction="mean", mask_kwargs=training_config.mask_kwargs)
                    # print("new_loss", new_loss.item())        
                    # trainable_sparse.greedy_optimize_permutation(permutation_dim,
                    #                                              training_config.mask_kwargs)
                    trainable_sparse.greedy_optimize_mask(
                        n_to_optimize=training_config.greedy_optimize_steps_per_iter
                    )
                    
                    new_loss = trainable_sparse.recon_loss(reduction="mean", mask_kwargs=training_config.mask_kwargs)
                    # print("new_loss", new_loss.item())        
                    # assert new_loss <= prev_loss + 1e-8, f"Loss should decrease but is {new_loss} > {prev_loss}"
                    print(f"prev_loss = {prev_loss.item()}, new_loss = {new_loss.item()}")
                    # if new_loss/prev_loss -1 > training_config.permutation_acceptance_rtol:
                    #     print("Loss increased, reverting to previous state")
                    #     trainable_sparse.load_state_dict(prev_state_dict)
                    #     assert trainable_sparse.recon_loss(reduction="mean", mask_kwargs=training_config.mask_kwargs) == prev_loss, f"Loss should be the same but is {trainable_sparse.recon_loss()} != {prev_loss}"
                    #     # P.load_state_dict(prev_state_dict)
                    #     # print("prev_loss", prev_loss.item())
                    #     # print("new_loss", new_loss.item())
                    # else:
                    l.append(new_loss.item())
                    # l_real.append(trainable_sparse.recon_loss(mask_kwargs={"return_hard": True}).item())
                    # raise ValueError("stop here")
                
            if np.isclose(
                loss.item(), new_loss.item(), rtol = training_config.loss_rtol, atol=training_config.loss_atol):
                remaining_patience -= 1

                if remaining_patience == 0:
                    if self.verbose:
                        print("Loss converged, stopping early")
                    break
            else:
                remaining_patience = training_config.overall_patience
            # )
                
            if len(l)%training_config.log_freq == training_config.log_freq-1 or i==0:
                log_str = f"Iter: {len(l)}, Loss: {loss.item()}, hard_loss: {trainable_sparse.recon_loss(mask_kwargs={"return_hard": True}).mean().item()}"
                if training_config.logging_path is not None:
                    with open(training_config.logging_path, "a") as f:
                        f.write(log_str + "\n")
                if self.verbose:
                    print(log_str)
        if self.verbose:
            print("Finished training") 
            
        c_0_norms.append(torch.norm(trainable_sparse.c_0, p=1, dim = 1).detach().cpu().numpy()/ trainable_sparse.d_out)
        c_1_norms.append(torch.norm(trainable_sparse.c_1, p=1, dim = 1).detach().cpu().numpy()/ trainable_sparse.d_in) 
                
        # if we are plotting, we will plot out 4 things and 2x2 subplots
        if training_config.plotting_path is not None:
            fig = plt.figure(figsize=(10, 10))
            axs = fig.subplots(1,1)
            plt.sca(axs)
            plt.plot(l, label="Loss")
            plt.legend()
            plt.title("Loss")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.yscale("log")
            plt.savefig(training_config.plotting_path)
            plt.close(fig)
            
            
            c_0_norms = np.array(c_0_norms).T
            c_1_norms = np.array(c_1_norms).T
            print("final c_0 norms", c_0_norms[:,-1])
            print("final c_1 norms", c_1_norms[:,-1])
            fig, axs = plt.subplots(2,1, figsize=(10, 10))
            plt.sca(axs[0])
            for i in range(c_0_norms.shape[0]):
                plt.plot(c_0_norms[i], label=f"Permutation {i}")
            plt.legend()
            plt.title("Permutation 0 norms")
            plt.xlabel("Iteration")
            plt.ylabel("Norm")
            plt.yscale("log")
            
            plt.sca(axs[1])
            for i in range(c_1_norms.shape[0]):
                plt.plot(c_1_norms[i], label=f"Permutation {i}")
            plt.legend()
            plt.title("Permutation 1 norms")
            plt.xlabel("Iteration")
            plt.ylabel("Norm")
            plt.yscale("log")
            plt.savefig(training_config.plotting_path.replace(".png", "_permutation_norms.png"))
            plt.close(fig)

        
        
        #create the sparse module
        self.sparse_module = sparse.UnstructuredSparse(
            self.out_features,
            self.in_features,
            permutation_config.N/permutation_config.M,
            self.original_weight.device,
            pattern=(permutation_config.N, permutation_config.M),
        )
        
        p = trainable_sparse.mask(return_hard=False, add_noise=False)
        self.sparse_module.sparse(p, trainable_sparse.X)
        # self.sparse_module.set_mask_and_values(trainable_sparse.mask, trainable_sparse.X)
        
        #save the permutation indicies and the permutation scales
        self.register_buffer("permutations_idx0", trainable_sparse.permutations_0)
        self.register_buffer("permutations_idx1", trainable_sparse.permutations_1)
        self.register_buffer("permutations_idx1_inv", torch.argsort(trainable_sparse.permutations_1, dim=1))
        
        self.permutation_scales0 = nn.Parameter(trainable_sparse.c_0.clone().detach(), requires_grad=True)
        self.permutation_scales1 = nn.Parameter(trainable_sparse.c_1.clone().detach(), requires_grad=True)
        
        # if trainable_sparse.low_rank:
        #     raise ValueError("Low rank not implemented yet")
        #     self.A = nn.Parameter(trainable_sparse.A.clone().detach(), requires_grad=True)
        #     self.B = nn.Parameter(trainable_sparse.B.clone().detach(), requires_grad=True)
        #     self.low_rank = True
        # else:
        #     self.low_rank = False
        
    def compress(self, 
                permutation_config: DictConfig,
                optimizer_config: DictConfig,
                training_config: DictConfig,
                normalizer: Optional[normalize.Normalizer] = None,
                normalizer_kwargs: Optional[dict] = None,
    ):
        self.compressed = True
        return self.permuted_sparse_(
            permutation_config=permutation_config,
            optimizer_config=optimizer_config,
            training_config=training_config,
            normalizer=normalizer,
            normalizer_kwargs=normalizer_kwargs,
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
            if self.low_rank:
                y_low_rank = F.linear(F.linear(x, self.B), self.A)
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
            if self.low_rank:
                y += y_low_rank
            # print("y post permutation:", y)
        # print("y post permutation:", y)
        # raise ValueError("stop here")
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
            
        if self.low_rank:
            weight += self.A @ self.B
            
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
            (permutation_config.n_permutations_0, self.original_weight.shape[0]
            ), dtype=torch.long
        )
        )
        self.register_buffer("permutations_idx1", torch.zeros(
            (permutation_config.n_permutations_1, self.original_weight.shape[1]
            ), dtype=torch.long
        )
        )
        self.register_buffer("permutations_idx1_inv", torch.zeros(
            (permutation_config.n_permutations_1, self.original_weight.shape[1]
            ), dtype=torch.long
        )
        )
        
        self.permutation_scales0 = nn.Parameter(torch.zeros(permutation_config.n_permutations_0, self.original_weight.shape[0]), requires_grad=True)
        self.permutation_scales1 = nn.Parameter(torch.zeros(permutation_config.n_permutations_1, self.original_weight.shape[1]), requires_grad=True)
        
        # if permutation_config.low_rank > 0:
        #     raise ValueError("Low rank not implemented yet")
        #     self.A = nn.Parameter(torch.randn(self.original_weight.shape[0], permutation_config.low_rank), requires_grad=True)
        #     self.B = nn.Parameter(torch.zeros(permutation_config.low_rank, self.original_weight.shape[1]), requires_grad=True)
        #     self.low_rank = True
        # else:
        #     self.low_rank = False
            
            
        self.to(self.original_weight.device)
        
    
        
        self.compressed = True
        
    def get_additional_bits(self):
        
        additional_bits = sum([a.numel() for a in [
            self.permutation_scales0,
            self.permutation_scales1,
            self.permutations_idx0,
            self.permutations_idx1,
        ]]) * 16 # 16 bits per float or int
        
        if self.low_rank:
            additional_bits += sum([a.numel() for a in [
                self.A,
                self.B,
            ]]) * 16
        return additional_bits
    
    def get_n_bits(self):
        n_bits = 0
        if self.compressed:
            n_bits = self.normalizer.get_n_bits() + self.get_additional_bits() + self.sparse_module.get_n_bits()
        return n_bits
    
    def get_n_nonzero(self):
        # if self.compressed:
        recon = self.sparse_module.reconstruct()
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
    weight_path = "/data/lliu/NoWAG/models/meta-llama/Llama-2-7b-hf/original_weights/layer_0/self_attn.q_proj.pt"
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
        **cfg.kwargs
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
        **cfg.kwargs)
    
    new_compression_module.load_state_dict(state_dict)
    
    assert torch.allclose(
        new_compression_module.reconstruct(), compression_module.reconstruct(), atol=1e-5
    ), "Weight does not match"
    
    y_blank_recreate = new_compression_module(x)
    assert torch.allclose(y_naive, y_blank_recreate, atol=1e-5), "Naive and blank recreate forward pass do not match"
    
    y_orig = F.linear(x, weight)
    print("y_orig:", y_orig)
    
    

if __name__ == "__main__":
    testing_main()