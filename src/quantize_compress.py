import os
import sys
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import numpy as np
from typing import Tuple, Optional, Union, List, Literal
import src.utils.sparse as sparse_utils
import src.compression_parent as compression_parent
import src.utils.quantizer as quantizer_utils
import src.utils.utils as utils 

from torch import jit
import warnings
import tqdm
import time


# @jit.script
def weighted_kmeans_assign(data:torch.FloatTensor,
                           weights:torch.FloatTensor,
                           centroids:torch.FloatTensor, 
                        #    batch_size:int = -1,
                           verbose:int = False)->Tuple[torch.FloatTensor, float]:
    """
    Assigns each data point to the nearest centroid using weighted distances,
    calculated in batches to avoid memory overflow.
    
    Args:
        data (torch.Tensor): Input data of shape (n, d)
        weights (torch.Tensor): Weights for each data point of shape (n, d)
        centroids (torch.Tensor): Current centroids of shape (n_clusters, d)
        
    Returns:
        torch.Tensor: Cluster assignments of shape (n,)
    """
    n = data.shape[0]
    n_clusters = centroids.shape[0]
    # assignments = torch.empty(n, dtype=torch.long, device=data.device)
    loss = 0.0
    try:

        #distances (x-c)\diag(w) (x-c) = x^T w x - 2x^T w c + c^T w c
        distances = torch.norm(data * torch.sqrt(weights),dim=1).unsqueeze(1)**2 \
            + weights @ (centroids **2).T \
                - 2*(data*weights) @ (centroids).T
        
        min_distances, assignments = torch.min(distances, dim=1)
        loss = torch.sum(min_distances).item()
    except torch.OutOfMemoryError:
        #try to divided it into 2 batches
        assignments = torch.empty(n, dtype=torch.long, device=data.device)
        loss = 0.0
        
        assignments[:n//2], loss1 = weighted_kmeans_assign(data[:n//2], weights[:n//2], centroids, verbose = verbose)
        assignments[n//2:], loss2 = weighted_kmeans_assign(data[n//2:], weights[n//2:], centroids, verbose = verbose)
        loss = loss1 + loss2
        # except RuntimeError 
    # for start in range(0, n, batch_size):
    #     end = min(start + batch_size, n)
    #     data_batch = data[start:end]  # Shape: (batch_size, d)
    #     weights_batch = weights[start:end]  # Shape: (batch_size, d)
        
    #     # Calculate weighted distances for the current batch
    #     # print((data_batch.unsqueeze(1) - centroids).shape)
    #     distances = torch.sum(weights_batch.unsqueeze(1) * (data_batch.unsqueeze(1) - centroids)**2, dim=2)
    #     # Assign each point to the closest centroid
    #     assignments[start:end] = torch.argmin(distances, dim=1)
    #     loss += torch.min(distances, dim=1)[0].sum().item()
    
    return assignments, loss


# @jit.script
def weighted_kmeans_update(data:torch.FloatTensor,
                           weights:torch.FloatTensor,
                           assignments:torch.FloatTensor,
                           n_clusters:int, verbose:bool = False)->torch.FloatTensor:
    """
    Updates centroids based on the current assignments.
    
    Args:
        data (torch.Tensor): Input data of shape (n, d)
        weights (torch.Tensor): Weights for each data point of shape (n, d)
        assignments (torch.Tensor): Current cluster assignments of shape (n,)
        n_clusters (int): Number of clusters
        
    Returns:
        torch.Tensor: Updated centroids of shape (n_clusters, d)
    """
    d = data.shape[1]
    centroids = torch.zeros((n_clusters, d), device=data.device)
    # n_samples = torch.zeros((n_clusters,), device=data.device)
    
    weighted_data = data * weights
    for k in range(n_clusters):
        # Mask for the current cluster
        mask = (assignments == k)
        # Sum weighted data points in this cluster
        masked_data = weighted_data[mask] # Shape: (n_samples[k], d)
        masked_weights = weights[mask] # Shape: (n_samples[k], d)

        # weights = weights[~mask]
        # data = data[~mask]
        # assignments = assignments[~mask]
        # weighted_data = weighted_data[~mask]

        # Calculate new centroid
        centroids[k] = torch.sum(masked_data, dim=0) / (torch.sum(masked_weights, dim=0) + 1e-8)
        
    return centroids

class LinearVQ(compression_parent.CompressedLinear):
    """K-means VQ quantizer"""
    name = "LinearVQ"

    @torch.no_grad()
    def quantize_(self, d:int = 4,
                        n_bits:Union[int, float] = 2,
                        n_inits:int = 1,
                        n_iter:int = 100,
                        ignore_norms = True,
                        normalizer_kwargs:dict = {},
                        normalizer:quantizer_utils.Normalizer = None,
                        **kwargs):
        """Quantize the weight matrix using K-means VQ

        Args:
            d (int, optional): subvector dimension. Defaults to 4.
            n_bits (Union[int, float], optional): number of bits per subvector. Defaults to 8. d*n_bits must be an integer.
            n_inits (int, optional): number of initializations for K-means. Defaults to 1.
            n_iter (int, optional): number of iterations for K-means. Defaults to 100.
            ignore_norms (bool, optional): whether to ignore the norms for k-means. Defaults to True.
            normalizer_kwargs (dict, optional): normalizer kwargs to create a normalizer. Defaults to {}.
            normalizer (quantizer_utils.Normalizer, optional): normalizer that was passed in. Defaults to None.
        """

        normalized_weight = self.initialize_normalizer(normalizer=normalizer, normalizer_kwargs=normalizer_kwargs)

        normalized_weight_use = normalized_weight.clone()

        k_mean_weights = self.get_hessianDiag().unsqueeze(0).repeat(self.out_features, 1) #shape of (out_features, in_features)
        if not ignore_norms:
            k_mean_weights *= self.normalizer.denormalize(torch.ones_like(self.original_weight), debias=False) ** 2
        

        #padding, check if we need to pad
        if self.in_features % d != 0:
            #we must pad the input
            print("Padding input to make it divisible by d")
            pad_size = d - self.in_features % d
            print("Pad size: ", pad_size)
            normalized_weight_use = F.pad(normalized_weight_use, (0, pad_size), value = torch.mean(normalized_weight_use).item())
            k_mean_weights = F.pad(k_mean_weights, (0, pad_size), value = 0)
            self.padded_in_features = normalized_weight_use.shape[1]
        else:
            self.padded_in_features = self.in_features

        #check that the number of bits is an integer
        assert d*n_bits == int(d*n_bits), "d*n_bits must be an integer"

        n_centriods = 2**(int(n_bits * d))
        print("Number of centriods: ", n_centriods)
        weight_subvectors = normalized_weight_use.reshape(-1, d)

        best_loss = float('inf')

        #reshape k_mean_weights to be the same shape as weight_subvectors
        k_mean_weights = k_mean_weights.reshape(-1, d)
        
        assign_time = []
        update_time = []
        for i in tqdm.tqdm(range(n_inits), desc="Initializing K-means", disable=not self.verbose):
            #initialize the codebook by randomly selecting n_centriods vectors from the data set
            codebook = weight_subvectors[torch.from_numpy(
                        np.random.choice(weight_subvectors.shape[0], n_centriods, replace=False))]
            
            for j in tqdm.tqdm(range(n_iter), desc="Iterating K-means", disable=not self.verbose):
                start = time.time()
                assignments,loss = weighted_kmeans_assign(weight_subvectors, k_mean_weights, codebook, verbose = self.verbose)
                assign_time.append(time.time() - start)
                
                start = time.time()
                codebook = weighted_kmeans_update(weight_subvectors, k_mean_weights,assignments, n_centriods)
                update_time.append(time.time() - start)

                if j > 0:
                    if torch.all(assignments == assignments_old):
                        break
                
                assignments_old = assignments.clone()

                # if self.verbose:
                #     tqdm.tqdm.write("Loss: {}".format(loss))

            if loss < best_loss:
                best_assignments = assignments.clone()
                best_codebook = codebook.clone()
        if self.verbose:
            print("Average assign time: ", "{:.2e}".format(np.mean(assign_time)),"+/-", "{:.2e}".format(np.std(assign_time)))
            print("Average update time: ", "{:.2e}".format(np.mean(update_time)),"+/-", "{:.2e}".format(np.std(update_time)))
        self.codebook = nn.Parameter(best_codebook)
        self.register_buffer("assignments", best_assignments)

    def compress(self, d:int = 4,
                        n_bits:Union[int, float] = 2,
                        n_inits:int = 1,
                        n_iter:int = 100,
                        ignore_norms = True,
                        normalizer_kwargs:dict = {},
                        normalizer:quantizer_utils.Normalizer = None,
                        **kwargs):
        self.compressed = True
        self.quantize_(d = d,
                        n_bits = n_bits,
                        n_inits = n_inits,
                        n_iter = n_iter,
                        ignore_norms = ignore_norms,
                        normalizer_kwargs = normalizer_kwargs,
                        normalizer = normalizer,
                        **kwargs)
        
    def reconstruct_(self, denormalize = True):  
        """Reconstruct the weight matrix from the codebook and assignments

        Args:
            denormalize (bool, optional): whether to denormalize the weight matrix. Defaults to True.
        """
        weight_subvectors = self.codebook[self.assignments]
        weight = weight_subvectors.reshape(self.out_features, self.padded_in_features)
        weight = weight[:, :self.in_features]
        if denormalize:
            weight = self.normalizer.denormalize(weight)
        return weight
    
    def _no_checkpoint_forward(self, x: torch.FloatTensor):
        if self.forward_method == "otf":
            warnings.warn("OTF forward method is not supported for LinearVQ, using reconstruct method instead")

        if self.denormalization_method == "otf":
            x = self.normalizer.denormalize_otf_in(x)
        
        y = F.linear(x, self.reconstruct_(denormalize=self.denormalization_method == "reconstruct"
                                          ), bias=self.bias)
        
        if self.denormalization_method == "otf":
            y = self.normalizer.denormalize_otf_out(y)
        
        return y

    def get_n_bits(self):
        n_unique = torch.unique(self.assignments).shape[0]
        n_bits = self.normalizer.get_n_bits()
        n_bits += math.log2(n_unique) * self.assignments.numel()
        n_bits += self.codebook.numel() * 16
        return n_bits
    
    def blank_recreate(self, d:int = 4,
                        n_bits:Union[int, float] = 2,
                        normalizer_kwargs:dict = {},
                        normalizer:quantizer_utils.Normalizer = None):
        
        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = quantizer_utils.Normalizer.blank_recreate(self.original_weight, **normalizer_kwargs)

        self.codebook = nn.Parameter(torch.zeros((2**(int(n_bits * d)), d), device=self.original_weight.device,
                                    dtype=self.original_weight.dtype))
        
        if self.in_features % d != 0:
            #we must pad the input
            pad_size = d - self.in_features % d
            n_subvectors = ((self.in_features + pad_size) * self.out_features) // d
            self.padded_in_features = self.in_features + pad_size
        else:
            
            n_subvectors = (self.in_features * self.out_features) // d 
            self.padded_in_features = self.in_features
        
        self.register_buffer("assignments", torch.zeros(n_subvectors, dtype=torch.long, device=self.original_weight.device))
    


class PlaceHolderKMeans:
    def __init__(self, assignments, centroids):
        self.assignments = assignments
        self.centroids = centroids
        self.done = False
    
    def to(self, device):
        self.assignments = self.assignments.to(device)
        self.centroids = self.centroids.to(device)
        return self


class LinearVQ_Halving(LinearVQ):
    """K-means VQ quantizer with halving"""
    name = "LinearVQ_Halving"

    @torch.no_grad()
    def quantize_(self, d:int = 4,
                        n_bits:Union[int, float] = 2,
                        n_inits:int = 8,
                        n_iter_before_halving:int = 10,
                        ignore_norms = True,
                        normalizer_kwargs:dict = {},
                        normalizer:quantizer_utils.Normalizer = None,
                        **kwargs):
        """Quantize the weight matrix using K-means VQ

        Args:
            d (int, optional): subvector dimension. Defaults to 4.
            n_bits (Union[int, float], optional): number of bits per subvector. Defaults to 8. d*n_bits must be an integer.
            n_inits (int, optional): number of initializations for K-means. Defaults to 1.
            n_iter (int, optional): number of iterations for K-means. Defaults to 100.
            ignore_norms (bool, optional): whether to ignore the norms for k-means. Defaults to True.
            normalizer_kwargs (dict, optional): normalizer kwargs to create a normalizer. Defaults to {}.
            normalizer (quantizer_utils.Normalizer, optional): normalizer that was passed in. Defaults to None.
        """

        normalized_weight = self.initialize_normalizer(normalizer=normalizer, normalizer_kwargs=normalizer_kwargs)

        normalized_weight_use = normalized_weight.clone()

        k_mean_weights = self.get_hessianDiag().unsqueeze(0).repeat(self.out_features, 1) #shape of (out_features, in_features)
        if not ignore_norms:
            k_mean_weights *= self.normalizer.denormalize(torch.ones_like(self.original_weight), debias=False) ** 2
        

        #padding, check if we need to pad
        if self.in_features % d != 0:
            #we must pad the input
            print("Padding input to make it divisible by d")
            pad_size = d - self.in_features % d
            print("Pad size: ", pad_size)
            normalized_weight_use = F.pad(normalized_weight_use, (0, pad_size), value = torch.mean(normalized_weight_use).item())
            k_mean_weights = F.pad(k_mean_weights, (0, pad_size), value = 0)
            self.padded_in_features = normalized_weight_use.shape[1]
        else:
            self.padded_in_features = self.in_features

        #check that the number of bits is an integer
        assert d*n_bits == int(d*n_bits), "d*n_bits must be an integer"

        n_centriods = 2**(int(n_bits * d))
        print("Number of centriods: ", n_centriods)
        weight_subvectors = normalized_weight_use.reshape(-1, d)

        best_loss = float('inf')

        #reshape k_mean_weights to be the same shape as weight_subvectors
        k_mean_weights = k_mean_weights.reshape(-1, d)
        
        k_means = {}
        active_kmeans = []
        for i in range(n_inits):
            #initialize the codebook by randomly selecting n_centriods vectors from the data set
            codebook = weight_subvectors[torch.from_numpy(
                        np.random.choice(weight_subvectors.shape[0], n_centriods, replace=False))]
                    
            assignments, initial_loss = weighted_kmeans_assign(weight_subvectors, k_mean_weights, codebook, verbose = self.verbose)
            k_means[i] = PlaceHolderKMeans(assignments, codebook).to("cpu") #move to cpu to save memory
            k_means[i].losses = [initial_loss]

        active_kmeans = list(k_means.keys())

        total_iter = n_iter_before_halving * math.log2(n_inits)

        while True:
    
            for i in active_kmeans:
                k_mean_use = k_means[i].to(weight_subvectors.device)
                if k_mean_use.done:
                    continue

                for j in tqdm.tqdm(range(n_iter_before_halving)):
                    k_mean_use.centroids = weighted_kmeans_update(weight_subvectors, k_mean_weights,k_mean_use.assignments, n_centriods)
                    k_mean_use.assignments,loss = weighted_kmeans_assign(weight_subvectors, k_mean_weights, k_mean_use.centroids, verbose = self.verbose)
                    # tqdm.tqdm.write("Loss: {}".format(loss))
                    k_mean_use.losses.append(loss)
                    if j > 0:
                        if torch.all(k_mean_use.assignments == assignments_old):
                            k_mean_use.done = True
                            break
                    
                    assignments_old = k_mean_use.assignments.clone()
                
                k_mean_use.to("cpu")
                utils.clean()
            
            #check the losses
            losses = [k_means[i].losses[-1] for i in active_kmeans]
            #if we have only one active kmeans, we are done
            if len(active_kmeans) == 1:
                best_k_means = active_kmeans[0]
                break
            
            #else find the best kmeans
            idxs = np.argsort(losses)
            active_kmeans = [active_kmeans[i] for i in idxs[:len(active_kmeans)//2]]
            n_iter_before_halving *= 2
                
        if self.verbose:
            #plot the losses
            import matplotlib.pyplot as plt
            for i in k_means.keys():
                plt.plot(k_means[i].losses, label = i)
            plt.legend()
            #log log
            plt.yscale("log")
            plt.xscale("log")
            plt.savefig("losses.png")
            plt.close()

        k_means[best_k_means].to(weight_subvectors.device)
        self.codebook = nn.Parameter(k_means[best_k_means].centroids)
        self.register_buffer("assignments", k_means[best_k_means].assignments)

    def compress(self, d:int = 4,
                        n_bits:Union[int, float] = 2,
                        n_inits:int = 1,
                        n_iter_before_halving:int = 10,
                        ignore_norms = True,
                        normalizer_kwargs:dict = {},
                        normalizer:quantizer_utils.Normalizer = None,
                        **kwargs):
        self.compressed = True
        self.quantize_(d = d,
                        n_bits = n_bits,
                        n_inits = n_inits,
                        n_iter_before_halving = n_iter_before_halving,
                        ignore_norms = ignore_norms,
                        normalizer_kwargs = normalizer_kwargs,
                        normalizer = normalizer,
                        **kwargs)

    

if __name__ == "__main__":
    import copy
    utils.seed(1234)
    suffix = "layer_0/self_attn.q_proj.pt" 
    test_weight = f"/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/original_weights/{suffix}"
    test_hessian = f"/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians_new/seed_0/pajama/128/{suffix}"

    device = torch.device("cuda:6")
    weight = torch.load(test_weight, map_location=device, weights_only=True)["weight"].float()
    hessian = torch.load(test_hessian, map_location=device, weights_only=True)["hessian"].float()
    print(weight.dtype)
    quantizer = LinearVQ(weight=weight,verbose = True)
    quantizer.hessian = hessian
    quantizer.compress(d=12, n_bits=1, n_inits=1, n_iter=100, ignore_norms=True, normalizer_kwargs={"norm_order":[0,1], "zero":[False, False]},
                       normalizer=None)
    # quantizer.compress(d=6, n_bits=2, n_inits=16, n_iter_before_halving=5, ignore_norms=True, normalizer_kwargs={"norm_order":[0,1], "zero":[False, False]},
    #                    normalizer=None)
    
    # print(quantizer.reconstruct())
    # print(weight)
    print(quantizer.get_reconstruction_error(hessian))

    x = torch.randn(1, weight.shape[1], device=device)
    y_prev = quantizer(x)
    for denormalization_method in ["otf", "reconstruct"]:
        for forward_method in ["otf", "reconstruct"]:
            print("Forward method: ", forward_method, "Denormalization method: ", denormalization_method)
            quantizer.change_forward_method(forward_method)
            quantizer.change_denormalization_method(denormalization_method)
            y = quantizer(x)
            assert torch.allclose(y, y_prev, atol = 1e-5), f"Forward method is not consistent: {y} != {y_prev}, large error: {torch.max(torch.abs(y/y_prev - 1))}"

    quantizer.change_denormalization_method("ignore")
    y_prev = quantizer(x)

    for forward_method in ["otf", "reconstruct"]:
        print("Forward method: ", forward_method, "Denormalization method: ignore")
        quantizer.change_forward_method(forward_method)
        y = quantizer(x)
        assert torch.allclose(y, y_prev, atol = 1e-5), f"Forward method is not consistent: {y} != {y_prev}, large error: {torch.max(torch.abs(y/y_prev - 1))}"
    

    quantizer_state_dict = copy.deepcopy(quantizer.state_dict())

    new_quantizer = LinearVQ(weight=weight,verbose = True)
    new_quantizer.blank_recreate(d=5, n_bits=2, normalizer_kwargs={"norm_order":[0,1], "zero":[False, False]},
                                 normalizer=None)
    
    # print("new quantizer state_dict keys: ", new_quantizer.state_dict())
    # print("quantizer state_dict keys: ", quantizer_state_dict)
    
    new_quantizer.load_state_dict(quantizer_state_dict)
    # new_quantizer.to(device)
    new_recon  = new_quantizer.reconstruct()
    recon = quantizer.reconstruct()
    assert torch.allclose(new_recon, recon), f"Reconstruction is not the same: {new_recon} != {recon}\n" \
                                            + f"large error abs: {torch.max(torch.abs(new_recon - recon))} rel: {torch.max(torch.abs(new_recon/recon - 1))}"


    





        


    

            


    
                  
