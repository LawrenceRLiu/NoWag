import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import tqdm
import torch.jit as jit
import os 
import sys 
from typing import Tuple, Optional, Union, List
if __name__ == "__main__":
    sys.path.append(os.getcwd())
import src.utils.quantizer as quantizer_utils
import src.utils.utils as utils
import src.quantizers.quantizer_parent as quantizer_parent

class VectorQuantizer(quantizer_parent.QuantizerParent):
    """Vector Quantizer without any sparse preservations"""
    def __init__(self, codes: torch.LongTensor,
                 codebook: torch.FloatTensor,
                 reconstructed_shape: Union[Tuple[int,int], torch.Size],
                 norms_1:Optional[torch.FloatTensor] = None,
                 norms_0:Optional[torch.FloatTensor] = None,
                 reference_weight:Optional[torch.FloatTensor] = None,
    ):
        """Vector Quantizer without any sparse preservations

        Args:
            codes (torch.LongTensor): the codes for the quantizer, of shape (n_values)
            codebook (torch.FloatTensor): the codebook for the quantizer, of shape (codebook_size, d)
            reconstructed_shape (Union[Tuple[int,int], torch.Size]): the size of the weight matrix we want to reshape to, after dequantization, expected to be (n_out,n_in) where n_out * n_in/d = n_values and n_in is divisible by d
            norms_1 (Optional[torch.FloatTensor], optional): The normalization of the weight matrix along the 0th dimension, of shape n_in. Defaults to None.
            norms_0 (Optional[torch.FloatTensor], optional): The normalization of the weight matrix along the 1st dimension, of shape n_out. Defaults to None.
            reference_weight (Optional[torch.FloatTensor], optional): The reference weight matrix, of shape (n_out,n_in). Defaults to None.
        """
        super(VectorQuantizer, self).__init__(codes, codebook, reconstructed_shape, reference_weight)
        if norms_1 is not None:
            self.register_buffer('norms_1', norms_1)
        else:
            self.norms_1 = None
        if norms_0 is not None:
            self.register_buffer('norms_0', norms_0)
        else:
            self.norms_0 = None
            
    def set_additional_attributes_as_trainable(self):
        if self.norms_1 is not None:
            utils.buffer_to_param(self, 'norms_1')
        if self.norms_0 is not None:
            utils.buffer_to_param(self, 'norms_0')
        
    def set_additional_attributes_as_buffer(self):
        if self.norms_1 is not None:
            utils.param_to_buffer(self, 'norms_1')
        if self.norms_0 is not None:
            utils.param_to_buffer(self, 'norms_0')
            
    def forward(self):
        
        reconstructed_weight = self.codebook[self.codes].view(self.reconstructed_shape)
        if self.norms_1 is not None:
            reconstructed_weight = reconstructed_weight * self.norms_1.unsqueeze(1)
        if self.norms_0 is not None:
            reconstructed_weight = reconstructed_weight * self.norms_0.unsqueeze(0)
            
        return reconstructed_weight
    

    @staticmethod
    def quantize(weight,hessian,
                d:int = 4,
                n_bits:int = 2, #number of bits per weight
                n_iter:int = 100,
                norm_order:list[int] = [0,1]):
        
        weight_use = weight.clone()
        norm_0,norm_1,weight_use = quantizer_utils.normalize(weight_use,norm_order)
    
        denormalize_matrix = torch.ones_like(weight_use)
        if norm_0 is not None:
            denormalize_matrix = denormalize_matrix * norm_0.unsqueeze(0)
        if norm_1 is not None:
            denormalize_matrix = denormalize_matrix * norm_1.unsqueeze(1)
            
            
        H_diag = torch.diag(hessian)
        H_diag = H_diag.reshape(-1,d)
        importances = (H_diag.unsqueeze(0).expand(weight.shape[0], -1, -1) * denormalize_matrix.reshape(denormalize_matrix.shape[0], -1, d)
                                                ).reshape(-1, d)
        
        weight_subvectors = weight_use.reshape(-1,d)
        n_subvectors = weight_subvectors.shape[0]
        n_centriods = 2**(n_bits * d)
        print(n_centriods)
        
        n_1 = torch.from_numpy(np.random.choice(n_subvectors, n_centriods, replace = False)).to(weight.device)
        # print("n_1", n_1)
        # print("max", torch.max(n_1), "min", torch.min(n_1))
        # print(X.shape)
        centriods = weight_subvectors[n_1, :]
        
        for i in range(n_iter):
            assignments = quantizer_utils.cluster_e_step(
                weight_subvectors, centriods, importances)
            # print(assignments)
            # print(assignments.shape)
            centriods = quantizer_utils.cluster_m_step(weight_subvectors, assignments, n_centriods, importances)
            if i > 0:
                if torch.all(assignments == assignments_old):
                    # print("breaking at iteration", i)
                    break
                # print("n_change:", torch.sum(assignments != assignments_old))
            assignments_old = assignments.clone()
            
        return VectorQuantizer(assignments, centriods, weight.shape, 
                               norm_1, norm_0, weight_use)

    def get_n_bits(self):
        n_bits = super().get_n_bits()
        #sum the bits of the norms
        if self.norms_0 is not None:
            n_bits += self.norms_0.numel() *16
        if self.norms_1 is not None:
            n_bits += self.norms_1.numel() * 16
        return n_bits

class VectorQuantizer_SampleReassign(VectorQuantizer):
    """Vector Quantizer with sample reassignment"""
    def __init__(self, codes: torch.LongTensor,
                 codebook: torch.FloatTensor,
                 reconstructed_shape: Union[Tuple[int,int], torch.Size],
                 norms_1:Optional[torch.FloatTensor] = None,
                 norms_0:Optional[torch.FloatTensor] = None,
                 reference_weight:Optional[torch.FloatTensor] = None,
                 reference_importances:Optional[torch.LongTensor] = None,
    ):
        super(VectorQuantizer_SampleReassign, self).__init__(codes, codebook, reconstructed_shape, norms_1, norms_0, reference_weight)
        if reference_importances is not None:
            self.register_buffer('reference_importances', reference_importances)
        else:
            self.reference_importances = None
            
    
    def update_discrete(self):
        
        self.codes = quantizer_utils.cluster_e_step(
                self.reference_weight, self.codebook, self.reference_importances)
    
    @staticmethod
    def quantize(weight:torch.FloatTensor,
                 hessian:torch.FloatTensor,
                d:int = 4,
                n_bits:int = 2, #number of bits per weight
                n_iter:int = 100,
                norm_order:list[int] = [0,1]):
        
        weight_use = weight.clone()
        norm_0,norm_1,weight_use = quantizer_utils.normalize(weight_use,norm_order)
    
        denormalize_matrix = torch.ones_like(weight_use)
        if norm_0 is not None:
            denormalize_matrix = denormalize_matrix * norm_0.unsqueeze(0)
        if norm_1 is not None:
            denormalize_matrix = denormalize_matrix * norm_1.unsqueeze(1)
            
            
        H_diag = torch.diag(hessian)
        H_diag = H_diag.reshape(-1,d)
        importances = (H_diag.unsqueeze(0).expand(weight.shape[0], -1, -1) * denormalize_matrix.reshape(denormalize_matrix.shape[0], -1, d) ** 0
                                                ).reshape(-1, d)
        
        weight_subvectors = weight_use.reshape(-1,d)
        n_subvectors = weight_subvectors.shape[0]
        n_centriods = 2**(n_bits * d)
        print(n_centriods)
        
        n_1 = torch.from_numpy(np.random.choice(n_subvectors, n_centriods, replace = False)).to(weight.device)
        # print("n_1", n_1)
        # print("max", torch.max(n_1), "min", torch.min(n_1))
        # print(X.shape)
        centriods = weight_subvectors[n_1, :]
        
        for i in range(n_iter):
            assignments = quantizer_utils.cluster_e_step(
                weight_subvectors, centriods, importances)
            # print(assignments)
            # print(assignments.shape)
            centriods = quantizer_utils.cluster_m_step(weight_subvectors, assignments, n_centriods, importances)
            if i > 0:
                if torch.all(assignments == assignments_old):
                    # print("breaking at iteration", i)
                    break
                # print("n_change:", torch.sum(assignments != assignments_old))
            assignments_old = assignments.clone()
            
        return VectorQuantizer_SampleReassign(assignments, centriods, weight.shape, 
                               norm_1, norm_0, weight_subvectors,
                               importances)
        
    

    def clean(self):
        super().clean()
        delattr(self, 'reference_importances')

    @staticmethod
    def blank_recreate(weight:torch.FloatTensor,
                d:int = 4,
                n_bits:int = 2, #number of bits per weight
                norm_order:list[int] = [0,1]):
        
        with torch.no_grad():
            weight_use = weight.clone()
            norm_0,norm_1,weight_use = quantizer_utils.normalize(weight_use,norm_order)

            codebook = torch.zeros(2**(n_bits * d), d).to(weight.device)    
            codes = torch.zeros((weight.shape[0] * weight.shape[1])//d, dtype = torch.long).to(weight.device)
            blank_quantizer = VectorQuantizer_SampleReassign(codes, codebook, weight.shape,
                                                norm_1, norm_0, None, None)
            blank_quantizer.clean()
        return blank_quantizer


    
        
if __name__ == "__main__":
    np.random.seed(0)
    torch.random.manual_seed(0)
    torch.cuda.random.manual_seed(0)
    
    device = torch.device("cuda:7")
    data = torch.load("test/weights_hessian.pt")
    W = data["weights"].to(device)
    hessian = data["hessian"].to(device)
    print(W.shape)
    
    vq = VectorQuantizer_SampleReassign.quantize(W, hessian, d = 4, n_bits = 2, n_iter = 100)
    print(vq.get_n_bits()/vq.get_n_original_parameters())
    vq.set_additional_attributes_as_trainable()
    # sys.path.append(os.getcwd())
    import src.utils.alignment.hessian_general_align as hessian_general_align
    
    hessian_use = hessian/hessian.shape[0]
    # hessian_use = torch.eye(W.shape[0]).to(W.device)
    hessian_general_align.align(vq,
                          W,
                          hessian_use,
                          None,
                          n_iters = 100,
                          val_every = -1,
                          patience = 100,
                          patience_scheduler= 1,
                          eps = 1e-4,
                          lr = 1e-3,
                          low_bound = 1e-6,
                          clip_grad = 1e-1,
                          discrete_update_every = 1,
                          lr_multiplier=0.9,
                          verbose = 10)
    
    # hessian_general_align.align(vq,
    #                       W,
    #                       hessian_use,
    #                       hessian_use,
    #                       n_iters = 100,
    #                       val_every = 1,
    #                       patience = 1000,
    #                       patience_scheduler= 1,
    #                       eps = 1e-4,
    #                       lr = 1e-3,
    #                       low_bound = 1e-6,
    #                       clip_grad = 1e-1,
    #                       discrete_update_every = 1,
    #                       lr_multiplier=0.9,
    #                       verbose = 1)
    
    print(vq().dtype)
    vq.clean()
    vq.to(torch.float16)
    print(vq().dtype)
    
    
    
    
    


    
