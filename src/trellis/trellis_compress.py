import os
import sys
import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, List, Literal

import src.compression_parent as compression_parent
import src.utils.normalizer as normalize
import src.utils.utils as utils

# Import TrellisCodebook and helper functions
from .trellis_codebook import TrellisCodebook, decode_1mad, decode_2mad, decode_3inst, quantlut, quantlut_sym

class LinearTrellis(compression_parent.CompressedLinear):
    """Trellis Quantization for linear layers
    
    This class implements Trellis Quantization, a method for compressing neural networks
    using a trellis-based codebook and the Viterbi algorithm for finding optimal quantization.
    
    The implementation preserves the normalization logic from quantize_compress.py while
    integrating the trellis approach from bitshift.py.
    """
    name = "LinearTrellis"

    @torch.no_grad()
    def quantize_(self, L=16, K=2, V=2, tlut_bits=16, decode_mode='lut', 
                  normalizer_kwargs={}, normalizer=None, td_x=None, td_y=None, 
                  tp_rank=8, dtype=torch.float16, ignore_norms=True, **kwargs):
        """Quantize the weight matrix using Trellis Quantization

        Args:
            L (int): Trellis window size controlling the context length.
            K (int): Bits per weight, determines compression ratio.
            V (int): Vector quantization dimension (usually 1 or 2).
            tlut_bits (int): Number of bits for the lookup table.
            decode_mode (str): Decode mode ('lut', '1mad', '2mad', '3inst', 'quantlut', 'quantlut_sym').
            normalizer_kwargs (dict): Parameters for the normalizer if not provided directly.
            normalizer (Normalizer): Pre-configured normalizer instance.
            td_x (int): Trellis dimension for output features.
            td_y (int): Trellis dimension for input features.
            tp_rank (int): Tensor parallel rank for distribution.
            dtype (torch.dtype): Data type for internal computations.
            ignore_norms (bool): Whether to ignore norms for Hessian weighting.
        """
        normalized_weight = self.initialize_normalizer(normalizer=normalizer, normalizer_kwargs=normalizer_kwargs)
        normalized_weight_use = normalized_weight.clone()
        
        # Save trellis parameters
        self.L = L
        self.K = K
        self.V = V
        self.tlut_bits = tlut_bits
        self.decode_mode = decode_mode
        self.register_buffer('tp_rank', torch.tensor(tp_rank))
        self.dtype = dtype
        

        self.register_buffer("SU", torch.ones(self.in_features, dtype=self.dtype))
        self.register_buffer("SV", torch.ones(self.out_features, dtype=torch.float32))
        
        # Determine trellis dimensions if not provided
        if td_x is None:
            factors = [f for f in range(4, min(32, self.out_features)) if self.out_features % f == 0]
            td_x = factors[0] if factors else 16  # Default to 16 if no factor found
        if td_y is None:
            factors = [f for f in range(4, min(32, self.in_features)) if self.in_features % f == 0]
            td_y = factors[0] if factors else 16  # Default to 16 if no factor found
        
        self.td_x = td_x
        self.td_y = td_y
        
        # Get kmean weights from hessian
        k_mean_weights = self.get_hessianDiag().unsqueeze(0).repeat(self.out_features, 1)  # shape (out_features, in_features)
        if not ignore_norms:
            k_mean_weights *= self.normalizer.denormalize(torch.ones_like(self.original_weight), debias=False) ** 2
            k_mean_weights /= torch.mean(k_mean_weights).item()

        self.pad_y = 0
        if self.in_features % td_y != 0:
            self.pad_y = td_y - (self.in_features % td_y)
            print(f"Padding input features from {self.in_features} to {self.in_features + self.pad_y}")
            normalized_weight_use = F.pad(normalized_weight_use, (0, self.pad_y), 
                                         value=torch.mean(normalized_weight_use).item())
            k_mean_weights = F.pad(k_mean_weights, (0, self.pad_y), value=0)
            self.padded_in_features = normalized_weight_use.shape[1]
        else:
            self.padded_in_features = self.in_features
            
        self.pad_x = 0
        if self.out_features % td_x != 0:
            self.pad_x = td_x - (self.out_features % td_x)
            print(f"Padding output features from {self.out_features} to {self.out_features + self.pad_x}")
            normalized_weight_use = F.pad(normalized_weight_use, (0, 0, 0, self.pad_x), 
                                         value=torch.mean(normalized_weight_use).item())
            k_mean_weights = F.pad(k_mean_weights, (0, 0, 0, self.pad_x), value=0)
            self.padded_out_features = normalized_weight_use.shape[0]
        else:
            self.padded_out_features = self.out_features
            
        # Reshape for trellis 
        weight_blocks = normalized_weight_use.reshape(
            self.padded_out_features // td_x, td_x, 
            self.padded_in_features // td_y, td_y
        ).permute(0, 2, 1, 3).reshape(-1, td_x * td_y)
        
        # reshape Hessian 
        k_mean_weight_blocks = k_mean_weights.reshape(
            self.padded_out_features // td_x, td_x, 
            self.padded_in_features // td_y, td_y
        ).permute(0, 2, 1, 3).reshape(-1, td_x * td_y)
        
        # Initialize codebook 
        self.codebook = TrellisCodebook(
            L=L, K=K, V=V, tlut_bits=tlut_bits, 
            decode_mode=decode_mode, tlut=None
        )
        
        if not ignore_norms:
            _, trellis_states = self.codebook.quantize_weighted(weight_blocks, weights=k_mean_weight_blocks)
        else:
            _, trellis_states = self.codebook.quantize(weight_blocks)
        

        packed_trellis = self.codebook.pack_trellis(trellis_states)
        
        self.register_buffer('trellis', packed_trellis)
        
        self.register_buffer('td_product', torch.tensor(td_x * td_y))
        

        try:
            from lib.utils.kernel_check import has_kernel
            self.has_kernel = has_kernel(decode_mode, L, K, V, tlut_bits, td_x, td_y)
        except ImportError:
            self.has_kernel = False

    def compress(self, L=16, K=2, V=2, tlut_bits=16, decode_mode='lut', 
                normalizer_kwargs={}, normalizer=None, td_x=None, td_y=None, 
                tp_rank=8, dtype=torch.float16, ignore_norms=True, **kwargs):
        """Compress the weight matrix using Trellis Quantization"""
        self.compressed = True
        self.quantize_(
            L=L, K=K, V=V, tlut_bits=tlut_bits, decode_mode=decode_mode,
            normalizer_kwargs=normalizer_kwargs, normalizer=normalizer,
            td_x=td_x, td_y=td_y, tp_rank=tp_rank, dtype=dtype,
            ignore_norms=ignore_norms, **kwargs
        )

    def get_hatW(self, unpacked_trellis, m, n):
        """Reconstruct weight matrix from unpacked trellis
        
        Args:
            unpacked_trellis: Unpacked trellis states
            m: Number of rows (padded output features)
            n: Number of columns (padded input features)
            
        Returns:
            Reconstructed weight matrix
        """

        hatW = self.codebook.recons(unpacked_trellis).transpose(0, 1).transpose(
            1, 2).reshape(m // self.td_x, n // self.td_y, self.td_x,
                        self.td_y).transpose(1, 2).reshape(m, n)
        
        if hasattr(self, 'pad_x') and self.pad_x > 0 and m == self.padded_out_features:
            hatW = hatW[:self.out_features]
        if hasattr(self, 'pad_y') and self.pad_y > 0 and n == self.padded_in_features:
            hatW = hatW[:, :self.in_features]
            
        return hatW

    def get_hatW_kernel(self, trellis, m, n):
        """Reconstruct weight matrix using CUDA kernel
        
        Args:
            trellis: Packed trellis states
            m: Number of rows (output features)
            n: Number of columns (input features)
            
        Returns:
            Reconstructed weight matrix
        """
        try:
            from lib.utils.kernel_decompress import decode_compressed

            out = decode_compressed(self.L, self.tlut_bits, self.K,
                                   int(math.log2(self.V)), m, n, trellis.view(-1),
                                   self.codebook.lut.T)
            
            if hasattr(self, 'pad_x') and self.pad_x > 0 and m == self.padded_out_features:
                out = out[:self.out_features]
            if hasattr(self, 'pad_y') and self.pad_y > 0 and n == self.padded_in_features:
                out = out[:, :self.in_features]
                
            return out
        except (ImportError, RuntimeError) as e:
            warnings.warn(f"CUDA kernel for decompression failed with error: {e}, falling back to CPU implementation")
            return self.get_hatW(self.codebook.unpack_trellis(trellis, self.td_x * self.td_y), m, n)

    def reconstruct_(self, denormalize=True):
        """Reconstruct the weight matrix from the trellis states
        
        Args:
            denormalize (bool): Whether to denormalize the reconstructed weights
            
        Returns:
            torch.Tensor: Reconstructed weight matrix
        """

        if hasattr(self, 'has_kernel') and self.has_kernel:
            try:
                blocks_recons = self.get_hatW_kernel(self.trellis, self.padded_out_features, self.padded_in_features)
            except Exception as e:
                warnings.warn(f"Error using CUDA kernel: {e}, falling back to CPU implementation")
                trellis = self.codebook.unpack_trellis(self.trellis, self.td_product)
                blocks_recons = self.get_hatW(trellis, self.padded_out_features, self.padded_in_features)
        else:
            trellis = self.codebook.unpack_trellis(self.trellis, self.td_product)
            
            blocks_recons = self.get_hatW(trellis, self.padded_out_features, self.padded_in_features)
        
        blocks_recons = blocks_recons.float() / 32.0
    
        if denormalize:
            blocks_recons = self.normalizer.denormalize(blocks_recons)
            
        return blocks_recons

    def _no_checkpoint_forward(self, x: torch.FloatTensor):
        """Forward pass implementation"""
        if self.forward_method == "otf":
            warnings.warn("OTF forward method not fully supported for LinearTrellis, using reconstruct method")
            
        if self.denormalization_method == "otf":
            x = self.normalizer.denormalize_otf_in(x)
        
        input_padded = False
        if hasattr(self, 'pad_y') and self.pad_y > 0:
            x_padded = F.pad(x, (0, self.pad_y), value=0)
            input_padded = True
        else:
            x_padded = x

        if hasattr(self, 'has_kernel') and self.has_kernel and hasattr(torch.ops, 'quip_lib'):
            try:

                x_scaled = x_padded * self.SU
                
                bs = x.shape[0]
                tp_rank = self.tp_rank.item()
                
                original_size = list(x.shape)
                original_size[-1] = self.out_features
                
                if bs == 1:
                    try:

                        wrapper_name = f"decompress_matvec_qtip_{self.padded_out_features}_1_{x_scaled.numel()}_{self.K}"
                        if hasattr(torch.ops.quip_lib, wrapper_name):
                            wrapper = getattr(torch.ops.quip_lib, wrapper_name)
                            y_padded = wrapper(self.trellis, x_scaled, self.codebook.tlut)
                        else:
                            wrapper_name = f"decompress_matvec_qtip_{self.padded_out_features//tp_rank}_{tp_rank}_{x_scaled.numel()}_{self.K}"
                            if hasattr(torch.ops.quip_lib, wrapper_name):
                                wrapper = getattr(torch.ops.quip_lib, wrapper_name)
                                y_padded = wrapper(self.trellis, x_scaled.reshape(-1, self.padded_in_features//tp_rank), self.codebook.tlut)
                                y_padded = y_padded.reshape(bs, self.padded_out_features)
                            else:
                                from lib.codebook.bitshift import BitshiftLinearKernelAG
                                y_padded = BitshiftLinearKernelAG.apply(
                                    x_scaled, self.trellis, self.padded_out_features, self.padded_in_features, 
                                    self.L, self.tlut_bits, self.K, self.V, self.codebook.lut).float()
                    except (AttributeError, RuntimeError) as e:
                        from lib.codebook.bitshift import BitshiftLinearKernelAG
                        y_padded = BitshiftLinearKernelAG.apply(
                            x_scaled, self.trellis, self.padded_out_features, self.padded_in_features, 
                            self.L, self.tlut_bits, self.K, self.V, self.codebook.lut).float()
                else:
                    print ("haha")
                

                y_padded = y_padded * (self.SV * 32)

                if hasattr(self, 'pad_x') and self.pad_x > 0:
                    y = y_padded[..., :self.out_features]
                else:
                    y = y_padded

                if self.denormalization_method == "otf":
                    y = self.normalizer.denormalize_otf_out(y)

                y = y.view(original_size)
                
                return y.to(x.dtype)
            except (AttributeError, RuntimeError, ImportError) as e:
                warnings.warn(f"CUDA kernel execution failed with error: {e}, using standard forward pass")
        
        if input_padded:
            x = x_padded

        y = F.linear(
            x, 
            self.reconstruct_(denormalize=self.denormalization_method == "reconstruct"), 
            bias=self.bias
        )
        
        if self.denormalization_method == "otf":

            y = self.normalizer.denormalize_otf_out(y)
        
        return y

    def get_n_bits(self):
        """Calculate the number of bits used for the compressed representation"""
        # Bits for the trellis
        trellis_bits = self.trellis.numel() * 16
        
        if hasattr(self.codebook, 'tlut'):
            lut_bits = self.codebook.tlut.numel() * 16
        else:
            lut_bits = 0
            
        normalizer_bits = self.normalizer.get_n_bits()
        
        return trellis_bits + lut_bits + normalizer_bits

    def blank_recreate(self, L=16, K=2, V=2, tlut_bits=16, decode_mode='lut',
                      normalizer_kwargs={}, normalizer=None, td_x=None, td_y=None, 
                      tp_rank=8, dtype=torch.float16, ignore_norms=True, **kwargs):
        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = normalize.Normalizer.blank_recreate(self.original_weight, **normalizer_kwargs)
            
        self.L = L
        self.K = K
        self.V = V
        self.tlut_bits = tlut_bits
        self.decode_mode = decode_mode
        self.register_buffer('tp_rank', torch.tensor(tp_rank))
        self.dtype = dtype
        

        self.register_buffer("SU", torch.ones(self.in_features, dtype=self.dtype))
        self.register_buffer("SV", torch.ones(self.out_features, dtype=torch.float32))
 
        if td_x is None:
            factors = [f for f in range(4, min(32, self.out_features)) if self.out_features % f == 0]
            td_x = factors[0] if factors else 16  # Default to 16 if no factor found
        if td_y is None:
            factors = [f for f in range(4, min(32, self.in_features)) if self.in_features % f == 0]
            td_y = factors[0] if factors else 16  # Default to 16 if no factor found
            
        self.td_x = td_x
        self.td_y = td_y
        

        self.pad_x = 0 if self.out_features % td_x == 0 else td_x - (self.out_features % td_x)
        self.pad_y = 0 if self.in_features % td_y == 0 else td_y - (self.in_features % td_y)
        self.padded_out_features = self.out_features + self.pad_x
        self.padded_in_features = self.in_features + self.pad_y
        

        self.codebook = TrellisCodebook(
            L=L, K=K, V=V, tlut_bits=tlut_bits, 
            decode_mode=decode_mode, tlut=None
        )
        

        trellis_size = (self.padded_out_features // td_x) * (self.padded_in_features // td_y)
        trellis_width = math.ceil((td_x * td_y) * K / 16)
        self.register_buffer('trellis', torch.zeros(trellis_size, trellis_width, dtype=torch.uint16))
        self.register_buffer('td_product', torch.tensor(td_x * td_y))

        try:
            from lib.utils.kernel_check import has_kernel
            self.has_kernel = has_kernel(decode_mode, L, K, V, tlut_bits, td_x, td_y)
        except ImportError:
            self.has_kernel = False
            
        self.compressed = True