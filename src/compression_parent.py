import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import numpy as np
import os
from typing import Tuple, Optional, Union, List, Literal
import src.utils.compress as compress_utils
from src.utils.normalizer import Normalizer
import src.alignment.hessian_general_align as hessian_general_align
import src.utils.utils as utils
from typing import Optional, Union
from dataclasses import dataclass


@dataclass
class PlaceholderOrigWeight:
    shape: Tuple[int, int]
    device: Optional[Union[str, torch.device]] = None
    dtype: Optional[torch.dtype] = None


class CompressedLinear(nn.Module):
    """Parent class of all compression algorithms for linear layers"""

    name = "CompressedLinear"

    def __init__(
        self,
        weight: torch.FloatTensor,
        bias: Optional[torch.FloatTensor] = None,
        add_bias: bool = False,
        verbose: bool = False,
    ):
        """quantized linear layer

        Args:
            weight (torch.FloatTensor): the original weight matrix of shape (out_features, in_features)
            bias (Optional[torch.FloatTensor], optional): the original bias vector of shape (out_features) or None.
            add_bias (bool, optional): should we add a bias to the layer or not. Defaults to False.
        """

        super(CompressedLinear, self).__init__()
        self.out_features, self.in_features = weight.shape
        self.original_weight = weight
        # print("original weight", self.original_weight[0])
        # self.register_buffer("original_weight", weight)
        self.original_parameters = self.in_features * self.out_features

        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=True)
            self.original_parameters += self.out_features

        else:
            if add_bias:
                self.bias = nn.Parameter(
                    torch.zeros(self.out_features), requires_grad=True
                )
            else:
                self.bias = None

        self.compressed = False
        self.grad_checkpoint = False
        self.verbose = verbose
        self.denormalization_method: Literal["otf", "reconstruct", "ignore"] = (
            "reconstruct"
        )
        self.forward_method: Literal["reconstruct", "otf"] = "reconstruct"

    def compress(
        self,
        normalizer_kwargs: Optional[dict] = None,
        normalizer: Optional[Normalizer] = None,
        **kwargs,
    ):
        """compress the weights, this is the main function to be implemented by the child classes"""
        self.compressed = True
        raise NotImplementedError

    # helper function to initialize the normalizer
    def initialize_normalizer(
        self,
        normalizer_kwargs: Optional[dict] = None,
        normalizer: Optional[Normalizer] = None,
    ):
        """Two ways to initialize the normalizer, either pass the normalizer or the normalizer_kwargs

        Args:
            normalizer_kwargs (Optional[dict], optional): kwargs for normalizer. Defaults to None.
            normalizer (Optional[compress_parent.Normalizer], optional): normalizer class. Defaults to None.
        """
        if normalizer is not None:
            self.normalizer = normalizer
            normalized_weight = self.normalizer.normalize(self.original_weight)
        else:
            # print("normalizer_kwargs", normalizer_kwargs)
            if normalizer_kwargs is None:
                print("Warning: normalizer_kwargs is None, using default")
                normalizer_kwargs = {}
            self.normalizer, normalized_weight = Normalizer.normalize_init(
                self.original_weight, **normalizer_kwargs
            )

        return normalized_weight

    def reconstruct_(self, denormalize: bool = True) -> torch.FloatTensor:
        """reconstructs the weigth matrix from the compressed version"""
        raise NotImplementedError

    def _no_checkpoint_forward(self, x: torch.FloatTensor):
        raise NotImplementedError

    def blank_recreate(self, **kwargs):
        """recreates the compressed layer without any compression"""
        raise NotImplementedError

    def get_n_bits(self) -> int:
        """returns the number of bits needed to store the compressed layer"""
        raise NotImplementedError

    # ================ Another Initialization Fns =================
    @classmethod
    def blank_init(
        cls,
        n_in: int,
        n_out: int,
        add_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        """creates a blank layer with the same shape as the original layer"""
        # print("device", device,"dtype", dtype)
        placeholder_weight = PlaceholderOrigWeight(
            (n_out, n_in), device=device, dtype=dtype
        )
        layer = cls(placeholder_weight, add_bias=add_bias)
        layer.blank_recreate(**kwargs)
        return layer

    # ================= Importance/Logging Fns =================
    def enable_hessian_logging(
        self,
        hessian: Optional[torch.FloatTensor] = None,
        logging_type: Literal["mean", "ema"] = "mean",
        **kwargs,
    ):
        """enable hessian logging"""
        if hessian is not None:
            self.register_buffer("hessian", hessian)
        else:
            self.register_buffer(
                "hessian",
                torch.zeros(
                    self.in_features,
                    self.in_features,
                    device=self.original_weight.device,
                    dtype=torch.float32,
                ),
            )
        if logging_type == "mean":
            self.n_samples = kwargs.get("n_samples", 0)  # allows us to continue logging
            self.hessian_handle = self.register_forward_pre_hook(
                compress_utils.hessian_mean_logging
            )
        elif logging_type == "ema":
            self.decay = kwargs.get("decay", 0.99)
            self.hessian_handle = self.register_forward_pre_hook(
                compress_utils.hessian_ema_logging
            )
        else:
            raise ValueError(f"logging_type {logging_type} not supported")

    def enable_hessianDiag_logging(
        self,
        hessianDiag: Optional[torch.FloatTensor] = None,
        logging_type: Literal["mean", "ema"] = "mean",
        **kwargs,
    ):
        """enable hessianDiag logging
        hessianDiag are just the diagonal of the hessian
        """
        if hessianDiag is not None:
            self.register_buffer("hessianDiag", hessianDiag)
        else:
            self.register_buffer(
                "hessianDiag",
                torch.zeros(
                    self.in_features,
                    device=self.original_weight.device,
                    dtype=torch.float32,
                ),
            )

        if logging_type == "mean":
            self.n_samples = kwargs.get("n_samples", 0)
            self.hessianDiag_handle = self.register_forward_pre_hook(
                compress_utils.hessianDiag_mean_logging
            )
        elif logging_type == "ema":
            self.decay = kwargs.get("decay", 0.99)
            self.hessianDiag_handle = self.register_forward_pre_hook(
                compress_utils.hessianDiag_ema_logging
            )

    def dump_hessian(self) -> List[torch.FloatTensor]:
        """gives the hessian calculated and stops logging the inputs for the hessian

        Returns:
            torch.FloatTensor: the hessian
        """
        hessian = self.hessian.clone()
        self.hessian_handle.remove()
        del self.hessian_handle
        del self.hessian
        del self.n_samples
        return [hessian]  # returning a list for consistency with the low rank sparse

    def get_hessian(self) -> torch.FloatTensor:
        if hasattr(self, "hessian"):
            return self.hessian
        else:
            raise Exception("No hessian found")

    def get_hessianDiag(self) -> torch.FloatTensor:
        if hasattr(self, "hessianDiag"):  # new format that saves space
            hessianDiag = self.hessianDiag
        elif hasattr(self, "hessian"):  # old format
            hessianDiag = torch.diag(self.hessian)
        else:
            raise Exception("No hessian found")
        return hessianDiag

    def dump_hessianDiag(self) -> List[torch.FloatTensor]:
        """gives the importances calculated and stops logging the inputs for the importances

        Returns:
            torch.FloatTensor: the importances
        """
        hessianDiag = self.hessianDiag.clone()
        self.hessianDiag_handle.remove()
        del self.hessianDiag_handle
        del self.hessianDiag
        del self.n_samples
        return [
            hessianDiag
        ]  # returning a list for consistency with the low rank sparse

    # ================= Forward Fns =================
    def forward(self, x: torch.FloatTensor):
        """forward pass of the linear layer"""
        if not self.compressed:
            return F.linear(x, self.original_weight, self.bias)
        if self.grad_checkpoint:
            return self._checkpoint_forward(x)
        else:
            return self._no_checkpoint_forward(x)

    def _checkpoint_forward(self, x: torch.FloatTensor):
        return torch.utils.checkpoint.checkpoint(
            self._no_checkpoint_forward, x, use_reentrant=True
        )

    # ================= Backwards Fns =================

    # ================= Reconstruction Fns =================
    def reconstruct(self, **kwargs) -> torch.FloatTensor:
        """reconstructs the weigth matrix from the quantized version"""
        if hasattr(self, "cached_reconstruct"):
            # print("current kwargs", kwargs)
            # print("reconstruct kwargs", self.reconstruct_kwargs)
            # raise ValueError("stop here")
            # print("returning cached")
            # check that the kwargs are the same
            if self.reconstruct_kwargs == kwargs:
                return self.cached_reconstruct
        if kwargs.get("cache", False):
            self.cache_reconstruct(**kwargs)
            return self.reconstruct(**kwargs)
        return self.reconstruct_(**kwargs)

    def cache_reconstruct(self, offload: bool = False, **kwargs):
        # print("caching")
        self.register_buffer("cached_reconstruct", self.reconstruct_(**kwargs))
        self.reconstruct_kwargs = kwargs
        # if we offload, then we offload everything besides the cached reconstruct and bias
        if offload:
            original_device = self.cached_reconstruct.device
            self.to("cpu")
            self.cached_reconstruct = self.cached_reconstruct.to(original_device)
            if self.bias is not None:
                self.bias = nn.Parameter(
                    self.bias.data.detach().clone().to(original_device)
                )

    def delete_cache_reconstruct(self):
        del self.cached_reconstruct

    def get_reconstruction_error(
        self,
        error_weight: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """returns the reconstruction error"""
        with torch.no_grad():
            # if weight is none, then just return the mean squared error
            if error_weight is None:
                return torch.mean((self.reconstruct() - self.original_weight) ** 2)
            # if its a 1d vector, then we assume its the diagonal of the hessian
            if len(error_weight.shape) == 1:
                return torch.mean(
                    (self.reconstruct() - self.original_weight) ** 2
                    * error_weight.unsqueeze(0)
                )
            else:
                return hessian_general_align.loss(
                    self.reconstruct(), self.original_weight, error_weight
                )

    # ================= Misc Fns =================
    def align(
        self,
        val_hessian: Optional[torch.FloatTensor] = None,
        lr: float = 1e-3,
        lr_multiplier: float = 1,  # decay the lr by this factor every time the val loss increases
        n_iters: int = 100,
        val_every: int = 1,
        discrete_update_every: int = 1,
        reinitialize_optimizer: bool = True,
        clip_grad: float = -1,
        verbose: Union[bool, int] = 10,
        low_bound: float = 1e-5,
        patience: int = 10,
        patience_scheduler: int = 2,
        eps: float = 1e-5,
        discrete_update_kwargs: Optional[dict] = {},
        **kwargs,
    ):
        """aligns the compression module to the hessian of the training dataset

        Args:
            val_hessian (Optional[torch.FloatTensor], optional): the hessian of the validation dataset, if None, we don't use it. Defaults to None.
            lr (float, optional): the learning rate for the optimizer. Defaults to 1e-3.
            lr_multiplier (float, optional): multiply the learning rate by this factor every time the validation loss increases. Defaults to 1.
            val_every (int, optional): validate the model every this number of iterations on the validation hessian. Defaults to 1.
            discrete_update_every (int, optional): update the discrete variables every this number of iteration. Defaults to 1.
            clip_grad (float, optional): clip the gradient norm to this value. Defaults to -1 in which case we don't clip the gradient.
            verbose (bool, optional): print every this number of iterations. If False, we don't print anything. If True, we print at the end only. Defaults to False.
            low_bound (float, optional): the lower bound for the error, below which we stop training. Defaults to 1e-5.
            patience (int, optional): the patience for the early stop, if the loss has not improved by eps for this number of iterations, we stop training. Defaults to 10.
            patience_scheduler (int, optional): the patience for the learning rate scheduler. Defaults to 2.
            eps (float, optional): the minimum improvement in the loss to consider it as an improvement. Defaults to 1e-5.

        """
        print(reinitialize_optimizer)
        _, best_loss = hessian_general_align.align(
            compression_module=self,
            original_weights=self.original_weight,
            train_hessian=self.hessian,
            val_hessian=val_hessian,
            lr=lr,
            lr_multiplier=lr_multiplier,
            n_iters=n_iters,
            val_every=val_every,
            discrete_update_every=discrete_update_every,
            reinitialize_optimizer=reinitialize_optimizer,
            clip_grad=clip_grad,
            verbose=verbose,
            low_bound=low_bound,
            patience=patience,
            patience_scheduler=patience_scheduler,
            eps=eps,
            discrete_update_kwargs=discrete_update_kwargs,
            **kwargs,
        )
        return best_loss

    def update_discrete(self, **kwargs):
        """updates the discrete values of the quantizer"""
        pass

    def clean(self):
        if hasattr(self, "original_weight"):
            del self.original_weight
        if hasattr(self, "hessian"):
            self.dump_hessian()
        if hasattr(self, "hessianDiag"):
            self.dump_hessianDiag()
        utils.recursive_apply(self, "clean")

    def get_n_original_parameters(self):
        return self.original_parameters

    def change_denormalization_method(
        self, new_method: Literal["otf", "reconstruct", "ignore"]
    ):
        self.denormalization_method = new_method

    def change_forward_method(self, new_method: Literal["reconstruct", "otf"]):
        self.forward_method = new_method

    def __str__(self):
        return self.name
