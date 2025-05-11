import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Literal

# a class of sparse stuff


class SparseParent(nn.Module):

    def __init__(self, n_out: int, n_in: int, device: torch.device):
        super(SparseParent, self).__init__()
        self.n_out = n_out
        self.n_in = n_in
        self.device = device

    def sparse(self, importances: torch.Tensor, remaining_weight: torch.Tensor, *args):
        raise NotImplementedError

    def update_fixed_mask(self, remaining_weight: torch.Tensor):
        raise NotImplementedError

    def reconstruct(self) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def get_n_bits(self):
        raise NotImplementedError


class Dim0_StructuredSparse(SparseParent):
    sparse_mask: torch.BoolTensor

    def __init__(self, n_out: int, n_in: int, frac_sparse: float, device: torch.device):

        super(Dim0_StructuredSparse, self).__init__(n_out, n_in, device)

        sparse_mask = torch.zeros(n_in, dtype=torch.bool, device=device)
        # print("n_in", n_in, "n_out", n_out)
        self.frac_sparse = frac_sparse
        self.n_sparse = int(frac_sparse * n_in)

        self.register_buffer("sparse_mask", sparse_mask)
        self.sparse_values = nn.Parameter(
            torch.zeros((self.n_out, self.n_sparse), device=device)
        )

    def sparse(
        self,
        importances: torch.FloatTensor,  # importance of shape (n_out, n_in)
        remaining_weight: torch.FloatTensor,  # importance of shape (n_out, n_in)
        pooling_fn: callable = lambda x: torch.mean(x, dim=0),  # pooling function
    ):

        # calculate the pooled importance
        pooled_importances = pooling_fn(importances)

        # sort the importances
        idxs_sorted = torch.argsort(pooled_importances, descending=True)

        new_mask = torch.zeros_like(self.sparse_mask)

        new_mask[idxs_sorted[: self.n_sparse]] = True

        self.register_buffer("sparse_mask", new_mask)

        self.sparse_values.data = remaining_weight[:, new_mask]
        # print("sparse_values", self.sparse_values)

    def reconstruct(self):
        # print("sparse_idxs", torch.where(self.sparse_mask))
        reconstructed_large = torch.zeros(
            self.n_out,
            self.n_in,
            device=self.sparse_values.device,
            dtype=self.sparse_values.dtype,
        )
        reconstructed_large[:, self.sparse_mask] = self.sparse_values
        return reconstructed_large

    def forward(
        self,
        x: torch.FloatTensor,  # shape (...,n_in)
        y: Optional[torch.FloatTensor] = None,
    ):
        return x[..., self.sparse_mask] @ self.sparse_values.T

    def get_n_bits(self):
        return self.sparse_values.numel() * 16 + self.sparse_mask.numel()

    def update_fixed_mask(self, remaining_weight: torch.FloatTensor):

        self.sparse_values.data = remaining_weight[:, self.sparse_mask]


class Dim1_StructuredSparse(SparseParent):
    sparse_mask: torch.BoolTensor

    def __init__(self, n_out: int, n_in: int, frac_sparse: float, device: torch.device):

        super(Dim1_StructuredSparse, self).__init__(n_out, n_in, device)

        sparse_mask = torch.zeros(n_out, dtype=torch.bool, device=device)
        # print("n_in", n_in, "n_out", n_out)
        self.frac_sparse = frac_sparse
        self.n_sparse = int(frac_sparse * n_out)

        self.register_buffer("sparse_mask", sparse_mask)
        self.sparse_values = nn.Parameter(
            torch.zeros((self.n_sparse, self.n_in), device=device)
        )

    def sparse(
        self,
        importances: torch.FloatTensor,  # importance of shape (n_out, n_in)
        remaining_weight: torch.FloatTensor,  # importance of shape (n_out, n_in)
        pooling_fn: callable = lambda x: torch.mean(x, dim=0),  # pooling function
    ):

        # calculate the pooled importance
        pooled_importances = pooling_fn(importances, dim=1)

        # sort the importances
        idxs_sorted = torch.argsort(pooled_importances, descending=True)

        new_mask = torch.zeros_like(self.sparse_mask)

        new_mask[idxs_sorted[: self.n_sparse]] = True

        self.register_buffer("sparse_mask", new_mask)

        self.sparse_values.data = remaining_weight[new_mask]
        # print("sparse_values", self.sparse_values)

    def forward(
        self,
        x: torch.FloatTensor,  # shape (...,n_in)
        y: Optional[torch.FloatTensor] = None,
    ):

        if y is None:
            y = torch.empty(
                x.shape[:-1] + (self.n_out,), device=x.device, dtype=x.dtype
            )

        y[..., self.sparse_mask] = x @ self.sparse_values.T
        return y

    def reconstruct(self):
        # print("sparse_idxs", torch.where(self.sparse_mask))
        reconstructed_large = torch.zeros(
            self.n_out,
            self.n_in,
            device=self.sparse_values.device,
            dtype=self.sparse_values.dtype,
        )
        reconstructed_large[self.sparse_mask, :] = self.sparse_values
        return reconstructed_large

    def get_n_bits(self):
        return self.sparse_values.numel() * 16 + self.sparse_mask.numel()

    def update_fixed_mask(self, remaining_weight: torch.FloatTensor):

        self.sparse_values.data = remaining_weight[self.sparse_mask]


def stochastic_sparse(importances: torch.FloatTensor, n_sparse: int, temp: float = 1.0):
    print("=" * 100)
    print("importances", importances.shape)
    probs = torch.softmax(importances.flatten() / temp, dim=0)
    print("probs", probs.shape, torch.sum(probs))
    # print(probs.shape)
    new_mask = torch.zeros_like(probs, dtype=torch.bool)
    frac_sparse = n_sparse / importances.numel()
    n_sparsed = 0
    idxs = torch.randperm(probs.numel())

    for i in range(0, probs.shape[0], 2**24):
        print(i)
        idx_section = idxs[i : i + 2**24]
        probs_section = probs[idx_section]
        new_mask_section = new_mask[idx_section]
        print("probs_section", probs_section.shape)
        if probs_section.shape[0] == 0:
            continue
        print("torch.sum(probs_section)", torch.sum(probs_section))
        n_sparse_section = (
            int(torch.sum(probs_section) * n_sparse)
            if i + 2**24 < probs.shape[0]
            else n_sparse - n_sparsed
        )
        if n_sparse_section == 0:
            continue
        if n_sparse_section > probs_section.shape[0]:
            n_sparse_section = probs_section.shape[0]
            new_mask_section[:] = True
            # add eps to the probabilities beyond
            probs[idxs[i + 2**24 :]] += 1e-8
        else:
            print("n_sparse_section", n_sparse_section)
            new_mask_section[torch.multinomial(probs_section, n_sparse_section)] = True
        new_mask[idx_section] = new_mask_section
        n_sparsed += n_sparse_section

    # new_mask[torch.multinomial(probs, n_sparse)] = True
    print(torch.sum(new_mask) / importances.numel())
    assert (
        torch.sum(new_mask) == n_sparse
    ), f"torch.sum(new_mask) != n_sparse, {torch.sum(new_mask)} != {n_sparse}"

    return new_mask.reshape(importances.shape)


class UnstructuredSparse(nn.Module):
    # completely unstructured sparse and a parent class
    sparse_mask: torch.BoolTensor
    compressed: bool

    def __init__(
        self,
        n_out: int,
        n_in: int,
        frac_sparse: float,
        device: torch.device,
        pattern: Optional[Tuple[int, int]] = None,
        sparse_group: Union[int, Literal["n_in"]] = -1,
        stochastic: bool = False,
        temp: float = 1.0,
    ):

        super(UnstructuredSparse, self).__init__()

        # create a placeholder sparse mask
        sparse_mask = torch.zeros(n_out, n_in, dtype=torch.bool, device=device)

        self.n_out = n_out
        self.n_in = n_in
        self.frac_sparse = frac_sparse
        self.n_sparse = int(frac_sparse * n_out * n_in)

        self.sparse_pattern = pattern
        self.sparse_group = sparse_group if sparse_group != "n_in" else n_in
        self.stochastic = stochastic
        self.temp = temp

        self.register_buffer("sparse_mask", sparse_mask)
        self.sparse_values = nn.Parameter(torch.zeros(self.n_sparse, device=device))
        # self.compressed = False #convert to csr tensor for

    @staticmethod
    def generate_mask(
        importances: torch.FloatTensor,
        n_sparse: int,
        stochastic: bool = False,
        temp: float = 1.0,
    ):
        if stochastic:
            return stochastic_sparse(importances, n_sparse, temp)
        else:
            idxs_sorted = torch.argsort(importances.flatten(), descending=True)
            idx0, idx1 = (
                idxs_sorted // importances.shape[1],
                idxs_sorted % importances.shape[1],
            )
            new_mask = torch.zeros_like(importances, dtype=torch.bool)
            new_mask[idx0[:n_sparse], idx1[:n_sparse]] = True
            return new_mask

    @staticmethod
    def generate_mask_grouped(
        importances: torch.FloatTensor, frac_sparse: float, n_group: int
    ):
        importances_reshaped = importances.reshape(
            -1, n_group
        )  # reshaped into the groups
        idxs_sorted = torch.argsort(importances_reshaped, dim=-1, descending=True)
        new_mask = torch.zeros_like(importances_reshaped, dtype=torch.bool)

        n_sparse_group = int(frac_sparse * n_group)
        assert n_sparse_group * importances_reshaped.shape[0] == int(
            frac_sparse * importances.shape[1] * importances.shape[0]
        ), f"n_sparse_group * importances_reshaped.shape[0] != int(frac_sparse * importances.shape[1] * importances.shape[0])"
        new_mask[
            torch.arange(new_mask.shape[0]).unsqueeze(1),
            idxs_sorted[:, :n_sparse_group],
        ] = True
        return new_mask.reshape(importances.shape)

    @staticmethod
    def generate_mask_pattern(importances: torch.FloatTensor, pattern: Tuple[int, int]):

        n_zero, n_pattern = pattern

        idxs_sorted = torch.argsort(
            importances.reshape(-1, n_pattern), dim=-1, descending=True
        )
        # print("generate_mask_pattern", idxs_sorted)
        mask_unshaped = torch.zeros_like(
            importances.reshape(-1, n_pattern),
            dtype=torch.bool,
            device=importances.device,
        )
        mask_unshaped[
            torch.arange(mask_unshaped.shape[0]).unsqueeze(1), idxs_sorted[:, :n_zero]
        ] = True
        return mask_unshaped.reshape(importances.shape)

    def sparse(
        self,
        importances: torch.FloatTensor,  # importance of shape (n_out, n_in)
        remaining_weight: torch.FloatTensor,  # importance of shape (n_out, n_in)
    ):

        if self.sparse_pattern is not None:
            new_mask = self.generate_mask_pattern(importances, self.sparse_pattern)
        elif self.sparse_group > 0:
            new_mask = self.generate_mask_grouped(
                importances, self.frac_sparse, self.sparse_group
            )
        else:
            new_mask = self.generate_mask(
                importances, self.n_sparse, self.stochastic, self.temp
            )
        # #sort the importances
        # idxs_sorted = torch.argsort(importances.flatten(), descending=True)
        # idx0, idx1 = idxs_sorted//self.n_in, idxs_sorted%self.n_in

        # new_mask = torch.zeros_like(self.sparse_mask)
        # new_mask[idx0[:self.n_sparse], idx1[:self.n_sparse]] = True
        # print("sparse_mask", torch.sum(new_mask)/new_mask.numel())
        self.register_buffer("sparse_mask", new_mask)

        self.sparse_values.data = remaining_weight[new_mask]

    def update_fixed_mask(self, remaining_weight: torch.FloatTensor):
        # print("remaining_error", remaining_error)
        self.sparse_values.data = remaining_weight[self.sparse_mask]

    def reconstruct(self):
        if hasattr(self, "cached_reconstruction"):
            return self.cached_reconstruction
        out = torch.zeros(
            (self.n_out, self.n_in),
            device=self.sparse_values.device,
            dtype=self.sparse_values.dtype,
        )

        out[self.sparse_mask] = self.sparse_values
        return out

    def cache_reconstruct(self):
        self.register_buffer("cached_reconstruction", self.reconstruct())

    def delete_cache_reconstruct(self):
        delattr(self, "cached_reconstruction")

    def forward(self, x):
        return F.linear(x, self.reconstruct())

    def get_n_bits(self):
        nse = torch.sum(self.sparse_mask)
        nrows = self.n_out
        # n_indicies_bits = torch.ceil(torch.log2(max(nse, nrows)))
        nbits = ((nrows * 16 + (16 + 16) * nse)).item()
        return nbits

    def set_mask_and_values(
        self,
        sparse_mask: torch.BoolTensor,
        sparse_values: torch.Tensor,
        n_sparse: Optional[int] = None,
    ):
        self.sparse_mask = sparse_mask
        self.sparse_values = nn.Parameter(sparse_values.detach().clone())
        if n_sparse is not None:
            self.n_sparse = n_sparse