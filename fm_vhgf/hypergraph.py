"""Separate modality hypergraphs and depth-aligned cross-modality updates.

Adapted from the recovered project's HGYolo_my.py. The cross update transfers
participation structure, rather than substituting the other modality's values.
"""

import math

import torch
from torch import nn


class AdaptiveHyperedgeGenerator(nn.Module):
    def __init__(self, dim: int, num_edges: int, num_heads: int, dropout: float):
        super().__init__()
        if dim % num_heads:
            raise ValueError("Embedding dimension must be divisible by hypergraph heads.")
        self.dim, self.num_edges, self.num_heads = dim, num_edges, num_heads
        self.prototype_base = nn.Parameter(torch.empty(num_edges, dim))
        nn.init.xavier_uniform_(self.prototype_base)
        self.context_net = nn.Linear(2 * dim, num_edges * dim)
        self.pre_head_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        batch, count, dim = nodes.shape
        context = torch.cat((nodes.mean(1), nodes.amax(1)), dim=-1)
        prototypes = self.prototype_base + self.context_net(context).reshape(
            batch, self.num_edges, dim
        )
        width = dim // self.num_heads
        queries = self.pre_head_proj(nodes).reshape(
            batch, count, self.num_heads, width
        ).transpose(1, 2)
        keys = prototypes.reshape(
            batch, self.num_edges, self.num_heads, width
        ).permute(0, 2, 3, 1)
        similarity = (queries @ keys).mean(1) / math.sqrt(width)
        return self.dropout(similarity).softmax(dim=1)


def _projection(dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim, dim), nn.GELU())


class HierarchicalHypergraphFusion(nn.Module):
    """Return updated PET and CT nodes, each with shape (batch, depth, dim)."""

    def __init__(self, dim=768, num_edges=8, num_heads=4, dropout=0.1):
        super().__init__()
        self.pet_generator = AdaptiveHyperedgeGenerator(dim, num_edges, num_heads, dropout)
        self.ct_generator = AdaptiveHyperedgeGenerator(dim, num_edges, num_heads, dropout)
        self.pet_edge, self.pet_node = _projection(dim), _projection(dim)
        self.ct_edge, self.ct_node = _projection(dim), _projection(dim)
        self.ct_to_pet_edge, self.ct_to_pet_node = _projection(dim), _projection(dim)
        self.pet_to_ct_edge, self.pet_to_ct_node = _projection(dim), _projection(dim)

    @staticmethod
    def _update(nodes, participation, edge_projection, node_projection):
        edges = edge_projection(participation.transpose(1, 2) @ nodes)
        return node_projection(participation @ edges)

    def forward(self, pet, ct, return_participation=False):
        if pet.ndim != 3 or pet.shape != ct.shape:
            raise ValueError("PET/CT nodes must have identical (batch, depth, dim) shapes.")
        ap, ac = self.pet_generator(pet), self.ct_generator(ct)
        # All updates use the original nodes; neither direction depends on update order.
        pet_out = (
            pet
            + self._update(pet, ap, self.pet_edge, self.pet_node)
            + self._update(pet, ac, self.ct_to_pet_edge, self.ct_to_pet_node)
        )
        ct_out = (
            ct
            + self._update(ct, ac, self.ct_edge, self.ct_node)
            + self._update(ct, ap, self.pet_to_ct_edge, self.pet_to_ct_node)
        )
        if return_participation:
            return pet_out, ct_out, {"pet": ap, "ct": ac}
        return pet_out, ct_out
