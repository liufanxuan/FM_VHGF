"""Volumetric transfer of the recovered BiomedCLIP ViT architecture.

Attention/MLP parameter names follow timm's ViT layout so the frozen 2D weights
can be checked and transferred without importing the old experimental fork.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .hypergraph import HierarchicalHypergraphFusion


@dataclass(frozen=True)
class ModelConfig:
    input_size: tuple = (96, 224, 224)
    patch_size: tuple = (16, 16, 16)
    dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: int = 4
    num_edges: int = 8
    hypergraph_heads: int = 4
    hypergraph_dropout: float = 0.1
    classifier_hidden: int = 128
    classifier_dropout: float = 0.5
    norm_eps: float = 1e-6

    def __post_init__(self):
        object.__setattr__(self, "input_size", tuple(self.input_size))
        object.__setattr__(self, "patch_size", tuple(self.patch_size))
        if len(self.input_size) != 3 or len(self.patch_size) != 3:
            raise ValueError("input_size and patch_size must be (depth, height, width).")
        if any(s <= 0 or p <= 0 or s % p for s, p in zip(self.input_size, self.patch_size)):
            raise ValueError("Each input dimension must be a positive patch-size multiple.")
        if any(value < 1 for value in (
            self.depth, self.dim, self.num_heads, self.hypergraph_heads,
            self.num_edges, self.mlp_ratio, self.classifier_hidden,
        )):
            raise ValueError("Model dimensions and head counts must be positive.")
        if self.dim % self.num_heads or self.dim % self.hypergraph_heads:
            raise ValueError("Invalid depth or attention/head dimensions.")
        if not all(0 <= rate < 1 for rate in (
            self.hypergraph_dropout, self.classifier_dropout,
        )):
            raise ValueError("Dropout probabilities must be in [0,1).")

    def to_dict(self):
        return asdict(self)


class DualNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.pet, self.ct = nn.LayerNorm(dim, eps=eps), nn.LayerNorm(dim, eps=eps)

    def forward(self, x, modality):
        return getattr(self, modality)(x)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.num_heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch, count, dim = x.shape
        q, k, v = self.qkv(x).reshape(
            batch, count, 3, self.num_heads, dim // self.num_heads
        ).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.proj(x.transpose(1, 2).reshape(batch, count, dim))


class MLP(nn.Module):
    def __init__(self, dim, ratio):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, dim * ratio), nn.Linear(dim * ratio, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = DualNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg.dim, cfg.num_heads)
        self.norm2 = DualNorm(cfg.dim, cfg.norm_eps)
        self.mlp = MLP(cfg.dim, cfg.mlp_ratio)

    def forward(self, x, modality):
        x = x + self.attn(self.norm1(x, modality))
        return x + self.mlp(self.norm2(x, modality))


class VolumetricEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.patch_embed_pet = nn.Conv3d(3, cfg.dim, cfg.patch_size, stride=cfg.patch_size)
        self.patch_embed_ct = nn.Conv3d(3, cfg.dim, cfg.patch_size, stride=cfg.patch_size)
        grid = [s // p for s, p in zip(cfg.input_size, cfg.patch_size)]
        count = grid[0] * grid[1] * grid[2]
        self.cls_token_pet = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.cls_token_ct = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.reg_token_pet = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.reg_token_ct = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.pos_embed = nn.Parameter(torch.empty(1, count + 2, cfg.dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for token in (self.cls_token_pet, self.cls_token_ct,
                      self.reg_token_pet, self.reg_token_ct):
            nn.init.trunc_normal_(token, std=0.02)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.depth)])
        self.norm = DualNorm(cfg.dim, cfg.norm_eps)
        # Frozen layers still participate in autograd so gradients reach trainable
        # projections and LayerNorms. Do not wrap this encoder in torch.no_grad().
        for block in self.blocks:
            for module in (block.attn, block.mlp):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    def forward_branch(self, volume, modality):
        if volume.ndim != 5 or tuple(volume.shape[2:]) != self.config.input_size:
            raise ValueError(f"Expected (B,C,{self.config.input_size}), got {tuple(volume.shape)}")
        if volume.shape[1] == 1:
            volume = volume.repeat(1, 3, 1, 1, 1)
        elif volume.shape[1] != 3:
            raise ValueError("Volumes must have one channel or three replicated channels.")
        x = getattr(self, "patch_embed_" + modality)(volume).flatten(2).transpose(1, 2)
        cls = getattr(self, "cls_token_" + modality).expand(x.shape[0], -1, -1)
        reg = getattr(self, "reg_token_" + modality).expand(x.shape[0], -1, -1)
        x = torch.cat((cls, reg, x), dim=1) + self.pos_embed
        descriptors = []
        for block in self.blocks:
            x = block(x, modality)
            readout = self.norm(x, modality)
            descriptors.append(readout[:, 0] + readout[:, 1])
        return torch.stack(descriptors, dim=1)

    def forward(self, pet, ct):
        if pet.shape != ct.shape:
            raise ValueError("Paired PET and CT must have identical shapes.")
        return self.forward_branch(pet, "pet"), self.forward_branch(ct, "ct")


class FMVHGF(nn.Module):
    """forward() returns one raw logit per patient; sigmoid is applied for prediction."""

    def __init__(self, config=None):
        super().__init__()
        cfg = config if config is not None else ModelConfig()
        self.config = cfg
        self.encoder = VolumetricEncoder(cfg)
        self.fusion = HierarchicalHypergraphFusion(
            cfg.dim, cfg.num_edges, cfg.hypergraph_heads, cfg.hypergraph_dropout
        )
        self.pool = nn.Conv1d(cfg.dim, cfg.dim, 2 * cfg.depth, groups=cfg.dim, bias=False)
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.classifier_dropout),
            nn.Linear(cfg.dim, cfg.classifier_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.classifier_dropout),
            nn.Linear(cfg.classifier_hidden, 1),
        )

    def forward(self, pet, ct, return_participation=False):
        pet_nodes, ct_nodes = self.encoder(pet, ct)
        if return_participation:
            pet_nodes, ct_nodes, weights = self.fusion(
                pet_nodes, ct_nodes, return_participation=True
            )
        else:
            pet_nodes, ct_nodes = self.fusion(pet_nodes, ct_nodes)
        nodes = torch.cat((pet_nodes, ct_nodes), dim=1)
        pooled = self.pool(nodes.transpose(1, 2)).squeeze(-1)
        logits = self.classifier(pooled).squeeze(-1)
        return (logits, weights) if return_participation else logits

    @torch.no_grad()
    def predict_proba(self, pet, ct):
        if self.training:
            raise RuntimeError("Call model.eval() before predicting probabilities.")
        return self(pet, ct).sigmoid()

    def parameter_counts(self):
        def count(module):
            return {"total": sum(p.numel() for p in module.parameters()),
                    "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad)}
        return {"encoder": count(self.encoder), "whole_model": count(self)}
