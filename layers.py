"""
layers.py

Building blocks shared by the single-stack models in `models.py` and the
multi-view models in `models_mv.py`. Imports nothing from either, so the
dependency graph stays acyclic:

    layers.py  <-  models_mv.py  <-  models.py (the registry)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp_head(dim: int, out_dim: int = 6) -> nn.Sequential:
    """The 6D regression head every model ends with."""
    return nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, out_dim))


def encoder_stack(embed_dim: int, depth: int, num_heads: int,
                  mlp_ratio: float = 4.0, dropout: float = 0.1) -> nn.TransformerEncoder:
    # Checked here rather than left to nn.MultiheadAttention, whose own message
    # ("embed_dim must be divisible by num_heads") arrives from three frames deep
    # with no hint of which of the two transformers in a multi-view model it came
    # from. A random hyper-parameter search hits this combination routinely.
    if embed_dim % num_heads != 0:
        raise ValueError(
            f"embed_dim ({embed_dim}) must be divisible by num_heads "
            f"({num_heads}); valid head counts here: "
            f"{[h for h in range(1, embed_dim + 1) if embed_dim % h == 0 and h <= 16]}")
    layer = nn.TransformerEncoderLayer(
        embed_dim, num_heads, int(embed_dim * mlp_ratio), dropout,
        activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, depth)


class FreezableBackbone:
    """Freeze/unfreeze semantics for a `self.backbone` submodule.

    A frozen backbone must stay in eval mode even under `model.train()`, or its
    dropout/stochastic-depth would keep perturbing features the head is trying
    to fit. Mixed in before `nn.Module` so `train()` overrides it.
    """

    def set_backbone_trainable(self, trainable: bool):
        self.freeze_backbone = not trainable
        for p in self.backbone.parameters():
            p.requires_grad = trainable
        self.backbone.train(trainable)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self


class ViTTrunk(nn.Module):
    """A compact from-scratch ViT trunk: patchify -> +CLS -> +pos -> encoder.

    Returns all tokens, `(B, 1 + n_patches, embed_dim)`, CLS first. Callers pick
    what they need -- `VanillaViT` takes the CLS, the multi-view models take the
    patch tokens and pool them per view.
    """

    def __init__(self, in_channels: int, img_size: int, patch_size: int,
                 embed_dim: int = 384, depth: int = 6, num_heads: int = 6,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.grid = img_size // patch_size
        self.embed_dim = embed_dim
        n_patches = self.grid ** 2

        self.patch = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.drop = nn.Dropout(dropout)
        self.encoder = encoder_stack(embed_dim, depth, num_heads, mlp_ratio, dropout)
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch(x).flatten(2).transpose(1, 2)      # (B, N, E)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        return self.norm(self.encoder(self.drop(x)))      # (B, 1 + N, E)


def split_views(x: torch.Tensor, n_views: int = 6) -> torch.Tensor:
    """(B, 3*V, T, T) -> (B*V, 3, T, T).

    `StackDataset._load_stack` (data.py:83-89) concatenates the V tiles on the
    channel axis in grid order, so the layout is already view-major and this is
    a pure reshape -- no data movement, and view v of sample b lands at row
    b*V + v.
    """
    B, C, H, W = x.shape
    if C != n_views * 3:
        raise ValueError(f"expected {n_views * 3} channels for {n_views} views, got {C}")
    return x.reshape(B * n_views, 3, H, W)


def tokens_to_grid(feat: torch.Tensor, num_prefix: int = 0) -> torch.Tensor:
    """Normalise a backbone's feature output to NCHW `(B, D, h, w)`.

    Handles the two layouts timm hands back:
      * ViT   `forward_features` -> `(B, num_prefix + N, D)`, tokens
      * SwinV2 `forward_features` / `features_only` -> `(B, h, w, D)`, NHWC
    """
    if feat.ndim == 4:                                    # NHWC
        return feat.permute(0, 3, 1, 2).contiguous()
    if feat.ndim != 3:
        raise ValueError(f"unexpected feature rank {feat.ndim} (shape {tuple(feat.shape)})")
    feat = feat[:, num_prefix:]                           # drop CLS / register tokens
    B, N, D = feat.shape
    side = int(math.isqrt(N))
    if side * side != N:
        raise ValueError(f"{N} patch tokens is not a square grid")
    return feat.transpose(1, 2).reshape(B, D, side, side)


def pool_grid(grid: torch.Tensor, token_grid: int) -> torch.Tensor:
    """(B, D, h, w) -> (B, token_grid**2, D), average-pooled to a coarse grid."""
    pooled = F.adaptive_avg_pool2d(grid, (token_grid, token_grid))
    return pooled.flatten(2).transpose(1, 2)


class GlobalFuser(nn.Module):
    """The cross-view transformer.

    Takes per-view token sets `(B, V, N, in_dim)`, projects them to a common
    width, tags each token with a learned *view* embedding and a learned
    *within-view position* embedding, flattens the views into one sequence,
    prepends a CLS, and runs a transformer over the lot. The CLS goes to the
    6D head.

    The two embeddings are what make this more than a concat-and-MLP: the view
    embedding tells the encoder which projection a token came from, so it can
    reason about a pose offset that is only observable by comparing views.
    """

    def __init__(self, in_dim: int, n_views: int, tokens_per_view: int,
                 out_dim: int = 6, embed_dim: int = 512, depth: int = 4,
                 num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.n_views = n_views
        self.tokens_per_view = tokens_per_view

        # LayerNorm after the projection, before the view/token embeddings are
        # added: raw feature magnitude varies ~4x across backbones (SwinV2's
        # final stage is much smaller than DINOv2's or an early Swin stage), so
        # without it a fixed-std view embedding would carry 8% of the token for
        # one backbone and 33% for another -- and the whole point of comparing
        # these variants is that the fuser is the same for all of them.
        self.proj = nn.Sequential(nn.Linear(in_dim, embed_dim),
                                  nn.LayerNorm(embed_dim))
        self.view_embed = nn.Parameter(torch.zeros(1, n_views, 1, embed_dim))
        self.tok_embed = nn.Parameter(torch.zeros(1, 1, tokens_per_view, embed_dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.drop = nn.Dropout(dropout)
        self.encoder = encoder_stack(embed_dim, depth, num_heads, mlp_ratio, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = mlp_head(embed_dim, out_dim)

        # The view/token embeddings get a much larger init than the ViT-standard
        # 0.02 used for the CLS. 0.02 is tuned for ~1k positions that a long
        # pretraining run grows into; here there are only 6 views and a short
        # fine-tune, and at 0.02 the fused output stays <1% sensitive to which
        # view a token came from -- i.e. very nearly a bag of views, which is
        # exactly what this architecture exists to avoid. At 0.25 sensitivity is
        # 12-20% with no measured cost to convergence.
        for p in (self.view_embed, self.tok_embed):
            nn.init.trunc_normal_(p, std=0.25)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):
        B, V, N, _ = x.shape
        if (V, N) != (self.n_views, self.tokens_per_view):
            raise ValueError(
                f"GlobalFuser built for {self.n_views} views x "
                f"{self.tokens_per_view} tokens, got {V} x {N}")
        x = self.proj(x) + self.view_embed + self.tok_embed
        x = x.reshape(B, V * N, -1)
        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)
        x = self.norm(self.encoder(self.drop(x)))
        return self.head(x[:, 0])
