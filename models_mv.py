"""
models_mv.py

Multi-view models. Where `models.py`'s backbones see the 6 projections as one
depth-stacked 18-channel image -- so the patch-embed conv sums all six into a
single token grid -- these keep the views separate:

    (B, 18, T, T) -> 6 x (3, T, T) -> shared local encoder -> per-view tokens
                  -> global transformer across views -> (B, 6)

The input contract is unchanged, so `data.py` and the train/infer/visualize
pipeline need no per-model special-casing: the reshape happens here, and it is
exact because `StackDataset` stacks the tiles view-major (data.py:83-89).

  - MultiViewViT : from scratch, a shared compact ViT per view.
  - MVDinoV2 / MVDinoV3 / MVSwinV2 : a shared *pretrained* timm extractor per
    view, frozen or fine-tuned.
  - MVSwinV2Low : SwinV2 tapped at stages 0-1 instead of the final stage --
    4x/8x reduction rather than 32x, so it keeps the fine detail that a
    millimetre-scale pose offset actually lives in.

Each backbone's timm name is overridable with VIT_HOPE_<MODEL>_BACKBONE, the
same convention as models.py.
"""

import os

import timm
import torch
import torch.nn as nn

from layers import (FreezableBackbone, GlobalFuser, ViTTrunk, pool_grid,
                    split_views, tokens_to_grid)

N_VIEWS = 6           # cfg.n_rows * cfg.n_cols
TOKEN_GRID = 2        # tokens each view contributes = TOKEN_GRID**2


class MultiViewViT(nn.Module):
    """From-scratch: one shared local ViT over each view, then a global fuser.

    Weights are shared across views (a single trunk applied to a (B*6) batch),
    which is both 6x cheaper than per-view trunks and the right inductive bias:
    the views are the same object under the same renderer, so what differs is
    *which* view a feature came from -- and that is carried by the fuser's view
    embedding, not by separate weights.
    """

    def __init__(self, in_channels: int, img_size: int, patch_size: int,
                 out_dim: int = 6, embed_dim: int = 256, depth: int = 4,
                 num_heads: int = 8, mlp_ratio: float = 4.0,
                 token_grid: int = TOKEN_GRID,
                 fuser_embed_dim: int = 512, fuser_depth: int = 4,
                 fuser_heads: int = 8, fuser_mlp_ratio: float = 4.0,
                 dropout: float = 0.1, n_views: int = N_VIEWS):
        super().__init__()
        if in_channels != n_views * 3:
            raise ValueError(f"expected {n_views * 3} input channels, got {in_channels}")
        self.n_views = n_views
        self.token_grid = token_grid

        self.local = ViTTrunk(3, img_size, patch_size, embed_dim=embed_dim,
                              depth=depth, num_heads=num_heads,
                              mlp_ratio=mlp_ratio, dropout=dropout)
        self.fuser = GlobalFuser(embed_dim, n_views, token_grid ** 2,
                                 out_dim=out_dim, embed_dim=fuser_embed_dim,
                                 depth=fuser_depth, num_heads=fuser_heads,
                                 mlp_ratio=fuser_mlp_ratio, dropout=dropout)

    def forward(self, x):
        B = x.shape[0]
        tokens = self.local(split_views(x, self.n_views))     # (B*V, 1+N, E)
        grid = tokens_to_grid(tokens, num_prefix=1)           # (B*V, E, g, g)
        pooled = pool_grid(grid, self.token_grid)             # (B*V, t, E)
        return self.fuser(pooled.reshape(B, self.n_views, pooled.shape[1], -1))


class MultiViewTimm(FreezableBackbone, nn.Module):
    """A shared pretrained timm extractor per view + the global fuser.

    The per-view analogue of `models.TimmRegressor`. The backbone is built with
    in_chans=3 (its native pretrained patch embed, not an 18-channel one
    re-inflated from it) and run on a (B*6) batch, so the pretrained weights are
    used exactly as they were trained.
    """

    BACKBONE = None

    def __init__(self, in_channels: int, img_size: int, out_dim: int = 6,
                 freeze_backbone: bool = True, token_grid: int = TOKEN_GRID,
                 fuser_embed_dim: int = 512, fuser_depth: int = 4,
                 fuser_heads: int = 8, fuser_mlp_ratio: float = 4.0,
                 dropout: float = 0.1, n_views: int = N_VIEWS, **kwargs):
        super().__init__()
        if in_channels != n_views * 3:
            raise ValueError(f"expected {n_views * 3} input channels, got {in_channels}")
        self.n_views = n_views
        self.token_grid = token_grid

        # Subclass shape knobs (e.g. MVSwinV2Low's stage_dim) are consumed before
        # _build_backbone/_post_backbone run, since those hooks read them.
        self._init_shape_opts(**kwargs)
        self.backbone = self._build_backbone(img_size)
        self._post_backbone()
        self.fuser = GlobalFuser(self._feature_dim(), n_views,
                                 self._tokens_per_view(), out_dim=out_dim,
                                 embed_dim=fuser_embed_dim, depth=fuser_depth,
                                 num_heads=fuser_heads,
                                 mlp_ratio=fuser_mlp_ratio, dropout=dropout)

        self.freeze_backbone = freeze_backbone
        self.set_backbone_trainable(not freeze_backbone)

    # ---- backbone-shape hooks, overridden by MVSwinV2Low ----
    def _init_shape_opts(self):
        """Absorb subclass-specific shape options. Base class takes none."""

    def _build_backbone(self, img_size: int):
        return timm.create_model(self.BACKBONE, pretrained=True, in_chans=3,
                                 img_size=(img_size, img_size), num_classes=0)

    def _post_backbone(self):
        """Register any submodules that depend on the backbone's shape."""

    def _feature_dim(self) -> int:
        return self.backbone.num_features

    def _tokens_per_view(self) -> int:
        return self.token_grid ** 2

    def _view_tokens(self, flat):
        """(B*V, 3, T, T) -> (B*V, tokens_per_view, feature_dim)."""
        feat = self.backbone.forward_features(flat)
        num_prefix = getattr(self.backbone, "num_prefix_tokens", 0)
        return pool_grid(tokens_to_grid(feat, num_prefix), self.token_grid)

    def forward(self, x):
        B = x.shape[0]
        tokens = self._view_tokens(split_views(x, self.n_views))
        return self.fuser(tokens.reshape(B, self.n_views, tokens.shape[1], -1))


class MVDinoV2(MultiViewTimm):
    BACKBONE = os.environ.get("VIT_HOPE_DINOV2_BACKBONE",
                              "vit_base_patch14_dinov2.lvd142m")


class MVDinoV3(MultiViewTimm):
    # Gated on the Hugging Face hub -- export HF_TOKEN before the first download.
    BACKBONE = os.environ.get("VIT_HOPE_DINOV3_BACKBONE",
                              "vit_base_patch16_dinov3.lvd1689m")


class MVSwinV2(MultiViewTimm):
    BACKBONE = os.environ.get("VIT_HOPE_SWINV2_BACKBONE",
                              "swinv2_base_window16_256.ms_in1k")


class MVSwinV2Low(MultiViewTimm):
    """SwinV2 tapped at its early stages, for small-detail sensitivity.

    The final stage is 32x-reduced (8x8 at 256px) and semantic; a pose offset of
    a few millimetres is a *sub-object* cue. Stages 0-1 are 4x/8x-reduced
    (64x64, 32x32) and keep it. Both stages are pooled to the same coarse grid
    and their tokens concatenated, so each view contributes 2 * token_grid**2
    tokens and the fuser can weigh the two scales itself.

    `features_only` returns NHWC maps and exposes no `num_features`, so the
    channel dims come from `feature_info` and a per-stage projection brings them
    to a common width.
    """

    BACKBONE = os.environ.get("VIT_HOPE_SWINV2_BACKBONE",
                              "swinv2_base_window16_256.ms_in1k")
    OUT_INDICES = (0, 1)
    STAGE_DIM = 256                # default `stage_dim`, kept as the public name

    def _init_shape_opts(self, stage_dim: int = None):
        self.stage_dim = self.STAGE_DIM if stage_dim is None else stage_dim

    def _build_backbone(self, img_size: int):
        return timm.create_model(self.BACKBONE, pretrained=True, in_chans=3,
                                 img_size=(img_size, img_size),
                                 features_only=True, out_indices=self.OUT_INDICES)

    def _post_backbone(self):
        self.stage_proj = nn.ModuleList(
            nn.Linear(c, self.stage_dim) for c in self.backbone.feature_info.channels())

    def _feature_dim(self) -> int:
        return self.stage_dim

    def _tokens_per_view(self) -> int:
        return len(self.OUT_INDICES) * self.token_grid ** 2

    def _view_tokens(self, flat):
        stages = self.backbone(flat)                          # list of (B*V, h, w, C)
        per_stage = [proj(pool_grid(tokens_to_grid(feat), self.token_grid))
                     for feat, proj in zip(stages, self.stage_proj)]
        return torch.cat(per_stage, dim=1)                    # (B*V, 2*t, STAGE_DIM)
