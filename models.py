"""
models.py

The model registry. Every model regresses the 6D pose offset from the same
(B, 18, T, T) input -- the 6 rendered projections stacked on the channel axis --
and returns (B, 6). They differ in how they read those 18 channels.

Stacked (this file): the 6 views go in as one 18-channel image, so the patch
embed sums them into a single token grid.

  - VanillaViT : a compact from-scratch ViT (baseline, trained end-to-end).
  - DinoV2ViT  : DINOv2 ViT-B/14 (timm).
  - DinoV3ViT  : DINOv3 ViT-B/16 (timm).
  - SwinV2     : Swin Transformer V2-B, window 16 @ 256 (timm).

Multi-view (`models_mv.py`): the views are encoded separately by a shared local
encoder and fused by a global transformer that knows which view each token came
from -- see that module's docstring.

  - MultiViewViT (mvit) : from scratch.
  - MVDinoV2 / MVDinoV3 / MVSwinV2 : shared pretrained extractor per view.
  - MVSwinV2Low (mvswinlo) : SwinV2 tapped at its early, high-resolution stages.

The pretrained ones share freeze/fine-tune semantics via `FreezableBackbone`,
and each backbone's timm name is overridable with a VIT_HOPE_<MODEL>_BACKBONE
env var (e.g. to trade back up to ViT-L).

Each backbone has its own tile-size constraint -- see MODEL_SPECS.
"""

import os

import timm
import torch.nn as nn

from layers import FreezableBackbone, ViTTrunk, mlp_head
from models_mv import (TOKEN_GRID, MultiViewViT, MVDinoV2, MVDinoV3, MVSwinV2,
                       MVSwinV2Low)

# Per-model input geometry and provenance.
#   patch_size : the divisor `tile_size` must respect
#   tile_size  : the default tile the weights expect (drives the dataset resize)
#   pretrained : False for from-scratch models -- decides the `scratch` vs
#                `frozen`/`finetune` checkpoint tag, and whether --freeze means
#                anything. Keep this as the single source of truth; train.py,
#                infer.py and slurm/run_all_objects.sh all read it.
MODEL_SPECS = {
    "vanilla":  {"patch_size": 14, "tile_size": 518, "pretrained": False},
    "dinov2":   {"patch_size": 14, "tile_size": 518, "pretrained": True},
    "dinov3":   {"patch_size": 16, "tile_size": 512, "pretrained": True},
    # Swin V2 needs tile % (patch 4 x window 16) == 0, not just % 4.
    "swinv2":   {"patch_size": 64, "tile_size": 256, "pretrained": True},

    # --- multi-view: 6 separate 3-channel views, fused by a global transformer.
    # A smaller tile than `vanilla`: the local encoder runs 6x per sample, and a
    # from-scratch model has no pretrained prior to justify 518.
    "mvit":     {"patch_size": 16, "tile_size": 224, "pretrained": False},
    "mvdinov2": {"patch_size": 14, "tile_size": 518, "pretrained": True},
    "mvdinov3": {"patch_size": 16, "tile_size": 512, "pretrained": True},
    "mvswinv2": {"patch_size": 64, "tile_size": 256, "pretrained": True},
    "mvswinlo": {"patch_size": 64, "tile_size": 256, "pretrained": True},
}
MODEL_KINDS = list(MODEL_SPECS)

# Which architectural hyper-parameters each model actually has. A pretrained
# backbone's width/depth/heads are fixed by its weights, so the only thing left
# to tune on a multi-view pretrained model is the cross-view fuser -- and on a
# stacked pretrained model, nothing at all (just lr/dropout, which live in
# train.py and are not architecture).
#
# This is the single source of truth for the sweep: train.py builds its CLI
# group from it, sweep.py restricts its search space with it, and build_model
# validates against it. Adding a model means adding one row here.
#   embed_dim/depth/num_heads/mlp_ratio -> the from-scratch trunk (the patch
#     embedding's output width is embed_dim, i.e. the linear projection knob)
#   fuser_*                             -> layers.GlobalFuser
#   stage_dim                           -> MVSwinV2Low's per-stage projection
_TRUNK_KEYS = ("embed_dim", "depth", "num_heads", "mlp_ratio")
_FUSER_KEYS = ("fuser_embed_dim", "fuser_depth", "fuser_heads", "fuser_mlp_ratio")

ARCH_KEYS = {
    "vanilla":  _TRUNK_KEYS,
    "dinov2":   (),
    "dinov3":   (),
    "swinv2":   (),
    "mvit":     _TRUNK_KEYS + _FUSER_KEYS,
    "mvdinov2": _FUSER_KEYS,
    "mvdinov3": _FUSER_KEYS,
    "mvswinv2": _FUSER_KEYS,
    "mvswinlo": _FUSER_KEYS + ("stage_dim",),
}
# Fail at import, not at sweep time, if the two tables ever drift apart.
assert set(ARCH_KEYS) == set(MODEL_SPECS), (
    f"ARCH_KEYS/MODEL_SPECS mismatch: {set(ARCH_KEYS) ^ set(MODEL_SPECS)}")

# Every knob any model exposes, in a stable order (for CLI + CSV columns).
ALL_ARCH_KEYS = _TRUNK_KEYS + _FUSER_KEYS + ("stage_dim",)


class VanillaViT(nn.Module):
    def __init__(self, in_channels: int, img_size: int, patch_size: int,
                 out_dim: int = 6, embed_dim: int = 384, depth: int = 6,
                 num_heads: int = 6, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.trunk = ViTTrunk(in_channels, img_size, patch_size,
                              embed_dim=embed_dim, depth=depth,
                              num_heads=num_heads, mlp_ratio=mlp_ratio,
                              dropout=dropout)
        self.head = mlp_head(embed_dim, out_dim)

    def forward(self, x):
        return self.head(self.trunk(x)[:, 0])           # CLS token -> (B, out_dim)


class TimmRegressor(FreezableBackbone, nn.Module):
    """A pretrained timm backbone + a 6D regression head."""

    BACKBONE = None

    def __init__(self, in_channels: int, img_size: int, out_dim: int = 6,
                 freeze_backbone: bool = True, dropout: float = 0.1):
        super().__init__()
        # timm adapts the patch-embed conv to `in_chans` and interpolates the
        # pretrained position embeddings to img_size.
        self.backbone = timm.create_model(
            self.BACKBONE, pretrained=True, in_chans=in_channels,
            img_size=(img_size, img_size), num_classes=0)
        embed_dim = self.backbone.num_features

        self.drop = nn.Dropout(dropout)
        self.head = mlp_head(embed_dim, out_dim)        # always trainable

        self.freeze_backbone = freeze_backbone
        self.set_backbone_trainable(not freeze_backbone)

    def forward(self, x):
        feats = self.backbone(x)                        # (B, embed_dim)
        return self.head(self.drop(feats))


class DinoV2ViT(TimmRegressor):
    # ViT-B (~86M), not ViT-L (~300M): an 18-channel 518x518 input makes ViT-L
    # OOM-prone on a single GPU. Set the env var to a timm name to override,
    # e.g. VIT_HOPE_DINOV2_BACKBONE=vit_large_patch14_dinov2.lvd142m
    BACKBONE = os.environ.get("VIT_HOPE_DINOV2_BACKBONE",
                              "vit_base_patch14_dinov2.lvd142m")


class DinoV3ViT(TimmRegressor):
    # Gated on the Hugging Face hub: accept the DINOv3 licence and export
    # HF_TOKEN before the first download.
    BACKBONE = os.environ.get("VIT_HOPE_DINOV3_BACKBONE",
                              "vit_base_patch16_dinov3.lvd1689m")


class SwinV2(TimmRegressor):
    # Already base-sized, and 256x256 tiles keep it the cheapest of the three.
    BACKBONE = os.environ.get("VIT_HOPE_SWINV2_BACKBONE",
                              "swinv2_base_window16_256.ms_in1k")


_PRETRAINED = {
    "dinov2": DinoV2ViT, "dinov3": DinoV3ViT, "swinv2": SwinV2,
    "mvdinov2": MVDinoV2, "mvdinov3": MVDinoV3,
    "mvswinv2": MVSwinV2, "mvswinlo": MVSwinV2Low,
}

# Models whose constructor takes the multi-view `token_grid` knob.
_MULTI_VIEW = {"mvit", "mvdinov2", "mvdinov3", "mvswinv2", "mvswinlo"}


def is_pretrained(kind: str) -> bool:
    """Whether `kind` starts from pretrained weights (so --freeze is meaningful)."""
    try:
        return MODEL_SPECS[kind]["pretrained"]
    except KeyError:
        raise ValueError(f"Unknown model kind: {kind!r} (use one of {MODEL_KINDS}).")


def check_arch(kind: str, arch: dict = None) -> dict:
    """Validate `arch` against ARCH_KEYS[kind] and drop unset (None) entries.

    Rejecting an inapplicable key rather than ignoring it matters for the sweep:
    silently dropping `--embed_dim` on `dinov2` would produce a table of trials
    that all secretly ran the same model.
    """
    if kind not in ARCH_KEYS:
        raise ValueError(f"Unknown model kind: {kind!r} (use one of {MODEL_KINDS}).")
    arch = {k: v for k, v in (arch or {}).items() if v is not None}
    allowed = ARCH_KEYS[kind]
    bad = [k for k in arch if k not in allowed]
    if bad:
        accepts = (str(list(allowed)) if allowed else
                   "no architecture overrides -- its width, depth and head "
                   "count are fixed by the pretrained weights, so only "
                   "lr/dropout vary")
        raise ValueError(
            f"model {kind!r} has no {', '.join(sorted(bad))} knob; it accepts "
            f"{accepts}")
    return arch


def build_model(kind: str, cfg, freeze_backbone: bool = True, dropout: float = 0.1,
                token_grid: int = TOKEN_GRID, arch: dict = None):
    arch = check_arch(kind, arch)
    mv = {"token_grid": token_grid} if kind in _MULTI_VIEW else {}
    if kind == "vanilla":
        return VanillaViT(cfg.in_channels, cfg.tile_size, cfg.patch_size,
                          dropout=dropout, **arch)
    if kind == "mvit":
        return MultiViewViT(cfg.in_channels, cfg.tile_size, cfg.patch_size,
                            dropout=dropout, **mv, **arch)
    if kind in _PRETRAINED:
        return _PRETRAINED[kind](cfg.in_channels, cfg.tile_size,
                                 freeze_backbone=freeze_backbone,
                                 dropout=dropout, **mv, **arch)
    raise ValueError(f"Unknown model kind: {kind!r} (use one of {MODEL_KINDS}).")


def migrate_state_dict(kind: str, state_dict: dict) -> dict:
    """Rename keys from checkpoints written before VanillaViT gained a `trunk`.

    VanillaViT's patch/cls/pos/encoder/norm used to sit at the top level; they
    now live in a shared `ViTTrunk`. Old `vanilla_scratch` weights (the ones
    behind the current RESULTS.md numbers) would otherwise fail to load.
    """
    if kind != "vanilla" or any(k.startswith("trunk.") for k in state_dict):
        return state_dict
    legacy = ("patch.", "cls", "pos", "encoder.", "norm.")
    return {(f"trunk.{k}" if k.startswith(legacy) else k): v
            for k, v in state_dict.items()}
