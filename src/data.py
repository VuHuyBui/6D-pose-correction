"""
data.py

StackDataset: one jittered sample -> an 18-channel tensor + its 6D target.

Each sample has 6 orthographic tiles rendered by the C++ renderer:
    <data_dir>/<output_folder>/<key>_tile-w={0,1}_tile-h={0,1,2}.png
The 6 RGB tiles are cropped to a single shared object box, resized to (T, T),
and concatenated on the channel axis -> (18, T, T) ("stacked in depth").

Augmentation (train only): one random colour-jitter (brightness/contrast/
saturation/hue) sampled per sample and applied *identically* to all 6 tiles,
so the joint appearance stays consistent across views.

Target: the 6D offset [dt_x,dt_y,dt_z, dr_x,dr_y,dr_z] scaled by JITTER_SCALE.
"""

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from config import Config, JITTER_COLUMNS, NORM_MEAN, NORM_STD

# colour-jitter ranges (match vit_jitter's stacked pipeline)
_BCS = 0.10          # brightness / contrast / saturation +/-
_HUE = 0.08          # hue +/-


class StackDataset(Dataset):
    def __init__(self, csv_path: str, data_dir: str, cfg: Config,
                 augment: bool = False):
        self.cfg = cfg
        self.data_dir = data_dir
        self.augment = augment
        self.to_tensor = transforms.ToTensor()

        df = pd.read_csv(csv_path)
        # keep only rows whose 6 tiles all exist on disk
        self.rows = [r for _, r in df.iterrows() if self._all_tiles_exist(r)]
        if not self.rows:
            raise RuntimeError(f"No renderable samples found under {data_dir}.")
        n_missing = len(df) - len(self.rows)
        if n_missing:
            print(f"[StackDataset] skipped {n_missing} rows with missing tiles.")

        self.crop_box = self._compute_shared_crop_box(self.rows[0])

    # ---- tile locations ----
    def _tile_paths(self, row):
        folder = os.path.join(self.data_dir, row["output_folder"])
        return [os.path.join(folder, f"{row['key']}_tile-w={r}_tile-h={c}.png")
                for r in range(self.cfg.n_rows) for c in range(self.cfg.n_cols)]

    def _all_tiles_exist(self, row):
        return all(os.path.exists(p) for p in self._tile_paths(row))

    # ---- shared object crop (computed once, reused everywhere) ----
    def _compute_shared_crop_box(self, row, thresh=8, margin=16):
        l = t = np.inf
        r = b = -np.inf
        for path in self._tile_paths(row):
            gray = np.asarray(Image.open(path).convert("L"))
            ys, xs = np.where(gray > thresh)
            if xs.size == 0:
                continue
            l, t = min(l, xs.min()), min(t, ys.min())
            r, b = max(r, xs.max()), max(b, ys.max())
        if not np.isfinite(l):          # all-black fallback: whole image
            with Image.open(self._tile_paths(row)[0]) as im:
                return (0, 0, im.width, im.height)
        with Image.open(self._tile_paths(row)[0]) as im:
            W, H = im.width, im.height
        return (max(0, int(l) - margin), max(0, int(t) - margin),
                min(W, int(r) + margin), min(H, int(b) + margin))

    # ---- stack builder ----
    def _load_stack(self, row) -> torch.Tensor:
        tiles = []
        for path in self._tile_paths(row):
            img = Image.open(path).convert("RGB").crop(self.crop_box)
            img = img.resize((self.cfg.tile_size, self.cfg.tile_size))
            tiles.append(self.to_tensor(img))          # (3, T, T) in [0,1]
        return torch.cat(tiles, dim=0)                 # (18, T, T)

    def _joint_color_jitter(self, stack: torch.Tensor) -> torch.Tensor:
        b = 1 + (torch.rand(1).item() * 2 - 1) * _BCS
        c = 1 + (torch.rand(1).item() * 2 - 1) * _BCS
        s = 1 + (torch.rand(1).item() * 2 - 1) * _BCS
        h = (torch.rand(1).item() * 2 - 1) * _HUE
        out = []
        for i in range(0, stack.shape[0], 3):
            tile = stack[i:i + 3]
            tile = TF.adjust_brightness(tile, b)
            tile = TF.adjust_contrast(tile, c)
            tile = TF.adjust_saturation(tile, s)
            tile = TF.adjust_hue(tile, h)
            out.append(tile)
        return torch.cat(out, dim=0).clamp(0, 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        stack = self._load_stack(row)
        if self.augment:
            stack = self._joint_color_jitter(stack)
        stack = (stack - NORM_MEAN) / NORM_STD

        target = torch.tensor(
            [row[c] for c in JITTER_COLUMNS], dtype=torch.float32
        ) * self.cfg.jitter_scale
        return stack, target
