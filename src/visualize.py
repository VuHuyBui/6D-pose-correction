"""
visualize.py

Consume predicted jitter from infer.py, correct poses, re-render, score, and
build side-by-side comparison images:

  1. Load predictions.csv (pred_* offsets + jittered tvec_*/rvec_*).
  2. Correct subtractively:  corrected_pose = jittered_pose - predicted_offset.
  3. Render clean (GT) poses once and corrected poses per run.
  4. Score corrected-vs-clean against the jittered-vs-clean baseline
     (MSE / MAE / masked SSIM over the 6 tiles) -> metrics.csv.
     SSIM is restricted to non-black pixels (union mask over the two images),
     matching vit_jitter's compute_masked_ssim: the tiles are an object on a
     pure black background, so an unmasked SSIM is dominated by empty space.
  5. For each test sample, write comparisons/{run}__{key}.png with three
     columns: Original | Ground Truth | Corrected (each a 2x3 tile montage).

Everything produced here goes to Config.results_dir (vit_hope/results). Only the
jittered *input* tiles are read from the renderer's own results/hope tree.

The model/tag is read off the `predictions_<model>_<tag>.csv` filename (override
with --model/--tag) and suffixes every output, so all four backbones can be
visualized into the same directory without clobbering each other:

    corrected_poses_<model>_<tag>.json   metrics_<model>_<tag>.csv
    corrected_<model>_<tag>/             comparisons_<model>_<tag>/

`clean/` is the ground-truth render, which does not depend on the model, so it
is shared and re-rendered only when missing (--force_clean to redo it).

Usage:
    python visualize.py --predictions results/predictions_dinov2_frozen.csv
    python visualize.py --predictions results/predictions_swinv2_frozen.csv
    python visualize.py --predictions preds.csv --model dinov3 --tag frozen
    python visualize.py --predictions preds.csv --no_render --no_comparisons
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

from config import Config, JITTER_COLUMNS
from models import MODEL_KINDS, MODEL_SPECS

POSE_COLUMNS = [
    "tvec_x", "tvec_y", "tvec_z",
    "rvec_x", "rvec_y", "rvec_z",
]

SSIM_THRESHOLD = 5      # non-black cutoff on the 0-255 scale (matches vit_jitter)
SSIM_KERNEL = 7         # uniform window, matches skimage's default

TAGS = ("scratch", "frozen", "finetune")   # train.py's checkpoint tags


def detect_variant(pred_path, model=None, tag=None):
    """(model, tag) for a predictions_<model>_<tag>.csv, or (None, None).

    Explicit --model/--tag win; anything not recoverable stays None and the
    outputs simply go unsuffixed (the pre-multi-model behaviour).
    """
    stem = os.path.splitext(os.path.basename(pred_path))[0]
    m = re.fullmatch(rf"predictions_([a-z0-9]+)_({'|'.join(TAGS)})", stem)
    if m and m.group(1) in MODEL_KINDS:
        model, tag = model or m.group(1), tag or m.group(2)
    elif model is None:
        print(f"[warn] cannot infer the model from {stem!r} -- outputs will be "
              f"unsuffixed and may overwrite another model's. Pass --model.")
    return model, tag


def variant_suffix(model, tag):
    return f"_{model}_{tag}" if model and tag else (f"_{model}" if model else "")


# --------------------------------------------------------------------------- #
# rendering helpers
# --------------------------------------------------------------------------- #
def want_xvfb(no_xvfb):
    return (not no_xvfb) and (not os.environ.get("DISPLAY")) and bool(shutil.which("xvfb-run"))


def render(cfg, payload, out_path, use_xvfb):
    """Run the C++ renderer, writing tiles into `out_path` (absolute).

    The renderer treats its argv[1] as a plain path prefix, so it happily writes
    outside its own repo; cwd stays at renderer_repo because the asset paths in
    the payload are relative to it.
    """
    os.makedirs(out_path, exist_ok=True)
    cmd = [f"./{cfg.renderer_binary}", out_path]
    if use_xvfb:
        cmd = ["xvfb-run", "-a", *cmd]
    res = subprocess.run(cmd, input=json.dumps(payload), text=True,
                         capture_output=True, cwd=cfg.renderer_repo)
    if res.returncode != 0:
        print(res.stderr[-1500:])
        raise RuntimeError(f"Renderer failed for {out_path}")


def correct_from_predictions(pred_df):
    """corrected pose = jittered pose - predicted offset; grouped by run folder."""
    by_run = {}
    for _, row in pred_df.iterrows():
        off = [float(row[f"pred_{c}"]) for c in JITTER_COLUMNS]
        tvec = [float(row["tvec_x"]) - off[0],
                float(row["tvec_y"]) - off[1],
                float(row["tvec_z"]) - off[2]]
        rvec = [float(row["rvec_x"]) - off[3],
                float(row["rvec_y"]) - off[4],
                float(row["rvec_z"]) - off[5]]
        by_run.setdefault(row["output_folder"], {})[row["key"]] = {
            "rvec": [[float(rvec[0])], [float(rvec[1])], [float(rvec[2])]],
            "tvec": [float(v) for v in tvec],
        }
    return by_run


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def load_tile(path):
    """PNG -> (3, H, W) float tensor in [0, 1]."""
    arr = np.array(Image.open(path).convert("RGB"))   # copy: from_numpy needs writable
    return torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)


def masked_ssim(pred, ref, threshold):
    """Per-tile SSIM averaged over non-black pixels only (union mask).

    pred/ref: (B, 3, H, W) in [0, 1]. Returns (ssim_per_tile, masked_px_count);
    the SSIM is NaN for tiles whose mask is empty (both images fully black).

    torchmetrics reflect-pads before a valid convolution, so the returned map has
    the same HxW as the input and the mask aligns pixel-for-pixel. The map is
    per-channel, so a flat mean over a channel-broadcast mask reproduces
    skimage's ssim_map[mask].mean() with channel_axis=-1.
    """
    _, full = ssim_fn(pred, ref, gaussian_kernel=False, kernel_size=SSIM_KERNEL,
                      data_range=1.0, return_full_image=True, reduction=None)
    mask = ((pred.amax(1) > threshold) | (ref.amax(1) > threshold)).unsqueeze(1)
    mask = mask.expand_as(full)
    n = mask.sum((1, 2, 3))
    s = (full * mask).sum((1, 2, 3)) / n.clamp(min=1)
    return torch.where(n > 0, s, torch.full_like(s, float("nan"))), n


def masked_ssim_chunked(pred, ref, threshold, batch):
    """masked_ssim over chunks of `batch` tiles, halving on CUDA OOM."""
    while True:
        try:
            outs = [masked_ssim(pred[i:i + batch], ref[i:i + batch], threshold)
                    for i in range(0, pred.shape[0], batch)]
            return (torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs]))
        except torch.OutOfMemoryError:
            if batch == 1:
                raise
            batch = max(1, batch // 2)
            torch.cuda.empty_cache()
            print(f"  [ssim] CUDA OOM -> retrying with --ssim_batch {batch}")


def sample_score(cfg, pred_dir, ref_dir, key, device, ssim_batch, threshold):
    """Mean MSE/MAE/masked-SSIM over the tiles of one sample.

    Returns (mse, mae, ssim_masked, n_tiles, mask_frac) or None if no tile pair
    exists / every tile's mask is empty.
    """
    preds, refs = [], []
    for r in range(cfg.n_rows):
        for c in range(cfg.n_cols):
            name = f"{key}_tile-w={r}_tile-h={c}.png"
            p, q = os.path.join(pred_dir, name), os.path.join(ref_dir, name)
            if not (os.path.exists(p) and os.path.exists(q)):
                continue
            a, b = load_tile(p), load_tile(q)
            if a.shape != b.shape:
                b = F.interpolate(b.unsqueeze(0), size=a.shape[-2:],
                                  mode="bilinear", align_corners=False).squeeze(0)
            preds.append(a); refs.append(b)
    if not preds:
        return None

    pred = torch.stack(preds).to(device, non_blocking=True)
    ref = torch.stack(refs).to(device, non_blocking=True)
    diff = pred - ref
    mse = diff.pow(2).mean((1, 2, 3)).mean().item()
    mae = diff.abs().mean((1, 2, 3)).mean().item()

    ssim, n_px = masked_ssim_chunked(pred, ref, threshold, ssim_batch)
    ssim = ssim.cpu().numpy()
    if np.isnan(ssim).all():
        return None
    # mask covers 3 channels; report the fraction of (H, W) positions selected.
    mask_frac = float(n_px.sum().item()) / float(pred.numel())
    return mse, mae, float(np.nanmean(ssim)), pred.shape[0], mask_frac


def print_ssim_summary(mdf, n_requested, threshold, label=""):
    """Masked SSIM aggregated across all scored samples -> terminal."""
    print(f"\n[{label}] Masked SSIM (non-black pixels, threshold={threshold}, "
          f"uniform {SSIM_KERNEL}x{SSIM_KERNEL} window)")
    print(f"  samples scored : {len(mdf)} / {n_requested}")
    if not len(mdf):
        print("  nothing to report -- no sample had a scorable tile pair.")
        return

    j, c = mdf.ssim_masked_jittered, mdf.ssim_masked_corrected
    print(f"  tiles per sample: {mdf.n_tiles.min()}-{mdf.n_tiles.max()}   "
          f"mean mask frac: {mdf.mask_frac_jittered.mean():.3f} (jittered) / "
          f"{mdf.mask_frac_corrected.mean():.3f} (corrected)")
    print(f"  {'':<10} {'jittered':>10} {'corrected':>11} {'delta':>10}")
    # the delta column describes the per-sample delta distribution, not the
    # difference of the jittered/corrected column statistics.
    for fn in ("mean", "std", "median", "min", "max"):
        jv, cv, dv = (getattr(x, fn)() for x in (j, c, mdf.ssim_masked_delta))
        print(f"  {fn:<10} {jv:10.4f} {cv:11.4f} {dv:+10.4f}")
    n_up = int((mdf.ssim_masked_delta > 0).sum())
    print(f"  {'improved':<10} {n_up}/{len(mdf)} ({100.0 * n_up / len(mdf):.1f}%)")
    print(f"  MSE {mdf.mse_jittered.mean():.5f} -> {mdf.mse_corrected.mean():.5f}   "
          f"MAE {mdf.mae_jittered.mean():.5f} -> {mdf.mae_corrected.mean():.5f}")


# --------------------------------------------------------------------------- #
# side-by-side collages
# --------------------------------------------------------------------------- #
def tile_paths(cfg, folder, key):
    return [os.path.join(folder, f"{key}_tile-w={r}_tile-h={c}.png")
            for r in range(cfg.n_rows) for c in range(cfg.n_cols)]


def montage_tiles(cfg, folder, key):
    """Load 6 tiles into a 2x3 grid image. Returns None if any tile is missing."""
    paths = tile_paths(cfg, folder, key)
    if not all(os.path.exists(p) for p in paths):
        return None
    imgs = [Image.open(p).convert("RGB") for p in paths]
    w, h = imgs[0].size
    grid = Image.new("RGB", (w * cfg.n_cols, h * cfg.n_rows), (0, 0, 0))
    for i, img in enumerate(imgs):
        if img.size != (w, h):
            img = img.resize((w, h), Image.Resampling.BILINEAR)
        r, c = divmod(i, cfg.n_cols)
        grid.paste(img, (c * w, r * h))
    return grid


def make_comparison(cfg, orig_dir, clean_dir, corr_dir, key, out_path,
                    corr_label="Corrected", label_h=36, gap=8):
    """Original | Ground Truth | Corrected side-by-side collage."""
    panels = []
    for folder in (orig_dir, clean_dir, corr_dir):
        m = montage_tiles(cfg, folder, key)
        if m is None:
            return False
        panels.append(m)

    target = panels[0].size
    panels = [p if p.size == target else p.resize(target, Image.Resampling.BILINEAR)
              for p in panels]

    pw, ph = target
    canvas_w = 3 * pw + 2 * gap
    canvas_h = label_h + ph
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    labels = ("Original", "Ground Truth", corr_label)
    for i, (panel, label) in enumerate(zip(panels, labels)):
        x = i * (pw + gap)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (pw - tw) // 2, (label_h - (bbox[3] - bbox[1])) // 2),
                  label, fill=(0, 0, 0), font=font)
        canvas.paste(panel, (x, label_h))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)
    return True


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", required=True,
                   help="CSV from infer.py (pred_* + tvec_*/rvec_* columns).")
    p.add_argument("--obj_id", type=int, default=Config.obj_id)
    p.add_argument("--model", choices=MODEL_KINDS, default=None,
                   help="Backbone that produced the CSV. Default: read off the "
                        "predictions_<model>_<tag>.csv filename.")
    p.add_argument("--tag", choices=TAGS, default=None,
                   help="Checkpoint tag. Default: read off the filename.")
    p.add_argument("--tile_size", type=int, default=None,
                   help="Default: the model's native tile size. Affects nothing "
                        "here but the Config sanity check.")
    p.add_argument(
        "--data_dir",
        default=None,
        help="Prepared render directory (jittered tiles). Default: Config.output_dir.",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Where everything this script produces goes: corrected_poses.json, "
             "metrics.csv, comparisons/, and the clean/ and corrected/ renders. "
             "Default: Config.results_dir (vit_hope/results).",
    )
    p.add_argument("--no_render", action="store_true",
                   help="Skip rendering; use existing clean/ and corrected/ tiles.")
    p.add_argument("--force_clean", action="store_true",
                   help="Re-render clean/ even if it already exists. It is the "
                        "GT render, shared by every model, so it is normally "
                        "rendered once and reused.")
    p.add_argument("--no_comparisons", action="store_true",
                   help="Skip the side-by-side collages (metrics-only run).")
    p.add_argument("--no_xvfb", action="store_true")
    p.add_argument("--device", default=None,
                   help="Torch device for the metrics. Default: cuda if available, else cpu.")
    p.add_argument("--ssim_batch", type=int, default=6,
                   help="Tiles per SSIM call; bounds VRAM. Halves automatically on OOM.")
    p.add_argument("--ssim_threshold", type=int, default=SSIM_THRESHOLD,
                   help="Non-black cutoff on the 0-255 scale for the SSIM mask.")
    args = p.parse_args()

    model, tag = detect_variant(args.predictions, args.model, args.tag)
    suffix = variant_suffix(model, tag)
    spec = MODEL_SPECS.get(model, MODEL_SPECS["dinov2"])

    cfg = Config(obj_id=args.obj_id,
                 tile_size=args.tile_size or spec["tile_size"],
                 patch_size=spec["patch_size"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    threshold = args.ssim_threshold / 255.0
    data_dir = os.path.abspath(args.data_dir or cfg.output_dir)
    out_dir = os.path.abspath(args.out_dir or cfg.results_dir)
    os.makedirs(out_dir, exist_ok=True)
    use_xvfb = want_xvfb(args.no_xvfb) and not args.no_render

    # Renders land in out_dir alongside the metrics; only the jittered input
    # tiles (data_dir) live in the renderer's tree. clean/ is the GT render and
    # is model-independent, so it stays unsuffixed and is shared.
    clean_dir = os.path.join(out_dir, "clean")
    corr_root = os.path.join(out_dir, f"corrected{suffix}")

    pred_df = pd.read_csv(args.predictions)
    required = (["output_folder", "key"] + POSE_COLUMNS
                + [f"pred_{c}" for c in JITTER_COLUMNS])
    missing = [c for c in required if c not in pred_df.columns]
    if missing:
        raise ValueError(f"predictions CSV missing columns: {missing}")

    print(f"Visualizing {len(pred_df)} samples from {args.predictions}")
    print(f"model={model or 'unknown'} tag={tag or 'unknown'} obj={cfg.obj_stem}")
    print(f"data={data_dir}  out={out_dir}  device={device}")

    by_run = correct_from_predictions(pred_df)
    corrected_json = os.path.join(out_dir, f"corrected_poses{suffix}.json")
    with open(corrected_json, "w") as f:
        json.dump(by_run, f, indent=2)
    print(f"wrote {corrected_json}")

    if not args.no_render:
        # Shared GT render: skip it if another model's run already produced it.
        if args.force_clean or not glob.glob(os.path.join(clean_dir, "*.png")):
            request_path = os.path.join(data_dir, "request.json")
            if not os.path.isfile(request_path):
                request_path = cfg.request_json_path
            with open(request_path) as f:
                clean_payload = json.load(f)
            render(cfg, clean_payload, clean_dir, use_xvfb)
        else:
            print(f"reusing existing clean render at {clean_dir} (--force_clean to redo)")

        for run_folder, data in by_run.items():
            payload = cfg.render_payload_header()
            payload["data"] = data
            render(cfg, payload, os.path.join(corr_root, run_folder), use_xvfb)

    # --- metrics ---
    records = []
    for _, r in pred_df.iterrows():
        key, run = r["key"], r["output_folder"]
        corr = sample_score(cfg, os.path.join(corr_root, run), clean_dir, key,
                            device, args.ssim_batch, threshold)
        base_s = sample_score(cfg, os.path.join(data_dir, run), clean_dir, key,
                              device, args.ssim_batch, threshold)
        if corr is None or base_s is None:
            continue
        records.append({
            # model/tag/obj travel with the rows so the four metrics CSVs can be
            # concatenated into one cross-model table.
            "model": model or "unknown", "tag": tag or "unknown",
            "obj_id": cfg.obj_id,
            "run": run, "key": key,
            "mse_jittered": base_s[0], "mse_corrected": corr[0],
            "mae_jittered": base_s[1], "mae_corrected": corr[1],
            "ssim_masked_jittered": base_s[2], "ssim_masked_corrected": corr[2],
            "ssim_masked_delta": corr[2] - base_s[2],
            "n_tiles": corr[3],
            "mask_frac_jittered": base_s[4], "mask_frac_corrected": corr[4],
        })
    mdf = pd.DataFrame(records)
    metrics_path = os.path.join(out_dir, f"metrics{suffix}.csv")
    mdf.to_csv(metrics_path, index=False)
    print_ssim_summary(mdf, len(pred_df), args.ssim_threshold,
                       label=f"{model or 'unknown'}/{tag or 'unknown'}")
    print(f"wrote {metrics_path}")

    # --- side-by-side comparisons ---
    if args.no_comparisons:
        print("skipped comparison images (--no_comparisons)")
    else:
        cmp_dir = os.path.join(out_dir, f"comparisons{suffix}")
        corr_label = f"Corrected ({model})" if model else "Corrected"
        n_ok = 0
        for _, r in pred_df.iterrows():
            key, run = r["key"], r["output_folder"]
            out_path = os.path.join(cmp_dir, f"{run}__{key}.png")
            if make_comparison(cfg,
                               os.path.join(data_dir, run),
                               clean_dir,
                               os.path.join(corr_root, run),
                               key, out_path, corr_label=corr_label):
                n_ok += 1
        print(f"wrote {n_ok}/{len(pred_df)} comparison images under {cmp_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
