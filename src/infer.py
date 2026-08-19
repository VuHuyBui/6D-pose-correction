"""
infer.py

Run a trained checkpoint on the held-out test split and write predicted 6D
jitter offsets (plus the jittered absolute poses needed for correction).

No rendering — safe to run on a GPU compute node.

Where the predictions CSV lands, in order of precedence:
    1. --out_dir
    2. $VIT_HOPE_OUT_ROOT/<obj_stem>   (the cluster case)
    3. Config.results_dir (vit_hope/results)

Examples:
    python infer.py --weights weights/obj_000006_dinov2_frozen.pt
    python infer.py --weights /data/me/vit_hope_rendered/obj_000002/obj_000002_dinov2_frozen.pt \\
        --data_dir /data/me/vit_hope_rendered/obj_000002 \\
        --out_dir  /data/me/vit_hope_rendered/obj_000002
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from config import Config, JITTER_COLUMNS
from data import StackDataset
from models import MODEL_SPECS, build_model, is_pretrained, migrate_state_dict
from models_mv import TOKEN_GRID

POSE_COLUMNS = [
    "tvec_x", "tvec_y", "tvec_z",
    "rvec_x", "rvec_y", "rvec_z",
]


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument(
        "--data_dir",
        default=None,
        help="Prepared render directory (jitter_all.csv + run_* tiles). "
             "Default: Config.output_dir from the checkpoint's obj_id.",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Where to write the predictions CSV. Default: "
             "$VIT_HOPE_OUT_ROOT/<obj_stem> if set, else vit_hope/results.",
    )
    p.add_argument(
        "--out_name",
        default=None,
        help="Predictions filename. Default: predictions_<model>_<tag>.csv "
             "derived from the checkpoint, or predictions.csv.",
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    spec = MODEL_SPECS[ck["model"]]
    cfg = Config(obj_id=ck["obj_id"], tile_size=ck["tile_size"],
                 patch_size=ck.get("patch_size", spec["patch_size"]))
    # `arch` is absent from checkpoints written before the hyper-parameter sweep;
    # .get keeps those loading with the built-in widths they were trained at.
    model = build_model(ck["model"], cfg, freeze_backbone=ck.get("freeze", True),
                        token_grid=ck.get("token_grid", TOKEN_GRID),
                        arch=ck.get("arch"))
    model.load_state_dict(migrate_state_dict(ck["model"], ck["state_dict"]))
    model.to(device).eval()
    return model, cfg, ck


def ckpt_tag(ck):
    """Match train.py's weight naming: scratch | frozen | finetune."""
    kind = ck["model"]
    if not is_pretrained(kind):
        return kind, "scratch"
    return kind, ("frozen" if ck.get("freeze", True) else "finetune")


def main():
    args = get_args()
    device = torch.device(args.device)
    model, cfg, ck = load_model(args.weights, device)

    data_dir = os.path.abspath(args.data_dir or cfg.output_dir)
    out_dir = os.path.abspath(args.out_dir or cfg.obj_out_dir)
    os.makedirs(out_dir, exist_ok=True)

    model_name, tag = ckpt_tag(ck)
    out_name = args.out_name or f"predictions_{model_name}_{tag}.csv"
    out_path = os.path.join(out_dir, out_name)

    jitter_csv = os.path.join(data_dir, "jitter_all.csv")
    if not os.path.isfile(jitter_csv):
        raise FileNotFoundError(f"Missing jitter CSV: {jitter_csv}")

    ds = StackDataset(jitter_csv, data_dir, cfg, augment=False)
    idx = ck.get("test_indices") or list(range(len(ds)))
    loader = DataLoader(Subset(ds, idx), batch_size=args.batch_size,
                        num_workers=args.workers)

    print(f"Inferring {len(idx)} test samples "
          f"(model={ck['model']} freeze={ck.get('freeze')} device={device})")
    print(f"data={data_dir}")

    preds = []
    with torch.no_grad():
        for x, _ in loader:
            preds.append(model(x.to(device)).cpu())
    offsets = torch.cat(preds).numpy() / cfg.jitter_scale  # (N, 6) unscaled

    rows = [ds.rows[i] for i in idx]
    true = np.array([[float(r[c]) for c in JITTER_COLUMNS] for r in rows],
                    dtype=np.float64)
    mse = float(np.mean((offsets - true) ** 2))
    mae = float(np.mean(np.abs(offsets - true)))
    print(f"offset MSE {mse:.6e}  MAE {mae:.6e}")

    records = []
    for r, off in zip(rows, offsets):
        rec = {
            "output_folder": r["output_folder"],
            "key": r["key"],
            **{c: float(r[c]) for c in POSE_COLUMNS},
            **{f"true_{c}": float(r[c]) for c in JITTER_COLUMNS},
            **{f"pred_{c}": float(o) for c, o in zip(JITTER_COLUMNS, off)},
        }
        records.append(rec)

    # Write-then-rename: os.replace is atomic on the same filesystem, so a job
    # killed at the walltime can never leave a truncated CSV behind. The sweep
    # scripts treat "predictions CSV exists" as "this object is done", and a
    # half-written file would make them skip an object that never finished.
    tmp_path = out_path + ".tmp"
    pd.DataFrame(records).to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
