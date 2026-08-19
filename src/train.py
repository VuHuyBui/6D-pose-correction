"""
train.py

Train any backbone to regress the 6D pose offset (MSE on the JITTER_SCALE-scaled
target). Same knobs for every model; the pretrained ones add freeze/unfreeze.

Examples:
    python train.py --model dinov2 --freeze --overfit_test   # wiring sanity check
    python train.py --model dinov2 --freeze                  # frozen backbone
    python train.py --model dinov3 --no_freeze --lr 3e-5     # full fine-tune
    python train.py --model swinv2 --freeze                  # Swin V2 @ 256
    python train.py --model vanilla                          # from-scratch baseline

Multi-view variants (views encoded separately, then fused -- see models_mv.py):
    python train.py --model mvit                             # from-scratch baseline
    python train.py --model mvdinov2 --freeze                # shared DINOv2 per view
    python train.py --model mvdinov2 --no_freeze             # ... fine-tuned
    python train.py --model mvswinlo --freeze                # SwinV2 early stages

Architecture overrides (unset = the model's built-in default, so a plain run is
byte-identical to before these flags existed). Which ones a model accepts is
models.ARCH_KEYS -- passing an inapplicable one is an error, not a no-op:
    python train.py --model vanilla --embed_dim 512 --num_heads 8 --depth 8
    python train.py --model mvit --embed_dim 192 --fuser_embed_dim 256

--tile_size defaults to whatever the chosen backbone expects (MODEL_SPECS).
"""

import argparse
import csv
import os
import time

import torch
from torch.utils.data import DataLoader, Subset

from config import Config
from data import StackDataset
from models import (ALL_ARCH_KEYS, ARCH_KEYS, MODEL_KINDS, MODEL_SPECS,
                    build_model, check_arch, is_pretrained)
from models_mv import TOKEN_GRID

# Written by --result_csv, one row per finished run. One schema for every model:
# a knob the model does not have is left blank rather than omitted, so sweeps of
# different models concatenate cleanly.
RESULT_COLUMNS = (
    ["run_tag", "model", "tag", "obj_id", "n_train", "n_val", "n_test",
     "epochs", "batch_size", "lr", "dropout"]
    + list(ALL_ARCH_KEYS)
    + ["token_grid", "tile_size", "seed", "best_val", "best_epoch",
       "final_train", "n_params", "seconds", "ckpt"]
)

# int-valued arch flags; mlp_ratio knobs are floats.
_ARCH_FLOAT = {"mlp_ratio", "fuser_mlp_ratio"}


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--obj_id", type=int, default=Config.obj_id)
    p.add_argument("--model", choices=MODEL_KINDS, default="dinov2")
    p.add_argument("--freeze", dest="freeze", action="store_true", default=True,
                   help="Freeze the pretrained backbone (default).")
    p.add_argument("--no_freeze", dest="freeze", action="store_false",
                   help="Fine-tune the whole pretrained backbone.")
    p.add_argument("--lr", type=float, default=None,
                   help="Default: 3e-5 for an unfrozen pretrained backbone, else 3e-4.")
    p.add_argument("--tile_size", type=int, default=None,
                   help="Default: the chosen backbone's native tile size.")
    p.add_argument("--token_grid", type=int, default=TOKEN_GRID,
                   help="Multi-view models only: each view contributes "
                        "token_grid**2 tokens to the global transformer. "
                        "1 = one pooled vector per view.")
    p.add_argument(
        "--data_dir",
        default=None,
        help="Prepared render directory containing jitter_all.csv and run_* "
             "tile folders. Default: the local renderer's results/hope.",
    )
    p.add_argument(
        "--weights_dir",
        default=None,
        help="Where to save the checkpoint. Default: $VIT_HOPE_WEIGHTS_DIR, "
             "else vit_hope/weights. On the cluster point this at the object's "
             "data directory so nothing large lands on /home.",
    )
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--augment", action="store_true", help="Enable colour jitter.")
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--test_frac", type=float, default=0.10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overfit_test", action="store_true",
                   help="Overfit 8 samples for 300 steps as a wiring check.")

    # --- architecture overrides, generated from models.ALL_ARCH_KEYS ---------- #
    arch = p.add_argument_group(
        "architecture overrides",
        "Default None = the model's own default. models.ARCH_KEYS decides which "
        "of these each model accepts; an inapplicable one is a hard error.")
    for key in ALL_ARCH_KEYS:
        arch.add_argument(f"--{key}", type=float if key in _ARCH_FLOAT else int,
                          default=None)

    # --- sweep bookkeeping --------------------------------------------------- #
    p.add_argument("--run_tag", default="",
                   help="Suffix for the checkpoint filename, so the trials of a "
                        "hyper-parameter sweep do not overwrite each other. "
                        "Empty (default) keeps the historical name.")
    p.add_argument("--result_csv", default=None,
                   help="Append one row of hyper-parameters + best val loss here "
                        "when the run finishes. Header written on creation.")
    return p.parse_args()


def append_result(path: str, row: dict):
    """Append one run's record, writing the header only when creating the file.

    Rewrite-and-replace rather than an in-place append: os.replace is atomic on
    the same filesystem, so a job killed at the walltime can never leave a torn
    final line behind -- and the sweep driver uses this file as its resume
    record, so a corrupt row would make it re-run or skip the wrong trial.
    """
    existing = []
    if os.path.isfile(path):
        with open(path, newline="") as fh:
            existing = list(csv.DictReader(fh))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for old in existing:
            w.writerow(old)
        w.writerow({k: row.get(k, "") for k in RESULT_COLUMNS})
    os.replace(tmp, path)


def split_indices(n, val_frac, test_frac, seed):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val, n_test = int(n * val_frac), int(n * test_frac)
    return (perm[n_val + n_test:], perm[:n_val], perm[n_val:n_val + n_test])


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            pred = model(x)
            loss = loss_fn(pred, y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += loss.item() * x.size(0)
        count += x.size(0)
    return total / max(count, 1)


def overfit_test(model, dataset, loss_fn, device, lr):
    """Overfit a handful of samples as a wiring check.

    Uses the run's real learning rate rather than a fixed one: the multi-view
    models stack a local and a global transformer, and at 1e-3 that stack is
    unstable enough to stall around the target variance -- which looks like a
    wiring bug when it is only a step-size problem.
    """
    idx = list(range(min(8, len(dataset))))
    loader = DataLoader(Subset(dataset, idx), batch_size=len(idx), shuffle=True)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model.train()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    for step in range(300):
        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0 or step == 299:
            print(f"  step {step:3d}  loss {loss.item():.4f}")
    print("overfit_test done (loss should be near 0).")


def main():
    args = get_args()
    spec = MODEL_SPECS[args.model]
    cfg = Config(obj_id=args.obj_id,
                 tile_size=args.tile_size or spec["tile_size"],
                 patch_size=spec["patch_size"])
    data_dir = os.path.abspath(args.data_dir or cfg.output_dir)
    weights_dir = os.path.abspath(args.weights_dir or cfg.weights_dir)
    jitter_csv = os.path.join(data_dir, "jitter_all.csv")
    device = torch.device(args.device)
    pretrained = is_pretrained(args.model)
    lr = args.lr if args.lr is not None else \
        (3e-5 if (pretrained and not args.freeze) else 3e-4)

    # Validated before the dataset is touched, so a bad sweep flag fails in
    # seconds rather than after the tiles have been read.
    arch = check_arch(args.model, {k: getattr(args, k) for k in ALL_ARCH_KEYS})

    print(f"model={args.model} freeze={args.freeze} lr={lr} device={device} "
          f"tile={cfg.tile_size} augment={args.augment}")
    if arch:
        print("arch: " + " ".join(f"{k}={v}" for k, v in arch.items()))
    elif ARCH_KEYS[args.model]:
        print(f"arch: defaults (tunable: {', '.join(ARCH_KEYS[args.model])})")
    print(f"data={data_dir}")
    print(f"weights={weights_dir}")

    model = build_model(args.model, cfg, freeze_backbone=args.freeze,
                        dropout=args.dropout, token_grid=args.token_grid,
                        arch=arch).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params:,}")
    loss_fn = torch.nn.MSELoss()

    if args.overfit_test:
        ds = StackDataset(jitter_csv, data_dir, cfg, augment=False)
        overfit_test(model, ds, loss_fn, device, lr)
        return

    # Two views over the same rows: augmented for train, clean for val/test.
    train_full = StackDataset(jitter_csv, data_dir, cfg, augment=args.augment)
    eval_full = StackDataset(jitter_csv, data_dir, cfg, augment=False)
    tr, va, te = split_indices(len(eval_full), args.val_frac, args.test_frac, args.seed)
    print(f"samples: train={len(tr)} val={len(va)} test={len(te)}")

    train_loader = DataLoader(Subset(train_full, tr), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(Subset(eval_full, va), batch_size=args.batch_size,
                            num_workers=args.workers)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    os.makedirs(weights_dir, exist_ok=True)
    tag = "frozen" if args.freeze else "finetune"
    if not pretrained:
        tag = "scratch"
    # The run_tag suffix is appended only when non-empty, so every existing
    # checkpoint name (and everything in RESULTS.md) is unchanged.
    stem = f"{cfg.obj_stem}_{args.model}_{tag}"
    if args.run_tag:
        stem = f"{stem}_{args.run_tag}"
    ckpt_path = os.path.join(weights_dir, f"{stem}.pt")

    best_val, best_epoch, tr_loss = float("inf"), 0, float("nan")
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss = run_epoch(model, train_loader, loss_fn, device, optimizer)
        va_loss = run_epoch(model, val_loader, loss_fn, device)
        print(f"epoch {epoch:3d}/{args.epochs}  train {tr_loss:.4f}  val {va_loss:.4f}")
        if va_loss < best_val:
            best_val, best_epoch = va_loss, epoch
            torch.save({
                "state_dict": model.state_dict(),
                "model": args.model, "freeze": args.freeze,
                "tile_size": cfg.tile_size, "patch_size": cfg.patch_size,
                "token_grid": args.token_grid,
                # Without this a tuned checkpoint cannot be rebuilt: infer.py
                # would construct the default widths and fail load_state_dict.
                "arch": arch,
                "obj_id": cfg.obj_id,
                "test_indices": te,
            }, ckpt_path)
    print(f"best val {best_val:.4f} (epoch {best_epoch}) -> {ckpt_path}")

    if args.result_csv:
        append_result(args.result_csv, {
            "run_tag": args.run_tag or "default", "model": args.model, "tag": tag,
            "obj_id": cfg.obj_id, "n_train": len(tr), "n_val": len(va),
            "n_test": len(te), "epochs": args.epochs,
            "batch_size": args.batch_size, "lr": lr, "dropout": args.dropout,
            **{k: arch.get(k, "") for k in ALL_ARCH_KEYS},
            "token_grid": args.token_grid, "tile_size": cfg.tile_size,
            "seed": args.seed, "best_val": f"{best_val:.6f}",
            "best_epoch": best_epoch, "final_train": f"{tr_loss:.6f}",
            "n_params": n_params, "seconds": round(time.time() - started, 1),
            "ckpt": ckpt_path,
        })
        print(f"appended result -> {args.result_csv}")


if __name__ == "__main__":
    main()
