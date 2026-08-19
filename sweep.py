"""
sweep.py

Emit a reproducible random-search plan for `train.py`. This script samples
configurations and prints them; it does not train. The slurm driver
(`slurm/sweep_hparams.sh`) reads the lines and runs `train.py` per trial per
object, which keeps the sampling logic here -- testable and seeded -- and the job
orchestration in shell, matching how `run_all_objects.sh` already splits things.

Output is one tab-separated line per trial:

    <run_tag>\t<flags to pass verbatim to train.py>

Trial 0 is always the *current default* configuration (no architecture flags, no
--lr, so train.py's own defaults apply) tagged `default`. That is the row every
other trial has to beat, which is the whole point of the exercise.

Which knobs get sampled comes from models.ARCH_KEYS, so this works for every key
in MODEL_SPECS with no per-model special-casing:

    python sweep.py --model mvit --n_trials 24            # trunk + fuser + lr
    python sweep.py --model mvdinov2 --n_trials 12        # fuser + lr only
    python sweep.py --model dinov2 --n_trials 8           # lr/dropout only
    python sweep.py --model mvdinov2 --no_freeze          # narrower lr range

`--seed` is mixed with the model name, so two models never draw correlated
configs, and re-running with the same seed reproduces the plan exactly -- which
is what makes a walltime-killed sweep resumable.
"""

import argparse
import hashlib
import math
import random

from models import ARCH_KEYS, MODEL_KINDS, is_pretrained

# --- the search space ------------------------------------------------------- #
# Widths are powers-of-two-ish values with plenty of small divisors, so a valid
# head count always exists. Head counts are drawn from the divisors of the width
# that was picked, which is why they are not an independent axis.
WIDTHS = (128, 192, 256, 384, 512)
HEAD_CHOICES = (2, 4, 6, 8, 12, 16)
DEPTHS = (2, 4, 6, 8, 12)
MLP_RATIOS = (2.0, 3.0, 4.0, 6.0)
STAGE_DIMS = (128, 256, 384)
DROPOUTS = (0.0, 0.1, 0.2)

# A frozen backbone (or a from-scratch model) trains only small new modules and
# tolerates a wide range; a full fine-tune does not. train.py:138-139 already
# encodes that 10x gap in its defaults, so mirror it here rather than spending
# trials on runs that were always going to diverge.
LR_RANGE = (1e-5, 1e-3)
LR_RANGE_FINETUNE = (3e-6, 1e-4)


def heads_for(width: int) -> list:
    return [h for h in HEAD_CHOICES if width % h == 0]


def sample_arch(rng: random.Random, keys) -> dict:
    """Draw one value for each architecture knob in `keys`."""
    out = {}
    if "embed_dim" in keys:
        out["embed_dim"] = rng.choice(WIDTHS)
        out["num_heads"] = rng.choice(heads_for(out["embed_dim"]))
        out["depth"] = rng.choice(DEPTHS)
        out["mlp_ratio"] = rng.choice(MLP_RATIOS)
    if "fuser_embed_dim" in keys:
        out["fuser_embed_dim"] = rng.choice(WIDTHS)
        out["fuser_heads"] = rng.choice(heads_for(out["fuser_embed_dim"]))
        out["fuser_depth"] = rng.choice(DEPTHS)
        out["fuser_mlp_ratio"] = rng.choice(MLP_RATIOS)
    if "stage_dim" in keys:
        out["stage_dim"] = rng.choice(STAGE_DIMS)
    return out


def loguniform(rng: random.Random, lo: float, hi: float) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def space_size(keys) -> int:
    """How many distinct configurations the space holds, lr/dropout aside.

    Used only to warn when --n_trials asks for more trials than the model has
    architecture to vary -- a stacked pretrained backbone has none at all, so its
    search is lr x dropout and a large n_trials just repeats itself.
    """
    n = len(DROPOUTS)
    if "embed_dim" in keys:
        n *= sum(len(heads_for(w)) for w in WIDTHS) * len(DEPTHS) * len(MLP_RATIOS)
    if "fuser_embed_dim" in keys:
        n *= sum(len(heads_for(w)) for w in WIDTHS) * len(DEPTHS) * len(MLP_RATIOS)
    if "stage_dim" in keys:
        n *= len(STAGE_DIMS)
    return n


def fmt(value) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def trials(model: str, n_trials: int, seed: int, freeze: bool = True):
    """Yield (run_tag, {flag: value}) pairs; the first is the default config."""
    keys = ARCH_KEYS[model]
    # Mix the model name into the seed: with a bare integer seed, `vanilla` and
    # `mvit` would draw the *same* trunk widths, and a fuser-only model would
    # draw the trunk numbers as its fuser numbers. Same seed still reproduces.
    salt = int(hashlib.sha256(f"{model}:{seed}".encode()).hexdigest()[:8], 16)
    rng = random.Random(salt)
    lo, hi = LR_RANGE_FINETUNE if (is_pretrained(model) and not freeze) else LR_RANGE

    yield "default", {}
    for i in range(1, n_trials + 1):
        cfg = sample_arch(rng, keys)
        cfg["lr"] = float(f"{loguniform(rng, lo, hi):.2e}")
        cfg["dropout"] = rng.choice(DROPOUTS)
        yield f"t{i:02d}", cfg


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODEL_KINDS, required=True)
    p.add_argument("--n_trials", type=int, default=24,
                   help="Random draws, on top of the always-present default trial.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--freeze", dest="freeze", action="store_true", default=True)
    p.add_argument("--no_freeze", dest="freeze", action="store_false",
                   help="Sample the narrower fine-tuning learning-rate range.")
    p.add_argument("--pretty", action="store_true",
                   help="Human-readable table instead of the driver's TSV.")
    return p.parse_args()


def main():
    args = get_args()
    keys = ARCH_KEYS[args.model]
    n_trials = args.n_trials
    if not keys and n_trials > 12:
        print(f"# note: {args.model} has no architecture knobs (see ARCH_KEYS) -- "
              f"capping {n_trials} trials at 12 lr/dropout draws")
        n_trials = 12

    plan = list(trials(args.model, n_trials, args.seed, args.freeze))
    if args.pretty:
        print(f"# {args.model}: {len(plan)} trials, "
              f"{space_size(keys)} distinct arch configs available")
        for tag, cfg in plan:
            body = "  ".join(f"{k}={fmt(v)}" for k, v in cfg.items()) or "(model defaults)"
            print(f"{tag:8s} {body}")
        return

    for tag, cfg in plan:
        flags = " ".join(f"--{k} {fmt(v)}" for k, v in cfg.items())
        print(f"{tag}\t{flags}")


if __name__ == "__main__":
    main()
