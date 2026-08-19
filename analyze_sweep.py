"""
analyze_sweep.py

Rank the trials of a hyper-parameter sweep against the default configuration.

Reads one or more CSVs written by `train.py --result_csv` (see
`slurm/sweep_hparams.sh`) and, per (model, tag), ranks each trial by its **mean
best-val loss across the sweep objects**. Mean, not sum: the sweep objects have
different sample counts, and a trial that is missing one object would otherwise
look better than one that completed all of them.

Trials that did not finish every object present in the file are marked and
excluded from the ranking -- an incomplete trial's mean is not comparable, and a
walltime-killed sweep always has one.

    python analyze_sweep.py /data/$USER/vit_hope_rendered/sweeps/sweep_mvit_scratch.csv
    python analyze_sweep.py /data/$USER/vit_hope_rendered/sweeps/*.csv --top 10
    python analyze_sweep.py sweep_mvit_scratch.csv --per_object

The bottom of each table prints the winning trial's flags, ready to paste into a
run_all_objects.sh submission for the all-28-object confirmation run.
"""

import argparse
import csv
import statistics
from collections import defaultdict

from models import ALL_ARCH_KEYS

# The knobs a trial varies: architecture (per models.ALL_ARCH_KEYS) plus the two
# optimisation ones that are not architecture and so live outside that tuple.
TUNED_KEYS = list(ALL_ARCH_KEYS) + ["lr", "dropout"]


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+", help="Sweep result CSV(s).")
    p.add_argument("--top", type=int, default=15, help="Rows to print per group.")
    p.add_argument("--per_object", action="store_true",
                   help="Also print each trial's per-object best val loss.")
    return p.parse_args()


def load(paths):
    """Read the CSVs, keeping the LAST row for each (model, tag, trial, object).

    A `FORCE=1` re-run appends rather than replaces, and passing overlapping
    files on the command line duplicates rows too. Without deduplication a trial
    with a repeated object looks like it covered more objects than exist, so the
    completeness check below rejects it and it silently vanishes from the
    ranking. Last-wins because a re-run is the more recent measurement.
    """
    seen = {}
    for path in paths:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                if not r.get("best_val"):
                    continue
                r["best_val"] = float(r["best_val"])
                seen[(r["model"], r["tag"], r["run_tag"], r["obj_id"])] = r
    return list(seen.values())


def flag_string(row) -> str:
    """Reconstruct the train.py flags this trial was run with."""
    return " ".join(f"--{k} {row[k]}" for k in TUNED_KEYS if row.get(k))


def main():
    args = get_args()
    rows = load(args.csv)
    if not rows:
        raise SystemExit("no completed runs found in the given CSV(s)")

    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["tag"])].append(r)

    for (model, tag), grp in sorted(groups.items()):
        objects = sorted({r["obj_id"] for r in grp}, key=int)
        by_trial = defaultdict(list)
        for r in grp:
            by_trial[r["run_tag"]].append(r)

        stats = {}
        for run_tag, runs in by_trial.items():
            vals = [r["best_val"] for r in runs]
            stats[run_tag] = {
                "mean": statistics.fmean(vals),
                "worst": max(vals),
                "n": len(runs),
                "complete": len(runs) == len(objects),
                "row": runs[0],
                "by_obj": {r["obj_id"]: r["best_val"] for r in runs},
            }

        base = stats.get("default")
        ranked = sorted((s for s in stats.values() if s["complete"]),
                        key=lambda s: s["mean"])

        print(f"\n{'=' * 78}")
        print(f"{model} / {tag}   {len(stats)} trials, "
              f"{len(objects)} objects ({', '.join(objects)})")
        if base is None:
            print("!! no `default` trial in this file -- nothing to compare against")
        elif not base["complete"]:
            print(f"!! the `default` trial covers only {base['n']}/{len(objects)} "
                  f"objects; deltas below are against an incomplete baseline")
        print("=" * 78)

        header = f"{'rank':>4} {'trial':<8} {'mean_val':>10} {'vs_default':>11} {'worst':>9}  config"
        print(header)
        for i, s in enumerate(ranked[:args.top], 1):
            rt = s["row"]["run_tag"]
            delta = "" if base is None else f"{s['mean'] - base['mean']:+.4f}"
            mark = " *" if rt == "default" else "  "
            print(f"{i:>4} {rt + mark:<8} {s['mean']:>10.4f} {delta:>11} "
                  f"{s['worst']:>9.4f}  {flag_string(s['row']) or '(model defaults)'}")

        incomplete = [s for s in stats.values() if not s["complete"]]
        if incomplete:
            print(f"\n  excluded as incomplete: " + ", ".join(
                f"{s['row']['run_tag']} ({s['n']}/{len(objects)})"
                for s in sorted(incomplete, key=lambda s: s["row"]["run_tag"])))

        if args.per_object and ranked:
            print(f"\n  per-object best val")
            print("  " + f"{'trial':<8}" + "".join(f"{o:>10}" for o in objects))
            for s in ranked[:args.top]:
                cells = "".join(f"{s['by_obj'].get(o, float('nan')):>10.4f}"
                                for o in objects)
                print("  " + f"{s['row']['run_tag']:<8}" + cells)

        if ranked and base is not None:
            best = ranked[0]
            if best["row"]["run_tag"] == "default":
                print(f"\n  VERDICT: the default config still wins "
                      f"({base['mean']:.4f}) -- tuning did not help here.")
            else:
                gain = (base["mean"] - best["mean"]) / base["mean"] * 100
                print(f"\n  VERDICT: {best['row']['run_tag']} beats the default "
                      f"{best['mean']:.4f} vs {base['mean']:.4f} ({gain:.1f}% lower)")
                print(f"  confirm over all 28 objects with:")
                print(f"    python train.py --model {model} --obj_id <N> "
                      f"{flag_string(best['row'])}")
                print(f"  trainable params: {best['row'].get('n_params', '?')} "
                      f"(default {base['row'].get('n_params', '?')})")


if __name__ == "__main__":
    main()
