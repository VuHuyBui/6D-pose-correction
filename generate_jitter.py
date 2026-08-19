"""
generate_jitter.py

Add Gaussian pose jitter to every clean pose in request.json and drive the C++
renderer to produce the 6 orthographic tiles per jittered pose. The 6D offset
(dt, dr) is the regression target.

Flow (mirrors vit_jitter/generate_jittered_data.py, cleaned):
  for run in 0..jitters_per_image-1:
      jittered_pose = clean_pose + N(0, std)     # keys kept identical
      write results/hope/run_<run>/request_jittered.json
      render -> results/hope/run_<run>/<key>_tile-w={0,1}_tile-h={0,1,2}.png
  concat all runs -> results/hope/jitter_all.csv

The renderer needs an OpenGL context; on a headless box we wrap it in
`xvfb-run` automatically (disable with --no_xvfb).

Usage:
    python generate_jitter.py --no_render          # write CSV/JSON only (no GPU)
    xvfb-run python generate_jitter.py --seed 42   # full render
"""

import argparse
import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd

from config import Config, JITTER_COLUMNS


def parse_pose(pose: dict):
    tvec = np.asarray(pose["tvec"], dtype=float).reshape(-1)
    rvec = np.asarray(pose["rvec"], dtype=float).reshape(-1)
    return tvec, rvec


def jitter_run(payload: dict, run_id: int, dt_std: float, dr_std: float,
               rng: np.random.Generator):
    """Return (jittered_payload, DataFrame) for one run; keys stay identical."""
    jittered_data, rows = {}, []
    for key, pose in payload["data"].items():
        tvec, rvec = parse_pose(pose)
        dt = rng.normal(0.0, dt_std, size=3)
        dr = rng.normal(0.0, dr_std, size=3)
        jt, jr = tvec + dt, rvec + dr

        jittered_data[key] = {
            "rvec": [[float(jr[0])], [float(jr[1])], [float(jr[2])]],
            "tvec": [float(jt[0]), float(jt[1]), float(jt[2])],
        }
        rows.append({
            "run_id": run_id,
            "output_folder": f"run_{run_id}",
            "key": key,
            **dict(zip(JITTER_COLUMNS, [*dt, *dr])),
            "real_tvec_x": tvec[0], "real_tvec_y": tvec[1], "real_tvec_z": tvec[2],
            "real_rvec_x": rvec[0], "real_rvec_y": rvec[1], "real_rvec_z": rvec[2],
            "tvec_x": jt[0], "tvec_y": jt[1], "tvec_z": jt[2],
            "rvec_x": jr[0], "rvec_y": jr[1], "rvec_z": jr[2],
        })

    out = dict(payload)
    out["data"] = jittered_data
    return out, pd.DataFrame(rows)


def render(cfg: Config, payload: dict, run_id: int, use_xvfb: bool):
    if not os.path.exists(cfg.binary_path):
        raise FileNotFoundError(
            f"Renderer binary not found: {cfg.binary_path}\n"
            f"Build it (see README) or pass --no_render."
        )
    out_rel = os.path.join(cfg.output_subdir, f"run_{run_id}")
    os.makedirs(os.path.join(cfg.renderer_repo, out_rel), exist_ok=True)

    cmd = [f"./{cfg.renderer_binary}", out_rel]
    if use_xvfb:
        cmd = ["xvfb-run", "-a", *cmd]

    print(f"\n=== render run_{run_id} ({len(payload['data'])} poses) ===")
    result = subprocess.run(cmd, input=json.dumps(payload), text=True,
                            capture_output=True, cwd=cfg.renderer_repo)
    if result.stdout:
        print(result.stdout[-2000:])
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise RuntimeError(f"Renderer failed (code {result.returncode}) on run_{run_id}.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--obj_id", type=int, default=Config.obj_id)
    p.add_argument("--jitters_per_image", type=int, default=Config.jitters_per_image)
    p.add_argument("--dt_std", type=float, default=Config.dt_std)
    p.add_argument("--dr_std", type=float, default=Config.dr_std)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--no_render", action="store_true", help="Write CSV/JSON only.")
    p.add_argument("--no_xvfb", action="store_true",
                   help="Do not wrap the renderer in xvfb-run.")
    args = p.parse_args()

    cfg = Config(obj_id=args.obj_id, jitters_per_image=args.jitters_per_image,
                 dt_std=args.dt_std, dr_std=args.dr_std, seed=args.seed)

    with open(cfg.request_json_path) as f:
        payload = json.load(f)
    if "data" not in payload or not payload["data"]:
        raise SystemExit("request.json has no poses. Run prepare_poses.py first.")

    # Auto-detect need for a virtual display.
    use_xvfb = (not args.no_xvfb) and (not args.no_render) \
        and (not os.environ.get("DISPLAY")) and shutil.which("xvfb-run")

    rng = np.random.default_rng(args.seed)
    print(f"{cfg.jitters_per_image} runs x {len(payload['data'])} poses "
          f"(dt_std={cfg.dt_std}, dr_std={cfg.dr_std})")

    all_rows = []
    for run_id in range(cfg.jitters_per_image):
        jittered, df = jitter_run(payload, run_id, cfg.dt_std, cfg.dr_std, rng)
        run_dir = os.path.join(cfg.output_dir, f"run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "request_jittered.json"), "w") as f:
            json.dump(jittered, f, indent=2)
        df.to_csv(os.path.join(run_dir, "jitter.csv"), index=False)

        if args.no_render:
            print(f"--no_render: skipped render for run_{run_id}")
        else:
            render(cfg, jittered, run_id, use_xvfb)
        all_rows.append(df)

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.to_csv(cfg.jitter_csv_path, index=False)
    print(f"\nWrote {len(df_all)} rows -> {cfg.jitter_csv_path}")


if __name__ == "__main__":
    main()
