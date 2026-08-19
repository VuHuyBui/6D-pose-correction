"""
prepare_poses.py

Turn real HOPE ground-truth poses for one object into renderer-ready input:

  1. Read every instance of `obj_id` from the HOPE `val` scenes' scene_gt.json.
  2. Convert each pose to the renderer's convention:
       - rotation  : cam_R_m2c (3x3) -> OpenCV Rodrigues rvec
       - translation: mm -> m, compensating for the (possibly) re-centred OBJ
  3. Save a per-instance projection image (the scene RGB masked to that
     instance) into the renderer's assets dir, named to match the pose key.
  4. Write a clean renderer request.json.

The pose maths (the important bit):
  The renderer's OBJ is centred and in metres, so a model point relates to the
  original HOPE model (mm) by  p_orig_mm = 1000 * p_obj + C_mm , where C_mm is
  the original AABB centre (from models_info.json). HOPE's GT maps the original
  model to the camera:  p_cam_mm = R @ p_orig_mm + t_mm . Substituting and
  converting to metres gives the pose to feed the renderer:
       rvec   = Rodrigues(R)
       tvec_m = (R @ C_mm + t_mm) / 1000
  (For obj_000006 the model is already centred so R @ C_mm is ~0, but this keeps
   the code correct for non-centred objects.)

Usage:
    python prepare_poses.py --obj_id 6
    python prepare_poses.py --obj_id 6 --limit 40    # cap #instances for a quick test
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

from config import Config


_RENDERER_AXIS_COMPENSATION = np.diag([1.0, -1.0, -1.0])


def model_center_mm(cfg: Config) -> np.ndarray:
    """AABB centre of the original HOPE model, in millimetres."""
    with open(cfg.models_info_path) as f:
        info = json.load(f)[str(cfg.obj_id)]
    return np.array([
        info["min_x"] + info["size_x"] / 2.0,
        info["min_y"] + info["size_y"] / 2.0,
        info["min_z"] + info["size_z"] / 2.0,
    ], dtype=np.float64)


def pose_to_renderer(R: np.ndarray, t_mm: np.ndarray, c_mm: np.ndarray):
    """(cam_R_m2c, cam_t_m2c mm) -> (rvec Rodrigues, tvec metres)."""
    # The renderer conjugates rotations with diag(1,-1,-1). Supplying R @ F
    # makes its resulting OpenGL rotation F @ R, which is the desired
    # OpenCV-to-OpenGL axis conversion.
    rvec = cv2.Rodrigues(R @ _RENDERER_AXIS_COMPENSATION)[0].reshape(3)
    tvec_m = (R @ c_mm + t_mm) / 1000.0
    return rvec, tvec_m


def _read_ply_ascii(path: str):
    """Minimal ASCII-PLY reader. Returns (verts NxP, faces list, name->col map)."""
    with open(path) as f:
        if f.readline().strip() != "ply":
            raise ValueError(f"{path}: not a PLY file")
        fmt = f.readline().strip()
        if "ascii" not in fmt:
            raise ValueError(f"{path}: only ASCII PLY is supported (got '{fmt}')")

        props, n_verts, n_faces, current = [], 0, 0, None
        for line in f:
            s = line.strip()
            if s.startswith("element vertex"):
                current, n_verts = "vertex", int(s.split()[-1])
            elif s.startswith("element face"):
                current, n_faces = "face", int(s.split()[-1])
            elif s.startswith("element"):
                current = "other"
            elif s.startswith("property") and current == "vertex":
                props.append(s.split()[-1])
            elif s == "end_header":
                break

        verts = np.array([[float(x) for x in f.readline().split()]
                          for _ in range(n_verts)], dtype=np.float64)
        faces = []
        for _ in range(n_faces):
            parts = f.readline().split()
            k = int(parts[0])
            faces.append([int(x) for x in parts[1:1 + k]])

    idx = {name: j for j, name in enumerate(props)}
    return verts, faces, idx


def write_model_obj(cfg: Config, c_mm: np.ndarray) -> str:
    """Convert the HOPE PLY (mm, un-centred) to the centred/metres OBJ the
    renderer loads, matching the pose maths: p_obj = (p_orig_mm - C_mm) / 1000.
    Written to assets/hope/<obj>.obj. Returns the output path."""
    verts, faces, idx = _read_ply_ascii(cfg.model_ply_path)
    xyz_m = (verts[:, [idx["x"], idx["y"], idx["z"]]] - c_mm) / 1000.0
    has_n = all(k in idx for k in ("nx", "ny", "nz"))
    has_uv = all(k in idx for k in ("texture_u", "texture_v"))

    out_path = os.path.join(cfg.assets_dir, cfg.model_obj)
    os.makedirs(cfg.assets_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"# {cfg.model_obj}: centred, metres; from {cfg.obj_stem}.ply\n")
        for v in xyz_m:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if has_uv:
            for vt in verts[:, [idx["texture_u"], idx["texture_v"]]]:
                f.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")
        if has_n:
            for vn in verts[:, [idx["nx"], idx["ny"], idx["nz"]]]:
                f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")
        for face in faces:
            refs = []
            for vi in face:              # OBJ is 1-based; attrs share the index
                j = vi + 1
                if has_uv and has_n:
                    refs.append(f"{j}/{j}/{j}")
                elif has_n:
                    refs.append(f"{j}//{j}")
                elif has_uv:
                    refs.append(f"{j}/{j}")
                else:
                    refs.append(f"{j}")
            f.write("f " + " ".join(refs) + "\n")
    return out_path


def write_renderer_intrinsics(cfg: Config) -> str:
    """Configure the renderer's existing intrinsics file for HOPE."""
    with open(cfg.camera_json_path) as f:
        camera = json.load(f)
    payload = {
        "width": camera["width"],
        "height": camera["height"],
        "mtx": [
            [camera["fx"], 0.0, camera["cx"]],
            [0.0, camera["fy"], camera["cy"]],
            [0.0, 0.0, 1.0],
        ],
        "distortion": [],
    }
    out_path = os.path.join(cfg.renderer_repo, "intrinsics.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def save_projection_image(cfg: Config, scene: str, img_idx: int,
                          inst_idx: int, key: str, mask: bool = False) -> bool:
    """Save the scene RGB as this instance's projection image.

    The renderer projects the *whole* image onto the mesh (the mesh geometry is
    the mask), so the raw scene RGB is all it needs -- this matches the cube
    setting. `mask=True` additionally zeros everything outside the instance's
    visible mask, which helps in cluttered HOPE scenes but is not required.
    Returns False if the required inputs are missing.
    """
    scene_dir = os.path.join(cfg.split_dir, scene)
    rgb_path = os.path.join(scene_dir, "rgb", f"{img_idx:06d}.png")
    if not os.path.exists(rgb_path):
        return False

    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)          # BGR, full frame
    if rgb is None:
        return False

    out = rgb
    if mask:
        mask_path = os.path.join(scene_dir, "mask_visib",
                                 f"{img_idx:06d}_{inst_idx:06d}.png")
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            return False
        out = cv2.bitwise_and(rgb, rgb, mask=(m > 0).astype(np.uint8) * 255)

    out_path = os.path.join(cfg.assets_dir, f"{key}.png")
    cv2.imwrite(out_path, out)
    return True


def build_request(cfg: Config, limit: int | None = None,
                  mask: bool = False) -> dict:
    c_mm = model_center_mm(cfg)
    os.makedirs(cfg.assets_dir, exist_ok=True)

    data = {}
    scenes = sorted(d for d in os.listdir(cfg.split_dir)
                    if os.path.isdir(os.path.join(cfg.split_dir, d)))

    for scene in scenes:
        gt_path = os.path.join(cfg.split_dir, scene, "scene_gt.json")
        if not os.path.exists(gt_path):
            continue
        with open(gt_path) as f:
            scene_gt = json.load(f)

        for img_key, instances in scene_gt.items():
            img_idx = int(img_key)
            for inst_idx, inst in enumerate(instances):
                if inst["obj_id"] != cfg.obj_id:
                    continue

                # nlohmann::json stores object keys lexicographically. Padding
                # keeps that order identical to numeric scene/image/instance
                # order, including duplicate objects such as IDs 2 and 16.
                key = f"{scene}_{img_idx:06d}_{inst_idx:06d}"
                if not save_projection_image(cfg, scene, img_idx, inst_idx, key,
                                             mask=mask):
                    print(f"  skip {key}: missing rgb/mask")
                    continue

                R = np.array(inst["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
                t_mm = np.array(inst["cam_t_m2c"], dtype=np.float64)
                rvec, tvec = pose_to_renderer(R, t_mm, c_mm)

                data[key] = {
                    "rvec": [[float(rvec[0])], [float(rvec[1])], [float(rvec[2])]],
                    "tvec": [float(tvec[0]), float(tvec[1]), float(tvec[2])],
                }
                if limit is not None and len(data) >= limit:
                    return data
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--obj_id", type=int, default=Config.obj_id)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of instances (handy for quick tests).")
    p.add_argument("--mask", action="store_true",
                   help="Zero everything outside the instance's visible mask "
                        "(helps in cluttered scenes). Default: use the raw RGB, "
                        "matching the cube setting.")
    args = p.parse_args()

    cfg = Config(obj_id=args.obj_id)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Produce the renderer's model: centred, metres OBJ from the HOPE PLY.
    c_mm = model_center_mm(cfg)
    obj_path = write_model_obj(cfg, c_mm)
    print(f"Wrote model -> {obj_path}")
    intrinsics_path = write_renderer_intrinsics(cfg)
    print(f"Wrote HOPE intrinsics -> {intrinsics_path}")

    mode = "masked" if args.mask else "raw RGB"
    print(f"Preparing poses for {cfg.obj_stem} from {cfg.split_dir} ({mode})")
    data = build_request(cfg, limit=args.limit, mask=args.mask)
    if not data:
        # Exit 2, distinct from the 1 that any other error produces, so a sweep
        # can tell "this object simply isn't in the split" from "prepare broke".
        print(f"No instances of {cfg.obj_stem} in {cfg.split_dir}.", file=sys.stderr)
        raise SystemExit(2)

    payload = cfg.render_payload_header()
    payload["data"] = data
    with open(cfg.request_json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(data)} poses -> {cfg.request_json_path}")
    print(f"Projection images -> {cfg.assets_dir}/<key>.png")


if __name__ == "__main__":
    main()
