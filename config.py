"""
config.py

Single source of truth for paths and hyper-parameters used across the
vit_hope pipeline. Everything is object-agnostic: change `obj_id` (and, if
needed, the paths) and the whole pipeline follows.

The task: an object is rendered by projecting its observed RGB onto its 3D mesh
from a *jittered* camera pose, producing 6 orthographic tiles. A ViT regresses
the 6D pose offset (dt_x,dt_y,dt_z, dr_x,dr_y,dr_z) so the pose can be corrected
by subtraction.
"""

from dataclasses import dataclass, field
import os


# 6D regression target column order (also the offset that gets corrected).
JITTER_COLUMNS = ["dt_x", "dt_y", "dt_z", "dr_x", "dr_y", "dr_z"]

# Normalisation applied to the stacked tensor (per channel, matches vit_jitter).
NORM_MEAN = 0.5
NORM_STD = 0.5


_DEF_HOPE_ROOT = "/home/vubui/daad-rise-2026/cnos/datasets/bop23_challenge/datasets/hope"
_DEF_RENDERER_REPO = "/home/vubui/daad-rise-2026/texture-projection-opengl-cpp"
_DEF_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
_DEF_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


@dataclass
class Config:
    # --- HOPE dataset (BOP format) ---
    # Paths default to the local checkout but can be relocated (e.g. to a
    # cluster /data mount) via env vars without editing this file.
    hope_root: str = field(
        default_factory=lambda: os.environ.get("VIT_HOPE_HOPE_ROOT", _DEF_HOPE_ROOT))
    obj_id: int = 6
    split: str = field(
        default_factory=lambda: os.environ.get("VIT_HOPE_SPLIT", "val"))  # only val/test carry GT

    # --- C++ renderer (texture-projection-opengl-cpp) ---
    renderer_repo: str = field(
        default_factory=lambda: os.environ.get("VIT_HOPE_RENDERER_REPO", _DEF_RENDERER_REPO))
    renderer_binary: str = "build/src/main"          # relative to renderer_repo
    assets_subdir: str = "assets/hope"               # projection PNGs + obj live here
    output_subdir: str = "results/hope"              # prepared/jittered tiles + csv

    # --- this repo's own outputs ---
    # Everything vit_hope produces (predictions, clean/corrected renders,
    # comparisons, metrics) lands here, not in the renderer's tree. The renderer
    # accepts an absolute output dir, so it can write straight into it.
    results_dir: str = field(
        default_factory=lambda: os.environ.get("VIT_HOPE_RESULTS_DIR", _DEF_RESULTS_DIR))

    # --- cluster output routing ---
    # On the cluster /home holds code and /data holds data, so training output
    # belongs next to the tiles it was trained on, not in the repo. Both are
    # empty-by-default overrides: unset, the pipeline behaves exactly as it does
    # on a laptop (weights -> vit_hope/weights, predictions -> vit_hope/results).
    weights_dir: str = field(
        default_factory=lambda: os.environ.get("VIT_HOPE_WEIGHTS_DIR", _DEF_WEIGHTS_DIR))
    out_root: str = field(
        default_factory=lambda: os.environ.get("VIT_HOPE_OUT_ROOT", ""))

    # --- jitter generation ---
    dt_std: float = 0.01   # translation std (metres)
    dr_std: float = 0.05   # rotation std (radians, on the Rodrigues vector)
    jitters_per_image: int = 5
    seed: int = 42

    # --- data / model ---
    tile_size: int = 518   # must be a multiple of patch_size (DINOv2 patch = 14)
    patch_size: int = 14
    n_rows: int = 2        # tile grid produced by the renderer (2 x 3 = 6 tiles)
    n_cols: int = 3
    jitter_scale: float = 100.0

    def __post_init__(self):
        if self.tile_size % self.patch_size != 0:
            raise ValueError(
                f"tile_size ({self.tile_size}) must be a multiple of "
                f"patch_size ({self.patch_size}) for the ViT patch embedding."
            )

    # ---- derived names ----
    @property
    def obj_stem(self) -> str:
        return f"obj_{self.obj_id:06d}"

    @property
    def model_obj(self) -> str:
        return f"{self.obj_stem}.obj"

    @property
    def in_channels(self) -> int:
        return self.n_rows * self.n_cols * 3  # 18

    # ---- absolute paths ----
    @property
    def split_dir(self) -> str:
        return os.path.join(self.hope_root, self.split)

    @property
    def models_info_path(self) -> str:
        return os.path.join(self.hope_root, "models", "models_info.json")

    @property
    def model_ply_path(self) -> str:
        """Source HOPE mesh (mm, un-centred) shipped as ASCII PLY."""
        return os.path.join(self.hope_root, "models", f"{self.obj_stem}.ply")

    @property
    def camera_json_path(self) -> str:
        return os.path.join(self.hope_root, "camera.json")

    @property
    def assets_dir(self) -> str:
        """Absolute dir where projection PNGs + the .obj live."""
        return os.path.join(self.renderer_repo, self.assets_subdir)

    @property
    def output_dir(self) -> str:
        """Absolute dir where the renderer writes tiles and we write the CSV."""
        return os.path.join(self.renderer_repo, self.output_subdir)

    @property
    def binary_path(self) -> str:
        return os.path.join(self.renderer_repo, self.renderer_binary)

    @property
    def obj_out_dir(self) -> str:
        """Where this object's predictions go.

        With VIT_HOPE_OUT_ROOT set (the cluster case) that is a per-object
        subdirectory of the out root, matching the layout slurm/run_all_objects.sh
        builds by hand; otherwise the flat local results dir.
        """
        if self.out_root:
            return os.path.join(self.out_root, self.obj_stem)
        return self.results_dir

    @property
    def request_json_path(self) -> str:
        return os.path.join(self.output_dir, "request.json")

    @property
    def jitter_csv_path(self) -> str:
        return os.path.join(self.output_dir, "jitter_all.csv")

    def render_payload_header(self) -> dict:
        """Common {path, model} header for every renderer request."""
        return {"path": self.assets_subdir, "model": self.model_obj}
