# vit_hope — architecture reference

A file-by-file, class-by-class, function-by-function account of the pipeline.
`README.md` tells you how to *run* it; this tells you what every piece *is*.

---

## 0. What the system does

One HOPE object is rendered by projecting its observed RGB onto its 3D mesh from a
**jittered** camera pose. The renderer emits **6 orthographic tiles** (a 2×3 grid) per
jittered pose. A transformer reads those 6 tiles and regresses the **6D pose offset**
`(dt_x, dt_y, dt_z, dr_x, dr_y, dr_z)` that was injected, so the pose can be repaired by
subtraction:

```
corrected_pose = jittered_pose − predicted_offset
```

Quality is then measured not in offset units but *visually*: re-render at the corrected
pose and compare to the ground-truth render (MSE / MAE / masked SSIM).

### The end-to-end flow

```
HOPE BOP dataset
      │
      │  prepare_poses.py          (local, needs the mesh + scene GT)
      ▼
request.json  +  assets/hope/<key>.png  +  obj_XXXXXX.obj  +  intrinsics.json
      │
      │  generate_jitter.py        (local, needs OpenGL → xvfb-run)
      ▼
run_<r>/<key>_tile-w=R_tile-h=C.png   ×6      +  jitter_all.csv
      │
      │  data.py  StackDataset             → (18, T, T) tensor + 6D target
      ▼
      │  train.py                  (cluster GPU, no OpenGL needed)
      ▼
obj_XXXXXX_<model>_<tag>.pt        (state_dict + tile_size + test_indices + …)
      │
      │  infer.py                  (cluster GPU)
      ▼
predictions_<model>_<tag>.csv      (jittered pose + true offset + predicted offset)
      │
      │  visualize.py              (local, needs OpenGL again)
      ▼
clean/  corrected_<model>_<tag>/  comparisons_<model>_<tag>/  metrics_<model>_<tag>.csv
      │
      │  results_comparison.ipynb
      ▼
cross-model / cross-object tables and plots
```

The split matters: **rendering is local, learning is remote.** The GPU compute nodes have
no OpenGL/GLFW context, so `prepare_poses.py`, `generate_jitter.py` and `visualize.py`
cannot run there — only `train.py` and `infer.py` can.

### Module dependency graph

```
config.py            (no project imports — the root)
   ▲   ▲   ▲   ▲
   │   │   │   └──────────── prepare_poses.py, generate_jitter.py
   │   │   └──────────────── data.py
   │   └──────────────────── train.py, infer.py, visualize.py
   │
layers.py            (no project imports either)
   ▲
models_mv.py
   ▲
models.py            (the registry — the only thing train/infer/visualize import)
```

`layers.py` deliberately imports from neither `models.py` nor `models_mv.py`, and
`models_mv.py` deliberately does not import `models.py`. That keeps the chain acyclic and
lets `models.py` stay the single registry that everything else talks to.

---

## 1. `config.py` — the single source of truth for paths and geometry

No project imports. Everything downstream constructs a `Config` and reads paths off it,
so relocating the dataset (laptop → cluster `/data`) is an env-var change, never an edit.

### Module-level constants

| Name | Value | Role |
|---|---|---|
| `JITTER_COLUMNS` | `["dt_x","dt_y","dt_z","dr_x","dr_y","dr_z"]` | The 6D target's **column order**. Used by `generate_jitter` (to write), `data` (to read the target), `infer` (to name `true_*`/`pred_*` columns) and `visualize` (to subtract). One list, so the ordering can never drift between writer and reader. |
| `NORM_MEAN`, `NORM_STD` | `0.5`, `0.5` | Per-channel normalisation `(x − 0.5) / 0.5`, mapping `[0,1] → [−1,1]`. Matches the earlier `vit_jitter` pipeline. |
| `_DEF_HOPE_ROOT` etc. | paths | Laptop defaults, each overridable by a `VIT_HOPE_*` env var. |

### `class Config` (dataclass)

Fields group into five blocks:

- **HOPE dataset** — `hope_root`, `obj_id` (default `6`), `split` (default `val`; only
  `val`/`test` carry ground-truth poses).
- **C++ renderer** — `renderer_repo`, `renderer_binary` (`build/src/main`),
  `assets_subdir` (`assets/hope`, where projection PNGs and the `.obj` live),
  `output_subdir` (`results/hope`, the renderer's fixed staging directory).
- **This repo's outputs** — `results_dir`.
- **Cluster routing** — `weights_dir`, `out_root`. Both empty-by-default, so unset the
  pipeline behaves exactly as on a laptop; set, output lands beside the data on `/data`
  instead of on `/home`.
- **Jitter generation** — `dt_std=0.01` m, `dr_std=0.05` rad (on the Rodrigues vector),
  `jitters_per_image=5`, `seed=42`.
- **Data/model geometry** — `tile_size=518`, `patch_size=14`, `n_rows=2`, `n_cols=3`,
  `jitter_scale=100.0`.

`jitter_scale` is a quiet but essential detail: raw offsets are ~0.01 in magnitude, so MSE
on them would be ~1e-4 and the optimiser would be working in the noise. The dataset
multiplies the target by 100; `infer.py` divides the prediction back by 100 before writing
the CSV. Any change to it invalidates every checkpoint's effective learning rate.

#### `__post_init__()`
Raises if `tile_size % patch_size != 0`. This is the guard that makes each backbone's
`MODEL_SPECS` entry meaningful — a mismatched `--tile_size` fails at construction rather
than deep inside a patch-embed convolution.

#### Derived-name properties

| Property | Returns |
|---|---|
| `obj_stem` | `f"obj_{obj_id:06d}"` — the filename stem used absolutely everywhere. |
| `model_obj` | `f"{obj_stem}.obj"` — the mesh filename the renderer loads. |
| `in_channels` | `n_rows * n_cols * 3` = **18**. The reason every model's input is `(B, 18, T, T)`. |

#### Absolute-path properties

| Property | Points at |
|---|---|
| `split_dir` | `hope_root/val` — the BOP scenes. |
| `models_info_path` | `hope_root/models/models_info.json` — AABB extents, needed for the centring maths. |
| `model_ply_path` | The source HOPE mesh, ASCII PLY, in **millimetres, un-centred**. |
| `camera_json_path` | HOPE's shared intrinsics. |
| `assets_dir` | `renderer_repo/assets/hope` — projection PNGs + the `.obj`. |
| `output_dir` | `renderer_repo/results/hope` — the renderer's **fixed** staging dir. |
| `binary_path` | The compiled renderer executable. |
| `obj_out_dir` | `out_root/obj_stem` if `out_root` is set (cluster), else the flat `results_dir` (laptop). This one property is what makes the same code work in both layouts. |
| `request_json_path` | `output_dir/request.json` — the clean-pose render request. |
| `jitter_csv_path` | `output_dir/jitter_all.csv` — the training manifest. |

#### `render_payload_header() -> dict`
Returns `{"path": assets_subdir, "model": model_obj}`, the common header of every renderer
request. Callers attach a `"data"` dict of poses to it.

**Why `output_dir` being fixed matters:** the renderer *always* writes to
`renderer_repo/results/hope`, so rendering object 2 then object 6 would clobber object 2.
`generate_jitter_all.sh` exists precisely to render-then-move into a per-object layout.

---

## 2. `prepare_poses.py` — HOPE ground truth → renderer input

Converts real BOP poses into the renderer's convention and produces everything the
renderer needs. Runs once per object.

### The pose maths (the load-bearing part)

The renderer's OBJ is **centred and in metres**; HOPE's model is **un-centred and in
millimetres**. So a model point relates to the original by

```
p_orig_mm = 1000 · p_obj + C_mm          (C_mm = original AABB centre)
```

HOPE's GT maps original-model → camera: `p_cam_mm = R · p_orig_mm + t_mm`. Substituting:

```
rvec   = Rodrigues(R)
tvec_m = (R · C_mm + t_mm) / 1000
```

For `obj_000006` the model is already centred so `R·C_mm ≈ 0`, but the general form keeps
non-centred objects correct.

### Functions

#### `model_center_mm(cfg) -> np.ndarray`
Reads `models_info.json` and returns the AABB centre `[min_x + size_x/2, …]` in mm — the
`C_mm` above.

#### `pose_to_renderer(R, t_mm, c_mm) -> (rvec, tvec_m)`
Applies the maths above, plus one subtlety: the rotation passed to `cv2.Rodrigues` is
`R @ diag(1, −1, −1)` (`_RENDERER_AXIS_COMPENSATION`). The renderer internally conjugates
rotations with that same matrix, so feeding it `R·F` makes its effective OpenGL rotation
`F·R` — which is the OpenCV→OpenGL axis flip you actually want. Get this wrong and every
render is silently mirrored.

#### `_read_ply_ascii(path) -> (verts, faces, idx)`
A minimal ASCII-PLY reader — no trimesh/open3d dependency for one file format. Parses the
header to learn the vertex property order, then reads `n_verts` rows and `n_faces`
variable-length face rows. Returns the raw vertex array, the face index lists, and a
`property-name → column` map so callers can ask for `x`, `nx`, `texture_u` by name.
Raises on non-`ply` and on binary PLY.

#### `write_model_obj(cfg, c_mm) -> str`
Converts the PLY to the OBJ the renderer loads, applying exactly the inverse of the pose
maths: `p_obj = (p_orig_mm − C_mm) / 1000`. Emits `v`, and conditionally `vt`/`vn` if the
PLY carried UVs/normals, then faces with the right `j`, `j/j`, `j//j` or `j/j/j` reference
form for whichever attributes exist. OBJ indices are 1-based and all attributes share the
vertex index here, hence the single `j = vi + 1`.

#### `write_renderer_intrinsics(cfg) -> str`
Reshapes HOPE's `camera.json` (`fx, fy, cx, cy, width, height`) into the renderer's
`intrinsics.json` (a 3×3 `mtx` plus an empty `distortion`), written into the renderer repo
root.

#### `save_projection_image(cfg, scene, img_idx, inst_idx, key, mask=False) -> bool`
Saves the scene RGB as this instance's projection texture, named `<key>.png` in the assets
dir. Default is the **raw full frame** — the mesh geometry is itself the mask when the
renderer projects, matching the original cube setting. `mask=True` additionally zeroes
everything outside the instance's `mask_visib`, which helps in cluttered HOPE scenes.
Returns `False` (rather than raising) when the RGB or mask is missing, so the caller can
skip and continue.

#### `build_request(cfg, limit=None, mask=False) -> dict`
Walks every scene in the split, every image, every instance, keeps the ones whose
`obj_id` matches, saves that instance's projection image, converts its pose, and
accumulates `{key: {"rvec": [[x],[y],[z]], "tvec": [x,y,z]}}`.

The key format `f"{scene}_{img_idx:06d}_{inst_idx:06d}"` is zero-padded on purpose:
`nlohmann::json` (the renderer's JSON library) stores object keys **lexicographically**, and
padding makes lexicographic order equal numeric order. Without it, image 10 would sort
before image 2 and the tiles would be mis-associated — including for objects that appear
twice in a scene.

#### `main()`
Builds the OBJ and the intrinsics, then the request. **Exits with code 2** (distinct from
the generic 1) when the object simply isn't in the split, so `generate_jitter_all.sh` can
tell "not present" from "prepare broke".

---

## 3. `generate_jitter.py` — clean poses → jittered renders + the training manifest

### `parse_pose(pose) -> (tvec, rvec)`
Flattens the renderer's nested pose representation (`rvec` is `[[x],[y],[z]]`) into two
1-D arrays.

### `jitter_run(payload, run_id, dt_std, dr_std, rng) -> (payload, DataFrame)`
One jitter run over all poses. For each key: draws `dt ~ N(0, dt_std²)` and
`dr ~ N(0, dr_std²)`, adds them to the clean pose, and emits **both** the jittered payload
(keys unchanged, so tiles line up with the clean render) and a DataFrame row recording:

- `run_id`, `output_folder` (`run_<r>`), `key` — the sample's identity;
- the 6 `dt_*`/`dr_*` columns — **the regression target**;
- `real_tvec_*`/`real_rvec_*` — the clean pose;
- `tvec_*`/`rvec_*` — the jittered pose, which `visualize.py` later corrects.

Rotation jitter is additive on the Rodrigues vector, not a proper `SO(3)` composition. At
`dr_std = 0.05` rad the difference is negligible, but it's a modelling choice worth knowing.

### `render(cfg, payload, run_id, use_xvfb)`
Pipes the payload as JSON on **stdin** to `./build/src/main <out_rel>`, with
`cwd = renderer_repo` (asset paths in the payload are relative to it). Wraps in
`xvfb-run -a` when headless. Raises with the tail of stderr on non-zero exit.

### `main()`
Loads `request.json`, auto-detects the need for `xvfb`, then for each of
`jitters_per_image` runs writes `run_<r>/request_jittered.json` and `run_<r>/jitter.csv`,
renders (unless `--no_render`), and finally concatenates every run into
`jitter_all.csv` — the one file `StackDataset` reads.

A single `np.random.default_rng(seed)` is threaded through all runs, so run 1's jitter is
not a repeat of run 0's, and the whole set is reproducible from `--seed`.

---

## 4. `data.py` — tiles on disk → `(18, T, T)` tensor + 6D target

### Module constants
`_BCS = 0.10` (brightness/contrast/saturation range), `_HUE = 0.08` — colour-jitter
amplitudes matching the earlier stacked pipeline.

### `class StackDataset(Dataset)`

#### `__init__(csv_path, data_dir, cfg, augment=False)`
Reads `jitter_all.csv` and keeps **only the rows whose 6 tiles all exist on disk**, so a
partial or interrupted render is usable rather than fatal; it prints how many rows it
dropped. Raises if nothing survives. Then computes the crop box **once** from the first
surviving row and reuses it for every sample.

#### `_tile_paths(row) -> list[str]`
`<data_dir>/<output_folder>/<key>_tile-w={r}_tile-h={c}.png` for `r` in `range(n_rows)`,
`c` in `range(n_cols)` — a nested loop with **`r` outer**. This iteration order is the
canonical view order and is what makes the multi-view reshape in `layers.split_views`
valid.

#### `_all_tiles_exist(row) -> bool`
The filter used by `__init__`.

#### `_compute_shared_crop_box(row, thresh=8, margin=16) -> (l, t, r, b)`
Unions the non-black bounding boxes across the 6 tiles of one sample (grayscale > 8),
pads by 16 px, and clamps to the image. Falls back to the whole image if every tile is
black.

Two decisions are embedded here. **Shared across views**: one box for all 6 tiles, so the
relative scale and placement between views is preserved — a per-view crop would destroy
exactly the cross-view geometry the multi-view models are built to exploit. **Computed
once, from row 0**: the object is at (nearly) the same place in every sample, so a
per-sample box would let the crop absorb part of the translation offset the model is
supposed to predict — i.e. it would leak the label into the preprocessing.

#### `_load_stack(row) -> torch.Tensor`
Opens each tile, crops to the shared box, resizes to `(T, T)`, converts to a `(3, T, T)`
float tensor in `[0,1]`, and `torch.cat(tiles, dim=0)` → `(18, T, T)`.

**This one line defines the input contract.** The concatenation is in `_tile_paths` order,
so channels `[0:3]` are view 0, `[3:6]` view 1, …, `[15:18]` view 5 — the layout is
**view-major**. That is what makes `x.reshape(B*6, 3, T, T)` an exact, zero-copy
decomposition later, and why the multi-view models needed no change to this file at all.

#### `_joint_color_jitter(stack) -> torch.Tensor`
Samples **one** `(brightness, contrast, saturation, hue)` quadruple per *sample* and
applies it identically to all 6 view slices, then clamps to `[0,1]`. Per-tile jitter would
teach the model that views can disagree about colour — they can't; they're the same object
under one renderer, and cross-view appearance consistency is signal.

#### `__len__` / `__getitem__`
`__getitem__` loads the stack, optionally augments, normalises with `NORM_MEAN`/`NORM_STD`,
and returns `(stack, target)` where the target is the 6 `JITTER_COLUMNS` values times
`cfg.jitter_scale`.

Note augmentation happens *before* normalisation (the `TF.adjust_*` ops expect `[0,1]`).

---

## 5. `layers.py` — primitives shared by both model families

Imports only `torch`. This file exists so `models.py` and `models_mv.py` can share the ViT
trunk, the head, and the freeze semantics without either importing the other.

### `mlp_head(dim, out_dim=6) -> nn.Sequential`
`Linear(dim, dim) → GELU → Linear(dim, out_dim)`. The regression head every model in the
repo ends with — identical across all nine variants, so head capacity is never a
confounder when comparing backbones.

### `encoder_stack(embed_dim, depth, num_heads, mlp_ratio=4.0, dropout=0.1) -> nn.TransformerEncoder`
A `TransformerEncoder` of `depth` layers with `activation="gelu"`, `batch_first=True`, and
**`norm_first=True`** (pre-LN). Pre-LN is what makes a from-scratch transformer trainable
without a warmup schedule — post-LN at this depth needs one.

### `class FreezableBackbone`
A **mixin, not an `nn.Module`**. Mixed in *before* `nn.Module` in the MRO so its `train()`
takes precedence.

- **`set_backbone_trainable(trainable)`** — sets `self.freeze_backbone`, flips
  `requires_grad` on every backbone parameter, and puts the backbone into the matching
  mode.
- **`train(mode=True)`** — calls `super().train(mode)`, then forces `self.backbone.eval()`
  if frozen, and returns `self`.

The `train()` override is the non-obvious half. `model.train()` recurses into *all*
children, so without it a "frozen" backbone would still run dropout and stochastic depth —
its features would jitter from step to step while the head tries to fit them. Frozen must
mean **deterministic**, not just "no gradients".

### `class ViTTrunk(nn.Module)`
A compact from-scratch ViT trunk: `Conv2d(in_ch, embed_dim, patch, stride=patch)` patchify
→ prepend CLS → add a learned position embedding → dropout → encoder → final LayerNorm.

Returns **all tokens**, `(B, 1 + n_patches, embed_dim)`, CLS first — deliberately not a
pooled vector, because the two consumers want different things: `VanillaViT` takes
`[:, 0]`, while `MultiViewViT` takes the patch tokens and pools them spatially per view.
Asserts `img_size % patch_size == 0`. Position and CLS embeddings init at `std=0.02`
(ViT standard).

### `split_views(x, n_views=6) -> Tensor`
`(B, 3V, T, T) → (B·V, 3, T, T)`. A pure `reshape` — no permute, no copy — because
`_load_stack` already laid the channels out view-major. View `v` of sample `b` lands at row
`b·V + v`, which is the inverse the callers rely on when they reshape back to
`(B, V, …)`. Validates the channel count so a wrong `n_views` fails loudly instead of
silently scrambling views.

### `tokens_to_grid(feat, num_prefix=0) -> Tensor`
Normalises whatever a backbone hands back into NCHW `(B, D, h, w)`, absorbing the two
layouts timm uses:

- **rank 4** — SwinV2 returns NHWC `(B, h, w, D)` → `permute(0,3,1,2)`.
- **rank 3** — ViTs return `(B, num_prefix + N, D)` → drop the `num_prefix` CLS/register
  tokens, then reshape the remaining `N` into a `√N × √N` grid.

Raises if `N` isn't a perfect square or the rank is neither 3 nor 4. This function is the
adapter that lets one `MultiViewTimm` class serve both ViT and Swin backbones.

### `pool_grid(grid, token_grid) -> Tensor`
`(B, D, h, w) → (B, token_grid², D)` via `adaptive_avg_pool2d` then flatten-transpose.
This is the **token-budget valve**. Raw concatenation of DINOv2 patch tokens at 518 px
would be 6 × 1369 = 8214 tokens into the global transformer — quadratically infeasible.
Pooling to a 2×2 grid gives 4 tokens per view, 24 + CLS = **25** total. `token_grid=1`
degenerates to a single pooled vector per view.

### `class GlobalFuser(nn.Module)`
The cross-view transformer. **Every one of the five multi-view variants uses this exact
module with the same hyper-parameters** (`embed_dim=512`, `depth=4`, `heads=8`), which is
what makes the comparison between them a comparison of *feature extractors* rather than of
fusers.

Forward, given `(B, V, N, in_dim)`:

1. `proj` — `Linear(in_dim, 512)` then **`LayerNorm`**.
2. Add `view_embed` `(1, V, 1, E)` and `tok_embed` `(1, 1, N, E)`, broadcast.
3. Flatten views into one sequence `(B, V·N, E)`.
4. Prepend CLS → `(B, 1 + V·N, E)`.
5. Dropout → encoder → LayerNorm.
6. `mlp_head` on the CLS → `(B, 6)`.

Validates `(V, N)` against what it was built for.

Two initialisation decisions here were *measured*, not guessed, and both defend the same
property:

**The `LayerNorm` after `proj`.** Raw feature magnitude varies ~4× across backbones
(SwinV2's final stage is much smaller than DINOv2's, or than an early Swin stage). Measured
projected-token std before the norm: mvit `0.249`, mvswinv2 `0.060`, mvswinlo `0.264`. With
a fixed-std view embedding, that same embedding would be 8% of the token for one backbone
and 33% for another — the fuser would not actually be the same across variants. After the
LayerNorm it is exactly 2.0% for all of them.

**`view_embed` / `tok_embed` init at `std=0.25`, not the ViT-standard `0.02`.** The 0.02
convention is tuned for ~1000 positions grown into over a long pretraining run. Here there
are 6 views and a short fine-tune, and at 0.02 the fused output was measured <1% sensitive
to *which view* a token came from — i.e. the architecture had degenerated into a
permutation-invariant bag of views, which is precisely what it exists to avoid. Sweeping
0.02 / 0.10 / 0.25 showed 0.25 gives 12–20% view sensitivity with no convergence cost. The
CLS stays at 0.02.

**Permutation sensitivity is the correctness property of this whole architecture.** If
shuffling the views leaves the output unchanged, the view embedding is inert and the model
is not doing multi-view reasoning. Measured at init on real data: 1.5–8.8%, growing 5–17×
under training.

---

## 6. `models_mv.py` — the multi-view family

Where `models.py`'s backbones see the 6 projections as one 18-channel image (the patch-embed
conv **sums** all six into a single token grid, so there is no per-view representation),
these keep the views separate:

```
(B, 18, T, T) → 6 × (3, T, T) → shared encoder → per-view tokens
              → global transformer across views → (B, 6)
```

The input contract is unchanged, so `data.py` and the whole train/infer/visualize pipeline
need no per-model special-casing.

### Constants
`N_VIEWS = 6` (= `cfg.n_rows * cfg.n_cols`), `TOKEN_GRID = 2` (tokens per view =
`TOKEN_GRID²`).

### `class MultiViewViT(nn.Module)` — registry key `mvit`
From scratch. A shared `ViTTrunk` over 3 channels, then `GlobalFuser`.

Forward: `split_views` → `local` trunk on the `(B·6)` batch → `tokens_to_grid(…, num_prefix=1)`
(drops the local CLS; the fuser has its own) → `pool_grid` → reshape to `(B, 6, t, E)` →
`fuser`.

**Weights are shared across views** — one trunk applied to a `(B·6)` batch, not six trunks.
That is 6× cheaper *and* the right inductive bias: the views are the same object under the
same renderer, so what differs is *which* view a feature came from, and that is carried by
the fuser's view embedding rather than by separate parameters.

`mvit` uses a 224/16 tile where `vanilla` uses 518/14, deliberately: it runs its encoder 6×
per sample, and a from-scratch model has no pretrained prior that would justify 518.

### `class MultiViewTimm(FreezableBackbone, nn.Module)`
The pretrained base class — the per-view analogue of `models.TimmRegressor`. Built with
**`in_chans=3`**, i.e. its native pretrained patch embed, not an 18-channel one re-inflated
from it, so the pretrained weights are used exactly as trained.

`__init__` validates the channel count, builds the backbone, calls `_post_backbone()`,
constructs the fuser from `_feature_dim()` / `_tokens_per_view()`, and applies the freeze
setting.

Five **shape hooks** exist so that `MVSwinV2Low` can differ without duplicating the class:

| Hook | Default | Purpose |
|---|---|---|
| `_build_backbone(img_size)` | `timm.create_model(BACKBONE, pretrained=True, in_chans=3, img_size=(T,T), num_classes=0)` | How the backbone is constructed. |
| `_post_backbone()` | no-op | Register submodules whose shape depends on the backbone. Exists as its own hook because creating them inside `_feature_dim()` would be a side effect in a getter. |
| `_feature_dim()` | `backbone.num_features` | Input width of the fuser. |
| `_tokens_per_view()` | `token_grid²` | Sequence length contributed per view. |
| `_view_tokens(flat)` | `forward_features` → `tokens_to_grid` → `pool_grid` | `(B·V, 3, T, T) → (B·V, tokens, dim)`. |

`forward` is then shared: `split_views` → `_view_tokens` → reshape to `(B, V, N, D)` →
`fuser`. Reading `num_prefix_tokens` off the backbone with `getattr(..., 0)` is what lets
the same path handle DINOv2 (1 CLS + registers) and Swin (no prefix).

### `MVDinoV2` / `MVDinoV3` / `MVSwinV2`
Each sets only `BACKBONE`, read from the same `VIT_HOPE_{DINOV2,DINOV3,SWINV2}_BACKBONE`
env vars as `models.py` so the stacked and multi-view variants can never end up comparing
different checkpoints of the same backbone. DINOv3 is gated on the HF hub — `HF_TOKEN`
must be exported before the first download.

### `class MVSwinV2Low(MultiViewTimm)` — registry key `mvswinlo`
SwinV2 tapped at **stages 0–1** instead of the final stage.

The motivation: SwinV2's final stage is 32×-reduced (8×8 at 256 px) and semantic, but a
pose offset of a few millimetres is a *sub-object* cue. Stages 0 and 1 are 4×/8×-reduced
(64×64, 32×32) and still carry it.

Overrides:
- `_build_backbone` — `features_only=True, out_indices=(0, 1)`, which returns a *list* of
  NHWC maps and exposes **no `num_features`**.
- `_post_backbone` — builds `stage_proj`, one `Linear(c, 256)` per stage, with the channel
  counts read from `backbone.feature_info.channels()` (the replacement for the missing
  `num_features`).
- `_feature_dim` → `STAGE_DIM = 256`; `_tokens_per_view` → `2 · token_grid²`.
- `_view_tokens` — pools each stage to the same coarse grid, projects each to the common
  width, and **concatenates along the token axis**, so each view contributes 8 tokens at
  the default and the fuser can weigh the two scales itself rather than having a fixed
  blend imposed.

Behaviour note from smoke testing: `mvswinlo` has a slow first ~80 steps (0.29 → 0.11 at
step 50) and then converges normally (0.004 by step 100). A 60-step run looks like a
failure; it isn't.

---

## 7. `models.py` — the registry and the stacked family

The only model module the rest of the pipeline imports.

### `MODEL_SPECS`
Per-key input geometry and provenance:

| key | patch | tile | pretrained | architecture |
|---|---|---|---|---|
| `vanilla` | 14 | 518 | ✗ | from-scratch ViT, 18-channel stack |
| `dinov2` | 14 | 518 | ✓ | DINOv2 ViT-B/14 |
| `dinov3` | 16 | 512 | ✓ | DINOv3 ViT-B/16 |
| `swinv2` | 64 | 256 | ✓ | SwinV2-B window16 |
| `mvit` | 16 | 224 | ✗ | shared local ViT + fuser |
| `mvdinov2` | 14 | 518 | ✓ | shared DINOv2 per view + fuser |
| `mvdinov3` | 16 | 512 | ✓ | shared DINOv3 per view + fuser |
| `mvswinv2` | 64 | 256 | ✓ | shared SwinV2 per view + fuser |
| `mvswinlo` | 64 | 256 | ✓ | SwinV2 stages 0–1 per view + fuser |

Two things to know about this table:

- **`swinv2`'s `patch_size: 64`** is not a real patch size (Swin's is 4). It is the
  constraint `tile % (patch 4 × window 16) == 0` expressed in the one field
  `Config.__post_init__` checks.
- **The `pretrained` field is the single source of truth** for "does `--freeze` mean
  anything, and is the checkpoint tag `scratch` or `frozen`/`finetune`?". It replaced a
  leaky `kind == "vanilla"` string test that appeared in `train.py` (twice), `infer.py`, and
  `run_all_objects.sh` — four independent places that had to agree. Now Python and shell
  both read this dict.

Registry keys are **lowercase-alphanumeric with no underscore** because
`visualize.detect_variant` and the notebook parse `predictions_<model>_<tag>.csv` with a
regex; an underscore in the model name would make the split ambiguous. That is why it's
`mvswinlo`, not `mv_swin_low`.

`MODEL_KINDS = list(MODEL_SPECS)` feeds `argparse` `choices=` in `train.py` and
`visualize.py`, so a new key becomes a valid CLI argument automatically.

### `class VanillaViT(nn.Module)`
`ViTTrunk` over all 18 channels + `mlp_head`; forward takes the CLS token. The from-scratch
baseline. Since the patch conv spans all 18 channels at once, the 6 views are summed into a
single token grid at the very first layer — this is the behaviour the multi-view family
exists to contrast against.

### `class TimmRegressor(FreezableBackbone, nn.Module)`
Pretrained backbone + dropout + `mlp_head`. Built with `in_chans=in_channels` (18) —
timm adapts the pretrained patch-embed conv to the extra channels and interpolates the
position embeddings to `img_size`. `forward` uses `self.backbone(x)` (pooled, `num_classes=0`)
rather than `forward_features`, so it gets a single `(B, embed_dim)` vector. The head is
always trainable regardless of the freeze setting.

`DinoV2ViT`, `DinoV3ViT`, `SwinV2` each set only `BACKBONE`. ViT-B rather than ViT-L
throughout: 18 channels at 518² makes ViT-L OOM-prone on one GPU, and the env vars let you
trade back up.

### Registry helpers

- **`_PRETRAINED`** — `{key → class}` for all seven pretrained variants (three stacked,
  four multi-view).
- **`_MULTI_VIEW`** — the five keys whose constructor accepts `token_grid`.
- **`is_pretrained(kind) -> bool`** — reads `MODEL_SPECS[kind]["pretrained"]`, converting
  a `KeyError` into a `ValueError` that names the valid keys. Called by `train.py` (learning
  rate + tag) and `infer.py` (tag).
- **`build_model(kind, cfg, freeze_backbone=True, dropout=0.1, token_grid=TOKEN_GRID)`** —
  the one constructor. Dispatches `vanilla` → `VanillaViT`, `mvit` → `MultiViewViT`,
  anything in `_PRETRAINED` → that class with `freeze_backbone`. `token_grid` is passed only
  to keys in `_MULTI_VIEW`, so the stacked models' signatures stay clean.
- **`migrate_state_dict(kind, state_dict) -> dict`** — a compatibility shim. `VanillaViT`'s
  `patch`/`cls`/`pos`/`encoder`/`norm` used to sit at the top level and now live inside
  `trunk`. Old `vanilla_scratch` checkpoints — the ones behind the current `RESULTS.md`
  numbers — would otherwise fail to load. It re-prefixes those five key families with
  `trunk.`, and is a no-op for any other kind or for a dict that already has `trunk.` keys.
  Applied in `infer.load_model`; verified to reproduce bit-identical output.

---

## 8. `train.py` — the training entry point

### `get_args()`
Notable flags:

- `--model` — `choices=MODEL_KINDS`, so new registry keys work with no edit here.
- `--freeze` / `--no_freeze` — a paired `store_true`/`store_false` on the same `dest`,
  default `True`.
- `--lr` — default `None`, resolved later to `3e-5` for an unfrozen pretrained backbone,
  `3e-4` otherwise.
- `--tile_size` — default `None` → the model's native size from `MODEL_SPECS`.
- `--token_grid` — multi-view only; `1` degenerates to one pooled vector per view.
- `--overfit_test` — the wiring check.
- `--data_dir` / `--weights_dir` — the cluster escape hatches that keep large files off `/home`.

### `split_indices(n, val_frac, test_frac, seed) -> (train, val, test)`
A seeded `randperm`, sliced val / test / train. Deterministic from `--seed`, which is what
lets `infer.py` reconstruct the exact same held-out set later — though in practice it
doesn't have to, because the indices are stored in the checkpoint.

### `run_epoch(model, loader, loss_fn, device, optimizer=None) -> float`
One pass. `optimizer is None` is the eval/train switch: it drives `model.train(train)` and
`torch.set_grad_enabled(train)`. Returns the sample-weighted mean loss, so a short final
batch doesn't distort the average.

### `overfit_test(model, dataset, loss_fn, device, lr)`
Takes 8 samples, one batch, 300 steps, printing every 50. A wiring check, not training:
if the loss doesn't go near zero, something in the data → model → loss → gradient path is
broken.

Two details:

- `filter(lambda p: p.requires_grad, ...)` — with a frozen backbone this must still be
  non-empty (the fuser + head). If it were empty, Adam would raise, and that error *is*
  part of the check.
- **It takes the run's real `lr`.** It used to hardcode `1e-3` while ignoring `--lr`
  entirely. The multi-view models stack a local *and* a global transformer, and at `1e-3`
  that stack is unstable enough to stall near the target variance: measured across seeds at
  `1e-3`, `mvit` gave 0.0073 / 0.0373 / 0.2661 / 0.1068 — which reads as a wiring bug when
  it is only a step-size problem. At `5e-4` and below it reliably reaches ~0.001–0.016.

### `main()`
Resolves the spec → `Config` → paths → device → `pretrained` → `lr`; builds the model;
short-circuits to `overfit_test` if asked (with `augment=False` — an overfit check should
see a fixed target).

Otherwise it constructs **two `StackDataset` objects over the same rows**: `train_full`
with augmentation and `eval_full` without. `Subset` then indexes into whichever is
appropriate. This is how val/test get clean, deterministic inputs while train gets jitter,
without a per-batch flag.

The checkpoint is saved on every val improvement and carries everything needed to rebuild
the model at inference time:

```python
{"state_dict", "model", "freeze", "tile_size", "patch_size",
 "token_grid", "obj_id", "test_indices"}
```

`test_indices` travelling with the weights is what guarantees `infer.py` evaluates on data
the model never saw, even if the split logic or seed later changes.

---

## 9. `infer.py` — checkpoint → predictions CSV

No rendering, so it is safe on a GPU compute node.

### `get_args()`
`--weights` is required; `--data_dir`, `--out_dir`, `--out_name` all default off the
checkpoint and the `Config`.

### `load_model(ckpt_path, device) -> (model, cfg, ck)`
Rebuilds `Config` from the checkpoint's `obj_id` / `tile_size` / `patch_size`, rebuilds the
model with the checkpoint's `freeze` and `token_grid` (defaulting to `TOKEN_GRID` for
pre-`token_grid` checkpoints), runs the state dict through `migrate_state_dict`, and puts
it in `eval()`.

The architecture is reconstructed **entirely from the checkpoint**, never from CLI flags —
which is why every geometry knob has to be written into it at save time. A `--token_grid`
that wasn't saved would silently build a differently-shaped fuser and fail to load.

### `ckpt_tag(ck) -> (model, tag)`
`scratch` when `not is_pretrained(kind)`, else `frozen`/`finetune`. Mirrors `train.py`'s
naming so the CSV filename matches the weights filename.

### `main()`
Loads the model; resolves the output path (`--out_dir` → `$VIT_HOPE_OUT_ROOT/<obj_stem>` →
`results_dir`); builds an un-augmented dataset and subsets it to `ck["test_indices"]`
(falling back to all rows for a checkpoint that lacks them); runs inference under
`no_grad`; divides by `cfg.jitter_scale` to undo the target scaling; prints the offset MSE
and MAE; and writes one row per sample containing:

- `output_folder`, `key` — identity;
- `tvec_*`, `rvec_*` — the **jittered** pose, which `visualize.py` will correct;
- `true_*` — the ground-truth offset;
- `pred_*` — the predicted offset.

**Write-then-rename.** The CSV goes to `out_path + ".tmp"` and is then `os.replace`d into
place, which is atomic on the same filesystem. The sweep scripts treat "predictions CSV
exists" as "this object is done", so a job killed at the walltime mid-write would otherwise
leave a truncated file that makes the resumed job skip an object that never finished.

---

## 10. `visualize.py` — predictions → corrected renders, metrics, collages

The evaluation half. Runs locally because it re-renders.

### Constants
`SSIM_THRESHOLD = 5` (non-black cutoff on the 0–255 scale), `SSIM_KERNEL = 7` (uniform
window, matching skimage's default), `TAGS = ("scratch", "frozen", "finetune")`.

### `detect_variant(pred_path, model=None, tag=None) -> (model, tag)`
Regex-matches `predictions_([a-z0-9]+)_(scratch|frozen|finetune)` against the filename stem
and validates the model against `MODEL_KINDS`. Explicit `--model`/`--tag` win. Anything
unrecoverable stays `None`, outputs go unsuffixed, and it warns loudly that they may
overwrite another model's.

This is where the "no underscores in registry keys" rule is enforced in practice — and it
is a second, independent implementation of the same `(model, tag)` derivation as
`infer.ckpt_tag`. Only `ckpt_tag` reads `MODEL_SPECS["pretrained"]`, so the two can drift;
this is a known, deliberately-unfixed maintenance item.

### `variant_suffix(model, tag) -> str`
`_<model>_<tag>`, or `_<model>`, or `""` — appended to every output name so all nine
variants can be visualized into one directory without collision.

### Rendering helpers

- **`want_xvfb(no_xvfb)`** — true when not disabled, `DISPLAY` is unset, and `xvfb-run`
  exists.
- **`render(cfg, payload, out_path, use_xvfb)`** — same stdin-JSON invocation as
  `generate_jitter.render`, but writing to an **absolute** path. The renderer treats
  `argv[1]` as a plain prefix so it will happily write outside its own repo; `cwd` still
  has to be `renderer_repo` because the asset paths inside the payload are relative to it.
- **`correct_from_predictions(pred_df) -> {run: {key: pose}}`** — the actual correction:
  `corrected = jittered − predicted`, per component, grouped by run folder so each run can
  be rendered as one batch. Emits the renderer's nested `rvec` shape.

### Metrics

- **`load_tile(path)`** — PNG → `(3, H, W)` float in `[0,1]`. Goes through
  `np.array(...)` rather than `np.asarray` because `torch.from_numpy` needs a writable
  buffer.
- **`masked_ssim(pred, ref, threshold) -> (ssim_per_tile, masked_px_count)`** — the key
  metric. The tiles are an object on pure black, so an unmasked SSIM would be dominated by
  agreement about empty space and would look excellent for any prediction. So: compute the
  full SSIM map, build a **union** mask of non-black pixels across both images, and average
  the map over that mask only. Returns `NaN` for a tile whose mask is empty. Two
  correctness notes are baked into the docstring — torchmetrics reflect-pads before a valid
  convolution so the map is the same H×W as the input and the mask aligns pixel-for-pixel;
  and the map is per-channel, so a flat mean over a channel-broadcast mask reproduces
  skimage's `ssim_map[mask].mean()` with `channel_axis=-1`.
- **`masked_ssim_chunked(pred, ref, threshold, batch)`** — chunks the call and **halves the
  batch on `torch.OutOfMemoryError`**, retrying down to 1 before giving up.
- **`sample_score(cfg, pred_dir, ref_dir, key, device, ssim_batch, threshold)`** — mean
  MSE/MAE/masked-SSIM over one sample's 6 tiles, plus tile count and mask fraction.
  Bilinearly resizes the reference if the shapes differ. Returns `None` when no tile pair
  exists or every mask is empty, so the caller can skip cleanly.
- **`print_ssim_summary(mdf, n_requested, threshold, label)`** — the terminal report:
  mean/std/median/min/max for jittered vs corrected, the **per-sample delta distribution**
  (explicitly not the difference of the two column statistics — those are different
  things), the improved-sample count, and mean MSE/MAE before → after.

### Collages

- **`tile_paths(cfg, folder, key)`** — the same `tile-w=R_tile-h=C` convention as
  `StackDataset._tile_paths`, duplicated here because `visualize.py` doesn't construct a
  dataset. A second known drift risk.
- **`montage_tiles(cfg, folder, key)`** — 6 tiles → one 2×3 `PIL.Image`; `None` if any tile
  is missing.
- **`make_comparison(...)`** — a three-panel **Original | Ground Truth | Corrected**
  canvas with centred labels, falling back to the default PIL font if DejaVu is absent.
  Returns `False` if any panel could not be built.

### `main()`
Detects the variant, builds the `Config`, reads the predictions CSV and **validates that
every required column is present** before doing any work. Writes `corrected_poses<suffix>.json`,
then renders:

- `clean/` — the ground-truth render. **Unsuffixed and shared**, because it doesn't depend
  on the model; re-rendered only when missing or `--force_clean`. This is a large saving
  across a nine-variant sweep.
- `corrected<suffix>/<run>/` — one render per run folder.

Then it scores every sample **twice** — corrected-vs-clean and jittered-vs-clean — so the
metrics CSV carries its own baseline and improvement is a within-row comparison. Each row
also carries `model`, `tag`, `obj_id`, so the per-object CSVs concatenate directly into one
cross-model table for the notebook.

---

## 11. The slurm layer

### `slurm/run_all_objects.sh` — the shared body

Not submitted directly. Sourced by a wrapper that has already set `MODEL`. It trains **and**
infers one model over **every** object with rendered data, sequentially, inside one job.

Inputs (all env, all with defaults): `DATA_ROOT`, `REPO_DIR`, `RENDERED_ROOT`, `OUT_ROOT`,
`MODEL` (required), `FREEZE`, `EPOCHS`, `BATCH_SIZE`, `TILE_SIZE`, `FORCE`, `OBJ_IDS`,
`RUN_TAG`, `TRAIN_EXTRA_ARGS`, `CHAIN`, `CHAIN_SCRIPT`.

**Freeze-aware batch defaults.** This is the **only** place per-model batch sizes live —
the wrappers deliberately don't set `BATCH_SIZE`, or the halving below could never fire for
them. Applied only when the caller didn't set one: `mvdinov2`/`mvdinov3` → 4, other `mv*` →
8, `vanilla`/`swinv2` → 16, everything else → 8; then **halved for a multi-view fine-tune**.
The halving is keyed on the resolved `TAG == finetune`, not on `FREEZE`, so a meaningless
`FREEZE=0` on the from-scratch `mvit` doesn't silently halve a batch that was never going to
store backbone activations. The multi-view models run their backbone once per view, so one
sample costs ~6× the activations of the equivalent stacked model. Frozen that is
forward-only and survivable; fine-tuning stores activations for all six views and needs the
extra headroom (an 8-sample `mvswinv2` fine-tune OOMs on an 8 GB card).

**Tag resolution via Python, not shell.** A small inline `python -c` imports `MODEL_SPECS`
and prints `frozen`/`finetune`/`scratch`, exiting with a message for an unknown key. This
runs after `cd $REPO_DIR` and the venv activation, so the flat imports resolve. Two
benefits: shell and Python can never disagree about a filename, and an invalid `MODEL`
fails **before any GPU time is spent**. Because the script runs `set -uo pipefail` *without*
`-e`, a failed `python` would leave `TAG` empty and silently mislabel every output — hence
the explicit non-empty check right after.

**Object discovery.** Globs `${RENDERED_ROOT}/obj_*/jitter_all.csv`, strips to a numeric id
via `$((10#${stem#obj_}))` (base-10 forced, or `000002` would parse as octal). `OBJ_IDS`
overrides.

**The per-object loop.** Skips an object with no `jitter_all.csv` (a partial render is
fine); skips one whose predictions CSV already exists unless `FORCE=1` (this is what makes a
walltime-killed job resumable); otherwise trains, then infers, logging and continuing past
any failure so one bad object doesn't lose the rest of the job. Weights and predictions both
land in `${OUT_ROOT}/obj_XXXXXX/` — on `/data`, never in the repo on `/home`.

**Running a tuned configuration.** `TRAIN_EXTRA_ARGS` is word-split and appended verbatim
to the `train.py` call; `RUN_TAG` adds `--run_tag` and suffixes the checkpoint, the
predictions CSV and the skip-existing check alike. Together they confirm a sweep winner over
all 28 objects: `--export=ALL,MODEL=mvdinov2,RUN_TAG=tuned,TRAIN_EXTRA_ARGS="--lr 2e-4 ..."`,
pasting the flag string `analyze_sweep.py` prints. **`RUN_TAG` is effectively mandatory for a
tuned run**: without it the outputs keep the default names, the skip-existing check finds the
baseline CSV for every object and the job does nothing — and `FORCE=1` "fixes" that by
overwriting the very RESULTS.md baseline the tuned run is meant to beat. An empty `RUN_TAG`
reproduces the historical filenames byte for byte, so untagged runs are unchanged. `sbatch`
splits `--export` on commas, so the flag string may contain spaces but not commas; every
flag `sweep.py` emits is `--key value`, so that holds. `visualize_all.sh` needs no change —
it treats the part after `predictions_` as an opaque suffix and pairs it with
`metrics_<suffix>.csv`.

**Walltime chaining.** With `CHAIN>0` the job submits its own successor before exiting, and
the skip-existing check makes the successor resume. Three guards against a runaway: it only
chains if this run completed ≥1 object (otherwise the successor hits the same wall), only if
objects remain, and `CHAIN` decrements each generation. `FORCE` is explicitly cleared in the
resubmission so the chain is guaranteed to converge. Only one job is ever queued per model,
so the cluster's 2-job limit holds.

Exits non-zero if anything failed.

### `slurm/vit_hope_model.slurm` — the generic launcher
The current way to launch anything. Takes the key via `--export`:

```
sbatch --export=ALL,MODEL=mvdinov2                 slurm/vit_hope_model.slurm
sbatch --export=ALL,MODEL=mvdinov2,FREEZE=0        slurm/vit_hope_model.slurm
sbatch -J vh_mvit --export=ALL,MODEL=mvit,CHAIN=8  slurm/vit_hope_model.slurm
```

`MODEL="${MODEL:?…}"` fails fast with a usage message. The job name is the static
`vh_model`, because `#SBATCH` directives are parsed before the shell runs and cannot
interpolate `$MODEL` — override per submission with `-J`. `CHAIN_SCRIPT` self-references so
chaining keeps working.

This replaced the plan of adding five more near-identical stubs; nine files differing only
in three variables is nine files to keep in sync.

### `slurm/vit_hope_{vanilla,dinov2,dinov3,swinv2}.slurm`
The four legacy per-model wrappers. Each sets `MODEL`, `FREEZE`, and a self-referencing
`CHAIN_SCRIPT`, then sources the same shared body. They predate the generic launcher and
**still work** — they set `MODEL` themselves, which is all the body requires. Fine-tune with
`--export=ALL,FREEZE=0`; the tag becomes `finetune`, so the output filenames differ from the
frozen run and nothing is clobbered or skipped.

### `slurm/vit_hope.slurm` / `slurm/vit_hope_infer.slurm`
The original single-object train+infer and infer-only jobs, superseded by the sweep scripts
but kept for one-off runs. Their headers document the prerequisite that rendering must
happen locally first because the compute node has no OpenGL.

---

## 12. Local orchestration shell scripts

All scripts below live under `bash/`; the Python files they invoke live under `src/`.

| Script | What it does |
|---|---|
| `generate_jitter_all.sh` | Renders every object locally, one at a time. Exists because the renderer always writes to the one fixed staging dir, so it renders there and then **moves** the result into `${DEST_ROOT}/obj_XXXXXX/` — the per-object layout `run_all_objects.sh` globs for. Handles `prepare_poses.py`'s exit-2 ("object not in the split") as a skip. Also deletes the ~250 MB of projection PNGs per object after rendering. |
| `visualize_all.sh` | Runs `visualize.py` over every predictions CSV that has come back. Object-outer, model-inner, for two reasons: the projection PNGs were deleted so `prepare_poses.py` must re-run **per object** before that object can be re-rendered, and `visualize.py` suffixes by model but *not* by object (and `clean/` is unsuffixed by design), so each object needs its own output directory or `run_0/` from one object lands on top of another's. |
| `push_code.sh` | rsyncs source only to the cluster. Excludes `results/`, `weights/`, `.venv/`, `*.pt` — pushing those would overwrite fresher cluster output. Never uses `--delete`. |
| `push_rendered.sh` | Stages the ~2 GB / ~32k-file rendered dataset. Tuned differently from `push_code.sh`: no `-z` (PNGs are already compressed), `--info=progress2` (one line, not 32k filenames), `--partial` (a dropped link resumes). Incremental, so an interrupted run is fixed by re-running. |
| `pull_data.sh` | Brings predictions, sweep result CSVs (`sweeps/*.csv` → `results/sweeps/`, for `analyze_sweep.py`) and slurm logs back; **weights are opt-in** via `--weights`, because the only local consumer is `visualize.py`, which reads a CSV and never loads a checkpoint. Filters hard so the input tiles don't come back with them. Additive, never `--delete`. |

---

## 13. Documentation and analysis

| File | Contents |
|---|---|
| `README.md` | How to run everything: setup, the model table split into Stacked and Multi-view families with parameter counts, `--token_grid`, the cluster workflow, the smoke-test command list. |
| `NEW_MODEL.md` | The multi-view proposal, now annotated with which class and registry key implements each part, and how `token_grid` realises "concatenate into a global patch". Its Status section records that the models are smoke-tested only. |
| `RESULTS.md` | Recorded metrics from the completed sweeps. **Unchanged by the multi-view work** — no real training run has been done for the new models. |
| `ORTHO_TILE_WINDOW.md` | Notes on the orthographic tile windowing in the renderer. |
| `results_comparison.ipynb` | Aggregates every `metrics_<model>_<tag>.csv` into cross-model / cross-object tables and plots. Parses the variant out of the filename with the same convention as `detect_variant`. **Asserts an 8-slot colour palette**, which 5 existing + up to 9 new variants will exceed — flagged in `NEW_MODEL.md`, not yet fixed. |
| `graphify-out/` | Generated knowledge graph: `graph.html`, `GRAPH_REPORT.md`, `graph.json` — 263 nodes / 636 edges / 16 communities over 29 files. |

---

## 14. Cross-cutting invariants

Things that are true across the codebase and that a change should not quietly break.

1. **Every model takes `(B, 18, T, T)` and returns `(B, 6)`.** This is the contract that
   lets `data.py`, `train.py`, `infer.py` and `visualize.py` contain zero per-model
   branching.
2. **Channel layout is view-major**, fixed by `_load_stack`'s `torch.cat` in `_tile_paths`
   order. `split_views` is exact *only* because of this. Reorder the tile loop and the
   multi-view models silently scramble their views.
3. **`MODEL_SPECS[kind]["pretrained"]` is the only place that knows whether a model is
   pretrained.** Read by `train.py`, `infer.py`, and (via `python -c`) `run_all_objects.sh`.
4. **The checkpoint is self-describing.** `model`, `freeze`, `tile_size`, `patch_size`,
   `token_grid`, `obj_id`, `test_indices` — enough to rebuild the exact architecture and the
   exact held-out split. Any new geometry knob must be added here or inference will
   reconstruct the wrong model.
5. **Registry keys are `[a-z0-9]+`** — no underscores, because the `predictions_<model>_<tag>`
   filename is parsed with a regex.
6. **A frozen backbone is in `eval()` mode, not merely `requires_grad=False`.** Enforced by
   `FreezableBackbone.train`.
7. **`GlobalFuser` is identical across all five multi-view variants**, and the `LayerNorm`
   after `proj` is what keeps that true in effect and not just on paper.
8. **Permutation sensitivity must be non-trivial.** If shuffling the six views doesn't change
   the output, the multi-view architecture has degenerated into a bag of views.
9. **Rendering is local, learning is remote.** The compute nodes have no OpenGL.
10. **Sweep scripts treat "predictions CSV exists" as "done".** Hence the atomic
    write-then-rename in `infer.py`.

## 15. Known drift risks

Reported, not fixed:

- `infer.ckpt_tag()` and `visualize.detect_variant()` independently derive `(model, tag)`;
  only the former reads `MODEL_SPECS["pretrained"]`.
- `visualize.tile_paths()` duplicates `StackDataset._tile_paths`'s filename convention.
- `results_comparison.ipynb`'s 8-slot palette is already too small for the variant count.
