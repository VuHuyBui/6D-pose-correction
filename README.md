# vit_hope

Pose-jitter correction for HOPE objects with a ViT. An object's observed RGB is
projected onto its 3D mesh from a **jittered** camera pose, producing 6
orthographic tiles; a ViT regresses the 6D pose offset
`(dt_x,dt_y,dt_z, dr_x,dr_y,dr_z)` so the pose can be corrected by subtraction.

The input is always the 6 RGB tiles concatenated on the channel axis →
`(18, T, T)`, with train-time joint colour-hue jitter. Models differ in how they
read those 18 channels, and fall into two families.

**Stacked** — the 6 views go in as one 18-channel image, so the patch embed sums
them into a single token grid:

| `--model` | backbone (timm) | tile | params | notes |
|-----------|-----------------|------|--------|-------|
| `vanilla` | from-scratch ViT | 518 | 13M | baseline, no pretraining |
| `dinov2`  | `vit_base_patch14_dinov2.lvd142m` | 518 | 89M | |
| `dinov3`  | `vit_base_patch16_dinov3.lvd1689m` | 512 | 89M | gated on HF — needs `HF_TOKEN` |
| `swinv2`  | `swinv2_base_window16_256.ms_in1k` | 256 | 88M | tile must be a multiple of 64 |

**Multi-view** (`models_mv.py`) — each view is encoded separately by a *shared*
local encoder, then a global transformer fuses the views. Tokens carry a learned
view embedding, so the fuser can reason about an offset that is only visible by
comparing projections:

| `--model` | per-view encoder | tile | params | notes |
|-----------|------------------|------|--------|-------|
| `mvit`     | from-scratch ViT | 224 | 16M | baseline, no pretraining |
| `mvdinov2` | `vit_base_patch14_dinov2.lvd142m` | 518 | 100M | |
| `mvdinov3` | `vit_base_patch16_dinov3.lvd1689m` | 512 | 99M | gated on HF — needs `HF_TOKEN` |
| `mvswinv2` | `swinv2_base_window16_256.ms_in1k` | 256 | 100M | final stage (32× reduced) |
| `mvswinlo` | same, stages 0–1 only | 256 | 15M | 4×/8× reduced — keeps small detail |

Each view contributes `--token_grid ** 2` tokens (default 2 → 4 tokens/view) to
the global transformer, average-pooled from the encoder's patch grid.
`--token_grid 1` degenerates to one pooled vector per view. Concatenating raw
patch tokens instead is not an option at this resolution: DINOv2 @ 518 would put
6 × 1369 = 8214 tokens into the fuser.

The multi-view models run the backbone once per view, so a sample costs ~6× the
activations of the stacked equivalent — use a smaller `--batch_size` at 518.
`mvswinlo` is the cheap one: `features_only` truncates SwinV2 after stage 1.

Pretrained backbones support **freeze** (`--freeze`, default) and **fine-tune**
(`--no_freeze`); `--tile_size` defaults per model. See `MODEL_SPECS` in
`models.py`, whose `pretrained` flag is also what decides the `scratch` vs
`frozen`/`finetune` checkpoint tag.

These are the **B**-sized variants: an 18-channel 518×518 input makes ViT-L
(~300M) OOM-prone on a single GPU. To trade back up, set the timm name via
`VIT_HOPE_DINOV2_BACKBONE` / `VIT_HOPE_DINOV3_BACKBONE` / `VIT_HOPE_SWINV2_BACKBONE`
— no code change needed. The `mv*` variants read the same three env vars.

Default object is HOPE `obj_000006`; pass `--obj_id N` to any script for others.

## Layout

| File | Role |
|------|------|
| `config.py` | Paths + hyper-parameters (one `Config` dataclass). |
| `prepare_poses.py` | HOPE `val` GT → renderer request.json + masked projection PNGs. |
| `generate_jitter.py` | Jitter poses + drive the renderer → tiles + `jitter_all.csv`. |
| `data.py` | `StackDataset`: 6 tiles → `(18,T,T)`, joint hue jitter, target×scale. |
| `models.py` | The `MODEL_SPECS` registry, the stacked models, `build_model`. |
| `models_mv.py` | The multi-view models: shared per-view encoder + global transformer. |
| `layers.py` | Blocks shared by both: `ViTTrunk`, `GlobalFuser`, `FreezableBackbone`. |
| `train.py` | Train either model from a prepared `--data_dir`; saves `<obj>_<model>_<tag>.pt` into `--weights_dir` (default `weights/`). |
| `infer.py` | Predict test-set 6D offsets → `predictions_<model>_<tag>.csv` (no renderer). |
| `sweep.py` | Emit a seeded random-search plan of `train.py` flags for one model (does not train). |
| `analyze_sweep.py` | Rank a sweep's trials by mean best-val loss against the default config. |
| `visualize.py` | Read predictions → correct → re-render → metrics + Original\|GT\|Corrected collages. |
| `visualize_all.sh` | Run `visualize.py` over every pulled predictions CSV, one output dir per object. |

## One-time prerequisite: configure the renderer for HOPE

Build the C++ renderer once:

```bash
cd ../texture-projection-opengl-cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

`prepare_poses.py` reads HOPE's `camera.json` and writes the focal lengths,
principal point, width, and height to the renderer's `intrinsics.json`.
No HOPE camera values need to be hard-coded.

The renderer needs an OpenGL context; on a headless machine the scripts wrap it
in `xvfb-run` automatically (disable with `--no_xvfb`).

## Pipeline

```bash
cd vit_hope

# 1. HOPE GT poses -> request.json + projection images
python prepare_poses.py --obj_id 6

# 2. jitter + render tiles (writes jitter_all.csv). Add --no_render for a dry run.
python generate_jitter.py --seed 42 --jitters_per_image 5

# 3. sanity check the model wiring (overfits 8 samples; loss should reach ~0)
python train.py --model dinov2 --freeze --overfit_test
python train.py --model vanilla --overfit_test
python train.py --model mvdinov2 --freeze --overfit_test
python train.py --model mvit --overfit_test

# 4. train
python train.py --model dinov2 --freeze --augment
python train.py --model dinov2 --no_freeze --lr 3e-5 --augment
python train.py --model vanilla --augment

# Train from prepared data stored somewhere else
python train.py --data_dir /path/to/results/hope --model dinov2 --freeze --augment

# 5. infer test-set jitter (no renderer; safe on a GPU node)
#    predictions land in vit_hope/results/ unless --out_dir or
#    $VIT_HOPE_OUT_ROOT says otherwise
python infer.py --weights weights/obj_000006_dinov2_frozen.pt
python infer.py --weights /data/USER/vit_hope_rendered/obj_000002/obj_000002_dinov2_frozen.pt \
  --data_dir /data/USER/vit_hope_rendered/obj_000002 \
  --out_dir  /data/USER/vit_hope_rendered/obj_000002
```

`T` (tile size) must be a multiple of 14 (DINOv2 patch size); default 518.

The 6 tiles are rendered through a **0.28 m orthographic window shared by every
object and every view**, so a pixel is the same 1/3.857 mm everywhere. See
[ORTHO_TILE_WINDOW.md](ORTHO_TILE_WINDOW.md) for why — and for why the navy
regions in the tiles are expected rather than a bug.

## Generate jitter for every object

`generate_jitter_all.sh` runs steps 1–2 for all 28 HOPE objects. It must run
**locally** — the renderer needs OpenGL, which the GPU nodes do not have.

```bash
./generate_jitter_all.sh                      # all objects
OBJ_IDS="2 6 14" ./generate_jitter_all.sh     # just these
NO_RENDER=1 ./generate_jitter_all.sh          # CSV/JSON only, quick check
FORCE=1 ./generate_jitter_all.sh              # redo finished objects
```

The renderer always writes to one fixed staging directory
(`Config.output_dir` = `../texture-projection-opengl-cpp/results/hope`), so the
script renders one object at a time and then **moves** the result into
`${DEST_ROOT}/obj_XXXXXX/` — the per-object layout the sweep slurm scripts
expect. `DEST_ROOT` defaults to `../vit_hope_rendered`.

- The repo `.venv` is activated automatically if one exists and none is active
  (`cv2` is not in the base interpreter).
- Objects with no ground truth in the `val` split are logged under `no GT` and
  skipped; only `val` carries GT poses. `prepare_poses.py` exits **2** for that
  case and **1** for anything genuinely broken, so a real error lands in
  `failed`, not silently in `no GT`.
- An object whose `jitter_all.csv` already exists is skipped (`FORCE=1` to redo),
  so an interrupted run just resumes.
- `jitter_all.csv` is moved **last**, so an interrupted move leaves no CSV and
  the object is correctly seen as unfinished next time.
- Budget ~1.1 GB of tiles per object (**~30 GB** for all 28). The 250 MB of
  projection PNGs in `assets/hope` are deleted after each object since they are
  only needed during that object's render (`KEEP_ASSETS=1` to retain them).
- Other overrides: `JITTERS_PER_IMAGE`, `DT_STD`, `DR_STD`, `SEED`, `LIMIT`, `MASK`.

## Train on a compute node

Rendering remains local. Copy the per-object directories to the compute node;
each must contain `jitter_all.csv` and the `run_*` folders. The compute node
does not need the renderer or `VIT_HOPE_RENDERER_REPO`.

```bash
# Local machine: copy prepared data (all objects)
rsync -av ../vit_hope_rendered/ USER@CLUSTER:/data/USER/vit_hope_rendered/

# ...or a single object's flat directory, for the older per-object slurm scripts
rsync -av ../texture-projection-opengl-cpp/results/hope/ \
  USER@CLUSTER:/data/USER/vit_hope_rendered/

# Cluster login node — train then infer
sbatch --export=ALL,RENDERED_DATA_DIR=/data/USER/vit_hope_rendered \
  vit_hope/slurm/vit_hope.slurm

# Inference only (already trained)
sbatch --export=ALL,OBJ_ID=2,RENDERED_DATA_DIR=/data/USER/vit_hope_rendered \
  vit_hope/slurm/vit_hope_infer.slurm

# Or point at a specific checkpoint
sbatch --export=ALL,WEIGHTS=/home/USER/daad-rise-2026/vit_hope/weights/obj_000002_dinov2_frozen.pt,\
RENDERED_DATA_DIR=/data/USER/vit_hope_rendered \
  vit_hope/slurm/vit_hope_infer.slurm
```

`vit_hope.slurm` passes `RENDERED_DATA_DIR` to `train.py --data_dir` and then
runs `infer.py`, writing `predictions_<model>_<tag>.csv` into that directory.
`vit_hope_infer.slurm` only runs inference.

## Sweep every object, one job per model

`slurm/vit_hope_model.slurm` trains **and** infers one model over **every object
that has rendered data**, sequentially inside a single job. Pass the model key
via `--export`; it works for any key in `MODEL_SPECS`:

```bash
sbatch --export=ALL,MODEL=mvdinov2 vit_hope/slurm/vit_hope_model.slurm
sbatch --export=ALL,MODEL=mvdinov2,FREEZE=0 vit_hope/slurm/vit_hope_model.slurm
sbatch -J vh_mvswinlo --export=ALL,MODEL=mvswinlo vit_hope/slurm/vit_hope_model.slurm
```

`-J` is worth passing because `#SBATCH` directives are parsed before the shell
runs, so the job name can't interpolate `${MODEL}` on its own.

The older `slurm/vit_hope_{dinov2,dinov3,swinv2,vanilla}.slurm` wrappers still
work — they just set `MODEL` themselves. All of them, old and new, source
`slurm/run_all_objects.sh`, which holds the loop.

Data layout on the node — one directory per object under `RENDERED_ROOT`:

```
/data/USER/vit_hope_rendered/obj_000002/jitter_all.csv + run_*/
/data/USER/vit_hope_rendered/obj_000006/...
```

```bash
sbatch --export=ALL,RENDERED_ROOT=/data/USER/vit_hope_rendered \
  vit_hope/slurm/vit_hope_dinov2.slurm
sbatch --export=ALL,RENDERED_ROOT=/data/USER/vit_hope_rendered \
  vit_hope/slurm/vit_hope_swinv2.slurm
# ...then dinov3 and vanilla once those two finish.
```

Both outputs land in `${OUT_ROOT}/obj_XXXXXX/` (`OUT_ROOT` defaults to
`RENDERED_ROOT`): the checkpoint `obj_XXXXXX_<model>_<tag>.pt` and the
predictions `predictions_<model>_<tag>.csv`. On Cassandra that is `/data`, so
nothing large is written to the repo on `/home`.

Behaviour worth knowing:

- Objects are **discovered by globbing** `obj_*/jitter_all.csv`, so a partial
  render is fine. Restrict explicitly with `OBJ_IDS="2 6 14"`.
- An object whose predictions CSV already exists is **skipped**, so a job killed
  by the walltime can just be resubmitted. `FORCE=1` redoes everything.
  `infer.py` writes that CSV atomically (temp file + `os.replace`), so a job
  killed mid-write can never leave a truncated CSV that makes the next run skip
  an object which never actually finished.
- A failing object is logged and the loop continues; the job exits non-zero at
  the end if any object failed, and the summary lists done/skipped/failed.
- Overrides: `FREEZE=0` (fine-tune), `EPOCHS`, `BATCH_SIZE`, `TILE_SIZE`,
  `REPO_DIR`, `DATA_ROOT`, `OUT_ROOT`.
- `BATCH_SIZE` defaults per model when you don't set it: 4 for `mvdinov2` /
  `mvdinov3` (6 views at 518 is ~6× the activations of the stacked version),
  8 otherwise.
- The `scratch` vs `frozen`/`finetune` tag is read from
  `MODEL_SPECS[MODEL]["pretrained"]` rather than restated in shell, so an
  unknown `MODEL` fails immediately, before any GPU time is spent.
- `dinov3` needs `HF_TOKEN` exported (gated weights) or every object fails at
  `timm.create_model`.

### Walltime chaining

The `gpu` partition caps jobs at **24h** (`sinfo -o "%P %l"`), which may not
cover every object. `CHAIN=N` makes a job submit its own successor before
exiting, and skip-existing makes that successor resume where the last one
stopped:

```bash
sbatch --export=ALL,CHAIN=8,RENDERED_ROOT=/data/USER/vit_hope_rendered \
  vit_hope/slurm/vit_hope_dinov2.slurm
```

Only one job ID is queued per model at a time, so the 2-job limit still holds.
Chaining is **off by default** (`CHAIN=0`) and guarded against runaway loops —
a job only resubmits if it completed at least one object *and* objects remain,
`CHAIN` decrements each generation and stops at 0, and the successor always
gets `FORCE=0` so the chain converges. A job that completes zero objects
refuses to chain and says so, rather than spinning on a broken config.

Confirm the backbone names resolve in the node's timm before burning a job:

```bash
python -c "import timm; print(timm.list_models('*dinov3*')); print(timm.list_models('swinv2_base*'))"
```

### Bring the results back

```bash
./pull_data.sh --dry-run    # preview
./pull_data.sh              # predictions CSVs + slurm logs
./pull_data.sh --weights    # ...and the .pt checkpoints too
```

Weights are **not** pulled by default. Nothing local reads them: `visualize.py`
corrects poses from a predictions CSV and re-renders, so a sweep's worth of
checkpoints would be transferred for nothing. Ask for them with `--weights`
when you actually want to resume or fine-tune from one. Predictions land in
`results/remote_predictions/obj_*/`, logs in `results/slurm_logs/`, weights
(flat) in `weights/`. Edit the `EDIT ME` block at the top of the script to
point it at your own account.

## Hyperparameter tuning on a compute node

`slurm/vit_hope_sweep.slurm` runs a seeded random search for one model over a
**small subset of objects**, so it finishes in days rather than weeks. Trial
`default` always runs first with no flags at all — it reproduces the current
configuration, and it is the row every other trial is measured against.

```bash
# the from-scratch pair first: cheapest, and the only models where embed_dim is
# the width of the patch-embed linear projection itself
sbatch -J vh_sw_mvit    --export=ALL,MODEL=mvit    vit_hope/slurm/vit_hope_sweep.slurm
sbatch -J vh_sw_vanilla --export=ALL,MODEL=vanilla vit_hope/slurm/vit_hope_sweep.slurm

# then the pretrained ones, where only the cross-view fuser is tunable
sbatch -J vh_sw_mvdino2 --export=ALL,MODEL=mvdinov2,N_TRIALS=12,CHAIN=4 vit_hope/slurm/vit_hope_sweep.slurm
sbatch -J vh_sw_dino2   --export=ALL,MODEL=dinov2,N_TRIALS=8            vit_hope/slurm/vit_hope_sweep.slurm

# read the answer
python analyze_sweep.py /data/$USER/vit_hope_rendered/sweeps/*.csv --per_object
```

Any `MODEL_SPECS` key works, as with `vit_hope_model.slurm`. What actually gets
searched comes from `models.ARCH_KEYS`, because a pretrained backbone's width,
depth and head count are fixed by its weights:

| model | searched |
|-------|----------|
| `vanilla` | `embed_dim` (the 18-channel patch-embed conv), `depth`, `num_heads`, `mlp_ratio`, `lr`, `dropout` |
| `mvit` | the above for the per-view patch embed + local trunk, **plus** `fuser_*`, `lr`, `dropout` |
| `mvdinov2` / `mvdinov3` / `mvswinv2` | `fuser_embed_dim`, `fuser_depth`, `fuser_heads`, `fuser_mlp_ratio`, `lr`, `dropout` |
| `mvswinlo` | the fuser four, plus `stage_dim`, `lr`, `dropout` |
| `dinov2` / `dinov3` / `swinv2` | `lr`, `dropout` only — fixed reference points |

Passing a flag a model does not have is an error, not a silent no-op, so a sweep
can never quietly train the same model 25 times.

Knobs: `EPOCHS`, `N_TRIALS`, `SWEEP_SEED`, `SWEEP_OBJ_IDS` (default `"21 2 11 6"`
— a spread of dataset sizes), `BATCH_SIZE`, `RESULT_CSV`, `INFER=1`, `FORCE=1`.
Every trial keeps `--seed 0`, so all trials share one train/val/test split and
their val losses are directly comparable. Resume granularity is the
(trial, object) pair keyed on the result CSV, so `CHAIN=4` picks up mid-trial
after a walltime kill. Smoke-test a submission with
`N_TRIALS=1,EPOCHS=2,SWEEP_OBJ_IDS=6` before spending a day of GPU.

The same flags work by hand, and the resolved architecture is stored in the
checkpoint so `infer.py` can rebuild a tuned model:

```bash
python train.py --model mvit --embed_dim 192 --num_heads 12 --depth 4 \
                --fuser_embed_dim 256 --lr 1e-4 --run_tag t01
```

Confirm the winner over all 28 objects with the normal
`slurm/vit_hope_model.slurm` path once the search has picked it.

## Visualize trained results locally

Inference produces the predictions CSV (on the cluster or locally). Copy that
CSV next to the local rendered dataset if needed, then run `visualize.py`
(requires the OpenGL renderer — and **not** a checkpoint):

```bash
# Correct poses, re-render, score metrics, write side-by-side collages
python visualize.py \
  --predictions results/predictions_dinov2_frozen.csv \
  --obj_id 2

# Skip re-rendering if clean/ and corrected/ tiles already exist
python visualize.py \
  --predictions results/predictions_dinov2_frozen.csv \
  --no_render

# Metrics only: no re-render, no collages
python visualize.py \
  --predictions results/predictions_dinov2_frozen.csv \
  --no_render --no_comparisons

# All four backbones into the same results/ directory — nothing collides
for f in results/predictions_*.csv; do python visualize.py --predictions "$f"; done
```

`visualize.py` reads the model and tag off the `predictions_<model>_<tag>.csv`
filename (override with `--model` / `--tag`) and **suffixes every output with
it**, so the four backbones can share one `results/` directory. `--tile_size`
now defaults to the detected model's native tile, so `swinv2` (256) no longer
trips the `tile_size % patch_size` check.

### Visualize every object at once

```bash
./visualize_all.sh                       # every CSV under results/remote_predictions/
OBJ_IDS="2 6 14" ./visualize_all.sh      # just these objects
DRY_RUN=1 ./visualize_all.sh             # print the commands, run nothing
NO_COMPARISONS=1 ./visualize_all.sh      # metrics only, no collages
FORCE=1 ./visualize_all.sh               # redo finished object/model pairs
```

The loop is **object-outer, model-inner**, for two reasons:

- The projection PNGs in `assets/hope` are deleted after each object renders
  (~250 MB each), so `prepare_poses.py` has to regenerate them before that
  object can be re-rendered — once per object, not once per model.
- `visualize.py` suffixes its outputs with `_<model>_<tag>` but **not** with the
  object, and `clean/` is unsuffixed by design. So each object gets its own
  `results/viz/obj_XXXXXX/`, otherwise `run_0/` from one object would land on
  top of `run_0/` from another.

Every model for a given object therefore reuses one `prepare_poses.py` run and
one `clean/` render. An object/model pair whose `metrics_<model>_<tag>.csv`
already exists is skipped, so an interrupted run just resumes. Objects with no
local tiles (`jitter_all.csv`) are reported under `no tiles` and skipped —
nothing to compare against. Budget a few hundred MB per object-model for the
collages.

Concatenate the per-object metrics into one cross-model table:

```bash
{ head -1 results/viz/obj_000001/metrics_*.csv | sed -n 2p; \
  awk 'FNR>1' results/viz/obj_*/metrics_*.csv; } > results/viz/metrics_all.csv
```

Everything vit_hope produces goes to `results/` (`Config.results_dir`, override
with `VIT_HOPE_RESULTS_DIR` or `--out_dir`). Two further env vars relocate
training output for the cluster, both unset by default so a laptop run is
unaffected: `VIT_HOPE_WEIGHTS_DIR` (checkpoint destination, else
`vit_hope/weights`, also `--weights_dir`) and `VIT_HOPE_OUT_ROOT` (predictions
go to `<root>/obj_XXXXXX/`, else `results/`). The renderer's own
`../texture-projection-opengl-cpp/results/hope` tree holds only the *inputs* —
`jitter_all.csv`, `request.json`, and the jittered `run_*` tiles.

- `predictions_<model>_<tag>.csv` — true/pred offsets + jittered poses (from `infer.py`).
- `corrected_poses_<model>_<tag>.json` — corrected poses.
- `metrics_<model>_<tag>.csv` — one row per test image, carrying `model`, `tag`
  and `obj_id` columns so the per-model CSVs concatenate into one cross-model
  table. Holds jittered versus corrected MSE, MAE, and **masked SSIM**
  (`ssim_masked_*`). SSIM is averaged over non-black pixels only
  (union mask, `--ssim_threshold` on the 0-255 scale, default 5), matching
  `vit_jitter`'s `compute_masked_ssim`; an unmasked SSIM on these tiles is
  dominated by the black background. Runs on the GPU by default (`--device`,
  `--ssim_batch`); `--no_comparisons` gives a metrics-only run.
- `clean/` — the GT render. Model-independent, so it is **shared**: it is
  rendered once and reused by later models (`--force_clean` to redo it).
- `corrected_<model>_<tag>/` — that model's corrected tile renders.
- `comparisons_<model>_<tag>/{run}__{key}.png` — Original | Ground Truth |
  Corrected (*model*) collage.

A CSV whose name does not match the pattern warns and falls back to unsuffixed
outputs (`metrics.csv`, `corrected/`, …) — the pre-multi-model layout.

## Notes

- Pose conversion (in `prepare_poses.py`): HOPE GT is `(cam_R_m2c, cam_t_m2c mm)`
  for the original model; the renderer's OBJ is centred + in metres, so
  the renderer-axis compensation is applied to `R`, while
  `tvec_m = (R @ C_mm + t_mm)/1000`, where `C_mm` is the original AABB centre
  from `models_info.json`.
- Only HOPE `val` scenes carry ground-truth poses, so they are the data source.
