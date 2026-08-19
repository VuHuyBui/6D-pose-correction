# vit_hope quiz

30 questions on the code as it currently stands. Answers at the bottom.
Sections: pipeline (1–6), config (7–10), data (11–15), models (16–24),
train/infer (25–28), visualize (29–30).

---

## Pipeline

1. Name the four scripts of the pipeline in execution order, and say in one
   phrase what each produces.
2. What exactly is the model's regression target, and how is the pose corrected
   once it is predicted?
3. `prepare_poses.py` converts a HOPE GT pose to renderer convention with
   `tvec_m = (R @ C_mm + t_mm) / 1000`. Why the `R @ C_mm` term, and when is it
   ~zero?
4. Why is the rotation passed to the renderer as `Rodrigues(R @ diag(1,-1,-1))`
   rather than `Rodrigues(R)`?
5. Why are the instance keys written as `{scene}_{img:06d}_{inst:06d}` with
   zero padding, rather than plain integers?
6. `prepare_poses.py` exits with code 2 in one specific situation. Which, and
   why is it not just exit 1?

## Config

7. `Config.__post_init__` enforces a single invariant. What is it, and which
   two backbones would break it if `tile_size` were left at 518?
8. What is `in_channels`, where does the number come from, and what does a
   single channel triple represent?
9. Which four env vars relocate paths, and what problem on the cluster do
   `VIT_HOPE_WEIGHTS_DIR` / `VIT_HOPE_OUT_ROOT` exist to solve?
10. `jitter_scale = 100.0`. Where is it applied, where is it undone, and what
    goes wrong if you forget the second one?

## Data

11. Describe how `StackDataset` turns one CSV row into a tensor, with the final
    shape.
12. The crop box is computed once, from `self.rows[0]`, and reused for every
    sample. State the assumption this bakes in and one way it could bite.
13. In `_compute_shared_crop_box`, what are `thresh=8` and `margin=16` for, and
    what is the fallback when no pixel exceeds the threshold?
14. Colour jitter samples *one* set of parameters per sample and applies it to
    all 6 tiles. Why not sample independently per tile?
15. `train.py` builds two `StackDataset` objects over the same CSV. Why two, and
    what would break if it reused one for train and val?

## Models

16. What is the architectural difference between the "stacked" models in
    `models.py` and the multi-view models in `models_mv.py`?
17. `split_views` is described as "a pure reshape -- no data movement". What
    property of `data.py` makes that true, and what would go wrong if `data.py`
    stacked tile-major instead?
18. Why must a frozen backbone stay in `eval()` mode even under `model.train()`?
19. `GlobalFuser` adds two learned embeddings to every token. Name both and say
    what capability the view embedding buys over a concat-and-MLP.
20. The view/token embeddings are initialised with `std=0.25` while the CLS uses
    `std=0.02`. Give the reasoning stated in the code.
21. Why does `GlobalFuser.proj` end in a `LayerNorm`, and what comparison would
    be invalidated without it?
22. `MVSwinV2Low` taps stages 0–1 instead of the final stage. What is the
    motivation, and how many tokens per view does it emit?
23. `MVSwinV2Low` overrides four hook methods from `MultiViewTimm`. Why can it
    not just use `self.backbone.num_features`?
24. What does `migrate_state_dict` fix, and which single model kind does it
    touch?

## Train / infer

25. What is the default learning rate for (a) `--model dinov2 --freeze`,
    (b) `--model dinov3 --no_freeze`, (c) `--model vanilla`?
26. `overfit_test` uses the run's real LR rather than a fixed 1e-3. What failure
    mode did the fixed value produce, and why was it misleading?
27. The checkpoint stores `test_indices`. Who consumes it, and what leak does
    storing it prevent?
28. `infer.py` writes to `out_path + ".tmp"` then `os.replace`. What concrete
    failure does that guard against?

## Visualize

29. Why is SSIM masked to non-black pixels instead of computed over the whole
    tile?
30. `clean/` is rendered once and shared across models, while `corrected_*/` is
    per-model. Why is that split correct?

---

# Answers

1. `prepare_poses.py` (renderer-ready `request.json`, the centred/metres `.obj`,
   per-instance projection PNGs, `intrinsics.json`) → `generate_jitter.py`
   (jittered poses per run + the rendered 6-tile sets + `jitter_all.csv`) →
   `train.py` (a checkpoint) → `infer.py` (`predictions_<model>_<tag>.csv`),
   with `visualize.py` optionally consuming the predictions to re-render and
   score.
2. The 6D offset `[dt_x, dt_y, dt_z, dr_x, dr_y, dr_z]` — translation in metres
   and a Rodrigues-vector delta — scaled by `jitter_scale`. Correction is
   subtractive: `corrected_pose = jittered_pose - predicted_offset`
   ([visualize.py:112](visualize.py#L112)).
3. The renderer's OBJ is centred and in metres, so `p_orig_mm = 1000*p_obj + C_mm`.
   Substituting into HOPE's `p_cam_mm = R @ p_orig_mm + t_mm` leaves the extra
   `R @ C_mm`. It is ~0 only when the source mesh is already centred (true for
   `obj_000006`); the term keeps the code correct for the others.
4. The renderer conjugates rotations with `diag(1,-1,-1)`, so feeding `R @ F`
   makes its OpenGL rotation come out as `F @ R` — the intended OpenCV→OpenGL
   axis flip.
5. `nlohmann::json` on the C++ side stores object keys in lexicographic order.
   Padding makes lexicographic order agree with numeric scene/image/instance
   order, including for objects appearing multiple times.
6. Exit 2 when the split contains no instances of the requested `obj_id`. It is
   distinct from 1 so a sweep can tell "object simply absent from this split"
   from "prepare actually broke".
7. `tile_size % patch_size == 0`, so the patch embedding tiles the image exactly.
   `dinov3` (patch 16 → 512) and `swinv2`/the MV Swin variants (effective 64 →
   256) would break at 518.
8. `n_rows * n_cols * 3 = 18`. Each consecutive channel triple is one
   orthographic RGB tile of the object, in grid order.
9. `VIT_HOPE_HOPE_ROOT`, `VIT_HOPE_RENDERER_REPO`, `VIT_HOPE_RESULTS_DIR`,
   `VIT_HOPE_WEIGHTS_DIR`, `VIT_HOPE_OUT_ROOT` (plus `VIT_HOPE_SPLIT` for the
   split, and the per-model `VIT_HOPE_<MODEL>_BACKBONE`). On the cluster `/home`
   holds code and `/data` holds data; the last two keep checkpoints and
   predictions next to the tiles rather than on `/home`. Both are
   empty-by-default, so laptop behaviour is unchanged.
10. Applied in `StackDataset.__getitem__` when building the target
    ([data.py:116](data.py#L116)); undone in `infer.py` by dividing the network
    output by `cfg.jitter_scale` ([infer.py:118](infer.py#L118)). Forgetting it
    yields offsets 100x too large, so subtractive correction destroys the pose
    and the reported MSE/MAE are meaningless.
11. Locate the 6 tiles at
    `<data_dir>/<output_folder>/<key>_tile-w={r}_tile-h={c}.png`, crop each to
    the shared box, resize to `(tile_size, tile_size)`, `ToTensor`, concatenate
    on the channel axis → `(18, T, T)`; then optional joint colour jitter and
    `(x - 0.5) / 0.5` normalisation. Target is the 6D scaled offset.
12. That every sample's object occupies roughly the same image region, i.e. the
    renderer's orthographic framing is fixed across poses and runs. If framing
    varies — a much closer/farther pose, or a different object extent — the box
    from row 0 can clip the object or leave it tiny inside a loose box.
13. `thresh=8` separates object pixels from the near-black background;
    `margin=16` pads the tight bounding box so the crop does not shave the
    silhouette. If no pixel exceeds the threshold in any tile (an all-black
    render), it falls back to the full image.
14. Because all 6 tiles are views of one object under one renderer, and the pose
    offset is only observable by comparing views. Independent per-tile jitter
    would inject appearance differences that mimic the very cross-view signal
    the model must read.
15. One with `augment=args.augment` for training, one with `augment=False` for
    val/test, indexed by the same split indices. Sharing one dataset would
    either evaluate on colour-jittered inputs (noisy, non-comparable val loss)
    or disable augmentation for training.
16. Stacked: the 6 views enter as one 18-channel image, so the patch-embed conv
    *sums* all six into a single token grid. Multi-view: the input is reshaped to
    `(B*6, 3, T, T)`, a shared local encoder produces per-view tokens, and a
    global transformer fuses them while knowing which view each token came from.
17. `_load_stack` concatenates tiles on the channel axis in grid order, so the
    layout is already view-major; `reshape(B*V, 3, H, W)` therefore lands view
    `v` of sample `b` at row `b*V + v` with no permute. Tile-major stacking (all
    reds, then all greens, …) would make the same reshape silently mix channels
    across views.
18. Its dropout / stochastic depth would keep perturbing the very features the
    trainable head is fitting. `FreezableBackbone.train` re-applies `eval()`
    after `super().train(mode)` (the mixin is listed before `nn.Module` so it
    wins).
19. A per-view embedding (`view_embed`, one vector per view) and a
    within-view position embedding (`tok_embed`). The view embedding lets
    attention condition on *which* projection a token came from, so the encoder
    can reason about an offset that is only visible in view-to-view comparison —
    a concat-and-MLP treats the views as an unordered bag.
20. 0.02 is tuned for ~1k positions and a long pretraining run that grows into
    them. Here there are only 6 views and a short fine-tune; at 0.02 the fused
    output is <1% sensitive to view identity, i.e. nearly a bag of views. At 0.25
    sensitivity is 12–20% with no measured convergence cost.
21. Raw feature magnitude varies ~4x across backbones (SwinV2's final stage vs
    DINOv2 vs an early Swin stage), so a fixed-std view embedding would be 8% of
    a token for one backbone and 33% for another. Without the norm, the
    like-for-like comparison of backbones under an identical fuser is invalid.
22. The final stage is 32x-reduced (8x8 at 256px) and semantic, while a
    millimetre-scale pose offset is a sub-object cue; stages 0–1 are 4x/8x
    reduced (64x64, 32x32) and retain it. Each view emits
    `len(OUT_INDICES) * token_grid**2 = 2 * token_grid**2` tokens (8 at the
    default `token_grid=2`).
23. `features_only=True` returns a list of NHWC maps and exposes no
    `num_features`, so channel dims must come from `feature_info.channels()`
    and a per-stage `Linear` brings the two stages to a common `STAGE_DIM=256`.
    It overrides `_build_backbone`, `_post_backbone`, `_feature_dim`,
    `_tokens_per_view` and `_view_tokens`.
24. It re-prefixes `patch./cls/pos/encoder./norm.` keys with `trunk.` for
    checkpoints written before `VanillaViT` moved those modules into the shared
    `ViTTrunk`, so the old `vanilla_scratch` weights behind RESULTS.md still
    load. It applies only to `kind == "vanilla"` (and is a no-op if `trunk.`
    keys are already present).
25. (a) 3e-4 — frozen, so only the head trains. (b) 3e-5 — pretrained and
    unfrozen. (c) 3e-4 — not pretrained, so the `--freeze` branch is irrelevant.
26. At a fixed 1e-3 the multi-view stack (local ViT + global fuser) is unstable
    enough to stall near the target variance. That looks exactly like a wiring
    bug — the thing the test exists to detect — when it is only a step-size
    problem.
27. `infer.py` reads `ck["test_indices"]` to rebuild the held-out subset. It
    prevents evaluating on samples the model trained on: the split is derived
    from a seed and fractions, so any later change to `--seed`, `--val_frac`,
    `--test_frac`, or the number of renderable rows would silently reshuffle it.
28. A job killed at walltime mid-write leaving a truncated CSV. The sweep scripts
    treat "predictions CSV exists" as "object done", so a partial file would make
    them skip an object that never finished. `os.replace` is atomic within one
    filesystem, so the destination only ever appears complete.
29. The tiles are an object on a pure black background, so an unmasked SSIM is
    dominated by empty space and barely moves with pose quality. The mask is the
    union of non-black pixels (threshold 5 on the 0–255 scale) over the two
    images being compared, matching `vit_jitter`'s `compute_masked_ssim`.
30. `clean/` is the ground-truth-pose render, which depends only on the object
    and not on any model, so re-rendering it per model is pure waste (it is
    redone only when missing, or with `--force_clean`). `corrected_*/` depends on
    that model's predicted offsets, so it must be suffixed per
    `<model>_<tag>` to avoid clobbering.
