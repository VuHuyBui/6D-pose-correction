# ViT pose refinement on HOPE — model comparison

**Task.** Given a jittered 6D object pose and the rendered tiles, predict the offset;
corrected pose is `jittered − predicted`. Translation only (rotation jitter is zero in the
current data), so pose errors are translation L2 in **millimetres**.

**Setup.** 28 HOPE objects × 7 variants, each evaluated on the same held-out 10% test
split per object — 453 samples per variant, 5–27 per object. No correction: **16.72 mm
mean / 16.56 mm median**. Masked SSIM before correction: **0.7485** (identical for all
variants, same jittered renders).

The `mv*` variants are multi-view heads (`models_mv.py`); everything else is single-view.

---

## Pose error

| variant | mean err | median err | median improvement | samples improved |
| --- | --- | --- | --- | --- |
| dinov2_frozen | 14.65 mm | 13.22 mm | +14.0% | 66% |
| mvswinv2_frozen | 14.78 mm | 13.95 mm | +11.3% | 64% |
| mvdinov2_frozen | 15.24 mm | 14.40 mm | +3.5% | 62% |
| dinov3_frozen | 15.27 mm | 13.95 mm | +7.7% | 62% |
| swinv2_frozen | 15.78 mm | 14.86 mm | +7.1% | 57% |
| dinov3_finetune | 15.94 mm | 15.18 mm | +1.1% | 54% |
| vanilla_scratch *(control)* | 16.50 mm | 15.55 mm | +0.2% | 51% |
| *no correction* | *16.72 mm* | *16.56 mm* | *—* | *—* |

`improvement` = per-sample `(err_before − err_after) / err_before`.

---

## Render quality (masked SSIM)

Corrected pose re-rendered and scored against ground truth. Before = 0.7485 for all.
Complete grid — every object rendered for every variant, so this ranking needs no
fair-subset restriction.

| variant | SSIM after | mean Δ | median Δ | samples improved | MSE | MAE |
| --- | --- | --- | --- | --- | --- | --- |
| dinov2_frozen | 0.7700 | +0.0215 | +0.0163 | 76% | 0.00068 | 0.00220 |
| mvswinv2_frozen | 0.7683 | +0.0198 | +0.0138 | 72% | 0.00071 | 0.00225 |
| dinov3_frozen | 0.7625 | +0.0140 | +0.0094 | 67% | 0.00078 | 0.00239 |
| swinv2_frozen | 0.7611 | +0.0126 | +0.0065 | 67% | 0.00080 | 0.00245 |
| mvdinov2_frozen | 0.7607 | +0.0122 | +0.0038 | 66% | 0.00082 | 0.00247 |
| dinov3_finetune | 0.7564 | +0.0079 | +0.0029 | 61% | 0.00083 | 0.00251 |
| vanilla_scratch *(control)* | 0.7514 | +0.0029 | +0.0006 | 53% | 0.00090 | 0.00265 |

Pearson r between mean pose improvement and mean ΔSSIM, over all 196 object × model
pairs: **+0.66**.

---

## Paired vs. from-scratch control

All variants share the identical test split per object, so this is a per-sample paired
comparison against `vanilla_scratch`. Every pretrained backbone beats the control.

| variant | Δ vs control | relative | samples beating control | objects beating control |
| --- | --- | --- | --- | --- |
| dinov2_frozen | −1.85 mm | −11.2% | 64% | 24 / 28 |
| mvswinv2_frozen | −1.73 mm | −10.5% | 62% | 27 / 28 |
| mvdinov2_frozen | −1.26 mm | −7.6% | 59% | 22 / 28 |
| dinov3_frozen | −1.23 mm | −7.5% | 57% | 23 / 28 |
| swinv2_frozen | −0.73 mm | −4.4% | 53% | 21 / 28 |
| dinov3_finetune | −0.57 mm | −3.4% | 56% | 19 / 28 |

`mvswinv2_frozen` is the most consistent — it wins on 27 of 28 objects, more than
`dinov2_frozen` does, even though its pooled mean is slightly worse.

---

## Per-object winner

Lowest mean pose error per object, out of 28:

| dinov2_frozen | mvdinov2_frozen | mvswinv2_frozen | dinov3_frozen | swinv2_frozen |
| --- | --- | --- | --- | --- |
| 10 | 7 | 7 | 3 | 1 |

`dinov3_finetune` and `vanilla_scratch` win no object.

---

## Multi-view vs single-view

Frozen-only comparison, the only one the runs support:

- `mvswinv2_frozen` (14.78 mm) clearly beats `swinv2_frozen` (15.78 mm) — multi-view helps
  the Swin backbone on both pose and SSIM.
- `mvdinov2_frozen` (15.24 mm) is *worse* than `dinov2_frozen` (14.65 mm), and its ΔSSIM
  drops from +0.0215 to +0.0122 — multi-view does not help DINOv2 here.

---

## Scope

- Translation only — rotation jitter is identically zero in the current data.
- 5–27 test samples per object/variant; an object only yields samples from the scenes it
  appears in.
- One seed per variant, no repeats.
- 7 of 27 possible variants run (9 backbones in `MODEL_SPECS` × 3 tags). dinov3 is the only
  backbone with both a frozen and a fine-tuned run; both multi-view runs are frozen-only.
- Masked SSIM is masked to non-black pixels (threshold 5/255), 6 tiles per sample.

*Source: `vit_hope/results_comparison.ipynb` — 28 objects × 7 variants, 3171 samples per
metric family, complete grid.*
