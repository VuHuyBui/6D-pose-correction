# New models

Two methods that keep the 6 projections **separate** instead of stacking them
into one 18-channel image. Both are implemented in `models_mv.py` and registered
in `MODEL_SPECS`; see the README's model table for tile sizes and parameter
counts.

The input contract is unchanged — `data.py` still yields `(18, T, T)` and the
reshape to 6 × `(3, T, T)` happens inside the model. That is exact, because
`StackDataset._load_stack` concatenates the tiles view-major.

### 1st model -- From scratch  →  `--model mvit`

Use a Multi-view Transformer. Local transformer for each image. Then concatenate
the results of the locals and use a global transformer across all views.

Implemented as `MultiViewViT`: one *shared* local ViT trunk applied to all 6
views (same object, same renderer — what differs is which view a feature came
from, and the fuser's view embedding carries that), then `GlobalFuser`.

### 2nd model -- Pretrained  →  `--model mvdinov2 | mvdinov3 | mvswinv2`

Pipeline:

6 projection images --> shared feature extractor (DINOv2/v3 and SwinV2) -->
concatenate into a global patch --> use a global transformer

Implemented as `MultiViewTimm`. "Concatenate into a global patch" is realised as
`--token_grid` tokens per view (default 2 → 4 tokens/view, 24 tokens total),
average-pooled from the backbone's patch grid. Concatenating raw patch tokens is
not feasible at this resolution: DINOv2 @ 518 would be 6 × 1369 = 8214 tokens.

Fine-tuning is the existing `--no_freeze` flag; `--freeze` (default) trains only
the fuser and head.

For SwinV2, the extra small-detail test is `--model mvswinlo` (`MVSwinV2Low`):
`features_only` taps stages 0–1 instead of the final stage, so features are 4×/8×
reduced rather than 32× and keep the fine structure a millimetre-scale offset
lives in. Both stages are pooled to the same grid and their tokens concatenated,
so the fuser weighs the two scales itself.

## Status

Wired and smoke-tested (`train.py --overfit_test` converges for every key). No
full training sweep has been run yet, so `RESULTS.md` does not include them.

Before aggregating a sweep: `results_comparison.ipynb` asserts an **8-slot**
colour palette, and 5 existing + up to 9 new variants exceeds it. Widen the
palette there first.
