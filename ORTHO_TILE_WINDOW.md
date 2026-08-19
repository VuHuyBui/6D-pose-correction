# The orthographic tile window

Why the renderer's ortho window is a hardcoded **0.28 m**, and why every tile
looks the way it does. Written down so the constant is not "tidied up" later.

## The bug it fixes

`texture-projection-opengl-cpp/src/new_projection.cpp` used to hardcode

```cpp
float orthoWidth  = 0.20f;
float orthoHeight = orthoWidth * heightCam / widthCam;   // 0.20 * 1080/1920
```

HOPE's `camera.json` is 1920×1080, so that made the window **200 × 112.5 mm**,
centred on the object. The *width* was fine — 0.20 m clears 27 of the 28 HOPE
objects. The *height* was not: 112.5 mm is shorter than 13 of the 28 objects, so
whenever an object's long axis landed on the vertical ortho axis, it was cut off.

Measured on `obj_000006` (167.2 mm long) before the fix:

| tile | % blue | clipped |
|------|--------|---------|
| `w=0,h=0` | 9.5% | 0/50 |
| `w=0,h=1` | 97.5% | 0/50 |
| `w=0,h=2` | 64.7% | **50/50** |
| `w=1,h=0` | 54.2% | **50/50** |
| `w=1,h=1` | 59.8% | 0/50 |
| `w=1,h=2` | 59.6% | 0/50 |

**100 of 300 tiles** had the silhouette running off the frame border. For obj 25
(249.9 mm) it was worse: only the middle ~45% of the box was ever in frame.

## What it is now

```cpp
static constexpr float ORTHO_EXTENT_M = 0.28f;   // vertical extent, metres
float orthoHeight = ORTHO_EXTENT_M;
float orthoWidth  = orthoHeight * widthCam / heightCam;
```

Two properties worth keeping:

- **One window for all 6 views of all 28 objects.** A pixel is 1/3.857 mm in
  every tile, so tiles are directly comparable across objects and views.
- **The aspect ratio stays tied to the viewport**, which keeps the render
  isotropic (3857 px/m on *both* axes). Making the window square while the
  viewport stays 16:9 would stretch the object horizontally — don't.

The horizontal window is therefore 0.498 m, mostly empty. That costs nothing:
`data.py:_compute_shared_crop_box` crops every tile to the object's bounding box
before resizing to `T×T`.

0.28 m was chosen to clear the largest HOPE object with ~12% headroom:

| extent | px/m | objects clipped | smallest object (68 mm) fills |
|---|---|---|---|
| 0.30 m | 3600 | 0 | 22.7% of tile height |
| **0.28 m** | **3857** | **0** | **24.3% (262 px)** |
| 0.22 m | 4909 | 1 (obj 25) | 30.9% |
| 0.20 m (old width) | 5400 | 1 (obj 25) | 34.0% |

Largest objects: obj 25 = 249.9 mm, obj 17 = 192.5 mm, obj 14 = 190.4 mm.
Smallest: obj 28 = 68.0 mm.

Both call sites must use the same value or the interactive view and the saved
PNGs disagree — `renderOrthographicSides()` and `saveResult()`.

### The trade-off

Shared intrinsics cost resolution on small objects. The vertical scale dropped
from 9600 px/m to 3857 px/m, so obj 28 now renders 262 px tall instead of 653 px
and `data.py` *upsamples* it to 518 rather than downsampling. If that measurably
hurts a small object's accuracy, the mitigation is to render the ortho tiles at a
fixed higher resolution instead of at `outW`/`outH`, which currently just inherit
`intrinsics.json`. Going back to a per-object window would also work but would
give up the shared scale.

## The blue regions are not a bug

Large parts of most tiles are navy `(0, 0, 128)`. That is deliberate:
`shaders/frag_shader_texture_projection.glsl:57` paints any surface with
`ndotl < 0.1` (facing away from the camera) navy, and `:74` does the same for
anything the projector's shadow map says is occluded. Navy means **the camera
never saw this surface**.

A single RGB frame only observes one side of an object, so this is unavoidable.
On `obj_000006` the camera-facing tile is 9.5% blue while the opposite tile is
97.5% blue.

HOPE offers no way around it: the 5 images in each `val` scene are **lighting
variations at an identical camera pose**, not extra viewpoints —
`cam_R_m2c`/`cam_t_m2c` are bit-identical across all 5 images for **184/184**
instances. Filling the unobserved surface would need either the meshes' own baked
UV textures (`models/obj_0000NN.png`, which HOPE does ship) or fusion across
different scenes. Both were considered and rejected.

## Consequences for the pipeline

- **Any render made before this change is stale.** Regenerate with
  `FORCE=1 ./bash/generate_jitter_all.sh`, then retrain — checkpoints and predictions
  from the old tiles are not comparable to new ones.
- **`data.py` needs no change.** The shared crop box is computed from `rows[0]`
  and reused for every sample, which stays valid: the mesh is drawn at
  `model = identity` with fixed ortho cameras, so the jitter moves the *projector*
  and never the silhouette. Verified — the silhouette is bit-identical across
  `run_0`/`run_1`/`run_2`.
- Per-object bbox cropping means the shared scale holds in the rendered PNGs but
  not in the final `(18,T,T)` tensor, where each object is normalised to fill the
  frame. That was a deliberate choice.

## Re-checking it

```bash
cd texture-projection-opengl-cpp && cmake --build build
cd ../vit_hope
DEST_ROOT=/tmp/verify OBJ_IDS="25 6" LIMIT=4 JITTERS_PER_IMAGE=1 \
  FORCE=1 ./bash/generate_jitter_all.sh
```

```python
import glob, re, json, numpy as np
from PIL import Image
mi = json.load(open('.../hope/models/models_info.json'))
bad, ext = 0, {}
for f in sorted(glob.glob('/tmp/verify/obj_*/run_*/*_tile-w=*.png')):
    obj = int(re.search(r'obj_(\d+)', f).group(1))
    a = np.asarray(Image.open(f).convert('RGB')).sum(2) > 15
    if a[0,:].any() or a[-1,:].any() or a[:,0].any() or a[:,-1].any(): bad += 1
    r, c = np.where(a.any(1))[0], np.where(a.any(0))[0]
    if r.size: ext.setdefault(obj, []).append(max(r[-1]-r[0]+1, c[-1]-c[0]+1))
print('clipped tiles:', bad)                      # expect 0
for o, v in sorted(ext.items()):                  # expect 3.857 px/mm for all
    m = mi[str(o)]
    print(o, max(v) / max(m['size_x'], m['size_y'], m['size_z']))
```

Last run: **0/48 clipped**, obj 6 at 3.857 px/mm and obj 25 at 3.858 px/mm
(0.03% spread), matching the predicted 1080 px / 280 mm = 3.857.
