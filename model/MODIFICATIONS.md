# Modifications to SAM3

This directory is a **modified fork of Meta's SAM3**, carrying the MTGF-SAM terrain
fusion modules. It remains under the **SAM License** (see `LICENSE`); Meta's original
copyright headers are retained in every file.

- Upstream project: https://github.com/facebookresearch/sam3
- Fork point: upstream commit `2530a8a` ("Drop per-session dict refs in close_session")
- Complete diff: [`mtgf_sam_vs_upstream.patch`](mtgf_sam_vs_upstream.patch)

Two upstream directories were removed from this release copy because they carry large
demo media and are unrelated to landslide segmentation: `assets/` (55 MB of gifs and
videos) and `examples/` (Jupyter demo notebooks), plus `scripts/eval/` (upstream
benchmark evaluation data). Nothing under `sam3/` was removed.

## Summary of changes

`1297 insertions, 20 deletions` across 13 files:

| file | + | what changed |
|---|---|---|
| `sam3/model/dem_fusion.py` | 648 | **new file** — the four MTGF fusion modules |
| `sam3/model_builder.py` | 161 | `enable_dem*` construction flags; wires the modules in |
| `sam3/model/sam3_image.py` | 102 | DEM tensor threaded through the forward pass |
| `sam3/model/maskformer_segmentation.py` | 99 | decoder-level terrain fusion + optional deep seg head |
| `sam3/model/vl_combiner.py` | 92 | prompt-level terrain/RGB conditioning of the text tokens |
| `sam3/train/data/sam3_image_dataset.py` | 51 | DEM channel loading in the training dataset |
| `sam3/model/sam3_image_processor.py` | 33 | DEM-aware preprocessing |
| `sam3/train/transforms/basic_for_api.py` | 24 | DEM-consistent geometric augmentation |
| `sam3/train/data/collator.py` | 11 | DEM batching |
| `sam3/perflib/fused.py` | 10 | grad-enabled path in `addmm_act` (needed for fine-tuning) |
| `sam3/model/data_misc.py` | 1 | export fix |
| `sam3/train/configs/roboflow_v100/roboflow_v100_landslide_dem.yaml` | 72 | landslide+DEM training config |
| `sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml` | 13 | full fine-tune config |

## The new module file

`sam3/model/dem_fusion.py` defines, in the order they act on the network:

| class | fusion point |
|---|---|
| `DEMEncoder` | builds the terrain feature pyramid from the 7-channel descriptor |
| `DEMCrossAttentionFusion` | encoder level — RGB queries attend over terrain keys/values |
| `DEMPromptFusion` | prompt level — pooled terrain + RGB conditions the text tokens |
| `DEMDecoderFusion` | decoder level — terrain enters the high-resolution pixel embedding |

`_SERefine`, the gated squeeze-excitation residual behind the `enable_dem_v3_refine`
flag, is used by the reported Landslide4Sense model (and switched off for Bijie and
Jiuzhaigou).

The file also contains `DEMSpatialPromptTokens`, `TextAdapter`,
`DecoupledGatedPromptFusion`, `VisualPromptAdapter`, `VerificationHead` and
`CrossModalAlignHead` — these were explored during development and are **not** enabled
in any reported model (no reported config sets `enable_dem_spatial_prompt`). They are
kept because the checkpoints and trainers reference the module namespace.

See `docs/ARCHITECTURE.md` in the repository root for the design rationale.

## Design invariant

Every fusion point enters through a **zero-initialised `tanh` gate**, so a freshly
constructed MTGF model is numerically identical to the RGB-only SAM3 it wraps. This
is what makes the staged warm-start chain safe and is worth preserving in any
further modification.

## Non-DEM change worth knowing about

`sam3/perflib/fused.py` — upstream's `addmm_act` takes a no-grad fast path that breaks
backpropagation. The fork adds a grad-enabled branch so the model can be fine-tuned at
all. This is independent of the DEM work and applies to anyone fine-tuning SAM3.
