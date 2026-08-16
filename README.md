# MTGF-SAM

**MTGF-SAM** is a terrain-aware landslide segmentation framework built on top of
Meta's Segment Anything Model 3 (SAM3). The model is designed for landslide
detection in remote sensing imagery by combining RGB visual features with
DEM-derived terrain information through multi-level terrain-gated fusion.

This cleaned public-release package focuses on the Jiuzhaigou landslide dataset.
It keeps only the model code, the core MTGF modules, the Jiuzhaigou dataset, and
a concise training pipeline aligned with the experimental settings described in
the paper.

## Model Overview

MTGF-SAM adapts SAM3 for geohazard segmentation by injecting terrain information
into the segmentation pipeline. The main idea is to preserve SAM3's strong
pre-trained visual prior while allowing the network to use topographic
constraints that are important for landslide recognition.

The framework contains the following key components:

- **SAM3 image backbone**: provides strong visual representation and text-prompt
  grounding for the target concept, here using the prompt `landslide`.
- **Terrain input branch**: reads the DEM associated with each RGB tile. In this
  release, the Jiuzhaigou dataset uses the raw single-channel DEM directly.
- **DEM encoder**: extracts terrain features from DEM input.
- **Cross-attention terrain-visual fusion**: aligns DEM features with visual
  features at multiple feature levels.
- **Terrain-aware prompt fusion**: injects global terrain context into prompt
  features.
- **Decoder-level DEM fusion**: refines the final mask prediction using terrain
  features.

## Datasets

### Included Dataset: JiuzhaigouDataset

This release includes the self-built **Jiuzhaigou landslide dataset** used in
the MTGF-SAM paper.

The dataset folder is:

```text
datasets/JiuzhaigouDataset/
```

It contains:

- `images/`: RGB remote sensing image tiles
- `dem/`: raw single-channel DEM tiles
- `masks/`: binary landslide masks
- `annotations.json`: COCO-style polygon annotations
- `combined_summary.json`: dataset summary information
- `README.md` and `README_zh.txt`: dataset notes

### Public Dataset: Bijie Landslide Dataset

The Bijie landslide dataset is a public remote sensing landslide dataset used as
one of the comparative datasets in the paper. It is **not bundled** in this
cleaned release.

Download / project page:

- [Bijie Landslide Dataset](https://gpcv.whu.edu.cn/data/Bijie_pages.html)

### Public Dataset: Landslide4Sense Dataset

Landslide4Sense is a public multi-source landslide benchmark dataset. It is used
for comparative evaluation in the paper, but it is **not bundled** in this
cleaned Jiuzhaigou release.

Download / project pages:

- [Landslide4Sense official page](https://www.iarai.ac.at/landslide4sense)
- [Landslide4Sense-2022 GitHub repository](https://github.com/iarai/Landslide4Sense-2022)

## Installation & Dependencies

### 1. Create an environment

Python 3.10 or newer is recommended. The original experiments were run with a
CUDA-enabled PyTorch environment.

```bash
conda create -n mtgf-sam python=3.10 -y
conda activate mtgf-sam
```

### 2. Install PyTorch

Install PyTorch and torchvision according to your CUDA version. For example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If you use a different CUDA version, follow the official PyTorch installation
selector.

### 3. Install the remaining packages

```bash
pip install -r requirements.txt
```

Main dependencies include:

- `torch` and `torchvision`
- `numpy`
- `scipy`
- `Pillow`
- `tqdm`
- `huggingface_hub`
- `iopath`
- `hydra-core`
- `omegaconf`
- `h5py` for Landslide4Sense-style `.h5` data handling if you extend this
  cleaned release to the L4S dataset

## SAM3 Base Weights

The cleaned release does not include the original SAM3 checkpoint. Download the
SAM3 base weights according to the official SAM3 instructions and place the
checkpoint at:

```text
sam3_weights/facebook/sam3/sam3.pt
```

Official SAM3 repository:

- [facebookresearch/sam3](https://github.com/facebookresearch/sam3)

You may also pass a custom checkpoint path when training:

```bash
python src/train/train_jiuzhaigou.py --sam3-checkpoint /path/to/sam3.pt
```

## Training

The Jiuzhaigou training entry point is:

```text
src/train/train_jiuzhaigou.py
```

Default training settings:

- Optimizer: AdamW
- Weight decay: `1e-4`
- Pre-trained encoder learning rate: `1e-4`
- Decoder / newly initialized module learning rate: `5e-4`
- Epochs: `30`
- Effective batch size: `8`
- Gradient clipping: `1.0`
- Learning-rate scheduler: `ReduceLROnPlateau`
- Scheduler factor: `0.5`
- Scheduler patience: `2` epochs without validation IoU improvement
- Augmentation: random horizontal flip and random vertical flip only
- RGB normalization: ImageNet mean and standard deviation
- Model selection: best checkpoint is saved according to validation IoU
- Terrain input: raw single-channel DEM

Run training:

```bash
python src/train/train_jiuzhaigou.py
```

The default script uses `batch_size=1` and `accum_steps=8`, which gives an
effective batch size of 8 while reducing GPU memory pressure:

```bash
python src/train/train_jiuzhaigou.py --batch-size 1 --accum-steps 8
```

If your GPU has enough memory, you can increase the physical batch size and
reduce accumulation accordingly:

```bash
python src/train/train_jiuzhaigou.py --batch-size 2 --accum-steps 4
```

Outputs are saved by default to:

```text
checkpoints/jiuzhaigou_raw_dem/
```

The folder will contain:

- `best_model.pt`
- `train_log.csv`
- `final_result.json`

## Evaluation Metrics

The evaluation utility is:

```text
src/eval/eval_metrics.py
```

It reports binary landslide segmentation metrics:

- Overall Accuracy (OA)
- Precision
- Recall
- F1-score
- Intersection over Union (IoU)
- TP / FP / FN / TN

The training script evaluates the validation split after each epoch and evaluates
both validation and test splits after loading the best checkpoint.
