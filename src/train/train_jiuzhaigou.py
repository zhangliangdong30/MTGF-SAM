from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm


ROOT = Path(os.environ.get("MTGF_ROOT", Path(__file__).resolve().parents[2]))
for extra in (ROOT / "sam3_dem", ROOT / "src" / "eval"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from sam3.model_builder import build_sam3_image_model  # noqa: E402
from sam3.model.data_misc import FindStage  # noqa: E402
from eval_metrics import calculate_metrics  # noqa: E402


@dataclass
class Sample:
    stem: str
    image: Path
    dem: Path
    mask: Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class JiuzhaigouDataset(Dataset):
    def __init__(self, root: Path, split: str, resolution: int = 1008, train: bool = False):
        self.root = root
        self.resolution = resolution
        self.train = train
        self.image_dir = root / "datasets" / "JiuzhaigouDataset" / "images"
        self.dem_dir = root / "datasets" / "JiuzhaigouDataset" / "dem"
        self.mask_dir = root / "datasets" / "JiuzhaigouDataset" / "masks"
        self.samples = self._load_split(split)
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def _load_split(self, split: str):
        split_file = self.root / "datasets" / f"jiuzhaigou_split_{split}.txt"
        samples = []
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                _, fname = line.strip().split(":", 1)
                stem = Path(fname).stem
                samples.append(
                    Sample(
                        stem=stem,
                        image=self.image_dir / fname,
                        dem=self.dem_dir / f"{stem}.tif",
                        mask=self.mask_dir / f"{stem}.png",
                    )
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _read_dem(self, path: Path) -> torch.Tensor:
        dem = np.asarray(Image.open(path).convert("F"), dtype=np.float32)
        lo, hi = float(dem.min()), float(dem.max())
        if hi - lo < 1e-6:
            dem = np.zeros_like(dem, dtype=np.float32)
        else:
            dem = (dem - lo) / (hi - lo)
        return torch.from_numpy(dem).unsqueeze(0)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        rgb = TF.to_tensor(Image.open(sample.image).convert("RGB"))
        dem = self._read_dem(sample.dem)
        mask = torch.from_numpy(np.asarray(Image.open(sample.mask).convert("L"), dtype=np.float32) / 255.0).unsqueeze(0)

        if self.train:
            if random.random() < 0.5:
                rgb, dem, mask = TF.hflip(rgb), TF.hflip(dem), TF.hflip(mask)
            if random.random() < 0.5:
                rgb, dem, mask = TF.vflip(rgb), TF.vflip(dem), TF.vflip(mask)

        if rgb.shape[-1] != self.resolution:
            rgb = F.interpolate(rgb.unsqueeze(0), size=(self.resolution, self.resolution), mode="bilinear", align_corners=False).squeeze(0)
            dem = F.interpolate(dem.unsqueeze(0), size=(self.resolution, self.resolution), mode="bilinear", align_corners=False).squeeze(0)
            mask = F.interpolate(mask.unsqueeze(0), size=(self.resolution, self.resolution), mode="nearest").squeeze(0)

        rgb = (rgb - self.rgb_mean) / self.rgb_std
        return rgb.float(), dem.float(), mask.float(), sample.stem


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    inter = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    return 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()


def segmentation_logits(outputs: dict, target_hw: tuple[int, int]) -> torch.Tensor:
    logits = outputs.get("semantic_seg")
    if logits is None:
        logits = outputs["pred_masks"][:, :1]
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    if logits.shape[-2:] != target_hw:
        logits = F.interpolate(logits.float(), size=target_hw, mode="bilinear", align_corners=False)
    return logits


def build_model(args):
    model = build_sam3_image_model(
        device=args.device,
        eval_mode=False,
        load_from_HF=False,
        checkpoint_path=args.sam3_checkpoint,
        enable_dem=True,
        dem_in_channels=1,
        dem_fuse_all_levels=True,
        enable_dem_prompt_fusion=True,
        enable_dem_prompt_rgb=True,
        enable_dem_decoder_fusion=True,
        enable_dem_v3_refine=False,
        enable_deep_seg_head=True,
    )
    encoder_params, other_params = [], []
    for name, param in model.named_parameters():
        param.requires_grad_(True)
        if "vision_backbone" in name:
            encoder_params.append(param)
        else:
            other_params.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.lr_encoder},
            {"params": other_params, "lr": args.lr_decoder},
        ],
        weight_decay=1e-4,
    )
    return model, optimizer


def forward_one(model, text_out, find_stage, rgb, dem):
    bb = model.backbone.forward_image(rgb, dem=dem)
    bb.update(text_out)
    return model.forward_grounding(
        backbone_out=bb,
        find_input=find_stage,
        geometric_prompt=model._get_dummy_prompt(),
        find_target=None,
    )


@torch.no_grad()
def evaluate(model, text_out, find_stage, loader, device):
    model.eval()
    totals = []
    for rgb, dem, mask, _stem in tqdm(loader, desc="eval", leave=False):
        rgb = rgb.to(device)
        dem = dem.to(device)
        mask = mask.to(device)
        with torch.autocast(device_type=device if device != "cpu" else "cpu", dtype=torch.bfloat16, enabled=device != "cpu"):
            out = forward_one(model, text_out, find_stage, rgb, dem)
        logits = segmentation_logits(out, mask.shape[-2:])
        pred = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
        gt = mask.cpu().numpy()
        totals.append(calculate_metrics(gt, pred))
    if not totals:
        return {"OA": 0.0, "F1": 0.0, "IoU": 0.0}
    return {k: float(np.mean([t[k] for t in totals if k in t])) if k not in {"TP", "FP", "FN", "TN"} else int(np.sum([t[k] for t in totals if k in t])) for k in totals[0].keys()}


def train(args):
    set_seed(args.seed)
    train_ds = JiuzhaigouDataset(ROOT, "train", resolution=args.resolution, train=True)
    val_ds = JiuzhaigouDataset(ROOT, "val", resolution=args.resolution, train=False)
    test_ds = JiuzhaigouDataset(ROOT, "test", resolution=args.resolution, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model, optimizer = build_model(args)
    model.to(args.device)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device == "cuda")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    with torch.no_grad():
        text_out = model.backbone.forward_text(["landslide"], device=args.device)
    find_stage = FindStage(
        img_ids=torch.tensor([0], device=args.device, dtype=torch.long),
        text_ids=torch.tensor([0], device=args.device, dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best_model.pt"
    best_iou = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, (rgb, dem, mask, _stem) in tqdm(enumerate(train_loader, start=1), total=len(train_loader), desc=f"epoch {epoch:02d}/{args.epochs}"):
            rgb = rgb.to(args.device)
            dem = dem.to(args.device)
            mask = mask.to(args.device)
            with torch.autocast(device_type=args.device if args.device != "cpu" else "cpu", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                out = forward_one(model, text_out, find_stage, rgb, dem)
                logits = segmentation_logits(out, mask.shape[-2:])
                loss = F.binary_cross_entropy_with_logits(logits, mask) + dice_loss(logits, mask)
                presence = out.get("presence_logit_dec")
                if presence is not None:
                    loss = loss + 0.1 * F.binary_cross_entropy_with_logits(presence.float(), torch.ones_like(presence, dtype=torch.float32))
                loss = loss / args.accum_steps
            scaler.scale(loss).backward()
            running += float(loss.item()) * args.accum_steps
            if step % args.accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        val_metrics = evaluate(model, text_out, find_stage, val_loader, args.device)
        scheduler.step(val_metrics.get("IoU", 0.0))
        if val_metrics.get("IoU", 0.0) > best_iou:
            best_iou = val_metrics["IoU"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_iou": best_iou}, best_path)

    ckpt = torch.load(best_path, map_location=args.device)
    model.load_state_dict(ckpt["model"], strict=True)
    val_metrics = evaluate(model, text_out, find_stage, val_loader, args.device)
    test_metrics = evaluate(model, text_out, find_stage, test_loader, args.device)
    with open(save_dir / "final_result.json", "w", encoding="utf-8") as f:
        json.dump({"best_iou": best_iou, "val": val_metrics, "test": test_metrics}, f, indent=2)
    print(json.dumps({"best_iou": best_iou, "val": val_metrics, "test": test_metrics}, indent=2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sam3-checkpoint", default=str(ROOT / "sam3_weights" / "facebook" / "sam3" / "sam3.pt"))
    p.add_argument("--save-dir", default=str(ROOT / "checkpoints" / "jiuzhaigou_raw_dem_no_teacher"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accum-steps", type=int, default=8)
    p.add_argument("--resolution", type=int, default=1008)
    p.add_argument("--lr-encoder", type=float, default=1e-4)
    p.add_argument("--lr-decoder", type=float, default=5e-4)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

