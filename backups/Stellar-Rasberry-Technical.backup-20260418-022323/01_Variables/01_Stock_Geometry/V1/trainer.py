"""
01_Stock_Geometry/V1/trainer.py
================================
ConvNeXt-Base trainer for V1 Stock Geometry. 5-resolution.
 
Architecture:
  Backbone: ConvNeXt-Base (heavier than MobileNetV3 — justified by 500+ ticker diversity)
  Input: 224×224 (resized from 680×680 rendered charts)
  Output: 3-class probability [P(bearish), P(neutral), P(strongly_bullish)]
  Loss: Focal Loss γ=2.0 with Label Smoothing ε=0.1
  Augmentation: Mixup α=0.3
  Calibration: Guo et al. (2017) post-hoc temperature scaling on val set
  Init: zero-gamma on final LayerNorm of each CNBlock (training stability)
 
Data Pipeline:
  Reads from 01_Stock_Geometry/V1/{resolution}/images/{TICKER}/{date}.png
  Labels from --label_csv with columns: date, ticker, label, split
  No yfinance. No network calls. Pure local I/O.
 
Hardware: M3 Ultra 96GB
  MPS backend, pin_memory=True, num_workers=8
  Batch size 256 fits ~12GB VRAM on MPS
 
Usage:
  python trainer.py --resolution 15min --data_dir 15min/images --label_csv labels.csv
  python trainer.py --resolution daily --data_dir daily/images --label_csv labels.csv
"""
 
import os, logging, argparse
from pathlib import Path
import numpy as np, pandas as pd
 
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
    from PIL import Image
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
 
import importlib as _il
B = _il.import_module("01_base_modules_")
 
IMG_SZ  = 224
NC      = 3
BS      = 256
NW      = 8
LR      = 1e-4
WD      = 1e-4
FOCAL_G = 2.0
LABEL_S = 0.1
MIXUP_A = 0.3
 
 
class FocalLoss(nn.Module):
    """
    Lin et al. (2017) Focal Loss with label smoothing.
    γ=2.0 down-weights easy (neutral) samples, focusing on
    the bull/bear boundary where classification matters most.
    """
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none",
                             label_smoothing=LABEL_S)
        pt = torch.exp(-ce)
        return ((1 - pt) ** FOCAL_G * ce).mean()
 
 
class ChartDataset(Dataset):
    """
    Reads pre-rendered chart images from local disk.
    No network calls. Path resolution uses glob-cached set for speed.
    """
    def __init__(self, img_dir: str, label_df: pd.DataFrame,
                 split: str, resolution: str):
        self.dir = Path(img_dir)
        self.data = label_df[label_df["split"] == split].reset_index(drop=True)
        self.res = resolution
        self.existing = B.build_image_set(str(self.dir))
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SZ, IMG_SZ)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        n_found = sum(1 for _, r in self.data.iterrows()
                      if self._path(r) in self.existing)
        logging.info(f"  ChartDataset [{split}]: {len(self.data):,} rows, "
                     f"{n_found:,} images on disk")
 
    def _path(self, row) -> str:
        tk = row["ticker"]
        if self.res in ("15min", "1hour"):
            fname = f"{B.fmt_15m(pd.Timestamp(row['date']))}.png"
        else:
            fname = f"{row['date']}.png"
        return str(self.dir / tk / fname)
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = self._path(row)
        if path in self.existing:
            img = Image.open(path).convert("RGB")
        else:
            img = Image.new("RGB", (IMG_SZ, IMG_SZ), (0, 0, 0))
        return self.transform(img), int(row.get("label", 1))
 
 
def build_convnext(device: str) -> nn.Module:
    """
    ConvNeXt-Base with zero-gamma init on final LayerNorm.
    This makes the residual blocks act as identity at init,
    improving training stability for fine-tuning.
    """
    model = models.convnext_base(
        weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features, NC)
    for mod in model.modules():
        if isinstance(mod, models.convnext.CNBlock):
            if hasattr(mod, "layer_scale"):
                nn.init.zeros_(mod.layer_scale)
    return model.to(device)
 
 
class TemperatureScaler(nn.Module):
    """Guo et al. (2017): post-hoc temperature scaling for calibration."""
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1) * 1.5)
 
    def forward(self, logits):
        return logits / self.T
 
 
def calibrate_temperature(model, val_loader, device) -> float:
    """Find optimal temperature on validation set via LBFGS."""
    scaler = TemperatureScaler().to(device)
    optimizer = torch.optim.LBFGS([scaler.T], lr=0.01, max_iter=50)
 
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            all_logits.append(model(x.to(device)))
            all_labels.append(y.to(device))
    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)
 
    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(scaler(logits_cat), labels_cat)
        loss.backward()
        return loss
 
    optimizer.step(closure)
    return scaler.T.item()
 
 
def train_model(args):
    if not HAS_TORCH:
        logging.error("PyTorch required. pip install torch torchvision")
        return
 
    device = args.device
    label_df = pd.read_csv(args.label_csv, parse_dates=["date"])
    label_df["date"] = label_df["date"].dt.strftime("%Y-%m-%d")
 
    train_ds = ChartDataset(args.data_dir, label_df, "train", args.resolution)
    val_ds   = ChartDataset(args.data_dir, label_df, "val",   args.resolution)
 
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True,
                              num_workers=NW, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BS, shuffle=False,
                              num_workers=NW, pin_memory=True)
 
    model = build_convnext(device)
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
 
    best_val_acc = 0.0
    best_state = None
 
    logging.info(f"V1 Trainer | {args.resolution} | ConvNeXt-Base | "
                 f"Focal γ={FOCAL_G} Mixup α={MIXUP_A}")
 
    for epoch in range(args.epochs):
        model.train()
        correct = total = 0
 
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
 
            lam = np.random.beta(MIXUP_A, MIXUP_A)
            perm = torch.randperm(x.size(0), device=device)
            x_mixed = lam * x + (1 - lam) * x[perm]
 
            logits = model(x_mixed)
            loss = (lam * criterion(logits, y) +
                    (1 - lam) * criterion(logits, y[perm]))
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
 
        scheduler.step()
 
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                preds = model(x.to(device)).argmax(1)
                val_correct += (preds == y.to(device)).sum().item()
                val_total += y.size(0)
 
        val_acc = val_correct / max(val_total, 1)
        if (epoch + 1) % 5 == 0:
            logging.info(f"  Ep {epoch+1}/{args.epochs} "
                         f"Train={correct/total:.3f} Val={val_acc:.3f}")
 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
 
    model.load_state_dict(best_state)
    model.to(device)
    opt_temp = calibrate_temperature(model, val_loader, device)
 
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"model_{args.resolution}.pth"
 
    torch.save({
        "model_state_dict": best_state,
        "optimal_temperature": opt_temp,
        "best_val_acc": best_val_acc,
        "backbone": "convnext_base",
        "render_size": B.PX,
        "resolution": args.resolution,
        "focal_gamma": FOCAL_G,
        "label_smoothing": LABEL_S,
        "mixup_alpha": MIXUP_A,
    }, save_path)
 
    logging.info(f"✓ {save_path} | Val={best_val_acc:.3f} T={opt_temp:.3f}")
 
 
def main():
    p = argparse.ArgumentParser(
        description="V1 Stock Geometry — ConvNeXt-Base Trainer")
    p.add_argument("--resolution", choices=B.RESOLUTIONS, required=True)
    p.add_argument("--data_dir", required=True,
                   help="Path to images/ directory for this resolution")
    p.add_argument("--label_csv", required=True,
                   help="CSV with columns: date, ticker, label, split")
    p.add_argument("--output_dir", default=".",
                   help="Directory to save model checkpoint")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", default="mps")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    train_model(args)
 
 
if __name__ == "__main__":
    main()