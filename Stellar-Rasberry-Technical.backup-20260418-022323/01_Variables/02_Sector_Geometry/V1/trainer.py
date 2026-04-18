"""
02_Sector_Geometry/V1/trainer.py
==================================
MobileNetV3-Large trainer for V2 Sector Geometry. 5-resolution.
 
Architecture:
  Backbone: MobileNetV3-Large (lighter than ConvNeXt — only 11 ETFs,
            less visual diversity than 500+ tickers in V1)
  Input: 224×224 (resized from 680×680 3-pane charts)
  Output: 3-class [P(bearish), P(neutral), P(strongly_bullish)]
  Loss: Focal γ=2.0, Label Smoothing ε=0.1
  Augmentation: Mixup α=0.3
  Calibration: Guo et al. (2017) temperature scaling
  Init: zero-gamma on final BatchNorm in last InvertedResidual block
 
Self-Reference Protocol (Zero-Bias, Non-Negotiable):
  check_self_reference(ticker, etf) → True if ticker weight > 5% in ETF
  When True: per-sample loss mask = 0.0 (gradient does not flow)
  When False: per-sample loss mask = 1.0 (normal training)
 
  This prevents AAPL (22% of XLK) from learning its own chart pattern
  via the XLK sector image. The mask is applied IN THE LOSS FUNCTION,
  not by zeroing features — the CNN still processes the image, but
  backprop is blocked for self-referencing samples.
 
  Heavy-weight constituents (>5%) defined in 01_base_modules_.HEAVY_WEIGHTS.
 
GICS Reclassification:
  The label_csv must include a v2_sector_etf column mapping each
  (ticker, date) to the correct ETF at that point in time:
    - Before 2018-09-24: GOOGL/META → XLK, DIS/NFLX → XLY (not XLC)
    - Before 2015-10-07: AMT/PLD → XLF (not XLRE)
  The get_sector_etf() function in 01_base_modules_ handles this.
 
Data Pipeline:
  Reads from 02_Sector_Geometry/V1/{resolution}/images/{ETF}/{date}.png
  No yfinance. No network. Pure local I/O.
 
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
 
 
class FocalLossWithMask(nn.Module):
    """
    Focal Loss with per-sample mask for self-reference enforcement.
    mask=0.0 blocks gradient for self-referencing samples.
    mask=1.0 for clean samples.
    """
    def forward(self, logits, targets, mask=None):
        ce = F.cross_entropy(logits, targets, reduction="none",
                             label_smoothing=LABEL_S)
        focal = (1 - torch.exp(-ce)) ** FOCAL_G * ce
        if mask is not None:
            focal = focal * mask
        return focal.mean()
 
 
class SectorDataset(Dataset):
    """
    Reads sector ETF chart images. Returns (image, label, self_ref_mask).
    self_ref_mask = 0.0 if ticker weight > 5% in its sector ETF.
    """
    def __init__(self, img_dir: str, label_df: pd.DataFrame,
                 split: str, resolution: str):
        self.dir = Path(img_dir)
        self.data = label_df[label_df["split"] == split].reset_index(drop=True)
        self.res = resolution
        self.existing = B.build_image_set(str(self.dir))
 
        if "ticker" in self.data.columns and "v2_sector_etf" in self.data.columns:
            self.data["_self_ref"] = self.data.apply(
                lambda r: B.check_self_reference(r["ticker"], r["v2_sector_etf"]),
                axis=1)
        else:
            self.data["_self_ref"] = False
 
        n_sr = self.data["_self_ref"].sum()
        logging.info(f"  SectorDataset [{split}]: {len(self.data):,} rows, "
                     f"self-ref masked: {n_sr:,}")
 
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SZ, IMG_SZ)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
 
    def _path(self, row) -> str:
        etf = row.get("v2_sector_etf", "SPY")
        if self.res in ("15min", "1hour"):
            fname = f"{B.fmt_15m(pd.Timestamp(row['date']))}.png"
        else:
            fname = f"{row['date']}.png"
        return str(self.dir / etf / fname)
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = self._path(row)
        if path in self.existing:
            img = Image.open(path).convert("RGB")
        else:
            img = Image.new("RGB", (IMG_SZ, IMG_SZ), (0, 0, 0))
 
        label = int(row.get("label", 1))
        mask = 0.0 if row["_self_ref"] else 1.0
        return self.transform(img), label, mask
 
 
def build_mobilenet(device: str) -> nn.Module:
    """
    MobileNetV3-Large with zero-gamma init on final BN.
    """
    model = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features, NC)
    last_block = model.features[-1]
    for mod in last_block.modules():
        if isinstance(mod, nn.BatchNorm2d) and mod.weight is not None:
            nn.init.zeros_(mod.weight)
    return model.to(device)
 
 
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1) * 1.5)
    def forward(self, logits):
        return logits / self.T
 
 
def calibrate_temperature(model, val_loader, device) -> float:
    scaler = TemperatureScaler().to(device)
    optimizer = torch.optim.LBFGS([scaler.T], lr=0.01, max_iter=50)
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y, _ in val_loader:
            all_logits.append(model(x.to(device)))
            all_labels.append(y.to(device))
    lc = torch.cat(all_logits); la = torch.cat(all_labels)
    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(scaler(lc), la)
        loss.backward()
        return loss
    optimizer.step(closure)
    return scaler.T.item()
 
 
def train_model(args):
    if not HAS_TORCH:
        logging.error("PyTorch required."); return
 
    device = args.device
    label_df = pd.read_csv(args.label_csv, parse_dates=["date"])
    label_df["date"] = label_df["date"].dt.strftime("%Y-%m-%d")
 
    train_ds = SectorDataset(args.data_dir, label_df, "train", args.resolution)
    val_ds   = SectorDataset(args.data_dir, label_df, "val",   args.resolution)
 
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True,
                              num_workers=NW, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BS, shuffle=False,
                              num_workers=NW, pin_memory=True)
 
    model = build_mobilenet(device)
    criterion = FocalLossWithMask()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
 
    best_val_acc = 0.0
    best_state = None
 
    logging.info(f"V2 Trainer | {args.resolution} | MobileNetV3 | "
                 f"Self-ref mask active")
 
    for epoch in range(args.epochs):
        model.train()
        correct = total = 0
 
        for x, y, mask in train_loader:
            x, y = x.to(device), y.to(device)
            mask = mask.to(device).float()
 
            lam = np.random.beta(MIXUP_A, MIXUP_A)
            perm = torch.randperm(x.size(0), device=device)
            x_mixed = lam * x + (1 - lam) * x[perm]
 
            logits = model(x_mixed)
            loss = (lam * criterion(logits, y, mask) +
                    (1 - lam) * criterion(logits, y[perm], mask))
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
 
        scheduler.step()
 
        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                preds = model(x.to(device)).argmax(1)
                vc += (preds == y.to(device)).sum().item()
                vt += y.size(0)
 
        va = vc / max(vt, 1)
        if (epoch + 1) % 5 == 0:
            logging.info(f"  Ep {epoch+1}/{args.epochs} "
                         f"Train={correct/total:.3f} Val={va:.3f}")
        if va > best_val_acc:
            best_val_acc = va
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
 
    model.load_state_dict(best_state)
    model.to(device)
    opt_temp = calibrate_temperature(model, val_loader, device)
 
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"model_{args.resolution}.pth"
 
    sr_count = train_ds.data["_self_ref"].sum()
    torch.save({
        "model_state_dict": best_state,
        "optimal_temperature": opt_temp,
        "best_val_acc": best_val_acc,
        "backbone": "mobilenet_v3_large",
        "render_size": B.PX,
        "resolution": args.resolution,
        "self_ref_masked": int(sr_count),
        "self_ref_method": "per-sample loss mask (weight>5% → mask=0.0)",
        "gics_reclassification": "XLC:2018-09-24, XLRE:2015-10-07",
        "focal_gamma": FOCAL_G,
        "label_smoothing": LABEL_S,
        "mixup_alpha": MIXUP_A,
    }, save_path)
 
    logging.info(f"✓ {save_path} | Val={best_val_acc:.3f} T={opt_temp:.3f} "
                 f"| Self-ref masked: {sr_count:,}")
 
 
def main():
    p = argparse.ArgumentParser(
        description="V2 Sector Geometry — MobileNetV3 Trainer")
    p.add_argument("--resolution", choices=B.RESOLUTIONS, required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--label_csv", required=True)
    p.add_argument("--output_dir", default=".")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", default="mps")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    train_model(args)
 
 
if __name__ == "__main__":
    main()