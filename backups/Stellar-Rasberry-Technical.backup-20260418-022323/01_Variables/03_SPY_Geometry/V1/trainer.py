"""
03_SPY_Geometry/V1/trainer.py
===============================
MobileNetV3-Large trainer for V3 SPY/VIX Geometry. 5-resolution.
 
Two separate models trained independently:
  --instrument SPY: learns market structure from SPY candlestick charts
  --instrument VIX: learns volatility regime from VIX line charts
 
Architecture:
  Backbone: MobileNetV3-Large (same as V2 — global instruments have
            less visual diversity than 500 individual stocks)
  Input: 224×224 (resized from 680×680)
  Output: 3-class [P(bearish), P(neutral), P(strongly_bullish)]
  Loss: Focal γ=2.0, Label Smoothing ε=0.1
  Augmentation: Mixup α=0.3
  Calibration: Guo et al. (2017) temperature scaling
  Init: zero-gamma on final BatchNorm
 
VIX Label Inversion (Non-Negotiable):
  Standard labels: 0=bearish, 1=neutral, 2=strongly_bullish
  VIX inversion:   labels.map({0: 2, 1: 1, 2: 0})
 
  WHY: when the MARKET is bearish (label=0), VIX RISES.
  A "bullish" VIX reading means stress is incoming.
  After inversion, P(bull) from the VIX CNN = P(stress rising).
  The meta-learner receives this inverted signal as-is —
  high v3_vix_bull_prob = high fear = suppress confidence.
 
  This ensures the VIX CNN's output is semantically consistent
  with the rest of the system: bull_prob always means "condition
  favoring the signal direction."
 
Data:
  SPY images: {data_dir}/images_spy/{date}.png
  VIX images: {data_dir}/images_vix/{date}.png
  GLOBAL: no ticker column needed. Each date has exactly 1 image.
 
Usage:
  python trainer.py --instrument SPY --resolution 15min --data_dir 15min
  python trainer.py --instrument VIX --resolution daily --data_dir daily
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
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none",
                             label_smoothing=LABEL_S)
        pt = torch.exp(-ce)
        return ((1 - pt) ** FOCAL_G * ce).mean()
 
 
class GlobalDataset(Dataset):
    """
    Reads global chart images (1 per date, no ticker dimension).
    For SPY: images_spy/. For VIX: images_vix/.
    Deduplicates on date — each date appears exactly once.
    """
    def __init__(self, img_dir: str, label_df: pd.DataFrame,
                 split: str, resolution: str, instrument: str):
        self.dir = Path(img_dir) / f"images_{instrument.lower()}"
        self.data = label_df[label_df["split"] == split].drop_duplicates(
            subset=["date"]).reset_index(drop=True)
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
        logging.info(f"  GlobalDataset [{instrument} {split}]: "
                     f"{len(self.data):,} rows, {n_found:,} on disk")
 
    def _path(self, row) -> str:
        if self.res in ("15min", "1hour"):
            fname = f"{B.fmt_15m(pd.Timestamp(row['date']))}.png"
        else:
            fname = f"{row['date']}.png"
        return str(self.dir / fname)
 
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
 
 
def build_mobilenet(device: str) -> nn.Module:
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
        for x, y in val_loader:
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
    inst = args.instrument.upper()
 
    label_df = pd.read_csv(args.label_csv, parse_dates=["date"])
    label_df["date"] = label_df["date"].dt.strftime("%Y-%m-%d")
 
    # ── VIX LABEL INVERSION (non-negotiable) ──
    if inst == "VIX" and "label" in label_df.columns:
        label_df["label"] = label_df["label"].map({0: 2, 1: 1, 2: 0})
        logging.info("VIX: labels inverted — P(bull) now = P(stress rising)")
 
    train_ds = GlobalDataset(args.data_dir, label_df, "train",
                             args.resolution, inst)
    val_ds   = GlobalDataset(args.data_dir, label_df, "val",
                             args.resolution, inst)
 
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True,
                              num_workers=NW, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BS, shuffle=False,
                              num_workers=NW, pin_memory=True)
 
    model = build_mobilenet(device)
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
 
    best_val_acc = 0.0
    best_state = None
 
    logging.info(f"V3 Trainer | {inst} {args.resolution} | MobileNetV3 | "
                 f"Label inverted={inst == 'VIX'}")
 
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
        vc = vt = 0
        with torch.no_grad():
            for x, y in val_loader:
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
    save_path = out_dir / f"model_{inst}_{args.resolution}.pth"
 
    torch.save({
        "model_state_dict": best_state,
        "optimal_temperature": opt_temp,
        "best_val_acc": best_val_acc,
        "backbone": "mobilenet_v3_large",
        "instrument": inst,
        "label_inverted": inst == "VIX",
        "render_size": B.PX,
        "resolution": args.resolution,
        "focal_gamma": FOCAL_G,
        "label_smoothing": LABEL_S,
        "mixup_alpha": MIXUP_A,
        "bear_filter": "L3 FULL CASH" if args.resolution == "monthly" else None,
    }, save_path)
 
    logging.info(f"✓ {save_path} | Val={best_val_acc:.3f} T={opt_temp:.3f} "
                 f"| Inverted={inst == 'VIX'}")
 
 
def main():
    p = argparse.ArgumentParser(
        description="V3 SPY/VIX Geometry — MobileNetV3 Trainer")
    p.add_argument("--instrument", choices=["SPY", "VIX"], required=True)
    p.add_argument("--resolution", choices=B.RESOLUTIONS, required=True)
    p.add_argument("--data_dir", required=True,
                   help="Resolution dir containing images_spy/ and images_vix/")
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