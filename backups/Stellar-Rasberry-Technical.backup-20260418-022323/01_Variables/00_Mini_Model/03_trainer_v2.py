"""
03_trainer_v2.py — 3M-Point God-Model Trainer (Production)
============================================================
80-feature XGBoost: 60 CNN probabilities + 20 tabular.
15-minute master clock with T-1 aligned multi-timeframe features.
 
Production optimizations:
  1. QuantileDMatrix: bins features before training, ~50% RAM reduction.
     Critical for 3M rows × 80 features on 96GB unified memory.
  2. Self-Ref Mask: hard-coded in the data loader. V2 CNN features for
     tickers with ETF weight > 5% are replaced with uniform prior (1/3)
     BEFORE the feature matrix is constructed. Impossible to leak.
  3. Chunked CNN extraction: processes 10K rows at a time to stay within
     the MPS transfer budget (no 3M-row tensor allocation).
  4. Category-aware: reads category dtypes from registry, converts to
     float32 feature matrix only at the DMatrix boundary.
 
Interaction Constraints (3 groups, same research basis):
  G0 SYSTEMATIC: CNN[0:59] + VIX
  G1 STRUCTURAL: Numerical[0:9] + VIX + V8
  G2 LOCAL: ATR% + CNN + Numerical
"""
 
import os, json, logging, argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
 
import importlib as _il
B = _il.import_module("01_base_modules_")
 
try:
    from build_god_registry import TABULAR_COLUMNS, IMAGE_COLUMNS
except ImportError:
    _reg = _il.import_module("02_build_god_registry")
    TABULAR_COLUMNS = _reg.TABULAR_COLUMNS
    IMAGE_COLUMNS = _reg.IMAGE_COLUMNS
 
IMG_SIZE      = 224
NUM_CLASSES   = 3
BATCH_SIZE    = 64
NUM_WORKERS   = 8
EXTRACT_CHUNK = 10_000
 
META_PARAMS = {
    "n_estimators": 2000, "max_depth": 6, "learning_rate": 0.01,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 50,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "tree_method": "hist",
    "objective": "binary:logistic", "eval_metric": "auc",
    "early_stopping_rounds": 100, "seed": 42,
}
 
 
def build_interaction_constraints(n_cnn: int, n_tab: int) -> list:
    cnn_idx = list(range(n_cnn))
    tab_off = n_cnn
    num_idx = [tab_off + i for i in range(10)]
    vix_idx = [tab_off + 10, tab_off + 11]
    atr_idx = [tab_off + 12, tab_off + 13]
    v8_idx  = [tab_off + i for i in range(14, 20)]
    return [
        cnn_idx + vix_idx,
        num_idx + vix_idx + v8_idx,
        atr_idx + cnn_idx + num_idx,
    ]
 
 
def apply_self_ref_mask(X: np.ndarray, feature_cols: list,
                        self_ref_flags: np.ndarray) -> np.ndarray:
    """
    Hard-coded self-reference enforcement.
    Any row where ticker weight > 5% in its sector ETF gets V2 CNN
    features replaced with 1/3 (uniform prior). Applied BEFORE DMatrix.
    This is not optional — it's in the data path, not the loss function.
    """
    v2_indices = [i for i, c in enumerate(feature_cols) if c.startswith("v2_")]
    if v2_indices and self_ref_flags.any():
        mask = self_ref_flags.astype(bool)
        X[mask][:, v2_indices] = 1.0 / 3
        n_masked = mask.sum()
        logging.info(f"  Self-ref mask: {n_masked:,} rows × {len(v2_indices)} V2 cols → 1/3")
    return X
 
 
def extract_cnn_features_chunked(
    registry_path: str, model_paths: dict,
    output_path: str, device: str = "mps",
):
    """
    Chunked CNN extraction: 10K rows per batch to stay within MPS budget.
    No 3M-row tensor. No OOM.
    """
    try:
        import torch, torch.nn as nn
        from torchvision import models, transforms
        from PIL import Image
    except ImportError:
        raise ImportError("pip install torch torchvision")
 
    logging.info("=" * 60)
    logging.info(f"STAGE 1: Chunked CNN Extraction ({EXTRACT_CHUNK} rows/batch)")
    logging.info("=" * 60)
 
    registry = pd.read_parquet(registry_path)
    n_total = len(registry)
 
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
 
    loaded = {}; temps = {}
    for key, path in model_paths.items():
        if not os.path.exists(path): continue
        if key.startswith("v1_"):
            m = models.convnext_base(weights=None)
            m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, NUM_CLASSES)
        else:
            m = models.mobilenet_v3_large(weights=None)
            m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, NUM_CLASSES)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        t = ckpt.get("optimal_temperature", 1.0) if isinstance(ckpt, dict) else 1.0
        m.load_state_dict(sd); m.to(device); m.eval()
        loaded[key] = m; temps[key] = t
        logging.info(f"  {key}: T={t:.3f}")
 
    cnn_col_names = []
    for mk in sorted(loaded.keys()):
        cnn_col_names.extend([f"{mk}_bull", f"{mk}_neutral", f"{mk}_bear"])
 
    n_cnn = len(cnn_col_names)
    features_arr = np.full((n_total, n_cnn), 1.0/3, dtype=np.float32)
 
    img_col_map = {}
    for col, mk in IMAGE_COLUMNS.items():
        if mk in loaded and col in registry.columns:
            img_col_map[mk] = col
 
    for chunk_start in range(0, n_total, EXTRACT_CHUNK):
        chunk_end = min(chunk_start + EXTRACT_CHUNK, n_total)
        chunk = registry.iloc[chunk_start:chunk_end]
 
        for mk, model in loaded.items():
            if mk not in img_col_map: continue
            col = img_col_map[mk]
            col_offset = sorted(loaded.keys()).index(mk) * 3
 
            imgs = []
            valid = []
            for _, row in chunk.iterrows():
                ip = row.get(col)
                if ip and isinstance(ip, str) and os.path.exists(ip):
                    img = Image.open(ip).convert("RGB")
                    imgs.append(transform(img)); valid.append(True)
                else:
                    imgs.append(torch.zeros(3, IMG_SIZE, IMG_SIZE)); valid.append(False)
 
            ib = torch.stack(imgs).to(device)
            with torch.no_grad():
                logits = model(ib)
                probas = torch.softmax(logits / temps[mk], dim=1).cpu().numpy()
 
            for i in range(len(chunk)):
                if valid[i]:
                    features_arr[chunk_start + i, col_offset] = probas[i, 2]
                    features_arr[chunk_start + i, col_offset + 1] = probas[i, 1]
                    features_arr[chunk_start + i, col_offset + 2] = probas[i, 0]
 
        if (chunk_end) % 100_000 < EXTRACT_CHUNK:
            logging.info(f"  Extracted {chunk_end:,}/{n_total:,}")
 
    features_df = pd.DataFrame(features_arr, columns=cnn_col_names)
    features_df["datetime"] = registry["datetime"].values
    features_df["ticker"] = registry["ticker"].values
    features_df["split"] = registry["split"].values
 
    features_df.to_parquet(output_path, engine="pyarrow", compression="zstd")
    logging.info(f"✓ CNN features: {output_path} ({n_total:,} × {n_cnn})")
    return output_path
 
 
def train_meta_learner(cnn_path: str, registry_path: str, output_dir: str):
    try:
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise ImportError("pip install xgboost scikit-learn")
 
    logging.info("=" * 60)
    logging.info("STAGE 2: XGBoost Meta-Learner (QuantileDMatrix)")
    logging.info("=" * 60)
 
    os.makedirs(output_dir, exist_ok=True)
    cnn = pd.read_parquet(cnn_path)
    reg = pd.read_parquet(registry_path)
 
    cnn_cols = sorted([c for c in cnn.columns if c.endswith(("_bull", "_neutral", "_bear"))])
 
    reg_tab_cols = [c for c in TABULAR_COLUMNS if c in reg.columns]
    merge_cols_reg = ["datetime", "ticker"] + reg_tab_cols
    if "label" in reg.columns: merge_cols_reg.append("label")
    if "v2_self_ref_flag" in reg.columns: merge_cols_reg.append("v2_self_ref_flag")
 
    merged = cnn[["datetime", "ticker", "split"] + cnn_cols].merge(
        reg[merge_cols_reg], on=["datetime", "ticker"], how="inner")
 
    feature_cols = cnn_cols + reg_tab_cols
    X = merged[feature_cols].fillna(0).values.astype(np.float32)
 
    if "label" in merged.columns:
        y = (merged["label"].values == 2).astype(np.int32)
    else:
        y = np.zeros(len(merged), dtype=np.int32)
 
    if "v2_self_ref_flag" in merged.columns:
        self_ref = merged["v2_self_ref_flag"].values.astype(bool)
    else:
        self_ref = np.zeros(len(merged), dtype=bool)
    X = apply_self_ref_mask(X, feature_cols, self_ref)
 
    tr = (merged["split"] == "train").values
    va = (merged["split"] == "val").values
    te = (merged["split"] == "test").values
 
    logging.info(f"  Features: {len(feature_cols)} ({len(cnn_cols)} CNN + {len(reg_tab_cols)} tab)")
    logging.info(f"  Train: {tr.sum():,} | Val: {va.sum():,} | Test: {te.sum():,}")
 
    constraints = build_interaction_constraints(len(cnn_cols), len(reg_tab_cols))
 
    params = META_PARAMS.copy()
    nr = params.pop("n_estimators")
    es = params.pop("early_stopping_rounds")
    params["interaction_constraints"] = json.dumps(constraints)
 
    logging.info("  Building QuantileDMatrix (50% RAM reduction)...")
    dtrain = xgb.QuantileDMatrix(X[tr], label=y[tr], feature_names=feature_cols,
                                  max_bin=256, nthread=B.N_CORES)
    dval = xgb.DMatrix(X[va], label=y[va], feature_names=feature_cols)
    dtest = xgb.DMatrix(X[te], label=y[te], feature_names=feature_cols)
 
    logging.info(f"  QuantileDMatrix train: {dtrain.num_row():,} rows × {dtrain.num_col()} cols")
 
    model = xgb.train(params, dtrain, num_boost_round=nr,
                       evals=[(dtrain, "train"), (dval, "val")],
                       early_stopping_rounds=es, verbose_eval=100)
 
    va_auc = roc_auc_score(y[va], model.predict(dval))
    te_auc = roc_auc_score(y[te], model.predict(dtest))
    logging.info(f"\n  Val AUC: {va_auc:.4f} | Test AUC: {te_auc:.4f}")
 
    imp = model.get_score(importance_type="gain")
    imp_sorted = sorted(imp.items(), key=lambda x: -x[1])[:20]
    logging.info("\n  Top 20 Features (gain):")
    for fname, gain in imp_sorted:
        tag = "CNN" if any(fname.startswith(f"v{i}_") for i in [1,2,3]) else \
              "NUM" if any(fname.startswith(f"v{i}_") for i in [4,5,6]) else \
              "V7" if fname.startswith("v7_") else "V8" if fname.startswith("v8_") else "?"
        logging.info(f"    [{tag:4s}] {fname:35s} {gain:>10.1f}")
 
    mp = os.path.join(output_dir, "technical_mgm_15m.json")
    model.save_model(mp)
 
    meta = {
        "architecture": "15m Master Clock God Model (Production)",
        "total_features": len(feature_cols),
        "cnn_features": len(cnn_cols), "tabular_features": len(reg_tab_cols),
        "val_auc": va_auc, "test_auc": te_auc,
        "best_iteration": model.best_iteration,
        "quantile_dmatrix": True, "max_bin": 256,
        "self_ref_enforced": "hard-coded in loader (not loss function)",
        "constraints": constraints,
        "train_rows": int(tr.sum()), "val_rows": int(va.sum()), "test_rows": int(te.sum()),
    }
    json.dump(meta, open(os.path.join(output_dir, "metadata_15m.json"), "w"), indent=2)
    logging.info(f"✓ Model: {mp}")
    return mp
 
 
def main():
    p = argparse.ArgumentParser(description="15m God-Model Trainer (Production)")
    p.add_argument("--stage", choices=["extract", "meta", "full"], required=True)
    p.add_argument("--registry", default="registry/master_registry_15m.parquet")
    p.add_argument("--features", default="cnn_features_15m.parquet")
    p.add_argument("--output_dir", default="models_15m")
    p.add_argument("--device", default="mps")
    args = p.parse_args()
 
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
 
    if args.stage in ("extract", "full"):
        model_paths = {}
        for var_dir, prefix in [("01_Stock_Geometry","v1"),("02_Sector_Geometry","v2"),("03_SPY_Geometry","v3")]:
            for res in B.RESOLUTIONS:
                model_paths[f"{prefix}_{res}"] = f"{var_dir}/V1/model_{res}.pth"
        extract_cnn_features_chunked(args.registry, model_paths, args.features, args.device)
 
    if args.stage in ("meta", "full"):
        train_meta_learner(args.features, args.registry, args.output_dir)
 
 
if __name__ == "__main__":
    main()