"""
trainer.py — Var 05 MACD Momentum (Trainer)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS DOES:
  Evaluates 24 valid MACD parameter combos across resolutions.
  Selects best combo via Spearman IC with Harvey t-stat gating
  and BLL bootstrap. DHS derivative is NOT tuned — it's a FIXED
  feature that passes through to the God Model unchanged.
 
CONSISTENCY:
  - Harvey, Liu & Zhu (2016): t ≥ 3.0 full, t ≥ 2.0 partial
  - BLL (1992): 1000 bootstrap permutations, p < 0.05
  - Temporal split: Train ≤ 2022, Val ≤ 2023, Test = 2024+
  - IC targets: Daily 0.04-0.08, Weekly 0.05-0.10
 
VARIABLE-UNIQUE:
  - MACD is PRO-momentum (trend-following), opposite to V04 RSI
  - DHS derivative: positive hist + accelerating = strong bull;
    positive hist + decelerating = momentum peak (early bear warning)
  - hist_delta_norm is TABULAR_COLUMNS[4:5] — always included
 
OUTPUT:
  selected_params_{resolution}.json
 
USAGE:
  python trainer.py --data_dir ../../00_Mini_Model/data/ohlcv --resolution daily
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
 
import argparse, json, logging, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
 
THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent / "00_Mini_Model"))
import importlib as _il
B = _il.import_module("01_base_modules_")
 
TRAIN_END   = B.TRAIN_END
VAL_END     = B.VAL_END
PROFIT_MULT = 2.0
STOP_MULT   = 1.0
VERT_D      = 20
VERT_W      = 12
N_BOOTSTRAP = 1000
 
 
def harvey_weight(t_stat):
    if t_stat >= 3.0: return 1.0
    if t_stat >= 2.0: return t_stat - 2.0
    return 0.0
 
 
def triple_barrier_labels(close, atr, profit_mult, stop_mult, horizon):
    n = len(close)
    labels = pd.Series(["neutral"] * n, index=close.index)
    for i in range(n - 1):
        entry = close.iloc[i]
        a = atr.iloc[i]
        if np.isnan(a) or a <= 0:
            continue
        tp = entry + profit_mult * a
        sl = entry - stop_mult * a
        for j in range(1, min(horizon + 1, n - i)):
            c = close.iloc[i + j]
            if c >= tp:
                labels.iloc[i] = "strongly_bullish"; break
            if c <= sl:
                labels.iloc[i] = "bearish"; break
    return labels
 
 
def compute_ic(signal, fwd_ret):
    mask = ~(np.isnan(signal) | np.isnan(fwd_ret))
    if mask.sum() < 30: return 0.0
    ic, _ = spearmanr(signal[mask], fwd_ret[mask])
    return ic if not np.isnan(ic) else 0.0
 
 
def bll_bootstrap(signal, fwd_ret, n_boot=N_BOOTSTRAP):
    mask = ~(np.isnan(signal) | np.isnan(fwd_ret))
    s, r = signal[mask], fwd_ret[mask]
    if len(s) < 30: return 1.0
    obs = spearmanr(s, r)[0]
    rng = np.random.default_rng(42)
    nulls = np.array([spearmanr(rng.permutation(s), r)[0] for _ in range(n_boot)])
    return float(np.mean(nulls >= obs))
 
 
def load_and_label(data_dir, resolution, tickers):
    horizon = VERT_D if resolution in ("daily", "15min", "1hour") else VERT_W
    frames = []
    for tk in tickers:
        p = Path(data_dir) / resolution / f"{tk}.parquet"
        if not p.exists(): continue
        df = pd.read_parquet(p)
        if "datetime" not in df.columns and df.index.name == "datetime":
            df = df.reset_index()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        if len(df) < 60: continue
 
        atr = B.wilder_atr(df["high"], df["low"], df["close"])
        df["fwd_ret"] = df["close"].pct_change(horizon).shift(-horizon)
        df["label"] = triple_barrier_labels(df["close"], atr, PROFIT_MULT, STOP_MULT, horizon)
        df["ticker"] = tk
        ts = pd.to_datetime(df["datetime"])
        df["split"] = "test"
        df.loc[ts <= pd.Timestamp(TRAIN_END), "split"] = "train"
        df.loc[(ts > pd.Timestamp(TRAIN_END)) & (ts <= pd.Timestamp(VAL_END)), "split"] = "val"
        frames.append(df[["datetime", "ticker", "split", "fwd_ret", "label"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
 
 
def evaluate_combos(signal_dir, labels_df):
    sig_path = Path(signal_dir)
    tickers = labels_df["ticker"].unique()
    results = []
 
    # Get unique combos from first available file
    sample_files = list(sig_path.glob("*.parquet"))
    if not sample_files: return []
    sample = pd.read_parquet(sample_files[0])
    combo_ids = sample["combo_id"].unique()
 
    for cid in combo_ids:
        all_signals = []
        for tk in tickers:
            sp = sig_path / f"{tk}.parquet"
            if not sp.exists(): continue
            sdf = pd.read_parquet(sp)
            combo = sdf[sdf["combo_id"] == cid]
            if len(combo) == 0: continue
            merged = combo.merge(labels_df[labels_df["ticker"] == tk],
                                 on=["datetime", "ticker"], how="inner")
            if len(merged) > 0:
                merged = merged.copy()
                merged["net_score"] = merged["bull_macd_score"] - merged["bear_macd_score"]
                all_signals.append(merged)
 
        if not all_signals: continue
        full = pd.concat(all_signals, ignore_index=True)
 
        tr = full["split"] == "train"
        va = full["split"] == "val"
        te = full["split"] == "test"
 
        ic_tr = compute_ic(full.loc[tr, "net_score"].values, full.loc[tr, "fwd_ret"].values)
        ic_va = compute_ic(full.loc[va, "net_score"].values, full.loc[va, "fwd_ret"].values)
        ic_te = compute_ic(full.loc[te, "net_score"].values, full.loc[te, "fwd_ret"].values)
 
        n_tr = int(tr.sum())
        t_stat = ic_tr * np.sqrt(n_tr) / np.sqrt(1 - ic_tr**2) if abs(ic_tr) > 1e-8 else 0.0
        bll_p = bll_bootstrap(full.loc[tr, "net_score"].values, full.loc[tr, "fwd_ret"].values)
 
        # Extract params from combo_id (e.g. "f12_s26_g9")
        parts = cid.split("_")
        fast = int(parts[0][1:])
        slow = int(parts[1][1:])
        sig = int(parts[2][1:])
 
        # DHS derivative IC (fixed feature — report but don't select on it)
        dhs_ic_tr = compute_ic(full.loc[tr, "hist_delta_norm"].values, full.loc[tr, "fwd_ret"].values)
 
        results.append({
            "combo_id": cid, "fast": fast, "slow": slow, "sig_period": sig,
            "ic_train": round(ic_tr, 5), "ic_val": round(ic_va, 5), "ic_test": round(ic_te, 5),
            "t_stat": round(t_stat, 3), "harvey_weight": round(harvey_weight(t_stat), 3),
            "bll_p_value": round(bll_p, 4),
            "dhs_ic_train": round(dhs_ic_tr, 5),
            "n_train": n_tr, "n_val": int(va.sum()), "n_test": int(te.sum()),
        })
 
    return sorted(results, key=lambda x: -x["ic_val"])
 
 
def main():
    p = argparse.ArgumentParser(description="V05 MACD Momentum — Trainer")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--resolution", choices=["daily", "weekly", "15min", "1hour"], required=True)
    p.add_argument("--tickers", nargs="*", default=None)
    args = p.parse_args()
 
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    logging.info(f"V05 MACD Trainer | Resolution: {args.resolution}")
 
    tf_dir = Path(args.data_dir) / args.resolution
    tickers = args.tickers or sorted(
        p.stem for p in tf_dir.glob("*.parquet") if not p.stem.startswith("_"))
    logging.info(f"  Tickers: {len(tickers)}")
 
    labels_df = load_and_label(args.data_dir, args.resolution, tickers)
    if len(labels_df) == 0:
        logging.error("No label data."); return
    logging.info(f"  Label rows: {len(labels_df):,}")
 
    signal_dir = THIS / args.resolution / "signals"
    results = evaluate_combos(str(signal_dir), labels_df)
    if not results:
        logging.error("No combo results."); return
 
    logging.info(f"\n{'='*90}")
    logging.info(f"  {'combo':>14} | {'IC_tr':>7} | {'IC_va':>7} | {'IC_te':>7} | "
                 f"{'t-stat':>7} | {'Harvey':>7} | {'BLL_p':>7} | {'DHS_IC':>7}")
    logging.info(f"  {'-'*14} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7}")
    for r in results:
        flag = "✓" if r["harvey_weight"] > 0 and r["bll_p_value"] < 0.05 else " "
        logging.info(f"{flag} {r['combo_id']:>14} | {r['ic_train']:>7.4f} | {r['ic_val']:>7.4f} | "
                     f"{r['ic_test']:>7.4f} | {r['t_stat']:>7.2f} | {r['harvey_weight']:>7.3f} | "
                     f"{r['bll_p_value']:>7.4f} | {r['dhs_ic_train']:>7.4f}")
    logging.info(f"{'='*90}")
 
    valid = [r for r in results if r["harvey_weight"] > 0 and r["bll_p_value"] < 0.05]
    if valid:
        best = max(valid, key=lambda x: x["ic_val"])
    else:
        logging.warning("  No combo passed gates. Selecting best val IC.")
        best = results[0]
 
    logging.info(f"\n  ★ Selected: {best['combo_id']} | IC_val={best['ic_val']:.4f} | "
                 f"t={best['t_stat']:.2f} | DHS_IC={best['dhs_ic_train']:.4f}")
 
    out_path = THIS / f"selected_params_{args.resolution}.json"
    with open(out_path, "w") as f:
        json.dump({"selected": best, "all_combos": results}, f, indent=2)
    logging.info(f"  → {out_path}")
 
 
if __name__ == "__main__":
    main()