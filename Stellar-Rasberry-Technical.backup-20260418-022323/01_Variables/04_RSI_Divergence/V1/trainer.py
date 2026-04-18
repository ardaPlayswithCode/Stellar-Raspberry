"""
trainer.py — Var 04 RSI Divergence (Trainer)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS DOES:
  Evaluates all 12 parameter combos (4 RSI periods × 3 div lookbacks)
  across daily and weekly resolutions. Selects the best combo per
  resolution using Spearman IC with Harvey t-stat gating (t ≥ 2.0)
  and BLL bootstrap validation (p < 0.05).
 
CONSISTENCY:
  - Harvey, Liu & Zhu (2016) RFS: t ≥ 3.0 full weight, t ≥ 2.0 partial
  - Brock, Lakonishok & LeBaron (1992) JF: bootstrap null distribution
  - Temporal split: Train ≤ 2022, Val ≤ 2023, Test = 2024+
  - Triple barrier labels from 01_base_modules_ constants
  - Local parquet only, no yfinance
 
VARIABLE-UNIQUE:
  - RSI is ANTI-momentum. High IC expected when partialing out J&T.
  - IC targets: Daily 0.03-0.07, Weekly 0.05-0.09
  - Half-life: 3-4 years
 
OUTPUT SCHEMA:
  selected_params_daily.json, selected_params_weekly.json
  Each: {combo_id, rsi_period, div_lookback, ic_train, ic_val, ic_test,
         t_stat, harvey_weight, bll_p_value}
  daily/signal/ and weekly/signal/: consolidated signal parquets
 
USAGE:
  python trainer.py --data_dir ../../00_Mini_Model/data/ohlcv --resolution daily
  python trainer.py --data_dir ../../00_Mini_Model/data/ohlcv --resolution weekly
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
 
TRAIN_END    = B.TRAIN_END
VAL_END      = B.VAL_END
ATR_P        = B.ATR_P
PROFIT_MULT  = 2.0
STOP_MULT    = 1.0
VERT_D       = 20
VERT_W       = 12
N_BOOTSTRAP  = 1000
LABELS       = ["strongly_bullish", "neutral", "bearish"]
 
RSI_PERIODS   = [7, 10, 14, 21]
DIV_LOOKBACKS = [5, 10, 20]
 
 
def harvey_weight(t_stat: float) -> float:
    if t_stat >= 3.0:
        return 1.0
    elif t_stat >= 2.0:
        return t_stat - 2.0
    return 0.0
 
 
def triple_barrier_labels(close: pd.Series, atr: pd.Series,
                          profit_mult: float, stop_mult: float,
                          horizon: int) -> pd.Series:
    """Vectorized triple barrier. Upper checked first (bullish bias)."""
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
                labels.iloc[i] = "strongly_bullish"
                break
            if c <= sl:
                labels.iloc[i] = "bearish"
                break
 
    return labels
 
 
def compute_ic(signal: np.ndarray, fwd_ret: np.ndarray) -> float:
    mask = ~(np.isnan(signal) | np.isnan(fwd_ret))
    if mask.sum() < 30:
        return 0.0
    ic, _ = spearmanr(signal[mask], fwd_ret[mask])
    return ic if not np.isnan(ic) else 0.0
 
 
def bll_bootstrap(signal: np.ndarray, fwd_ret: np.ndarray,
                  n_boot: int = N_BOOTSTRAP) -> float:
    """BLL (1992) bootstrap: shuffle signal, compute null IC distribution."""
    mask = ~(np.isnan(signal) | np.isnan(fwd_ret))
    sig_clean = signal[mask]
    ret_clean = fwd_ret[mask]
    if len(sig_clean) < 30:
        return 1.0
    observed_ic = spearmanr(sig_clean, ret_clean)[0]
    rng = np.random.default_rng(42)
    null_ics = np.array([
        spearmanr(rng.permutation(sig_clean), ret_clean)[0]
        for _ in range(n_boot)
    ])
    p_value = np.mean(null_ics >= observed_ic)
    return p_value
 
 
def load_and_label(data_dir: str, resolution: str, tickers: list) -> pd.DataFrame:
    """Load OHLCV, compute ATR, triple-barrier labels, forward returns."""
    horizon = VERT_D if resolution == "daily" else VERT_W
    all_frames = []
 
    for tk in tickers:
        p = Path(data_dir) / resolution / f"{tk}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if "datetime" not in df.columns and df.index.name == "datetime":
            df = df.reset_index()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        if len(df) < 60:
            continue
 
        atr = B.wilder_atr(df["high"], df["low"], df["close"])
        df["atr"] = atr
        df["fwd_ret"] = df["close"].pct_change(horizon).shift(-horizon)
        df["label"] = triple_barrier_labels(df["close"], atr, PROFIT_MULT, STOP_MULT, horizon)
        df["ticker"] = tk
 
        ts = pd.to_datetime(df["datetime"])
        df["split"] = "test"
        df.loc[ts <= pd.Timestamp(TRAIN_END), "split"] = "train"
        df.loc[(ts > pd.Timestamp(TRAIN_END)) & (ts <= pd.Timestamp(VAL_END)), "split"] = "val"
 
        all_frames.append(df[["datetime", "ticker", "split", "fwd_ret", "label"]])
 
    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)
 
 
def evaluate_combos(signal_dir: str, labels_df: pd.DataFrame,
                    resolution: str) -> list:
    """Evaluate all 12 combos across all tickers."""
    sig_path = Path(signal_dir)
    tickers = labels_df["ticker"].unique()
    results = []
 
    for period in RSI_PERIODS:
        for lookback in DIV_LOOKBACKS:
            combo_id = f"p{period}_lb{lookback}"
            all_signals = []
 
            for tk in tickers:
                sp = sig_path / f"{tk}.parquet"
                if not sp.exists():
                    continue
                sdf = pd.read_parquet(sp)
                combo = sdf[sdf["combo_id"] == combo_id].copy()
                if len(combo) == 0:
                    continue
 
                merged = combo.merge(
                    labels_df[labels_df["ticker"] == tk],
                    on=["datetime", "ticker"], how="inner"
                )
                if len(merged) > 0:
                    # Net score: bull - bear (anti-momentum signal)
                    merged["net_score"] = merged["bull_rsi_score"] - merged["bear_rsi_score"]
                    all_signals.append(merged)
 
            if not all_signals:
                continue
 
            full = pd.concat(all_signals, ignore_index=True)
 
            # IC by split
            train_mask = full["split"] == "train"
            val_mask = full["split"] == "val"
            test_mask = full["split"] == "test"
 
            ic_train = compute_ic(full.loc[train_mask, "net_score"].values,
                                  full.loc[train_mask, "fwd_ret"].values)
            ic_val = compute_ic(full.loc[val_mask, "net_score"].values,
                                full.loc[val_mask, "fwd_ret"].values)
            ic_test = compute_ic(full.loc[test_mask, "net_score"].values,
                                 full.loc[test_mask, "fwd_ret"].values)
 
            # Harvey t-stat on train set
            n_train = train_mask.sum()
            t_stat = ic_train * np.sqrt(n_train) / np.sqrt(1 - ic_train ** 2) if abs(ic_train) > 1e-8 else 0.0
 
            # BLL bootstrap on train set
            bll_p = bll_bootstrap(
                full.loc[train_mask, "net_score"].values,
                full.loc[train_mask, "fwd_ret"].values
            )
 
            results.append({
                "combo_id": combo_id,
                "rsi_period": period,
                "div_lookback": lookback,
                "ic_train": round(ic_train, 5),
                "ic_val": round(ic_val, 5),
                "ic_test": round(ic_test, 5),
                "t_stat": round(t_stat, 3),
                "harvey_weight": round(harvey_weight(t_stat), 3),
                "bll_p_value": round(bll_p, 4),
                "n_train": int(n_train),
                "n_val": int(val_mask.sum()),
                "n_test": int(test_mask.sum()),
            })
 
    return sorted(results, key=lambda x: -x["ic_val"])
 
 
def main():
    p = argparse.ArgumentParser(description="V04 RSI Divergence — Trainer")
    p.add_argument("--data_dir", required=True, help="Path to data/ohlcv")
    p.add_argument("--resolution", choices=["daily", "weekly", "15min", "1hour"],
                   required=True)
    p.add_argument("--tickers", nargs="*", default=None)
    args = p.parse_args()
 
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    logging.info(f"V04 RSI Trainer | Resolution: {args.resolution}")
 
    # Discover tickers
    tf_dir = Path(args.data_dir) / args.resolution
    if args.tickers:
        tickers = args.tickers
    else:
        tickers = sorted(p.stem for p in tf_dir.glob("*.parquet") if not p.stem.startswith("_"))
 
    logging.info(f"  Tickers: {len(tickers)}")
 
    # Load labels
    logging.info("  Computing triple-barrier labels...")
    labels_df = load_and_label(args.data_dir, args.resolution, tickers)
    if len(labels_df) == 0:
        logging.error("No label data. Run generators first.")
        return
 
    logging.info(f"  Label rows: {len(labels_df):,}")
    logging.info(f"  Distribution: {labels_df['label'].value_counts().to_dict()}")
 
    # Evaluate combos
    signal_dir = THIS / args.resolution / "signals"
    logging.info(f"  Evaluating 12 combos from {signal_dir}...")
    results = evaluate_combos(str(signal_dir), labels_df, args.resolution)
 
    if not results:
        logging.error("No combo results. Check signal files exist.")
        return
 
    # Report
    logging.info(f"\n{'='*80}")
    logging.info(f"  {'combo':>12} | {'IC_tr':>7} | {'IC_va':>7} | {'IC_te':>7} | "
                 f"{'t-stat':>7} | {'Harvey':>7} | {'BLL_p':>7}")
    logging.info(f"  {'-'*12} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7}")
    for r in results:
        flag = "✓" if r["harvey_weight"] > 0 and r["bll_p_value"] < 0.05 else " "
        logging.info(f"{flag} {r['combo_id']:>12} | {r['ic_train']:>7.4f} | {r['ic_val']:>7.4f} | "
                     f"{r['ic_test']:>7.4f} | {r['t_stat']:>7.2f} | {r['harvey_weight']:>7.3f} | "
                     f"{r['bll_p_value']:>7.4f}")
    logging.info(f"{'='*80}")
 
    # Select best combo: highest val IC with Harvey weight > 0 and BLL p < 0.05
    valid = [r for r in results if r["harvey_weight"] > 0 and r["bll_p_value"] < 0.05]
    if valid:
        best = max(valid, key=lambda x: x["ic_val"])
    else:
        logging.warning("  No combo passed Harvey + BLL gates. Selecting best val IC anyway.")
        best = results[0]
 
    logging.info(f"\n  ★ Selected: {best['combo_id']} | IC_val={best['ic_val']:.4f} | "
                 f"t={best['t_stat']:.2f} | Harvey={best['harvey_weight']:.3f}")
 
    # Save
    out_path = THIS / f"selected_params_{args.resolution}.json"
    with open(out_path, "w") as f:
        json.dump({"selected": best, "all_combos": results}, f, indent=2)
    logging.info(f"  → {out_path}")
 
 
if __name__ == "__main__":
    main()