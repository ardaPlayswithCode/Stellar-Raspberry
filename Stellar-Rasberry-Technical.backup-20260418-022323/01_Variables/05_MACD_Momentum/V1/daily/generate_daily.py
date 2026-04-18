"""
generate_daily.py — Var 05 MACD Momentum (Daily Resolution)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS DOES:
  Computes MACD momentum scores for all tickers at daily resolution.
  Grid: fast[8,12,16] × slow[21,26,34] × signal[7,9,12] = 24 valid combos.
  Constraint: slow - fast >= 8 (eliminates fast=16/slow=21).
 
CONSISTENCY:
  - EMA: Standard span-based (alpha=2/(span+1), adjust=False) — NOT Wilder
  - MOP (2012) normalization: histogram / ATR
  - DHS (1998) derivative: tanh((hist[t]-hist[t-1])/atr) — FIXED column, not tuned
  - ATR: Wilder EWM span=14 (from 01_base_modules_)
  - Local parquet only, zstd compression, vectorized
 
RESEARCH:
  Appel (1985) — original MACD
  Daniel, Hirshleifer & Subrahmanyam (1998) JF — DHS momentum derivative
  Moskowitz, Ooi & Pedersen (2012) JFE — MOP vol normalization
 
OUTPUT SCHEMA:
  signals/{TICKER}.parquet → [datetime, ticker, combo_id,
    fast, slow, sig_period, bull_macd_score, bear_macd_score, hist_delta_norm]
 
USAGE:
  python generate_daily.py --data_dir ../../00_Mini_Model/data/ohlcv --workers 24
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
 
import argparse, logging, sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
 
THIS = Path(__file__).resolve().parent
V1   = THIS.parent
sys.path.insert(0, str(V1.parent.parent / "00_Mini_Model"))
import importlib as _il
B = _il.import_module("01_base_modules_")
 
RESOLUTION  = "daily"
FETCH_START = "2005-01-01"
OUT         = THIS / "signals"
FAST_GRID   = [8, 12, 16]
SLOW_GRID   = [21, 26, 34]
SIG_GRID    = [7, 9, 12]
N_CORES     = min(cpu_count(), 24)
 
 
def macd_histogram(close: pd.Series, fast: int, slow: int, sig: int) -> pd.Series:
    """Standard EMA MACD — Appel/Neftci convention. NOT Wilder."""
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    macd_line = ef - es
    signal_line = macd_line.ewm(span=sig, adjust=False).mean()
    return macd_line - signal_line
 
 
def valid_combos():
    """27 raw → 24 valid. Constraint: slow - fast >= 8."""
    combos = []
    for f in FAST_GRID:
        for s in SLOW_GRID:
            if s - f < 8:
                continue
            for sig in SIG_GRID:
                combos.append((f, s, sig))
    return combos
 
 
COMBOS = valid_combos()  # 24 combos, computed once
 
 
def process_ticker(args):
    data_dir, ticker = args
    p = Path(data_dir) / RESOLUTION / f"{ticker}.parquet"
    if not p.exists():
        return None
 
    df = pd.read_parquet(p)
    if "datetime" not in df.columns and df.index.name == "datetime":
        df = df.reset_index()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df[df["datetime"] >= FETCH_START].reset_index(drop=True)
    if len(df) < 60:
        return None
 
    close = df["close"]
    atr = B.wilder_atr(df["high"], df["low"], close)
    atr_safe = atr.replace(0, np.nan)
    rows = []
 
    for fast, slow, sig in COMBOS:
        hist = macd_histogram(close, fast, slow, sig)
 
        # MOP normalization (Moskowitz, Ooi & Pedersen 2012)
        hist_norm = (hist / atr_safe).fillna(0)
        bull = np.clip(np.tanh(hist_norm.values), 0, 1).astype(np.float32)
        bear = np.clip(np.tanh(-hist_norm.values), 0, 1).astype(np.float32)
 
        # DHS derivative — FIXED column, not a parameter
        hist_delta = hist.diff()
        hist_delta_norm = np.tanh((hist_delta / atr_safe).fillna(0).values).astype(np.float32)
 
        combo_id = f"f{fast}_s{slow}_g{sig}"
        combo_df = pd.DataFrame({
            "datetime": df["datetime"].values,
            "ticker": ticker,
            "combo_id": combo_id,
            "fast": np.int8(fast),
            "slow": np.int8(slow),
            "sig_period": np.int8(sig),
            "bull_macd_score": bull,
            "bear_macd_score": bear,
            "hist_delta_norm": hist_delta_norm,
        })
        rows.append(combo_df)
 
    if not rows:
        return None
    result = pd.concat(rows, ignore_index=True)
    out_path = OUT / f"{ticker}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    return ticker
 
 
def main():
    p = argparse.ArgumentParser(description="V05 MACD Momentum — Daily Generator")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--workers", type=int, default=N_CORES)
    p.add_argument("--tickers", nargs="*", default=None)
    args = p.parse_args()
 
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    if args.tickers:
        tickers = args.tickers
    else:
        tf_dir = Path(args.data_dir) / RESOLUTION
        tickers = sorted(p.stem for p in tf_dir.glob("*.parquet") if not p.stem.startswith("_"))
 
    logging.info(f"V05 MACD Daily | Tickers: {len(tickers)} | Combos: {len(COMBOS)} (24 valid)")
 
    jobs = [(args.data_dir, tk) for tk in tickers]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.map(process_ticker, jobs, chunksize=16)
    else:
        results = [process_ticker(j) for j in jobs]
 
    ok = [r for r in results if r is not None]
    logging.info(f"✓ Daily signals: {len(ok)}/{len(tickers)} tickers → {OUT}/")
 
 
if __name__ == "__main__":
    main()