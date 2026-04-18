"""
generate_15min.py — Var 04 RSI Divergence (15-Minute Resolution)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS DOES:
  Computes RSI divergence scores at 15-minute (LCM master clock) resolution.
  Grid: 4 periods × 3 lookbacks = 12 parameter combinations.
  Lookback window: 60 bars (15 hours of market time).
 
CONSISTENCY:
  - Wilder EWM: alpha = 1/period, adjust=False
  - Zero indicator drift — same RSI periods across all resolutions
  - Local parquet only, zstd compression, vectorized
  - Chunked ticker processing for 3M+ row scale
 
OUTPUT SCHEMA:
  signals/{TICKER}.parquet → [datetime, ticker, combo_id,
    rsi_period, div_lookback, bull_rsi_score, bear_rsi_score]
 
USAGE:
  python generate_15min.py --data_dir ../../00_Mini_Model/data/ohlcv --workers 24
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
 
RESOLUTION    = "15min"
OUT           = THIS / "signals"
RSI_PERIODS   = [7, 10, 14, 21]
DIV_LOOKBACKS = [5, 10, 20]
SCALE         = 0.5
N_CORES       = min(cpu_count(), 24)
 
 
def wilder_rsi_flex(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    ag = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    al = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)
 
 
def measure_divergence(rsi: pd.Series, close: pd.Series, lookback: int) -> pd.Series:
    price_min = close.rolling(lookback, min_periods=lookback).min()
    price_max = close.rolling(lookback, min_periods=lookback).max()
    price_rank = (close - price_min) / (price_max - price_min).replace(0, np.nan)
 
    rsi_min = rsi.rolling(lookback, min_periods=lookback).min()
    rsi_max = rsi.rolling(lookback, min_periods=lookback).max()
    rsi_rank = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
 
    return (rsi_rank - price_rank).fillna(0)
 
 
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
 
    if len(df) < 60:
        return None
 
    close = df["close"]
    rows = []
 
    for period in RSI_PERIODS:
        rsi = wilder_rsi_flex(close, period)
        for lookback in DIV_LOOKBACKS:
            div = measure_divergence(rsi, close, lookback)
            bull = np.clip(np.tanh(div.values / SCALE), 0, 1).astype(np.float32)
            bear = np.clip(np.tanh(-div.values / SCALE), 0, 1).astype(np.float32)
            combo_id = f"p{period}_lb{lookback}"
 
            combo_df = pd.DataFrame({
                "datetime": df["datetime"].values,
                "ticker": ticker,
                "combo_id": combo_id,
                "rsi_period": np.int8(period),
                "div_lookback": np.int8(lookback),
                "bull_rsi_score": bull,
                "bear_rsi_score": bear,
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
    p = argparse.ArgumentParser(description="V04 RSI Divergence — 15min Generator")
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
 
    logging.info(f"V04 RSI 15min | Tickers: {len(tickers)} | Grid: 12 combos")
 
    jobs = [(args.data_dir, tk) for tk in tickers]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.map(process_ticker, jobs, chunksize=16)
    else:
        results = [process_ticker(j) for j in jobs]
 
    ok = [r for r in results if r is not None]
    logging.info(f"✓ 15min signals: {len(ok)}/{len(tickers)} tickers → {OUT}/")
 
 
if __name__ == "__main__":
    main()