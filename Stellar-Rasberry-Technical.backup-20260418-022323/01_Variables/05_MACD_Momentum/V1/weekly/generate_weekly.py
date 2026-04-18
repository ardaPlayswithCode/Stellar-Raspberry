"""
generate_weekly.py — Var 05 MACD Momentum (Weekly Resolution)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS DOES:
  MACD momentum scores at W-FRI resolution. 24 valid parameter combos.
  MOP (2012) normalization, DHS (1998) derivative as fixed column.
 
CONSISTENCY:
  - Standard EMA (NOT Wilder) — Appel/Neftci convention
  - FETCH_START = "2004-01-01" (weekly needs extra warmup year)
  - W-FRI anchor, local parquet only, zstd, vectorized
 
USAGE:
  python generate_weekly.py --data_dir ../../00_Mini_Model/data/ohlcv --workers 24
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
 
RESOLUTION  = "weekly"
FETCH_START = "2004-01-01"
OUT         = THIS / "signals"
FAST_GRID   = [8, 12, 16]
SLOW_GRID   = [21, 26, 34]
SIG_GRID    = [7, 9, 12]
N_CORES     = min(cpu_count(), 24)
 
 
def macd_histogram(close, fast, slow, sig):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    ml = ef - es
    return ml - ml.ewm(span=sig, adjust=False).mean()
 
 
def valid_combos():
    return [(f, s, g) for f in FAST_GRID for s in SLOW_GRID
            for g in SIG_GRID if s - f >= 8]
 
 
COMBOS = valid_combos()
 
 
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
    if len(df) < 40:
        return None
 
    close = df["close"]
    atr = B.wilder_atr(df["high"], df["low"], close)
    atr_safe = atr.replace(0, np.nan)
    rows = []
 
    for fast, slow, sig in COMBOS:
        hist = macd_histogram(close, fast, slow, sig)
        hist_norm = (hist / atr_safe).fillna(0)
        bull = np.clip(np.tanh(hist_norm.values), 0, 1).astype(np.float32)
        bear = np.clip(np.tanh(-hist_norm.values), 0, 1).astype(np.float32)
        hist_delta_norm = np.tanh((hist.diff() / atr_safe).fillna(0).values).astype(np.float32)
 
        combo_df = pd.DataFrame({
            "datetime": df["datetime"].values,
            "ticker": ticker,
            "combo_id": f"f{fast}_s{slow}_g{sig}",
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
    p = argparse.ArgumentParser(description="V05 MACD Momentum — Weekly Generator")
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
 
    logging.info(f"V05 MACD Weekly | Tickers: {len(tickers)} | Combos: {len(COMBOS)}")
    jobs = [(args.data_dir, tk) for tk in tickers]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.map(process_ticker, jobs, chunksize=16)
    else:
        results = [process_ticker(j) for j in jobs]
    ok = [r for r in results if r is not None]
    logging.info(f"✓ Weekly signals: {len(ok)}/{len(tickers)} tickers → {OUT}/")
 
 
if __name__ == "__main__":
    main()