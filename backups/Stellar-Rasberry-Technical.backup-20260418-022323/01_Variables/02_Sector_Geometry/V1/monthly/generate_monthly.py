"""
02_Sector_Geometry/V1/monthly/generate_monthly.py
===================================================
Monthly (ME) sector ETF charts. L2 Bear Filter.
 
Source: data/ohlcv/monthly/{ETF}.parquet (ME resampled by 00_ingest)
  12-bar lookback = 1 year of monthly sector bars.
 
Rendering: 680×680 2-pane (candlestick + volume). No RSI panel.
  Monthly sector geometry captures macro distribution/accumulation
  patterns. Pure price structure for the CNN.
 
Bear Filter Hierarchy:
  L1 (Stock Monthly): per-ticker position reduction
  L2 (Sector Monthly): if sector CNN → bearish, reduce ALL constituent positions
  L3 (SPY Monthly): FULL CASH override
 
Output: 02_Sector_Geometry/V1/monthly/images/{ETF}/{YYYY-MM-DD}.png
"""
 
import logging, argparse
from pathlib import Path
import numpy as np, pandas as pd
 
import importlib as _il
B = _il.import_module("01_base_modules_")
 
THIS = Path(__file__).resolve().parent
OUT  = THIS / "images"
LB   = B.LOOKBACK_MAP["monthly"]
 
 
def load_etf(data_dir, etf):
    p = Path(data_dir) / "monthly" / f"{etf}.parquet"
    if not p.exists(): return pd.DataFrame()
    df = pd.read_parquet(p)
    col = "datetime" if "datetime" in df.columns else df.columns[0]
    df["datetime"] = pd.to_datetime(df[col])
    return df.sort_values("datetime").reset_index(drop=True)
 
 
def build_jobs(df, etf, existing):
    jobs = []
    for i in range(LB, len(df)):
        dt = df["datetime"].iloc[i]
        path_str = str(OUT / etf / f"{B.fmt_date(dt)}.png")
        if path_str in existing: continue
        s = slice(i - LB + 1, i + 1)
        jobs.append((B._ohlcv_dict(df.iloc[s]), path_str))
    return jobs
 
 
def generate(args):
    existing = B.build_image_set(str(OUT)) if OUT.exists() else set()
    logging.info(f"V2 Sector Monthly (L2 Bear) | Existing: {len(existing):,}")
    all_jobs = []
    for etf in B.SECTOR_ETFS:
        df = load_etf(args.data_dir, etf)
        if df.empty or len(df) < LB + 3: continue
        jobs = build_jobs(df, etf, existing)
        all_jobs.extend(jobs)
        logging.info(f"  {etf}: {len(jobs):,}")
    logging.info(f"Total: {len(all_jobs):,}")
    if all_jobs:
        B.render_batch_2p(all_jobs, n_workers=args.workers)
    logging.info(f"✓ V2 Monthly → {OUT}")
 
 
def main():
    p = argparse.ArgumentParser(description="V2 Sector — Monthly (L2 Bear)")
    p.add_argument("--data_dir", default="data/ohlcv")
    p.add_argument("--workers", type=int, default=B.N_CORES)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    generate(args)
 
if __name__ == "__main__":
    main()