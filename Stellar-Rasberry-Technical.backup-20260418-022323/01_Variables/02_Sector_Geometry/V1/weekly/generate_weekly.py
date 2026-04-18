"""
02_Sector_Geometry/V1/weekly/generate_weekly.py
=================================================
Weekly (W-FRI) sector ETF charts. Production architecture.
 
Source: data/ohlcv/weekly/{ETF}.parquet (W-FRI resampled by 00_ingest)
  52-bar lookback = 1 year of weekly sector bars.
 
Output: 02_Sector_Geometry/V1/weekly/images/{ETF}/{YYYY-MM-DD}.png
"""
 
import logging, argparse
from pathlib import Path
import numpy as np, pandas as pd
 
import importlib as _il
B = _il.import_module("01_base_modules_")
 
THIS   = Path(__file__).resolve().parent
OUT    = THIS / "images"
LB     = B.LOOKBACK_MAP["weekly"]
WARMUP = max(B.RSI_P, LB)
 
 
def load_etf(data_dir, etf):
    p = Path(data_dir) / "weekly" / f"{etf}.parquet"
    if not p.exists(): return pd.DataFrame()
    df = pd.read_parquet(p)
    col = "datetime" if "datetime" in df.columns else df.columns[0]
    df["datetime"] = pd.to_datetime(df[col])
    return df.sort_values("datetime").reset_index(drop=True)
 
 
def build_jobs(df, etf, existing):
    rsi = B.wilder_rsi(df["close"]).values
    jobs = []
    for i in range(WARMUP, len(df)):
        dt = df["datetime"].iloc[i]
        path_str = str(OUT / etf / f"{B.fmt_date(dt)}.png")
        if path_str in existing: continue
        s = slice(i - LB + 1, i + 1)
        jobs.append((B._ohlcv_dict(df.iloc[s]), rsi[s], path_str))
    return jobs
 
 
def generate(args):
    existing = B.build_image_set(str(OUT)) if OUT.exists() else set()
    logging.info(f"V2 Sector Weekly | Existing: {len(existing):,}")
    all_jobs = []
    for etf in B.SECTOR_ETFS:
        df = load_etf(args.data_dir, etf)
        if df.empty or len(df) < WARMUP + 10: continue
        jobs = build_jobs(df, etf, existing)
        all_jobs.extend(jobs)
        logging.info(f"  {etf}: {len(jobs):,}")
    logging.info(f"Total: {len(all_jobs):,}")
    if all_jobs:
        B.render_batch_3p(all_jobs, n_workers=args.workers)
    logging.info(f"✓ V2 Weekly → {OUT}")
 
 
def main():
    p = argparse.ArgumentParser(description="V2 Sector — Weekly")
    p.add_argument("--data_dir", default="data/ohlcv")
    p.add_argument("--workers", type=int, default=B.N_CORES)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    generate(args)
 
if __name__ == "__main__":
    main()