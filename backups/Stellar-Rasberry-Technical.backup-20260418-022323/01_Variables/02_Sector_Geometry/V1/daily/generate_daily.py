"""
02_Sector_Geometry/V1/daily/generate_daily.py
===============================================
Daily sector ETF charts. Production architecture.
 
Source: data/ohlcv/daily/{ETF}.parquet
  60-bar lookback = ~3 months of daily sector bars.
 
GICS Reclassification (point-in-time accuracy):
  XLC inception: 2018-06-18. Data before this date does not exist.
  XLRE inception: 2015-10-08. Data before this date does not exist.
  00_ingest_historical.py handles these gaps — if an ETF's parquet
  starts later than 2005, the generator simply produces fewer images.
  The registry forward-fills to cover the gap.
 
Output: 02_Sector_Geometry/V1/daily/images/{ETF}/{YYYY-MM-DD}.png
"""
 
import logging, argparse
from pathlib import Path
import numpy as np, pandas as pd
 
import importlib as _il
B = _il.import_module("01_base_modules_")
 
THIS     = Path(__file__).resolve().parent
OUT      = THIS / "images"
LB       = B.LOOKBACK_MAP["daily"]
WARMUP   = max(B.RSI_P, LB)
 
 
def load_etf(data_dir, etf):
    p = Path(data_dir) / "daily" / f"{etf}.parquet"
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
    logging.info(f"V2 Sector Daily | GICS-aware | Existing: {len(existing):,}")
    all_jobs = []
    for etf in B.SECTOR_ETFS:
        df = load_etf(args.data_dir, etf)
        if df.empty or len(df) < WARMUP + 10:
            logging.info(f"  [SKIP] {etf}")
            continue
        jobs = build_jobs(df, etf, existing)
        all_jobs.extend(jobs)
        start = df["datetime"].iloc[0].date()
        logging.info(f"  {etf}: {len(jobs):,} new (data from {start})")
    logging.info(f"Total: {len(all_jobs):,}")
    if all_jobs:
        B.render_batch_3p(all_jobs, n_workers=args.workers)
    logging.info(f"✓ V2 Daily → {OUT}")
 
 
def main():
    p = argparse.ArgumentParser(description="V2 Sector — Daily")
    p.add_argument("--data_dir", default="data/ohlcv")
    p.add_argument("--workers", type=int, default=B.N_CORES)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    generate(args)
 
if __name__ == "__main__":
    main()