"""
02_Sector_Geometry/V1/15min/generate_15min.py
===============================================
15-minute sector ETF chart generator. Production architecture.
 
Source: data/ohlcv/15min/{ETF}.parquet (from 00_ingest_historical.py)
  11 SPDR sector ETFs: XLK XLF XLV XLY XLC XLI XLE XLP XLRE XLU XLB
 
Rendering: 680×680 3-pane (candlestick + volume + RSI)
  Panel ratios [6.5, 2, 1.5]. No MACD panel — sector charts are
  structurally smoother than individual stocks; RSI alone captures
  overbought/oversold for the CNN without redundant oscillator noise.
 
  60-bar lookback = 15 hours of 15m sector bars.
 
GICS Reclassification:
  XLC (Communication Services) created 2018-09-24. Prior: tickers in XLK/XLY.
  XLRE (Real Estate) created 2015-10-07. Prior: tickers in XLF.
  00_ingest handles data availability; generators only render what exists.
 
Self-Reference: NOT handled here (rendering is ETF-level, not ticker-level).
  Self-ref masking is enforced in trainer.py at the per-sample loss level.
 
Output: 02_Sector_Geometry/V1/15min/images/{ETF}/{YYYY-MM-DD_HHMM}.png
"""
 
import logging, argparse
from pathlib import Path
import numpy as np, pandas as pd
 
import importlib as _il
B = _il.import_module("01_base_modules_")
 
THIS     = Path(__file__).resolve().parent
OUT      = THIS / "images"
LB       = B.LOOKBACK_MAP["15min"]
WARMUP   = max(B.RSI_P, LB)
 
 
def load_etf(data_dir: str, etf: str) -> pd.DataFrame:
    p = Path(data_dir) / "15min" / f"{etf}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)
 
 
def build_jobs(df: pd.DataFrame, etf: str, existing: set) -> list:
    rsi = B.wilder_rsi(df["close"]).values
    jobs = []
    for i in range(WARMUP, len(df)):
        dt = df["datetime"].iloc[i]
        path_str = str(OUT / etf / f"{B.fmt_15m(dt)}.png")
        if path_str in existing:
            continue
        s = slice(i - LB + 1, i + 1)
        od = B._ohlcv_dict(df.iloc[s])
        jobs.append((od, rsi[s], path_str))
    return jobs
 
 
def generate(args):
    existing = B.build_image_set(str(OUT)) if OUT.exists() else set()
    logging.info(f"V2 Sector 15min | 11 ETFs | LB={LB} | Existing: {len(existing):,}")
 
    all_jobs = []
    for etf in B.SECTOR_ETFS:
        df = load_etf(args.data_dir, etf)
        if df.empty or len(df) < WARMUP + 10:
            logging.info(f"  [SKIP] {etf}: insufficient data")
            continue
        jobs = build_jobs(df, etf, existing)
        all_jobs.extend(jobs)
        logging.info(f"  {etf}: {len(jobs):,} new images")
 
    logging.info(f"Total jobs: {len(all_jobs):,}")
    if all_jobs:
        B.render_batch_3p(all_jobs, n_workers=args.workers)
    logging.info(f"✓ V2 15min complete → {OUT}")
 
 
def main():
    p = argparse.ArgumentParser(description="V2 Sector Geometry — 15min Generator")
    p.add_argument("--data_dir", default="data/ohlcv")
    p.add_argument("--workers", type=int, default=B.N_CORES)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    generate(args)
 
 
if __name__ == "__main__":
    main()