"""
01_Stock_Geometry/V1/weekly/generate_weekly.py
===============================================
Weekly (W-FRI) stock chart generator. Production architecture.
 
Source: data/ohlcv/weekly/{TICKER}.parquet (W-FRI resampled by 00_ingest)
  52-bar lookback = 1 year of weekly bars per image.
  J&T (1993) JF 48(1): 12-1 month skip-momentum visible at this scale.
 
Rendering: 680×680 4-pane. Same indicator math as 15min/1hour/daily.
  The CNN learns weekly structural patterns (cup-and-handle, bases, etc.)
  that are invisible at intraday resolution.
 
Output: 01_Stock_Geometry/V1/weekly/images/{TICKER}/{YYYY-MM-DD}.png
  Date = Friday close date (W-FRI anchor). Forward-filled by registry.
"""
 
import logging, argparse
from pathlib import Path
import numpy as np, pandas as pd
 
import importlib as _il
B = _il.import_module("01_base_modules_")
 
THIS     = Path(__file__).resolve().parent
OUT      = THIS / "images"
LB       = B.LOOKBACK_MAP["weekly"]
WARMUP   = max(B.MACD_S + B.MACD_SIG, B.RSI_P, LB)
 
 
def load_ticker(data_dir: str, ticker: str) -> pd.DataFrame:
    p = Path(data_dir) / "weekly" / f"{ticker}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    col = "datetime" if "datetime" in df.columns else df.columns[0]
    df["datetime"] = pd.to_datetime(df[col])
    return df.sort_values("datetime").reset_index(drop=True)
 
 
def build_jobs(df: pd.DataFrame, ticker: str, existing: set) -> list:
    rsi  = B.wilder_rsi(df["close"]).values
    hist = B.macd_hist(df["close"]).values
    jobs = []
    for i in range(WARMUP, len(df)):
        dt = df["datetime"].iloc[i]
        path_str = str(OUT / ticker / f"{B.fmt_date(dt)}.png")
        if path_str in existing:
            continue
        s = slice(i - LB + 1, i + 1)
        od = B._ohlcv_dict(df.iloc[s])
        jobs.append((od, rsi[s], hist[s], path_str))
    return jobs
 
 
def generate(args):
    data_dir = args.data_dir
    existing = B.build_image_set(str(OUT)) if OUT.exists() else set()
    logging.info(f"V1 Stock Weekly | LB={LB} | Existing: {len(existing):,}")
 
    universe_path = Path(data_dir) / "_universe.parquet"
    if universe_path.exists():
        tickers = pd.read_parquet(universe_path)["ticker"].tolist()
    else:
        tickers = sorted(p.stem for p in (Path(data_dir) / "weekly").glob("*.parquet"))
 
    all_jobs = []
    loaded = 0
    for idx, tk in enumerate(tickers):
        df = load_ticker(data_dir, tk)
        if df.empty or len(df) < WARMUP + 10:
            continue
        all_jobs.extend(build_jobs(df, tk, existing))
        loaded += 1
        if (idx + 1) % 100 == 0:
            logging.info(f"  [{idx+1}/{len(tickers)}] loaded={loaded} jobs={len(all_jobs):,}")
 
    logging.info(f"Total jobs: {len(all_jobs):,} | {loaded} tickers")
    if all_jobs:
        B.render_batch_4p(all_jobs, n_workers=args.workers)
    logging.info(f"✓ V1 Weekly complete → {OUT}")
 
 
def main():
    p = argparse.ArgumentParser(description="V1 Stock Geometry — Weekly Generator")
    p.add_argument("--data_dir", default="data/ohlcv")
    p.add_argument("--workers", type=int, default=B.N_CORES)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    generate(args)
 
 
if __name__ == "__main__":
    main()