"""
01_Stock_Geometry/V1/daily/generate_daily.py
=============================================
Daily stock chart generator. Production architecture.
 
Source: data/ohlcv/daily/{TICKER}.parquet (resampled by 00_ingest)
  60-bar lookback = ~3 calendar months of daily bars per image.
 
Rendering: 680×680 4-pane. Physics identical across all 5 resolutions.
Output: 01_Stock_Geometry/V1/daily/images/{TICKER}/{YYYY-MM-DD}.png
"""
 
import logging, argparse
from pathlib import Path
import numpy as np, pandas as pd
 
import importlib as _il
B = _il.import_module("01_base_modules_")
 
THIS     = Path(__file__).resolve().parent
OUT      = THIS / "images"
LB       = B.LOOKBACK_MAP["daily"]
WARMUP   = max(B.MACD_S + B.MACD_SIG, B.RSI_P, LB)
 
 
def load_ticker(data_dir: str, ticker: str) -> pd.DataFrame:
    p = Path(data_dir) / "daily" / f"{ticker}.parquet"
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
    logging.info(f"V1 Stock Daily | LB={LB} | Existing: {len(existing):,}")
 
    universe_path = Path(data_dir) / "_universe.parquet"
    if universe_path.exists():
        tickers = pd.read_parquet(universe_path)["ticker"].tolist()
    else:
        tickers = sorted(p.stem for p in (Path(data_dir) / "daily").glob("*.parquet"))
 
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
    logging.info(f"✓ V1 Daily complete → {OUT}")
 
 
def main():
    p = argparse.ArgumentParser(description="V1 Stock Geometry — Daily Generator")
    p.add_argument("--data_dir", default="data/ohlcv")
    p.add_argument("--workers", type=int, default=B.N_CORES)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    generate(args)
 
 
if __name__ == "__main__":
    main()