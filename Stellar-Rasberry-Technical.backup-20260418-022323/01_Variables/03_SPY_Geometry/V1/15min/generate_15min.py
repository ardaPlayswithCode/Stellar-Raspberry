"""
03_SPY_Geometry/V1/15min/generate_15min.py
============================================
15-minute SPY + VIX chart generator. Production architecture.
 
Dual Output:
  images_spy/{YYYY-MM-DD_HHMM}.png — SPY candlestick + volume (2-pane)
  images_vix/{YYYY-MM-DD_HHMM}.png — VIX line chart (raw index level)
 
Source: data/ohlcv/15min/SPY.parquet, data/ohlcv/15min/^VIX.parquet
  (from 00_ingest_historical.py)
 
GLOBAL: these images are the SAME for every ticker on a given timestamp.
  The registry maps all 500+ tickers to the same SPY/VIX image per date.
  This is the "market pulse" — systematic risk context for all positions.
 
Rendering:
  SPY: 680×680 2-pane (candlestick + volume). Panel ratios [3, 1].
       No RSI/MACD — the CNN learns pure market structure.
  VIX: 680×680 single-pane line chart with fill.
       Orange (#e67e22) line, 15% alpha fill.
       Reference lines at VIX=20 (normal), VIX=30 (elevated).
       Whaley (2000) JPM 26(3): VIX = forward-looking 30-day implied vol.
 
60-bar lookback = 15 hours of market data.
"""
 
import logging, argparse
from pathlib import Path
import numpy as np, pandas as pd
 
import importlib as _il
B = _il.import_module("01_base_modules_")
 
THIS    = Path(__file__).resolve().parent
SPY_OUT = THIS / "images_spy"
VIX_OUT = THIS / "images_vix"
LB      = B.LOOKBACK_MAP["15min"]
 
 
def load_instrument(data_dir: str, symbol: str) -> pd.DataFrame:
    for name in [symbol, symbol.replace("^", "")]:
        p = Path(data_dir) / "15min" / f"{name}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["datetime"] = pd.to_datetime(df["datetime"])
            return df.sort_values("datetime").reset_index(drop=True)
    return pd.DataFrame()
 
 
def generate(args):
    spy_df = load_instrument(args.data_dir, "SPY")
    vix_df = load_instrument(args.data_dir, "^VIX")
 
    if spy_df.empty or vix_df.empty:
        logging.error("SPY or VIX 15min parquet not found. Run 00_ingest first.")
        return
 
    spy_idx = pd.DatetimeIndex(spy_df["datetime"])
    vix_idx = pd.DatetimeIndex(vix_df["datetime"])
    common = spy_idx.intersection(vix_idx)
    logging.info(f"V3 SPY 15min | SPY: {len(spy_df):,} | VIX: {len(vix_df):,} | "
                 f"Common: {len(common):,} | LB={LB}")
 
    spy_df = spy_df.set_index("datetime").loc[common].reset_index()
    vix_df = vix_df.set_index("datetime").loc[common].reset_index()
 
    spy_existing = B.build_image_set(str(SPY_OUT)) if SPY_OUT.exists() else set()
    vix_existing = B.build_image_set(str(VIX_OUT)) if VIX_OUT.exists() else set()
    logging.info(f"  Existing: SPY={len(spy_existing):,} VIX={len(vix_existing):,}")
 
    spy_jobs = []
    vix_jobs = []
 
    for i in range(LB, len(spy_df)):
        dt = spy_df["datetime"].iloc[i]
        s = slice(i - LB + 1, i + 1)
        fname = f"{B.fmt_15m(dt)}.png"
 
        spy_path = str(SPY_OUT / fname)
        if spy_path not in spy_existing:
            spy_jobs.append((B._ohlcv_dict(spy_df.iloc[s]), spy_path))
 
        vix_path = str(VIX_OUT / fname)
        if vix_path not in vix_existing:
            vix_vals = vix_df["close"].iloc[s].values
            vix_jobs.append((vix_vals, vix_path))
 
    logging.info(f"  New: SPY={len(spy_jobs):,} VIX={len(vix_jobs):,}")
 
    if spy_jobs:
        B.render_batch_2p(spy_jobs, n_workers=args.workers)
    if vix_jobs:
        B.render_batch_vix(vix_jobs, n_workers=args.workers)
 
    logging.info(f"✓ V3 15min → {SPY_OUT}\n               {VIX_OUT}")
 
 
def main():
    p = argparse.ArgumentParser(description="V3 SPY/VIX — 15min Generator")
    p.add_argument("--data_dir", default="data/ohlcv")
    p.add_argument("--workers", type=int, default=B.N_CORES)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    generate(args)
 
 
if __name__ == "__main__":
    main()