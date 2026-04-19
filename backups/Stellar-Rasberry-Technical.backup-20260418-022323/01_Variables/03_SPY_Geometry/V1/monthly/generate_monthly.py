"""
03_SPY_Geometry/V1/monthly/generate_monthly.py
================================================
Monthly (ME) SPY + VIX charts. L3 Bear Filter — FULL CASH.
 
Source: data/ohlcv/monthly/SPY.parquet, data/ohlcv/monthly/^VIX.parquet
  12-bar lookback = 1 year of monthly bars.
 
Bear Filter Hierarchy (the nuclear chain):
  L1 (Stock Monthly):  per-ticker position reduction
  L2 (Sector Monthly): per-sector position reduction
  L3 (SPY Monthly):    if CNN → bearish, OVERRIDE ALL SIGNALS → FULL CASH
       This is the "circuit breaker." Monthly SPY bearish means the
       macro environment is hostile. No individual stock or sector
       signal can override this. Trading Dynamics receives confidence=0.0
       for ALL tickers when the L3 flag is active.
 
Rendering:
  SPY: 680×680 2-pane (candlestick + volume). Monthly bars show
       major distribution/accumulation phases visible only at this scale.
  VIX: 680×680 line chart. Monthly VIX peaks correspond to crisis events
       (2008, 2020 March, 2022 rate shock).
 
Output: 03_SPY_Geometry/V1/monthly/{images_spy,images_vix}/{YYYY-MM-DD}.png
"""
 
import logging, argparse
from pathlib import Path
import numpy as np, pandas as pd
 
import importlib as _il
B = _il.import_module("01_base_modules_")
 
THIS    = Path(__file__).resolve().parent
SPY_OUT = THIS / "images_spy"
VIX_OUT = THIS / "images_vix"
LB      = B.LOOKBACK_MAP["monthly"]
 
 
def load_instrument(data_dir, symbol):
    for name in [symbol, symbol.replace("^", "")]:
        p = Path(data_dir) / "monthly" / f"{name}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            col = "datetime" if "datetime" in df.columns else df.columns[0]
            df["datetime"] = pd.to_datetime(df[col])
            return df.sort_values("datetime").reset_index(drop=True)
    return pd.DataFrame()
 
 
def generate(args):
    spy_df = load_instrument(args.data_dir, "SPY")
    vix_df = load_instrument(args.data_dir, "^VIX")
    if spy_df.empty or vix_df.empty:
        logging.error("Missing monthly SPY or VIX."); return
 
    common = pd.DatetimeIndex(spy_df["datetime"]).intersection(
        pd.DatetimeIndex(vix_df["datetime"]))
    spy_df = spy_df.set_index("datetime").loc[common].reset_index()
    vix_df = vix_df.set_index("datetime").loc[common].reset_index()
    logging.info(f"V3 SPY Monthly (L3 FULL CASH) | Common: {len(common):,}")
 
    spy_existing = B.build_image_set(str(SPY_OUT)) if SPY_OUT.exists() else set()
    vix_existing = B.build_image_set(str(VIX_OUT)) if VIX_OUT.exists() else set()
 
    spy_jobs, vix_jobs = [], []
    for i in range(LB, len(spy_df)):
        dt = spy_df["datetime"].iloc[i]
        s = slice(i - LB + 1, i + 1)
        fname = f"{B.fmt_date(dt)}.png"
        sp = str(SPY_OUT / fname)
        if sp not in spy_existing:
            spy_jobs.append((B._ohlcv_dict(spy_df.iloc[s]), sp))
        vp = str(VIX_OUT / fname)
        if vp not in vix_existing:
            vix_jobs.append((vix_df["close"].iloc[s].values, vp))
 
    logging.info(f"  New: SPY={len(spy_jobs):,} VIX={len(vix_jobs):,}")
    if spy_jobs: B.render_batch_2p(spy_jobs, n_workers=args.workers)
    if vix_jobs: B.render_batch_vix(vix_jobs, n_workers=args.workers)
    logging.info(f"✓ V3 Monthly (L3) complete")
    logging.info(f"  REMINDER: monthly bearish → confidence=0.0 for ALL tickers")
 
 
def main():
    p = argparse.ArgumentParser(description="V3 SPY/VIX — Monthly (L3 FULL CASH)")
    p.add_argument("--data_dir", default="data/ohlcv")
    p.add_argument("--workers", type=int, default=B.N_CORES)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    generate(args)
 
if __name__ == "__main__":
    main()