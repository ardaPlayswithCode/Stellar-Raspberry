"""
03_SPY_Geometry/V1/1hour/generate_1hour.py
============================================
1-hour SPY + VIX charts. Production architecture.
 
Dual Output:
  images_spy/{YYYY-MM-DD_HHMM}.png — SPY 2-pane
  images_vix/{YYYY-MM-DD_HHMM}.png — VIX line
 
Source: data/ohlcv/1hour/SPY.parquet, data/ohlcv/1hour/^VIX.parquet
  60-bar lookback ≈ 8.6 trading days.
  The 1hour SPY chart is the "Market Structural Horizon" —
  defines whether the broad market environment supports long entries.
 
Adrian & Brunnermeier (2016) AER 106(7): at this scale,
  the VIX CNN detects the volatility paradox — low VIX regimes
  where hidden systemic risk accumulates before rapid expansion.
 
Output: 03_SPY_Geometry/V1/1hour/{images_spy,images_vix}/{YYYY-MM-DD_HHMM}.png
"""
 
import logging, argparse
from pathlib import Path
import numpy as np, pandas as pd
 
import importlib as _il
B = _il.import_module("01_base_modules_")
 
THIS    = Path(__file__).resolve().parent
SPY_OUT = THIS / "images_spy"
VIX_OUT = THIS / "images_vix"
LB      = B.LOOKBACK_MAP["1hour"]
 
 
def load_instrument(data_dir, symbol):
    for name in [symbol, symbol.replace("^", "")]:
        p = Path(data_dir) / "1hour" / f"{name}.parquet"
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
        logging.error("Missing 1hour SPY or VIX parquet."); return
 
    common = pd.DatetimeIndex(spy_df["datetime"]).intersection(
        pd.DatetimeIndex(vix_df["datetime"]))
    spy_df = spy_df.set_index("datetime").loc[common].reset_index()
    vix_df = vix_df.set_index("datetime").loc[common].reset_index()
    logging.info(f"V3 SPY 1hour | Common: {len(common):,}")
 
    spy_existing = B.build_image_set(str(SPY_OUT)) if SPY_OUT.exists() else set()
    vix_existing = B.build_image_set(str(VIX_OUT)) if VIX_OUT.exists() else set()
 
    spy_jobs, vix_jobs = [], []
    for i in range(LB, len(spy_df)):
        dt = spy_df["datetime"].iloc[i]
        s = slice(i - LB + 1, i + 1)
        fname = f"{B.fmt_15m(dt)}.png"
        sp = str(SPY_OUT / fname)
        if sp not in spy_existing:
            spy_jobs.append((B._ohlcv_dict(spy_df.iloc[s]), sp))
        vp = str(VIX_OUT / fname)
        if vp not in vix_existing:
            vix_jobs.append((vix_df["close"].iloc[s].values, vp))
 
    logging.info(f"  New: SPY={len(spy_jobs):,} VIX={len(vix_jobs):,}")
    if spy_jobs: B.render_batch_2p(spy_jobs, n_workers=args.workers)
    if vix_jobs: B.render_batch_vix(vix_jobs, n_workers=args.workers)
    logging.info(f"✓ V3 1hour complete")
 
 
def main():
    p = argparse.ArgumentParser(description="V3 SPY/VIX — 1hour")
    p.add_argument("--data_dir", default="data/ohlcv")
    p.add_argument("--workers", type=int, default=B.N_CORES)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    generate(args)
 
if __name__ == "__main__":
    main()