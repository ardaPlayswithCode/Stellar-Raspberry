"""
generate_15min.py — Var 06 Volume Trend (15-Minute Resolution)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dollar volume surprise at LCM master clock. 18 combos.
GKM (2001): volume shock. Hendershott (2011): dollar volume.
Lookback 60 bars = 15 hours market time.
 
USAGE:
  python generate_15min.py --data_dir ../../00_Mini_Model/data/ohlcv --workers 24
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
 
RESOLUTION = "15min"
OUT        = THIS / "signals"
LOOKBACKS  = [5, 10, 20]
NORMS      = ["z_score", "market_ratio", "rank"]
EWM_FLAGS  = [True, False]
N_CORES    = min(cpu_count(), 24)
 
 
def compute_vol_signal(dollar_vol, lookback, norm, use_ewm):
    if use_ewm:
        mu = dollar_vol.ewm(span=lookback, min_periods=lookback, adjust=False).mean()
        sigma = dollar_vol.ewm(span=lookback, min_periods=lookback, adjust=False).std()
    else:
        mu = dollar_vol.rolling(lookback, min_periods=lookback).mean()
        sigma = dollar_vol.rolling(lookback, min_periods=lookback).std()
    if norm == "z_score":
        return ((dollar_vol - mu) / sigma.replace(0, np.nan)).fillna(0)
    elif norm == "market_ratio":
        return ((dollar_vol / mu.replace(0, np.nan)).fillna(1) - 1.0)
    elif norm == "rank":
        return (dollar_vol.rolling(lookback, min_periods=lookback).rank(pct=True) * 2 - 1).fillna(0)
    return pd.Series(0, index=dollar_vol.index)
 
 
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
    if len(df) < 60:
        return None
 
    dollar_vol = (df["close"] * df["volume"]).astype(np.float64)
    rows = []
    for lb in LOOKBACKS:
        for norm in NORMS:
            for ewm in EWM_FLAGS:
                signal = compute_vol_signal(dollar_vol, lb, norm, ewm)
                bull = np.clip(np.tanh(signal.values), 0, 1).astype(np.float32)
                bear = np.clip(np.tanh(-signal.values), 0, 1).astype(np.float32)
                vol_roc = np.tanh(signal.diff().fillna(0).values).astype(np.float32)
                combo_df = pd.DataFrame({
                    "datetime": df["datetime"].values, "ticker": ticker,
                    "combo_id": f"lb{lb}_{norm}_{'ewm' if ewm else 'sma'}",
                    "lookback": np.int8(lb), "norm_method": norm, "use_ewm": ewm,
                    "bull_vol_score": bull, "bear_vol_score": bear, "vol_roc": vol_roc,
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
    p = argparse.ArgumentParser(description="V06 Volume Trend — 15min Generator")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--workers", type=int, default=N_CORES)
    p.add_argument("--tickers", nargs="*", default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    tickers = args.tickers or sorted(
        p.stem for p in (Path(args.data_dir) / RESOLUTION).glob("*.parquet")
        if not p.stem.startswith("_"))
    logging.info(f"V06 Volume 15min | Tickers: {len(tickers)} | Combos: 18")
    jobs = [(args.data_dir, tk) for tk in tickers]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.map(process_ticker, jobs, chunksize=16)
    else:
        results = [process_ticker(j) for j in jobs]
    ok = [r for r in results if r is not None]
    logging.info(f"✓ 15min signals: {len(ok)}/{len(tickers)} tickers → {OUT}/")
 
 
if __name__ == "__main__":
    main()