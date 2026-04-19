"""
generate_daily.py — Var 06 Volume Trend (Daily Resolution)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS DOES:
  Computes volume SHOCK scores (dollar volume surprise vs baseline).
  Grid: lookback[5,10,20] × norm[z_score,market_ratio,rank] × ewm[T/F] = 18 combos.
  Fixed feature: vol_roc = tanh(Δ vol_zscore) — GKM acceleration.
 
RESEARCH:
  GKM (2001) JF 56(3) — Volume SHOCK predicts returns 1-20 days.
    NOT absolute volume — the unexpected component. Dollar volume canonical.
    Effect strongest when |same_bar_return| rank < 30th pctl (normal bars).
  Chordia/Roll/Subrahmanyam (2008) JFE 87(2) — AR(12) innovation only.
    Raw volume IC ≈ 0 in modern markets; vol-surprise IC survives.
  Hendershott/Jones/Menkveld (2011) JF 66(1) — AT regime shift Jan 2003.
    Dollar volume preferred over share count.
 
CONSISTENCY:
  - dollar_volume = close × volume (Hendershott 2011)
  - ATR: Wilder EWM span=14 from 01_base_modules_
  - FETCH_START = "2005-01-01"
  - Local parquet only, zstd, vectorized, np.float32
 
OUTPUT:
  signals/{TICKER}.parquet → [datetime, ticker, combo_id, lookback,
    norm_method, use_ewm, bull_vol_score, bear_vol_score, vol_roc]
 
USAGE:
  python generate_daily.py --data_dir ../../00_Mini_Model/data/ohlcv --workers 24
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
 
RESOLUTION  = "daily"
FETCH_START = "2005-01-01"
OUT         = THIS / "signals"
LOOKBACKS   = [5, 10, 20]
NORMS       = ["z_score", "market_ratio", "rank"]
EWM_FLAGS   = [True, False]
N_CORES     = min(cpu_count(), 24)
 
 
def compute_vol_signal(dollar_vol: pd.Series, lookback: int,
                       norm: str, use_ewm: bool) -> pd.Series:
    """
    Volume surprise score. Three normalization methods:
      z_score:      (dv - μ) / σ  over rolling window
      market_ratio: dv / rolling_mean  (ratio to baseline)
      rank:         rolling percentile rank ∈ [0,1] centered to [-1,1]
    EWM option: exponential vs simple rolling window.
    """
    if use_ewm:
        mu = dollar_vol.ewm(span=lookback, min_periods=lookback, adjust=False).mean()
        sigma = dollar_vol.ewm(span=lookback, min_periods=lookback, adjust=False).std()
    else:
        mu = dollar_vol.rolling(lookback, min_periods=lookback).mean()
        sigma = dollar_vol.rolling(lookback, min_periods=lookback).std()
 
    if norm == "z_score":
        return ((dollar_vol - mu) / sigma.replace(0, np.nan)).fillna(0)
    elif norm == "market_ratio":
        ratio = (dollar_vol / mu.replace(0, np.nan)).fillna(1)
        return (ratio - 1.0)  # centered: 0 = normal, >0 = above-average
    elif norm == "rank":
        rank = dollar_vol.rolling(lookback, min_periods=lookback).rank(pct=True)
        return (rank * 2 - 1).fillna(0)  # [-1, +1]
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
    df = df[df["datetime"] >= FETCH_START].reset_index(drop=True)
    if len(df) < 60:
        return None
 
    # Dollar volume (Hendershott 2011: preferred over share count)
    dollar_vol = (df["close"] * df["volume"]).astype(np.float64)
    rows = []
 
    for lb in LOOKBACKS:
        for norm in NORMS:
            for ewm in EWM_FLAGS:
                signal = compute_vol_signal(dollar_vol, lb, norm, ewm)
                bull = np.clip(np.tanh(signal.values), 0, 1).astype(np.float32)
                bear = np.clip(np.tanh(-signal.values), 0, 1).astype(np.float32)
 
                # GKM acceleration — FIXED column (not tuned)
                vol_roc = np.tanh(signal.diff().fillna(0).values).astype(np.float32)
 
                ewm_tag = "ewm" if ewm else "sma"
                combo_id = f"lb{lb}_{norm}_{ewm_tag}"
 
                combo_df = pd.DataFrame({
                    "datetime": df["datetime"].values,
                    "ticker": ticker,
                    "combo_id": combo_id,
                    "lookback": np.int8(lb),
                    "norm_method": norm,
                    "use_ewm": ewm,
                    "bull_vol_score": bull,
                    "bear_vol_score": bear,
                    "vol_roc": vol_roc,
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
    p = argparse.ArgumentParser(description="V06 Volume Trend — Daily Generator")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--workers", type=int, default=N_CORES)
    p.add_argument("--tickers", nargs="*", default=None)
    args = p.parse_args()
 
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    tickers = args.tickers or sorted(
        p.stem for p in (Path(args.data_dir) / RESOLUTION).glob("*.parquet")
        if not p.stem.startswith("_"))
 
    logging.info(f"V06 Volume Daily | Tickers: {len(tickers)} | "
                 f"Grid: {len(LOOKBACKS)}×{len(NORMS)}×{len(EWM_FLAGS)}=18 combos")
 
    jobs = [(args.data_dir, tk) for tk in tickers]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.map(process_ticker, jobs, chunksize=16)
    else:
        results = [process_ticker(j) for j in jobs]
 
    ok = [r for r in results if r is not None]
    logging.info(f"✓ Daily signals: {len(ok)}/{len(tickers)} tickers → {OUT}/")
 
 
if __name__ == "__main__":
    main()