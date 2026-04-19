"""
generate_1hour.py — Var 07 Volatility Context (1-Hour Resolution)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STATE VARIABLE — No trainer. No IC. No labels.
 
LOOKAHEAD GUARD: VIX uses previous trading day's close (.shift(1)
on daily VIX before merging). Folder name "1hour" — non-negotiable.
 
USAGE:
  python generate_1hour.py --data_dir ../../00_Mini_Model/data/ohlcv
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
 
import argparse, logging, sys
from pathlib import Path
import numpy as np
import pandas as pd
 
THIS = Path(__file__).resolve().parent
V1   = THIS.parent
sys.path.insert(0, str(V1.parent.parent / "00_Mini_Model"))
import importlib as _il
B = _il.import_module("01_base_modules_")
 
RESOLUTION = "1hour"
OUT        = THIS / "signals"
PARKINSON_N = 60  # 60 × 1h ≈ 8.6 trading days
SIGMA_N     = 60
ATR_SMA_WINDOWS = [20, 60, 120]
ANNUALIZE_1H = np.sqrt(252 * 6.5)  # ~6.5 bars/day
 
 
def load_vix_daily(data_dir):
    for name in ["^VIX", "VIX"]:
        p = Path(data_dir) / "daily" / f"{name}.parquet"
        if p.exists():
            vdf = pd.read_parquet(p)
            if "datetime" not in vdf.columns and vdf.index.name == "datetime":
                vdf = vdf.reset_index()
            vdf["datetime"] = pd.to_datetime(vdf["datetime"])
            vdf = vdf.sort_values("datetime").reset_index(drop=True)
            vdf["vix_close"] = vdf["close"].shift(1).astype(np.float32)
            vdf["date"] = vdf["datetime"].dt.date
            return vdf[["date", "vix_close"]].dropna()
    return pd.DataFrame(columns=["date", "vix_close"])
 
 
def parkinson_vol(high, low, n=PARKINSON_N):
    log_hl_sq = (np.log(high / low.replace(0, np.nan))) ** 2
    return np.sqrt(log_hl_sq.rolling(n, min_periods=n).mean() / (4 * np.log(2)))
 
 
def process_ticker(args):
    data_dir, ticker, vix_daily = args
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
 
    close, high, low = df["close"], df["high"], df["low"]
    atr = B.wilder_atr(high, low, close)
    atr_pct = (atr / close.replace(0, np.nan) * 100).astype(np.float32)
    park_vol = parkinson_vol(high, low).astype(np.float32)
 
    ratios = {}
    for w in ATR_SMA_WINDOWS:
        sma = atr.rolling(w, min_periods=min(w, 20)).mean()
        ratios[f"atr_ratio_{w}"] = (atr / sma.replace(0, np.nan)).fillna(1.0).astype(np.float32)
 
    log_ret = np.log(close / close.shift(1))
    ann_sigma = (log_ret.rolling(SIGMA_N, min_periods=SIGMA_N).std() * ANNUALIZE_1H).astype(np.float32)
 
    result = pd.DataFrame({
        "datetime": df["datetime"].values, "ticker": ticker,
        "atr_pct": atr_pct.values, "parkinson_vol": park_vol.values,
        "atr_ratio_20": ratios["atr_ratio_20"].values,
        "atr_ratio_60": ratios["atr_ratio_60"].values,
        "atr_ratio_120": ratios["atr_ratio_120"].values,
        "annual_sigma": ann_sigma.values,
    })
 
    if len(vix_daily) > 0:
        result["date"] = pd.to_datetime(result["datetime"]).dt.date
        result = result.merge(vix_daily, on="date", how="left")
        result["vix_close"] = result["vix_close"].ffill().astype(np.float32)
        result.drop(columns=["date"], inplace=True)
    else:
        result["vix_close"] = np.float32(np.nan)
 
    out_path = OUT / f"{ticker}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    return ticker
 
 
def main():
    p = argparse.ArgumentParser(description="V07 Volatility Context — 1hour (STATE)")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--tickers", nargs="*", default=None)
    args = p.parse_args()
 
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    tickers = args.tickers or sorted(
        pp.stem for pp in (Path(args.data_dir) / RESOLUTION).glob("*.parquet")
        if not pp.stem.startswith("_") and not pp.stem.startswith("^"))
 
    vix_daily = load_vix_daily(args.data_dir)
    logging.info(f"V07 Volatility 1hour (STATE) | Tickers: {len(tickers)} | VIX days: {len(vix_daily)} (T-1)")
    results = [process_ticker((args.data_dir, tk, vix_daily)) for tk in tickers]
    ok = [r for r in results if r is not None]
    logging.info(f"✓ 1hour state: {len(ok)}/{len(tickers)} tickers → {OUT}/")
 
 
if __name__ == "__main__":
    main()