"""
generate_weekly.py — Var 07 Volatility Context (Weekly Resolution)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STATE VARIABLE — No trainer. No IC. No labels.
 
Weekly VIX = Friday close (W-FRI anchor). ATR% and Parkinson computed
on weekly OHLCV bars. FETCH_START = "2004-01-01".
 
USAGE:
  python generate_weekly.py --data_dir ../../00_Mini_Model/data/ohlcv
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
 
RESOLUTION  = "weekly"
FETCH_START = "2004-01-01"
OUT         = THIS / "signals"
PARKINSON_N = 12
SIGMA_N     = 12
ATR_SMA_WINDOWS = [12, 26, 52]
ANNUALIZE_WEEKLY = np.sqrt(52)
 
 
def load_vix(data_dir):
    for name in ["^VIX", "VIX"]:
        # VIX weekly: resample from daily or load directly
        p_w = Path(data_dir) / RESOLUTION / f"{name}.parquet"
        p_d = Path(data_dir) / "daily" / f"{name}.parquet"
        target = p_w if p_w.exists() else p_d if p_d.exists() else None
        if target:
            vdf = pd.read_parquet(target)
            if "datetime" not in vdf.columns and vdf.index.name == "datetime":
                vdf = vdf.reset_index()
            vdf["datetime"] = pd.to_datetime(vdf["datetime"])
            vdf = vdf.sort_values("datetime")
            if target == p_d:
                # Resample daily VIX to weekly (last close of week)
                vdf = vdf.set_index("datetime")
                vdf = vdf["close"].resample("W-FRI").last().dropna().reset_index()
                vdf.columns = ["datetime", "vix_close"]
            else:
                vdf = vdf[["datetime", "close"]].rename(columns={"close": "vix_close"})
            return vdf
    return pd.DataFrame(columns=["datetime", "vix_close"])
 
 
def parkinson_vol(high, low, n=PARKINSON_N):
    log_hl_sq = (np.log(high / low.replace(0, np.nan))) ** 2
    return np.sqrt(log_hl_sq.rolling(n, min_periods=n).mean() / (4 * np.log(2)))
 
 
def process_ticker(args):
    data_dir, ticker, vix_df = args
    p = Path(data_dir) / RESOLUTION / f"{ticker}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "datetime" not in df.columns and df.index.name == "datetime":
        df = df.reset_index()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df[df["datetime"] >= FETCH_START].reset_index(drop=True)
    if len(df) < 30:
        return None
 
    close, high, low = df["close"], df["high"], df["low"]
    atr = B.wilder_atr(high, low, close)
    atr_pct = (atr / close.replace(0, np.nan) * 100).astype(np.float32)
    park_vol = parkinson_vol(high, low).astype(np.float32)
 
    ratios = {}
    for w in ATR_SMA_WINDOWS:
        sma = atr.rolling(w, min_periods=min(w, 12)).mean()
        ratios[f"atr_ratio_{w}"] = (atr / sma.replace(0, np.nan)).fillna(1.0).astype(np.float32)
 
    log_ret = np.log(close / close.shift(1))
    ann_sigma = (log_ret.rolling(SIGMA_N, min_periods=SIGMA_N).std() * ANNUALIZE_WEEKLY).astype(np.float32)
 
    result = pd.DataFrame({
        "datetime": df["datetime"].values, "ticker": ticker,
        "atr_pct": atr_pct.values, "parkinson_vol": park_vol.values,
        "atr_ratio_12": ratios["atr_ratio_12"].values,
        "atr_ratio_26": ratios["atr_ratio_26"].values,
        "atr_ratio_52": ratios["atr_ratio_52"].values,
        "annual_sigma": ann_sigma.values,
    })
 
    if len(vix_df) > 0:
        result = result.merge(vix_df, on="datetime", how="left")
        result["vix_close"] = result["vix_close"].ffill().astype(np.float32)
    else:
        result["vix_close"] = np.float32(np.nan)
 
    out_path = OUT / f"{ticker}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    return ticker
 
 
def main():
    p = argparse.ArgumentParser(description="V07 Volatility Context — Weekly (STATE)")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--tickers", nargs="*", default=None)
    args = p.parse_args()
 
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    tickers = args.tickers or sorted(
        pp.stem for pp in (Path(args.data_dir) / RESOLUTION).glob("*.parquet")
        if not pp.stem.startswith("_") and not pp.stem.startswith("^"))
 
    vix_df = load_vix(args.data_dir)
    logging.info(f"V07 Volatility Weekly (STATE) | Tickers: {len(tickers)} | VIX rows: {len(vix_df)}")
    results = [process_ticker((args.data_dir, tk, vix_df)) for tk in tickers]
    ok = [r for r in results if r is not None]
    logging.info(f"✓ Weekly state: {len(ok)}/{len(tickers)} tickers → {OUT}/")
 
 
if __name__ == "__main__":
    main()