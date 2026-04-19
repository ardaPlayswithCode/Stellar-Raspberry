"""
generate_daily.py — Var 07 Volatility Context (Daily Resolution)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STATE VARIABLE — No trainer. No IC. No labels. No normalization.
 
WHAT THIS DOES:
  Provides environmental physics for the God Model. Two orthogonal
  risk dimensions (Ang et al. 2006):
    Global:  VIX close (systematic, forward-looking 30-day implied vol)
    Local:   ATR% (idiosyncratic, per-ticker realized vol)
 
SENSORS:
  vix_close         — Raw CBOE VIX daily close (Whaley 2000)
  atr_pct           — Wilder ATR(14) / Close × 100
  parkinson_vol     — Range-based σ, 2.5-5× more efficient (Parkinson 1980)
  atr_ratio_20/60/120 — ATR(14) / SMA(ATR, window). >1.0 = expanding
  annual_sigma      — 20-bar rolling log-return σ, annualized
 
RESEARCH:
  Whaley (2000) JPM 26(3) — VIX = forward-looking. Raw close IS state.
  Parkinson (1980) JBus 53(1) — Range estimator efficiency.
  Schwert (1989) JF 44(5) — Vol persistent (AR=0.8994). No z-score needed.
  Ang et al. (2006) JF 61(1) — VIX ⊥ ATR%. Both needed.
  Adrian & Brunnermeier (2016) AER 106(7) — Low VIX = hidden risk.
 
OUTPUT:
  signals/{TICKER}.parquet → [datetime, ticker, vix_close, atr_pct,
    parkinson_vol, atr_ratio_20, atr_ratio_60, atr_ratio_120, annual_sigma]
 
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
N_CORES     = min(cpu_count(), 24)
PARKINSON_N = 20  # rolling window for Parkinson estimator
SIGMA_N     = 20  # rolling window for annualized sigma
ATR_SMA_WINDOWS = [20, 60, 120]
ANNUALIZE_DAILY = np.sqrt(252)
 
 
def load_vix(data_dir: str) -> pd.DataFrame:
    """Load VIX daily close. Try ^VIX.parquet then ^GSPC fallback patterns."""
    for name in ["^VIX", "VIX", "^VIX.parquet"]:
        p = Path(data_dir) / RESOLUTION / f"{name.replace('.parquet','')}.parquet"
        if p.exists():
            vdf = pd.read_parquet(p)
            if "datetime" not in vdf.columns and vdf.index.name == "datetime":
                vdf = vdf.reset_index()
            vdf["datetime"] = pd.to_datetime(vdf["datetime"])
            vdf = vdf.sort_values("datetime").reset_index(drop=True)
            return vdf[["datetime", "close"]].rename(columns={"close": "vix_close"})
    logging.warning("VIX parquet not found. vix_close will be NaN.")
    return pd.DataFrame(columns=["datetime", "vix_close"])
 
 
def parkinson_vol(high: pd.Series, low: pd.Series, n: int = PARKINSON_N) -> pd.Series:
    """
    Parkinson (1980) range-based volatility estimator.
    σ = sqrt( (1 / (4·n·ln2)) × Σ(ln(H/L))² )
    Rolling implementation over n bars.
    """
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
    if len(df) < 60:
        return None
 
    close = df["close"]
    high = df["high"]
    low = df["low"]
 
    # ── Local sensors (per-ticker, idiosyncratic) ─────────────────────
    atr = B.wilder_atr(high, low, close)
    atr_pct = (atr / close.replace(0, np.nan) * 100).astype(np.float32)
 
    # Parkinson (1980) range-based vol
    park_vol = parkinson_vol(high, low).astype(np.float32)
 
    # ATR expansion ratios: ATR(14) / SMA(ATR, window)
    ratios = {}
    for w in ATR_SMA_WINDOWS:
        sma = atr.rolling(w, min_periods=w).mean()
        ratios[f"atr_ratio_{w}"] = (atr / sma.replace(0, np.nan)).fillna(1.0).astype(np.float32)
 
    # Annualized log-return sigma (classical measure)
    log_ret = np.log(close / close.shift(1))
    ann_sigma = (log_ret.rolling(SIGMA_N, min_periods=SIGMA_N).std() * ANNUALIZE_DAILY).astype(np.float32)
 
    # ── Build result ──────────────────────────────────────────────────
    result = pd.DataFrame({
        "datetime": df["datetime"].values,
        "ticker": ticker,
        "atr_pct": atr_pct.values,
        "parkinson_vol": park_vol.values,
        "atr_ratio_20": ratios["atr_ratio_20"].values,
        "atr_ratio_60": ratios["atr_ratio_60"].values,
        "atr_ratio_120": ratios["atr_ratio_120"].values,
        "annual_sigma": ann_sigma.values,
    })
 
    # ── Global sensor (VIX — same for all tickers) ────────────────────
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
    p = argparse.ArgumentParser(description="V07 Volatility Context — Daily Generator (STATE)")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--workers", type=int, default=N_CORES)
    p.add_argument("--tickers", nargs="*", default=None)
    args = p.parse_args()
 
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    tickers = args.tickers or sorted(
        pp.stem for pp in (Path(args.data_dir) / RESOLUTION).glob("*.parquet")
        if not pp.stem.startswith("_") and not pp.stem.startswith("^"))
 
    vix_df = load_vix(args.data_dir)
    logging.info(f"V07 Volatility Daily (STATE) | Tickers: {len(tickers)} | VIX rows: {len(vix_df)}")
 
    # Can't pickle vix_df through Pool easily — use global or serial for VIX merge
    jobs = [(args.data_dir, tk, vix_df) for tk in tickers]
    # Serial processing due to shared vix_df (lightweight per-ticker anyway)
    results = [process_ticker(j) for j in jobs]
 
    ok = [r for r in results if r is not None]
    logging.info(f"✓ Daily state: {len(ok)}/{len(tickers)} tickers → {OUT}/")
 
 
if __name__ == "__main__":
    main()