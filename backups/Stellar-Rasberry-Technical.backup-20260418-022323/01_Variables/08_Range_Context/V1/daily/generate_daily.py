"""
08_Range_Context / V1 / generate_daily.py
==========================================
STATE VARIABLE — reports where price sits relative to its structural
anchors. Bias shift for the MGM: changes how V1–V6 signals are
INTERPRETED, not just whether to trust them.
 
Output columns feeding TABULAR_COLUMNS [14–19]:
  v6_score              — (close−low_52w)/(high_52w−low_52w) ∈ [0,1]
  bars_since_52w_high   — Huddart (2009) LHIGH analog
  bars_since_52w_low    — Huddart (2009) LLOW analog
 
Extended features for regime classification:
  zscore_sma200   — (Close − SMA200) / StdDev(Close, 200)
                    Z-score stationarity: |Z| > 2 = overextended.
                    |Z| < 0.5 = mean-reverting. Between = trending.
  dist_52w_high   — (52w_high − Close) / Close × 100 (% below high)
  dist_52w_low    — (Close − 52w_low) / Close × 100 (% above low)
  range_pctile_63 — percentile rank of (H−L)/C within 63-day window
                    0 = narrowest range, 100 = widest. Identifies
                    compression → expansion transitions.
 
LOOKAHEAD GUARD: shift(1) on all rolling H/L operations.
 
Research:
  Huddart et al (2009) ManSci 55(1): threshold effect at 52w extremes.
  Grinblatt & Han (2005) JFE 78(2): v6_score subsumes momentum.
  Kahneman & Tversky (1979): λ=2.25 loss aversion at reference points.
  Baker et al (2012) JF 67(1): 52w high = purely psychological anchor.
 
Output: range_context_daily.parquet
Schema: date, ticker, v6_score, bars_since_52w_high, bars_since_52w_low,
        zscore_sma200, dist_52w_high, dist_52w_low, range_pctile_63, split
"""
 
import sys
import warnings
from pathlib import Path
 
import numpy as np
import pandas as pd
import yfinance as yf
 
warnings.filterwarnings("ignore")
 
import importlib as _il
_bm = _il.import_module("01_base_modules_")
StateVariableModule = _bm.StateVariableModule
 
FETCH_START   = "2005-01-01"
TRAIN_END     = "2022-12-31"
VAL_END       = "2023-12-31"
MIN_AVG_VOL   = 500_000
LOOKBACK_52W  = 252
SMA_PERIOD    = 200
RANGE_PCTILE_W = 63  # 3-month window for range percentile
 
THIS_DIR = Path(__file__).resolve().parent
OUT_PATH = THIS_DIR / "range_context_daily.parquet"
 
 
def _universe():
    try:
        t = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        cur = t[0]["Symbol"].tolist()
        h = t[1] if len(t) > 1 else pd.DataFrame()
        a = h["Added"]["Ticker"].dropna().tolist() if not h.empty and "Added" in h.columns.get_level_values(0) else []
        r = h["Removed"]["Ticker"].dropna().tolist() if not h.empty and "Removed" in h.columns.get_level_values(0) else []
        return sorted(set(s.replace(".","-").strip().upper() for s in set(cur+a+r) if 1<=len(s.strip())<=6))
    except Exception:
        return ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","UNH","XOM"]
 
 
def compute_range_features(close, high, low):
    """
    Core range computation with shift(1) lookahead guard.
    v6_score = (close − low_52w) / (high_52w − low_52w)
    """
    shifted = close.shift(1)
    high_52w = shifted.rolling(LOOKBACK_52W, min_periods=LOOKBACK_52W//2).max()
    low_52w = shifted.rolling(LOOKBACK_52W, min_periods=LOOKBACK_52W//2).min()
    range_52w = high_52w - low_52w
 
    v6_raw = (close - low_52w) / range_52w.replace(0, np.nan)
    v6_score = v6_raw.clip(0.0, 1.0).fillna(0.5)
 
    # Distance from 52w extremes (%)
    dist_high = ((high_52w - close) / close.replace(0, np.nan) * 100).clip(0, 200)
    dist_low = ((close - low_52w) / close.replace(0, np.nan) * 100).clip(0, 200)
 
    return v6_score, high_52w, low_52w, dist_high, dist_low
 
 
def compute_staleness(close):
    """Huddart (2009) LHIGH/LLOW: bars since rolling 52w max/min was set."""
    shifted = close.shift(1)
    bsh = shifted.rolling(LOOKBACK_52W, min_periods=LOOKBACK_52W//2).apply(
        lambda x: len(x) - int(np.argmax(x)) - 1, raw=True)
    bsl = shifted.rolling(LOOKBACK_52W, min_periods=LOOKBACK_52W//2).apply(
        lambda x: len(x) - int(np.argmin(x)) - 1, raw=True)
    return bsh, bsl
 
 
def compute_zscore_sma200(close):
    """
    Z-score of close relative to SMA(200).
    Z = (Close − SMA200) / σ(Close, 200)
    |Z| > 2: overextended. |Z| < 0.5: mean-reverting.
    """
    sma = close.rolling(SMA_PERIOD, min_periods=SMA_PERIOD//2).mean()
    std = close.rolling(SMA_PERIOD, min_periods=SMA_PERIOD//2).std()
    return ((close - sma) / std.replace(0, np.nan)).clip(-5, 5).fillna(0)
 
 
def compute_range_percentile(high, low, close, window=RANGE_PCTILE_W):
    """
    3-month range percentile: where does today's (H−L)/C rank
    within the last 63 days? 0 = narrowest, 100 = widest.
    Identifies compression → expansion transitions.
    """
    daily_range = (high - low) / close.replace(0, np.nan)
    return daily_range.rolling(window, min_periods=window//2).apply(
        lambda x: (x[-1] > x[:-1]).sum() / max(len(x)-1, 1) * 100, raw=True
    ).fillna(50)
 
 
def assign_split(dates):
    s = pd.Series("test", index=dates.index)
    s[dates <= TRAIN_END] = "train"
    s[(dates > TRAIN_END) & (dates <= VAL_END)] = "val"
    return s
 
 
def generate():
    print(f"{'='*60}\nV8 Range Context — Daily Generator")
    print(f"v6_score + Huddart staleness + Z-score(SMA200) + range pctile")
    print(f"TABULAR: [14] v6_score [15] bars_since_high [16] bars_since_low\n{'='*60}")
 
    tickers = _universe()
    all_records = []
    fail = 0
 
    for idx, tk in enumerate(tickers):
        if (idx+1)%100==0: print(f"  [{idx+1}/{len(tickers)}] fail={fail}")
        try:
            raw = yf.download(tk, start=FETCH_START, auto_adjust=True, progress=False)
            if raw.empty or len(raw) < LOOKBACK_52W + 50: fail+=1; continue
            df = raw[["Open","High","Low","Close","Volume"]].copy()
            df.columns = ["open","high","low","close","volume"]
            df = df[df["volume"].rolling(20).mean() >= MIN_AVG_VOL].copy()
            if len(df) < LOOKBACK_52W + 50: fail+=1; continue
 
            # Core range
            v6, h52, l52, dh, dl = compute_range_features(
                df["close"], df["high"], df["low"])
 
            # Staleness
            bsh, bsl = compute_staleness(df["close"])
 
            # Z-score from SMA200
            zscore = compute_zscore_sma200(df["close"])
 
            # Range percentile
            rpct = compute_range_percentile(df["high"], df["low"], df["close"])
 
            out = pd.DataFrame({
                "date": df.index, "ticker": tk,
                "v6_score": v6.values,
                "bars_since_52w_high": bsh.values,
                "bars_since_52w_low": bsl.values,
                "zscore_sma200": zscore.values,
                "dist_52w_high": dh.values,
                "dist_52w_low": dl.values,
                "range_pctile_63": rpct.values,
            })
 
            out = out.dropna(subset=["v6_score"])
            if len(out) < 50: fail+=1; continue
 
            out["split"] = assign_split(out["date"]).values
            all_records.append(out)
 
        except Exception as e:
            fail += 1
            if fail <= 3: print(f"  [W] {tk}: {e}")
 
    if not all_records: print("[ERROR] No data."); sys.exit(1)
 
    combined = pd.concat(all_records, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["date","ticker"]).reset_index(drop=True)
 
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH, index=False)
 
    train = combined[combined["split"]=="train"]
    print(f"\n{'─'*60}")
    print(f"Output: {OUT_PATH}")
    print(f"Rows: {len(combined):,} | Tickers: {combined['ticker'].nunique()}")
    print(f"v6_score (train): mean={train['v6_score'].mean():.3f} median={train['v6_score'].median():.3f}")
    print(f"Z-score SMA200 (train): mean={train['zscore_sma200'].mean():.2f} std={train['zscore_sma200'].std():.2f}")
    print(f"bars_since_high (train): mean={train['bars_since_52w_high'].dropna().mean():.1f}")
    print(f"range_pctile_63 (train): mean={train['range_pctile_63'].mean():.1f}")
 
 
if __name__ == "__main__":
    generate()