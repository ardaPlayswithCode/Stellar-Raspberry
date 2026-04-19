"""
08_Range_Context / V1 / generate_weekly.py
===========================================
Weekly (W-FRI) range positioning. Same features as daily on weekly bars.
52-bar lookback = 52 weeks (same economic window).
 
Huddart et al (2009) used WEEKLY data — this generator is the
native resolution of their study. bars_since values = WEEKS.
 
Output: range_context_weekly.parquet
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
 
FETCH_START   = "2004-01-01"
TRAIN_END     = "2022-12-31"
VAL_END       = "2023-12-31"
MIN_AVG_VOL   = 500_000
LOOKBACK_52W  = 52
SMA_PERIOD    = 40   # ~200 daily ÷ 5 = 40 weekly bars
RANGE_PCTILE_W = 13  # 13 weeks ≈ 3 months
RESAMPLE      = "W-FRI"
 
THIS_DIR = Path(__file__).resolve().parent
OUT_PATH = THIS_DIR / "range_context_weekly.parquet"
 
 
def _universe():
    try:
        t = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        cur = t[0]["Symbol"].tolist()
        h = t[1] if len(t) > 1 else pd.DataFrame()
        a = h["Added"]["Ticker"].dropna().tolist() if not h.empty and "Added" in h.columns.get_level_values(0) else []
        r = h["Removed"]["Ticker"].dropna().tolist() if not h.empty and "Removed" in h.columns.get_level_values(0) else []
        return sorted(set(s.replace(".","-").strip().upper() for s in set(cur+a+r) if 1<=len(s.strip())<=6))
    except Exception:
        return ["AAPL","MSFT","GOOGL","AMZN","NVDA","META"]
 
 
def _resample(df):
    w = df.resample(RESAMPLE).agg({"open":"first","high":"max","low":"min",
                                    "close":"last","volume":"sum"})
    return w.dropna(subset=["close"])
 
 
def assign_split(dates):
    s = pd.Series("test", index=dates.index)
    s[dates <= TRAIN_END] = "train"
    s[(dates > TRAIN_END) & (dates <= VAL_END)] = "val"
    return s
 
 
def generate():
    print(f"{'='*60}\nV8 Range Context — Weekly (W-FRI)")
    print(f"52-bar lookback | Huddart native resolution (weeks)\n{'='*60}")
 
    tickers = _universe()
    all_records = []
    fail = 0
 
    for idx, tk in enumerate(tickers):
        if (idx+1)%100==0: print(f"  [{idx+1}/{len(tickers)}] fail={fail}")
        try:
            raw = yf.download(tk, start=FETCH_START, auto_adjust=True, progress=False)
            if raw.empty or len(raw)<260: fail+=1; continue
            df = raw[["Open","High","Low","Close","Volume"]].copy()
            df.columns = ["open","high","low","close","volume"]
            df = df[df["volume"].rolling(20).mean()>=MIN_AVG_VOL].copy()
            if len(df)<260: fail+=1; continue
            df = _resample(df)
            if len(df)<LOOKBACK_52W+10: fail+=1; continue
 
            close = df["close"]; high = df["high"]; low = df["low"]
 
            # v6_score with shift(1) guard
            shifted = close.shift(1)
            h52 = shifted.rolling(LOOKBACK_52W, min_periods=LOOKBACK_52W//2).max()
            l52 = shifted.rolling(LOOKBACK_52W, min_periods=LOOKBACK_52W//2).min()
            r52 = h52 - l52
            v6_raw = (close - l52) / r52.replace(0, np.nan)
            v6_score = v6_raw.clip(0, 1).fillna(0.5)
 
            # Staleness (weekly bars = weeks)
            bsh = shifted.rolling(LOOKBACK_52W, min_periods=LOOKBACK_52W//2).apply(
                lambda x: len(x)-int(np.argmax(x))-1, raw=True)
            bsl = shifted.rolling(LOOKBACK_52W, min_periods=LOOKBACK_52W//2).apply(
                lambda x: len(x)-int(np.argmin(x))-1, raw=True)
 
            # Z-score from SMA(40 weeks ≈ 200 days)
            sma = close.rolling(SMA_PERIOD, min_periods=SMA_PERIOD//2).mean()
            std = close.rolling(SMA_PERIOD, min_periods=SMA_PERIOD//2).std()
            zscore = ((close-sma)/std.replace(0,np.nan)).clip(-5,5).fillna(0)
 
            # Distance from 52w extremes (%)
            dh = ((h52-close)/close.replace(0,np.nan)*100).clip(0,200)
            dl = ((close-l52)/close.replace(0,np.nan)*100).clip(0,200)
 
            # Range percentile (13 weeks ≈ 3 months)
            dr = (high-low)/close.replace(0,np.nan)
            rpct = dr.rolling(RANGE_PCTILE_W, min_periods=RANGE_PCTILE_W//2).apply(
                lambda x: (x[-1]>x[:-1]).sum()/max(len(x)-1,1)*100, raw=True).fillna(50)
 
            out = pd.DataFrame({
                "date": df.index, "ticker": tk,
                "v6_score": v6_score.values,
                "bars_since_52w_high": bsh.values,
                "bars_since_52w_low": bsl.values,
                "zscore_sma200": zscore.values,
                "dist_52w_high": dh.values,
                "dist_52w_low": dl.values,
                "range_pctile_63": rpct.values,
            })
            out = out.dropna(subset=["v6_score"])
            if len(out)<20: fail+=1; continue
            out["split"] = assign_split(out["date"]).values
            all_records.append(out)
 
        except Exception as e:
            fail+=1
            if fail<=3: print(f"  [W] {tk}: {e}")
 
    if not all_records: print("[ERROR] No data."); sys.exit(1)
 
    combined = pd.concat(all_records, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["date","ticker"]).reset_index(drop=True)
 
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH, index=False)
 
    train = combined[combined["split"]=="train"]
    print(f"\n{'─'*60}")
    print(f"Output: {OUT_PATH}\nRows: {len(combined):,}")
    print(f"v6_score (train): mean={train['v6_score'].mean():.3f}")
    print(f"bars_since_high in WEEKS (train): mean={train['bars_since_52w_high'].dropna().mean():.1f}")
    print(f"Z-score SMA (train): mean={train['zscore_sma200'].mean():.2f}")
 
 
if __name__=="__main__":
    generate()