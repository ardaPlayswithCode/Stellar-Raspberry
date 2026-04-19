"""
01_Stock_Geometry / V1 / monthly / generate_monthly.py
=======================================================
Per-ticker monthly charts (ME anchor). 680×680 PNG for ConvNeXt-Base.
12-bar lookback (~1 year of months). Layer 1 bear filter:
  If monthly CNN predicts bearish → suppresses daily/weekly bull signals.

Thinner data: ~240 monthly bars per ticker (2005–2025).
2-pane layout: candlestick (75%) | volume (25%). No RSI/MACD —
insufficient monthly bars for oscillator stability.

Output: monthly/images/{TICKER}/{YYYY-MM-DD}.png  (date = month-end)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")

FETCH_START   = "2004-01-01"
MIN_AVG_VOL   = 500_000
IMG_PX        = 680
DPI           = 100
FIG           = IMG_PX / DPI
LOOKBACK      = 12
RESAMPLE      = "ME"
UP, DOWN, BG  = "#26a69a", "#ef5350", "black"

THIS_DIR = Path(__file__).resolve().parent
IMG_DIR  = THIS_DIR / "images"


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


def _resample(df):
    m = df.resample(RESAMPLE).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
    return m.dropna(subset=["close"])


def _render(ohlcv, path):
    n = len(ohlcv)
    if n < 6: return
    o,h,l,c,v = (ohlcv["open"].values, ohlcv["high"].values, ohlcv["low"].values,
                  ohlcv["close"].values, ohlcv["volume"].values)

    fig, ax = plt.subplots(2,1,figsize=(FIG,FIG),dpi=DPI,
                           gridspec_kw={"height_ratios":[3,1],"hspace":0})
    fig.patch.set_facecolor(BG)

    a=ax[0]; a.set_facecolor(BG); a.set_xlim(-0.5,n-0.5)
    a.set_ylim(l.min()*0.995, h.max()*1.005)
    for i in range(n):
        cl = UP if c[i]>=o[i] else DOWN
        a.plot([i,i],[l[i],h[i]],color=cl,lw=0.8)
        bl,bh=min(o[i],c[i]),max(o[i],c[i])
        a.add_patch(Rectangle((i-0.4,bl),0.8,max(bh-bl,(h.max()-l.min())*0.003),fc=cl,ec="none"))
    a.axis("off")

    a=ax[1]; a.set_facecolor(BG)
    a.bar(range(n),v/(v.max()+1e-10),color=[UP if c[i]>=o[i] else DOWN for i in range(n)],
          width=0.8,alpha=0.7)
    a.set_xlim(-0.5,n-0.5); a.axis("off")

    plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path,dpi=DPI,facecolor=BG,bbox_inches="tight",pad_inches=0)
    plt.close(fig)


def generate():
    print(f"{'='*60}\nV1 Stock Geometry — Monthly (ME) | {IMG_PX}×{IMG_PX} | Bear Filter\n{'='*60}")
    tickers = _universe()
    ok = fail = 0

    for idx, tk in enumerate(tickers):
        if (idx+1)%100==0: print(f"  [{idx+1}/{len(tickers)}] ok={ok} fail={fail}")
        try:
            raw = yf.download(tk,start=FETCH_START,auto_adjust=True,progress=False)
            if raw.empty or len(raw)<260: fail+=1; continue
            df = raw[["Open","High","Low","Close","Volume"]].copy()
            df.columns = ["open","high","low","close","volume"]
            df = df[df["volume"].rolling(20).mean()>=MIN_AVG_VOL].copy()
            if len(df)<260: fail+=1; continue
            df = _resample(df)
            if len(df)<LOOKBACK+3: fail+=1; continue

            for dt in df.index[LOOKBACK:]:
                loc = df.index.get_loc(dt)
                if loc<LOOKBACK: continue
                p = IMG_DIR / tk / f"{dt.date()}.png"
                if p.exists(): continue
                _render(df.iloc[loc-LOOKBACK+1:loc+1], p)
            ok += 1
        except Exception as e:
            fail+=1
            if fail<=3: print(f"  [W] {tk}: {e}")

    print(f"\n{'─'*60}\nDone: {ok} tickers | {fail} failed\nOutput: {IMG_DIR}")

if __name__=="__main__":
    generate()