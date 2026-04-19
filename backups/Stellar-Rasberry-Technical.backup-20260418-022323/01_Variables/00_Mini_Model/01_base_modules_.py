"""
01_base_modules_.py — 3M-Point Production Foundation
======================================================
Vectorized T-1 alignment: numpy searchsorted on int64 ns arrays.
Parallel rendering: multiprocessing Pool saturates 24 M3 Ultra cores.
Glob-based path resolution: build set of existing images once, then map.
Physics constants: zero drift from daily architecture.
"""
 
from pathlib import Path
from typing import Dict, List, Optional
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
 
PX            = 680
DPI           = 100
FIG           = PX / DPI
RSI_P         = 14
MACD_F        = 12
MACD_S        = 26
MACD_SIG      = 9
ATR_P         = 14
UP            = "#26a69a"
DOWN          = "#ef5350"
BG            = "black"
VIX_CLR       = "#e67e22"
RSI_CLR       = "#ab47bc"
 
RESOLUTIONS   = ["15min", "1hour", "daily", "weekly", "monthly"]
RESAMPLE_MAP  = {"1hour": "1h", "daily": "1D", "weekly": "W-FRI", "monthly": "ME"}
LOOKBACK_MAP  = {"15min": 60, "1hour": 60, "daily": 60, "weekly": 52, "monthly": 12}
PANEL_4       = [6, 1.5, 1.25, 1.25]
PANEL_3       = [6.5, 2, 1.5]
PANEL_2       = [3, 1]
 
SECTOR_ETFS   = ["XLK","XLF","XLV","XLY","XLC","XLI","XLE","XLP","XLRE","XLU","XLB"]
HEAVY_WEIGHTS = {
    "XLK":{"AAPL","MSFT","NVDA","AVGO","CRM"},"XLF":{"BRK-B","JPM","V","MA","BAC"},
    "XLV":{"UNH","JNJ","LLY","ABBV","MRK"},"XLY":{"AMZN","TSLA","HD","MCD","NKE"},
    "XLC":{"META","GOOGL","GOOG","NFLX","DIS"},"XLI":{"GE","CAT","UNP","HON","RTX"},
    "XLE":{"XOM","CVX","COP","SLB","EOG"},"XLP":{"PG","COST","KO","PEP","PM"},
    "XLRE":{"AMT","PLD","CCI","EQIX","PSA"},"XLU":{"NEE","SO","DUK","SRE","AEP"},
    "XLB":{"LIN","APD","SHW","FCX","ECL"},
}
XLC_DATE      = pd.Timestamp("2018-09-24")
XLRE_DATE     = pd.Timestamp("2015-10-07")
XLC_FROM_XLK  = {"GOOGL","GOOG","META"}
XLC_FROM_XLY  = {"DIS","CMCSA","NFLX"}
XLC_ALL       = XLC_FROM_XLK | XLC_FROM_XLY | {"VZ","T","CHTR","ATVI","EA","TTWO","TMUS","FOXA","FOX","VIAC"}
XLRE_ALL      = {"AMT","PLD","CCI","EQIX","PSA","SPG","DLR","O","WELL","AVB","EQR","VTR"}
TRAIN_END     = "2022-12-31"
VAL_END       = "2023-12-31"
N_CORES       = min(cpu_count(), 24)
 
 
def wilder_rsi(c: pd.Series) -> pd.Series:
    d = c.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    ag = g.ewm(alpha=1/RSI_P, min_periods=RSI_P, adjust=False).mean()
    al = l.ewm(alpha=1/RSI_P, min_periods=RSI_P, adjust=False).mean()
    return (100 - 100/(1 + ag/al.replace(0, np.nan))).fillna(50)
 
def macd_hist(c: pd.Series) -> pd.Series:
    ef = c.ewm(span=MACD_F, adjust=False).mean()
    es = c.ewm(span=MACD_S, adjust=False).mean()
    ml = ef - es
    return ml - ml.ewm(span=MACD_SIG, adjust=False).mean()
 
def wilder_atr(h, l, c, p=ATR_P):
    pc = c.shift(1)
    tr = pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(span=p, min_periods=p, adjust=False).mean()
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# VECTORIZED T-1 ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════
 
def t_minus_1_align(master_ns: np.ndarray, child_ns: np.ndarray) -> np.ndarray:
    """
    For each master timestamp find the last COMPLETED child bar.
    Pure numpy searchsorted on int64 ns — O(N log M), no Python loop.
    Returns int64 ns array; -1 where no prior bar exists.
    """
    pos = np.searchsorted(child_ns, master_ns, side="right") - 1
    result = np.where(pos >= 0, child_ns[np.clip(pos, 0, len(child_ns)-1)], -1)
    result = np.where((result >= 0) & (result < master_ns), result, -1)
    return result
 
def build_t1_matrix(master_idx: pd.DatetimeIndex,
                    child_bars: Dict[str, pd.DatetimeIndex]) -> pd.DataFrame:
    """Full T-1 matrix. One searchsorted per timeframe. ~2s for 3M rows × 4 TFs."""
    master_ns = master_idx.astype(np.int64)
    cols = {}
    for tf, idx in child_bars.items():
        if idx is None or len(idx) == 0:
            cols[f"t1_{tf}"] = pd.array([pd.NaT] * len(master_ns), dtype="datetime64[ns]")
            continue
        child_ns = idx.sort_values().astype(np.int64)
        aligned = t_minus_1_align(master_ns, child_ns)
        cols[f"t1_{tf}"] = pd.array(
            np.where(aligned >= 0, aligned, np.datetime64("NaT")).astype("datetime64[ns]"),
            dtype="datetime64[ns]")
    return pd.DataFrame(cols, index=master_idx)
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# GLOB-BASED PATH RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════════
 
def build_image_set(image_dir: str) -> set:
    """Glob all .png paths into a set. O(1) lookup per image vs O(stat) for .exists()."""
    return set(str(p) for p in Path(image_dir).rglob("*.png"))
 
def resolve_paths_vectorized(
    master_idx: pd.DatetimeIndex,
    image_set: set,
    path_template: str,
    ticker: str,
    is_intraday: bool = False,
) -> pd.Series:
    """
    Build image path strings vectorized, check against pre-globbed set.
    path_template: e.g. "01_Stock_Geometry/V1/daily/images/{ticker}/{date}.png"
    """
    if is_intraday:
        dates_str = master_idx.strftime("%Y-%m-%d_%H%M")
    else:
        dates_str = master_idx.strftime("%Y-%m-%d")
    paths = np.array([path_template.format(ticker=ticker, date=d) for d in dates_str])
    mask = np.array([p in image_set for p in paths])
    result = np.where(mask, paths, None)
    return pd.Series(result, index=master_idx)
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# FORWARD-FILL (vectorized, no pd.reindex)
# ═══════════════════════════════════════════════════════════════════════════════
 
def ffill_paths(master_idx: pd.DatetimeIndex,
                img_dates: np.ndarray,
                img_paths: np.ndarray) -> pd.Series:
    """Forward-fill image paths via searchsorted. Pure numpy for 3M rows."""
    dates_ns = pd.DatetimeIndex(img_dates).astype(np.int64)
    master_ns = master_idx.astype(np.int64)
    pos = np.searchsorted(dates_ns, master_ns, side="right") - 1
    pos = np.clip(pos, 0, len(img_paths) - 1)
    return pd.Series(img_paths[pos], index=master_idx)
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# SPLIT / SECTOR
# ═══════════════════════════════════════════════════════════════════════════════
 
def assign_split_vec(dates: pd.Series) -> pd.Series:
    ts = pd.to_datetime(dates)
    result = np.full(len(ts), "test", dtype=object)
    result[ts <= pd.Timestamp(TRAIN_END)] = "train"
    result[(ts > pd.Timestamp(TRAIN_END)) & (ts <= pd.Timestamp(VAL_END))] = "val"
    return pd.Series(result, index=dates.index)
 
def assign_split(dt):
    if dt <= pd.Timestamp(TRAIN_END): return "train"
    if dt <= pd.Timestamp(VAL_END): return "val"
    return "test"
 
def check_self_reference(ticker: str, etf: str) -> bool:
    return ticker in HEAVY_WEIGHTS.get(etf, set())
 
def get_sector_etf(ticker: str, as_of: pd.Timestamp, sector_gics: str = None) -> str:
    if as_of < XLC_DATE and ticker in XLC_ALL:
        return "XLK" if ticker in XLC_FROM_XLK else "XLY"
    if as_of < XLRE_DATE and ticker in XLRE_ALL:
        return "XLF"
    _m = {"Information Technology":"XLK","Financials":"XLF","Health Care":"XLV",
          "Consumer Discretionary":"XLY","Communication Services":"XLC","Industrials":"XLI",
          "Energy":"XLE","Consumer Staples":"XLP","Real Estate":"XLRE",
          "Utilities":"XLU","Materials":"XLB"}
    return _m.get(sector_gics, "SPY") if sector_gics else "SPY"
 
def resample_ohlcv(df, freq):
    w = df.resample(freq).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
    return w.dropna(subset=["close"])
 
def get_universe():
    try:
        t = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        cur = t[0]["Symbol"].tolist()
        h = t[1] if len(t) > 1 else pd.DataFrame()
        a = h["Added"]["Ticker"].dropna().tolist() if not h.empty and "Added" in h.columns.get_level_values(0) else []
        r = h["Removed"]["Ticker"].dropna().tolist() if not h.empty and "Removed" in h.columns.get_level_values(0) else []
        return sorted(set(s.replace(".","-").strip().upper() for s in set(cur+a+r) if 1<=len(s.strip())<=6))
    except Exception:
        return ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","UNH","XOM"]
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# FORMATTERS
# ═══════════════════════════════════════════════════════════════════════════════
 
def fmt_15m(dt) -> str: return dt.strftime("%Y-%m-%d_%H%M")
def fmt_date(dt) -> str: return str(dt.date() if hasattr(dt, 'date') else dt)
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# PARALLEL RENDERING — saturates 24 M3 Ultra cores
# ═══════════════════════════════════════════════════════════════════════════════
 
def _ohlcv_dict(df_slice):
    return {"o":df_slice["open"].values,"h":df_slice["high"].values,
            "l":df_slice["low"].values,"c":df_slice["close"].values,
            "v":df_slice["volume"].values}
 
def _r4p(args):
    od, rsi, hist, ps = args
    p = Path(ps)
    if p.exists(): return
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    n=len(rsi)
    if n<10: return
    o,h,l,c,v=od["o"],od["h"],od["l"],od["c"],od["v"]
    fig,ax=plt.subplots(4,1,figsize=(FIG,FIG),dpi=DPI,gridspec_kw={"height_ratios":PANEL_4,"hspace":0})
    fig.patch.set_facecolor(BG)
    a=ax[0];a.set_facecolor(BG);a.set_xlim(-0.5,n-0.5);a.set_ylim(l.min()*0.998,h.max()*1.002)
    for i in range(n):
        cl=UP if c[i]>=o[i] else DOWN;a.plot([i,i],[l[i],h[i]],color=cl,lw=0.6)
        bl,bh=min(o[i],c[i]),max(o[i],c[i])
        a.add_patch(Rectangle((i-0.35,bl),0.7,max(bh-bl,(h.max()-l.min())*0.002),fc=cl,ec="none"))
    a.axis("off")
    a=ax[1];a.set_facecolor(BG);a.bar(range(n),v/(v.max()+1e-10),color=[UP if c[i]>=o[i] else DOWN for i in range(n)],width=0.7,alpha=0.7);a.set_xlim(-0.5,n-0.5);a.axis("off")
    a=ax[2];a.set_facecolor(BG);a.plot(range(n),rsi,color=RSI_CLR,lw=0.8);a.axhline(70,color="#555",lw=0.3,ls="--");a.axhline(30,color="#555",lw=0.3,ls="--");a.set_xlim(-0.5,n-0.5);a.set_ylim(0,100);a.axis("off")
    a=ax[3];a.set_facecolor(BG);a.bar(range(n),hist,color=[UP if x>=0 else DOWN for x in hist],width=0.7,alpha=0.8);a.axhline(0,color="#555",lw=0.3);a.set_xlim(-0.5,n-0.5);a.axis("off")
    plt.subplots_adjust(left=0,right=1,top=1,bottom=0);p.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(p,dpi=DPI,facecolor=BG,bbox_inches="tight",pad_inches=0);plt.close(fig)
 
def _r3p(args):
    od,rsi,ps=args;p=Path(ps)
    if p.exists():return
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    n=len(rsi);
    if n<10:return
    o,h,l,c,v=od["o"],od["h"],od["l"],od["c"],od["v"]
    fig,ax=plt.subplots(3,1,figsize=(FIG,FIG),dpi=DPI,gridspec_kw={"height_ratios":PANEL_3,"hspace":0})
    fig.patch.set_facecolor(BG)
    a=ax[0];a.set_facecolor(BG);a.set_xlim(-0.5,n-0.5);a.set_ylim(l.min()*0.998,h.max()*1.002)
    for i in range(n):
        cl=UP if c[i]>=o[i] else DOWN;a.plot([i,i],[l[i],h[i]],color=cl,lw=0.6)
        bl,bh=min(o[i],c[i]),max(o[i],c[i])
        a.add_patch(Rectangle((i-0.35,bl),0.7,max(bh-bl,(h.max()-l.min())*0.002),fc=cl,ec="none"))
    a.axis("off")
    a=ax[1];a.set_facecolor(BG);a.bar(range(n),v/(v.max()+1e-10),color=[UP if c[i]>=o[i] else DOWN for i in range(n)],width=0.7,alpha=0.7);a.set_xlim(-0.5,n-0.5);a.axis("off")
    a=ax[2];a.set_facecolor(BG);a.plot(range(n),rsi,color=RSI_CLR,lw=0.8);a.axhline(70,color="#555",lw=0.3,ls="--");a.axhline(30,color="#555",lw=0.3,ls="--");a.set_xlim(-0.5,n-0.5);a.set_ylim(0,100);a.axis("off")
    plt.subplots_adjust(left=0,right=1,top=1,bottom=0);p.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(p,dpi=DPI,facecolor=BG,bbox_inches="tight",pad_inches=0);plt.close(fig)
 
def _r2p(args):
    od,ps=args;p=Path(ps)
    if p.exists():return
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    o,h,l,c,v=od["o"],od["h"],od["l"],od["c"],od["v"];n=len(o)
    if n<6:return
    fig,ax=plt.subplots(2,1,figsize=(FIG,FIG),dpi=DPI,gridspec_kw={"height_ratios":PANEL_2,"hspace":0})
    fig.patch.set_facecolor(BG)
    a=ax[0];a.set_facecolor(BG);a.set_xlim(-0.5,n-0.5);a.set_ylim(l.min()*0.995,h.max()*1.005)
    for i in range(n):
        cl=UP if c[i]>=o[i] else DOWN;a.plot([i,i],[l[i],h[i]],color=cl,lw=0.8)
        bl,bh=min(o[i],c[i]),max(o[i],c[i])
        a.add_patch(Rectangle((i-0.4,bl),0.8,max(bh-bl,(h.max()-l.min())*0.003),fc=cl,ec="none"))
    a.axis("off")
    a=ax[1];a.set_facecolor(BG);a.bar(range(n),v/(v.max()+1e-10),color=[UP if c[i]>=o[i] else DOWN for i in range(n)],width=0.8,alpha=0.7);a.set_xlim(-0.5,n-0.5);a.axis("off")
    plt.subplots_adjust(left=0,right=1,top=1,bottom=0);p.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(p,dpi=DPI,facecolor=BG,bbox_inches="tight",pad_inches=0);plt.close(fig)
 
def _rvix(args):
    va,ps=args;p=Path(ps)
    if p.exists():return
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    n=len(va)
    if n<10:return
    fig,ax=plt.subplots(figsize=(FIG,FIG),dpi=DPI)
    fig.patch.set_facecolor(BG);ax.set_facecolor(BG)
    ax.plot(range(n),va,color=VIX_CLR,lw=1.2);ax.fill_between(range(n),va,alpha=0.15,color=VIX_CLR)
    ax.axhline(20,color="#555",lw=0.3,ls="--");ax.axhline(30,color="#ef5350",lw=0.4,ls="--",alpha=0.6)
    ax.set_xlim(-0.5,n-0.5);ax.set_ylim(max(0,va.min()*0.9),va.max()*1.1);ax.axis("off")
    plt.subplots_adjust(left=0,right=1,top=1,bottom=0);p.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(p,dpi=DPI,facecolor=BG,bbox_inches="tight",pad_inches=0);plt.close(fig)
 
def render_batch_4p(jobs, n_workers=N_CORES):
    with Pool(n_workers) as pool: pool.map(_r4p, jobs, chunksize=64)
 
def render_batch_3p(jobs, n_workers=N_CORES):
    with Pool(n_workers) as pool: pool.map(_r3p, jobs, chunksize=64)
 
def render_batch_2p(jobs, n_workers=N_CORES):
    with Pool(n_workers) as pool: pool.map(_r2p, jobs, chunksize=64)
 
def render_batch_vix(jobs, n_workers=N_CORES):
    with Pool(n_workers) as pool: pool.map(_rvix, jobs, chunksize=64)
 
def render_candles_4p(ohlcv,rsi,hist,path): _r4p((_ohlcv_dict(ohlcv),rsi,hist,str(path)))
def render_candles_3p(ohlcv,rsi,path): _r3p((_ohlcv_dict(ohlcv),rsi,str(path)))
def render_candles_2p(ohlcv,path): _r2p((_ohlcv_dict(ohlcv),str(path)))
def render_vix_line(vix_vals,path):
    v=vix_vals.values if hasattr(vix_vals,'values') else np.array(vix_vals)
    _rvix((v,str(path)))