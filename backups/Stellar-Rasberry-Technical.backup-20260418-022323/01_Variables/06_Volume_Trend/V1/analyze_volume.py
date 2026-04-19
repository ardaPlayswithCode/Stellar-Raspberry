"""
analyze_volume.py — Var 06 Volume Trend (Analyzer)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PLOTS (→ V1/plots/):
  P01: Train/Val/Test IC bar chart
  P02: IC decay vs 1/3/5/10/20 bar horizons
  P03: IC by year
  P04: Signal ACF + Ljung-Box
  P05: Ticker coverage by year
  P06: Rolling IC with split boundaries
  P07: Decile returns (Monotonic Staircase)
  P08: Score distribution (bull/bear)
  P09: GKM conditioning — IC for normal-return vs all bars
  P10: Hendershott AT-era regime — IC pre/post Jan 2003
  P11: Parameter grid heatmap (18 combos)
 
USAGE:
  python analyze_volume.py --data_dir ../../00_Mini_Model/data/ohlcv --resolution daily
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
 
import argparse, json, logging, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
 
THIS  = Path(__file__).resolve().parent
PLOTS = THIS / "plots"
sys.path.insert(0, str(THIS.parent.parent / "00_Mini_Model"))
import importlib as _il
B = _il.import_module("01_base_modules_")
 
TRAIN_END = B.TRAIN_END
VAL_END   = B.VAL_END
GKM_PCTL  = 0.30
AT_DATE   = pd.Timestamp("2003-01-01")  # Hendershott NYSE autoquote
 
 
def load_selected(res):
    p = THIS / f"selected_params_{res}.json"
    if not p.exists(): raise FileNotFoundError(f"Run trainer.py first: {p}")
    with open(p) as f: return json.load(f)
 
 
def load_signals_and_prices(data_dir, resolution, combo_id):
    sig_dir = THIS / resolution / "signals"
    frames = []
    for sp in sorted(sig_dir.glob("*.parquet")):
        tk = sp.stem
        sdf = pd.read_parquet(sp)
        combo = sdf[sdf["combo_id"] == combo_id].copy()
        if len(combo) == 0: continue
        pp = Path(data_dir) / resolution / f"{tk}.parquet"
        if not pp.exists(): continue
        pdf = pd.read_parquet(pp)
        if "datetime" not in pdf.columns and pdf.index.name == "datetime":
            pdf = pdf.reset_index()
        pdf["datetime"] = pd.to_datetime(pdf["datetime"])
        merged = combo.merge(pdf[["datetime", "close", "high", "low", "volume"]],
                             on="datetime", how="inner")
        if len(merged) == 0: continue
 
        merged["net_score"] = merged["bull_vol_score"] - merged["bear_vol_score"]
        merged["same_bar_ret_abs"] = merged["close"].pct_change().abs()
        merged["ret_rank"] = merged["same_bar_ret_abs"].rolling(60, min_periods=20).rank(pct=True)
        for h in [1, 3, 5, 10, 20]:
            merged[f"fwd_{h}"] = merged["close"].pct_change(h).shift(-h)
        ts = pd.to_datetime(merged["datetime"])
        merged["split"] = "test"
        merged.loc[ts <= pd.Timestamp(TRAIN_END), "split"] = "train"
        merged.loc[(ts > pd.Timestamp(TRAIN_END)) & (ts <= pd.Timestamp(VAL_END)), "split"] = "val"
        merged["year"] = ts.dt.year
        merged["ticker"] = tk
        frames.append(merged)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
 
 
def ic(s, r):
    m = ~(np.isnan(s) | np.isnan(r))
    if m.sum() < 30: return 0
    v, _ = spearmanr(s[m], r[m])
    return v if not np.isnan(v) else 0
 
 
def plot_all(df, params, resolution):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTS.mkdir(parents=True, exist_ok=True)
    cid = params["selected"]["combo_id"]
 
    # P01: IC by split
    fig, ax = plt.subplots(figsize=(6, 4))
    ics = [ic(df.loc[df["split"]==s, "net_score"].values, df.loc[df["split"]==s, "fwd_5"].values)
           for s in ["train","val","test"]]
    ax.bar(["train","val","test"], ics, color=["#26a69a","#42a5f5","#ef5350"], edgecolor="black", lw=0.5)
    ax.set_title(f"V06 Volume IC by Split ({resolution} | {cid})"); ax.set_ylabel("IC")
    ax.axhline(0, color="gray", lw=0.5)
    fig.tight_layout(); fig.savefig(PLOTS/f"P01_ic_by_split_{resolution}.png", dpi=150); plt.close(fig)
 
    # P02: IC decay
    horizons = [1,3,5,10,20]
    decay = [ic(df["net_score"].dropna().values,
                df.loc[df["net_score"].notna(), f"fwd_{h}"].values) for h in horizons]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(horizons, decay, "o-", color="#26a69a", lw=2, ms=6)
    ax.set_xlabel("Forward Bars"); ax.set_ylabel("IC"); ax.set_xticks(horizons)
    ax.set_title(f"V06 Volume IC Decay ({resolution})"); ax.axhline(0, color="gray", lw=0.5)
    fig.tight_layout(); fig.savefig(PLOTS/f"P02_ic_decay_{resolution}.png", dpi=150); plt.close(fig)
 
    # P03: IC by year
    years = sorted(df["year"].dropna().unique())
    y_ics = [ic(df.loc[df["year"]==y, "net_score"].values, df.loc[df["year"]==y, "fwd_5"].values)
             for y in years]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([str(y) for y in years], y_ics, color="#26a69a", alpha=0.7)
    ax.set_title(f"V06 Volume IC by Year ({resolution})"); ax.axhline(0, color="gray", lw=0.5)
    plt.xticks(rotation=45)
    fig.tight_layout(); fig.savefig(PLOTS/f"P03_ic_by_year_{resolution}.png", dpi=150); plt.close(fig)
 
    # P04: ACF
    first_tk = df["ticker"].iloc[0]
    ts = df[df["ticker"]==first_tk]["net_score"].dropna().values
    if len(ts) > 100:
        ml = min(50, len(ts)//4)
        acf = [np.corrcoef(ts[:-l], ts[l:])[0,1] if l < len(ts) else 0 for l in range(1, ml+1)]
        acf = [a if not np.isnan(a) else 0 for a in acf]
        q = len(ts) * sum(a**2 for a in acf[:20])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(1, ml+1), acf, color="#42a5f5", alpha=0.7)
        ax.set_title(f"V06 ACF ({first_tk} | {resolution}) | Q₂₀={q:.1f}")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axhline(1.96/np.sqrt(len(ts)), color="red", lw=0.5, ls="--")
        ax.axhline(-1.96/np.sqrt(len(ts)), color="red", lw=0.5, ls="--")
        fig.tight_layout(); fig.savefig(PLOTS/f"P04_signal_acf_{resolution}.png", dpi=150); plt.close(fig)
 
    # P05: Coverage
    cov = df.groupby("year")["ticker"].nunique()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(cov.index.astype(str), cov.values, color="#26a69a", alpha=0.7)
    ax.set_title(f"V06 Ticker Coverage ({resolution})"); plt.xticks(rotation=45)
    fig.tight_layout(); fig.savefig(PLOTS/f"P05_ticker_coverage_{resolution}.png", dpi=150); plt.close(fig)
 
    # P06: Rolling IC
    ds = df.sort_values("datetime").reset_index(drop=True)
    w = 252 if resolution in ("daily","15min","1hour") else 52
    rd, ri = [], []
    for i in range(w, len(ds), w//4):
        ch = ds.iloc[i-w:i]
        m = ~(ch["net_score"].isna()|ch["fwd_5"].isna())
        if m.sum()>30:
            v,_ = spearmanr(ch.loc[m,"net_score"], ch.loc[m,"fwd_5"])
            rd.append(ch["datetime"].iloc[-1]); ri.append(v if not np.isnan(v) else 0)
    if rd:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(rd, ri, color="#26a69a", lw=1); ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(pd.Timestamp(TRAIN_END), color="blue", lw=1, ls="--", label="Train end")
        ax.axvline(pd.Timestamp(VAL_END), color="red", lw=1, ls="--", label="Val end")
        ax.set_title(f"V06 Rolling IC ({resolution})"); ax.legend()
        fig.tight_layout(); fig.savefig(PLOTS/f"P06_rolling_ic_{resolution}.png", dpi=150); plt.close(fig)
 
    # P07: Decile returns
    valid = df.dropna(subset=["net_score","fwd_5"]).copy()
    if len(valid) > 100:
        valid["decile"] = pd.qcut(valid["net_score"], 10, labels=False, duplicates="drop")
        dr = valid.groupby("decile")["fwd_5"].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        c = ["#ef5350" if v<0 else "#26a69a" for v in dr.values]
        ax.bar(dr.index, dr.values*100, color=c, edgecolor="black", lw=0.3)
        ax.set_xlabel("Volume Shock Decile"); ax.set_ylabel("Mean 5-bar Fwd Ret (%)")
        ax.set_title(f"V06 Monotonic Staircase ({resolution})"); ax.axhline(0, color="gray", lw=0.5)
        fig.tight_layout(); fig.savefig(PLOTS/f"P07_decile_returns_{resolution}.png", dpi=150); plt.close(fig)
 
    # P08: Score distribution
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df["bull_vol_score"].dropna(), bins=50, color="#26a69a", alpha=0.7)
    axes[0].set_title("Bull Vol Score")
    axes[1].hist(df["bear_vol_score"].dropna(), bins=50, color="#ef5350", alpha=0.7)
    axes[1].set_title("Bear Vol Score")
    fig.suptitle(f"V06 Score Distribution ({resolution} | {cid})")
    fig.tight_layout(); fig.savefig(PLOTS/f"P08_score_distribution_{resolution}.png", dpi=150); plt.close(fig)
 
    # P09: GKM conditioning — IC for normal-return vs all bars
    fig, ax = plt.subplots(figsize=(6, 5))
    all_ic = ic(df["net_score"].values, df["fwd_5"].values)
    normal_mask = df["ret_rank"] < GKM_PCTL
    gkm_ic = ic(df.loc[normal_mask, "net_score"].values, df.loc[normal_mask, "fwd_5"].values)
    abnormal_mask = df["ret_rank"] >= GKM_PCTL
    abn_ic = ic(df.loc[abnormal_mask, "net_score"].values, df.loc[abnormal_mask, "fwd_5"].values)
    labels = ["All Bars", f"Normal\n(|ret| rank<{int(GKM_PCTL*100)}%)", f"Abnormal\n(|ret| rank≥{int(GKM_PCTL*100)}%)"]
    vals = [all_ic, gkm_ic, abn_ic]
    colors = ["#42a5f5", "#26a69a", "#ef5350"]
    ax.bar(labels, vals, color=colors, edgecolor="black", lw=0.5)
    ax.set_ylabel("Spearman IC")
    ax.set_title(f"V06 GKM Conditioning ({resolution})\nGKM (2001): normal-return bars ~double IC")
    ax.axhline(0, color="gray", lw=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.001 if v >= 0 else v - 0.003, f"{v:.4f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(PLOTS/f"P09_gkm_conditioning_{resolution}.png", dpi=150); plt.close(fig)
 
    # P10: AT-era regime (Hendershott 2011: Jan 2003 NYSE autoquote)
    pre_at = df[pd.to_datetime(df["datetime"]) < AT_DATE]
    post_at = df[pd.to_datetime(df["datetime"]) >= AT_DATE]
    fig, ax = plt.subplots(figsize=(6, 5))
    ic_pre = ic(pre_at["net_score"].values, pre_at["fwd_5"].values) if len(pre_at) > 100 else 0
    ic_post = ic(post_at["net_score"].values, post_at["fwd_5"].values) if len(post_at) > 100 else 0
    ax.bar(["Pre-AT\n(<2003)", "Post-AT\n(≥2003)"], [ic_pre, ic_post],
           color=["#ffcc80", "#42a5f5"], edgecolor="black", lw=0.5)
    ax.set_ylabel("Spearman IC")
    ax.set_title(f"V06 AT-Era Regime ({resolution})\nHendershott (2011): AT = ~73% vol by 2009")
    ax.axhline(0, color="gray", lw=0.5)
    fig.tight_layout(); fig.savefig(PLOTS/f"P10_at_era_regime_{resolution}.png", dpi=150); plt.close(fig)
 
    # P11: Parameter grid heatmap
    all_combos = params.get("all_combos", [])
    if all_combos:
        lookbacks = sorted(set(c["lookback"] for c in all_combos))
        norms = sorted(set(c["norm_method"] for c in all_combos))
        # Average over EWM flag
        grid = np.full((len(lookbacks), len(norms)), np.nan)
        for c in all_combos:
            li = lookbacks.index(c["lookback"])
            ni = norms.index(c["norm_method"])
            if np.isnan(grid[li, ni]):
                grid[li, ni] = c["ic_val"]
            else:
                grid[li, ni] = (grid[li, ni] + c["ic_val"]) / 2
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(grid, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(norms))); ax.set_xticklabels(norms)
        ax.set_yticks(range(len(lookbacks))); ax.set_yticklabels(lookbacks)
        ax.set_xlabel("Normalization"); ax.set_ylabel("Lookback")
        ax.set_title(f"V06 Val IC Grid ({resolution}) — avg over EWM")
        for i in range(len(lookbacks)):
            for j in range(len(norms)):
                if not np.isnan(grid[i,j]):
                    ax.text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, label="IC (val)")
        fig.tight_layout(); fig.savefig(PLOTS/f"P11_param_grid_{resolution}.png", dpi=150); plt.close(fig)
 
    logging.info(f"✓ All plots saved to {PLOTS}/")
 
 
def main():
    p = argparse.ArgumentParser(description="V06 Volume Trend — Analyzer")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--resolution", choices=["daily","weekly","15min","1hour"], required=True)
    args = p.parse_args()
 
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    logging.info(f"V06 Volume Analyzer | Resolution: {args.resolution}")
 
    params = load_selected(args.resolution)
    cid = params["selected"]["combo_id"]
    logging.info(f"  Selected: {cid}")
 
    df = load_signals_and_prices(args.data_dir, args.resolution, cid)
    if len(df) == 0:
        logging.error("No data loaded."); return
    logging.info(f"  Rows: {len(df):,} | Tickers: {df['ticker'].nunique()}")
    plot_all(df, params, args.resolution)
 
 
if __name__ == "__main__":
    main()