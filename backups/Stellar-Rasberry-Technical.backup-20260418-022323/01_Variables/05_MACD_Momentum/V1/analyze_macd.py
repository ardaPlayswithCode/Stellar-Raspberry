"""
analyze_macd.py — Var 05 MACD Momentum (Analyzer)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS DOES:
  Diagnostic plots for selected MACD combo. All output → V1/plots/.
 
PLOTS:
  P01: Train/Val/Test IC 3-bar chart
  P02: IC decay curve vs 1/3/5/10/20 bar horizons
  P03: IC by year
  P04: Signal ACF + Ljung-Box
  P05: Ticker coverage by year
  P06: Rolling IC with split boundaries
  P07: Decile returns (Monotonic Staircase)
  P08: Score distribution (bull/bear)
  P09: DHS derivative analysis (hist_delta_norm IC + quadrant plot)
  P10: Parameter grid heatmap (IC across 24 combos)
 
USAGE:
  python analyze_macd.py --data_dir ../../00_Mini_Model/data/ohlcv --resolution daily
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
 
 
def load_selected(resolution):
    p = THIS / f"selected_params_{resolution}.json"
    if not p.exists():
        raise FileNotFoundError(f"Run trainer.py first: {p}")
    with open(p) as f:
        return json.load(f)
 
 
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
 
        merged = combo.merge(pdf[["datetime", "close", "high", "low"]], on="datetime", how="inner")
        if len(merged) == 0: continue
 
        atr = B.wilder_atr(merged["high"], merged["low"], merged["close"])
        merged["atr"] = atr
        merged["net_score"] = merged["bull_macd_score"] - merged["bear_macd_score"]
        for h in [1, 3, 5, 10, 20]:
            merged[f"fwd_{h}"] = merged["close"].pct_change(h).shift(-h)
 
        ts = pd.to_datetime(merged["datetime"])
        merged["split"] = "test"
        merged.loc[ts <= pd.Timestamp(TRAIN_END), "split"] = "train"
        merged.loc[(ts > pd.Timestamp(TRAIN_END)) & (ts <= pd.Timestamp(VAL_END)), "split"] = "val"
        merged["year"] = ts.dt.year
        frames.append(merged)
 
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
 
 
def plot_all(df, params, resolution):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    PLOTS.mkdir(parents=True, exist_ok=True)
    cid = params["selected"]["combo_id"]
 
    # P01: IC by split
    fig, ax = plt.subplots(figsize=(6, 4))
    splits = ["train", "val", "test"]
    ics = []
    for s in splits:
        m = df["split"] == s
        if m.sum() > 30:
            ic, _ = spearmanr(df.loc[m, "net_score"], df.loc[m, "fwd_5"])
            ics.append(ic if not np.isnan(ic) else 0)
        else:
            ics.append(0)
    ax.bar(splits, ics, color=["#26a69a", "#42a5f5", "#ef5350"], edgecolor="black", lw=0.5)
    ax.set_title(f"V05 MACD IC by Split ({resolution} | {cid})")
    ax.set_ylabel("Spearman IC"); ax.axhline(0, color="gray", lw=0.5)
    fig.tight_layout(); fig.savefig(PLOTS / f"P01_ic_by_split_{resolution}.png", dpi=150); plt.close(fig)
 
    # P02: IC decay
    horizons = [1, 3, 5, 10, 20]
    decay = []
    for h in horizons:
        m = ~(df["net_score"].isna() | df[f"fwd_{h}"].isna())
        ic, _ = spearmanr(df.loc[m, "net_score"], df.loc[m, f"fwd_{h}"]) if m.sum() > 30 else (0, 0)
        decay.append(ic if not np.isnan(ic) else 0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(horizons, decay, "o-", color="#e67e22", lw=2, markersize=6)
    ax.set_xlabel("Forward Bars"); ax.set_ylabel("Spearman IC")
    ax.set_title(f"V05 MACD IC Decay ({resolution} | {cid})")
    ax.axhline(0, color="gray", lw=0.5); ax.set_xticks(horizons)
    fig.tight_layout(); fig.savefig(PLOTS / f"P02_ic_decay_{resolution}.png", dpi=150); plt.close(fig)
 
    # P03: IC by year
    years = sorted(df["year"].dropna().unique())
    y_ics = []
    for y in years:
        m = df["year"] == y
        if m.sum() > 30:
            ic, _ = spearmanr(df.loc[m, "net_score"], df.loc[m, "fwd_5"])
            y_ics.append(ic if not np.isnan(ic) else 0)
        else:
            y_ics.append(0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([str(y) for y in years], y_ics, color="#e67e22", alpha=0.7)
    ax.set_title(f"V05 MACD IC by Year ({resolution})"); ax.set_ylabel("IC")
    ax.axhline(0, color="gray", lw=0.5); plt.xticks(rotation=45)
    fig.tight_layout(); fig.savefig(PLOTS / f"P03_ic_by_year_{resolution}.png", dpi=150); plt.close(fig)
 
    # P04: ACF
    first_tk = df["ticker"].iloc[0]
    tk_sig = df[df["ticker"] == first_tk]["net_score"].dropna().values
    if len(tk_sig) > 100:
        max_lag = min(50, len(tk_sig) // 4)
        acf = [np.corrcoef(tk_sig[:-lag], tk_sig[lag:])[0, 1] if lag < len(tk_sig) else 0
               for lag in range(1, max_lag + 1)]
        acf = [a if not np.isnan(a) else 0 for a in acf]
        n = len(tk_sig)
        q = n * sum(a**2 for a in acf[:20])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(1, max_lag + 1), acf, color="#42a5f5", alpha=0.7)
        ax.set_title(f"V05 MACD ACF ({first_tk} | {resolution}) | Q₂₀={q:.1f}")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axhline(1.96/np.sqrt(n), color="red", lw=0.5, ls="--")
        ax.axhline(-1.96/np.sqrt(n), color="red", lw=0.5, ls="--")
        fig.tight_layout(); fig.savefig(PLOTS / f"P04_signal_acf_{resolution}.png", dpi=150); plt.close(fig)
 
    # P05: Coverage
    cov = df.groupby("year")["ticker"].nunique()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(cov.index.astype(str), cov.values, color="#26a69a", alpha=0.7)
    ax.set_title(f"V05 MACD Ticker Coverage ({resolution})"); ax.set_ylabel("Unique Tickers")
    plt.xticks(rotation=45)
    fig.tight_layout(); fig.savefig(PLOTS / f"P05_ticker_coverage_{resolution}.png", dpi=150); plt.close(fig)
 
    # P06: Rolling IC
    ds = df.sort_values("datetime").reset_index(drop=True)
    window = 252 if resolution in ("daily", "15min", "1hour") else 52
    rd, ri = [], []
    for i in range(window, len(ds), window // 4):
        ch = ds.iloc[i - window:i]
        m = ~(ch["net_score"].isna() | ch["fwd_5"].isna())
        if m.sum() > 30:
            ic, _ = spearmanr(ch.loc[m, "net_score"], ch.loc[m, "fwd_5"])
            rd.append(ch["datetime"].iloc[-1]); ri.append(ic if not np.isnan(ic) else 0)
    if rd:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(rd, ri, color="#e67e22", lw=1); ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(pd.Timestamp(TRAIN_END), color="blue", lw=1, ls="--", label="Train end")
        ax.axvline(pd.Timestamp(VAL_END), color="red", lw=1, ls="--", label="Val end")
        ax.set_title(f"V05 MACD Rolling IC ({resolution})"); ax.legend()
        fig.tight_layout(); fig.savefig(PLOTS / f"P06_rolling_ic_{resolution}.png", dpi=150); plt.close(fig)
 
    # P07: Decile returns
    valid = df.dropna(subset=["net_score", "fwd_5"]).copy()
    if len(valid) > 100:
        valid["decile"] = pd.qcut(valid["net_score"], 10, labels=False, duplicates="drop")
        dr = valid.groupby("decile")["fwd_5"].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors_d = ["#ef5350" if v < 0 else "#26a69a" for v in dr.values]
        ax.bar(dr.index, dr.values * 100, color=colors_d, edgecolor="black", lw=0.3)
        ax.set_xlabel("MACD Decile (0=bearish, 9=bullish)")
        ax.set_ylabel("Mean 5-bar Fwd Return (%)")
        ax.set_title(f"V05 MACD Monotonic Staircase ({resolution})")
        ax.axhline(0, color="gray", lw=0.5)
        fig.tight_layout(); fig.savefig(PLOTS / f"P07_decile_returns_{resolution}.png", dpi=150); plt.close(fig)
 
    # P08: Score distribution
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df["bull_macd_score"].dropna(), bins=50, color="#26a69a", alpha=0.7)
    axes[0].set_title("Bull MACD Score")
    axes[1].hist(df["bear_macd_score"].dropna(), bins=50, color="#ef5350", alpha=0.7)
    axes[1].set_title("Bear MACD Score")
    fig.suptitle(f"V05 Score Distribution ({resolution} | {cid})")
    fig.tight_layout(); fig.savefig(PLOTS / f"P08_score_distribution_{resolution}.png", dpi=150); plt.close(fig)
 
    # P09: DHS derivative quadrant analysis
    if "hist_delta_norm" in df.columns:
        valid_dhs = df.dropna(subset=["net_score", "hist_delta_norm", "fwd_5"]).copy()
        if len(valid_dhs) > 100:
            fig, ax = plt.subplots(figsize=(7, 6))
            # 4 quadrants: MACD direction × DHS acceleration
            q1 = valid_dhs[(valid_dhs["net_score"] > 0) & (valid_dhs["hist_delta_norm"] > 0)]["fwd_5"].mean()
            q2 = valid_dhs[(valid_dhs["net_score"] > 0) & (valid_dhs["hist_delta_norm"] <= 0)]["fwd_5"].mean()
            q3 = valid_dhs[(valid_dhs["net_score"] <= 0) & (valid_dhs["hist_delta_norm"] > 0)]["fwd_5"].mean()
            q4 = valid_dhs[(valid_dhs["net_score"] <= 0) & (valid_dhs["hist_delta_norm"] <= 0)]["fwd_5"].mean()
 
            labels = ["Bull+Accel", "Bull+Decel", "Bear+Accel", "Bear+Decel"]
            vals = [q1 * 100, q2 * 100, q3 * 100, q4 * 100]
            colors = ["#26a69a", "#a5d6a7", "#ef9a9a", "#ef5350"]
            ax.bar(labels, vals, color=colors, edgecolor="black", lw=0.5)
            ax.set_ylabel("Mean 5-bar Fwd Return (%)")
            ax.set_title(f"V05 DHS Quadrant Analysis ({resolution})")
            ax.axhline(0, color="gray", lw=0.5)
            fig.tight_layout(); fig.savefig(PLOTS / f"P09_dhs_quadrant_{resolution}.png", dpi=150); plt.close(fig)
 
    # P10: Parameter grid heatmap
    all_combos = params.get("all_combos", [])
    if all_combos:
        # Use fast×slow (aggregated over signal periods)
        pairs = {}
        for c in all_combos:
            key = (c["fast"], c["slow"])
            if key not in pairs:
                pairs[key] = []
            pairs[key].append(c["ic_val"])
        # Average IC across signal periods
        fasts = sorted(set(k[0] for k in pairs))
        slows = sorted(set(k[1] for k in pairs))
        grid = np.full((len(fasts), len(slows)), np.nan)
        for (f, s), ics_list in pairs.items():
            fi = fasts.index(f); si = slows.index(s)
            grid[fi, si] = np.mean(ics_list)
 
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(grid, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(slows))); ax.set_xticklabels(slows)
        ax.set_yticks(range(len(fasts))); ax.set_yticklabels(fasts)
        ax.set_xlabel("Slow EMA"); ax.set_ylabel("Fast EMA")
        ax.set_title(f"V05 MACD Val IC Grid ({resolution}) — avg over signal periods")
        for i in range(len(fasts)):
            for j in range(len(slows)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, label="IC (val)")
        fig.tight_layout(); fig.savefig(PLOTS / f"P10_param_grid_{resolution}.png", dpi=150); plt.close(fig)
 
    logging.info(f"✓ All plots saved to {PLOTS}/")
 
 
def main():
    p = argparse.ArgumentParser(description="V05 MACD Momentum — Analyzer")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--resolution", choices=["daily", "weekly", "15min", "1hour"], required=True)
    args = p.parse_args()
 
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    logging.info(f"V05 MACD Analyzer | Resolution: {args.resolution}")
 
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