"""
analyze_rsi.py — Var 04 RSI Divergence (Analyzer)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS DOES:
  Generates diagnostic plots for the selected RSI divergence parameters.
  All output → V1/plots/ directory.
 
PLOTS:
  P01: Train/Val/Test IC 3-bar chart
  P02: IC decay curve vs 1/3/5/10/20 bar horizons
  P03: Label quality by year + touch_bar histogram
  P04: Signal ACF + Ljung-Box statistic
  P05: Ticker coverage by year
  P06: Rolling IC (252d) with train/val/test boundaries
  P07: Decile returns by RSI divergence score (Monotonic Staircase)
  P08: Bull vs Bear score distribution (histogram)
  P09: RSI anti-momentum validation (RSI IC vs MACD IC if available)
  P10: Parameter grid heatmap (IC across 12 combos)
 
CONSISTENCY:
  - Reads selected_params_{resolution}.json for chosen combo
  - Reads signal parquets from {resolution}/signals/
  - All matplotlib — this is the ONLY file that uses matplotlib for V04
  - Outputs to plots/ at V1 level
 
USAGE:
  python analyze_rsi.py --data_dir ../../00_Mini_Model/data/ohlcv --resolution daily
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
 
import argparse, json, logging, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
 
THIS = Path(__file__).resolve().parent
PLOTS = THIS / "plots"
sys.path.insert(0, str(THIS.parent.parent / "00_Mini_Model"))
import importlib as _il
B = _il.import_module("01_base_modules_")
 
TRAIN_END = B.TRAIN_END
VAL_END   = B.VAL_END
ATR_P     = B.ATR_P
 
 
def load_selected(resolution: str) -> dict:
    p = THIS / f"selected_params_{resolution}.json"
    if not p.exists():
        raise FileNotFoundError(f"Run trainer.py first: {p}")
    with open(p) as f:
        return json.load(f)
 
 
def load_signals_and_prices(data_dir: str, resolution: str,
                            combo_id: str) -> pd.DataFrame:
    """Load signal parquets for selected combo, merge with price data."""
    sig_dir = THIS / resolution / "signals"
    all_frames = []
 
    for sp in sorted(sig_dir.glob("*.parquet")):
        tk = sp.stem
        sdf = pd.read_parquet(sp)
        combo = sdf[sdf["combo_id"] == combo_id].copy()
        if len(combo) == 0:
            continue
 
        # Load price for forward returns
        pp = Path(data_dir) / resolution / f"{tk}.parquet"
        if not pp.exists():
            continue
        pdf = pd.read_parquet(pp)
        if "datetime" not in pdf.columns and pdf.index.name == "datetime":
            pdf = pdf.reset_index()
        pdf["datetime"] = pd.to_datetime(pdf["datetime"])
 
        merged = combo.merge(pdf[["datetime", "close", "high", "low"]],
                             on="datetime", how="inner")
        if len(merged) == 0:
            continue
 
        atr = B.wilder_atr(merged["high"], merged["low"], merged["close"])
        merged["atr"] = atr
        merged["net_score"] = merged["bull_rsi_score"] - merged["bear_rsi_score"]
 
        # Multiple forward return horizons
        for h in [1, 3, 5, 10, 20]:
            merged[f"fwd_{h}"] = merged["close"].pct_change(h).shift(-h)
 
        ts = pd.to_datetime(merged["datetime"])
        merged["split"] = "test"
        merged.loc[ts <= pd.Timestamp(TRAIN_END), "split"] = "train"
        merged.loc[(ts > pd.Timestamp(TRAIN_END)) & (ts <= pd.Timestamp(VAL_END)), "split"] = "val"
        merged["year"] = ts.dt.year
 
        all_frames.append(merged)
 
    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)
 
 
def plot_all(df: pd.DataFrame, params: dict, resolution: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    PLOTS.mkdir(parents=True, exist_ok=True)
    combo_id = params["selected"]["combo_id"]
 
    # ── P01: Train/Val/Test IC bar chart ──────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    splits = ["train", "val", "test"]
    ics = []
    for s in splits:
        mask = df["split"] == s
        if mask.sum() > 30:
            ic, _ = spearmanr(df.loc[mask, "net_score"], df.loc[mask, "fwd_5"])
            ics.append(ic if not np.isnan(ic) else 0)
        else:
            ics.append(0)
    colors = ["#26a69a", "#42a5f5", "#ef5350"]
    ax.bar(splits, ics, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title(f"V04 RSI IC by Split ({resolution} | {combo_id})")
    ax.set_ylabel("Spearman IC")
    ax.axhline(0, color="gray", lw=0.5)
    fig.tight_layout()
    fig.savefig(PLOTS / f"P01_ic_by_split_{resolution}.png", dpi=150)
    plt.close(fig)
 
    # ── P02: IC decay curve ───────────────────────────────────────────
    horizons = [1, 3, 5, 10, 20]
    decay_ics = []
    for h in horizons:
        col = f"fwd_{h}"
        mask = ~(df["net_score"].isna() | df[col].isna())
        if mask.sum() > 30:
            ic, _ = spearmanr(df.loc[mask, "net_score"], df.loc[mask, col])
            decay_ics.append(ic if not np.isnan(ic) else 0)
        else:
            decay_ics.append(0)
 
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(horizons, decay_ics, "o-", color="#ab47bc", lw=2, markersize=6)
    ax.set_xlabel("Forward Bars")
    ax.set_ylabel("Spearman IC")
    ax.set_title(f"V04 RSI IC Decay ({resolution} | {combo_id})")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xticks(horizons)
    fig.tight_layout()
    fig.savefig(PLOTS / f"P02_ic_decay_{resolution}.png", dpi=150)
    plt.close(fig)
 
    # ── P03: Label quality by year ────────────────────────────────────
    if "year" in df.columns and "fwd_5" in df.columns:
        years = sorted(df["year"].dropna().unique())
        yearly_ics = []
        for y in years:
            mask = df["year"] == y
            if mask.sum() > 30:
                ic, _ = spearmanr(df.loc[mask, "net_score"], df.loc[mask, "fwd_5"])
                yearly_ics.append(ic if not np.isnan(ic) else 0)
            else:
                yearly_ics.append(0)
 
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar([str(y) for y in years], yearly_ics, color="#ab47bc", alpha=0.7)
        ax.set_title(f"V04 RSI IC by Year ({resolution})")
        ax.set_ylabel("Spearman IC")
        ax.axhline(0, color="gray", lw=0.5)
        plt.xticks(rotation=45)
        fig.tight_layout()
        fig.savefig(PLOTS / f"P03_ic_by_year_{resolution}.png", dpi=150)
        plt.close(fig)
 
    # ── P04: Signal ACF ───────────────────────────────────────────────
    # Use first ticker's signal for autocorrelation
    first_tk = df["ticker"].iloc[0] if "ticker" in df.columns else None
    if first_tk:
        tk_sig = df[df["ticker"] == first_tk]["net_score"].dropna().values
        if len(tk_sig) > 100:
            max_lag = min(50, len(tk_sig) // 4)
            acf_vals = []
            for lag in range(1, max_lag + 1):
                c = np.corrcoef(tk_sig[:-lag], tk_sig[lag:])[0, 1]
                acf_vals.append(c if not np.isnan(c) else 0)
 
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(range(1, max_lag + 1), acf_vals, color="#42a5f5", alpha=0.7)
            # Ljung-Box approximate: Q = n * sum(acf^2)
            n = len(tk_sig)
            q_stat = n * sum(a ** 2 for a in acf_vals[:20])
            ax.set_title(f"V04 RSI Signal ACF ({first_tk} | {resolution}) | Q₂₀={q_stat:.1f}")
            ax.set_xlabel("Lag")
            ax.set_ylabel("Autocorrelation")
            ax.axhline(0, color="gray", lw=0.5)
            ax.axhline(1.96 / np.sqrt(n), color="red", lw=0.5, ls="--")
            ax.axhline(-1.96 / np.sqrt(n), color="red", lw=0.5, ls="--")
            fig.tight_layout()
            fig.savefig(PLOTS / f"P04_signal_acf_{resolution}.png", dpi=150)
            plt.close(fig)
 
    # ── P05: Ticker coverage by year ──────────────────────────────────
    if "year" in df.columns:
        cov = df.groupby("year")["ticker"].nunique()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(cov.index.astype(str), cov.values, color="#26a69a", alpha=0.7)
        ax.set_title(f"V04 RSI Ticker Coverage ({resolution})")
        ax.set_ylabel("Unique Tickers")
        plt.xticks(rotation=45)
        fig.tight_layout()
        fig.savefig(PLOTS / f"P05_ticker_coverage_{resolution}.png", dpi=150)
        plt.close(fig)
 
    # ── P06: Rolling IC (252d) ────────────────────────────────────────
    df_sorted = df.sort_values("datetime").reset_index(drop=True)
    window = 252 if resolution == "daily" else 52
    roll_dates, roll_ics = [], []
    for i in range(window, len(df_sorted), window // 4):
        chunk = df_sorted.iloc[i - window:i]
        mask = ~(chunk["net_score"].isna() | chunk["fwd_5"].isna())
        if mask.sum() > 30:
            ic, _ = spearmanr(chunk.loc[mask, "net_score"], chunk.loc[mask, "fwd_5"])
            roll_dates.append(chunk["datetime"].iloc[-1])
            roll_ics.append(ic if not np.isnan(ic) else 0)
 
    if roll_dates:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(roll_dates, roll_ics, color="#ab47bc", lw=1)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(pd.Timestamp(TRAIN_END), color="blue", lw=1, ls="--", label="Train end")
        ax.axvline(pd.Timestamp(VAL_END), color="red", lw=1, ls="--", label="Val end")
        ax.set_title(f"V04 RSI Rolling IC ({resolution} | window={window})")
        ax.set_ylabel("Spearman IC")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS / f"P06_rolling_ic_{resolution}.png", dpi=150)
        plt.close(fig)
 
    # ── P07: Decile returns (Monotonic Staircase) ─────────────────────
    valid = df.dropna(subset=["net_score", "fwd_5"])
    if len(valid) > 100:
        valid = valid.copy()
        valid["decile"] = pd.qcut(valid["net_score"], 10, labels=False, duplicates="drop")
        dec_ret = valid.groupby("decile")["fwd_5"].mean()
 
        fig, ax = plt.subplots(figsize=(8, 5))
        colors_d = ["#ef5350" if v < 0 else "#26a69a" for v in dec_ret.values]
        ax.bar(dec_ret.index, dec_ret.values * 100, color=colors_d, edgecolor="black", lw=0.3)
        ax.set_xlabel("RSI Divergence Decile (0=bearish, 9=bullish)")
        ax.set_ylabel("Mean 5-bar Forward Return (%)")
        ax.set_title(f"V04 RSI Monotonic Staircase ({resolution})")
        ax.axhline(0, color="gray", lw=0.5)
        fig.tight_layout()
        fig.savefig(PLOTS / f"P07_decile_returns_{resolution}.png", dpi=150)
        plt.close(fig)
 
    # ── P08: Score distribution ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df["bull_rsi_score"].dropna(), bins=50, color="#26a69a", alpha=0.7)
    axes[0].set_title("Bull RSI Score")
    axes[1].hist(df["bear_rsi_score"].dropna(), bins=50, color="#ef5350", alpha=0.7)
    axes[1].set_title("Bear RSI Score")
    fig.suptitle(f"V04 RSI Score Distribution ({resolution} | {combo_id})")
    fig.tight_layout()
    fig.savefig(PLOTS / f"P08_score_distribution_{resolution}.png", dpi=150)
    plt.close(fig)
 
    # ── P10: Parameter grid heatmap ───────────────────────────────────
    all_combos = params.get("all_combos", [])
    if all_combos:
        periods = sorted(set(c["rsi_period"] for c in all_combos))
        lookbacks = sorted(set(c["div_lookback"] for c in all_combos))
        grid = np.full((len(periods), len(lookbacks)), np.nan)
        for c in all_combos:
            pi = periods.index(c["rsi_period"])
            li = lookbacks.index(c["div_lookback"])
            grid[pi, li] = c["ic_val"]
 
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(grid, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(lookbacks)))
        ax.set_xticklabels(lookbacks)
        ax.set_yticks(range(len(periods)))
        ax.set_yticklabels(periods)
        ax.set_xlabel("Divergence Lookback")
        ax.set_ylabel("RSI Period")
        ax.set_title(f"V04 RSI Val IC Grid ({resolution})")
        for i in range(len(periods)):
            for j in range(len(lookbacks)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, label="IC (val)")
        fig.tight_layout()
        fig.savefig(PLOTS / f"P10_param_grid_{resolution}.png", dpi=150)
        plt.close(fig)
 
    logging.info(f"✓ All plots saved to {PLOTS}/")
 
 
def main():
    p = argparse.ArgumentParser(description="V04 RSI Divergence — Analyzer")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--resolution", choices=["daily", "weekly", "15min", "1hour"],
                   required=True)
    args = p.parse_args()
 
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    logging.info(f"V04 RSI Analyzer | Resolution: {args.resolution}")
 
    params = load_selected(args.resolution)
    combo_id = params["selected"]["combo_id"]
    logging.info(f"  Selected combo: {combo_id}")
 
    df = load_signals_and_prices(args.data_dir, args.resolution, combo_id)
    if len(df) == 0:
        logging.error("No data loaded.")
        return
 
    logging.info(f"  Rows: {len(df):,} | Tickers: {df['ticker'].nunique()}")
    plot_all(df, params, args.resolution)
 
 
if __name__ == "__main__":
    main()