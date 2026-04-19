01_Stock_Geometry/V1/analyze_cnn.py
=====================================
V1 Stock Geometry — Cross-Resolution Analysis Suite.
 
Plots:
  S01: Resolution IC comparison
  S02: Physics drift audit
  S03: Rolling IC (252-day window)
  S04: Decile returns by CNN confidence
  S05: Coverage heatmap
  S06: Label distribution by resolution
=====================================
"""
 
import argparse, logging
from pathlib import Path
import numpy as np, pandas as pd
 
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import PercentFormatter
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
 
import importlib as _il
# Note: Ensure the path to 01_base_modules_ is in your PYTHONPATH
B = _il.import_module("01_base_modules_")
 
CLR = {
    "train": "#2c3e50", "val": "#f39c12", "test": "#e74c3c",
    "bull": "#2ecc71", "neutral": "#95a5a6", "bear": "#e74c3c",
    "15min": "#3498db", "1hour": "#2ecc71", "daily": "#9b59b6",
    "weekly": "#e67e22", "monthly": "#e74c3c",
}
 
# Anchor to the directory where this script lives (V1 folder)
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = THIS_DIR / "plots"
 
class V1Analyzer:
    def __init__(self, signal_path: str, output_dir: str = str(DEFAULT_OUT)):
        self.df = pd.read_parquet(signal_path)
        if "datetime" in self.df.columns:
            self.df["datetime"] = pd.to_datetime(self.df["datetime"])
        
        # Ensures 'plots/' folder is created exactly under V1/
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        
        self.train = self.df[self.df["split"] == "train"]
 
        if HAS_MPL:
            plt.rcParams.update({
                "figure.figsize": (14, 8), "font.size": 10,
                "axes.grid": True, "grid.alpha": 0.3,
            })
 
        logging.info(f"V1 Analyzer loaded: {len(self.df):,} rows | "
                     f"Train: {len(self.train):,}")
 
    def s01_resolution_ic(self):
        """Compare Spearman IC across all 5 resolutions."""
        if not HAS_MPL or "forward_ret_5" not in self.train.columns:
            return
 
        bull_cols = sorted([c for c in self.train.columns
                            if "_bull" in c and "v1_" in c])
        if not bull_cols:
            return
 
        fig, ax = plt.subplots(figsize=(10, 6))
        names, vals, colors = [], [], []
 
        for col in bull_cols:
            sub = self.train[[col, "forward_ret_5"]].dropna()
            if len(sub) < 30:
                continue
            ic = sub[col].corr(sub["forward_ret_5"], method="spearman")
            res_name = col.replace("v1_", "").replace("_bull", "")
            names.append(res_name)
            vals.append(ic)
            colors.append(CLR.get(res_name, "#3498db"))
 
        ax.bar(range(len(vals)), vals, color=colors, alpha=0.7)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Spearman IC")
        ax.set_title("S01: V1 Bull Probability IC by Resolution (Train)")
        ax.axhline(0, color="black", ls="--", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9)
 
        plt.tight_layout()
        plt.savefig(self.out / "S01_resolution_ic.png", dpi=150)
        plt.close()
        logging.info("  ✓ S01 Resolution IC")
 
    def s02_physics_drift(self):
        """Scatter: 15min bull_prob vs daily bull_prob."""
        if not HAS_MPL:
            return
 
        b15 = [c for c in self.train.columns if "v1_15min_bull" in c]
        bd  = [c for c in self.train.columns if "v1_daily_bull" in c]
        if not b15 or not bd:
            return
 
        sub = self.train[[b15[0], bd[0]]].dropna()
        if len(sub) < 100:
            return
 
        sample = sub.sample(min(5000, len(sub)), random_state=42)
        rho = sample[b15[0]].corr(sample[bd[0]], method="spearman")
 
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(sample[bd[0]], sample[b15[0]],
                   alpha=0.1, s=2, color=CLR["15min"])
        ax.plot([0, 1], [0, 1], "--", color="#555", lw=0.5, alpha=0.5)
        ax.set_xlabel("Daily Bull Prob")
        ax.set_ylabel("15min Bull Prob")
        ax.set_title(f"S02: Physics Drift — 15min vs Daily (ρ={rho:.3f})\n"
                     f"Same RSI(14)/MACD(12,26,9) at different scales")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(self.out / "S02_physics_drift.png", dpi=150)
        plt.close()
        logging.info(f"  ✓ S02 Physics Drift (ρ={rho:.3f})")
 
    def s03_rolling_ic(self):
        """Rolling 252-day IC with temporal split markers."""
        if not HAS_MPL or "forward_ret_5" not in self.df.columns:
            return
 
        bull_col = [c for c in self.df.columns if "v1_daily_bull" in c]
        if not bull_col:
            return
 
        col = bull_col[0]
        dt_col = "datetime" if "datetime" in self.df.columns else "date"
        sub = self.df[[dt_col, col, "forward_ret_5"]].dropna()
        sub = sub.sort_values(dt_col).set_index(dt_col)
 
        rolling_corr = sub[[col, "forward_ret_5"]].rolling(252).corr()
        ic = rolling_corr.xs("forward_ret_5", level=1)[col]
 
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(ic.index, ic.values, color=CLR["train"], lw=0.7, alpha=0.8)
        ax.axhline(0, color="black", ls="--", alpha=0.3)
        ax.axvline(pd.Timestamp("2023-01-01"), color=CLR["val"],
                   ls="--", alpha=0.7, label="Val start")
        ax.axvline(pd.Timestamp("2024-01-01"), color=CLR["test"],
                   ls="--", alpha=0.7, label="Test start")
        ax.set_ylabel("Rolling Spearman IC (252d)")
        ax.set_title("S03: V1 Daily Bull — Temporal IC Stability")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(self.out / "S03_rolling_ic.png", dpi=150)
        plt.close()
        logging.info("  ✓ S03 Rolling IC")
 
    def s04_decile_returns(self):
        """Forward returns by CNN confidence decile."""
        if not HAS_MPL or "forward_ret_5" not in self.train.columns:
            return
 
        bull_col = [c for c in self.train.columns
                    if "_bull" in c and "v1_" in c]
        if not bull_col:
            return
 
        col = bull_col[0]
        df = self.train[[col, "forward_ret_5"]].dropna()
        df["decile"] = pd.qcut(df[col], 10, labels=False, duplicates="drop")
 
        fig, ax = plt.subplots(figsize=(10, 6))
        means = df.groupby("decile")["forward_ret_5"].mean()
        ax.bar(means.index, means.values, color=CLR["bull"], alpha=0.7)
        ax.axhline(0, color="black", ls="--", alpha=0.3)
        ax.set_xlabel("Bull Probability Decile (0=most bearish, 9=most bullish)")
        ax.set_ylabel("Mean 5-Bar Forward Return")
        ax.set_title("S04: V1 Forward Returns by CNN Confidence (Train)\n"
                     "Monotonic = CNN learned real structure")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.tight_layout()
        plt.savefig(self.out / "S04_decile_returns.png", dpi=150)
        plt.close()
        logging.info("  ✓ S04 Decile Returns")
 
    def s05_coverage(self):
        """Coverage heatmap: tickers × year-month."""
        if not HAS_MPL:
            return
 
        dt_col = "datetime" if "datetime" in self.df.columns else "date"
        df = self.df.copy()
        df["ym"] = pd.to_datetime(df[dt_col]).dt.to_period("M")
        coverage = df.groupby(["ticker", "ym"]).size().unstack(fill_value=0)
 
        if coverage.shape[0] > 50:
            top_tickers = df["ticker"].value_counts().head(50).index
            coverage = coverage.loc[coverage.index.isin(top_tickers)]
 
        fig, ax = plt.subplots(figsize=(16, max(8, coverage.shape[0] * 0.15)))
        im = ax.imshow(coverage.values > 0, aspect="auto", cmap="YlGn")
        ax.set_ylabel("Ticker")
        ax.set_title("S05: V1 Image Coverage (top 50 tickers × month)")
        ax.set_xticks(range(0, coverage.shape[1], max(1, coverage.shape[1]//12)))
        ax.set_xticklabels([str(coverage.columns[i])
                            for i in range(0, coverage.shape[1],
                                           max(1, coverage.shape[1]//12))],
                           rotation=45, fontsize=7)
        plt.tight_layout()
        plt.savefig(self.out / "S05_coverage.png", dpi=150)
        plt.close()
        logging.info("  ✓ S05 Coverage")
 
    def s06_label_dist(self):
        """Label distribution per split."""
        if not HAS_MPL or "label" not in self.df.columns:
            return
 
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        label_map = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
        label_colors = [CLR["bear"], CLR["neutral"], CLR["bull"]]
 
        for ax, split in zip(axes, ["train", "val", "test"]):
            sub = self.df[self.df["split"] == split]
            if sub.empty:
                continue
            counts = sub["label"].map(label_map).value_counts()
            ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
                   colors=label_colors)
            ax.set_title(f"{split.upper()}\n(n={len(sub):,})")
 
        plt.suptitle("S06: Triple Barrier Label Distribution",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.out / "S06_label_dist.png", dpi=150)
        plt.close()
        logging.info("  ✓ S06 Label Distribution")
 
    def run_all(self):
        logging.info("=" * 60)
        logging.info("V1 STOCK GEOMETRY — Analysis Suite (6 plots)")
        logging.info("=" * 60)
        self.s01_resolution_ic()
        self.s02_physics_drift()
        self.s03_rolling_ic()
        self.s04_decile_returns()
        self.s05_coverage()
        self.s06_label_dist()
        logging.info(f"✓ All plots → {self.out}/")
        logging.info("=" * 60)
 
 
def main():
    p = argparse.ArgumentParser(
        description="V1 Stock Geometry — Cross-Resolution Analyzer")
    p.add_argument("--signal_parquet", required=True,
                   help="Path to cnn_features.parquet or signal parquet")
    # Using str(DEFAULT_OUT) to anchor to V1/plots/
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    V1Analyzer(args.signal_parquet, args.output_dir).run_all()
 
 
if __name__ == "__main__":
    main()