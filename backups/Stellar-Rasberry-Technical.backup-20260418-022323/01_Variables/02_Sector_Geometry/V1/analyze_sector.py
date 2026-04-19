"""02_Sector_Geometry/V1/analyze_cnn.py
======================================
V2 Sector Geometry — Cross-Resolution Analysis Suite.
 
Plots:
  S01: Self-reference impact — clean vs masked forward returns
  S02: Resolution IC comparison (15min through monthly)
  S03: Sector rotation heatmap
  S04: GICS reclassification coverage
  S05: Per-ETF decile returns
  S06: Cross-sector correlation matrix
======================================
"""
 
import argparse, logging
from pathlib import Path
import numpy as np, pandas as pd
 
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import PercentFormatter
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
 
import importlib as _il
# Note: Ensure the path to 01_base_modules_ is in your PYTHONPATH
B = _il.import_module("01_base_modules_")
 
CLR = {
    "bull": "#2ecc71", "bear": "#e74c3c", "neutral": "#95a5a6",
    "train": "#2c3e50", "val": "#f39c12", "test": "#e74c3c",
}
 
# Anchor to the directory where this script lives
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = THIS_DIR / "plots"
 
class V2Analyzer:
    def __init__(self, signal_path: str, output_dir: str = str(DEFAULT_OUT)):
        self.df = pd.read_parquet(signal_path)
        if "datetime" in self.df.columns:
            self.df["datetime"] = pd.to_datetime(self.df["datetime"])
        
        # This ensures the 'plots/' folder is created exactly under V1/
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        
        self.train = self.df[self.df["split"] == "train"]
        logging.info(f"V2 Analyzer: {len(self.df):,} rows | "
                     f"Train: {len(self.train):,}")
 
    def s01_self_ref_impact(self):
        """Clean vs masked: does self-ref masking prevent leakage?"""
        if not HAS_MPL: return
        if "v2_self_ref_flag" not in self.train.columns: return
        if "forward_ret_5" not in self.train.columns: return
 
        flagged = self.train[self.train["v2_self_ref_flag"] == True]
        clean = self.train[self.train["v2_self_ref_flag"] == False]
        if len(flagged) < 10 or len(clean) < 10: return
 
        fig, ax = plt.subplots(figsize=(10, 6))
        data = [clean["forward_ret_5"].dropna().values,
                flagged["forward_ret_5"].dropna().values]
        bp = ax.boxplot(
            data,
            labels=[f"Clean\n(n={len(clean):,})",
                    f"Self-ref (W>5%)\n(n={len(flagged):,})"],
            showfliers=False, patch_artist=True)
        bp["boxes"][0].set_facecolor(CLR["bull"])
        bp["boxes"][1].set_facecolor(CLR["bear"])
        for b in bp["boxes"]: b.set_alpha(0.6)
        ax.axhline(0, color="black", ls="--", alpha=0.3)
        ax.set_ylabel("5-Bar Forward Return")
        ax.set_title("S01: Self-Reference Impact\n"
                     "If distributions differ → self-ref leaks alpha → mask is necessary")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.tight_layout()
        plt.savefig(self.out / "S01_self_ref.png", dpi=150)
        plt.close()
        logging.info("  ✓ S01 Self-Ref Impact")
 
    def s02_resolution_ic(self):
        """Spearman IC across all 5 resolutions."""
        if not HAS_MPL or "forward_ret_5" not in self.train.columns: return
        bull_cols = sorted([c for c in self.train.columns
                            if "_bull" in c and "v2_" in c])
        if not bull_cols: return
 
        fig, ax = plt.subplots(figsize=(10, 6))
        names, vals = [], []
        for col in bull_cols:
            sub = self.train[[col, "forward_ret_5"]].dropna()
            if len(sub) < 30: continue
            ic = sub[col].corr(sub["forward_ret_5"], method="spearman")
            names.append(col.replace("v2_", "").replace("_bull", ""))
            vals.append(ic)
        ax.bar(range(len(vals)), vals, color="#3498db", alpha=0.7)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Spearman IC")
        ax.set_title("S02: V2 Sector Bull IC by Resolution (Train)")
        ax.axhline(0, color="black", ls="--", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9)
        plt.tight_layout()
        plt.savefig(self.out / "S02_resolution_ic.png", dpi=150)
        plt.close()
        logging.info("  ✓ S02 Resolution IC")
 
    def s03_sector_rotation(self):
        """Monthly bull probability heatmap by sector."""
        if not HAS_MPL or "v2_sector_etf" not in self.df.columns: return
        bull_col = [c for c in self.df.columns if "_bull" in c and "v2_" in c]
        if not bull_col: return
 
        col = bull_col[0]
        dt_col = "datetime" if "datetime" in self.df.columns else "date"
        df = self.df.copy()
        df["month"] = pd.to_datetime(df[dt_col]).dt.to_period("M")
        pivot = df.pivot_table(index="month", columns="v2_sector_etf",
                               values=col, aggfunc="mean")
 
        fig, ax = plt.subplots(figsize=(16, 10))
        sns.heatmap(pivot.T, cmap="RdYlGn", center=0.33, ax=ax,
                    xticklabels=max(1, len(pivot) // 12))
        ax.set_title("S03: Sector Rotation — Monthly Mean Bull Probability\n"
                     "XLC appears 2018-09. XLRE appears 2015-10.")
        plt.tight_layout()
        plt.savefig(self.out / "S03_sector_rotation.png", dpi=150)
        plt.close()
        logging.info("  ✓ S03 Sector Rotation")
 
    def s04_gics_coverage(self):
        """GICS reclassification coverage — data availability per ETF."""
        if not HAS_MPL or "v2_sector_etf" not in self.df.columns: return
 
        dt_col = "datetime" if "datetime" in self.df.columns else "date"
        fig, ax = plt.subplots(figsize=(14, 6))
        for i, etf in enumerate(B.SECTOR_ETFS):
            sub = self.df[self.df["v2_sector_etf"] == etf]
            if sub.empty: continue
            dates = pd.to_datetime(sub[dt_col])
            ax.barh(i, (dates.max() - dates.min()).days,
                    left=dates.min().toordinal(), height=0.7,
                    color="#3498db", alpha=0.7)
        ax.set_yticks(range(len(B.SECTOR_ETFS)))
        ax.set_yticklabels(B.SECTOR_ETFS)
        ax.axvline(pd.Timestamp("2018-09-24").toordinal(),
                   color="red", ls="--", alpha=0.7, label="XLC created")
        ax.axvline(pd.Timestamp("2015-10-07").toordinal(),
                   color="orange", ls="--", alpha=0.7, label="XLRE created")
        ax.legend()
        ax.set_title("S04: GICS Reclassification — Data Coverage by ETF")
        plt.tight_layout()
        plt.savefig(self.out / "S04_gics_coverage.png", dpi=150)
        plt.close()
        logging.info("  ✓ S04 GICS Coverage")
 
    def s05_per_etf_decile(self):
        """Decile returns broken down by sector ETF."""
        if not HAS_MPL: return
        if "forward_ret_5" not in self.train.columns: return
        if "v2_sector_etf" not in self.train.columns: return
        bull_col = [c for c in self.train.columns if "_bull" in c and "v2_" in c]
        if not bull_col: return
 
        col = bull_col[0]
        etfs = self.train["v2_sector_etf"].unique()
        n_etfs = min(len(etfs), 11)
        fig, axes = plt.subplots(3, 4, figsize=(18, 12))
        axes = axes.flatten()
 
        for i, etf in enumerate(sorted(etfs)[:n_etfs]):
            sub = self.train[self.train["v2_sector_etf"] == etf][[col, "forward_ret_5"]].dropna()
            if len(sub) < 100: continue
            sub["decile"] = pd.qcut(sub[col], 5, labels=False, duplicates="drop")
            means = sub.groupby("decile")["forward_ret_5"].mean()
            axes[i].bar(means.index, means.values, color=CLR["bull"], alpha=0.7)
            axes[i].axhline(0, color="black", ls="--", alpha=0.3)
            axes[i].set_title(etf, fontsize=10)
 
        for j in range(n_etfs, len(axes)):
            axes[j].set_visible(False)
 
        plt.suptitle("S05: Per-ETF Decile Returns (Train)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.out / "S05_per_etf_decile.png", dpi=150)
        plt.close()
        logging.info("  ✓ S05 Per-ETF Decile")
 
    def s06_cross_sector_corr(self):
        """Cross-sector CNN correlation matrix."""
        if not HAS_MPL or "v2_sector_etf" not in self.df.columns: return
        bull_col = [c for c in self.df.columns if "_bull" in c and "v2_" in c]
        if not bull_col: return
 
        col = bull_col[0]
        dt_col = "datetime" if "datetime" in self.df.columns else "date"
        df = self.df.copy()
        df["month"] = pd.to_datetime(df[dt_col]).dt.to_period("M")
        pivot = df.pivot_table(index="month", columns="v2_sector_etf",
                               values=col, aggfunc="mean")
        corr = pivot.corr(method="spearman")
 
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    ax=ax, square=True)
        ax.set_title("S06: Cross-Sector CNN Correlation\n"
                     "Low correlation = sectors provide independent signals")
        plt.tight_layout()
        plt.savefig(self.out / "S06_cross_sector_corr.png", dpi=150)
        plt.close()
        logging.info("  ✓ S06 Cross-Sector Correlation")
 
    def run_all(self):
        logging.info("=" * 60)
        logging.info("V2 SECTOR GEOMETRY — Analysis Suite (6 plots)")
        logging.info("=" * 60)
        self.s01_self_ref_impact()
        self.s02_resolution_ic()
        self.s03_sector_rotation()
        self.s04_gics_coverage()
        self.s05_per_etf_decile()
        self.s06_cross_sector_corr()
        logging.info(f"✓ All plots → {self.out}/")
        logging.info("=" * 60)
 
 
def main():
    p = argparse.ArgumentParser(
        description="V2 Sector Geometry — Cross-Resolution Analyzer")
    p.add_argument("--signal_parquet", required=True)
    # Using str(DEFAULT_OUT) ensures it defaults to the 'plots/' folder next to this script
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    V2Analyzer(args.signal_parquet, args.output_dir).run_all()
 
 
if __name__ == "__main__":
    main()