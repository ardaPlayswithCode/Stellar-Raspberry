"""
03_SPY_Geometry/V1/analyze.py
===============================
V3 SPY/VIX Geometry — Cross-Resolution Analysis Suite.
 
ALL PLOTS saved to: 03_SPY_Geometry/V1/plots/
  (same level as this file and trainer.py)
 
Plots:
  S01: Resolution IC — SPY bull probability across 5 timeframes.
 
  S02: SPY×VIX Cross-Signal — forward returns in 4 quadrants:
       SPY↑ VIX↓ (best): bull market with declining fear
       SPY↑ VIX↑ (conflict): rally into rising vol — fragile
       SPY↓ VIX↓ (conflict): selloff with no fear — correction
       SPY↓ VIX↑ (worst): crash regime
 
  S03: VIX Regime Conditioning — Adrian & Brunnermeier (2016):
       Split SPY signal IC by VIX quintile. The volatility paradox
       predicts that signals in low-VIX regimes APPEAR reliable but
       are actually the most fragile.
 
  S04: Drawdown Overlay — SPY CNN confidence vs equity drawdown.
       Tests whether the CNN provides early warning before drawdowns.
 
  S05: L3 Bear Filter Backtest — monthly SPY bearish events:
       what are forward 1m/3m/6m returns when L3 fires?
       Validates the FULL CASH override decision.
 
  S06: VIX Inversion Validation — IC with and without label inversion.
       Confirms that .map({0:2,1:1,2:0}) improves predictive alignment.
"""
 
import argparse, logging
from pathlib import Path
import numpy as np, pandas as pd
 
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
 
import importlib as _il
B = _il.import_module("01_base_modules_")
 
CLR = {
    "bull": "#2ecc71", "bear": "#e74c3c", "neutral": "#95a5a6",
    "vix": "#e67e22", "train": "#2c3e50", "val": "#f39c12",
    "test": "#e74c3c", "spy": "#3498db",
}
 
# Output directory: V1/plots/ (same level as this file)
THIS = Path(__file__).resolve().parent
PLOT_DIR = THIS / "plots"
 
 
class V3Analyzer:
    def __init__(self, spy_signal_path: str,
                 vix_signal_path: str = None):
        self.spy = pd.read_parquet(spy_signal_path)
        if "datetime" in self.spy.columns:
            self.spy["datetime"] = pd.to_datetime(self.spy["datetime"])
        self.vix = None
        if vix_signal_path and Path(vix_signal_path).exists():
            self.vix = pd.read_parquet(vix_signal_path)
            if "datetime" in self.vix.columns:
                self.vix["datetime"] = pd.to_datetime(self.vix["datetime"])
        self.out = PLOT_DIR
        self.out.mkdir(parents=True, exist_ok=True)
        self.train = self.spy[self.spy["split"] == "train"]
 
        if HAS_MPL:
            plt.rcParams.update({
                "figure.figsize": (14, 8), "font.size": 10,
                "axes.grid": True, "grid.alpha": 0.3,
            })
 
        logging.info(f"V3 Analyzer: SPY {len(self.spy):,}"
                     + (f" | VIX {len(self.vix):,}" if self.vix is not None else ""))
        logging.info(f"  Plots → {self.out}/")
 
    def s01_resolution_ic(self):
        """SPY bull IC across all 5 resolutions."""
        if not HAS_MPL or "forward_ret_5" not in self.train.columns:
            return
 
        bull_cols = sorted([c for c in self.train.columns
                            if "_bull" in c and "v3_spy" in c])
        if not bull_cols:
            return
 
        fig, ax = plt.subplots(figsize=(10, 6))
        names, vals = [], []
        for col in bull_cols:
            sub = self.train[[col, "forward_ret_5"]].dropna()
            if len(sub) < 30: continue
            ic = sub[col].corr(sub["forward_ret_5"], method="spearman")
            names.append(col.split("_bull")[0].replace("v3_spy_", ""))
            vals.append(ic)
 
        ax.bar(range(len(vals)), vals, color=CLR["spy"], alpha=0.7)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Spearman IC")
        ax.set_title("S01: V3 SPY Bull IC by Resolution (Train)")
        ax.axhline(0, color="black", ls="--", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9)
 
        plt.tight_layout()
        plt.savefig(self.out / "S01_resolution_ic.png", dpi=150)
        plt.close()
        logging.info("  ✓ S01 Resolution IC")
 
    def s02_cross_signal(self):
        """SPY↑/↓ × VIX↑/↓ → forward returns (4 quadrants)."""
        if not HAS_MPL or self.vix is None:
            return
        if "forward_ret_5" not in self.train.columns:
            return
 
        spy_b = [c for c in self.spy.columns if "v3_spy_15min_bull" in c]
        vix_b = [c for c in self.vix.columns if "v3_vix_15min_bull" in c]
        if not spy_b: spy_b = [c for c in self.spy.columns if "v3_spy" in c and "_bull" in c]
        if not vix_b: vix_b = [c for c in self.vix.columns if "v3_vix" in c and "_bull" in c]
        if not spy_b or not vix_b: return
 
        dt_col = "datetime" if "datetime" in self.spy.columns else "date"
        merged = self.spy[[dt_col, spy_b[0], "forward_ret_5", "split"]].merge(
            self.vix[[dt_col, vix_b[0]]].drop_duplicates(dt_col),
            on=dt_col, how="inner")
        tr = merged[merged["split"] == "train"]
        if len(tr) < 100: return
 
        q67_s = tr[spy_b[0]].quantile(0.67)
        q33_s = tr[spy_b[0]].quantile(0.33)
        q67_v = tr[vix_b[0]].quantile(0.67)
        q33_v = tr[vix_b[0]].quantile(0.33)
 
        scenarios = {
            "SPY↑ VIX↓\n(best)":     (tr[spy_b[0]] > q67_s) & (tr[vix_b[0]] < q33_v),
            "SPY↑ VIX↑\n(conflict)": (tr[spy_b[0]] > q67_s) & (tr[vix_b[0]] > q67_v),
            "SPY↓ VIX↓\n(conflict)": (tr[spy_b[0]] < q33_s) & (tr[vix_b[0]] < q33_v),
            "SPY↓ VIX↑\n(worst)":    (tr[spy_b[0]] < q33_s) & (tr[vix_b[0]] > q67_v),
        }
        colors = [CLR["bull"], CLR["vix"], CLR["vix"], CLR["bear"]]
 
        fig, ax = plt.subplots(figsize=(12, 6))
        data, labels = [], []
        for label, mask in scenarios.items():
            rets = tr.loc[mask, "forward_ret_5"].dropna()
            if len(rets) > 10:
                data.append(rets.values)
                labels.append(f"{label}\n(n={len(rets):,})")
 
        if data:
            bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
            for patch, c in zip(bp["boxes"], colors[:len(bp["boxes"])]):
                patch.set_facecolor(c); patch.set_alpha(0.6)
            ax.axhline(0, color="black", ls="--", alpha=0.3)
            ax.set_ylabel("5-Bar Forward Return")
            ax.set_title("S02: SPY×VIX Cross-Signal (15min)\n"
                         "Best: bull SPY + declining VIX. Worst: bear SPY + rising VIX.")
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
 
        plt.tight_layout()
        plt.savefig(self.out / "S02_cross_signal.png", dpi=150)
        plt.close()
        logging.info("  ✓ S02 Cross-Signal")
 
    def s03_vix_regime(self):
        """VIX quintile conditioning — Adrian & Brunnermeier (2016)."""
        if not HAS_MPL or self.vix is None:
            return
        if "forward_ret_5" not in self.train.columns:
            return
 
        vix_b = [c for c in self.vix.columns if "_bull" in c and "v3_vix" in c]
        if not vix_b: return
 
        dt_col = "datetime" if "datetime" in self.spy.columns else "date"
        merged = self.spy[[dt_col, "forward_ret_5", "split"]].merge(
            self.vix[[dt_col, vix_b[0]]].drop_duplicates(dt_col),
            on=dt_col, how="inner")
        tr = merged[merged["split"] == "train"]
        if len(tr) < 200: return
 
        tr["vix_q"] = pd.qcut(tr[vix_b[0]], 5,
                               labels=["Q1\n(Low\nFear)", "Q2", "Q3", "Q4", "Q5\n(High\nFear)"])
 
        fig, ax = plt.subplots(figsize=(10, 6))
        means = tr.groupby("vix_q")["forward_ret_5"].agg(["mean", "std", "count"])
        se = means["std"] / np.sqrt(means["count"])
        ax.bar(range(5), means["mean"], yerr=se, capsize=5,
               color=CLR["vix"], alpha=0.7)
        ax.set_xticks(range(5))
        ax.set_xticklabels(means.index)
        ax.axhline(0, color="black", ls="--", alpha=0.3)
        ax.set_ylabel("Mean 5-Bar Forward Return")
        ax.set_title("S03: Forward Returns by VIX CNN Quintile\n"
                     "Adrian & Brunnermeier (2016): low fear ≠ low risk")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.tight_layout()
        plt.savefig(self.out / "S03_vix_regime.png", dpi=150)
        plt.close()
        logging.info("  ✓ S03 VIX Regime")
 
    def s04_drawdown_overlay(self):
        """CNN confidence vs equity drawdown."""
        if not HAS_MPL: return
        bull_col = [c for c in self.spy.columns if "_bull" in c and "v3_spy" in c]
        if not bull_col or "forward_ret_1" not in self.spy.columns: return
 
        dt_col = "datetime" if "datetime" in self.spy.columns else "date"
        df = self.spy.drop_duplicates(dt_col).sort_values(dt_col).set_index(dt_col)
        col = bull_col[0]
 
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        ax1.plot(df.index, df[col], color=CLR["spy"], lw=0.6, alpha=0.8)
        ax1.axhline(0.5, color="#555", ls="--", lw=0.3)
        ax1.axhline(0.33, color=CLR["bear"], ls="--", lw=0.3, alpha=0.5)
        ax1.set_ylabel("SPY Bull Prob")
        ax1.set_title("S04: SPY CNN Confidence vs Market Drawdowns")
        ax1.axvline(pd.Timestamp("2023-01-01"), color=CLR["val"], ls="--", alpha=0.5)
        ax1.axvline(pd.Timestamp("2024-01-01"), color=CLR["test"], ls="--", alpha=0.5)
 
        cum = (1 + df["forward_ret_1"]).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        ax2.fill_between(dd.index, dd.values, 0, color=CLR["bear"], alpha=0.4)
        ax2.set_ylabel("Drawdown")
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
 
        plt.tight_layout()
        plt.savefig(self.out / "S04_drawdown.png", dpi=150)
        plt.close()
        logging.info("  ✓ S04 Drawdown Overlay")
 
    def s05_l3_backtest(self):
        """L3 bear filter: what happens when monthly SPY → bearish?"""
        if not HAS_MPL: return
        bull_col = [c for c in self.spy.columns
                    if "v3_spy_monthly_bull" in c]
        if not bull_col: return
 
        col = bull_col[0]
        dt_col = "datetime" if "datetime" in self.spy.columns else "date"
        df = self.spy.drop_duplicates(dt_col).sort_values(dt_col)
 
        fwd_cols = [c for c in ["forward_ret_5", "forward_ret_10", "forward_ret_20"]
                    if c in df.columns]
        if not fwd_cols: return
 
        bearish = df[df[col] < 0.33]
        neutral = df[(df[col] >= 0.33) & (df[col] <= 0.67)]
        bullish = df[df[col] > 0.67]
 
        fig, axes = plt.subplots(1, len(fwd_cols), figsize=(5 * len(fwd_cols), 6))
        if len(fwd_cols) == 1: axes = [axes]
 
        for ax, fc in zip(axes, fwd_cols):
            data, labels, clrs = [], [], []
            for sub, lbl, c in [(bullish, "Bull", CLR["bull"]),
                                (neutral, "Neutral", CLR["neutral"]),
                                (bearish, "Bear\n(L3 CASH)", CLR["bear"])]:
                vals = sub[fc].dropna()
                if len(vals) > 5:
                    data.append(vals.values)
                    labels.append(f"{lbl}\n(n={len(vals):,})")
                    clrs.append(c)
            if data:
                bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
                for patch, c in zip(bp["boxes"], clrs):
                    patch.set_facecolor(c); patch.set_alpha(0.6)
                ax.axhline(0, color="black", ls="--", alpha=0.3)
                ax.set_title(fc.replace("forward_ret_", "") + "-bar")
                ax.yaxis.set_major_formatter(PercentFormatter(1.0))
 
        plt.suptitle("S05: L3 Bear Filter Backtest — Monthly SPY CNN Signal\n"
                     "Bear = FULL CASH override. Does avoiding losses justify missed gains?",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.out / "S05_l3_backtest.png", dpi=150)
        plt.close()
        logging.info("  ✓ S05 L3 Backtest")
 
    def s06_inversion_validation(self):
        """Confirm that VIX label inversion improves IC."""
        if not HAS_MPL or self.vix is None:
            return
        if "forward_ret_5" not in self.spy.columns:
            return
 
        vix_b = [c for c in self.vix.columns if "_bull" in c and "v3_vix" in c]
        if not vix_b: return
 
        dt_col = "datetime" if "datetime" in self.spy.columns else "date"
        merged = self.spy[[dt_col, "forward_ret_5", "split"]].merge(
            self.vix[[dt_col, vix_b[0]]].drop_duplicates(dt_col),
            on=dt_col, how="inner")
        tr = merged[merged["split"] == "train"]
        if len(tr) < 100: return
 
        ic_inverted = tr[vix_b[0]].corr(tr["forward_ret_5"], method="spearman")
        ic_original = (1 - tr[vix_b[0]]).corr(tr["forward_ret_5"], method="spearman")
 
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(["Original\n(high VIX = bear)", "Inverted\n(high VIX bull = bear)"],
                       [ic_original, ic_inverted],
                       color=[CLR["neutral"], CLR["vix"]], alpha=0.7)
        ax.axhline(0, color="black", ls="--", alpha=0.3)
        ax.set_ylabel("Spearman IC")
        ax.set_title("S06: VIX Label Inversion Validation\n"
                     ".map({0:2, 1:1, 2:0}) — does inversion improve alignment?")
        for bar, val in zip(bars, [ic_original, ic_inverted]):
            ax.text(bar.get_x() + bar.get_width() / 2, val,
                    f"{val:.4f}", ha="center",
                    va="bottom" if val >= 0 else "top", fontsize=11)
        plt.tight_layout()
        plt.savefig(self.out / "S06_inversion.png", dpi=150)
        plt.close()
        logging.info(f"  ✓ S06 Inversion (orig={ic_original:.4f} inv={ic_inverted:.4f})")
 
    def run_all(self):
        logging.info("=" * 60)
        logging.info("V3 SPY/VIX GEOMETRY — Analysis Suite (6 plots)")
        logging.info(f"  Output: {self.out}/")
        logging.info("=" * 60)
        self.s01_resolution_ic()
        self.s02_cross_signal()
        self.s03_vix_regime()
        self.s04_drawdown_overlay()
        self.s05_l3_backtest()
        self.s06_inversion_validation()
        logging.info(f"✓ All plots → {self.out}/")
        logging.info("=" * 60)
 
 
def main():
    p = argparse.ArgumentParser(
        description="V3 SPY/VIX Geometry — Cross-Resolution Analyzer")
    p.add_argument("--spy_parquet", required=True,
                   help="Path to SPY CNN features parquet")
    p.add_argument("--vix_parquet", default=None,
                   help="Path to VIX CNN features parquet")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    V3Analyzer(args.spy_parquet, args.vix_parquet).run_all()
 
 
if __name__ == "__main__":
    main()