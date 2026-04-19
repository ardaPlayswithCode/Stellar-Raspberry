"""
trainer.py — Var 06 Volume Trend (Trainer)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS DOES:
  Evaluates 18 volume surprise combos. Selects best via Spearman IC
  with Harvey t-stat gating and BLL bootstrap. Also computes GKM-
  conditioned IC (normal-return bars only, |ret| rank < 30th pctl).
 
RESEARCH:
  GKM (2001) JF 56(3) — Filtering to normal-return bars ~DOUBLES IC.
  Chordia (2008) JFE 87(2) — Only innovation coefficient significant.
  Harvey, Liu & Zhu (2016) RFS — t ≥ 3.0 full, t ≥ 2.0 partial.
  BLL (1992) JF — Bootstrap null, p < 0.05.
 
IC TARGETS: Daily 0.03-0.07, Weekly 0.05-0.10. Half-life 4-5 years.
 
OUTPUT:
  selected_params_{resolution}.json
 
USAGE:
  python trainer.py --data_dir ../../00_Mini_Model/data/ohlcv --resolution daily
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
 
import argparse, json, logging, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
 
THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent / "00_Mini_Model"))
import importlib as _il
B = _il.import_module("01_base_modules_")
 
TRAIN_END   = B.TRAIN_END
VAL_END     = B.VAL_END
PROFIT_MULT = 2.0
STOP_MULT   = 1.0
VERT_D      = 20
VERT_W      = 12
N_BOOTSTRAP = 1000
GKM_PCTL    = 0.30  # GKM Table 4: normal-return filter threshold
 
 
def harvey_weight(t):
    if t >= 3.0: return 1.0
    if t >= 2.0: return t - 2.0
    return 0.0
 
 
def triple_barrier_labels(close, atr, pm, sm, hz):
    n = len(close)
    labels = pd.Series(["neutral"] * n, index=close.index)
    for i in range(n - 1):
        e, a = close.iloc[i], atr.iloc[i]
        if np.isnan(a) or a <= 0: continue
        tp, sl = e + pm * a, e - sm * a
        for j in range(1, min(hz + 1, n - i)):
            c = close.iloc[i + j]
            if c >= tp: labels.iloc[i] = "strongly_bullish"; break
            if c <= sl: labels.iloc[i] = "bearish"; break
    return labels
 
 
def compute_ic(s, r):
    m = ~(np.isnan(s) | np.isnan(r))
    if m.sum() < 30: return 0.0
    ic, _ = spearmanr(s[m], r[m])
    return ic if not np.isnan(ic) else 0.0
 
 
def bll_bootstrap(s, r, n=N_BOOTSTRAP):
    m = ~(np.isnan(s) | np.isnan(r))
    sc, rc = s[m], r[m]
    if len(sc) < 30: return 1.0
    obs = spearmanr(sc, rc)[0]
    rng = np.random.default_rng(42)
    nulls = np.array([spearmanr(rng.permutation(sc), rc)[0] for _ in range(n)])
    return float(np.mean(nulls >= obs))
 
 
def load_and_label(data_dir, resolution, tickers):
    hz = VERT_D if resolution in ("daily", "15min", "1hour") else VERT_W
    frames = []
    for tk in tickers:
        p = Path(data_dir) / resolution / f"{tk}.parquet"
        if not p.exists(): continue
        df = pd.read_parquet(p)
        if "datetime" not in df.columns and df.index.name == "datetime":
            df = df.reset_index()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        if len(df) < 60: continue
 
        atr = B.wilder_atr(df["high"], df["low"], df["close"])
        df["fwd_ret"] = df["close"].pct_change(hz).shift(-hz)
        df["same_bar_ret"] = df["close"].pct_change().abs()
        df["label"] = triple_barrier_labels(df["close"], atr, PROFIT_MULT, STOP_MULT, hz)
        df["ticker"] = tk
 
        ts = pd.to_datetime(df["datetime"])
        df["split"] = "test"
        df.loc[ts <= pd.Timestamp(TRAIN_END), "split"] = "train"
        df.loc[(ts > pd.Timestamp(TRAIN_END)) & (ts <= pd.Timestamp(VAL_END)), "split"] = "val"
 
        # GKM filter: rank of |same_bar_return|
        df["ret_rank"] = df["same_bar_ret"].rolling(60, min_periods=20).rank(pct=True)
 
        frames.append(df[["datetime", "ticker", "split", "fwd_ret", "label", "ret_rank"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
 
 
def evaluate_combos(signal_dir, labels_df):
    sig_path = Path(signal_dir)
    tickers = labels_df["ticker"].unique()
    sample_files = list(sig_path.glob("*.parquet"))
    if not sample_files: return []
    combo_ids = pd.read_parquet(sample_files[0])["combo_id"].unique()
    results = []
 
    for cid in combo_ids:
        all_signals = []
        for tk in tickers:
            sp = sig_path / f"{tk}.parquet"
            if not sp.exists(): continue
            sdf = pd.read_parquet(sp)
            combo = sdf[sdf["combo_id"] == cid]
            if len(combo) == 0: continue
            merged = combo.merge(labels_df[labels_df["ticker"] == tk],
                                 on=["datetime", "ticker"], how="inner")
            if len(merged) > 0:
                merged = merged.copy()
                merged["net_score"] = merged["bull_vol_score"] - merged["bear_vol_score"]
                all_signals.append(merged)
        if not all_signals: continue
        full = pd.concat(all_signals, ignore_index=True)
 
        tr = full["split"] == "train"
        va = full["split"] == "val"
        te = full["split"] == "test"
 
        ic_tr = compute_ic(full.loc[tr, "net_score"].values, full.loc[tr, "fwd_ret"].values)
        ic_va = compute_ic(full.loc[va, "net_score"].values, full.loc[va, "fwd_ret"].values)
        ic_te = compute_ic(full.loc[te, "net_score"].values, full.loc[te, "fwd_ret"].values)
 
        # GKM conditioned IC: normal-return bars only (|ret| rank < 30th pctl)
        gkm_mask = tr & (full["ret_rank"] < GKM_PCTL)
        ic_gkm = compute_ic(full.loc[gkm_mask, "net_score"].values,
                            full.loc[gkm_mask, "fwd_ret"].values)
 
        n_tr = int(tr.sum())
        t_stat = ic_tr * np.sqrt(n_tr) / np.sqrt(1 - ic_tr**2) if abs(ic_tr) > 1e-8 else 0.0
        bll_p = bll_bootstrap(full.loc[tr, "net_score"].values, full.loc[tr, "fwd_ret"].values)
 
        # vol_roc IC (fixed feature)
        roc_ic = compute_ic(full.loc[tr, "vol_roc"].values, full.loc[tr, "fwd_ret"].values)
 
        # Parse combo_id
        parts = cid.split("_")
        lb = int(parts[0][2:])
        norm = parts[1] if len(parts) == 3 else f"{parts[1]}_{parts[2]}"
        ewm_tag = parts[-1]
 
        results.append({
            "combo_id": cid, "lookback": lb, "norm_method": norm, "use_ewm": ewm_tag == "ewm",
            "ic_train": round(ic_tr, 5), "ic_val": round(ic_va, 5), "ic_test": round(ic_te, 5),
            "ic_gkm_conditioned": round(ic_gkm, 5),
            "t_stat": round(t_stat, 3), "harvey_weight": round(harvey_weight(t_stat), 3),
            "bll_p_value": round(bll_p, 4), "vol_roc_ic": round(roc_ic, 5),
            "n_train": n_tr, "n_val": int(va.sum()), "n_test": int(te.sum()),
            "n_gkm_normal": int(gkm_mask.sum()),
        })
 
    return sorted(results, key=lambda x: -x["ic_val"])
 
 
def main():
    p = argparse.ArgumentParser(description="V06 Volume Trend — Trainer")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--resolution", choices=["daily", "weekly", "15min", "1hour"], required=True)
    p.add_argument("--tickers", nargs="*", default=None)
    args = p.parse_args()
 
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    logging.info(f"V06 Volume Trainer | Resolution: {args.resolution}")
 
    tf_dir = Path(args.data_dir) / args.resolution
    tickers = args.tickers or sorted(
        p.stem for p in tf_dir.glob("*.parquet") if not p.stem.startswith("_"))
    logging.info(f"  Tickers: {len(tickers)}")
 
    labels_df = load_and_label(args.data_dir, args.resolution, tickers)
    if len(labels_df) == 0:
        logging.error("No label data."); return
    logging.info(f"  Label rows: {len(labels_df):,}")
 
    signal_dir = THIS / args.resolution / "signals"
    results = evaluate_combos(str(signal_dir), labels_df)
    if not results:
        logging.error("No combo results."); return
 
    logging.info(f"\n{'='*100}")
    logging.info(f"  {'combo':>22} | {'IC_tr':>7} | {'IC_va':>7} | {'IC_te':>7} | "
                 f"{'GKM_IC':>7} | {'t':>6} | {'Hrvy':>5} | {'BLL_p':>6} | {'RoC_IC':>7}")
    logging.info(f"  {'-'*22} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*6} | {'-'*5} | {'-'*6} | {'-'*7}")
    for r in results:
        f = "✓" if r["harvey_weight"] > 0 and r["bll_p_value"] < 0.05 else " "
        logging.info(f"{f} {r['combo_id']:>22} | {r['ic_train']:>7.4f} | {r['ic_val']:>7.4f} | "
                     f"{r['ic_test']:>7.4f} | {r['ic_gkm_conditioned']:>7.4f} | {r['t_stat']:>6.2f} | "
                     f"{r['harvey_weight']:>5.2f} | {r['bll_p_value']:>6.3f} | {r['vol_roc_ic']:>7.4f}")
    logging.info(f"{'='*100}")
 
    valid = [r for r in results if r["harvey_weight"] > 0 and r["bll_p_value"] < 0.05]
    best = max(valid, key=lambda x: x["ic_val"]) if valid else results[0]
    if not valid:
        logging.warning("  No combo passed gates. Selecting best val IC.")
 
    logging.info(f"\n  ★ Selected: {best['combo_id']} | IC_val={best['ic_val']:.4f} | "
                 f"GKM_IC={best['ic_gkm_conditioned']:.4f} | t={best['t_stat']:.2f}")
 
    out_path = THIS / f"selected_params_{args.resolution}.json"
    with open(out_path, "w") as f:
        json.dump({"selected": best, "all_combos": results}, f, indent=2)
    logging.info(f"  → {out_path}")
 
 
if __name__ == "__main__":
    main()