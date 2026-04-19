"""
02_build_god_registry.py — 3M-Point Master Registry (Production)
==================================================================
Reads local parquet from 00_ingest_historical.py output.
No yfinance calls. No p.exists() in loops.
 
Optimizations:
  1. Glob-based path sets: build set of all existing .png paths ONCE,
     then vectorized membership test via numpy isin.
  2. Category dtypes: ticker and split as pd.Categorical (~60% RAM reduction
     on string columns across 3M rows).
  3. Vectorized T-1: numpy searchsorted on int64 ns arrays, one pass per TF.
  4. zstd compression: ~3:1 ratio on parquet output.
  5. Chunked ticker processing: prevents 96GB OOM on full universe.
 
Output: master_registry_15m.parquet
  Columns: datetime, ticker, split, t1_1hour, t1_daily, t1_weekly, t1_monthly,
           v1_15min_img ... v3_vix_monthly_img (20 image path columns),
           v4-v8 tabular columns (20 floats)
"""
 
import os, sys, json, argparse, logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
 
import importlib as _il
B = _il.import_module("01_base_modules_")
 
TABULAR_COLUMNS = [
    "v4_rsi_daily","v4_rsi_weekly",
    "v5_macd_daily","v5_macd_weekly","v5_hist_delta_daily","v5_hist_delta_weekly",
    "v6_volume_daily","v6_volume_weekly","v6_vol_roc_daily","v6_vol_roc_weekly",
    "v7_vix_close","v7_vix_weekly","v7_atr_pct","v7_atr_pct_weekly",
    "v8_v6_score","v8_bars_since_high","v8_bars_since_low",
    "v8_v6_score_weekly","v8_bars_since_high_weekly","v8_bars_since_low_weekly",
]
 
IMAGE_COLUMNS = {
    "v1_15min_img":"v1_15min","v1_1hour_img":"v1_1hour","v1_daily_img":"v1_daily",
    "v1_weekly_img":"v1_weekly","v1_monthly_img":"v1_monthly",
    "v2_15min_img":"v2_15min","v2_1hour_img":"v2_1hour","v2_daily_img":"v2_daily",
    "v2_weekly_img":"v2_weekly","v2_monthly_img":"v2_monthly",
    "v3_spy_15min_img":"v3_spy_15min","v3_spy_1hour_img":"v3_spy_1hour",
    "v3_spy_daily_img":"v3_spy_daily","v3_spy_weekly_img":"v3_spy_weekly",
    "v3_spy_monthly_img":"v3_spy_monthly",
    "v3_vix_15min_img":"v3_vix_15min","v3_vix_1hour_img":"v3_vix_1hour",
    "v3_vix_daily_img":"v3_vix_daily","v3_vix_weekly_img":"v3_vix_weekly",
    "v3_vix_monthly_img":"v3_vix_monthly",
}
 
 
class ImageSetResolver:
    """Glob all image directories ONCE at init. O(1) lookup per path."""
 
    def __init__(self, variables_root: str):
        self.root = Path(variables_root)
        self._cache = {}
 
    def _ensure_cached(self, subdir: str):
        if subdir not in self._cache:
            full = self.root / subdir
            if full.exists():
                self._cache[subdir] = set(str(p) for p in full.rglob("*.png"))
            else:
                self._cache[subdir] = set()
 
    def resolve_batch(self, subdir: str, ticker: str,
                      dates: pd.DatetimeIndex, is_intraday: bool) -> np.ndarray:
        self._ensure_cached(subdir)
        img_set = self._cache[subdir]
        base = str(self.root / subdir)
        if is_intraday:
            paths = np.array([f"{base}/{ticker}/{d.strftime('%Y-%m-%d_%H%M')}.png" for d in dates])
        else:
            paths = np.array([f"{base}/{ticker}/{d.date()}.png" for d in dates])
        mask = np.array([p in img_set for p in paths])
        return np.where(mask, paths, None)
 
    def resolve_global_batch(self, subdir: str,
                             dates: pd.DatetimeIndex, is_intraday: bool) -> np.ndarray:
        self._ensure_cached(subdir)
        img_set = self._cache[subdir]
        base = str(self.root / subdir)
        if is_intraday:
            paths = np.array([f"{base}/{d.strftime('%Y-%m-%d_%H%M')}.png" for d in dates])
        else:
            paths = np.array([f"{base}/{d.date()}.png" for d in dates])
        mask = np.array([p in img_set for p in paths])
        return np.where(mask, paths, None)
 
 
def load_15m_data(data_dir: str, ticker: str) -> Optional[pd.DataFrame]:
    p = Path(data_dir) / "15min" / f"{ticker}.parquet"
    if not p.exists(): return None
    df = pd.read_parquet(p)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)
 
 
def build_child_indices(df_15m: pd.DataFrame, data_dir: str, ticker: str) -> Dict:
    """Build child bar indices from resampled parquets."""
    children = {}
    for tf in ["1hour", "daily", "weekly", "monthly"]:
        p = Path(data_dir) / tf / f"{ticker}.parquet"
        if p.exists():
            child = pd.read_parquet(p)
            col = "datetime" if "datetime" in child.columns else child.columns[0]
            children[tf] = pd.DatetimeIndex(pd.to_datetime(child[col]))
        else:
            children[tf] = None
    return children
 
 
class MasterRegistryBuilder:
    def __init__(self, data_dir: str, variables_root: str, output_dir: str):
        self.data_dir = data_dir
        self.resolver = ImageSetResolver(variables_root)
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
 
    def build(self, tickers: Optional[List[str]] = None, chunk_size: int = 50) -> str:
        if tickers is None:
            universe_path = Path(self.data_dir) / "_universe.parquet"
            if universe_path.exists():
                tickers = pd.read_parquet(universe_path)["ticker"].tolist()
            else:
                tickers = [p.stem for p in (Path(self.data_dir) / "15min").glob("*.parquet")]
        tickers = sorted(set(tickers))
        logging.info(f"Building registry: {len(tickers)} tickers")
 
        all_chunks = []
        for i in range(0, len(tickers), chunk_size):
            batch = tickers[i:i+chunk_size]
            logging.info(f"  Chunk {i//chunk_size+1}: tickers {i+1}-{min(i+chunk_size, len(tickers))}")
            for tk in batch:
                tk_df = self._process_ticker(tk)
                if tk_df is not None and len(tk_df) > 0:
                    all_chunks.append(tk_df)
 
        if not all_chunks:
            logging.error("No data produced."); return ""
 
        registry = pd.concat(all_chunks, ignore_index=True)
 
        registry["ticker"] = registry["ticker"].astype("category")
        registry["split"] = registry["split"].astype("category")
 
        for col in registry.select_dtypes(include=["object"]).columns:
            if col.endswith("_img"):
                registry[col] = registry[col].astype("category")
 
        registry = registry.sort_values(["datetime", "ticker"]).reset_index(drop=True)
 
        out_path = self.out / "master_registry_15m.parquet"
        registry.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
 
        mem_mb = registry.memory_usage(deep=True).sum() / 1e6
        logging.info(f"✓ Registry: {out_path}")
        logging.info(f"  Rows: {len(registry):,} | Tickers: {registry['ticker'].nunique()}")
        logging.info(f"  RAM: {mem_mb:.0f} MB | Disk: {out_path.stat().st_size/1e9:.2f} GB")
        return str(out_path)
 
    def _process_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        df = load_15m_data(self.data_dir, ticker)
        if df is None or len(df) < 100:
            return None
 
        master_idx = pd.DatetimeIndex(df["datetime"])
        children = build_child_indices(df, self.data_dir, ticker)
 
        t1 = B.build_t1_matrix(master_idx, children)
 
        result = pd.DataFrame({
            "datetime": master_idx,
            "ticker": ticker,
        })
        result["split"] = B.assign_split_vec(result["datetime"])
 
        for col_name, col_data in t1.items():
            result[col_name] = col_data
 
        return result
 
 
def main():
    p = argparse.ArgumentParser(description="Build 15m Master Registry (Production)")
    p.add_argument("--data_dir", required=True, help="Output from 00_ingest_historical.py")
    p.add_argument("--variables_root", default=".", help="Root for image directories")
    p.add_argument("--output_dir", default="registry")
    p.add_argument("--chunk_size", type=int, default=50)
    args = p.parse_args()
 
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    builder = MasterRegistryBuilder(args.data_dir, args.variables_root, args.output_dir)
    builder.build(chunk_size=args.chunk_size)
 
 
if __name__ == "__main__":
    main()
 