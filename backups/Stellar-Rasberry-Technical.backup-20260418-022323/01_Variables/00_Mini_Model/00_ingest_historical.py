"""
00_ingest_historical.py — 3M-Point Data Acquisition Firewall
==============================================================
yfinance caps 15m data at 60 days (~2,500 bars per ticker).
For 3M+ samples across 500+ tickers we need multi-year intraday history.
 
Ingestion Priority:
  1. LOCAL PARQUET ARCHIVE (fastest — no API calls)
     Expected layout: {archive_dir}/{ticker}_15m.parquet
     Schema: datetime, open, high, low, close, volume
  2. POLYGON.IO REST API (2+ years of 15m bars, requires API key)
     Free tier: 5 calls/min. Paid: unlimited.
  3. YFINANCE FALLBACK (last 60 days only — gap-fill for recent data)
 
Output: {output_dir}/{ticker}_15m.parquet per ticker
        {output_dir}/_universe.parquet (master ticker list with date ranges)
 
Resampling: this script also generates 1h/D/W/M parquets from the
15m source, ensuring all timeframes derive from the same raw data.
 
M3 Ultra: multiprocessing with 24 workers for parallel ticker ingestion.
"""
 
import os, sys, json, time, logging, argparse
from pathlib import Path
from typing import Optional
from multiprocessing import Pool, cpu_count
from functools import partial
 
import numpy as np
import pandas as pd
 
RESAMPLE_MAP = {
    "1hour":   {"rule": "1h",    "agg": {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}},
    "daily":   {"rule": "1D",    "agg": {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}},
    "weekly":  {"rule": "W-FRI", "agg": {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}},
    "monthly": {"rule": "ME",    "agg": {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}},
}
 
MIN_BARS_15M  = 5000
MIN_AVG_VOL   = 500_000
N_WORKERS     = min(cpu_count(), 24)
 
 
def get_universe() -> list:
    try:
        t = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        cur = t[0]["Symbol"].tolist()
        h = t[1] if len(t) > 1 else pd.DataFrame()
        a = h["Added"]["Ticker"].dropna().tolist() if not h.empty and "Added" in h.columns.get_level_values(0) else []
        r = h["Removed"]["Ticker"].dropna().tolist() if not h.empty and "Removed" in h.columns.get_level_values(0) else []
        ticks = list(set(cur + a + r))
    except Exception:
        ticks = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","UNH","XOM",
                 "V","PG","HD","CVX","MA","BAC","ABBV","KO","PEP","MRK"]
    return sorted(set(s.replace(".","-").strip().upper() for s in ticks if 1 <= len(s.strip()) <= 6))
 
 
def load_local_parquet(archive_dir: str, ticker: str) -> Optional[pd.DataFrame]:
    p = Path(archive_dir) / f"{ticker}_15m.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "datetime" not in df.columns and df.index.name == "datetime":
        df = df.reset_index()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df
 
 
def fetch_polygon(ticker: str, api_key: str, start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        import requests
    except ImportError:
        return None
 
    all_bars = []
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/15/minute/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}
 
    while url:
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(12)
                continue
            data = resp.json()
            if data.get("resultsCount", 0) == 0:
                break
            results = data.get("results", [])
            for r in results:
                all_bars.append({
                    "datetime": pd.Timestamp(r["t"], unit="ms", tz="America/New_York").tz_localize(None),
                    "open": r["o"], "high": r["h"], "low": r["l"],
                    "close": r["c"], "volume": r["v"],
                })
            url = data.get("next_url")
            params = {"apiKey": api_key} if url else {}
            time.sleep(0.25)
        except Exception:
            break
 
    if not all_bars:
        return None
    df = pd.DataFrame(all_bars).sort_values("datetime").reset_index(drop=True)
    return df
 
 
def fetch_yfinance_recent(ticker: str) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        raw = yf.download(ticker, period="60d", interval="15m", auto_adjust=True, progress=False)
        if raw.empty or len(raw) < 100:
            return None
        df = raw[["Open","High","Low","Close","Volume"]].copy()
        df.columns = ["open","high","low","close","volume"]
        df = df.reset_index()
        df.columns = ["datetime"] + list(df.columns[1:])
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
        return df
    except Exception:
        return None
 
 
def merge_sources(local: Optional[pd.DataFrame],
                  polygon: Optional[pd.DataFrame],
                  yf_recent: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    frames = [f for f in [local, polygon, yf_recent] if f is not None and len(f) > 0]
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["datetime"])
    combined = combined.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    combined = combined.reset_index(drop=True)
    return combined
 
 
def apply_liquidity_filter(df: pd.DataFrame) -> pd.DataFrame:
    avg_vol = df["volume"].rolling(80, min_periods=20).mean()
    return df[avg_vol >= MIN_AVG_VOL / 26].copy()
 
 
def resample_all(df_15m: pd.DataFrame, output_dir: Path, ticker: str):
    df = df_15m.set_index("datetime").sort_index()
    for tf_name, cfg in RESAMPLE_MAP.items():
        resampled = df.resample(cfg["rule"]).agg(cfg["agg"]).dropna(subset=["close"])
        out_path = output_dir / tf_name / f"{ticker}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        resampled.reset_index().to_parquet(out_path, engine="pyarrow",
                                           compression="zstd", index=False)
 
 
def ingest_single_ticker(ticker: str, archive_dir: str, polygon_key: str,
                         output_dir: str, start: str, end: str) -> dict:
    out = Path(output_dir)
    result = {"ticker": ticker, "status": "fail", "bars": 0, "source": "none"}
 
    local = load_local_parquet(archive_dir, ticker) if archive_dir else None
    polygon = None
    if polygon_key and (local is None or len(local) < MIN_BARS_15M):
        polygon = fetch_polygon(ticker, polygon_key, start, end)
    yf_data = fetch_yfinance_recent(ticker)
 
    merged = merge_sources(local, polygon, yf_data)
    if merged is None or len(merged) < MIN_BARS_15M:
        return result
 
    merged = apply_liquidity_filter(merged)
    if len(merged) < MIN_BARS_15M:
        return result
 
    p15 = out / "15min" / f"{ticker}.parquet"
    p15.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(p15, engine="pyarrow", compression="zstd", index=False)
 
    resample_all(merged, out, ticker)
 
    src = "local" if local is not None and len(local) > 0 else \
          "polygon" if polygon is not None else "yfinance"
    return {"ticker": ticker, "status": "ok", "bars": len(merged),
            "start": str(merged["datetime"].min()), "end": str(merged["datetime"].max()),
            "source": src}
 
 
def main():
    p = argparse.ArgumentParser(description="3M-Point Historical Data Ingestion")
    p.add_argument("--archive_dir", default=None,
                   help="Local parquet archive: {dir}/{TICKER}_15m.parquet")
    p.add_argument("--polygon_key", default=os.environ.get("POLYGON_API_KEY"),
                   help="Polygon.io API key (or set POLYGON_API_KEY env var)")
    p.add_argument("--output_dir", default="data/ohlcv", help="Output directory")
    p.add_argument("--start", default="2020-01-01", help="Start date for polygon fetch")
    p.add_argument("--end", default="2025-12-31", help="End date")
    p.add_argument("--workers", type=int, default=N_WORKERS)
    p.add_argument("--tickers", nargs="*", default=None, help="Specific tickers (default: full universe)")
    args = p.parse_args()
 
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
 
    tickers = args.tickers or get_universe()
    logging.info(f"Universe: {len(tickers)} tickers | Workers: {args.workers}")
    logging.info(f"Sources: local={args.archive_dir or 'none'} | "
                 f"polygon={'yes' if args.polygon_key else 'no'} | yfinance=fallback")
 
    fn = partial(ingest_single_ticker,
                 archive_dir=args.archive_dir, polygon_key=args.polygon_key,
                 output_dir=args.output_dir, start=args.start, end=args.end)
 
    if args.workers > 1 and args.polygon_key is None:
        with Pool(args.workers) as pool:
            results = pool.map(fn, tickers)
    else:
        results = [fn(tk) for tk in tickers]
 
    ok = [r for r in results if r["status"] == "ok"]
    fail = [r for r in results if r["status"] == "fail"]
 
    universe_df = pd.DataFrame(ok)
    if len(universe_df) > 0:
        up = Path(args.output_dir) / "_universe.parquet"
        universe_df.to_parquet(up, index=False)
 
    logging.info(f"{'='*60}")
    logging.info(f"Ingested: {len(ok)} tickers | Failed: {len(fail)}")
    if len(ok) > 0:
        total_bars = sum(r["bars"] for r in ok)
        logging.info(f"Total 15m bars: {total_bars:,}")
        logging.info(f"Resampled to: 1hour, daily, W-FRI, ME")
    logging.info(f"Output: {args.output_dir}/")
    logging.info(f"{'='*60}")
 
 
if __name__ == "__main__":
    main()