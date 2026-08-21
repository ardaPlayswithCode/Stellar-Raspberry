# CLAUDE.md — Stellar-Rasberry Sovereign Extraction Engine
# THE COMPLETE PROJECT BRIEF — READ ENTIRELY BEFORE DOING ANYTHING
# Version 12.0 | April 18, 2026 | Builder: Arda Turgut, UC Berkeley Freshman, Age 18
# Architecture: 15-Minute LCM Master Clock (v3.0)
# Data Source: IBKR TWS API via ib_insync (replaces yfinance/Polygon)
# File Structure: Timeframe-first, Version-nested (V1/ inside each timeframe)
# Companion: Stellar-Rasberry Technical (Live Execution Interceptor)

---

## ══════════════════════════════════════════════════════
## SECTION 0 — NEW CHAT ORIENTATION (READ FIRST)
## ══════════════════════════════════════════════════════

This document is the complete project brain. Every architectural decision,
every constant, every research paper finding, every formula, every file name,
every business rule is recorded here. When starting a new chat, read this
entire file before writing a single line of code. Never assume — everything
is defined below.

### Who You Are Working With

Arda Turgut | UC Berkeley Freshman | Age 18 | Palo Alto, CA

- Father worked at dotdata (automated feature engineering company).
  → Proactively suggest dotdata research papers when Arda asks "what else
    can I add" to the trading system. Relevant names: Kondo & Uchiyama,
    DFLR (Dynamic Feature Learning for Regression).
- Goal: Build fully autonomous systematic swing trading system →
  subscription signals business → prop fleet → personal AUM → $15M+ net.
- Hardware: MacBook Air M1 (current, light scripting only).
  Mac Studio M3 Ultra 96GB **ordered early April 2026, arriving ~April 2026**.
- DO NOT TRAIN ON MACBOOK AIR M1 (confirmed: thermal limits, batch=8
  forces noise scale g=2.58 which destroys training).
- IBKR TWS API handshake **SUCCESSFUL** on April 18, 2026 — account
  authenticated, live port 7496 verified from 127.0.0.1.

### Two-Project Architecture (v12.0 — unchanged from v11.0)

```
Stellar-Rasberry/                    ← RESEARCH VESSEL (full 3M-point backtest engine)
Stellar-Rasberry-Technical/          ← HIGH-SPEED INTERCEPTOR (live execution, prop firms)
```

Both projects share the same variable logic (V01–V08) but serve different
purposes. The Research Vessel trains and validates. The Technical
Interceptor executes live trades minute-by-minute against prop firm
accounts and personal capital.

### Current Build Status (v12.0 — April 18, 2026)

```
STRUCTURAL FOUNDATION (00_Mini_Model):
  00_ingest_historical.py     COMPLETE — REWRITTEN FOR IBKR (April 18, 2026)
                                         ib_insync, sequential, 12s pacing,
                                         resume-safe, back to IPO date
  01_base_modules_.py         COMPLETE — vectorized T-1, parallel render,
                                         glob sets, all physics constants
  02_build_god_registry.py    COMPLETE — QuantileDMatrix-ready, category dtypes
  03_trainer_v2.py            COMPLETE — 80-feature XGBoost, self-ref mask
                                         in data path, chunked extraction

CNN TIER (5-resolution: 15min/1hour/daily/weekly/monthly):
  01_Stock_Geometry           COMPLETE — 7 files per V1, ConvNeXt-Base
                                         trainer.py UPDATED April 18, 2026:
                                           per-epoch checkpointing
                                           metrics_{res}.csv fine-grained output
                                         analyze_cnn.py UPDATED April 18, 2026:
                                           S07 training-curves panel added
  02_Sector_Geometry          COMPLETE — 7 files per V1, MobileNetV3 + self-ref
  03_SPY_Geometry             COMPLETE — 7 files per V1, MobileNetV3 ×2, VIX inverted

NUMERICAL TIER (4 resolutions each):
  04_RSI_Divergence           COMPLETE — 6 files, Wilder EWM, 12-combo grid
  05_MACD_Momentum            COMPLETE — 6 files, standard EMA, 24-combo grid
  06_Volume_Trend             COMPLETE — 6 files, dollar volume, 18-combo grid

STATE TIER (no trainers — context only):
  07_Volatility_Context       COMPLETE — 5 files, VIX + ATR% + Parkinson
  08_Range_Context            COMPLETE — 5 files, 52w position + Huddart staleness

STELLAR-RASBERRY TECHNICAL (LIVE EXECUTION LAYER):
  variables/                  DESIGNED — maps V01-V08 to live inference wrappers
  god_model/                  DESIGNED — go.py + correlation_logic.py
  trading_dynamics/           DESIGNED — scaling_rules.py + broker_connection.py + repeat.py
  ibkr_fetch_api.md           DOCUMENTED — IBKR setup + troubleshooting reference

NOT YET BUILT (RESEARCH VESSEL):
  02_God_Model                DESIGNED — not built
  03_Trading_Dynamics         DESIGNED — not built
  04_Order_Execution          DESIGNED — not built
  05_Alpha_Tester             DESIGNED — not built
  06_Stellar_Archive          DESIGNED — not built
```

### WAITING ON MAC STUDIO (timeline moved from future tense to imminent)

All variable code is written. CNN and numerical trainers cannot run on
MacBook Air M1 (batch=8 → g=2.58 → catastrophic noise scale). Once the
Mac Studio M3 Ultra 96GB is fully deployed, the execution order is:

1. Launch TWS and confirm API handshake on port 7496 (DONE — April 18).
2. Run `00_ingest_historical.py` to populate `data/ohlcv/` via IBKR.
   - Validate with 10-ticker slice first (AAPL MSFT GOOGL AMZN NVDA
     META TSLA JPM UNH XOM) before launching the full 7-day backfill.
3. Generate chart PNGs for all CNN variables across all 5 resolutions.
4. Generate numerical signals for V04–V06 across all 4 resolutions.
5. Run triple-barrier labelers per (variable × resolution).
6. Run CNN trainers V01–V03 (5 resolutions each).
7. Run numerical trainers V04–V06 (4 resolutions each, grid search +
   Harvey + BLL).
8. Build master_registry_15m.parquet → train God Model XGBoost.
9. Deploy trained weights to Stellar-Rasberry Technical for live execution.

### FIRST TRAINING RUN PRIORITY

**V1 Stock Geometry, 1hour resolution** is the designated first training
run. Reason: 1hour is the "Structural Horizon" — mid-weight data volume,
strong signal/noise, and produces diagnostic output useful for validating
the entire pipeline before scaling to 15min (4× more data) or daily
(different filename format).

Expected first-run artifacts after successful training:

```
01_Stock_Geometry/1hour/V1/
  ├── model_1hour.pth           ← ConvNeXt-Base weights + temperature
  ├── metrics_1hour.csv         ← per-epoch fine-grained metrics
  └── checkpoints/
        ├── latest.pth          ← rolling resume point
        ├── ckpt_epoch_005.pth
        ├── ckpt_epoch_010.pth
        ├── ckpt_epoch_015.pth
        ├── ckpt_epoch_020.pth
        ├── ckpt_epoch_025.pth
        └── ckpt_epoch_030.pth
```

### CRITICAL ARCHITECTURAL PIVOTS (historical record)

**v9 → v10 (March 2026):** pivoted from daily-resolution master clock
to 15-minute LCM (Least Common Multiple) master clock. All CNN variables
now generate images at 5 resolutions (15min, 1hour, daily, weekly, monthly).
The registry maps every 15m tick to its T-1 aligned higher-timeframe
context, producing ~3M+ time-aligned image references per variable.

**v11 → v12 (April 18, 2026):** three simultaneous pivots:

1. **File structure**: V1 moved from parent of timeframes to child of each
   timeframe. Was `VAR/V1/{timeframe}/`, now `VAR/{timeframe}/V1/`.
   Each timeframe version-trains independently and drops its model into
   its own `V1/` folder. Rationale: each timeframe is trained as an
   independent model with its own parameter search. Subsequent versions
   (V2, V3) of a single timeframe sit alongside V1 *within* that
   timeframe, not at the variable level.

2. **Data source**: yfinance/Polygon replaced with IBKR TWS API via
   `ib_insync`. yfinance caps 15m data at 60 days, Polygon free tier
   caps at 2 years, and Polygon Advanced is $2,400/year. IBKR is free
   for account holders (Arda already has an IBKR account with $10,321
   starting capital), goes back to each ticker's IPO date (1990s-2000s
   for S&P 500 names), and it's the same API powering the live execution
   fleet. One credential, one mental model.

3. **Trainer instrumentation**: per-epoch checkpointing (latest.pth
   rolling + milestone every 5 epochs) and fine-grained metrics CSV
   (train_loss, train_acc, val_loss, val_acc, val_top2_acc, lr,
   epoch_time_s per epoch). S07 added to analyze_cnn.py renders a
   6-panel diagnostic grid from the metrics CSV. Rationale from Arda:
   "in addition to the nearly 10 plots that analyze makes for each time
   frame for each variable, I also want to track the output of the
   training, at a very fine detail, to see, for each parameter it trains,
   how did the model do across different metrics."

---

## ══════════════════════════════════════════════════════
## SECTION 1 — UNIVERSAL CONSTANTS (NON-NEGOTIABLE)
## ══════════════════════════════════════════════════════

```python
# ─── Core Label Set ─────────────────────────────────────────────────────
LABELS = ["strongly_bullish", "neutral", "bearish"]
# Display/reference list. Stored labels in label CSVs and parquets are
# INTEGERS: 0=bearish, 1=neutral, 2=strongly_bullish. trainer.py does
# `int(row.get("label", 1))` at load time.

# ─── ATR — Wilder EWM (applies to EVERY variable — no exceptions) ────────
ATR_PERIOD = 14
# Wilder formula: alpha = 1/period (NOT standard alpha = 2/(n+1))
# Implementation: tr.ewm(span=14, min_periods=14, adjust=False).mean()
# True Range = max(H-L, |H-PrevC|, |L-PrevC|)

# ─── Triple Barrier Labeling ──────────────────────────────────────────────
PROFIT_MULT     = 2.0    # upper barrier = entry + 2.0 × ATR
STOP_MULT       = 1.0    # lower barrier = entry - 1.0 × ATR
VERTICAL_BARS   = 20     # daily: 20 bars ≈ 1 month
VERTICAL_BARS_W = 12     # weekly: 12 bars ≈ 3 months
VERTICAL_BARS_1H   = 40  # 1hour: 40 bars ≈ 6 trading days (swing horizon)
VERTICAL_BARS_15M  = 80  # 15min: 80 bars ≈ 2-3 trading days
# Upper barrier checked FIRST (intentional bullish bias for long-only)

# ─── Temporal Split ───────────────────────────────────────────────────────
TRAIN_END = "2022-12-31"   # extended to capture 2022 rate-hiking cycle
VAL_END   = "2023-12-31"   # test = 2024-01-01 → present (ever-expanding)

# ─── Universe ─────────────────────────────────────────────────────────────
MIN_AVG_VOLUME = 500_000   # 20-day rolling average daily volume
UNIVERSE       = "Wikipedia S&P 500 historical constituents (~800 tickers)"
# Survivorship-bias-free: current + added + removed
# IBKR ticker format: BRK.B → "BRK B" (space instead of dot); saved as BRK_B.parquet

# ─── Fetch Windows ────────────────────────────────────────────────────────
FETCH_START_DAILY  = "2005-01-01"
FETCH_START_WEEKLY = "2004-01-01"
# 15m history: back to IPO date per ticker (IBKR historical depth permitting)

# ─── Weekly Anchor ────────────────────────────────────────────────────────
RESAMPLE_FREQ = "W-FRI"   # Friday 16:00 EST close — universal

# ─── Image Resolution ────────────────────────────────────────────────────
PX  = 680    # 680×680 pixels, all chart images, all resolutions
DPI = 100    # FIG = PX/DPI = 6.8 inches

# ─── Indicator Physics (ZERO DRIFT — same across all resolutions) ─────
RSI_P    = 14      # Wilder EWM: alpha=1/14, adjust=False
MACD_F   = 12      # Standard EMA: span=12, adjust=False (Neftci, NOT Wilder)
MACD_S   = 26      # Standard EMA: span=26, adjust=False
MACD_SIG = 9       # Standard EMA: span=9, adjust=False

# ─── Net Return Multiplier ────────────────────────────────────────────────
NET_MULTIPLIER = 0.41
# = (1-0.003)(1-0.31)(1-0.18)(1-0.05)
# = (1 - 0.30% slippage)(1 - 31% tax)(1 - 18% IS decay)(1 - 5% missed signals)
NET_TARGET     = 0.25     # 25% post-tax annually
```

---

## ══════════════════════════════════════════════════════
## SECTION 2 — 15-MINUTE LCM ARCHITECTURE
## ══════════════════════════════════════════════════════

### 2.1 The Multi-Timeframe Alignment (MTA)

**Master Clock**: 15-minute bars (the Least Common Multiple of all execution
timeframes).

| Resolution | Lookback | Folder Name | Role | Status |
|------------|----------|-------------|------|--------|
| 15min | 60 bars (15h) | `15min` | LCM master clock / technical execution | BUILT |
| 1hour | 60 bars (8.6 days) | `1hour` | Structural horizon / trend bias | BUILT |
| Daily | 60 bars (~3 months) | `daily` | Swing context | BUILT |
| Weekly (W-FRI) | 52 bars (1 year) | `weekly` | J&T 12-1 skip momentum | BUILT |
| Monthly (ME) | 12 bars (1 year) | `monthly` | Bear filter (L1/L2/L3) | BUILT |

**Folder names are `15min` and `1hour` — NOT `15m`/`1h`. Non-negotiable.**

### 2.2 T-1 Alignment (No Look-Ahead)

For each 15m tick at time `t`, higher timeframe context uses:

```
img_τ(t) = img_τ(max{s ∈ T_τ : s < t})
```

Implementation: `np.searchsorted(child_ns, master_ns, side="right") - 1`
on int64 nanosecond arrays. O(N log M) for 3M master rows. No Python loop.

### 2.3 Forward-Fill Persistence (SSD Throughput Optimization)

Higher-TF images are forward-filled across 15m ticks:

- Monthly images change ~12×/year → ffill across ~500K ticks.
- Weekly changes ~52×/year → ffill across ~115K ticks.
- Daily changes ~252×/year → ffill across ~26 ticks.

This minimizes disk I/O on the M3 Ultra's 7.4 GB/s SSD.

### 2.4 The Pipeline

```
00_ingest_historical.py (IBKR)
         ↓
data/ohlcv/{15min,1hour,daily,weekly,monthly}/{TICKER}.parquet
         ↓
V1–V3 Generators (×5 res)  →  680×680 PNG charts via Pool(24)
V4–V8 Generators (×4 res)  →  float scores in parquet
         ↓
label_{res}.py per (variable × resolution)  →  labels_{res}.csv
         ↓
trainer.py per (variable × resolution)  →  model_{res}.pth +
                                            metrics_{res}.csv in V1/
         ↓
02_build_god_registry.py  →  master_registry_15m.parquet (~20M rows)
         ↓
03_trainer_v2.py Stage 1  →  CNN probabilities (60 features)
03_trainer_v2.py Stage 2  →  XGBoost meta-learner (80 features → [0,1])
         ↓
Trading Dynamics  →  position sizing, scale in/out, prop firm compliance
```

### 2.5 First Training Run Recipe (V1 / Stock Geometry / 1hour)

This is the canonical first-run pipeline as of April 18, 2026, with the
Mac Studio M3 Ultra deployed and TWS API verified:

```bash
# From Stellar-Rasberry-Technical/01_Variables/

# (0) Keep the Mac awake for the duration of the run
caffeinate -d &

# (1) IBKR data ingest — populates data/ohlcv/{15min,1hour,daily,weekly,monthly}/
python3 00_Mini_Model/00_ingest_historical.py --port 7496

# (2) Triple-Barrier label generation for 1hour
python3 01_Stock_Geometry/1hour/label_1hour.py --data_dir data/ohlcv --workers 24

# (3) Chart PNG generation for 1hour
python3 01_Stock_Geometry/1hour/generate_1hour.py --data_dir data/ohlcv --workers 24

# (4) ConvNeXt-Base training — outputs to 01_Stock_Geometry/1hour/V1/
python3 01_Stock_Geometry/trainer.py \
  --resolution 1hour \
  --data_dir   01_Stock_Geometry/1hour/images \
  --label_csv  01_Stock_Geometry/1hour/labels_1hour.csv \
  --output_dir 01_Stock_Geometry/1hour/V1 \
  --epochs 30 --device mps --num_workers 4
```

---

## ══════════════════════════════════════════════════════
## SECTION 3 — PRODUCTION CONSTRAINTS (NON-NEGOTIABLE)
## ══════════════════════════════════════════════════════

| # | Constraint | Implementation |
|---|-----------|----------------|
| 1 | **No yfinance anywhere** | Only `00_ingest_historical.py` touches the network, and only via IBKR TWS API on localhost:7496 |
| 2 | **No `p.exists()` in render loops** | `build_image_set()` → `rglob("*.png")` into `set()` for O(1) lookup, then dropped before DataLoader forks |
| 3 | **Pool(24)** for all image rendering | `render_batch_{4p,3p,2p,vix}(jobs, n_workers=N_CORES)`, `chunksize=64` |
| 4 | **Local parquet only** | All data from `data/ohlcv/{resolution}/{TICKER}.parquet` |
| 5 | **zstd compression** | All parquet outputs use `compression="zstd"` |
| 6 | **Category dtypes** | ticker, split, img columns → `pd.Categorical` (~60% RAM reduction) |
| 7 | **QuantileDMatrix** | `xgb.QuantileDMatrix(max_bin=256)` for 3M+ rows (~50% RAM) |
| 8 | **Path anchoring** | `THIS = Path(__file__).resolve().parent` everywhere |
| 9 | **Self-ref mask in data path** | `apply_self_ref_mask()` BEFORE DMatrix — not in loss function |
| 10 | **Chunked CNN extraction** | 10K rows/batch — no 3M-row MPS tensor OOM |
| 11 | **Glob-set dropped pre-fork** | `del existing` in ChartDataset before DataLoader spawns workers |
| 12 | **Per-epoch checkpointing** | `latest.pth` overwritten every epoch; `ckpt_epoch_{N}.pth` every 5 |

### API Firewall (00_ingest_historical.py) — IBKR-BASED

The previous yfinance/Polygon firewall design (v11.0) is obsolete.
Current firewall (v12.0, April 18, 2026):

| Priority | Source | Coverage | Rate Limit |
|----------|--------|----------|------------|
| 1 | Local parquet archive | Unlimited | None (disk) — optional, supplied via `--archive_dir` |
| 2 | IBKR TWS API (ib_insync) | Back to IPO | 60 requests / 10 min (documented); we use 12s pacing = 50 req/10min safe margin |

All higher timeframes resampled FROM the 15m source → perfect timestamp
alignment.

**IBKR specifics:**
- Port 7496 live, 7497 paper (TWS); 4001 live, 4002 paper (IB Gateway).
- Arda's setup verified on port 7496 (live TWS) from 127.0.0.1 on April 18, 2026.
- Sequential fetching by design — IBKR pacing is global per account.
- Resume-safe — tickers with ≥5,000 15m bars on disk are skipped.
- Full 800-ticker × 20-year backfill estimate: **5–7 days continuous**.
- Daily incremental updates after initial backfill: minutes.

---

## ══════════════════════════════════════════════════════
## SECTION 4 — FILE STRUCTURE (v12.0 — CURRENT)
## ══════════════════════════════════════════════════════

**Structural change from v11.0**: version (V1/) is now a child of each
timeframe folder, not a parent of timeframes. Each timeframe trains
independently and stores its trained model(s) in its own V1/. Subsequent
versions of a timeframe (V2, V3) sit alongside V1 within that timeframe.

```
Stellar-Rasberry/                                  ← RESEARCH VESSEL
├── CLAUDE.md                                      ← THIS FILE
├── .gitignore
│
├── 01_Variables/
│   ├── 00_Mini_Model/                             ← STRUCTURAL FOUNDATION
│   │   ├── 00_ingest_historical.py                (IBKR ib_insync, ~270 lines)
│   │   ├── 01_base_modules_.py                    (physics, T-1, render, ~310 lines)
│   │   ├── 02_build_god_registry.py               (glob sets, categories, ~200 lines)
│   │   └── 03_trainer_v2.py                       (XGBoost meta-learner, ~300 lines)
│   │
│   ├── 01_Stock_Geometry/                         ← CNN, ConvNeXt-Base
│   │   ├── 15min/
│   │   │   ├── generate_15min.py
│   │   │   ├── label_15min.py
│   │   │   └── V1/
│   │   │         ├── model_15min.pth
│   │   │         ├── metrics_15min.csv
│   │   │         └── checkpoints/
│   │   ├── 1hour/
│   │   │   ├── generate_1hour.py                  ← ACTIVE (first training target)
│   │   │   ├── label_1hour.py                     ← ACTIVE
│   │   │   └── V1/                                ← trained model lands here
│   │   │         ├── model_1hour.pth
│   │   │         ├── metrics_1hour.csv
│   │   │         └── checkpoints/
│   │   ├── daily/
│   │   │   ├── generate_daily.py
│   │   │   ├── label_daily.py
│   │   │   └── V1/
│   │   ├── weekly/
│   │   │   ├── generate_weekly.py
│   │   │   ├── label_weekly.py
│   │   │   └── V1/
│   │   ├── monthly/                               ← L1 Bear Filter (2-pane)
│   │   │   ├── generate_monthly.py
│   │   │   └── V1/
│   │   ├── trainer.py                             ← SHARED across resolutions
│   │   └── analyze_cnn.py                         ← SHARED, → plots/
│   │
│   ├── 02_Sector_Geometry/                        ← CNN, MobileNetV3
│   │   ├── 15min/ (generate + label + V1)
│   │   ├── 1hour/ (generate + label + V1)
│   │   ├── daily/ (generate + label + V1)        ← GICS reclassification aware
│   │   ├── weekly/ (generate + label + V1)
│   │   ├── monthly/ (generate + V1)               ← L2 Bear Filter (2-pane)
│   │   ├── trainer.py                             ← Self-ref per-sample mask
│   │   └── analyze_sector.py → plots/
│   │
│   ├── 03_SPY_Geometry/                           ← CNN, MobileNetV3 ×2
│   │   ├── 15min/ (dual: images_spy/ + images_vix/, V1)
│   │   ├── 1hour/ (...)
│   │   ├── daily/ (...)
│   │   ├── weekly/ (...)
│   │   ├── monthly/ (...)                         ← L3 FULL CASH Bear Filter
│   │   ├── trainer.py                             ← --instrument SPY|VIX, VIX inverted
│   │   └── analyze.py → plots/
│   │
│   ├── 04_RSI_Divergence/                         ← NUMERICAL, Wilder EWM
│   │   ├── 15min/ (generate + V1 with selected_params_15min.json)
│   │   ├── 1hour/ (...)
│   │   ├── daily/ (...)
│   │   ├── weekly/ (...)
│   │   ├── trainer.py                             ← Harvey + BLL gating
│   │   └── analyze_rsi.py → plots/                ← 10 diagnostic plots
│   │
│   ├── 05_MACD_Momentum/                          ← NUMERICAL, standard EMA
│   │   ├── 15min/ | 1hour/ | daily/ | weekly/
│   │   ├── trainer.py                             ← Harvey + BLL + DHS IC
│   │   └── analyze_macd.py → plots/               ← 10 plots + DHS quadrant
│   │
│   ├── 06_Volume_Trend/                           ← NUMERICAL, dollar volume
│   │   ├── 15min/ | 1hour/ | daily/ | weekly/
│   │   ├── trainer.py                             ← Harvey + BLL + GKM conditioning
│   │   └── analyze_volume.py → plots/             ← 11 plots incl. GKM + AT-era
│   │
│   ├── 07_Volatility_Context/                     ← STATE, no trainer
│   │   ├── 15min/ (generate + V1; VIX T-1 lookahead guard)
│   │   ├── 1hour/ (generate + V1; VIX T-1 guard)
│   │   ├── daily/ (generate + V1)
│   │   ├── weekly/ (generate + V1)
│   │   └── context_schema.json                    ← Regime thresholds + gating
│   │
│   └── 08_Range_Context/                          ← STATE, no trainer
│       ├── 15min/ (generate + V1; close.shift(1) lookahead guard)
│       ├── 1hour/ (generate + V1; close.shift(1) guard)
│       ├── daily/ (generate + V1)
│       ├── weekly/ (generate + V1)
│       └── context_schema.json                    ← Kahneman asymmetry + Huddart
│
├── 02_God_Model/                                  ← DESIGNED, not built
├── 03_Trading_Dynamics/                           ← DESIGNED, not built
├── 04_Order_Execution/                            ← DESIGNED, not built
├── 05_Retraining/                                 ← DESIGNED, not built
└── 06_Archive/                                    ← DESIGNED, not built

Stellar-Rasberry-Technical/                        ← LIVE EXECUTION INTERCEPTOR
├── 01_Variables/                                  ← mirror of Research Vessel variables
│   ├── 00_Mini_Model/
│   ├── 01_Stock_Geometry/                         ← SAME structure as Research Vessel
│   ├── 02_Sector_Geometry/
│   ├── 03_SPY_Geometry/
│   ├── 04_RSI_Divergence/
│   ├── 05_MACD_Momentum/
│   ├── 06_Volume_Trend/
│   ├── 07_Volatility_Context/
│   └── 08_Range_Context/
│
├── 02_God_Model/
│   ├── 01_correlation_logic.py                    ← 20-period rolling corr, sector diversify
│   └── 02_go.py                                   ← scan→infer→rank→filter→order
│
├── 03_Trading_Dynamics/
│   ├── 01_Scaling_Rules.py                        ← Floor, hysteresis, linear scaling
│   ├── 02_Firm_Rules.py                           ← Per-prop-firm drawdown tracking
│   ├── 03_Risk_Manager.py                         ← Kill-switch integration
│   ├── 04_Broker_Connection.py                    ← Mid-point limit orders
│   └── 05_Repeat.py                               ← 60s watchdog loop
│
└── ibkr_fetch_api.md                              ← IBKR setup + troubleshooting
```

### File Naming Convention (CANONICAL — v12.0)

```
Folder hierarchy:   VAR/{resolution}/{V1|V2|V3}/model_{resolution}.pth
                    VAR/{resolution}/generate_{resolution}.py
                    VAR/{resolution}/label_{resolution}.py
                    VAR/trainer.py           (shared)
                    VAR/analyze_{x}.py       (shared)

Folder names:       15min, 1hour, daily, weekly, monthly (NEVER 15m, 1h)
Version folders:    V1, V2, V3 (within each timeframe, not above)
Generators:         generate_{resolution}.py at timeframe root
Labelers:           label_{resolution}.py at timeframe root
Trainers:           trainer.py at variable root (shared across resolutions)
                    Signal vars V04-V06 only; NOT V07-V08 (state vars)
Analyzers:          analyze_cnn.py (V1/V2), analyze.py (V3)
                    Variable-specific: analyze_rsi.py, analyze_macd.py,
                    analyze_volume.py, analyze_sector.py
Trained models:     {res}/V1/model_{resolution}.pth
Metrics:            {res}/V1/metrics_{resolution}.csv
Checkpoints:        {res}/V1/checkpoints/latest.pth
                    {res}/V1/checkpoints/ckpt_epoch_{NNN}.pth
Plot output:        plots/ directory at variable root
Schemas:            context_schema.json for state variables (V07, V08)
Selected params:    {res}/V1/selected_params_{resolution}.json (numerical vars)
```

### Why The New Structure?

The v11.0 structure (`VAR/V1/{timeframe}/`) implied a single version
applies to all timeframes. In practice, each timeframe is a separate
model with its own training run, its own hyperparameter search, its
own failure modes. The 15min and 1hour models may converge on different
augmentation strategies; the weekly model may need different label
horizons. Collapsing them under a single V1 was misleading.

The v12.0 structure (`VAR/{timeframe}/V1/`) makes the independence
explicit. If V1 of the 1hour model works but V1 of the weekly model
needs revision, you create `weekly/V2/` without touching `1hour/V1/`.
Versions are per-timeframe, not per-variable.

---

## ══════════════════════════════════════════════════════
## SECTION 5 — VARIABLE TAXONOMY
## ══════════════════════════════════════════════════════

```
CNN SIGNAL VARIABLES (01, 02, 03):
  → 5-resolution image generators (15min/1hour/D/W/M)
  → CNN backbone extracts 3-class probability [P(bear), P(neutral), P(bull)]
  → 5 res × 3 classes = 15 probabilities per variable
  → Bear filter hierarchy: L1 (stock) → L2 (sector) → L3 (SPY=FULL CASH)

NUMERICAL SIGNAL VARIABLES (04, 05, 06):
  → bull_score + bear_score ∈ [0, 1] via tanh/MOP normalization
  → Grid search over parameter combos, Harvey t-stat gating
  → IC targets, BLL bootstrap, HRP weights
  → V04: 12-combo grid (RSI periods × div lookbacks), Wilder EWM
  → V05: 24-combo grid (fast × slow × signal EMAs), standard EMA,
          DHS derivative FIXED
  → V06: 18-combo grid (lookback × norm × ewm), dollar volume,
          GKM conditioning

STATE VARIABLES (07, 08) — "Ghost Variables":
  → NO IC target, NO trainer, NO Harvey weight, NO BLL bootstrap
  → V07 = Reliability Multiplier (HOW MUCH to trust) — VIX + ATR% + Parkinson
  → V08 = Bias Shift (WHICH WAY to interpret) — v6_score + staleness + range_pctl
  → Feed XGBoost meta-learner via Feature Interaction Constraints
  → Intraday lookahead guards: V07 uses VIX.shift(1), V08 uses close.shift(1)
```

### Standalone-Per-Ticker Constraint (NEVER VIOLATE)

All variables in 01_Variables must be computable per-ticker in ISOLATION.
No cross-sectional computations. Cross-sectional ops belong in 02_God_Model.

### Training Horizons (VERTICAL_BARS per resolution)

| Resolution | VERTICAL_BARS | Approx span | Rationale |
|------------|---------------|-------------|-----------|
| 15min | 80 | 2-3 trading days | Enough room for TP at 2×ATR within typical volatility |
| 1hour | 40 | ~6 trading days | Swing-appropriate; matches prop firm EOD horizons |
| daily | 20 | ~1 month | CLAUDE.md canonical; J&T formation-period-lite |
| weekly | 12 | ~3 months | CLAUDE.md canonical |
| monthly | — | — | No labels; bear filter only |

---

## ══════════════════════════════════════════════════════
## SECTION 6 — VAR 01: STOCK GEOMETRY (CNN)
## ══════════════════════════════════════════════════════

### Architecture

- **Backbone**: ConvNeXt-Base (heavier — justified by 500+ ticker diversity).
- **Weights**: `models.ConvNeXt_Base_Weights.IMAGENET1K_V1` (pretrained).
- **Input**: 224×224 (resized from 680×680 rendered charts).
- **Output**: 3-class softmax [P(bearish), P(neutral), P(strongly_bullish)].
- **Panels**: 4-pane (candle + volume + RSI + MACD histogram) for
  15min/1hour/daily/weekly. 2-pane (candle + volume) for monthly
  (L1 bear filter).
- **Panel ratios**: [6, 1.5, 1.25, 1.25] (4-pane), [3, 1] (2-pane).

### Training

- **Loss**: Focal Loss γ=2.0 (Lin et al. 2017) + Label Smoothing ε=0.1.
- **Augmentation**: Mixup α=0.3 (Zhang et al. 2018).
- **Init**: Zero-gamma on final LayerNorm of each CNBlock. NOT all BN
  layers — only `CNBlock.layer_scale`.
- **Calibration**: Guo et al. (2017) post-hoc temperature scaling on
  val set, LBFGS with max_iter=50.
- **Optimizer**: AdamW, lr=1e-4, weight_decay=1e-4.
- **Scheduler**: CosineAnnealingLR, T_max=epochs.
- **Batch size**: 256 (fits ~12GB VRAM on MPS).
- **`--resolution` flag** selects image directory.

### Checkpointing (NEW — April 18, 2026)

Per-epoch writes (keeps long runs resumable):

- `V1/checkpoints/latest.pth` — full state (model, optimizer, scheduler,
  best_val_acc, epoch), overwritten every epoch.
- `V1/checkpoints/ckpt_epoch_{NNN}.pth` — milestone snapshot every
  CKPT_EVERY=5 epochs. Model weights + val_acc only (smaller than latest).
- `V1/model_{res}.pth` — final best-val-acc weights + temperature,
  written once at the end.

Rationale: a 30-epoch training run that crashes at epoch 28 loses
everything without rolling checkpoints. With them, resumption is one
flag away.

### Fine-Grained Metrics (NEW — April 18, 2026)

`V1/metrics_{res}.csv` written row-by-row each epoch. Columns:

| Column | Meaning |
|--------|---------|
| epoch | 1-indexed epoch number |
| train_loss | Mean Focal Loss on train set (mixup-adjusted during training loop) |
| train_acc | Top-1 accuracy on training mini-batches |
| val_loss | Mean Focal Loss on val set (no mixup) |
| val_acc | Top-1 accuracy on val set |
| val_top2_acc | Top-2 accuracy on val set (fraction where true class in top 2 predictions) |
| lr | Current learning rate (for cosine-anneal audit) |
| epoch_time_s | Wall-clock seconds per epoch (thermal-throttle detector) |

Consumed by `analyze_cnn.py` S07 which renders a 6-panel grid:

1. Loss curves (train vs val overlaid).
2. Accuracy curves (top-1 train, top-1 val, top-2 val, 1/3 random baseline).
3. Generalization gap (train_acc − val_acc) with overfitting shading.
4. Learning rate schedule (log scale).
5. Epoch wall-clock time (thermal throttling detection).
6. Loss ratio (val_loss / train_loss — smooth ≈ 1 = healthy).

### Data Pipeline

- Reads from `data/ohlcv/{resolution}/{TICKER}.parquet` (from IBKR ingest).
- `ChartDataset.__init__` globs all PNGs under the images dir into a
  Python set, filters label-DataFrame rows to those with existing
  images, then **drops the set before DataLoader spawns workers**. This
  avoids fork CoW refcount mutation duplicating a multi-million-entry
  set across N workers.
- `render_batch_4p()` with `Pool(24), chunksize=64`.

### Output (per resolution)

Images:  `01_Stock_Geometry/{resolution}/images/{TICKER}/{date}.png`

- Intraday date format: `YYYY-MM-DD_HHMM` (e.g. `2023-03-15_1045.png`).
- Daily/weekly/monthly format: `YYYY-MM-DD` (e.g. `2023-03-15.png`).

Models: `01_Stock_Geometry/{resolution}/V1/model_{resolution}.pth`

### Analyzer Plots (→ plots/ at variable root)

- **S01**: Resolution IC comparison (15min vs 1hour vs daily vs weekly vs monthly).
- **S02**: Physics drift audit (15m bull_prob vs daily bull_prob scatter — ρ ∈ [0.3, 0.7] expected).
- **S03**: Rolling IC (252d) with train/val/test boundaries.
- **S04**: Decile returns by CNN confidence (must be monotonic — "Monotonic Staircase").
- **S05**: Coverage heatmap (tickers × months).
- **S06**: Label distribution by split (train/val/test pie charts).
- **S07**: Training curves — 6-panel diagnostics from metrics CSV (NEW).

### V1 / 1hour — First Training Run Specifics

Label horizon: `VERTICAL_BARS_1H = 40` bars ≈ 6 trading days.
Reason: Claude.md §1 only specifies 20 (daily) and 12 (weekly). For
1hour, 40 is 2/3 of the LOOKBACK=60, which is a swing-appropriate
horizon given the prop firm EOD drawdown framework.

LOOKBACK: 60 bars (~8.6 trading days of 7-hour RTH).

Expected training cost (M3 Ultra 96GB, BS=256, 30 epochs, full S&P 500):
~12-18 hours for full-universe training. Validate pipeline with 10
tickers first.

---

## ══════════════════════════════════════════════════════
## SECTION 7 — VAR 02: SECTOR GEOMETRY (CNN)
## ══════════════════════════════════════════════════════

### Architecture

- **Backbone**: MobileNetV3-Large (lighter — only 11 ETFs).
- **Panels**: 3-pane (candle + volume + RSI) for 15min/1hour/D/W.
  2-pane (candle + volume) for monthly (L2 bear filter).
- **Panel ratios**: [6.5, 2, 1.5] (3-pane).
- **Entities**: 11 SPDR Sector ETFs: XLK XLF XLV XLY XLC XLI XLE XLP XLRE XLU XLB.

### Self-Reference Masking (Non-Negotiable)

When `check_self_reference(ticker, etf) → True` (ticker weight > 5% in ETF):

- **Loss mask = 0.0** in trainer → gradient does NOT flow for that sample.
- **Feature mask = 1/3** in 03_trainer_v2.py → `apply_self_ref_mask()`
  replaces V2 CNN features with uniform prior BEFORE DMatrix construction.

```python
HEAVY_WEIGHTS = {
    "XLK": {"AAPL","MSFT","NVDA","AVGO","CRM"},
    "XLF": {"BRK-B","JPM","V","MA","BAC"},
    "XLV": {"UNH","JNJ","LLY","ABBV","MRK"},
    "XLY": {"AMZN","TSLA","HD","MCD","NKE"},
    "XLC": {"META","GOOGL","GOOG","NFLX","DIS"},
    "XLI": {"GE","CAT","UNP","HON","RTX"},
    "XLE": {"XOM","CVX","COP","SLB","EOG"},
    "XLP": {"PG","COST","KO","PEP","PM"},
    "XLRE":{"AMT","PLD","CCI","EQIX","PSA"},
    "XLU": {"NEE","SO","DUK","SRE","AEP"},
    "XLB": {"LIN","APD","SHW","FCX","ECL"},
}
```

### GICS Reclassification (Point-in-Time)

| Date | Event | Pre-reclassification |
|------|-------|---------------------|
| 2018-09-24 | XLC created | GOOGL/META → XLK, DIS/NFLX → XLY |
| 2015-10-07 | XLRE created | AMT/PLD/CCI → XLF |

`get_sector_etf(ticker, as_of, sector_gics)` in `01_base_modules_` handles
this. For dates before XLC_DATE, XLC constituents are routed to their
prior sector. For dates before XLRE_DATE, XLRE constituents are routed
to XLF (financials).

### Key Papers

- Moskowitz & Grinblatt (1999) JF 54(4) — Industry momentum explains
  most of J&T.
- Chan, Hameed & Tong (2000) JF 55(3) — Industry returns predict
  individual stock returns.
- Asness, Moskowitz & Pedersen (2013) JF 68(3) — Momentum in sector ETFs.
- Fama & French (1997) JFE 43(2) — Industry as systematic risk factor.

---

## ══════════════════════════════════════════════════════
## SECTION 8 — VAR 03: SPY GEOMETRY (CNN)
## ══════════════════════════════════════════════════════

### Architecture

- **Backbone**: MobileNetV3-Large ×2 (separate models for SPY and VIX).
- **SPY rendering**: 680×680 2-pane (candlestick + volume) → `images_spy/`.
- **VIX rendering**: 680×680 line chart (orange #e67e22, 15% fill) → `images_vix/`.
- **Scope**: GLOBAL — same image for all 500+ tickers on a given timestamp.
- **L3 Bear Filter**: Monthly SPY bearish → confidence=0.0 for ALL tickers → FULL CASH.

### VIX Label Inversion (Non-Negotiable)

```python
if instrument == "VIX":
    labels = labels.map({0: 2, 1: 1, 2: 0})
```

**Why**: When market is bearish (label=0), VIX RISES. After inversion,
P(bull) from VIX CNN = P(stress rising). High `v3_vix_bull_prob` =
high fear. This lets the meta-learner treat the VIX CNN's output with
the same bull-probability semantics as the SPY CNN's output, without
special-casing the sign.

### Bear Filter Hierarchy

```
L1 (Stock Monthly bearish)   → reduce THAT ticker's position size
L2 (Sector Monthly bearish)  → reduce ALL tickers in that sector
L3 (SPY Monthly bearish)     → OVERRIDE ALL SIGNALS → FULL CASH
```

L3 is the circuit breaker. Nothing can override it.

### Relationship to V07 Volatility Context

V07 gives the RAW VIX level (state variable, no transformation).
V03 gives the PATTERN of VIX (compression coil, spike reversal, breakout).
They are orthogonal. Both needed.

### Key Papers

- Whaley (2000) JPM 26(3) — VIX fear gauge.
- Schwert (1989) JF 44(5) — Vol persistence (AR sum=0.8994).
- Adrian & Brunnermeier (2016) AER 106(7) — Volatility paradox.
- Rapach, Strauss & Zhou (2010) RFS — Market timing improves stock prediction.

---

## ══════════════════════════════════════════════════════
## SECTION 9 — VAR 04: RSI DIVERGENCE (NUMERICAL) ✓ COMPLETE
## ══════════════════════════════════════════════════════

### Key Design Facts

```python
# RSI: Wilder EWM (alpha=1/period) — NOT standard EWM
# Period grid: [7, 10, 14, 21] × divergence lookback [5, 10, 20] = 12 combos
# Divergence = rsi_rank - price_rank within rolling window
# Positive divergence = bullish (RSI stronger than price suggests)
rsi_raw        = wilder_rsi(close, period)
rsi_divergence = measure_divergence(rsi_raw, close, lookback)
bull_rsi_score = np.clip(np.tanh(rsi_divergence / scale), 0, 1)
bear_rsi_score = np.clip(np.tanh(-rsi_divergence / scale), 0, 1)
```

### KEY INSIGHT — RSI is ANTI-momentum (mean-reversion)

RSI divergence is structurally OPPOSITE to MACD. Partialing out J&T
momentum from RSI IC should INCREASE residual IC. High HRP diversity
weight expected.

### IC Targets

Daily: 0.03-0.07 | Weekly: 0.05-0.09 | Half-life: 3-4 years

### Build Status: ✓ ALL 6 FILES COMPLETE

- 4 generators (15min/1hour/daily/weekly) with wilder_rsi_flex().
- trainer.py with Harvey t-stat gating + BLL bootstrap (1000 permutations).
- analyze_rsi.py with 10 diagnostic plots (P01-P10).

### Key Papers

- Wilder (1978) "New Concepts in Technical Trading Systems" — original RSI.
- Lo, Mamaysky & Wang (2000) JF 55(4) — technical patterns, predictability.

---

## ══════════════════════════════════════════════════════
## SECTION 10 — VAR 05: MACD MOMENTUM (NUMERICAL) ✓ COMPLETE
## ══════════════════════════════════════════════════════

### Key Design Facts

```python
# EMA type: STANDARD span-based (alpha=2/(span+1)) — Appel/Neftci convention
# NOT Wilder's EWM. MACD and RSI use DIFFERENT EMA formulas. Keep distinct.
# Grid: fast[8,12,16] × slow[21,26,34] × signal[7,9,12] = 27 raw → 24 valid
# Constraint: slow - fast >= 8 (eliminates fast=16/slow=21)

# MOP normalization (Moskowitz, Ooi & Pedersen 2012):
hist_norm       = histogram / atr
bull_macd_score = np.clip(np.tanh(hist_norm), 0, 1)
bear_macd_score = np.clip(np.tanh(-hist_norm), 0, 1)

# DHS histogram derivative (Daniel, Hirshleifer & Subrahmanyam 1998):
hist_delta_norm = np.tanh((histogram[t] - histogram[t-1]) / atr)
# FIXED FEATURE — TABULAR_COLUMNS[4:5]. NOT a parameter.
# Not folded into net score.
# Positive hist + accelerating = strong bull
# Positive hist + decelerating = momentum peak (early bear warning)
```

### IC Targets

Daily: 0.04-0.08 | Weekly: 0.05-0.10 | Half-life: 3-4 years

### Build Status: ✓ ALL 6 FILES COMPLETE

- 4 generators (15min/1hour/daily/weekly) with valid_combos() filter.
- trainer.py with Harvey + BLL + DHS derivative IC reporting.
- analyze_macd.py with 10 plots including P09 DHS quadrant analysis.

### Key Papers

- Appel (1985) — original MACD.
- Brock, Lakonishok & LeBaron (1992) JF 47 — technical rules.
- Daniel, Hirshleifer & Subrahmanyam (1998) JF — DHS momentum derivative.
- Moskowitz, Ooi & Pedersen (2012) JFE — MOP vol normalization.
- Neftci (1991) — non-linear cycle detection via EMA crossovers.

---

## ══════════════════════════════════════════════════════
## SECTION 11 — VAR 06: VOLUME TREND (NUMERICAL) ✓ COMPLETE
## ══════════════════════════════════════════════════════

### Research Foundation

**GKM (2001) JF 56(3)** — Volume SHOCK (surprise vs baseline) predicts
returns 1-20 days. NOT absolute volume — the unexpected component.
Dollar volume is canonical. Effect strongest when |same_bar_return|
rank < 30th percentile (normal return bars only). Filtering to
normal-return bars roughly DOUBLES IC — validates GKM Table 4.

**Chordia/Roll/Subrahmanyam (2008) JFE 87(2)** — AR(12) shows ONLY
the innovation coefficient is significant. Raw volume IC ≈ 0 in modern
markets; vol-surprise IC survives.

**Hendershott/Jones/Menkveld (2011) JF 66(1)** — NYSE autoquote Jan
2003 = AT regime shift. AT = ~73% of volume by 2009. Dollar volume
preferred over share count.

### Key Design

```python
dollar_volume   = close × volume
vol_zscore      = (dollar_vol - rolling_mean(lookback)) / rolling_std(lookback)
bull_vol_score  = np.clip(np.tanh(vol_zscore), 0, 1)
bear_vol_score  = np.clip(np.tanh(-vol_zscore), 0, 1)
vol_roc         = np.tanh(delta_vol_zscore)   # GKM acceleration — FIXED column
# Grid: lookback[5,10,20] × norm[z_score, market_ratio, rank] × ewm[T,F] = 18 combos
# Three normalizations: z_score, market_ratio (dv/baseline-1), rank (rolling pctl)
```

### IC Targets

Daily: 0.03-0.07 | Weekly: 0.05-0.10 | Half-life: 4-5 years (GKM durable)

### Build Status: ✓ ALL 6 FILES COMPLETE

- 4 generators with dollar_volume, 3 normalization methods, EWM toggle.
- trainer.py with GKM conditioning (normal-return filter, ic_gkm_conditioned).
- analyze_volume.py with 11 plots: P09 GKM conditioning + P10 AT-era regime.

---

## ══════════════════════════════════════════════════════
## SECTION 12 — VAR 07: VOLATILITY CONTEXT (STATE) ✓ COMPLETE
## ══════════════════════════════════════════════════════

STATE VARIABLE. No trainer. No IC target. No Harvey weight. No labels.

### Sensors

```
GLOBAL:  vix_close      — CBOE VIX daily close (same for ALL tickers)
LOCAL:   atr_pct        — Wilder ATR(14) / Close × 100 (per-ticker)
         parkinson_vol  — sqrt(mean(ln(H/L)²) / (4·ln2)) over rolling window
         atr_ratio_*    — ATR(14) / SMA(ATR, window). >1.0 = expanding
         annual_sigma   — Rolling log-return σ, annualized
```

### Intraday Lookahead Guard (CRITICAL)

15min and 1hour generators use PREVIOUS day's VIX close via `.shift(1)`
on daily VIX before merging by date key. The state is known at market
open.

### Regime Thresholds (context_schema.json)

```
VIX:   Low <15 (Complacent) | Normal 15-22 | High 22-32 | Extreme 32+
ATR%:  Compressed <1% | Normal 1-2.5% | Volatile 2.5-5% | Distressed 5%+
Ratio: Compressing <0.8 | Stable 0.8-1.2 | Expanding 1.2-2.0 | Exploding 2.0+
```

### Build Status: ✓ ALL 5 FILES COMPLETE

- 4 generators (15min/1hour with VIX T-1 guard, daily, weekly).
- context_schema.json with regime thresholds + interaction constraint docs.

### Research

- **Whaley (2000)** JPM 26(3) — VIX = forward-looking 30-day implied vol.
- **Parkinson (1980)** JBus 53(1) — Range estimator 2.5-5× more efficient.
- **Schwert (1989)** JF 44(5) — Vol persistent (AR=0.8994). No z-score needed.
- **Ang et al. (2006)** JF 61(1) — VIX (systematic) ⊥ ATR% (idiosyncratic).
- **Adrian & Brunnermeier (2016)** AER 106(7) — Volatility paradox:
  low VIX = hidden risk.

---

## ══════════════════════════════════════════════════════
## SECTION 13 — VAR 08: RANGE CONTEXT (STATE) ✓ COMPLETE
## ══════════════════════════════════════════════════════

STATE VARIABLE. Bias Shift — changes how V01-V06 signals are INTERPRETED.

### Core Formula

```python
shifted_close = close.shift(1)   # LOOKAHEAD GUARD — window stops at T-1
high_52w = shifted_close.rolling(252, min_periods=126).max()
low_52w  = shifted_close.rolling(252, min_periods=126).min()
v6_score = clip((close - low_52w) / (high_52w - low_52w), 0, 1)
```

### Extended Features

- Z-score from SMA(200): `(Close − SMA200) / σ(Close, 200)`. |Z| > 2 =
  overextended.
- Huddart staleness: `bars_since_52w_high`, `bars_since_52w_low`.
- 3-month range percentile (0-100): compression → expansion transitions.

### Regime Thresholds (context_schema.json)

```
v6_score: Deep Value 0-0.15 | Value 0.15-0.35 | Neutral 0.35-0.65 |
          Momentum 0.65-0.85 | Overextended 0.85-1.0
Z-score:  Deep Discount Z<-2 | Fair Value |Z|<1 | Extreme Premium Z>2
```

### Build Status: ✓ ALL 5 FILES COMPLETE

- 4 generators (15min/1hour with resolution-adapted windows, daily, weekly).
- context_schema.json with Kahneman asymmetry docs + Huddart regime map.

### Research (Full Detail)

**Huddart et al. (2009)** ManSci 55(1) — ZERO significant effects at
intermediate deciles. ONLY MAX and MIN significant. Longer since prior
extreme → STRONGER volume spike (0.0028/week). At 52w LOW:
attention-driven BUY orders.

**Grinblatt & Han (2005)** JFE 78(2) — v6_score proxies aggregate
unrealized capital gains. Once controlled, intermediate-horizon returns
= ZERO (t=0.68). High v6 → momentum reliable. Low v6 → bounce risk HIGH.

**Kahneman & Tversky (1979)** Econometrica 47(2) — λ=2.25 loss aversion.
v6=0.25 generates 2.25× stronger distortion than v6=0.75. NOT symmetric
around 0.5. XGBoost tree splits learn this asymmetry organically.

**Baker, Pan & Wurgler (2012)** JF 67(1) — 52w high = discontinuous
psychological anchor. Acceptance probability jumps 4.4-6.4 ppt at
threshold. Institutional participation → will NOT be arbitraged away.

---

## ══════════════════════════════════════════════════════
## SECTION 14 — META-LEARNER (GOD MODEL)
## ══════════════════════════════════════════════════════

### Feature Vector (80 features)

**CNN Features (60)** — 5 res × 3 classes × 4 model outputs:

| Source | Count | Description |
|--------|-------|-------------|
| V1 Stock | 15 | 5 res × 3 cls (ConvNeXt-Base, per-ticker) |
| V2 Sector | 15 | 5 res × 3 cls (MobileNetV3, per-sector ETF) |
| V3 SPY | 15 | 5 res × 3 cls (MobileNetV3, global market) |
| V3 VIX | 15 | 5 res × 3 cls (MobileNetV3, vol regime, labels inverted) |

**Tabular Features (20)** — TABULAR_COLUMNS:

| Index | Feature | Source |
|-------|---------|--------|
| 0–1 | v4_rsi_daily, v4_rsi_weekly | RSI divergence net score |
| 2–3 | v5_macd_daily, v5_macd_weekly | MACD momentum net score |
| 4–5 | v5_hist_delta_daily, v5_hist_delta_weekly | DHS (1998) derivative — FIXED |
| 6–7 | v6_volume_daily, v6_volume_weekly | GKM (2001) volume net score |
| 8–9 | v6_vol_roc_daily, v6_vol_roc_weekly | GKM acceleration — FIXED |
| 10–11 | v7_vix_close, v7_vix_weekly | Whaley (2000) global fear |
| 12–13 | v7_atr_pct, v7_atr_pct_weekly | Parkinson (1980) local vol |
| 14 | v8_v6_score | Grinblatt & Han (2005) range position |
| 15–16 | v8_bars_since_high, v8_bars_since_low | Huddart (2009) staleness |
| 17–19 | v8 weekly variants | Forward-filled from W-FRI |

### XGBoost Interaction Constraints (3 Groups)

```
G0 SYSTEMATIC: CNN[0:59] + v7_vix_close[10,11]
   → VIX gates ALL chart patterns (Ang et al. 2006)
   → Adrian & Brunnermeier (2016): volatility paradox gating

G1 STRUCTURAL: Numerical[0:9] + VIX[10,11] + V8[14:19]
   → VIX + range position gate numerical signal interpretation
   → Grinblatt & Han (2005): v6_score changes what RSI/MACD "mean"

G2 LOCAL: ATR%[12,13] + CNN[0:59] + Numerical[0:9]
   → Per-ticker idiosyncratic vol gates THAT ticker's signals
   → Ang et al. (2006): idiosyncratic vol orthogonal to VIX
```

### XGBoost Parameters

```python
META_PARAMS = {
    "n_estimators": 2000, "max_depth": 6, "learning_rate": 0.01,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 50,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "tree_method": "hist",
    "objective": "binary:logistic", "eval_metric": "auc",
    "early_stopping_rounds": 100, "seed": 42,
}
```

### HRP Implementation

```python
def get_hrp_weights(signal_returns_df):
    cov  = signal_returns_df.cov()
    corr = signal_returns_df.corr()
    dist = ((1 - corr) / 2.0) ** 0.5
    link = sch.linkage(dist, "single")
    sortIx = get_quasi_diag(link)
    weights = recursive_bisection(cov, sortIx)
    return weights   # sum=1.0
# Run QUARTERLY. Min 252 trading days. RSI+MACD get high diversity weight.
```

### Harvey Weight

```python
def harvey_weight(t_stat):
    if t_stat >= 3.0: return 1.0
    elif t_stat >= 2.0: return (t_stat - 2.0)
    else: return 0.0
```

### Markowitz & Xu (1994) Data Mining Corrections (θ-shrinkage)

**New in v12.0** — from Markowitz Chapter 3 research.

The core test (Model III, the general case from Markowitz & Xu 1994):

```
gᵢ = μᵢ + εᵢ          # log-return of model i
E(εᵢ) = 0
cov(εᵢ, εⱼ) can be ≠ 0    # models are NOT required to be independent
θ ∈ [0, 1]            # shrinkage coefficient regressing best-model return
                        toward grand-average across all tested models
```

**The practical rule**: If you tested N model variants and selected the
best, expected out-of-sample excess return is `θ × backtested_excess_return`,
NOT the full backtested excess return. Markowitz & Xu's original DPOS test
yielded θ = 0.59 (59% of excess returns persist out-of-sample). Guerard's
MQ model yielded θ = 0.47.

**Where this applies in Stellar-Rasberry**:

- **Harvey t-stat gating already approximates this** — but DMC is a
  complementary shrinkage estimator that survives even when t-stats look
  fine. Add to V04–V06 trainers AFTER first full training run completes.
- **God Model: shrink XGBoost OOF Sharpe by θ before go-live business-plan
  projections.** If backtested Sharpe = 2.0 and θ = 0.50, plan for Sharpe = 1.0.
- **Business plan §19.4**: The projected $15M over 5 years assumes full
  persistence of backtested alpha. Apply θ = 0.50 haircut as a
  pessimistic floor scenario.

**Implementation priority**: After V1–V8 + God Model v1 train successfully.
Do not bolt onto individual variables before they have baseline IC.

### Correlation-Budget Diagnostic (post-training)

Compute pairwise Spearman between all 80 features on the train split.
Any pair with |ρ| > 0.90 should be flagged. Feature pruning is a
next-pass refinement, not a first-training concern.

V07 vix_close and V03 VIX probabilities are partially redundant (both
measure fear). Keep both but watch SHAP interaction values on training
completion — if they dominate each other, one is noise.

---

## ══════════════════════════════════════════════════════
## SECTION 15 — STELLAR-RASBERRY TECHNICAL (LIVE EXECUTION)
## ══════════════════════════════════════════════════════

### 15.1 Philosophy

Stellar-Rasberry Technical is the live execution companion — a high-speed
interceptor stripped of fundamental/sentiment data. It uses the trained
weights from the Research Vessel to execute minute-by-minute against
prop firm accounts and personal capital ($10,321 starting).

```
Research Vessel:  trains models, validates, backtests (offline, Mac Studio)
Technical:        runs inference, selects stocks, executes orders (live, 60s loop)
```

### 15.2 Three Power Blocks

**Block 1: variables/** — The Feature & Intelligence Factory

- 00_meta_learner.py: Trained weights. V07/V08 act as gating mechanisms.
- v01_cnn.py through v08_range.py: Signal generators processing last
  15-20 bars of OHLCV data to produce independent confidence scores
  ∈ [0, 1].
- Output: Raw matrix of confidence scores for all 500 stocks.

**Block 2: god_model/** — The Orchestrator & Filter

- go.py: Master execution script. Pipeline:
  1. Market Scan — pull latest minute-bars for S&P 500.
  2. Inference — feed into variables/ to get meta-learner confidence.
  3. Raw Rank — identify Top 15 highest-confidence stocks.
  4. Filter — call correlation_logic.py to prune to Top 10.
- correlation_logic.py: 20-period rolling correlation. If the model
  loves five semiconductors, only the strongest one (or two) survives.
  Forces diversification into uncorrelated winners.

**Block 3: trading_dynamics/** — Execution & Heartbeat

- scaling_rules.py: Mathematical Shield
  - Floor: confidence < 60% → full exit.
  - Hysteresis: sold at 60% → locked until 80% (institutional-swing profile, UPDATED v12.0).
  - Linear Scaling: position size ∝ confidence (more capital at 95% vs 70%).
- broker_connection.py: API bridge. Mid-point limit orders (middle of
  bid/ask) to avoid the spread tax. Pre-earnings exit: always 3 days
  before.
- repeat.py: 60-second watchdog. If go.py crashes, repeat.py re-launches
  on the next minute. Network-resilient.

### 15.3 The Minute-by-Minute Execution Loop

```
repeat.py → go.py → Var Generation → Top 15 → Corr Filter → Top 10 → Limit Orders
```

| Time | Action |
|------|--------|
| T-60s | repeat.py triggers go.py |
| T-50s | Data scraped; V01-V08 generate scores; meta-learner produces confidence |
| T-30s | correlation_logic.py evaluates Top 15 → Top 10 uncorrelated |
| T-15s | go.py reads scaling_rules.py. Compare holdings vs target Top 10 |
| T-05s | broker_connection.py sends mid-point limit orders (trim/add) |
| T-00s | Loop resets |

### 15.4 Position Parameters (Technical — UPDATED v12.0)

```python
STARTING_CAPITAL = 10_321
MAX_POSITIONS    = 10          # configurable in scaling_rules.py
CONFIDENCE_FLOOR = 0.60       # exit trigger
HYSTERESIS_CLEAR = 0.80       # re-entry threshold (institutional-swing profile)
TOP_N_RAW        = 15         # pre-filter candidate count
TOP_N_FINAL      = 10         # post-correlation diversified count
CORR_WINDOW      = 20         # rolling correlation lookback (bars)
CORR_THRESHOLD   = 0.70       # max allowed pairwise correlation
CORR_FALLBACK    = 0.80       # relaxed threshold if < 10 names survive
CORR_MIN_NAMES   = 8          # never drop below 8 names; idiosyncratic variance dominates below
```

### 15.5 Benchmark Comparison — Entry/Exit Profiles (v12.0)

Comparative table from April 2026 research notes, used to calibrate
`01_Scaling_Rules.py`:

| System Style | Entry / Exit | Outcome |
|--------------|--------------|---------|
| Retail / Basic Quant | 70 / 65 | High churn. Shaken out by 15m noise during healthy 1h pullbacks. |
| **Institutional Swing** | **80 / 60** | **High efficiency. Elite setups. Maximizes Sharpe. ← our target** |
| Mean Reversion | 90 / 50 | Extreme conviction. Low frequency, high win rate, fat tails. |

**Decision**: Institutional Swing profile, `FLOOR=0.60`, `HYSTERESIS_CLEAR=0.80`.
60-80% is the dead zone between exit and re-entry.

### 15.6 Timeframe Decay Weighting (v12.0)

Apply in meta-learner feature rollup:

- Weekly + Daily (Structural Anchor): **0.70** of composite weight.
- 1h + 15m (Precision Layer): **0.30** of composite weight.

**Why**: if 15m drops to zero during a flash dip, composite only falls
~0.10. Trade survives through noise. If daily or weekly break structure,
composite collapses past the 60% floor — clean exit.

### 15.7 Why This Architecture

- **Modular Recovery**: Change "10" to "5" in scaling_rules.py without
  touching the AI or data pipeline.
- **Pure Technical Alpha**: V07/V08 gating = aggressive in stable regimes,
  defensive when VIX spikes.
- **Zero-Waste Scaling**: Mid-point limits + minute-by-minute scaling =
  never guessing an exit. Mathematically dissolving positions as
  probability fades.

---

## ══════════════════════════════════════════════════════
## SECTION 16 — MSV (MACRO STATE VECTOR)
## ══════════════════════════════════════════════════════

NOT a filter — smoothly modulates god_score thresholds.

```python
MSV_COMPONENTS = {
    "VIX_norm":  "^VIX",
    "HY_norm":   "FRED BAMLH0A0HYM2 — ICE BofA HY OAS spread",
    "VIX_term":  "VIX9D / VIX ratio — inverted = panic",
    "DXY_norm":  "DX-Y.NYB — US Dollar Index",
    "BDI_norm":  "Quandl BCOM/BDI — Baltic Dry Index",
    "FBX_norm":  "Freightos FBX — global container freight rate",
}

stress_score = tanh(0.30*VIX_z + 0.20*HY_z + 0.15*VIX_term
                    + 0.15*DXY_z + 0.10*BDI_z + 0.10*FBX_z)
modulated_score = raw_god_score - (stress_score × 0.15)
```

### Fear & Greed Integration (v12.0 — deferred)

Research note from April 2026: incorporate CNN Fear & Greed Index as
contrarian MSV component. "Buy when fearful" — extreme fear readings
(<20) historically precede multi-week rallies. Extreme greed (>80)
precedes mean-reversion. Deferred to post-God-Model-v1 (AUG 2026+).

---

## ══════════════════════════════════════════════════════
## SECTION 17 — TRADING DYNAMICS
## ══════════════════════════════════════════════════════

### Triple Barrier

```python
def triple_barrier_label(close, atr, entry_idx, profit_mult=2.0, stop_mult=1.0, horizon=20):
    entry = close.iloc[entry_idx]
    tp = entry + profit_mult * atr.iloc[entry_idx]
    sl = entry - stop_mult * atr.iloc[entry_idx]
    for i in range(1, min(horizon+1, len(close)-entry_idx)):
        c = close.iloc[entry_idx+i]
        if c >= tp: return "strongly_bullish", i   # UPPER CHECKED FIRST
        if c <= sl: return "bearish", i
    return "neutral", horizon
```

Per-resolution horizons (as implemented in label_{res}.py):

- 15min: 80 bars
- 1hour: 40 bars (current first-run target)
- daily: 20 bars
- weekly: 12 bars

### Position Sizing

```python
MAX_POSITION_PCT = 0.20     # never >20% in one stock
MAX_DAILY_RISK   = 0.02     # never risk >2% of account per trade
# Kelly: f = (b×p - q) / b where b=2.0 (TP/SL ratio)
# Use quarter-Kelly (conservative)
```

### Almgren-Chriss Execution Cost

```python
# Cost ∝ position²/volume (NOT flat haircut)
MINIMUM_EV_RATIO = 2.0   # trade only if gross EV >= 2× execution cost
# Pre-earnings exit: ALWAYS exit 3 days before (spreads widen 2-4×)
```

### Scaling Rules Target Values (v12.0 — per-firm lookup)

Built into `03_Trading_Dynamics/01_Scaling_Rules.py` and
`02_Firm_Rules.py`:

```python
FIRM_RULES = {
    "GFT":          {"dd_daily": 0.04, "dd_total": 0.10, "drawdown_type": "EOD"},
    "FTMO":         {"dd_daily": 0.05, "dd_total": 0.10, "drawdown_type": "EOD"},
    "The5%ers":     {"dd_daily": 0.04, "dd_total": 0.10, "drawdown_type": "EOD"},
    "Apex":         {"dd_daily": None, "dd_total": 0.06, "drawdown_type": "trailing"},
    "Topstep":      {"dd_daily": 0.03, "dd_total": 0.06, "drawdown_type": "EOD"},
    "FundingPips":  {"dd_daily": 0.05, "dd_total": 0.10, "drawdown_type": "EOD",
                     "news_blackout_min": 5},
}
```

---

## ══════════════════════════════════════════════════════
## SECTION 18 — KEY EQUATIONS (COMPLETE)
## ══════════════════════════════════════════════════════

```python
# ─── T-1 Alignment (vectorized) ──────────────────────────────────────────
pos = np.searchsorted(child_ns, master_ns, side="right") - 1
result = np.where((pos >= 0) & (child_ns[pos] < master_ns), child_ns[pos], -1)

# ─── Noise Scale (Smith 2018) ────────────────────────────────────────────
g = epsilon × N / B       # optimal: 0.5-1.5

# ─── HRP Distance ────────────────────────────────────────────────────────
d[i,j] = sqrt(0.5 × (1 - rho[i,j]))

# ─── IC t-Statistic ──────────────────────────────────────────────────────
t = IC × sqrt(N) / sqrt(1 - IC²)

# ─── Net Return Multiplier ───────────────────────────────────────────────
NET = (1-0.003)(1-0.31)(1-0.18)(1-0.05) = 0.41

# ─── Wilder ATR ──────────────────────────────────────────────────────────
atr = tr.ewm(span=14, min_periods=14, adjust=False).mean()

# ─── Wilder RSI ──────────────────────────────────────────────────────────
ag = gains.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
al = losses.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
rsi = 100 - 100/(1 + ag/al)

# ─── MACD (standard EMA, NOT Wilder) ─────────────────────────────────────
ef = close.ewm(span=12, adjust=False).mean()
es = close.ewm(span=26, adjust=False).mean()
hist = (ef - es) - (ef - es).ewm(span=9, adjust=False).mean()

# ─── MOP Normalization ───────────────────────────────────────────────────
bull_score = clip(tanh(raw_signal / (k × atr)), 0, 1)

# ─── V08 Range Position ──────────────────────────────────────────────────
v6_score = clip((close - low_52w) / (high_52w - low_52w), 0, 1)
# LOOKAHEAD GUARD: rolling H/L computed on close.shift(1)

# ─── Parkinson Volatility ────────────────────────────────────────────────
sigma = sqrt(rolling_mean(ln(H/L)^2) / (4 * ln(2)))

# ─── DHS Derivative ──────────────────────────────────────────────────────
hist_delta_norm = tanh((histogram[t] - histogram[t-1]) / atr)

# ─── J&T Momentum ────────────────────────────────────────────────────────
mom_daily  = close.shift(21) / close.shift(252) - 1   # skip 1m, form 12m
mom_weekly = close.shift(4) / close.shift(52) - 1

# ─── Kelly Position Size ─────────────────────────────────────────────────
kelly = (2.0 × god_score - (1 - god_score)) / 2.0
position_pct = min(kelly × 0.25, 0.20)

# ─── MSV Stress Modulation ───────────────────────────────────────────────
modulated_god_score = raw_god_score - (stress × 0.15)

# ─── Almgren-Chriss ──────────────────────────────────────────────────────
eta  = (bid_ask/2) / (0.01 × daily_volume)
cost = (bid_ask/2)×shares×price + eta×(shares/T)²×T×price + 0.5×(eta×0.1)×shares²×price

# ─── BLL Bootstrap ───────────────────────────────────────────────────────
null_ics = [spearmanr(shuffle(signal), fwd_return) for _ in range(1000)]
p_value  = mean(null_ics >= observed_ic)   # threshold: p < 0.05

# ─── Technical: Scaling Rules (UPDATED v12.0) ────────────────────────────
position_size = (confidence - FLOOR) / (1.0 - FLOOR) × max_allocation
# confidence < 0.60 → exit. 0.60–0.80 hysteresis dead zone (institutional-swing).

# ─── Markowitz & Xu (1994) DMC — expected OOS excess return ──────────────
oos_expected_excess = theta × backtested_excess
# theta ∈ [0, 1]; Markowitz DPOS θ≈0.59, Guerard MQ θ≈0.47.
# Use θ=0.50 as conservative default for business-plan floor scenarios.
```

---

## ══════════════════════════════════════════════════════
## SECTION 19 — TECHNICAL GOD SOVEREIGN WEALTH ROADMAP
## ══════════════════════════════════════════════════════

### 19.1 Core Account Unit: GFT 2-Step $150K

```
Evaluation Fee:    ~$364 (target 50% discount windows)
Phase 1 Target:    8% ($12,000)
Phase 2 Target:    5% ($7,500)
Daily Drawdown:    4% ($6,000)
Total Drawdown:    10% ($15,000)
Profit Split:      80/20 (trader keeps 80%)
Payout Cycle:      Bi-weekly
```

### 19.2 The 'Sovereign' Growth Protocol

**Model Yield Target**: 1% asset gain bi-weekly across an uncorrelated
10-asset fleet. This is the God Model's steady-state extraction rate.

**Plateau Strategy**: Once an account hits $165,000 (10% buffer above
the $150K base), initiate **Harvest Mode**. All subsequent profits are
withdrawn to the Personal Fund. The account never grows beyond $165K —
it becomes a perpetual cash-flow machine.

**The 'Infinite Return' Logic**: The $364 evaluation fee is a one-time
admission ticket for perpetual AUM access. Risk is capped at the fee +
drawdown rules. Upside is systemic and ongoing. Each passed account
generates ~$154K/year in prop net for a $364 entry cost.

### 19.3 Leverage Tapering (The Shield)

```
$0 – $400K AUM:    3× Leverage  (early acceleration, small capital at risk)
$400K – $4M AUM:   2× Leverage  (growth phase, moderate risk)
$4M+ AUM:          1× Leverage  (Indestructibility — zero leverage risk)
```

At 1× leverage with $12M AUM, the fleet is mathematically indestructible
against any single-day event. Position sizing becomes pure
confidence-weighted allocation with no borrowed exposure.

### 19.4 The Definitive 5-Year Projection (Post-Tax @ 20%)

This is the North Star. All progress tracking references this table.

| Month | AUM | Leverage | Prop Net (Payout) | Personal Gain | Total Monthly | Cumulative Wealth |
|-------|-----|----------|-------------------|---------------|---------------|-------------------|
| 1 | $0.2M | 3× | $4,677 | $0 | $4,677 | $4,677 |
| 12 | $0.6M | 2× | $9,308 | $1,841 | $11,149 | $102,733 |
| 24 | $12.0M | 1× | $154,368 | $22,971 | $177,339 | $1,320,168 |
| 36 | $12.0M | 1× | $154,368 | $70,805 | $225,173 | $3,747,825 |
| 48 | $12.0M | 1× | $154,368 | $131,542 | $285,910 | $6,830,306 |
| 60 | $12.0M | 1× | $154,368 | $208,662 | $363,030 | $10,744,238 |

**Key inflection points**:

- Month 12: Fleet reaches $600K AUM, leverage drops to 2×.
- Month 24: Full $12M AUM deployed across 6 firms, leverage = 1×.
- Month 36: Cumulative wealth passes $3.7M (812 GTS eligible).
- Month 48: Personal Fund gains ($131K/mo) approach prop net ($154K/mo).
- Month 60: $10.7M cumulative. Personal Fund alone generates $208K/mo.

**Post Month 60**: Personal Fund at ~$10M+ generates more than prop
fleet. Begin winding down prop accounts. Invest personal capital only.

### 19.5 Markowitz-Adjusted Pessimistic Floor (v12.0)

Applying DMC θ-shrinkage to the above projections:

| Month | Optimistic (above) | θ=0.50 Floor | θ=0.30 Worst-Case |
|-------|-------------------|---------------|-------------------|
| 12 | $102K | $51K | $31K |
| 24 | $1.32M | $660K | $396K |
| 36 | $3.75M | $1.87M | $1.12M |
| 48 | $6.83M | $3.42M | $2.05M |
| 60 | $10.74M | $5.37M | $3.22M |

Even under θ=0.30 (pessimistic: 70% of backtested alpha evaporates out of
sample), the 5-year cumulative still exceeds $3M, which is the GTS-eligible
threshold. Business plan remains viable under Markowitz shrinkage.

### 19.6 The Dual Engine

```
ENGINE 1 — PROP FLEET (Capped at $12M AUM)
  80 × $150K accounts across 6 firms = $12M theoretical max
  Plateau at $165K per account → harvest all excess
  Prop Net: ~$154K/month at steady state (post-tax @ 20%)

ENGINE 2 — PERSONAL FUND (Uncapped, 0× leverage)
  100% of post-tax payouts reinvested
  Same God Model, same 1% bi-weekly yield
  Compounds freely: $0 → $208K/mo by Month 60
  This is the true wealth engine — prop fleet is the bootstrap
```

---

## ══════════════════════════════════════════════════════
## SECTION 20 — BUSINESS PLAN (SOVEREIGN EXECUTION)
## ══════════════════════════════════════════════════════

### Phase 1: Foundation & First Account (Months 1–4)

```
Objective: Pass ONE GFT 2-Step evaluation. Validate live execution.
  - Deploy Stellar-Rasberry Technical with trained weights
  - Target 50% discount window for $364 → ~$182 entry
  - Phase 1: 8% in evaluation period (~2-4 weeks with God Model)
  - Phase 2: 5% in verification (~2-3 weeks)
  - First funded payout validates the entire pipeline
  - NO paper trading — backtesting/training IS the validation

Capital at risk: $364 (one evaluation fee)
Monthly net by Month 4: ~$4,677
```

### Phase 2: Fleet Scaling (Months 5–24)

```
Year 1 cadence: Add ONE new $150K account every 4 months.
  - Month 1: 1 account ($150K) → 3× leverage
  - Month 5: 2 accounts ($300K) → 3× leverage
  - Month 9: 3 accounts ($450K) → leverage drops to 2×
  - Month 13: 4 accounts ($600K) → 2× leverage

Year 2: Aggressively scale to 6 DIFFERENT firms.
  - Target firms: GFT, FTMO, The5%ers, Apex, Topstep, Funding Pips
  - Each firm: multiple $150K accounts (within firm TOS)
  - Month 24 target: $12M AUM → leverage drops to 1× (Indestructibility)

Focus during this phase:
  - V01-V08 variable hardening (fix any IC decay observed in live)
  - Bayesian retraining every Saturday
  - correlation_logic.py tuning (10-asset sector diversification)
  - Kill-switch calibration against real market events
```

### Phase 3: Harvest Mode (Months 25–60)

```
AUM capped at $12M across all firms.
Every account plateaus at $165K (10% buffer).
All excess profit → Personal Fund (0× leverage, same model).

Prop Net:     ~$154K/month (stable, post-tax)
Personal Net: $23K/month (Month 24) → $209K/month (Month 60)
Combined:     $177K/month → $363K/month

Cumulative wealth trajectory:
  Month 36: $3.7M  → 812 GTS PURCHASE
  Month 48: $6.8M
  Month 60: $10.7M → Personal Fund dominates
```

### Phase 4: Personal Capital Only (Month 60+)

```
At ~$10M+ personal capital:
  $10M × 1% bi-weekly × 2 = ~$200K/month (0× leverage)
  This exceeds prop net of $154K/month

Wind down prop accounts. No more evaluation fees.
No more payout splits. No more drawdown rules.
Same God Model. Same 10-asset uncorrelated fleet.
Pure personal alpha extraction.
```

### Prop Firm Selection Criteria

```
MANDATORY:
  ✓ Allows overnight/swing positions
  ✓ EOD drawdown (not intraday-only trailing)
  ✓ 70-80%+ profit split
  ✓ Bi-weekly or faster payout cycle
  ✓ Proven payout history (community-verified)
  ✓ No news blackout restrictions OR manageable ones

TARGET FIRMS (evaluate at deployment time):
  GFT (primary — $150K 2-Step, $364 fee at 50% discount)
  FTMO (swing accounts, weekend holding)
  The5%ers (native scaling program)
  Apex Trader Funding (trailing drawdown, 30% consistency)
  Topstep (EOD drawdown, 50% cap on payouts)
  Funding Pips (news blackout — evaluate compatibility)
```

### Prop Firm Compliance Rules (NON-NEGOTIABLE)

```
✓ Multiple accounts across DIFFERENT firms = ACCEPTABLE
  (Each firm is a separate legal entity with separate TOS)
✗ Multiple accounts at the SAME firm beyond TOS limits = VIOLATION
✓ Virtual stops only (no broker-side stop orders visible to firms)
✓ Quantower bridge with 500-2000ms jitter for detection bypass
✓ Pre-earnings exit: 3 days before (spreads widen 2-4×)
✓ Respect each firm's specific drawdown rules (EOD vs trailing)
✓ Account plateau at $165K — never let accounts grow unchecked
```

### Anti-Correlation Rule (Portfolio Construction — UPDATED v12.0)

```
Maintain 10 assets across DISTINCT sectors at all times.
correlation_logic.py enforces (Markowitz-informed):
  - 20-period rolling correlation between all held positions
  - Max pairwise correlation: 0.70 (base threshold)
  - Greedy-add: sort candidates by meta-learner confidence,
    admit a name iff max pairwise |ρ| with already-held names < 0.70
  - Fallback: if greedy produces < 10 names, relax threshold to 0.80
  - Minimum: never drop below 8 names (idiosyncratic variance dominates)
  - If model loves 5 semiconductors, only the strongest 1-2 survives
  - Forces diversification into uncorrelated winners
  - This IS the portfolio-level risk management
```

### Transfer Coefficient Awareness (v12.0)

From Clarke, de Silva & Thorley (2002) via JLM (2005): long-only
constraint drops the transfer coefficient — the correlation between
active weights and forecast residual returns — from 0.98 to 0.58 at
5.0 TE. **Implication**: over-pruning the Top 15 → Top 10 correlation
filter can collapse the transfer coefficient and destroy the alpha
the CNN variables paid 800-ticker training to extract. The 0.70 → 0.80
fallback is deliberate — stay above 8 names.

### Kill-Switch Integration

```
03_Risk_Manager.py (within Trading Dynamics) must:
  - Flatten ALL positions across ALL firms if total fleet heat > 1.1% in 24H
  - Per-account monitoring against individual firm drawdown rules
  - Daily drawdown: 4% ($6,000 per $150K account)
  - Total drawdown: 10% ($15,000 per $150K account)
  - TIER_0 CATASTROPHIC: full cash in 60 seconds
  - TIER_1 SEVERE: tighten stops to 0.5×ATR, halt entries 24H
```

### No Paper Trading Rationale

```
Paper trading is redundant when:
  1. Walk-forward backtesting covers 2024+ out-of-sample data
  2. God Model validated on temporal test split (never seen in training)
  3. BLL bootstrap confirms signal ≠ random (p < 0.05)
  4. Harvey t-stat confirms signal survives multiple testing (t ≥ 2.0)
  5. Markowitz DMC θ-shrinkage calibrates realistic expectations
  6. Slow capital scaling (1 account for first 4 months) IS live validation

The first evaluation + 3 months on ONE funded account serves as the
live validation phase — real money, real execution, minimal capital risk.
Total downside: $364 evaluation fee.
```

---

## ══════════════════════════════════════════════════════
## SECTION 21 — RESEARCH PAPER LIBRARY (37 PAPERS)
## ══════════════════════════════════════════════════════

```
 1. Tan & Le (2019) ICML — EfficientNet compound scaling
 2. Lopez de Prado (2018) — Triple Barrier + Purged CV
 3. Lin et al. (2017) ICCV — Focal Loss γ=2.0
 4. Zhang et al. (2018) ICLR — Mixup α=0.3
 5. Guo et al. (2017) ICML — Temperature Scaling
 6. Lundberg & Lee (2017) NeurIPS — SHAP (qualitative audit only)
 7. Davis & Goadrich (2006) ICML — PR-AUC (correct under imbalance)
 8. Smith et al. (2018) ICLR — Noise Scale g=ε×N/B
 9. Goyal et al. (2017) CVPR — Linear Scaling Rule (sqrt for transfer)
10. He et al. (2019) CVPR — Bag of Tricks (zero gamma, no bias WD, LS)
11. Lopez de Prado (2016) JOPM — HRP
12. Chen & Guestrin (2016) KDD — XGBoost
13. Wolpert (1992) — Stacked Generalization (OOF only)
14. Almgren & Chriss (2000) JRI — Optimal Execution (cost ∝ pos²/vol)
15. Brogaard & Pan (2022) JF — Dark Pool (FINRA ATS, free data)
16. Jegadeesh & Titman (1993) JF — J&T 12-1 momentum
17. Harvey, Liu & Zhu (2016) RFS — Multiple testing (t≥3.0 full)
18. Brock, Lakonishok & LeBaron (1992) JF — BLL Bootstrap
19. Moskowitz, Ooi & Pedersen (2012) JFE — MOP Vol Normalization
20. Daniel, Hirshleifer & Subrahmanyam (1998) JF — DHS Derivative
21. George & Hwang (2004) JF 59(5) — PTH Anomaly
22. George, Hwang & Li (2018) JFE 128(1) — PTH q-Theory
23. Jeon & Byun (2023) FAJ 79(2) — Panic State
24. Büsing, Mohrschladt & Siedhoff (2021) — MOM=HTP+PTH decomposition
25. Huddart, Lang & Yetman (2009) ManSci 55(1) — 52w extremes
26. Grinblatt & Han (2005) JFE 78(2) — Capital gains overhang
27. Kahneman & Tversky (1979) Econometrica 47(2) — Prospect Theory
28. Baker, Pan & Wurgler (2012) JF 67(1) — 52w anchor in M&A
29. Whaley (2000) JPM 26(3) — VIX Fear Gauge
30. Parkinson (1980) JBus 53(1) — Range Variance Estimator
31. Schwert (1989) JF 44(5) — Vol Persistence
32. Ang, Hodrick, Xing & Zhang (2006) JF 61(1) — Vol Cross-Section
33. Moskowitz & Grinblatt (1999) JF 54(4) — Industry Momentum
34. Adrian & Brunnermeier (2016) AER 106(7) — Volatility Paradox
35. Markowitz & Xu (1994) JPM 21(1) — Data Mining Corrections (DMC, θ-shrinkage)  [NEW v12.0]
36. Jacobs, Levy & Markowitz (2005) OR 53(4) — Trimability + 130/30 + transfer coefficient  [NEW v12.0]
37. Farrell (1974) J.Bus. 47 — Homogeneous Stock Groupings via covariance clustering  [NEW v12.0]
```

### v12.0 Paper Integration Notes

**Markowitz & Xu (1994) — DMC θ-shrinkage**

Source paper: Markowitz, H. M. and Xu, G. L. 1994. "Data Mining
Corrections." *Journal of Portfolio Management*, 21(1), 60–69.

Key formula:
```
yit = μᵢ + zt + εᵢt     (Model III, general case)
θ ∈ [0, 1] = shrinkage coefficient from regressing best model's
              excess return toward grand average
Expected OOS excess return = θ × backtested excess
```

Empirical θ values from Markowitz Chapter 3 research:
- DPOS (Japan, 1974-1990): θ = 0.59
- Guerard MQ model: θ = 0.47
- Conservative default for Stellar-Rasberry: θ = 0.50

**Jacobs, Levy & Markowitz (2005) — Trimability + Transfer Coefficient**

Source paper: Jacobs, B. I., Levy, K. N., and Markowitz, H. M. 2005.
"Portfolio Optimization with Factors, Scenarios, and Realistic Short
Positions." *Operations Research* 53(4), 586–599.

Key concept for Stellar-Rasberry: factor-model decomposition converts
dense covariance matrix to near-diagonal. Inversion cheap. Transfer
coefficient (Clarke/de Silva/Thorley 2002) = correlation between active
weights and forecast residual returns. Long-only constraint drops it
from 0.98 → 0.58 at 5.0 TE. Informs `correlation_logic.py` greedy
filter design.

**Farrell (1974) — Homogeneous Stock Groupings**

Historical foundation for HRP. Uses correlation structure to cluster
equities into homogeneous risk groups. Relevant when thinking about
sector diversification and the bottom-up HRP weight construction.

---

## ══════════════════════════════════════════════════════
## SECTION 22 — AGENT ARCHITECTURE
## ══════════════════════════════════════════════════════

```
TRADING AGENTS (Research Vessel):
  Agent 1: Weekly Retrainer (Saturday Bayesian loop, quarterly HRP)
  Agent 2: Nightly Signal Generator (Mon-Fri after 16:30 EST)
  Agent 3: Position Monitor (every minute, market hours)
  Agent 4: Universe Scanner + Bear Filter (Sunday)
  Agent 5A: Data Quality Monitor (daily pre-market)
  Agent 5B: Kill Switch (5-min polling, build AUG-NOV 2026)

TRADING AGENTS (Technical Interceptor):
  Agent T1: repeat.py — 60s watchdog loop, crash recovery
  Agent T2: go.py — scan → infer → rank → filter → execute
  Agent T3: scaling_rules.py — floor/hysteresis/linear scaling
  Agent T4: broker_connection.py — mid-point limit orders

TRAINING AGENTS (v12.0 NEW):
  Agent T5: Training Progress Watcher
    - Runs on Mac Studio during long training sessions
    - Every 30 min, reads latest metrics_{res}.csv and checks:
      • val_loss NaN → email alert, halt training
      • val_loss > 3× best val_loss → divergence alert
      • train_acc − val_acc > 0.30 → overfitting alert
      • epoch_time_s > 3× median → thermal throttle alert
    - Does NOT halt training automatically — alerts only. Human decides.

BUSINESS AGENTS:
  Agent 6: Weekly Report Generator (Saturday PDF)
  Agent 7: Track Record Updater (daily → public dashboard)
  Agent 8: Subscriber Onboarding (Stripe → Discord → email)
  Agent 9: Alpha Tester (5-min IC monitor during market hours)
```

### Kill Switch Tiers (Build: AUG-NOV 2026)

```
TIER_0 CATASTROPHIC: FULL CASH in 60 seconds (2 sources, 0.92 confidence)
TIER_1 SEVERE:       Tighten stops to 0.5×ATR, halt entries 24H
TIER_2 VOLATILE:     Zero-weight affected sector for 48H
TIER_NEG1 NOISE:     Log only
```

---

## ══════════════════════════════════════════════════════
## SECTION 23 — BAYESIAN RETRAINING PROTOCOL
## ══════════════════════════════════════════════════════

Never cold-start. Expanding window with importance decay.

```python
BAYES_FACTOR_THRESHOLD = 3.0   # Jeffrey's scale: BF > 3.0 = "Substantial"
alpha_decay = 0.95             # per month. 24-month-old: 0.95^24 = 0.292
```

### Anchor Test (before any deployment)

```python
ANCHOR_SHARPE_FLOOR  = current_sharpe * 0.95
ANCHOR_DRAWDOWN_CEIL = current_drawdown * 1.10
ANCHOR_IC_CORR       = 0.85
```

### Saturday Loop

1. Alpha report → 2. Bayes Factor → 3. If BF>3: posterior weights →
4. Anchor test → 5. If pass: deploy → 6. If fail: keep prior, investigate.

### Checkpoint-Aware Resume (v12.0)

Weekly retrainer should:
- Load prior weights from `{res}/V1/model_{res}.pth` — not from scratch.
- Resume optimizer state from `{res}/V1/checkpoints/latest.pth` if
  training was interrupted mid-run.
- Preserve `metrics_{res}.csv` from prior runs by appending with an
  epoch offset or timestamp marker (append-mode, not overwrite).

---

## ══════════════════════════════════════════════════════
## SECTION 24 — HOW TO CREATE A NEW VARIABLE (v12.0 STRUCTURE)
## ══════════════════════════════════════════════════════

### Step 1: Directory Structure

```
XX_Variable_Name/
  ├── 15min/
  │   ├── generate_15min.py
  │   ├── label_15min.py          (signal vars only, NOT state)
  │   └── V1/
  │         ├── model_15min.pth
  │         ├── metrics_15min.csv
  │         └── checkpoints/
  ├── 1hour/
  │   ├── generate_1hour.py
  │   ├── label_1hour.py
  │   └── V1/
  ├── daily/ (same pattern)
  ├── weekly/ (same pattern)
  ├── monthly/ (CNN only — generate + V1 only, no label)
  ├── trainer.py                  (signal vars only, NOT state vars)
  ├── analyze_{name}.py           (signal vars only)
  ├── context_schema.json         (state vars only)
  └── plots/
```

### Step 2: Generator Boilerplate

```python
import importlib as _il
B = _il.import_module("01_base_modules_")

THIS   = Path(__file__).resolve().parent
OUT    = THIS / "signals"              # numerical: parquet
# OUT = THIS / "images"               # CNN: PNG

def load_ticker(data_dir, ticker):
    """Read from 00_ingest (IBKR) output. NEVER call yfinance or IBKR directly."""
    p = Path(data_dir) / "{resolution}" / f"{ticker}.parquet"
    ...
```

### Step 3: Labeler Boilerplate (v12.0 NEW — explicit per-resolution)

```python
# 01_Stock_Geometry/1hour/label_1hour.py is the reference
import importlib as _il
B = _il.import_module("01_base_modules_")

THIS          = Path(__file__).resolve().parent
OUT_CSV       = THIS / "labels_1hour.csv"
VERTICAL_BARS = 40                                 # per §1 constants
PROFIT_MULT   = 2.0
STOP_MULT     = 1.0

# Triple-barrier labeling per-ticker, parallelized with ProcessPoolExecutor(24)
# Output: date, ticker, label (0/1/2), split (train/val/test)
```

### Step 4: Trainer Checklist (Signal Variables V04-V06)

- [ ] Grid search over parameter combos.
- [ ] Harvey t-stat gating (≥2.0 partial, ≥3.0 full).
- [ ] BLL bootstrap (1000 permutations, p < 0.05).
- [ ] Spearman IC on train/val/test splits.
- [ ] `{res}/V1/selected_params_{resolution}.json` output.

### Step 5: Trainer Checklist (CNN Variables V01-V03)

- [ ] Correct backbone (ConvNeXt for V1, MobileNetV3 for V2-V3).
- [ ] Focal γ=2.0, Label Smoothing 0.1, Mixup α=0.3.
- [ ] Zero-gamma init (CNBlock.layer_scale or final BN).
- [ ] Guo (2017) temperature scaling.
- [ ] `--resolution` flag.
- [ ] Self-ref mask if applicable.
- [ ] VIX inversion if applicable.
- [ ] **Per-epoch checkpointing (v12.0)**: `latest.pth` every epoch,
      `ckpt_epoch_{NNN}.pth` every 5.
- [ ] **Per-epoch metrics CSV (v12.0)**: `metrics_{res}.csv` with 8 columns
      (epoch, train_loss, train_acc, val_loss, val_acc, val_top2_acc, lr,
      epoch_time_s).
- [ ] Outputs land in `{res}/V1/`, NOT at variable root.

### Step 6: Analyzer Validation (REQUIRED before God Model)

- [ ] P01: Train/Val/Test IC bar chart.
- [ ] P02: IC decay curve.
- [ ] P06: Rolling IC (252d) — no degradation train→val→test.
- [ ] P07: **Monotonic Staircase** in decile returns.
- [ ] **S07 (v12.0 CNN only)**: Training curves 6-panel from metrics CSV.
- [ ] All plots → `plots/` at variable root.

### Step 7: Registry Integration

Add tabular features to `TABULAR_COLUMNS` in `02_build_god_registry.py`.
Update interaction constraints in `03_trainer_v2.py`. Point
`MODEL_PATHS` to `{var}/{res}/V1/model_{res}.pth` (v12.0 path convention).

---

## ══════════════════════════════════════════════════════
## SECTION 25 — HARDWARE
## ══════════════════════════════════════════════════════

### MacBook Air M1 (legacy) — DO NOT TRAIN

Light scripting, planning, Claude Code, git.

### Mac Studio M3 Ultra 96GB (active as of April 2026)

```
CPU: 28-core | GPU: 60-core | Neural Engine: 32-core
RAM: 96GB unified | Bandwidth: 819 GB/s (50% faster than M4 Max)
SSD: 7.4 GB/s sequential read
MPS: device = torch.device("mps")
Workers: N_CORES = min(cpu_count(), 24)
DataLoader: num_workers=4 default (lowered from 8 to avoid CoW
            duplication of ChartDataset glob-set across workers)
Price: ~$3,780 (+ Samsung T9 4TB)
```

### Power Management

```bash
# REQUIRED for any overnight training session:
caffeinate -d &
# macOS is aggressive about power saving. Without caffeinate,
# training can be interrupted by system sleep.
# Kill the caffeinate process when training completes.
```

### IBKR Infrastructure (v12.0)

- TWS live port: 7496 (Arda's verified setup).
- TWS paper port: 7497.
- IB Gateway live: 4001.
- IB Gateway paper: 4002.
- Localhost-only: "Allow connections from localhost only" checked in
  Global Configuration.
- No Trusted IPs needed when localhost-only is active.
- `pip3 install ib_insync pandas pyarrow` — mandatory dependencies.
- Use `python3` and `pip3` explicitly on macOS.
- iCloud paths: wrap in quotes due to space in "Mobile Documents".

### Ethernet

Cat 6A recommended for 10GbE backbone. Used for:
- IBKR TWS API handshake (low-latency sensitivity).
- M3 Ultra ↔ local NAS for parquet staging.
- Future live execution: minimizes tail-latency in prop firm order flow.

---

## ══════════════════════════════════════════════════════
## SECTION 26 — THINGS NEVER TO DO
## ══════════════════════════════════════════════════════

```
TRAINING:
✗ DO NOT train on MacBook Air M1 (batch=8 → g=2.58 → catastrophe)
✗ DO NOT use random K-fold CV on time series (purged time series CV only)
✗ DO NOT apply zero-gamma init to ALL BN layers (LAST BN / CNBlock.layer_scale ONLY)
✗ DO NOT use horizontal flip on chart images (violates temporal causality)
✗ DO NOT use yfinance ANYWHERE (only 00_ingest_historical.py touches the network via IBKR)
✗ DO NOT use p.exists() in render loops (use build_image_set() glob cache)
✗ DO NOT keep the ChartDataset glob-set alive into DataLoader.fork time
✗ DO NOT raise num_workers above 4 without verifying RAM headroom under load
✗ DO NOT skip caffeinate -d for overnight runs

ARCHITECTURE:
✗ DO NOT compute cross-sectional features in 01_Variables (God Model only)
✗ DO NOT store training data as CSV > 10K rows (Parquet only)
✗ DO NOT change ATR_PERIOD=14 or VERTICAL_BARS without horizon-specific rationale
✗ DO NOT change NET_MULTIPLIER=0.41 without full business model review
✗ DO NOT use integer labels in display strings (stored labels are ints,
           display uses LABELS list)
✗ DO NOT use folder names "15m" or "1h" (must be "15min" and "1hour")
✗ DO NOT put V1 above timeframes — v12.0 structure is {var}/{res}/V1/
✗ DO NOT put self-ref mask in loss function only (must be in data path too)
✗ DO NOT confuse Wilder EWM (RSI/ATR: alpha=1/n) with standard EMA
           (MACD: alpha=2/(n+1))

DATA:
✗ DO NOT run IBKR backfill in parallel (sequential, 12s pacing — pacing
           is global per account)
✗ DO NOT try to fetch 15m bars from yfinance (60-day cap; will silently
           produce stale/wrong data)
✗ DO NOT submit to Polygon free tier for backfill (5 req/min, 2-year
           window; both insufficient)
✗ DO NOT bypass the resume check in 00_ingest_historical.py (completed
           tickers are skipped for a reason — re-downloading wastes
           pacing budget and risks account warnings)

BUSINESS:
✗ DO NOT set up multiple simultaneous prop firm accounts at the same
           firm beyond its TOS
✗ DO NOT accept subscriber funds before legal review (California)
✗ DO NOT manage client money without ERA/RIA registration
✗ DO NOT claim specific return guarantees
✗ DO NOT buy 812 GTS before $2M net worth
✗ DO NOT project business-plan returns at θ=1.0 — use θ=0.50 Markowitz
           shrinkage as the floor scenario

TIMING:
✗ DO NOT build Kill Switch before August 2026
✗ DO NOT launch before 6 months verified paper trading OR equivalent
           live-capital validation via Phase 1 prop firm evaluation

TECHNICAL INTERCEPTOR:
✗ DO NOT deploy Technical live before Research Vessel training completes
✗ DO NOT use more than ONE prop firm account simultaneously before
           validation is complete
✗ DO NOT skip the hysteresis dead zone (60%-80% institutional-swing)
✗ DO NOT execute market orders — always mid-point limit
✗ DO NOT let correlation filter drop below 8 names (idiosyncratic
           variance dominates; transfer coefficient collapses)
```

---

## ══════════════════════════════════════════════════════
## SECTION 27 — DEVELOPMENT TIMELINE
## ══════════════════════════════════════════════════════

```
COMPLETED (March-April 2026):
  CLAUDE.md v9.0 → v10.0 → v11.0 → v12.0 (April 18)
  Architecture pivot to 15min LCM master clock (v10)
  File structure pivot to timeframe-first, version-nested (v12)
  Data source pivot from yfinance/Polygon → IBKR TWS API (v12)
  00_ingest, 01_base_modules_, 02_registry, 03_trainer
  01_Stock_Geometry   7 files (5 res + trainer + analyzer)
                      trainer.py + analyze_cnn.py UPDATED April 18:
                        per-epoch checkpointing, metrics CSV, S07 plot
  02_Sector_Geometry  7 files (5 res + trainer + analyzer)
  03_SPY_Geometry     7 files (5 res + trainer + analyzer)
  04_RSI_Divergence   6 files (4 res + trainer + analyzer)
  05_MACD_Momentum    6 files (4 res + trainer + analyzer)
  06_Volume_Trend     6 files (4 res + trainer + analyzer)
  07_Volatility_Ctx   5 files (4 res + context_schema.json)
  08_Range_Context    5 files (4 res + context_schema.json)
  Stellar-Rasberry Technical architecture designed
  IBKR TWS API handshake verified (April 18, 2026)

APRIL 2026 (Mac Studio arrived — IMMINENT EXECUTION):
  [ ] First training run: V1 Stock Geometry 1hour (10-ticker validation)
  [ ] Full-universe 1hour training + metrics/checkpoints + S07 diagnostics
  [ ] Expand to other resolutions (15min, daily, weekly, monthly)
  [ ] V02-V03 CNN training (5 resolutions × 2 variables = 10 training runs)
  [ ] V04-V06 numerical trainer runs (parameter selection from grid search)
  [ ] End-to-end pipeline test with real 3M-point data

MAY 2026:
  Build master_joiner.py (merge_asof 3-channel tensor)
  Build Technical MGM (XGBoost with interaction constraints)
  Build macro_state_vector.py
  Begin Stellar-Rasberry Technical: variables/ wrappers
  Implement Markowitz DMC θ-shrinkage for meta-learner Sharpe

JUN-JUL 2026:
  God Model V1 training
  Trading Dynamics V1 (EV engine, position sizer, trailing stops)
  Technical: god_model/go.py + correlation_logic.py (greedy Markowitz filter)
  Technical: trading_dynamics/ (scaling, broker, repeat)
  Order Execution V1 (broker connector, IBKR live)

AUG-NOV 2026:
  FinBERT NLP + Kill Switch (Agent 5B)
  Alpha Tester V1 + Stellar Archive V1
  Technical Interceptor: live paper trading against prop firm sim

FEB-AUG 2027:
  6 months paper trading (public timestamps) — BOTH projects
  Bayesian retraining every weekend

AUGUST 2027 — LAUNCH (Research Vessel + Technical Interceptor)
  First GFT evaluation via Stellar-Rasberry Technical
  $364 entry → $150K funded account → $4,677/month prop net
```

---

## ══════════════════════════════════════════════════════
## SECTION 28 — IC TARGETS
## ══════════════════════════════════════════════════════

```
Variable                    IC Daily    IC Weekly   Half-Life  Harvey Min T
────────────────────────────────────────────────────────────────────────────
01 Stock Geometry (CNN)      0.04-0.08   0.05-0.10   3-5 yrs    2.0
02 Sector Geometry (CNN)     0.03-0.07   0.04-0.08   3-5 yrs    2.0
03 SPY Geometry (CNN)        0.04-0.08   0.05-0.10   3-5 yrs    2.0
04 RSI Divergence            0.03-0.07   0.05-0.09   3-4 yrs    2.0
05 MACD Momentum             0.04-0.08   0.05-0.10   3-4 yrs    2.0
06 Volume Trend              0.03-0.07   0.05-0.10   4-5 yrs    2.0
07 Volatility Context        N/A (state) N/A (state) N/A         N/A
08 Range Context             N/A (state) N/A (state) N/A         N/A
Dark Pool (future)           0.05-0.10   0.08-0.15   5-7 yrs    2.0
────────────────────────────────────────────────────────────────────────────
V2 trigger: IC < 60% of baseline for 2+ consecutive quarters
```

### Markowitz-Adjusted Realistic Floor (v12.0)

Apply θ=0.50 shrinkage to backtested IC for realistic live expectations:

```
Variable                    Backtested IC (target)    Expected Live (θ=0.50)
────────────────────────────────────────────────────────────────────────────
01 Stock Geometry (Daily)    0.04-0.08                 0.02-0.04
04 RSI Divergence (Daily)    0.03-0.07                 0.015-0.035
05 MACD Momentum (Daily)     0.04-0.08                 0.02-0.04
06 Volume Trend (Daily)      0.03-0.07                 0.015-0.035
God Model Sharpe             2.0 backtested            1.0 live expected
────────────────────────────────────────────────────────────────────────────
```

Individual variable ICs live at ~0.02-0.04 are still aggregate-useful
when combined via the 80-feature XGBoost meta-learner; Sharpe 1.0 on
an uncorrelated 10-asset fleet with 1.3× prop leverage still hits the
§19.4 Phase-1 monthly targets.

---

## ══════════════════════════════════════════════════════
## SECTION 29 — ALT DATA & FUTURE SOURCES
## ══════════════════════════════════════════════════════

### FINRA ATS Dark Pool (FREE — Start Collecting Now)

```
URL: finra.org/finra-data/alternative-trading-systems
Weekly, back to 2014. No API key.
signal = delta_dark_ratio (4-week change in dark %)
Bullish: +0.03 AND price flat (hidden accumulation)
```

### Dewey Data (Check Berkeley subscription)

```
ConsumerEdge, Advan Research, LinkUp, Context Analytics
TOS WARNING: academic license prohibits live trading.
Backtest only → purchase commercial license before live.
```

### dotdata Research

Father worked at dotdata. Search: Kondo & Uchiyama papers, DFLR.
Relevant for automated feature discovery on financial time series.

### Fear & Greed Index (deferred — v12.0 candidate)

CNN Fear & Greed Index composite: 7 indicators (momentum, strength,
breadth, put/call, junk bond demand, volatility, safe haven demand).
Daily updates since 2011. Candidate for MSV §16 integration once God
Model v1 is trained. Research hypothesis: "buy when fearful" —
readings <20 historically precede multi-week mean-reversion rallies.

### Claude as Summarization Layer (deferred — v12.0 candidate)

Use Anthropic's Claude API as an altdata summarization layer between
web-scraped sources (10-Ks, earnings transcripts, news feeds) and the
trading system. Candidate replacement / complement for FinBERT NLP
component of Agent 5B. Evaluate post-training of core V01-V08.

---

## ══════════════════════════════════════════════════════
## SECTION 30 — v12.0 CHANGELOG (April 18, 2026)
## ══════════════════════════════════════════════════════

This section records what changed from v11.0 → v12.0 for auditability.

### Code Changes (shipped)

**`00_Mini_Model/00_ingest_historical.py`** — full rewrite.
Replaced yfinance/Polygon firewall with IBKR TWS API via `ib_insync`.
Sequential fetching with 12s pacing. Resume-safe. Back to IPO date.
New CLI flags: `--host`, `--port`, `--client_id`. Default port 7496
(Arda's verified TWS live setup). Output schema unchanged (bytes-
identical parquet).

**`01_Stock_Geometry/trainer.py`** — two additions.

1. Per-epoch checkpointing:
   - `{out}/checkpoints/latest.pth` overwritten each epoch with full
     state (model, optimizer, scheduler, best_val_acc).
   - `{out}/checkpoints/ckpt_epoch_{NNN}.pth` every CKPT_EVERY=5
     epochs with model weights + val_acc.
2. Fine-grained metrics CSV:
   - `{out}/metrics_{res}.csv` written row-by-row per epoch.
   - 8 columns: epoch, train_loss, train_acc, val_loss, val_acc,
     val_top2_acc, lr, epoch_time_s.
   - Enables S07 training-curves plot in analyze_cnn.py.

Other changes:
- `ChartDataset.__init__` pre-filters label DataFrame by glob membership,
  then drops the set before DataLoader workers spawn (OOM fix).
- Date format dispatcher fixed for intraday resolutions
  (`YYYY-MM-DD_HHMM` for 15min/1hour, `YYYY-MM-DD` otherwise).
- `NUM_WORKERS` default lowered from 8 → 4. Exposed as CLI flag.
- `--output_dir` now required (was defaulted to `.`).

**`01_Stock_Geometry/analyze_cnn.py`** — new S07 section.
- `S07_training_curves_{res}.png` — 6-panel grid per resolution.
- Panels: loss, accuracy, generalization gap, LR schedule, epoch time,
  loss ratio.
- `_resolve_metrics_csv(res)` tries three fallback paths, prefers
  `{res}/V1/metrics_{res}.csv` (v12.0 canonical).

### Structural Changes (shipped — directory reorganization)

Version folder moved from parent-of-timeframes to child-of-timeframe.

Before (v11.0):
```
01_Stock_Geometry/
  V1/
    1hour/
      generate_1hour.py
    trainer.py
    analyze_cnn.py
```

After (v12.0):
```
01_Stock_Geometry/
  1hour/
    generate_1hour.py
    label_1hour.py
    V1/
      model_1hour.pth
      metrics_1hour.csv
      checkpoints/
  trainer.py
  analyze_cnn.py
```

### Documentation Changes (this document)

- Section 0: updated build status, added "first training run" block.
- Section 2: added §2.5 "First Training Run Recipe" with canonical
  terminal sequence.
- Section 3: replaced yfinance/Polygon firewall table with IBKR
  specifics.
- Section 4: rewrote file structure to reflect v12.0 hierarchy.
- Section 6: added checkpoint, metrics, and V1/1hour-specific
  subsections.
- Section 14: added Markowitz DMC θ-shrinkage block.
- Section 15: updated §15.4 HYSTERESIS_CLEAR 65 → 80; added §15.5
  benchmark table; added §15.6 timeframe decay weighting.
- Section 18: added DMC equation.
- Section 19: added §19.5 Markowitz-adjusted pessimistic floor table.
- Section 20: updated anti-correlation rule with greedy Markowitz
  filter; added transfer-coefficient awareness block.
- Section 21: expanded from 34 → 37 papers (Markowitz & Xu 1994, JLM
  2005, Farrell 1974); added v12.0 paper integration notes.
- Section 22: added Agent T5 Training Progress Watcher.
- Section 23: added §"Checkpoint-Aware Resume" block.
- Section 24: rewrote "How to Create a New Variable" for v12.0 paths.
- Section 25: added IBKR infrastructure subsection.
- Section 26: expanded "Never To Do" list with v12.0-specific items
  (IBKR backfill parallelism, old directory structure, θ=1.0 business
  projections).
- Section 27: updated timeline — April 2026 moved from future to
  present; "IMMINENT EXECUTION" block.
- Section 28: added Markowitz-adjusted realistic floor table.
- Section 29: added Fear & Greed and Claude-as-summarization candidates.
- Section 30 (THIS): new — changelog.

### Deferred Items (recorded for future CLAUDE.md updates)

- Markowitz DMC integration into V04-V06 trainers (after first successful
  training run establishes baseline IC).
- Fear & Greed Index integration into MSV §16.
- Claude as altdata summarization layer (FinBERT replacement
  consideration).
- correlation_logic.py implementation (currently designed, not built).
- 80/60 institutional-swing parameters live on Technical
  `01_Scaling_Rules.py` (currently in design).

### Arda Feedback Items — Status Tracker

| Item | Status | Where Implemented |
|------|--------|-------------------|
| Fear & Greed index | Deferred | §16 — post-training phase |
| Claude as altdata summarization | Deferred | §29 — FinBERT replacement candidate |
| 80/60 entry + 70/30 TF weighting | Documented | §15.5, §15.6 — implementation pending `01_Scaling_Rules.py` |
| Markowitz DMC for meta-learner | Documented | §14, §18, §19.5 — implementation post-first-training |
| Markowitz correlation for top-10 | Documented | §20 anti-correlation rule (greedy filter) — pending `correlation_logic.py` |
| Checkpoints every 5–10 epochs | **DONE** | `trainer.py` CKPT_EVERY=5, latest.pth every epoch |
| Fine-grained training metrics | **DONE** | `trainer.py` → `metrics_{res}.csv`, `analyze_cnn.py` → S07 |
| caffeinate -d | Documented | §25 Hardware power management |
| Normalization check | Already correct | ImageNet stats in ChartDataset.transform, unchanged |
| Batch size / MPS | Already correct | BS=256, device='mps', unchanged |
| IBKR API for data | **DONE** | `00_ingest_historical.py` full rewrite |
| File structure: version under timeframe | **DONE** | Directory reorganization April 18, 2026 |

---

## ══════════════════════════════════════════════════════
## END OF CLAUDE.md — VERSION 12.0
## Builder: Arda Turgut | UC Berkeley | April 18, 2026
## Architecture: 15-Minute LCM Master Clock (v3.0)
## Data Source: IBKR TWS API via ib_insync (port 7496 verified)
## File Structure: {var}/{resolution}/{V1|V2|V3}/ (v12.0)
## Companion: Stellar-Rasberry Technical (Live Execution)
## Stellar-Rasberry Sovereign Extraction Engine
## 46 production files | 37 research papers
## V01-V08 variable code COMPLETE — first training run April 2026
## trainer.py + analyze_cnn.py updated April 18, 2026:
##   per-epoch checkpointing + metrics CSV + S07 training-curves plot
## TWS API handshake successful — live trading node authorized
## ══════════════════════════════════════════════════════