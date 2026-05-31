# Phase 0 Summary — Scaffolding

**Date:** 2026-05-30
**Status:** COMPLETE — all acceptance criteria met

---

## What was built

### Directory tree
Full project skeleton per `PROJECT_PLAN.md §3`, including all stub files for phases 1–5:

```
financial_forecasting_system/
├── README.md
├── CLAUDE.md                  (pre-existing, not touched)
├── PROJECT_PLAN.md            (pre-existing, not touched)
├── requirements.txt           (pinned, Python 3.11)
├── pyproject.toml             (editable install via setuptools.build_meta)
├── .gitignore                 (.env, data/, outputs/models/, __pycache__, etc.)
├── .env.example               (FRED_API_KEY=your_key_here)
├── config/config.yaml         (universe, dates, horizon, costs, splitter params)
├── seed/eval_harness_reference.py   (moved from root; reference only — not imported)
├── src/
│   ├── __init__.py
│   ├── config.py              (load + validate config, fully implemented)
│   ├── data/                  (universe.py, prices.py, macro.py — stubs)
│   ├── features/              (price_features.py, macro_features.py — stubs)
│   ├── targets.py             (stub)
│   ├── eval/                  (splitter.py, metrics.py, backtester.py — stubs)
│   ├── models/                (base.py, baselines.py, linear.py, gbm.py — stubs)
│   └── pipeline.py            (stub)
├── scripts/                   (fetch_data.py, run_backtest.py — stubs)
├── outputs/figures/, reports/, models/
├── notebooks/
└── tests/
    ├── test_config.py         (7 smoke tests, all passing)
    ├── test_splitter.py       (stub, commented — Phase 2)
    ├── test_targets.py        (stub — Phase 4)
    └── test_metrics.py        (stub — Phase 2)
```

### `config/config.yaml` parameters
- **Universe:** metals (GC=F, SI=F, PL=F, PA=F), ETFs (GLD, SLV, GDX), macro context (SPY, TLT, UUP, ^VIX)
- **FRED series:** DFII10, T10YIE, DGS10, DTWEXBGS, VIXCLS
- **Dates:** 2010-01-01 to 2024-12-31
- **Horizon:** 1 trading day
- **Cost:** 5 bps round-trip
- **Splitter:** 8 folds, 3-year train, 1-year test, 5-day embargo (> horizon=1, leakage-safe)
- **Seed:** 42

### `src/config.py`
- Loads and validates `config.yaml` into typed dataclasses (`Config`, `SplitterConfig`, `PathsConfig`)
- Raises `ValueError` on any missing key or constraint violation
- Enforces `embargo >= horizon` at load time (this is the core leakage-prevention invariant)
- All paths resolved to absolute `Path` objects relative to repo root

### Python environment
- Python 3.11.13
- Virtualenv at `.venv/`
- All pinned dependencies installed successfully via `pip install -e .`

---

## Acceptance criteria results

| Criterion | Result |
|---|---|
| `pip install -e .` works | PASS |
| `python -c "from src.config import load_config; print(load_config())"` prints parsed config | PASS |
| `pytest` runs without import errors | PASS — 7 tests, 0 failures |

---

## Decisions made / ambiguities resolved

1. **pytest as dev dependency:** `pytest` is listed in `requirements.txt` but not in `pyproject.toml`'s runtime `dependencies` (it's test-only). Installed separately into the venv. This is intentional — `pip install -e .` installs runtime deps; `pip install pytest` installs test tools.

2. **`pyproject.toml` build backend:** Used `setuptools.build_meta` (standard) rather than the newer `setuptools.backends.legacy:build` alias, which is not available in the bundled pip's setuptools version.

3. **Config validation at load time:** Added `embargo >= horizon` check in `_validate_config`. This fires immediately if someone misconfigures the splitter, preventing a silent leakage bug from propagating into Phase 2+.

4. **No logic in stub files:** All Phase 1–5 stubs contain only a comment line. They import cleanly and will be replaced in their respective phases.

---

## What is uncertain / to watch for in Phase 1

- `yfinance` API stability: the pinned version (0.2.61) should be stable, but yfinance sometimes breaks on ticker symbol changes (e.g., futures roll conventions). The Phase 1 loader should log any download failures explicitly rather than silently returning empty DataFrames.
- FRED series lag: `DFII10` and `T10YIE` are released with a 1-day lag. A `TODO` reminder is planned in `macro.py` to note that point-in-time/vintage data would be the rigorous fix; for now, lag every macro feature by ≥ 1 day.
- `^VIX` via yfinance: spot VIX is available but has gaps. Phase 1 should log gap counts.

---

**Ready for Phase 1 review.**
