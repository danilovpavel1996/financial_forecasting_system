# CLAUDE.md — Working agreement for this repo

You are building a **daily financial return-forecasting research system**. Read
`PROJECT_PLAN.md` for the full phased build plan. This file holds the rules you
must follow on **every** session, in every phase. These are non-negotiable.

## The one-paragraph mission
Forecast **forward returns** (never price levels) of a basket of assets
(precious metals first), at a **daily** horizon, inside a **leakage-free,
cost-aware, walk-forward evaluation harness**. The harness is the product. The
models are swappable parts. A model only "works" if it beats the random-walk
and drift baselines out-of-sample, after transaction costs, with stable IC.

## Non-negotiable principles (violating any of these is a bug)

1. **Targets are forward returns, never price levels.** Predicting price level
   yields a fake R² of ~0.99 from autocorrelation and is worthless. Always
   transform to forward log returns.

2. **No look-ahead, ever.** Every feature at time `t` must use ONLY data
   available at or before `t`. Rolling stats, normalization, and any fitted
   transform (scalers, PCA) are fit on TRAIN ONLY and applied forward. If you
   are unsure whether a step leaks, assume it does and write a test.

3. **Walk-forward with embargo.** Train always precedes test in time. There is
   an embargo gap of at least `horizon` samples between train end and test
   start so labels straddling the boundary cannot leak future prices. There is
   a unit test that asserts this.

4. **Baselines are first-class.** Every model comparison includes RandomWalk
   (predict 0) and Drift (predict trailing mean). A model that does not clearly
   beat these is reported as "no edge" — do not bury it.

5. **Costs are always charged.** Strategy P&L is net of transaction costs on
   turnover. Report turnover alongside Sharpe. A high-IC model that churns is
   not a win.

6. **Honest reporting over impressive numbers.** If results are marginal or
   negative, say so plainly in the run report. Do NOT tune until the backtest
   looks good — that is overfitting the test set. Suspiciously high Sharpe
   (> ~2.5 on daily single-asset) is a red flag to investigate for leakage,
   not celebrate.

7. **Config-driven, reproducible.** No hard-coded tickers, dates, or paths in
   logic. Everything goes through `config/config.yaml`. Set random seeds.

8. **This is research tooling, not investment advice.** Code and reports must
   not phrase outputs as recommendations to buy/sell.

## Workflow rules
- Work **one phase at a time** (see PROJECT_PLAN.md). Do not jump ahead.
- At the end of each phase, STOP and write a short `outputs/reports/phaseN_summary.md`
  stating what was built, the acceptance-criteria results, and anything ambiguous
  you had to decide. The human reviews this before you continue.
- Write tests as you go. The leakage tests (`tests/test_splitter.py`) are the
  most important code in the repo — treat them as such.
- Prefer small, reviewable commits with clear messages.
- If a phase's acceptance criteria cannot be met, do not fake it — document the
  blocker in the phase summary and stop.

## Tech stack
Python 3.11+. Core: numpy, pandas, scipy, scikit-learn. Data: yfinance, fredapi
(or pandas-datareader). Models: lightgbm; arch (for GARCH later). Viz:
matplotlib. Config: pyyaml. Tests: pytest. Keep dependencies minimal and pinned.

## Secrets
The FRED API key lives in a local `.env` (gitignored), read via env var
`FRED_API_KEY`. Never commit keys. Never print them.
