# Phase 20 Crypto Sweep Results (10 tokens, cost_bps=20)

*Format: Sharpe (net, non-overlapping) / CS-RIC (mean OOS)*

*Embargo = horizon for each run. All results are out-of-sample walk-forward.*

*Universe (10 tokens): BTC-USD, ETH-USD, XRP-USD, LTC-USD, ADA-USD, LINK-USD, DOGE-USD, SOL-USD, AVAX-USD, DOT-USD*

*Data filtered to business days (Mon-Fri) — weekend returns absorbed into Monday bar.*

*History start: 2020-01-01  |  End: 2024-12-31*

*Cost: 20 bps round-trip (Binance/Kraken ~0.1% per side)*

---

## Crypto Sweep Matrix

| Model | h=5 | h=21 | h=63 |
| --- | --- | --- | --- |
| MeanReversion | -0.18 / 0.0138 | -0.28 / -0.0508 | 0.26 / -0.0747 |
| LightGBM | -0.54 / -0.0502 | 0.01 / 0.0130 | -0.89 / -0.0449 |
| LambdaMART | -0.52 / 0.0329 | 0.22 / 0.0531 | 0.25 / 0.1997 |
| LightGBM PredAvg21 | -0.65 / -0.1240 | -0.54 / -0.0276 | -0.93 / -0.0774 |

---

## Turnover Matrix

| Model | h=5 | h=21 | h=63 |
| --- | --- | --- | --- |
| MeanReversion | 0.390 | 0.388 | 0.390 |
| LightGBM | 0.194 | 0.255 | 0.219 |
| LambdaMART | 0.361 | 0.228 | 0.197 |
| LightGBM PredAvg21 | 0.056 | 0.050 | 0.047 |

---

## Summary

**Crypto best:** **MeanReversion** at h=63 — Sharpe 0.26

**Forex reference (Phase 18):** LightGBM B3 h=5 — Sharpe 0.94

**Min viable threshold:** Sharpe > 0.3 after 20 bps costs (confirms cross-asset approach)


## Leakage flags

- LambdaMART h=63: CS-RIC=0.1997
---

## Notes

- Sharpe uses non-overlapping subsampling every `horizon` steps.
- CS-RIC is mean cross-sectional rank IC across all OOS test dates.
- Crypto has ~5 years of data (2020–2024). h=5 is most robust; h=63 has only ~20 independent periods.
- Costs: 20 bps round-trip is conservative (Binance Spot: 0.1% maker + 0.1% taker).
- With 10 tokens: top-3 long, bottom-3 short.
- High crypto vol (50-100% ann.) means high absolute P&L variance — interpret Sharpe carefully.
- Weekend data is dropped: Monday bars contain Fri-close to Mon-close move (Sat+Sun included).

*Research tooling only — not investment advice.*
