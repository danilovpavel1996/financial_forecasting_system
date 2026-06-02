# PHASE_20_PLAN.md — Forex Expansion + Crypto Universe + Cost Modeling

> Read CLAUDE.md first. All non-negotiable principles still apply.
> Focus: maximize Sharpe on assets tradeable with $1,000 or less.

---

## 1. Goal

Push the small-capital-accessible Sharpe above 1.0 (ideally 1.5) by:
A. Expanding the forex universe from 7 to 15+ pairs (larger cross-section)
B. Adding crypto as a new asset class (high vol, low fees, $10 minimum)
C. Modeling realistic costs per asset class (not a flat 5 bps for everything)

---

## 2. Part A: Forex Expansion

### Current: 7 major USD pairs
EURUSD=X, GBPUSD=X, USDJPY=X, AUDUSD=X, USDCAD=X, USDCHF=X, NZDUSD=X

### Expanded: add 8 cross pairs (15 total)
```yaml
forex:
  ranked_assets:
    # Majors (existing)
    - "EURUSD=X"
    - "GBPUSD=X"
    - "USDJPY=X"
    - "AUDUSD=X"
    - "USDCAD=X"
    - "USDCHF=X"
    - "NZDUSD=X"
    # Crosses (new)
    - "EURJPY=X"
    - "GBPJPY=X"
    - "EURGBP=X"
    - "AUDJPY=X"
    - "EURAUD=X"
    - "GBPAUD=X"
    - "AUDNZD=X"
    - "CADJPY=X"
```

### Why crosses help
- 15 assets give much better ranking stability than 7 (same effect as
  expanding commodities from 4 to 9)
- JPY crosses capture the carry trade dynamic (high-yielders vs JPY)
- Cross pairs add genuinely different return drivers (EUR vs GBP is
  about UK/EU divergence, not USD strength)
- With 15 assets: top-3 long, bottom-3 short (instead of top-2/bottom-2)

### Data availability
Most cross pairs have yfinance history back to 2003–2005. Verify on fetch.

---

## 3. Part B: Crypto Universe

### Universe: 10 major cryptocurrencies
```yaml
crypto:
  ranked_assets:
    - "BTC-USD"     # Bitcoin (since ~2014)
    - "ETH-USD"     # Ethereum (since ~2017)
    - "XRP-USD"     # Ripple (since ~2017)
    - "LTC-USD"     # Litecoin (since ~2014)
    - "ADA-USD"     # Cardano (since ~2017)
    - "LINK-USD"    # Chainlink (since ~2019)
    - "DOGE-USD"    # Dogecoin (since ~2019)
    - "SOL-USD"     # Solana (since ~2020)
    - "AVAX-USD"    # Avalanche (since ~2020)
    - "DOT-USD"     # Polkadot (since ~2020)
  context:
    - "^VIX"
    - "SPY"
    - "TLT"
  start_date: "2020-01-01"  # common start for 10 assets
  cost_bps: 20              # crypto: 0.1% per side = 20 bps round-trip
```

### Crypto-specific considerations

**Calendar:** crypto trades 24/7, including weekends. yfinance provides
daily data including Saturday/Sunday. For compatibility with the existing
pipeline (which uses pd.bdate_range for business days):
- **Filter crypto data to business days only (Mon-Fri).** Weekend returns
  get absorbed into Monday's open-to-close. This is slightly inexact but
  keeps the pipeline compatible without major refactoring.
- Do NOT use pd.bdate_range for the crypto reference calendar — use the
  filtered daily index directly.

**Costs:** crypto exchanges charge ~0.1% maker + 0.1% taker = 20 bps
round-trip (Binance, Kraken). This is 4× higher than forex. The model
needs to overcome this higher cost bar. Set cost_bps = 20 in the crypto
pipeline.

**Volatility:** crypto vol is 50–100% annualized (vs 5–15% for forex).
Higher vol means higher absolute returns for the same Sharpe, but also
higher drawdowns. Vol targeting is important here.

**History:** SOL, AVAX, DOT start around 2020. With start_date 2020-01-01,
we get ~5 years of data, which is thin for h=63 (only ~20 independent
periods). The h=5 signal will be more statistically robust for crypto.

---

## 4. Part C: Realistic Cost Modeling

### Per-universe cost parameters

The current system uses a flat `cost_bps = 5` for everything. This is
wrong — costs vary dramatically by asset class and broker.

Add `cost_bps` as a per-universe config parameter:

```yaml
forex:
  cost_bps: 3    # spread-only broker: ~1.0-1.5 pip = 1-2 bps/side

crypto:
  cost_bps: 20   # Binance/Kraken: 0.1% per side

commodities:
  cost_bps: 5    # futures: ~2-3 bps spread + commission

equity_sectors:
  cost_bps: 3    # ETF: ~1-2 bps spread + $0.005/share commission
```

Update ALL pipelines to read `cost_bps` from their respective universe
config instead of the global `cost_bps`.

### Impact on backtest results
Re-running the forex sweep with cost_bps=3 (instead of 5) should slightly
improve the forex Sharpe (less cost drag). Re-running crypto with
cost_bps=20 will be a harder bar to clear.

---

## 5. What to build

### 5a. Config updates
- Expand `forex.ranked_assets` to 15 pairs
- Add `crypto` section with 10 tokens + start_date + cost_bps
- Add `cost_bps` to each universe section

### 5b. Universe helpers (`src/data/universe.py`)
- Add `crypto_tickers(cfg)`, `crypto_context_tickers(cfg)`,
  `crypto_price_tickers(cfg)`, `crypto_start_date(cfg)`, `crypto_cost_bps(cfg)`
- Add `forex_cost_bps(cfg)` (returns 3)
- Add `universe_cost_bps(cfg, universe_name)` generic helper

### 5c. Crypto pipeline (`src/pipeline_ranking_crypto.py`)
Mirror the forex/sectors pipeline. Key differences:
- Filter crypto data to business days (drop weekends)
- Use `cost_bps=20` (from config)
- No COT, no carry_proxy, no late-close lag
- Skip macro features if they don't align (crypto trades on holidays
  when FRED doesn't publish — forward-fill handles this, same as forex)

### 5d. Sweep scripts
- `scripts/run_sweep_forex_expanded.py` — 15-pair forex sweep
- `scripts/run_sweep_crypto.py` — crypto sweep
- Both run: 3 horizons [5, 21, 63] × 3 models [MeanReversion, LightGBM,
  LambdaMART] + B3 PredAvg21 variant for LightGBM

### 5e. Cost-adjusted re-sweep for forex
Re-run the 15-pair forex sweep with cost_bps=3 (realistic for spread-only
broker) instead of the previous 5 bps.

### 5f. Multi-asset ensemble (if results are positive)
If crypto or expanded forex produces Sharpe > 0.5, build a small-capital
ensemble:
- Forex (15 pairs, cost_bps=3)
- Crypto (10 tokens, cost_bps=20)
- Weights: to be determined by sweep

### 5g. Dashboard updates
- Add "Crypto" to Live Signal universe selector
- Add "Forex (15 pairs)" as an option (distinct from old 7-pair)
- Show cost_bps per universe in the Signal Context panel
- Update Portfolio Simulator with crypto position sizing
  (crypto: position = allocation / price, fractional allowed,
  minimum ~$10 on most exchanges)

### 5h. IB ticker mapping for new assets
Add to `dashboard/ib_tickers.py`:
- New forex crosses (EURJPY, GBPJPY, etc.)
- Crypto tickers (note: IB uses "BTC" for Bitcoin futures,
  but for spot crypto the operator would use Binance/Kraken directly —
  show both the exchange ticker and the IB equivalent where applicable)

---

## 6. Execution order

1. Update config with expanded forex + crypto sections + per-universe cost_bps
2. Add universe helpers for crypto
3. Fetch all new data (8 new forex pairs + 10 crypto tokens)
4. Build crypto pipeline (mirror forex pipeline, filter to bdays)
5. Run expanded forex sweep (15 pairs, cost_bps=3)
6. Run crypto sweep (10 tokens, cost_bps=20)
7. Compare all results in one table
8. If any new signal beats forex h=5 Sharpe 0.94: build ensemble
9. Update Live Signal dashboard
10. Write phase20_summary.md with all sweep matrices

---

## 7. Definition of done

- [ ] 15 forex pairs fetched and verified
- [ ] 10 crypto tokens fetched and verified (filtered to bdays)
- [ ] Expanded forex sweep completed (cost_bps=3)
- [ ] Crypto sweep completed (cost_bps=20)
- [ ] All results compared in one summary table
- [ ] Per-universe cost_bps used in all pipelines
- [ ] If improved: ensemble built and compared
- [ ] Live Signal supports Crypto universe
- [ ] Dashboard shows cost_bps per universe
- [ ] All existing tests pass
- [ ] `phase20_summary.md` with complete results

---

## 8. What success looks like

- **Best case:** crypto h=5 Sharpe > 1.0 (high vol + mean-reversion
  in a herd-driven market). Combined forex+crypto small-capital
  ensemble Sharpe > 1.5.
- **Good case:** expanded forex (15 pairs) Sharpe improves from 0.94
  to > 1.0 due to larger cross-section + lower realistic costs.
- **Minimum:** crypto signal is positive (Sharpe > 0.3) after the
  higher 20 bps costs, confirming the approach works across asset
  classes.

---

*Research tooling only — not investment advice.*
