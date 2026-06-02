"""Interactive Brokers ticker mapping.

Maps yfinance tickers (used for data fetching) to IB instrument details
(used for display and contract-size calculations).

Fields per instrument:
  ib         — IB symbol / currency pair string
  name       — human-readable name
  exchange   — primary IB exchange
  multiplier — contract multiplier (oz, bbl, bu, shares, or 1 for forex)
  unit       — unit of the multiplier
  micro      — micro-contract IB symbol (None if not available)
  micro_mult — micro-contract multiplier (None if not available)
"""
from __future__ import annotations

YFINANCE_TO_IB: dict[str, dict] = {
    # ── Commodity futures ─────────────────────────────────────────────────────
    "GC=F": {
        "ib": "GC", "name": "Gold", "exchange": "COMEX",
        "multiplier": 100, "unit": "oz",
        "micro": "MGC", "micro_mult": 10,
    },
    "SI=F": {
        "ib": "SI", "name": "Silver", "exchange": "COMEX",
        "multiplier": 5000, "unit": "oz",
        "micro": None, "micro_mult": None,
    },
    "PL=F": {
        "ib": "PL", "name": "Platinum", "exchange": "NYMEX",
        "multiplier": 50, "unit": "oz",
        "micro": None, "micro_mult": None,
    },
    "PA=F": {
        "ib": "PA", "name": "Palladium", "exchange": "NYMEX",
        "multiplier": 100, "unit": "oz",
        "micro": None, "micro_mult": None,
    },
    "CL=F": {
        "ib": "CL", "name": "Crude Oil", "exchange": "NYMEX",
        "multiplier": 1000, "unit": "bbl",
        "micro": "MCL", "micro_mult": 100,
    },
    "NG=F": {
        "ib": "NG", "name": "Natural Gas", "exchange": "NYMEX",
        "multiplier": 10000, "unit": "mmBtu",
        "micro": None, "micro_mult": None,
    },
    "HG=F": {
        "ib": "HG", "name": "Copper", "exchange": "COMEX",
        "multiplier": 25000, "unit": "lbs",
        "micro": None, "micro_mult": None,
    },
    "ZC=F": {
        "ib": "ZC", "name": "Corn", "exchange": "CBOT",
        "multiplier": 5000, "unit": "bu",
        "micro": None, "micro_mult": None,
    },
    "ZS=F": {
        "ib": "ZS", "name": "Soybeans", "exchange": "CBOT",
        "multiplier": 5000, "unit": "bu",
        "micro": None, "micro_mult": None,
    },
    # ── Forex ─────────────────────────────────────────────────────────────────
    "EURUSD=X": {
        "ib": "EUR.USD", "name": "EUR/USD", "exchange": "IDEALPRO",
        "multiplier": 1, "unit": "units",
        "micro": None, "micro_mult": None,
    },
    "GBPUSD=X": {
        "ib": "GBP.USD", "name": "GBP/USD", "exchange": "IDEALPRO",
        "multiplier": 1, "unit": "units",
        "micro": None, "micro_mult": None,
    },
    "USDJPY=X": {
        "ib": "USD.JPY", "name": "USD/JPY", "exchange": "IDEALPRO",
        "multiplier": 1, "unit": "units",
        "micro": None, "micro_mult": None,
    },
    "AUDUSD=X": {
        "ib": "AUD.USD", "name": "AUD/USD", "exchange": "IDEALPRO",
        "multiplier": 1, "unit": "units",
        "micro": None, "micro_mult": None,
    },
    "USDCAD=X": {
        "ib": "USD.CAD", "name": "USD/CAD", "exchange": "IDEALPRO",
        "multiplier": 1, "unit": "units",
        "micro": None, "micro_mult": None,
    },
    "USDCHF=X": {
        "ib": "USD.CHF", "name": "USD/CHF", "exchange": "IDEALPRO",
        "multiplier": 1, "unit": "units",
        "micro": None, "micro_mult": None,
    },
    "NZDUSD=X": {
        "ib": "NZD.USD", "name": "NZD/USD", "exchange": "IDEALPRO",
        "multiplier": 1, "unit": "units",
        "micro": None, "micro_mult": None,
    },
    # ── Forex cross pairs (Phase 20) ──────────────────────────────────────────
    "EURJPY=X": {
        "ib": "EUR.JPY", "name": "EUR/JPY", "exchange": "IDEALPRO",
        "multiplier": 1, "unit": "units",
        "micro": None, "micro_mult": None,
    },
    "GBPJPY=X": {
        "ib": "GBP.JPY", "name": "GBP/JPY", "exchange": "IDEALPRO",
        "multiplier": 1, "unit": "units",
        "micro": None, "micro_mult": None,
    },
    "EURGBP=X": {
        "ib": "EUR.GBP", "name": "EUR/GBP", "exchange": "IDEALPRO",
        "multiplier": 1, "unit": "units",
        "micro": None, "micro_mult": None,
    },
    "AUDJPY=X": {
        "ib": "AUD.JPY", "name": "AUD/JPY", "exchange": "IDEALPRO",
        "multiplier": 1, "unit": "units",
        "micro": None, "micro_mult": None,
    },
    "EURAUD=X": {
        "ib": "EUR.AUD", "name": "EUR/AUD", "exchange": "IDEALPRO",
        "multiplier": 1, "unit": "units",
        "micro": None, "micro_mult": None,
    },
    "GBPAUD=X": {
        "ib": "GBP.AUD", "name": "GBP/AUD", "exchange": "IDEALPRO",
        "multiplier": 1, "unit": "units",
        "micro": None, "micro_mult": None,
    },
    "AUDNZD=X": {
        "ib": "AUD.NZD", "name": "AUD/NZD", "exchange": "IDEALPRO",
        "multiplier": 1, "unit": "units",
        "micro": None, "micro_mult": None,
    },
    "CADJPY=X": {
        "ib": "CAD.JPY", "name": "CAD/JPY", "exchange": "IDEALPRO",
        "multiplier": 1, "unit": "units",
        "micro": None, "micro_mult": None,
    },
    # ── Crypto (Phase 20) — trade via Binance/Kraken; IB crypto shown as reference ──
    # IB offers crypto via PAXOS/IBKR Crypto; for small capital use exchange directly.
    # multiplier=1, fractional=True: position = allocation / price (no integer rounding).
    "BTC-USD": {
        "ib": "BTC", "name": "Bitcoin", "exchange": "PAXOS",
        "multiplier": 1, "unit": "BTC", "fractional": True,
        "micro": None, "micro_mult": None,
        "alt_exchange": "Binance/Kraken (recommended for small capital)",
    },
    "ETH-USD": {
        "ib": "ETH", "name": "Ethereum", "exchange": "PAXOS",
        "multiplier": 1, "unit": "ETH", "fractional": True,
        "micro": None, "micro_mult": None,
        "alt_exchange": "Binance/Kraken",
    },
    "XRP-USD": {
        "ib": "XRP", "name": "Ripple", "exchange": "PAXOS",
        "multiplier": 1, "unit": "XRP", "fractional": True,
        "micro": None, "micro_mult": None,
        "alt_exchange": "Binance/Kraken",
    },
    "LTC-USD": {
        "ib": "LTC", "name": "Litecoin", "exchange": "PAXOS",
        "multiplier": 1, "unit": "LTC", "fractional": True,
        "micro": None, "micro_mult": None,
        "alt_exchange": "Binance/Kraken",
    },
    "ADA-USD": {
        "ib": "ADA", "name": "Cardano", "exchange": "PAXOS",
        "multiplier": 1, "unit": "ADA", "fractional": True,
        "micro": None, "micro_mult": None,
        "alt_exchange": "Binance/Kraken",
    },
    "LINK-USD": {
        "ib": "LINK", "name": "Chainlink", "exchange": "PAXOS",
        "multiplier": 1, "unit": "LINK", "fractional": True,
        "micro": None, "micro_mult": None,
        "alt_exchange": "Binance/Kraken",
    },
    "DOGE-USD": {
        "ib": "DOGE", "name": "Dogecoin", "exchange": "PAXOS",
        "multiplier": 1, "unit": "DOGE", "fractional": True,
        "micro": None, "micro_mult": None,
        "alt_exchange": "Binance/Kraken",
    },
    "SOL-USD": {
        "ib": "SOL", "name": "Solana", "exchange": "PAXOS",
        "multiplier": 1, "unit": "SOL", "fractional": True,
        "micro": None, "micro_mult": None,
        "alt_exchange": "Binance/Kraken",
    },
    "AVAX-USD": {
        "ib": "AVAX", "name": "Avalanche", "exchange": "PAXOS",
        "multiplier": 1, "unit": "AVAX", "fractional": True,
        "micro": None, "micro_mult": None,
        "alt_exchange": "Binance/Kraken",
    },
    "DOT-USD": {
        "ib": "DOT", "name": "Polkadot", "exchange": "PAXOS",
        "multiplier": 1, "unit": "DOT", "fractional": True,
        "micro": None, "micro_mult": None,
        "alt_exchange": "Binance/Kraken",
    },
    # ── Equity sector ETFs ────────────────────────────────────────────────────
    "XLB": {
        "ib": "XLB", "name": "Materials", "exchange": "ARCA",
        "multiplier": 1, "unit": "shares",
        "micro": None, "micro_mult": None,
    },
    "XLE": {
        "ib": "XLE", "name": "Energy", "exchange": "ARCA",
        "multiplier": 1, "unit": "shares",
        "micro": None, "micro_mult": None,
    },
    "XLF": {
        "ib": "XLF", "name": "Financials", "exchange": "ARCA",
        "multiplier": 1, "unit": "shares",
        "micro": None, "micro_mult": None,
    },
    "XLI": {
        "ib": "XLI", "name": "Industrials", "exchange": "ARCA",
        "multiplier": 1, "unit": "shares",
        "micro": None, "micro_mult": None,
    },
    "XLK": {
        "ib": "XLK", "name": "Technology", "exchange": "ARCA",
        "multiplier": 1, "unit": "shares",
        "micro": None, "micro_mult": None,
    },
    "XLP": {
        "ib": "XLP", "name": "Staples", "exchange": "ARCA",
        "multiplier": 1, "unit": "shares",
        "micro": None, "micro_mult": None,
    },
    "XLU": {
        "ib": "XLU", "name": "Utilities", "exchange": "ARCA",
        "multiplier": 1, "unit": "shares",
        "micro": None, "micro_mult": None,
    },
    "XLV": {
        "ib": "XLV", "name": "Healthcare", "exchange": "ARCA",
        "multiplier": 1, "unit": "shares",
        "micro": None, "micro_mult": None,
    },
    "XLY": {
        "ib": "XLY", "name": "Cons Disc", "exchange": "ARCA",
        "multiplier": 1, "unit": "shares",
        "micro": None, "micro_mult": None,
    },
}


def ib_symbol(yf_ticker: str) -> str:
    """Return IB symbol for a yfinance ticker, falling back to the ticker itself."""
    return YFINANCE_TO_IB.get(yf_ticker, {}).get("ib", yf_ticker)


def ib_exchange(yf_ticker: str) -> str:
    return YFINANCE_TO_IB.get(yf_ticker, {}).get("exchange", "—")


def contracts(yf_ticker: str, allocation: float, price: float) -> tuple[int | float, float, str | None]:
    """Compute contract/share count and notional per contract.

    For fractional assets (crypto), returns a float token quantity with a
    note about the exchange. For integer-lot instruments returns (int, notional, micro_note).

    Returns (n_contracts, notional_per_contract, micro_note).
    micro_note is a hint string if micro contracts exist or if exchange guidance applies.
    """
    info = YFINANCE_TO_IB.get(yf_ticker, {})
    mult = info.get("multiplier", 1)
    unit = info.get("unit", "units")

    # Fractional assets (crypto) — return token quantity, no min notional
    if info.get("fractional", False):
        if price <= 0:
            return 0.0, 0.0, None
        qty = allocation / price
        alt_exch = info.get("alt_exchange", "")
        note = None
        if alt_exch:
            note = (
                f"Fractional crypto: buy {qty:.6g} {unit} at ${price:,.4g}. "
                f"Trade on {alt_exch} (min ~$10). IB symbol: {info.get('ib', yf_ticker)} on {info.get('exchange', '—')}."
            )
        return qty, price, note

    notional = price * mult
    if notional <= 0:
        return 0, notional, None

    n = int(allocation // notional)

    micro_note = None
    if n == 0:
        micro = info.get("micro")
        micro_mult = info.get("micro_mult")
        if micro and micro_mult:
            micro_notional = price * micro_mult
            micro_n = int(allocation // micro_notional)
            micro_note = (
                f"Min notional: ${notional:,.0f} ({mult} {unit}). "
                f"Micro {micro} (×{micro_mult}): ${micro_notional:,.0f}/contract"
                + (f" → {micro_n} micro contract(s)" if micro_n > 0 else " → still < 1 micro contract")
            )
        else:
            micro_note = f"Min notional: ${notional:,.0f} ({mult} {unit}). No micro available."

    return n, notional, micro_note
