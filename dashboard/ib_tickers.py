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


def contracts(yf_ticker: str, allocation: float, price: float) -> tuple[int, float, str | None]:
    """Compute integer contract/share count and notional per contract.

    Returns (n_contracts, notional_per_contract, micro_note).
    micro_note is a hint string if micro contracts exist, else None.
    """
    info = YFINANCE_TO_IB.get(yf_ticker, {})
    mult = info.get("multiplier", 1)
    unit = info.get("unit", "units")

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
