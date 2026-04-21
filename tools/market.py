"""
Market data tool -- fetches post-market closing data via yfinance.

Returns:
- Indian indices (Nifty, Sensex) with daily change
- Top Nifty 50 gainers/losers
- Sector indices (Bank, IT, Pharma, FMCG) with daily change
- India VIX (volatility/fear index)
- 52-week context (distance from high/low)
- Global context (S&P 500, Crude, Gold, USD/INR)
- FII/DII flow data (scraped from moneycontrol)
"""

import yfinance as yf
import requests
from langchain_core.tools import tool

from config import (
    INDIA_VIX_TICKER,
    INDIAN_INDICES,
    NIFTY_50_TICKERS,
    GLOBAL_TICKERS,
    SECTOR_INDICES,
    TOP_MOVERS_COUNT,
)


def _fetch_daily_change(ticker: str) -> dict | None:
    """Fetch today's closing price and % change for a single ticker."""
    try:
        # Use a slightly wider window to avoid "only 1 row" edge-cases
        # for some futures symbols (e.g., Brent) around market hours.
        hist = yf.Ticker(ticker).history(period="5d")
        if len(hist) < 2:
            return None
        prev_close = hist["Close"].iloc[-2]
        curr_close = hist["Close"].iloc[-1]
        pct_change = ((curr_close - prev_close) / prev_close) * 100
        return {
            "price": round(float(curr_close), 2),
            "change_pct": round(float(pct_change), 2),
        }
    except Exception:
        return None


def _fetch_indices() -> dict:
    """Fetch closing data for Nifty 50 and Sensex."""
    results = {}
    for name, ticker in INDIAN_INDICES.items():
        data = _fetch_daily_change(ticker)
        if data:
            results[name] = data
    return results


def _fetch_top_movers() -> dict:
    """Scan all Nifty 50 stocks, return top gainers and losers."""
    stock_changes = []
    for ticker in NIFTY_50_TICKERS:
        data = _fetch_daily_change(ticker)
        if data:
            stock_changes.append({"ticker": ticker, **data})

    stock_changes.sort(key=lambda x: x["change_pct"], reverse=True)

    green = sum(1 for s in stock_changes if s["change_pct"] > 0)
    red = len(stock_changes) - green

    return {
        "gainers": stock_changes[:TOP_MOVERS_COUNT],
        "losers": stock_changes[-TOP_MOVERS_COUNT:],
        "breadth": {"green": green, "red": red, "total": len(stock_changes)},
    }


def _fetch_sectors() -> dict:
    """Fetch closing data for sector indices (Bank, IT, Pharma, FMCG)."""
    results = {}
    for name, ticker in SECTOR_INDICES.items():
        data = _fetch_daily_change(ticker)
        if data:
            results[name] = data
    return results


def _fetch_vix() -> dict | None:
    """Fetch India VIX (fear/volatility index)."""
    try:
        hist = yf.Ticker(INDIA_VIX_TICKER).history(period="5d")
        if hist.empty:
            return None
        current = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else current
        change_pct = ((current - prev) / prev) * 100 if prev else 0
        avg_5d = float(hist["Close"].mean())
        return {
            "value": round(current, 2),
            "change_pct": round(change_pct, 2),
            "avg_5d": round(avg_5d, 2),
            "elevated": current > 20,
        }
    except Exception:
        return None


def _fetch_52week_context() -> dict | None:
    """Fetch Nifty 52-week high/low and distance from each."""
    try:
        hist = yf.Ticker("^NSEI").history(period="1y")
        if hist.empty:
            return None
        high_52w = float(hist["High"].max())
        low_52w = float(hist["Low"].min())
        current = float(hist["Close"].iloc[-1])
        return {
            "current": round(current, 2),
            "high_52w": round(high_52w, 2),
            "low_52w": round(low_52w, 2),
            "from_high_pct": round(((current - high_52w) / high_52w) * 100, 1),
            "from_low_pct": round(((current - low_52w) / low_52w) * 100, 1),
        }
    except Exception:
        return None


def _fetch_fii_dii() -> dict | None:
    """
    Fetch FII/DII daily flow data from moneycontrol.

    Returns net buy/sell amounts for both FII and DII in Rs Cr.
    Falls back gracefully if the page structure changes.
    """
    try:
        url = "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/data.json"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list) and len(data) > 0:
            latest = data[0]
            return {
                "date": latest.get("date", ""),
                "fii_net": latest.get("fii_net", ""),
                "dii_net": latest.get("dii_net", ""),
            }
    except Exception:
        pass

    try:
        url = "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.ok and "FII" in resp.text:
            return {"note": "FII/DII data available at moneycontrol but parsing failed. Check manually."}
    except Exception:
        pass

    return None


def _fetch_global_context() -> dict:
    """Fetch closing data for S&P 500, Crude, Gold, USD/INR with significance flags."""
    SIGNIFICANT_THRESHOLD = 1.5

    results = {}
    for name, ticker in GLOBAL_TICKERS.items():
        data = _fetch_daily_change(ticker)
        if data:
            data["significant_move"] = abs(data["change_pct"]) >= SIGNIFICANT_THRESHOLD
            results[name] = data

    significant = [f"{n} ({d['change_pct']:+.2f}%)" for n, d in results.items() if d.get("significant_move")]
    results["_significant_moves"] = significant or ["No major commodity/currency moves today"]

    return results


@tool
def get_market_data() -> dict:
    """Fetch comprehensive post-market data: indices, movers, sectors, VIX, 52-week context, FII/DII flows, and global context."""
    return {
        "indices": _fetch_indices(),
        "top_movers": _fetch_top_movers(),
        "sectors": _fetch_sectors(),
        "india_vix": _fetch_vix(),
        "nifty_52week": _fetch_52week_context(),
        "fii_dii": _fetch_fii_dii(),
        "global": _fetch_global_context(),
    }
