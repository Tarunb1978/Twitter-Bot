"""
Macro economic data tool -- fetches key India economic indicators.

Returns:
- RBI Policy Rate (current + previous)
- India CPI (inflation rate)
- USD/INR spot and forward expectations
- Bond yields (10Y Gsec)
- GDP growth forecasts/latest
- FII/DII net flows
"""

import requests
from langchain_core.tools import tool
from datetime import datetime


def _fetch_rbi_rate() -> dict | None:
    """
    Fetch RBI Policy Rate.
    Returns current rate, previous rate, and direction.
    
    Note: In production, this would connect to RBI's API or data provider.
    For now, we'll try to fetch from a financial data source.
    """
    try:
        # Attempt to fetch from a financial API
        # This is a placeholder - in production you'd use RBI data or a financial API
        url = "https://api.example.com/rbi/rate"  # Placeholder
        response = requests.get(url, timeout=5)
        if response.ok:
            data = response.json()
            return {
                "current_rate": data.get("current_rate"),
                "previous_rate": data.get("previous_rate"),
                "change_bps": (data.get("current_rate") - data.get("previous_rate")) * 100,
            }
    except Exception:
        pass
    
    # Fallback: Return None and let prompt know it's unavailable
    return None


def _fetch_inflation_cpi() -> dict | None:
    """
    Fetch India CPI (inflation) data.
    Returns latest CPI YoY change and previous month.
    """
    try:
        # Placeholder for CPI data fetch
        # In production, connect to MOSPI or financial data provider
        return {
            "latest_cpi_yoy": None,  # Will be populated from API
            "previous_cpi_yoy": None,
            "month": "Feb 2026",
            "note": "CPI data requires connection to MOSPI or financial API"
        }
    except Exception:
        pass
    
    return None


def _fetch_bond_yields() -> dict | None:
    """
    India 10Y G-Sec yield.

    Yahoo symbol ^INBOND10Y is not available (404 / delisted); calling yfinance on it
    only spams errors. Use RBI/CCIL or a bond data API in production.
    """
    return {
        "yield_10y": None,
        "note": "India 10Y G-Sec not on Yahoo (^INBOND10Y unavailable); use RBI/CCIL or a bond data feed.",
    }


def _fetch_usdinr_forward() -> dict | None:
    """
    Fetch USD/INR spot and implied forward rupee weakness.
    Returns spot rate, 1M/3M/6M forward expectations.
    """
    try:
        import yfinance as yf
        
        hist = yf.Ticker("USDINR=X").history(period="5d")
        if not hist.empty:
            current = float(hist["Close"].iloc[-1])
            prev_week = float(hist["Close"].iloc[0]) if len(hist) > 0 else current
            week_change = current - prev_week
            
            return {
                "spot_rate": round(current, 2),
                "week_change": round(week_change, 2),
                "direction": "weakening INR" if week_change > 0 else "strengthening INR",
                "implication": "INR weakness boosts IT export margins" if week_change > 0 else "INR strength pressures exporters",
            }
    except Exception:
        pass
    
    return None


def _fetch_fii_dii_flows() -> dict | None:
    """
    Fetch FII/DII net flows for the day/period.
    Returns net buy/sell amounts in Rs Cr.
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
                "fii_net_buy": latest.get("fii_net", ""),
                "dii_net_buy": latest.get("dii_net", ""),
                "note": "Positive = net buying, Negative = net selling"
            }
    except Exception:
        pass
    
    # Fallback
    return {
        "note": "FII/DII flow data requires moneycontrol connection or financial API"
    }


def _fetch_gdp_forecast() -> dict | None:
    """
    Fetch India GDP growth forecast/latest.
    Returns consensus forecast and previous actual.
    """
    try:
        # In production, connect to consensus forecast APIs or RBI projections
        return {
            "fy26_forecast": None,  # Will be populated from API
            "fy25_actual": None,
            "note": "GDP forecast requires connection to RBI projections or consensus APIs"
        }
    except Exception:
        pass
    
    return None


@tool
def get_macro_data() -> dict:
    """
    Fetch macro economic indicators for India: RBI rate, inflation, USD/INR, 
    bond yields, GDP forecasts, FII/DII flows.
    
    Returns a dict with each indicator (may be None if unavailable).
    """
    return {
        "rbi_rate": _fetch_rbi_rate(),
        "cpi_inflation": _fetch_inflation_cpi(),
        "bond_yields_10y": _fetch_bond_yields(),
        "usdinr": _fetch_usdinr_forward(),
        "fii_dii_flows": _fetch_fii_dii_flows(),
        "gdp_forecast": _fetch_gdp_forecast(),
        "fetch_time": datetime.now().isoformat(),
    }