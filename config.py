import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys (loaded from .env file) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY")
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")

# --- Email ---
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_TO = os.getenv("EMAIL_TO")
EMAIL_SMTP_SERVER = "smtp.gmail.com"
EMAIL_SMTP_PORT = 587

# --- LLM ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WRITER_MODEL = "gpt-4o"            # Premium model for narrative quality
AUDITOR_MODEL = "gpt-4o"           # Premium model for catching hallucinations

# --- Indian Indices ---
INDIAN_INDICES = {
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
}

# --- Nifty 50 Constituents (for finding top movers) ---
NIFTY_50_TICKERS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",
    "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "BEL.NS", "BPCL.NS", "BHARTIARTL.NS", "BRITANNIA.NS",
    "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS",
    "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS",
    "ITC.NS", "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS",
    "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS",
    "NTPC.NS", "NESTLEIND.NS", "ONGC.NS", "POWERGRID.NS",
    "RELIANCE.NS", "SBILIFE.NS", "SHRIRAMFIN.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS",
    "TATASTEEL.NS", "TECHM.NS", "TITAN.NS", "TRENT.NS",
    "ULTRACEMCO.NS", "WIPRO.NS",
]

# --- Global Context Tickers ---
GLOBAL_TICKERS = {
    "S&P 500": "^GSPC",
    "Dow Jones": "^DJI",
    "Crude Oil": "CL=F",
    "Gold": "GC=F",
    "USD/INR": "USDINR=X",
}

# --- Sector Indices ---
SECTOR_INDICES = {
    "Nifty Bank": "^NSEBANK",
    "Nifty IT": "^CNXIT",
    "Nifty Pharma": "^CNXPHARMA",
    "Nifty FMCG": "^CNXFMCG",
}

# --- Volatility ---
INDIA_VIX_TICKER = "^INDIAVIX"

# --- Top Movers ---
TOP_MOVERS_COUNT = 3  # Top 3 gainers + top 3 losers

# --- News Keywords ---
NEWS_KEYWORDS = [
    "India", "RBI", "Nifty", "Sensex", "Fed", "tariffs",
    "crude oil", "inflation", "emerging markets", "FII", "rupee",
]

# --- Agent ---
TWEET_BATCH_SIZE = 3      # Tweets generated per batch
TOTAL_NEEDED = 5          # Total good tweets needed before emailing
AUDIT_PASS_THRESHOLD = 9  # Min score (out of 12) to approve a tweet
MAX_AUDIT_CYCLES = 5      # Max batches before force-emailing best available
MAX_TWEET_CHARS = 1500    # Max characters per tweet/thread
