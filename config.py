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
EMAIL_SMTP_TIMEOUT_SEC = 25
EMAIL_SEND_RETRIES = 2

# smtp: send Gmail as today. terminal: skip SMTP; main prints full premium + standalone.
_raw_email_delivery = (os.getenv("EMAIL_DELIVERY") or "smtp").strip().lower()
EMAIL_DELIVERY = _raw_email_delivery if _raw_email_delivery in ("smtp", "terminal") else "smtp"

# Banner-style print() of final tweets in main.py. Default on so smtp runs still show output in the console.
_raw_print_final = (os.getenv("PRINT_FINAL_TO_TERMINAL") or "true").strip().lower()
PRINT_FINAL_TO_TERMINAL = _raw_print_final not in ("0", "false", "no", "off")

# --- Single-run lock (main.py) ---
# If a run is killed mid-flight, run.lock may remain; break it after this many minutes
# or when the recorded PID is no longer running.
RUN_LOCK_STALE_MINUTES = 45

# --- LLM ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WRITER_MODEL = "gpt-4o"            # Premium model for narrative quality
AUDITOR_MODEL = "gpt-4o"           # Premium model for catching hallucinations

# All LLM chat steps (writer, auditor, synthesizer, prioritizer): "openai" (default) or
# "ollama" for local models (e.g. llama3.1:8b). OpenAI model names apply only when provider is openai.
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL") or "llama3.1:8b"
_ollama_ctx = os.getenv("OLLAMA_NUM_CTX")
OLLAMA_NUM_CTX = int(_ollama_ctx) if _ollama_ctx and _ollama_ctx.isdigit() else None


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip(), 10)
    except ValueError:
        return default


# OpenAI 429 / burst protection (used when LLM_PROVIDER=openai). Throttle spaces out calls;
# retries use exponential backoff with jitter after RateLimitError or 429-like messages.
OPENAI_MIN_CALL_INTERVAL_SEC = max(0.0, _env_float("OPENAI_MIN_CALL_INTERVAL_SEC", 0.65))
OPENAI_RATE_LIMIT_MAX_ATTEMPTS = max(1, _env_int("OPENAI_RATE_LIMIT_MAX_ATTEMPTS", 8))
OPENAI_RETRY_BASE_DELAY_SEC = max(0.1, _env_float("OPENAI_RETRY_BASE_DELAY_SEC", 1.25))
OPENAI_RETRY_MAX_DELAY_SEC = max(OPENAI_RETRY_BASE_DELAY_SEC, _env_float("OPENAI_RETRY_MAX_DELAY_SEC", 90.0))
OPENAI_SDK_MAX_RETRIES = max(0, _env_int("OPENAI_SDK_MAX_RETRIES", 2))

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
    "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS",
    "TATASTEEL.NS", "TECHM.NS", "TITAN.NS", "TRENT.NS",
    "ULTRACEMCO.NS", "WIPRO.NS",
]

# --- Global Context Tickers ---
GLOBAL_TICKERS = {
    "S&P 500": "^GSPC",
    "Dow Jones": "^DJI",
    "Kospi": "^KS11",
    "Nikkei 225": "^N225",
    "Crude Oil": "CL=F",
    "Brent": "BZ=F",
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
AUDIT_PASS_THRESHOLD = 10  # Min score (out of 12) to approve a tweet - raised for higher insight bar
MAX_AUDIT_CYCLES = 5      # Max batches before force-emailing best available
MAX_TWEET_CHARS = 1100    # Max characters per tweet/thread - reduced for Kobeissi ideal length

# --- Story Prioritizer (Phase A + B) ---
PRIORITIZER_MODE = "deterministic"  # deterministic | hybrid
PRIORITIZER_TOP_K = 7
PRIORITIZER_MIN_SCORE = 6
PRIORITIZER_MAX_PER_THEME = 2
PRIORITIZER_REQUIRE_MACRO = True
PRIORITIZER_REQUIRE_IDIOSYNCRATIC_IF_AVAILABLE = True
PRIORITIZER_MEMORY_WINDOW = 7

# Rolling editorial mix targets for Phase B allocation penalties.
PRIORITIZER_THEME_TARGETS = {
    "indiaEquityTape": 0.22,
    "riskOffUnwind": 0.18,
    "commodityShock": 0.20,
    "fxStress": 0.12,
    "policyRepricing": 0.12,
    "idiosyncraticCorporate": 0.10,
    "geopoliticsEnergy": 0.06,
}

# Synthetic tool-anchored rows when futures/FX/VIX move sharply (no headline required)
PRIORITIZER_BRENT_ANCHOR_PCT = 1.2
PRIORITIZER_GOLD_ANCHOR_PCT = 1.0
PRIORITIZER_CRUDE_ANCHOR_PCT = 1.2
PRIORITIZER_VIX_ANCHOR_CHANGE_PCT = 5.0
PRIORITIZER_FX_WEEK_ANCHOR = 0.45
SYNTHETIC_STORY_SCORE_BOOST = 3