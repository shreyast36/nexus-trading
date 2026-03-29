"""
Configuration Module
====================
Central configuration for the Predictive Market Decision Engine.
All thresholds, API endpoints, and asset definitions live here.

To modify behavior:
- Adjust thresholds in THRESHOLDS dict
- Add new assets to ASSETS dict
- Modify RSS feeds in RSS_FEEDS dict

For Streamlit Cloud deployment:
- Add secrets in the Streamlit Cloud dashboard under Settings > Secrets
- Format: GROQ_API_KEY = "your-api-key"
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from credentials.env (check parent dir too)
env_path = Path('credentials.env')
if not env_path.exists():
    env_path = Path('../credentials.env')
load_dotenv(env_path)


def _get_secret(key: str, default=None):
    """Get secret from Streamlit secrets (Cloud) or environment variables (local)."""
    # Try Streamlit secrets first (for Cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    # Fall back to environment variable
    return os.getenv(key, default)


# =============================================================================
# API ENDPOINTS
# =============================================================================

POLYMARKET_CLOB_HOST = "https://clob.polymarket.com"
POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com/markets"
POLYMARKET_CHAIN_ID = 137  # Polygon mainnet


# =============================================================================
# CREDENTIALS (from Streamlit secrets or environment)
# =============================================================================

POLY_API_KEY = _get_secret('POLY_API_KEY')
POLY_SECRET = _get_secret('POLY_SECRET')
POLY_PASSPHRASE = _get_secret('POLY_PASSPHRASE')
POLY_PRIVATE_KEY = _get_secret('POLY_PRIVATE_KEY')
GROQ_API_KEY = _get_secret('GROQ_API_KEY')


# =============================================================================
# TRADING THRESHOLDS
# =============================================================================
# These control when trades are triggered and position sizing.
# Tune these based on backtesting results.

THRESHOLDS = {
    # Technical score thresholds for direction
    "technical_long": 0.25,      # Score >= this = Long signal
    "technical_short": -0.25,    # Score <= this = Short signal
    
    # Confidence thresholds for position sizing
    "confidence_trade": 35,      # Min confidence to trade
    "confidence_quarter": 55,    # Quarter position
    "confidence_half": 70,       # Half position
    "confidence_three_quarter": 85,  # Three-quarter position
    
    # Risk thresholds
    "caution_low": 30,           # Below = Tradeable
    "caution_medium": 55,        # Below = Cautious
    "caution_high": 75,          # Below = High Risk, Above = Avoid
}


# =============================================================================
# ASSET DEFINITIONS
# =============================================================================
# Maps user input (e.g., "btc", "bitcoin") to:
#   - ticker: Display name
#   - yf_symbol: yfinance symbol for price data
#   - keywords: Search terms for Polymarket/News

ASSETS = {
    # ── Major Layer 1s ───────────────────────────────────────────────────────
    "btc":      {"ticker": "BTC",   "yf_symbol": "BTC-USD",   "keywords": ["bitcoin", "btc"]},
    "bitcoin":  {"ticker": "BTC",   "yf_symbol": "BTC-USD",   "keywords": ["bitcoin", "btc"]},
    "eth":      {"ticker": "ETH",   "yf_symbol": "ETH-USD",   "keywords": ["ethereum", "eth"]},
    "ethereum": {"ticker": "ETH",   "yf_symbol": "ETH-USD",   "keywords": ["ethereum", "eth"]},
    "sol":      {"ticker": "SOL",   "yf_symbol": "SOL-USD",   "keywords": ["solana", "sol"]},
    "solana":   {"ticker": "SOL",   "yf_symbol": "SOL-USD",   "keywords": ["solana", "sol"]},
    "bnb":      {"ticker": "BNB",   "yf_symbol": "BNB-USD",   "keywords": ["binance", "bnb"]},
    "xrp":      {"ticker": "XRP",   "yf_symbol": "XRP-USD",   "keywords": ["ripple", "xrp"]},
    "ada":      {"ticker": "ADA",   "yf_symbol": "ADA-USD",   "keywords": ["cardano", "ada"]},
    "cardano":  {"ticker": "ADA",   "yf_symbol": "ADA-USD",   "keywords": ["cardano", "ada"]},
    "avax":     {"ticker": "AVAX",  "yf_symbol": "AVAX-USD",  "keywords": ["avalanche", "avax"]},
    "dot":      {"ticker": "DOT",   "yf_symbol": "DOT-USD",   "keywords": ["polkadot", "dot"]},

    # ── Layer 2 ──────────────────────────────────────────────────────────────
    "matic":    {"ticker": "MATIC", "yf_symbol": "MATIC-USD", "keywords": ["polygon", "matic"]},

    # ── DeFi ─────────────────────────────────────────────────────────────────
    "link":     {"ticker": "LINK",  "yf_symbol": "LINK-USD",  "keywords": ["chainlink", "link"]},
    "uni":      {"ticker": "UNI",   "yf_symbol": "UNI-USD",   "keywords": ["uniswap", "uni"]},
    "aave":     {"ticker": "AAVE",  "yf_symbol": "AAVE-USD",  "keywords": ["aave"]},

    # ── Meme ─────────────────────────────────────────────────────────────────
    "doge":     {"ticker": "DOGE",  "yf_symbol": "DOGE-USD",  "keywords": ["dogecoin", "doge"]},
    "shib":     {"ticker": "SHIB",  "yf_symbol": "SHIB-USD",  "keywords": ["shiba inu", "shib"]},

    # ── Established Alts ─────────────────────────────────────────────────────
    "ltc":      {"ticker": "LTC",   "yf_symbol": "LTC-USD",   "keywords": ["litecoin", "ltc"]},
    "bch":      {"ticker": "BCH",   "yf_symbol": "BCH-USD",   "keywords": ["bitcoin cash", "bch"]},
    "atom":     {"ticker": "ATOM",  "yf_symbol": "ATOM-USD",  "keywords": ["cosmos", "atom"]},
}


# =============================================================================
# POLYMARKET DIRECTION KEYWORDS
# =============================================================================
# Used to classify prediction market questions as bullish or bearish.

BULLISH_KEYWORDS = [
    "above", "hit", "reach", "reserve", "etf approval", "approved",
    "bull", "all-time high", "new high", "surge", "rally", "breakout",
]

BEARISH_KEYWORDS = [
    "below", "ban", "collapse", "hack", "recession", "crash",
    "bear", "down", "dump", "fail", "reject", "plunge",
]

# Per-asset keyword overrides for more accurate direction tagging.
# Falls back to the global lists above for assets not listed here.
PER_ASSET_BULLISH_KEYWORDS = {
    "BTC": ["above", "hit", "reach", "etf", "approval", "approved", "reserve",
            "strategic reserve", "all-time high", "new high", "bull", "surge",
            "rally", "breakout", "institutional", "adopt"],
    "ETH": ["above", "hit", "reach", "etf", "approval", "approved", "upgrade",
            "dencun", "pectra", "all-time high", "new high", "bull", "surge",
            "rally", "breakout", "staking"],
    "SOL": ["above", "hit", "reach", "ecosystem", "meme coin", "surge", "rally",
            "breakout", "new high", "firedancer", "adoption", "bull"],
}

PER_ASSET_BEARISH_KEYWORDS = {
    "BTC": ["below", "ban", "collapse", "hack", "crash", "recession", "down",
            "dump", "reject", "fail", "regulatory crackdown", "bear", "plunge"],
    "ETH": ["below", "ban", "collapse", "hack", "crash", "down", "dump",
            "reject", "fail", "bear", "plunge", "delay", "vulnerability"],
    "SOL": ["below", "ban", "collapse", "hack", "outage", "crash", "down",
            "dump", "reject", "fail", "bear", "plunge", "congestion"],
}

THEME_KEYWORDS = [
    "regulation", "sec", "fed", "halving", "etf", "defi", "layer 2",
    "l2", "institutional", "whale", "liquidation", "stablecoin",
    "cbdc", "airdrop", "governance",
]

# Words that confirm the text is about cryptocurrency (not a false positive).
# If a short ticker like "sol" or "op" matches, we require at least one of these
# context words to also appear, so "solar panel" or "opinion" gets filtered out.
CRYPTO_CONTEXT_WORDS = [
    "crypto", "token", "coin", "blockchain", "defi", "nft",
    "trading", "trader", "price", "market cap", "exchange",
    "binance", "coinbase", "kraken", "bybit", "okx",
    "wallet", "staking", "yield", "airdrop", "mainnet",
    "testnet", "protocol", "chain", "layer 2", "l2",
    "bullish", "bearish", "whale", "hodl", "altcoin",
    "etf", "sec", "regulation", "mining", "halving",
    "polymarket", "prediction market", "dex", "swap",
    "$", "usd", "usdt", "usdc", "satoshi", "gwei",
]

# Tickers/keywords that are too short or too common as English words.
# These REQUIRE crypto context to be considered a valid match.
AMBIGUOUS_KEYWORDS = {
    "sol", "dot", "link", "uni", "ada", "atom", "matic",
}


# =============================================================================
# RSS NEWS FEEDS
# =============================================================================
# Categorized by source type for diverse sentiment coverage.

RSS_FEEDS = {
    "Retail": [
        "https://cointelegraph.com/rss",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://decrypt.co/feed",
        "https://bitcoinmagazine.com/.rss/full/",
        "https://www.newsbtc.com/feed/",
    ],
    "Institutional": [
        "https://www.theblock.co/rss.xml",
        "https://www.bankless.com/rss.xml",
        "https://www.dlnews.com/arc/outboundfeeds/rss/",
    ],
    "Security": [
        "https://rekt.news/feed.xml",
        "https://blog.chainalysis.com/feed/",
    ],
    "Regulatory": [
        "https://unchainedcrypto.com/feed/",
        "https://www.coinlaw.io/feed",
    ],
    "DeFi": [
        "https://thedefiant.io/feed",
        "https://www.defipulse.com/blog/rss.xml",
    ],
    "General": [
        "https://www.wired.com/feed/tag/cryptocurrency/latest/rss",
        "https://techcrunch.com/tag/cryptocurrency/feed/",
    ],
}


# =============================================================================
# FUSION WEIGHTS
# =============================================================================
# Weights for combining signals in the fusion engine.
# Must sum to ~1.0 for the positive components.

FUSION_WEIGHTS = {
    "technical": 0.55,          # Technical confidence weight
    "pm_confirmation": 0.25,   # PM confirmation (same-direction) weight
    "pm_quality": 0.10,         # PM market quality bonus
    "pm_conflict": -0.10,       # PM conflict (opposite-direction) penalty
    "caution": -0.25,           # CautionScore penalty
}


# =============================================================================
# HELPER FUNCTION
# =============================================================================

def resolve_asset(query: str) -> dict:
    """
    Convert user input to standardized asset configuration.
    
    Args:
        query: User input like "btc", "bitcoin", "ETH"
        
    Returns:
        Asset config dict with ticker, yf_symbol, keywords
        
    Example:
        >>> resolve_asset("btc")
        {"ticker": "BTC", "yf_symbol": "BTC-USD", "keywords": ["bitcoin", "btc"]}
    """
    q = query.strip().lower()
    
    if q in ASSETS:
        return ASSETS[q]
    
    # Fallback: treat as generic keyword search
    return {
        "ticker": query.upper(),
        "yf_symbol": f"{query.upper()}-USD",
        "keywords": [q],
    }
