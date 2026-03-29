"""
Technical Analysis Engine
=========================
Fetches price data from yfinance and computes technical indicators.

Indicators computed:
- SMA (20, 50 day)
- RSI (14 period)
- MACD (12/26/9)
- Bollinger Bands (20 period, 2 std)
- ATR (14 period)
- Momentum (20 day)
- Volatility (20 day rolling std)
- Drawdown (from rolling max)

Signals are weighted and combined into a single TechnicalScore.
"""

import numpy as np
import pandas as pd
import requests as _tech_requests
import yfinance as yf
from config import THRESHOLDS


# =============================================================================
# KRAKEN FALLBACK
# =============================================================================

KRAKEN_PAIR_MAP = {
    'BTC-USD': 'XXBTZUSD', 'ETH-USD': 'XETHZUSD', 'SOL-USD': 'SOLUSD',
    'BNB-USD': 'BNBUSD', 'XRP-USD': 'XXRPZUSD', 'ADA-USD': 'ADAUSD',
    'AVAX-USD': 'AVAXUSD', 'DOGE-USD': 'XDGUSD', 'DOT-USD': 'DOTUSD',
    'LINK-USD': 'LINKUSD', 'MATIC-USD': 'MATICUSD', 'UNI-USD': 'UNIUSD',
    'LTC-USD': 'XLTCZUSD', 'BCH-USD': 'BCHUSD', 'ATOM-USD': 'ATOMUSD',
    'SHIB-USD': 'SHIBUSD', 'AAVE-USD': 'AAVEUSD',
}


def _period_to_dates(period: str):
    """Convert period string to (start, end) Timestamps."""
    end = pd.Timestamp.now().normalize()
    offsets = {
        '7d': pd.DateOffset(days=7), '30d': pd.DateOffset(days=30),
        '90d': pd.DateOffset(days=90), '6mo': pd.DateOffset(months=6),
        '1y': pd.DateOffset(years=1), '2y': pd.DateOffset(years=2),
    }
    return end - offsets.get(period, pd.DateOffset(days=90)), end


def _fetch_kraken(symbol: str, period: str):
    """Fetch OHLCV from Kraken REST API."""
    pair = KRAKEN_PAIR_MAP.get(symbol)
    if not pair:
        return None
    try:
        start, _ = _period_to_dates(period)
        resp = _tech_requests.get(
            'https://api.kraken.com/0/public/OHLC',
            params={'pair': pair, 'interval': 1440, 'since': int(start.timestamp())},
            timeout=10,
        )
        data = resp.json()
        if data.get('error') or not data.get('result'):
            return None
        key = [k for k in data['result'] if k != 'last']
        if not key:
            return None
        rows = data['result'][key[0]]
        df = pd.DataFrame(rows, columns=[
            'time', 'Open', 'High', 'Low', 'Close', 'vwap', 'Volume', 'count'
        ])
        df.index = pd.to_datetime(df['time'].astype(int), unit='s')
        for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df if len(df) >= 50 else None
    except Exception:
        return None


def _generate_synthetic(symbol: str, period: str):
    """Generate synthetic OHLCV as last-resort fallback."""
    start, end = _period_to_dates(period)
    idx = pd.bdate_range(start, end)
    n = len(idx)
    if n < 50:
        return None
    rng = np.random.default_rng(hash(symbol) & 0xFFFFFFFF)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    return pd.DataFrame({
        'Open': prices * 0.995, 'High': prices * 1.01,
        'Low': prices * 0.99, 'Close': prices,
        'Volume': rng.integers(1_000_000, 50_000_000, n).astype(float),
    }, index=idx)


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_price_data(symbol: str, period: str = "90d", interval: str = "1d"):
    """
    Fetch OHLCV data with fallback chain: yfinance -> Kraken -> synthetic.
    """
    # Primary: yfinance
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df is not None and not df.empty:
            if df.index.tz:
                df.index = df.index.tz_localize(None)
            return df
    except Exception as e:
        print(f"  [!] yfinance error for {symbol}: {e}")

    # Fallback: Kraken (daily only)
    if interval == "1d":
        df = _fetch_kraken(symbol, period)
        if df is not None:
            print(f"  [i] Using Kraken data for {symbol}")
            return df

    # Last resort: synthetic
    df = _generate_synthetic(symbol, period)
    if df is not None:
        print(f"  [!] Using synthetic data for {symbol}")
    return df


# =============================================================================
# INDICATOR CALCULATIONS
# =============================================================================

def compute_indicators(df):
    """
    Calculate all technical indicators on price DataFrame.
    
    Adds columns:
        SMA_20, SMA_50: Simple moving averages
        RSI: Relative Strength Index
        MACD, MACD_Signal, MACD_Hist: MACD components
        BB_Upper, BB_Mid, BB_Lower: Bollinger Bands
        ATR: Average True Range
        Momentum_20: 20-day momentum
        Volatility_20D: 20-day rolling volatility
        Drawdown: Current drawdown from peak
        
    Args:
        df: DataFrame with OHLCV columns
        
    Returns:
        DataFrame with indicator columns added, or None if insufficient data
    """
    if df is None or len(df) < 50:
        return None
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # -------------------------------------------------------------------------
    # Moving Averages
    # -------------------------------------------------------------------------
    df['SMA_20'] = close.rolling(20).mean()
    df['SMA_50'] = close.rolling(50).mean()
    
    # -------------------------------------------------------------------------
    # RSI (Relative Strength Index)
    # RSI < 30 = oversold (bullish), RSI > 70 = overbought (bearish)
    # -------------------------------------------------------------------------
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # -------------------------------------------------------------------------
    # MACD (Moving Average Convergence Divergence)
    # MACD_Hist > 0 = bullish momentum, < 0 = bearish
    # -------------------------------------------------------------------------
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # -------------------------------------------------------------------------
    # Bollinger Bands
    # Price above upper = overbought breakout, below lower = oversold breakdown
    # -------------------------------------------------------------------------
    df['BB_Mid'] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * bb_std
    df['BB_Lower'] = df['BB_Mid'] - 2 * bb_std
    
    # -------------------------------------------------------------------------
    # ATR (Average True Range) - Volatility measure
    # -------------------------------------------------------------------------
    tr = np.maximum(
        high - low,
        np.maximum(
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        )
    )
    df['ATR'] = tr.rolling(14).mean()
    
    # -------------------------------------------------------------------------
    # Momentum & Volatility
    # -------------------------------------------------------------------------
    df['Momentum_20'] = close.pct_change(20)
    df['Volatility_20D'] = close.pct_change().rolling(20).std()
    
    # -------------------------------------------------------------------------
    # Drawdown (from rolling maximum)
    # -------------------------------------------------------------------------
    rolling_max = close.cummax()
    df['Drawdown'] = (close - rolling_max) / rolling_max
    
    # -------------------------------------------------------------------------
    # Daily Return & Volatility Spike (for CautionScore)
    # VolSpike: current vol vs its 60-day rolling mean
    # -------------------------------------------------------------------------
    df['Return_1D'] = close.pct_change()
    vol_mean_60 = df['Volatility_20D'].rolling(60).mean()
    df['VolSpike'] = np.clip(
        ((df['Volatility_20D'] / vol_mean_60.replace(0, np.nan)) - 1) * 100,
        0, 100
    )
    
    return df


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def compute_signals(df) -> dict:
    """
    Generate trading signals from indicator values.
    
    Signal weights (must sum to 1.0):
        - Trend (price vs SMA20): 35%
        - MACD Histogram: 25%
        - RSI: 20%
        - Bollinger Breakout: 10%
        - Momentum: 10%
        
    Args:
        df: DataFrame with indicator columns
        
    Returns:
        Dict with:
            price, rsi, macd_hist, volatility, drawdown,
            signals (individual), tech_score, direction, confidence
    """
    if df is None or len(df) < 50:
        return None
    
    latest = df.iloc[-1]
    close = latest['Close']
    
    # -------------------------------------------------------------------------
    # Individual Signals (-1, 0, or +1)
    # -------------------------------------------------------------------------
    
    # RSI Signal: Oversold = buy, Overbought = sell
    if latest['RSI'] < 30:
        rsi_signal = 1   # Oversold - bullish
    elif latest['RSI'] > 70:
        rsi_signal = -1  # Overbought - bearish
    else:
        rsi_signal = 0   # Neutral
    
    # Trend Signal: Price vs SMA20
    if close > latest['SMA_20']:
        trend_signal = 1   # Above SMA - bullish
    elif close < latest['SMA_20']:
        trend_signal = -1  # Below SMA - bearish
    else:
        trend_signal = 0
    
    # MACD Signal: Histogram direction
    if latest['MACD_Hist'] > 0:
        macd_signal = 1   # Positive histogram - bullish
    elif latest['MACD_Hist'] < 0:
        macd_signal = -1  # Negative histogram - bearish
    else:
        macd_signal = 0
    
    # Breakout Signal: Bollinger Band position
    if close > latest['BB_Upper']:
        breakout_signal = 1   # Above upper band - breakout
    elif close < latest['BB_Lower']:
        breakout_signal = -1  # Below lower band - breakdown
    else:
        breakout_signal = 0
    
    # Momentum Signal: 20-day direction
    if latest['Momentum_20'] > 0:
        momentum_signal = 1
    elif latest['Momentum_20'] < 0:
        momentum_signal = -1
    else:
        momentum_signal = 0
    
    # -------------------------------------------------------------------------
    # Weighted Technical Score
    # -------------------------------------------------------------------------
    tech_score = (
        0.35 * trend_signal +
        0.25 * macd_signal +
        0.20 * rsi_signal +
        0.10 * breakout_signal +
        0.10 * momentum_signal
    )
    
    # -------------------------------------------------------------------------
    # Direction based on thresholds
    # -------------------------------------------------------------------------
    if tech_score >= THRESHOLDS['technical_long']:
        direction = "Long"
    elif tech_score <= THRESHOLDS['technical_short']:
        direction = "Short"
    else:
        direction = "Flat"
    
    # Confidence: magnitude of score (0-100)
    confidence = min(100, abs(tech_score) * 100)
    
    return {
        "price": close,
        "rsi": latest['RSI'],
        "macd_hist": latest['MACD_Hist'],
        "sma_20": latest['SMA_20'],
        "sma_50": latest['SMA_50'],
        "bb_upper": latest['BB_Upper'],
        "bb_lower": latest['BB_Lower'],
        "volatility": latest['Volatility_20D'],
        "drawdown": latest['Drawdown'],
        "atr": latest['ATR'],
        "vol_spike": latest.get('VolSpike', 0) if not np.isnan(latest.get('VolSpike', 0)) else 0,
        "drawdown_score": float(np.clip(abs(latest['Drawdown']) * 200, 0, 100)),
        "signals": {
            "rsi": rsi_signal,
            "trend": trend_signal,
            "macd": macd_signal,
            "breakout": breakout_signal,
            "momentum": momentum_signal,
        },
        "tech_score": tech_score,
        "direction": direction,
        "confidence": confidence,
    }


# =============================================================================
# MULTI-TIMEFRAME ANALYSIS
# =============================================================================

def get_weekly_trend(symbol: str) -> str:
    """
    Get weekly timeframe trend for confluence analysis.
    
    Used to confirm or contradict daily signals.
    Daily Long + Weekly Bullish = Strong signal
    Daily Long + Weekly Bearish = Counter-trend (risky)
    
    Args:
        symbol: yfinance ticker
        
    Returns:
        "Bullish", "Bearish", "Neutral", or "Unknown"
    """
    df = fetch_price_data(symbol, period="6mo", interval="1wk")
    
    if df is None or len(df) < 10:
        return "Unknown"
    
    close = df['Close'].iloc[-1]
    sma_10w = df['Close'].rolling(10).mean().iloc[-1]
    
    # 2% threshold for clear trend
    if close > sma_10w * 1.02:
        return "Bullish"
    elif close < sma_10w * 0.98:
        return "Bearish"
    return "Neutral"


def get_btc_trend() -> str:
    """
    Get BTC trend for market leader correlation.
    
    BTC leads the crypto market ~80% of the time.
    Going long altcoins while BTC dumps is risky.
    
    Returns:
        "Bullish", "Bearish", "Neutral", or "Unknown"
    """
    df = fetch_price_data("BTC-USD", period="30d", interval="1d")
    
    if df is None or len(df) < 20:
        return "Unknown"
    
    close = df['Close'].iloc[-1]
    sma_20 = df['Close'].rolling(20).mean().iloc[-1]
    
    if close > sma_20:
        return "Bullish"
    elif close < sma_20:
        return "Bearish"
    return "Neutral"


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def analyze(symbol: str) -> dict:
    """
    Full technical analysis pipeline.
    
    Args:
        symbol: yfinance ticker (e.g., "BTC-USD")
        
    Returns:
        Dict with all technical data and signals
    """
    df = fetch_price_data(symbol)
    df = compute_indicators(df)
    return compute_signals(df)


# =============================================================================
# FEATURE LAB (DataFrame-level for backtesting)
# =============================================================================

def build_feature_lab(df):
    """
    Build full feature set as DataFrame columns (matches notebook).
    Adds per-row: TechnicalScore, TechnicalConfidence, BaseDirection.
    Used by the backtest engine and trade simulator.
    """
    if df is None or len(df) < 50:
        return df

    df = compute_indicators(df)
    if df is None:
        return None

    close = df['Close']

    # Per-row signal vectors
    df['sig_trend'] = np.where(close > df['SMA_20'], 1,
                               np.where(close < df['SMA_20'], -1, 0))
    df['sig_macd'] = np.where(df['MACD_Hist'] > 0, 1,
                              np.where(df['MACD_Hist'] < 0, -1, 0))
    df['sig_rsi'] = np.where(df['RSI'] < 30, 1,
                             np.where(df['RSI'] > 70, -1, 0))
    df['sig_breakout'] = np.where(close > df['BB_Upper'], 1,
                                  np.where(close < df['BB_Lower'], -1, 0))
    df['sig_momentum'] = np.where(df['Momentum_20'] > 0, 1,
                                  np.where(df['Momentum_20'] < 0, -1, 0))

    df['TechnicalScore'] = (
        0.35 * df['sig_trend'] + 0.25 * df['sig_macd'] +
        0.20 * df['sig_rsi'] + 0.10 * df['sig_breakout'] +
        0.10 * df['sig_momentum']
    )
    df['TechnicalConfidence'] = np.clip(df['TechnicalScore'].abs() * 100, 0, 100)
    df['BaseDirection'] = np.where(
        df['TechnicalScore'] >= THRESHOLDS['technical_long'], 'Long',
        np.where(df['TechnicalScore'] <= THRESHOLDS['technical_short'], 'Short', 'Flat')
    )

    return df
