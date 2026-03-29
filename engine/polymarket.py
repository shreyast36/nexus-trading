"""
Polymarket Engine
=================
Fetches prediction markets from Polymarket and aggregates sentiment.

Features:
- Search markets by keywords
- Tag questions as bullish/bearish/ambiguous
- Aggregate into sentiment score, quality, dispersion

Note: This uses the public Gamma API (no auth needed for reading).
The CLOB client is only needed for placing orders.
"""

import re
import json
import requests
import numpy as np
from config import (
    POLYMARKET_GAMMA_API,
    BULLISH_KEYWORDS,
    BEARISH_KEYWORDS,
    CRYPTO_CONTEXT_WORDS,
    AMBIGUOUS_KEYWORDS,
    PER_ASSET_BULLISH_KEYWORDS,
    PER_ASSET_BEARISH_KEYWORDS,
)


# =============================================================================
# MARKET FETCHING
# =============================================================================

def search_markets(keywords: list, max_results: int = 15) -> list:
    """
    Search Polymarket for markets matching keywords.
    
    Args:
        keywords: List of search terms (e.g., ["bitcoin", "btc"])
        max_results: Maximum markets to return
        
    Returns:
        List of market dicts with question, odds, volume, liquidity
    """
    if not keywords:
        return []
    
    try:
        resp = requests.get(
            POLYMARKET_GAMMA_API,
            params={"limit": 500, "active": "true", "closed": "false"},
            timeout=5
        )
        resp.raise_for_status()
        all_markets = resp.json()
        
    except requests.RequestException as e:
        print(f"  [!] Polymarket API error: {e}")
        return []
    
    # Filter by keywords — match ONLY on question + slug, NOT description.
    # The description field is extremely verbose and causes false positives
    # (e.g., GTA VI market description mentioning "Bitcoin hitting $1M").
    results = []
    for m in all_markets:
        if len(results) >= max_results:
            break
        
        if not m.get('active', True):
            continue

        question = m.get('question', '').lower()
        slug = m.get('slug', '').lower().replace('-', ' ')
        search_text = f"{question} {slug}"

        # Check each keyword with word-boundary matching
        matched = False
        for kw in keywords:
            kw_lower = kw.lower()
            if not re.search(rf'\b{re.escape(kw_lower)}\b', search_text):
                continue
            # Short / ambiguous tickers need crypto context in search text
            if kw_lower in AMBIGUOUS_KEYWORDS:
                if not any(ctx in search_text for ctx in CRYPTO_CONTEXT_WORDS):
                    continue
            matched = True
            break

        if not matched:
            continue

        # Parse outcome prices
        yes_odds, no_odds = _parse_outcome_prices(m.get('outcomePrices', '[]'))

        # Spread: deviation from tight market (yes + no = 1.0)
        spread = abs((yes_odds or 0.5) + (no_odds or 0.5) - 1.0)

        # Per-market quality: volume, liquidity, spread
        vol_raw = float(m.get('volume', 0) or 0)
        liq_raw = float(m.get('liquidity', 0) or 0)
        vol_score = min(1.0, vol_raw / 500_000)
        liq_score = min(1.0, liq_raw / 200_000)
        spread_score = max(0.0, 1.0 - spread * 2)
        mkt_quality = 0.4 * vol_score + 0.35 * liq_score + 0.25 * spread_score

        results.append({
            "question": m.get('question', ''),
            "slug": m.get('slug', ''),
            "yes_odds": yes_odds,
            "no_odds": no_odds,
            "volume": m.get('volume'),
            "liquidity": m.get('liquidity'),
            "spread": spread,
            "market_quality_score": mkt_quality,
            "clobTokenIds": m.get('clobTokenIds', ''),
        })
    
    return results


def _parse_outcome_prices(prices) -> tuple:
    """
    Parse outcome prices from API response.
    
    Args:
        prices: String or list of prices
        
    Returns:
        Tuple of (yes_price, no_price) or (None, None)
    """
    try:
        if isinstance(prices, str):
            prices = json.loads(prices)
        
        yes_price = float(prices[0]) if len(prices) > 0 else None
        no_price = float(prices[1]) if len(prices) > 1 else None
        return yes_price, no_price
        
    except (json.JSONDecodeError, IndexError, TypeError, ValueError):
        return None, None


# =============================================================================
# QUESTION TAGGING
# =============================================================================

def tag_direction(question: str, asset_ticker: str = None) -> str:
    """
    Classify a prediction market question as bullish, bearish, or ambiguous.
    
    Uses per-asset keyword overrides when available, otherwise falls back
    to the global BULLISH_KEYWORDS / BEARISH_KEYWORDS lists.
    
    Args:
        question: The market question text
        asset_ticker: Optional asset ticker (e.g., "BTC") for per-asset keywords
        
    Returns:
        "bullish", "bearish", or "ambiguous"
    """
    q = question.lower()
    
    bull_kw = PER_ASSET_BULLISH_KEYWORDS.get(asset_ticker, BULLISH_KEYWORDS) if asset_ticker else BULLISH_KEYWORDS
    bear_kw = PER_ASSET_BEARISH_KEYWORDS.get(asset_ticker, BEARISH_KEYWORDS) if asset_ticker else BEARISH_KEYWORDS
    
    bull_count = sum(1 for kw in bull_kw if kw in q)
    bear_count = sum(1 for kw in bear_kw if kw in q)
    
    if bull_count > bear_count:
        return "bullish"
    elif bear_count > bull_count:
        return "bearish"
    return "ambiguous"


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate(markets: list, asset_ticker: str = None) -> dict:
    """
    Aggregate Polymarket data into summary features.
    
    Matches the notebook's aggregate_polymarket_snapshot output:
    - sentiment: Liquidity-weighted net sentiment (-1 to +1)
    - quality: Composite market quality score (0-1)
    - spread_mean: Average market spread (0-1)
    - event_risk: Fraction of markets near resolution (0-1)
    - dispersion: Std dev of YES prices (disagreement)
    - count, markets: For display
    """
    if not markets:
        return {
            "count": 0,
            "sentiment": 0.0,
            "bullish_avg": None,
            "bearish_avg": None,
            "dispersion": 0.0,
            "quality": 0.0,
            "spread_mean": 0.0,
            "event_risk": 0.0,
            "markets": [],
        }
    
    bullish_probs = []
    bearish_probs = []
    all_probs = []
    signed_prices = []
    weights = []
    
    for m in markets:
        direction = tag_direction(m['question'], asset_ticker)
        yes_prob = m.get('yes_odds')
        liq = max(1.0, float(m.get('liquidity', 0) or 0))
        
        # Store direction info on each market for display
        m['direction'] = direction
        m['direction_sign'] = 1 if direction == "bullish" else (-1 if direction == "bearish" else 0)
        
        if yes_prob is not None:
            all_probs.append(yes_prob)
            signed_price = m['direction_sign'] * yes_prob
            signed_prices.append(signed_price)
            weights.append(liq)
            
            if direction == "bullish":
                bullish_probs.append(yes_prob)
            elif direction == "bearish":
                bearish_probs.append(yes_prob)
    
    # -------------------------------------------------------------------------
    # Liquidity-Weighted Net Sentiment (notebook style)
    # -------------------------------------------------------------------------
    if signed_prices and sum(weights) > 0:
        sentiment = float(np.clip(
            sum(s * w for s, w in zip(signed_prices, weights)) / sum(weights),
            -1, 1
        ))
    else:
        sentiment = 0.0
    
    # -------------------------------------------------------------------------
    # Dispersion (disagreement between markets)
    # -------------------------------------------------------------------------
    dispersion = float(np.std(all_probs)) if len(all_probs) > 1 else 0.0
    
    # -------------------------------------------------------------------------
    # Quality: Composite of per-market quality scores (notebook formula)
    # -------------------------------------------------------------------------
    quality_scores = [m.get('market_quality_score', 0) for m in markets]
    quality = float(np.mean(quality_scores)) if quality_scores else 0.0
    
    # -------------------------------------------------------------------------
    # Spread Mean
    # -------------------------------------------------------------------------
    spreads = [m.get('spread', 0) for m in markets]
    spread_mean = float(np.mean(spreads)) if spreads else 0.0
    
    # -------------------------------------------------------------------------
    # Event Risk: Fraction of markets with extreme odds (near resolution)
    # -------------------------------------------------------------------------
    if all_probs:
        extreme = sum(1 for p in all_probs if p > 0.85 or p < 0.15)
        event_risk = extreme / len(all_probs)
    else:
        event_risk = 0.0
    
    return {
        "count": len(markets),
        "sentiment": sentiment,
        "bullish_avg": np.mean(bullish_probs) if bullish_probs else None,
        "bearish_avg": np.mean(bearish_probs) if bearish_probs else None,
        "dispersion": dispersion,
        "quality": quality,
        "spread_mean": spread_mean,
        "event_risk": event_risk,
        "markets": markets[:5],  # Top 5 for display
    }


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def analyze(keywords: list, asset_ticker: str = None) -> dict:
    """
    Full Polymarket analysis pipeline.
    
    Args:
        keywords: Search terms for the asset
        asset_ticker: Optional ticker (e.g., "BTC") for per-asset keyword tagging
        
    Returns:
        Aggregated Polymarket features
    """
    markets = search_markets(keywords)
    return aggregate(markets, asset_ticker=asset_ticker)


# =============================================================================
# HISTORICAL PM DATA (token-level price history)
# =============================================================================

def _parse_token_ids(market: dict) -> list:
    """Extract token IDs from a market dict."""
    raw = market.get('clobTokenIds', '')
    if not raw:
        return []
    try:
        tokens = json.loads(raw) if isinstance(raw, str) else raw
        return [str(t) for t in tokens if t]
    except (json.JSONDecodeError, TypeError):
        return []


def fetch_token_price_history(token_id: str, start_ts: int, end_ts: int,
                              fidelity_minutes: int = 1440) -> list:
    """
    Fetch token-level price history from Polymarket CLOB.
    Chunked into 30-day windows to respect API limits.
    """
    history = []
    chunk = 86400 * 30  # 30 days
    current = start_ts
    while current < end_ts:
        chunk_end = min(current + chunk, end_ts)
        try:
            resp = requests.get(
                'https://clob.polymarket.com/prices-history',
                params={
                    'market': token_id,
                    'startTs': str(int(current)),
                    'endTs': str(int(chunk_end)),
                    'fidelity': str(int(fidelity_minutes)),
                },
                timeout=8,
            )
            if resp.ok:
                for pt in resp.json().get('history', []):
                    history.append({'timestamp': int(pt['t']), 'price': float(pt['p'])})
        except Exception:
            pass
        current = chunk_end
    return history


def build_historical_polymarket_panel(tagged_markets: list, start_date: str,
                                      end_date: str, asset_index,
                                      fidelity_minutes: int = 1440,
                                      max_markets: int = 12):
    """
    Build time-series PM features aligned to market data dates.
    Returns DataFrame with columns:
      pm_hist_liq_weighted_sentiment, pm_hist_net_sentiment,
      pm_hist_mean_price, pm_hist_market_quality
    """
    import pandas as pd

    empty = pd.DataFrame(
        0.0, index=asset_index,
        columns=['pm_hist_liq_weighted_sentiment', 'pm_hist_net_sentiment',
                 'pm_hist_mean_price', 'pm_hist_market_quality']
    )
    if not tagged_markets:
        return empty

    start_ts = int(pd.Timestamp(start_date).timestamp())
    end_ts = int(pd.Timestamp(end_date).timestamp())

    series_list = []
    for m in tagged_markets[:max_markets]:
        token_ids = _parse_token_ids(m)
        if not token_ids:
            continue
        hist = fetch_token_price_history(token_ids[0], start_ts, end_ts,
                                         fidelity_minutes)
        if not hist:
            continue
        hdf = pd.DataFrame(hist)
        hdf['date'] = pd.to_datetime(hdf['timestamp'], unit='s').dt.normalize()
        daily = hdf.groupby('date')['price'].last()
        series_list.append({
            'prices': daily,
            'direction_sign': m.get('direction_sign', 0),
            'liquidity': max(1.0, float(m.get('liquidity', 0) or 0)),
            'quality': float(m.get('market_quality_score', 0)),
        })

    if not series_list:
        return empty

    weighted_sent = pd.Series(0.0, index=asset_index)
    total_weight = pd.Series(0.0, index=asset_index)
    net_sent = pd.Series(0.0, index=asset_index)
    mean_price_sum = pd.Series(0.0, index=asset_index)
    quality_sum = pd.Series(0.0, index=asset_index)
    count = pd.Series(0, index=asset_index)

    for s in series_list:
        aligned = s['prices'].reindex(asset_index).ffill()
        mask = aligned.notna()
        signed = s['direction_sign'] * aligned
        weighted_sent += signed.fillna(0) * s['liquidity']
        total_weight += mask.astype(float) * s['liquidity']
        net_sent += signed.fillna(0)
        mean_price_sum += aligned.fillna(0)
        quality_sum += mask.astype(float) * s['quality']
        count += mask.astype(int)

    result = pd.DataFrame(index=asset_index)
    result['pm_hist_liq_weighted_sentiment'] = (
        weighted_sent / total_weight.replace(0, np.nan)
    ).fillna(0)
    result['pm_hist_net_sentiment'] = (
        net_sent / count.replace(0, np.nan)
    ).fillna(0)
    result['pm_hist_mean_price'] = (
        mean_price_sum / count.replace(0, np.nan)
    ).fillna(0)
    result['pm_hist_market_quality'] = (
        quality_sum / count.replace(0, np.nan)
    ).fillna(0)
    return result


# =============================================================================
# THEME CLASSIFICATION
# =============================================================================

THEME_KEYWORDS_MAP = {
    'price':      ['above', 'below', 'hit', 'reach', '$', 'price'],
    'regulation': ['sec', 'regulation', 'ban', 'approval', 'approved', 'reserve'],
    'macro':      ['fed', 'rates', 'recession', 'inflation', 'treasury'],
    'adoption':   ['etf', 'fund', 'treasury', 'reserve', 'adoption'],
    'risk':       ['hack', 'collapse', 'bankruptcy', 'attack', 'outage'],
}


def infer_theme(text: str) -> str:
    """Classify a market question by dominant theme."""
    text = text.lower()
    scores = {theme: sum(kw in text for kw in keywords)
              for theme, keywords in THEME_KEYWORDS_MAP.items()}
    best = max(scores, key=scores.get) if scores else 'other'
    return best if scores.get(best, 0) > 0 else 'other'


# =============================================================================
# ORDER BOOK (CLOB) SNAPSHOTS
# =============================================================================

def normalize_book_levels(levels) -> list:
    """Extract price/size from both object and dict formats."""
    out = []
    for level in levels or []:
        price = getattr(level, 'price', None)
        size = getattr(level, 'size', None)
        if isinstance(level, dict):
            price = level.get('price', price)
            size = level.get('size', size)
        if price is not None:
            out.append({'price': float(price),
                        'size': float(size) if size is not None else float('nan')})
    return out


def fetch_order_book_snapshot(token_id: str, clob_host: str = None,
                              timeout: int = 8) -> dict:
    """
    Fetch live bid/ask prices from CLOB with multiple endpoint fallbacks.

    Returns:
        Dict with 'bids', 'asks', 'source' keys.
    """
    if not token_id:
        return {'bids': [], 'asks': [], 'source': 'no_token'}

    from config import POLYMARKET_CLOB_HOST
    host = (clob_host or POLYMARKET_CLOB_HOST).rstrip('/')

    attempts = [
        ('GET',  f'{host}/book', {'params': {'token_id': token_id}}),
        ('GET',  f'{host}/book', {'params': {'asset_id': token_id}}),
        ('POST', f'{host}/books', {'json': {'token_ids': [token_id]}}),
    ]
    for method, url, kwargs in attempts:
        try:
            resp = requests.request(method, url, timeout=timeout, **kwargs)
            if not resp.ok:
                continue
            data = resp.json()
            if isinstance(data, list):
                data = data[0] if data else {}
            if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
                data = data['data'][0] if data['data'] else {}
            bids = normalize_book_levels(data.get('bids', []))
            asks = normalize_book_levels(data.get('asks', []))
            if bids or asks:
                return {'bids': bids, 'asks': asks, 'source': f'rest:{method}'}
        except Exception:
            continue
    return {'bids': [], 'asks': [], 'source': 'unavailable'}


# =============================================================================
# WEIGHTED MEAN HELPER
# =============================================================================

def weighted_mean(series, weights):
    """NaN-safe liquidity-weighted average."""
    import pandas as pd
    series = pd.Series(series)
    weights = pd.Series(weights)
    mask = series.notna() & weights.notna()
    if mask.sum() == 0 or weights[mask].sum() == 0:
        return float('nan')
    return float(np.average(series[mask], weights=weights[mask]))
