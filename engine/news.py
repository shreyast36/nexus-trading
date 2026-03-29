"""
News Sentiment Engine
=====================
Fetches crypto news from RSS feeds and analyzes sentiment using VADER.

Features:
- Parallel RSS feed fetching (10 workers)
- Strict keyword matching in titles
- VADER sentiment analysis (-1 to +1 compound score)
- Aggregation into bullish/bearish/neutral counts

Sources are categorized:
- Retail: CoinTelegraph, CoinDesk, Decrypt
- Institutional: The Block, Bankless
- Security: Rekt News, Chainalysis
- Regulatory: Unchained, CoinLaw
- DeFi: The Defiant
- General: Wired, TechCrunch
"""

import re
import numpy as np
import requests as _req
import feedparser
from concurrent.futures import ThreadPoolExecutor, as_completed
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import RSS_FEEDS, CRYPTO_CONTEXT_WORDS, AMBIGUOUS_KEYWORDS


# =============================================================================
# SENTIMENT ANALYZER (singleton for efficiency)
# =============================================================================

_analyzer = SentimentIntensityAnalyzer()


# =============================================================================
# FEED FETCHING
# =============================================================================

def _fetch_single_feed(args: tuple) -> list:
    """
    Fetch articles from a single RSS feed.
    
    Args:
        args: Tuple of (url, category, keywords)
        
    Returns:
        List of article dicts with title, source, category, sentiment
    """
    url, category, keywords = args
    articles = []
    
    try:
        resp = _req.get(url, timeout=4, headers={"User-Agent": "Mozilla/5.0"})
        feed = feedparser.parse(resp.content)
        
        for entry in feed.entries[:10]:  # Max 10 per feed
            title = entry.get('title', '')
            summary = entry.get('summary', '')
            title_lower = title.lower()
            
            # ─────────────────────────────────────────────────────────────────
            # Strict keyword matching: word-boundary match in title.
            # For short/ambiguous tickers ("sol", "op", etc.) we also
            # require a crypto-context word in the title+summary so
            # "solar panels" or "opinion" doesn't slip through.
            # ─────────────────────────────────────────────────────────────────
            full_text_lower = f"{title_lower} {summary.lower()}"
            is_relevant = False
            for kw in keywords:
                if not re.search(rf'\b{re.escape(kw)}\b', title_lower):
                    continue
                # Ambiguous short tickers need crypto context nearby
                if kw.lower() in AMBIGUOUS_KEYWORDS:
                    if not any(ctx in full_text_lower for ctx in CRYPTO_CONTEXT_WORDS):
                        continue
                is_relevant = True
                break

            if not is_relevant:
                continue
            
            # ─────────────────────────────────────────────────────────────────
            # VADER Sentiment Analysis
            # Compound score: -1 (negative) to +1 (positive)
            # ─────────────────────────────────────────────────────────────────
            text = f"{title} {summary}"
            sentiment = _analyzer.polarity_scores(text)['compound']
            
            articles.append({
                "title": title,
                "source": feed.feed.get('title', 'Unknown'),
                "category": category,
                "sentiment": sentiment,
            })
            
    except Exception:
        pass  # Silently skip failed feeds
    
    return articles


def fetch_news(keywords: list, max_articles: int = 20) -> list:
    """
    Fetch news articles matching keywords from all RSS feeds.
    
    Uses parallel execution for speed (10 workers).
    
    Args:
        keywords: List of search terms
        max_articles: Maximum articles to return
        
    Returns:
        List of articles sorted by sentiment (positive first)
    """
    # Build task list for parallel execution
    tasks = []
    for category, feeds in RSS_FEEDS.items():
        for url in feeds:
            tasks.append((url, category, keywords))
    
    # Fetch all feeds in parallel
    articles = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(_fetch_single_feed, task) for task in tasks]
        
        for future in as_completed(futures):
            articles.extend(future.result())
    
    # Sort by sentiment (most positive first)
    articles.sort(key=lambda x: x['sentiment'], reverse=True)
    
    return articles[:max_articles]


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate(articles: list) -> dict:
    """
    Aggregate news articles into summary features.
    
    Returns:
        count: Total articles found
        sentiment: Average sentiment (-1 to +1)
        bullish_count: Articles with sentiment >= 0.05
        bearish_count: Articles with sentiment <= -0.05
        neutral_count: Articles with -0.05 < sentiment < 0.05
        dispersion: Standard deviation (disagreement)
        articles: Top articles for display
    """
    if not articles:
        return {
            "count": 0,
            "sentiment": 0.0,
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "dispersion": 0.0,
            "articles": [],
        }
    
    sentiments = [a['sentiment'] for a in articles]
    
    # Count by category
    bullish = sum(1 for s in sentiments if s >= 0.05)
    bearish = sum(1 for s in sentiments if s <= -0.05)
    neutral = len(sentiments) - bullish - bearish
    
    return {
        "count": len(articles),
        "sentiment": float(np.mean(sentiments)),
        "bullish_count": bullish,
        "bearish_count": bearish,
        "neutral_count": neutral,
        "dispersion": float(np.std(sentiments)) if len(sentiments) > 1 else 0.0,
        "articles": articles[:5],  # Top 5 for display
    }


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def analyze(keywords: list) -> dict:
    """
    Full news analysis pipeline.
    
    Args:
        keywords: Search terms for the asset
        
    Returns:
        Aggregated news features
    """
    articles = fetch_news(keywords)
    return aggregate(articles)


# =============================================================================
# HELPERS
# =============================================================================

def sentiment_label(score: float) -> str:
    """
    Convert sentiment score to display label.
    
    Args:
        score: Sentiment value (-1 to +1)
        
    Returns:
        "[+]" for positive, "[-]" for negative, "[=]" for neutral
    """
    if score >= 0.05:
        return "[+]"
    elif score <= -0.05:
        return "[-]"
    return "[=]"
