"""
Display Module
==============
Formatted output for the decision dashboard.

Sections:
1. Technical Analysis - Price, indicators, signals
2. Polymarket - Prediction market sentiment
3. News - RSS feed sentiment
4. Fusion - Combined decision with confidence bars

Uses ASCII art for terminal compatibility.
"""

from polymarket import tag_direction
from news import sentiment_label


# =============================================================================
# MAIN DASHBOARD
# =============================================================================

def dashboard(
    asset: dict,
    tech: dict,
    pm: dict,
    news: dict,
    decision: dict,
    weekly_trend: str,
    btc_trend: str
):
    """
    Print the complete decision dashboard.
    
    Args:
        asset: Asset configuration dict
        tech: Technical analysis output
        pm: Polymarket aggregated data
        news: News aggregated data
        decision: Fusion engine output
        weekly_trend: Weekly timeframe trend
        btc_trend: BTC market leader trend
    """
    ticker = asset['ticker']
    
    # Header
    print(f"\n{'='*65}")
    print(f"  DECISION DASHBOARD: {ticker}")
    print(f"{'='*65}")
    
    # Sections
    _print_technical(asset, tech, weekly_trend)
    _print_polymarket(pm)
    _print_news(news)
    _print_fusion(decision, ticker, btc_trend)
    
    print(f"{'='*65}\n")


# =============================================================================
# SECTION PRINTERS
# =============================================================================

def _print_technical(asset: dict, tech: dict, weekly_trend: str):
    """Print technical analysis section."""
    print(f"\n[TECHNICAL] {asset['yf_symbol']}")
    print("-" * 65)
    
    if not tech:
        print("  No technical data available")
        return
    
    # Direction arrow
    if tech['direction'] == "Long":
        arrow = "▲"
    elif tech['direction'] == "Short":
        arrow = "▼"
    else:
        arrow = "—"
    
    # Price line
    print(f"  Price: ${tech['price']:,.2f} | "
          f"Trend: {tech['direction']} {arrow} | "
          f"RSI: {tech['rsi']:.0f}")
    
    # Indicators line
    print(f"  MACD: {tech['macd_hist']:+.2f} | "
          f"Vol: {tech['volatility']*100:.1f}% | "
          f"DD: {tech['drawdown']*100:.1f}%")
    
    # Score line
    print(f"  Technical Score: {tech['tech_score']:+.2f} | "
          f"Confidence: {tech['confidence']:.0f}%")
    
    # Weekly trend
    print(f"  Weekly Trend: {weekly_trend}")


def _print_polymarket(pm: dict):
    """Print Polymarket section."""
    print(f"\n[POLYMARKET] {pm['count']} relevant markets")
    print("-" * 65)
    
    if pm['count'] == 0:
        print("  No relevant markets found")
        return
    
    # Sentiment label
    sent_label = sentiment_label(pm['sentiment'])
    
    print(f"  Sentiment: {pm['sentiment']:+.2f} {sent_label} | "
          f"Quality: {pm['quality']*100:.0f}% | "
          f"Dispersion: {pm['dispersion']:.2f}")
    
    # Top markets
    for m in pm.get('markets', [])[:3]:
        question = m['question']
        if len(question) > 50:
            question = question[:50] + "..."
        
        direction = tag_direction(m['question']).upper()
        odds = f"YES: {m['yes_odds']*100:.0f}%" if m.get('yes_odds') else ""
        
        print(f"  • {question}")
        print(f"    {odds} | {direction}")


def _print_news(news: dict):
    """Print news sentiment section."""
    print(f"\n[NEWS] {news['count']} articles")
    print("-" * 65)
    
    if news['count'] == 0:
        print("  No relevant news found")
        return
    
    # Summary line
    sent_label = sentiment_label(news['sentiment'])
    print(f"  Sentiment: {news['sentiment']:+.2f} {sent_label} | "
          f"+:{news['bullish_count']} "
          f"=:{news['neutral_count']} "
          f"-:{news['bearish_count']}")
    
    # Top articles
    for a in news.get('articles', [])[:3]:
        s_label = sentiment_label(a['sentiment'])
        title = a['title']
        if len(title) > 45:
            title = title[:45] + "..."
        
        print(f"  {s_label} {a['sentiment']:+.2f} | {title}")


def _print_fusion(decision: dict, ticker: str, btc_trend: str):
    """Print fusion/decision section."""
    print(f"\n[FUSION]")
    print("-" * 65)
    
    # Agreement
    print(f"  Signal Agreement: {decision['agreement']['agreement']}")
    
    # BTC market leader (for altcoins)
    if ticker != "BTC":
        penalty_text = f" (penalty: -{decision['btc_penalty']})" if decision['btc_penalty'] > 0 else ""
        print(f"  BTC Market Leader: {btc_trend}{penalty_text}")
    
    # Timeframe confluence
    if decision['timeframe_boost'] > 0:
        print(f"  Timeframe Confluence: +{decision['timeframe_boost']} (Daily + Weekly align)")
    if decision['timeframe_penalty'] > 0:
        print(f"  Timeframe Conflict: -{decision['timeframe_penalty']} (Counter-trend)")
    
    # Confidence bars
    conf_filled = int(decision['confidence'] / 10)
    conf_bar = "█" * conf_filled + "░" * (10 - conf_filled)
    
    caut_filled = int(decision['caution'] / 10)
    caut_bar = "█" * caut_filled + "░" * (10 - caut_filled)
    
    print(f"\n  Final Confidence: {decision['confidence']:5.1f}% {conf_bar}")
    print(f"  Caution Score:    {decision['caution']:5.1f}% {caut_bar}")
    print(f"  Risk Zone: {decision['risk_zone']}")
    print(f"  Position Size: {decision['position_size']:.0%}")
    
    # Final action box
    print(f"\n  ╔{'═'*40}╗")
    print(f"  ║  >>> RECOMMENDED ACTION: {decision['action']:^10} <<<  ║")
    print(f"  ║      Confidence: {decision['confidence']:.0f}/100                  ║")
    
    size_text = f"{decision['position_size']:.0%} of normal position"
    print(f"  ║      Size: {size_text:<25}  ║")
    print(f"  ╚{'═'*40}╝")


# =============================================================================
# PROGRESS INDICATOR
# =============================================================================

def progress(step: int, total: int, message: str):
    """
    Print a progress step.
    
    Args:
        step: Current step number
        total: Total steps
        message: Description of current step
    """
    print(f"  [{step}/{total}] {message}")
