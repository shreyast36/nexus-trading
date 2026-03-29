"""
Yhack Predictive Market Decision Engine
=======================================
Entry point that orchestrates all modules.

Usage:
    python main.py
    
Then enter an asset like: btc, eth, sol, bitcoin, ethereum, etc.

Modules:
    config.py    - Configuration and thresholds
    technical.py - yfinance price data and indicators
    polymarket.py - Prediction market sentiment
    news.py      - RSS news sentiment
    fusion.py    - Signal combination logic
    display.py   - Formatted output

NO AI PREDICTIONS - all rules are transparent and auditable.
"""

# Local modules
from config import resolve_asset
import technical
import polymarket
import news
import fusion
import display


def main():
    """Main entry point."""
    
    # =========================================================================
    # USER INPUT
    # =========================================================================
    
    query = input("Asset (BTC/ETH/SOL or keyword): ").strip()
    
    if not query:
        print("No asset specified.")
        return
    
    print(f"\nAnalyzing {query.upper()}...")
    print("─" * 40)
    
    # =========================================================================
    # RESOLVE ASSET
    # =========================================================================
    
    asset = resolve_asset(query)
    is_btc = asset['ticker'] == "BTC"
    
    # =========================================================================
    # 1. TECHNICAL ANALYSIS
    # =========================================================================
    
    display.progress(1, 5, "Fetching price data...")
    price_df = technical.fetch_price_data(asset['yf_symbol'])
    price_df = technical.compute_indicators(price_df)
    tech_data = technical.compute_signals(price_df)
    
    # =========================================================================
    # 2. WEEKLY TREND (Multi-Timeframe Confluence)
    # =========================================================================
    
    display.progress(2, 5, "Checking weekly trend...")
    weekly_trend = technical.get_weekly_trend(asset['yf_symbol'])
    
    # =========================================================================
    # 3. BTC MARKET LEADER (for altcoins)
    # =========================================================================
    
    if not is_btc:
        display.progress(3, 5, "Checking BTC market leader...")
        btc_trend = technical.get_btc_trend()
    else:
        display.progress(3, 5, "BTC market leader: skipped (analyzing BTC)")
        btc_trend = "N/A"
    
    # =========================================================================
    # 4. POLYMARKET DATA
    # =========================================================================
    
    display.progress(4, 5, "Fetching Polymarket data...")
    pm_data = polymarket.analyze(asset['keywords'], asset_ticker=asset['ticker'])
    
    # =========================================================================
    # 5. NEWS SENTIMENT
    # =========================================================================
    
    display.progress(5, 5, "Fetching news sentiment...")
    news_data = news.analyze(asset['keywords'])
    
    # =========================================================================
    # FUSION
    # =========================================================================
    
    decision = fusion.fuse(
        tech_data=tech_data,
        pm_data=pm_data,
        news_data=news_data,
        is_btc=is_btc,
        btc_trend=btc_trend,
        weekly_trend=weekly_trend
    )
    
    # =========================================================================
    # DISPLAY
    # =========================================================================
    
    display.dashboard(
        asset=asset,
        tech=tech_data,
        pm=pm_data,
        news=news_data,
        decision=decision,
        weekly_trend=weekly_trend,
        btc_trend=btc_trend
    )


if __name__ == "__main__":
    main()
