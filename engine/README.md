
# YHack 2026: Crypto Trading Research System with Prediction Market Intelligence

## Overview

This project is a comprehensive research platform that combines **technical analysis** with **prediction market intelligence** to improve cryptocurrency trading decisions. It uses Polymarket data as an information layer to enhance trade selection, position sizing, risk management, and scenario analysis.

### Core Value Proposition

Rather than trading prediction contracts directly, this system leverages Polymarket as a **confidence and risk management layer** to validate and optimize trades in liquid crypto assets (BTC, ETH, SOL).

---

## Quick Start

### Prerequisites
- Python 3.9+
- Conda or pip
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/linmat1/YHack2026.git
cd YHack2026/engine

# Create and activate environment
conda create -n genAiEnv python=3.9
conda activate genAiEnv

# Install dependencies
pip install -r requirements.txt
```

### API Key Configuration

This project requires API keys for Polymarket and Groq. **Never commit your credentials to version control.**

1. **Copy the template:**
   ```bash
   cp .env.example credentials.env
   ```

2. **Edit `credentials.env` with your actual API keys:**
   ```
   POLY_API_KEY=your_polymarket_api_key
   POLY_SECRET=your_polymarket_secret
   POLY_PASSPHRASE=your_polymarket_passphrase
   GROQ_API_KEY=your_groq_api_key
   ```

3. **Get your API keys:**
   - **Polymarket**: Create an account at [polymarket.com](https://polymarket.com), then go to Account Settings → API to generate credentials
   - **Groq**: Sign up at [console.groq.com](https://console.groq.com) and create an API key

> ⚠️ **Security Note:** The `credentials.env` file is listed in `.gitignore` and will not be pushed to GitHub. Keep your keys safe!

### Running the Application

**NEXUS Terminal (Streamlit Dashboard):**
```bash
streamlit run app.py
```

**Research Notebook (Recommended for analysis):**
```bash
jupyter notebook yhack_prediction_market_dashboard_lab_iter_2.ipynb
```

**Run Full Scan Pipeline (CLI):**
```bash
python main.py
```

---

## Architecture

### Project Structure

```
YHack2026/
├── engine/
│   ├── app.py                      # NEXUS Terminal — main Streamlit application
│   ├── main.py                     # Orchestrator — runs full scan pipeline
│   ├── config.py                   # API keys, credentials (Groq, Polymarket)
│   ├── technical.py                # Technical analysis engine (RSI, MACD, BB, ATR)
│   ├── polymarket.py               # Polymarket API integration (Gamma + CLOB)
│   ├── news.py                     # News sentiment aggregation
│   ├── fusion.py                   # Signal fusion — merges tech + PM + news
│   ├── backtest.py                 # Dual-strategy backtester (Base vs PM-Enhanced)
│   ├── display.py                  # Console display utilities
│   ├── conversational_ai.py        # Experimental RAG pipeline (standalone)
│   ├── credentials.env             # API keys (git-ignored)
│   ├── requirements.txt            # Python dependencies
│   ├── Data/                       # Pre-computed CSV exports (64 files)
│   │   ├── iter2_portfolio_summary.csv
│   │   ├── iter2_{btc,eth,sol}_with_pm_summary.csv
│   │   ├── iter2_{asset}_fused_data.csv
│   │   ├── iter2_{asset}_with_pm_trades.csv
│   │   ├── iter2_{asset}_pm_snapshot.csv
│   │   ├── iter2_resolution_table.csv
│   │   ├── iter2_education.csv
│   │   ├── iter2_hedge_profiles.csv
│   │   └── iter2_probability_payoff.csv
│   └── __pycache__/
├── yhack_prediction_market_dashboard_lab.ipynb  # Research notebook
└── README.md
```

### NEXUS Terminal (app.py)

The Streamlit terminal features a cinematic UI with:
- **Landing Page** — Animated trader scene with typing keyboard, ticker tape, terminal boot sequence
- **Scan Engine** — Live technical + Polymarket + news fusion per asset
- **Oracle AI** — Floating chatbot panel powered by Groq (LLaMA 3.3 70B) with full CSV data context
- **Shutdown Sequence** — Fullscreen cinematic power-down overlay

### Oracle AI Architecture

Oracle uses direct context injection (not RAG) for maximum reliability:
1. `_load_csv_context(ticker)` loads curated summaries from 64 CSV files into the system prompt
2. Live scan data (price, RSI, MACD, Polymarket probs, news sentiment, fusion weights) is appended
3. Sent to Groq's `llama-3.3-70b-versatile` (128K context, OpenAI-compatible API)
4. Rendered in a `@st.fragment` floating panel — chat reruns independently without page shake

### Data Flow

1. **Market Data Acquisition**
   - Live crypto candles from Kraken (BTC-USD, ETH-USD, SOL-USD)
   - Technical indicators computed in-memory

2. **Polymarket Integration**
   - Market discovery via Gamma API
   - Live order-book snapshots via CLOB API
   - Historical token prices via CLOB `prices-history`

3. **Signal Fusion**
   - Technical analysis (RSI, MACD, Bollinger Bands, ATR)
   - Polymarket sentiment aggregation
   - Risk zone classification (Tradeable → Avoid)

4. **Strategy Execution**
   - Two parallel strategies: Base (technical only) vs. PM-Enhanced
   - Risk-based position sizing
   - Comprehensive trade simulation

---

## Strategy Logic

### Base Market Strategy

**Entry/Exit Signal:**
- Buy when: `Close > Rolling Mean`
- Sell when: `Close < Rolling Mean`

**Risk Management:**
```
Position Size = (Cash × Risk Per Trade) / Stop Distance
Max Position = Min(Calculated Size, Max Portfolio Exposure)
```

**Default Parameters:**
- Initial Capital: $10,000
- Risk Per Trade: 1.16%
- Stop Loss: 0.32%
- Take Profit: 0.99%
- Slippage: 0.05%
- Fee: 0.10%

**Exit Hierarchy:**
1. Stop-Loss (hard exit)
2. Take-Profit (hard exit)
3. Technical Signal (soft exit)
4. PM Defensive Exit (PM-Enhanced strategy only)

### Polymarket Integration

**Confidence Layers:**
- Confirmation Score – Agreement between technical signal and PM sentiment
- Conflict Penalty – Divergence between indicators
- Quality Score – Market depth and liquidity assessment
- Event Urgency – Days to resolution impact
- Spread Stress – Order-book stability metric

**Risk Zone Classification:**

| Zone | Action | Position Size |
|------|--------|---------------|
| Tradeable | Allow entry | Full size |
| Cautious | Allow entry | 50% size |
| High Risk | Consider skip | 25% size |
| Avoid | Block entry | No position |

---

## Research Notebook Architecture

The Jupyter notebook is organized into 15 sequential research cells:

1. Problem Framing
2. Config & Inputs
3. Market Data
4. Polymarket Live Pull
5. Question Tagging
6. Historical Alignment
7. Signal Fusion
8. Decision Dashboard
9. Scenario Analysis
10. Baseline Backtest
11. Trade Simulator
12. Payoff Explorer
13. Hedge Lab
14. Portfolio Diagnostics
15. CSV Export

### Generated Outputs

**Strategy Exports:**
- `without_pm_roundtrip.csv` – Control strategy trades
- `with_pm_roundtrip.csv` – PM-enhanced strategy trades

**Analysis Tables:**
- `market_data.csv` – Raw OHLCV
- `polymarket_board.csv` – Live market snapshots
- `tagged_polymarket.csv` – Sentiment classification
- `fused_data.csv` – Technical + PM combined

---

## Streamlit Dashboard

### Interface Tabs

| Tab | Purpose |
|-----|---------|
| **Overview** | Key metrics and strategy comparison |
| **Trade Engine** | Live signal generation and backtest |
| **Market Board** | Polymarket sentiment dashboard |
| **Payoff Studio** | Probability-weighted P&L explorer |
| **Timing + Portfolio** | Multi-asset timing and aggregation |
| **Education** | Diagnostic contribution analysis |

### Customizable Controls

- Asset selection (BTC, ETH, SOL)
- Date range and interval
- Strategy parameters (rolling window, stop-loss, take-profit)
- Polymarket settings (confidence thresholds, market count)
- Portfolio options (asset weights, hedge ratios)

---

## Configuration

### Environment Variables

Create `.env` from `.env.example`:

```bash
POLYMARKET_GAMMA_HOST=https://gamma-api.polymarket.com
POLYMARKET_CLOB_HOST=https://clob.polymarket.com
POLYMARKET_DATA_HOST=https://data-api.polymarket.com
POLYMARKET_API_KEY=<your-api-key>
POLYMARKET_SECRET=<your-secret>
POLYMARKET_PASSPHRASE=<your-passphrase>
```

**Note:** Analytics workflow is public-data-first. Private key not required for research.

---

## Known Limitations

- Question tagging is keyword-based, not semantic
- Public Polymarket data has limited historical depth
- BTC features stronger market coverage than altcoins
- Hedge lab and payoff studio are research visualizations, not production-grade
- Results depend on API availability and live market conditions

---

## Recommended Improvements

- [ ] Replace keyword tagging with semantic relevance scoring
- [ ] Reconstruct historical spread and liquidity data
- [ ] Optimize per-market selection for PM aggregation
- [ ] Test alternative technical entry models
- [ ] Add portfolio constraints and risk budgeting
- [ ] Expose CSV export from Streamlit dashboard

---

## Interpretation Guide

### Strategy Variants

**Without PM (Control):**
- Pure technical analysis
- Rolling-mean entry/exit
- Standard risk model
- No prediction market filtering

**With PM (Enhanced):**
- Same core trading engine
- PM sentiment confirms/discounts entries
- Dynamic position sizing
- Defensive early exits

### Reading Results

PM layer adds value when it:
- Improves final equity
- Reduces maximum drawdown
- Increases win rate
- Avoids false-positive entries

---

## Project Positioning

**Official Description:**

> A crypto trading research system that uses prediction-market intelligence to improve entry quality, position sizing, risk control, and scenario understanding.

---

## Support

For issues:
1. Check existing documentation
2. Review GitHub issues
3. Enable debug logging
4. Provide complete error traceback

---

## Contributing

- Create feature branches from `main`
- Write clear commit messages
- Include tests for new functionality
- Update documentation
- Submit pull requests with detailed descriptions

---

## License & Attribution

- **Project Lead:** sabya-chow
- **Frameworks:** Jupyter, Streamlit, Plotly
- **Data Sources:** Kraken, Polymarket
- **Last Updated:** 2026-03-29

---

**Status:** Active Development | **Version:** Iteration 2.0
```

---

## **Key Improvements:**

✅ **Fixed all broken file paths**  
✅ **Professional structure & formatting**  
✅ **Clear installation instructions**  
✅ **Removed personal file system paths**  
✅ **Added configuration guide**  
✅ **Better visual hierarchy (tables, code blocks)**  
✅ **Concise and scannable content**  
✅ **Enterprise-ready tone**  
✅ **Actionable sections**  

