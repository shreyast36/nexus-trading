"""
NEXUS Trading Terminal
======================
High-tech crypto prediction market terminal with a modern dark UI.

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from config import resolve_asset, ASSETS, THRESHOLDS, GROQ_API_KEY
import technical
import polymarket
import news
import fusion
import backtest

# ── Groq LLM Integration (free, fast, OpenAI-compatible) ─────────────────────
from openai import OpenAI
from pathlib import Path

DATA_DIR = Path(__file__).parent / "Data"

@st.cache_data(ttl=300)
def _load_csv_context(ticker: str | None = None):
    """Load key CSVs from Data/ and return a compact text summary for the LLM."""
    sections = []

    # ── Portfolio-level summaries ──
    try:
        df = pd.read_csv(DATA_DIR / "iter2_portfolio_summary.csv")
        sections.append("PORTFOLIO SUMMARY:\n" + df.to_string(index=False))
    except Exception:
        pass

    # ── Per-asset backtest summaries (with PM) ──
    for sym in ("btc", "eth", "sol"):
        try:
            df = pd.read_csv(DATA_DIR / f"iter2_{sym}_with_pm_summary.csv")
            sections.append(f"{sym.upper()} BACKTEST (With PM):\n" + df.to_string(index=False))
        except Exception:
            pass

    # ── Resolution table ──
    try:
        df = pd.read_csv(DATA_DIR / "iter2_resolution_table.csv")
        sections.append("RESOLUTION TABLE:\n" + df.to_string(index=False))
    except Exception:
        pass

    # ── Education / component breakdown ──
    try:
        df = pd.read_csv(DATA_DIR / "iter2_education.csv")
        sections.append("CONFIDENCE COMPONENTS:\n" + df.to_string(index=False))
    except Exception:
        pass

    # ── Hedge profiles (key rows only) ──
    try:
        df = pd.read_csv(DATA_DIR / "iter2_hedge_profiles.csv")
        key = df[df["AssetReturn"].isin([-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20])]
        sections.append("HEDGE PROFILE SCENARIOS:\n" + key.to_string(index=False))
    except Exception:
        pass

    # ── Asset-specific deep data (if a ticker is active) ──
    if ticker:
        sym = ticker.lower().replace("-usd", "").replace("usdt", "")

        # Recent fused data (last 10 trading days)
        try:
            df = pd.read_csv(DATA_DIR / f"iter2_{sym}_fused_data.csv")
            tail = df.tail(10)[["Date", "Close", "RSI", "MACD_Hist", "BB_Z",
                                "Volatility_20D", "Drawdown", "FinalAction",
                                "FinalConfidence", "RiskZone", "PositionSize",
                                "CautionScore", "PMAgreement"]].copy()
            for c in tail.select_dtypes("float").columns:
                tail[c] = tail[c].round(4)
            sections.append(f"{sym.upper()} RECENT SIGNALS (last 10 days):\n" + tail.to_string(index=False))
        except Exception:
            pass

        # Recent trades
        try:
            df = pd.read_csv(DATA_DIR / f"iter2_{sym}_with_pm_trades.csv")
            tail = df.tail(15)[["timestamp", "side", "reason", "fill_price",
                                "qty", "notional", "cash_after", "pm_multiplier"]].copy()
            for c in tail.select_dtypes("float").columns:
                tail[c] = tail[c].round(4)
            sections.append(f"{sym.upper()} RECENT TRADES (last 15):\n" + tail.to_string(index=False))
        except Exception:
            pass

        # PM snapshot
        try:
            df = pd.read_csv(DATA_DIR / f"iter2_{sym}_pm_snapshot.csv")
            sections.append(f"{sym.upper()} POLYMARKET SNAPSHOT:\n" + df.to_string(index=False))
        except Exception:
            pass

        # Probability / payoff scenarios for this asset
        try:
            df = pd.read_csv(DATA_DIR / "iter2_probability_payoff.csv")
            df = df[df["Asset"] == sym.upper()]
            if not df.empty:
                for c in df.select_dtypes("float").columns:
                    df[c] = df[c].round(4)
                sections.append(f"{sym.upper()} PROBABILITY-PAYOFF SCENARIOS:\n" + df.to_string(index=False))
        except Exception:
            pass

    return "\n\n".join(sections)


def _ask_oracle(messages, scan_data=None):
    """Send messages to Groq LLM with scan data + CSV context."""
    api_key = GROQ_API_KEY
    if not api_key:
        return "⚠ GROQ_API_KEY not set in credentials.env. Get a free key at https://console.groq.com"

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    # Determine active ticker for CSV loading
    ticker = None
    if scan_data:
        ticker = (scan_data.get('asset', {}) or {}).get('ticker')

    # Load rich CSV context
    csv_context = _load_csv_context(ticker)

    system = """You are ORACLE AI — the expert analyst powering the NEXUS Prediction-Market Terminal.
You have access to comprehensive backtest results, live technical signals, Polymarket data,
trade histories, hedge profiles, and probability-payoff scenarios from the engine's Data pipeline.

RULES:
• Keep responses SHORT and conversational — 2-4 sentences by default.
• Do NOT dump raw metrics or data tables unless the user explicitly asks for them.
• Answer the specific question asked. Do not volunteer extra info, stats, or disclaimers.
• Only cite exact numbers when they directly answer what the user asked.
• If the user asks to elaborate or wants more detail, THEN provide a thorough data-rich breakdown.
• Never start a response by listing key metrics — get straight to the insight.
• If something isn't in the data, say so briefly — never fabricate numbers.
• Sound like a sharp trading desk analyst, not a report generator."""

    # Append real-time scan data
    if scan_data:
        d = scan_data
        tech = d.get('tech', {}) or {}
        pm = d.get('pm', {}) or {}
        nws = d.get('news', {}) or {}
        dec = d.get('decision', {}) or {}
        asset = d.get('asset', {}) or {}

        system += f"""

LIVE SCAN DATA:
Asset: {asset.get('ticker','N/A')} | Price: ${tech.get('price',0):,.2f}
Decision: {dec.get('action','N/A')} | Confidence: {dec.get('confidence',0):.1f}% | Caution: {dec.get('caution',0):.1f}
Position Size: {dec.get('position_size',0)*100:.0f}% | Risk Zone: {dec.get('risk_zone','N/A')}

Technical: RSI={tech.get('rsi',0):.1f} | MACD Signal={tech.get('macd_signal','N/A')} | BB Position={tech.get('bb_position','N/A')} | Trend={tech.get('trend','N/A')}
SMA20={tech.get('sma_20',0):.2f} | Volatility={tech.get('volatility',0):.4f}

Polymarket: Prob Up={pm.get('prob_up',0):.2f} | Prob Down={pm.get('prob_down',0):.2f} | PM Signal={pm.get('pm_signal','N/A')} | Spread={pm.get('spread',0):.3f}

News Sentiment: Score={nws.get('sentiment_score',0):.2f} | Label={nws.get('sentiment_label','N/A')} | Headlines={nws.get('num_articles',0)}

Fusion Weights: Technical={dec.get('weights',{}).get('technical',0):.0%} | Polymarket={dec.get('weights',{}).get('polymarket',0):.0%} | News={dec.get('weights',{}).get('news',0):.0%}"""

    # Append CSV-backed historical data
    if csv_context:
        system += "\n\nHISTORICAL DATA FROM ENGINE:\n" + csv_context

    api_messages = [{"role": "system", "content": system}]
    for m in messages:
        api_messages.append({"role": m["role"], "content": m["content"]})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=api_messages,
        max_tokens=1024,
        temperature=0.7,
    )
    return response.choices[0].message.content

# ── Page config ──────────────────────────────────────────────────────────────
_entered = st.session_state.get('entered', False)
st.set_page_config(page_title="NEXUS TERMINAL", page_icon="N", layout="wide",
                   initial_sidebar_state="expanded" if _entered is True else "collapsed")

# ── Theme constants ──────────────────────────────────────────────────────────
CYAN   = "#00e5ff"
GREEN  = "#00ffaa"
RED    = "#ff3d5a"
AMBER  = "#ffab00"
PURPLE = "#b388ff"
DIM    = "#5a6272"
BG     = "#05080f"
PANEL  = "#0b1120"
BORDER = "#1a2744"

# ── Modern CSS ───────────────────────────────────────────────────────────────
st.markdown(\"\"\"<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">\"\"\", unsafe_allow_html=True)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ── base ── */
.stApp {{ background: {BG}; }}
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #080e1a 0%, #0a1628 100%);
    border-right: 1px solid {BORDER};
}}
.block-container {{ padding: 0.6rem 1.2rem 0 1.2rem; }}

/* ── typography ── */
html, body, [class*="css"], p, span, label, li, td, th {{
    font-family: 'JetBrains Mono', 'Consolas', monospace !important;
    color: #c0c8d8;
}}
h1,h2,h3,h4,h5,h6 {{
    font-family: 'Inter', sans-serif !important;
    color: #eef2f7 !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 700;
}}

/* ── glassmorphism panels ── */
.nx-panel {{
    background: linear-gradient(135deg, rgba(11,17,32,0.95) 0%, rgba(15,25,50,0.9) 100%);
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 8px;
    backdrop-filter: blur(12px);
    position: relative;
    overflow: hidden;
}}

/* ── staggered fly-in for dashboard sections ── */
@keyframes flyInUp {{
  0%   {{ opacity:0; transform:translateY(60px) scale(0.92); }}
  60%  {{ opacity:1; }}
  80%  {{ transform:translateY(-4px) scale(1.01); }}
  100% {{ opacity:1; transform:translateY(0) scale(1); }}
}}
@keyframes flyInLeft {{
  0%   {{ opacity:0; transform:translateX(-50px) scale(0.93); }}
  60%  {{ opacity:1; }}
  80%  {{ transform:translateX(3px) scale(1.005); }}
  100% {{ opacity:1; transform:translateX(0) scale(1); }}
}}
@keyframes flyInRight {{
  0%   {{ opacity:0; transform:translateX(50px) scale(0.93); }}
  60%  {{ opacity:1; }}
  80%  {{ transform:translateX(-3px) scale(1.005); }}
  100% {{ opacity:1; transform:translateX(0) scale(1); }}
}}
.nx-cascade-1 {{ animation: flyInLeft 0.5s cubic-bezier(0.22,1,0.36,1) 0.05s both; }}
.nx-cascade-2 {{ animation: flyInUp 0.5s cubic-bezier(0.22,1,0.36,1) 0.12s both; }}
.nx-cascade-3 {{ animation: flyInUp 0.5s cubic-bezier(0.22,1,0.36,1) 0.19s both; }}
.nx-cascade-4 {{ animation: flyInRight 0.5s cubic-bezier(0.22,1,0.36,1) 0.26s both; }}
.nx-cascade-5 {{ animation: flyInLeft 0.5s cubic-bezier(0.22,1,0.36,1) 0.34s both; }}
.nx-cascade-6 {{ animation: flyInUp 0.5s cubic-bezier(0.22,1,0.36,1) 0.42s both; }}
.nx-cascade-7 {{ animation: flyInRight 0.5s cubic-bezier(0.22,1,0.36,1) 0.50s both; }}
@keyframes tickerSlide {{
  from {{ opacity:0; transform:translateY(-15px); }}
  to {{ opacity:1; transform:translateY(0); }}
}}
.nx-ticker {{ animation: tickerSlide 0.4s ease both; }}
.nx-panel::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, {CYAN}, {PURPLE}, {CYAN});
    border-radius: 8px 8px 0 0;
}}
.nx-panel-title {{
    color: {CYAN};
    font-family: 'Inter', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(26,39,68,0.6);
    display: flex;
    align-items: center;
    gap: 6px;
}}
.nx-panel-title::before {{
    content: '';
    font-size: 0.8rem;
    color: {CYAN};
}}
.nx-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    font-size: 1.1rem;
}}
.nx-label {{ color: {DIM}; font-size: 1.05rem; }}
.nx-val   {{ color: #eef2f7; font-weight: 600; }}
.nx-cyan  {{ color: {CYAN}; font-weight: 700; }}
.nx-green {{ color: {GREEN}; font-weight: 700; }}
.nx-red   {{ color: {RED}; font-weight: 700; }}
.nx-amber {{ color: {AMBER}; font-weight: 700; }}
.nx-big   {{ font-size: 2.2rem; font-weight: 800; line-height: 1; }}

/* ── action banners ── */
.nx-action-long {{
    background: linear-gradient(135deg, rgba(0,60,30,0.5), rgba(0,40,20,0.3));
    border: 1px solid {GREEN};
    border-radius: 8px;
    text-align: center;
    padding: 18px 14px;
    box-shadow: 0 0 20px rgba(0,255,170,0.08), inset 0 0 30px rgba(0,255,170,0.03);
}}
.nx-action-short {{
    background: linear-gradient(135deg, rgba(60,10,15,0.5), rgba(40,5,10,0.3));
    border: 1px solid {RED};
    border-radius: 8px;
    text-align: center;
    padding: 18px 14px;
    box-shadow: 0 0 20px rgba(255,61,90,0.08), inset 0 0 30px rgba(255,61,90,0.03);
}}
.nx-action-flat {{
    background: linear-gradient(135deg, rgba(20,25,40,0.5), rgba(15,20,35,0.3));
    border: 1px solid {DIM};
    border-radius: 8px;
    text-align: center;
    padding: 18px 14px;
    box-shadow: 0 0 20px rgba(90,98,114,0.05);
}}

/* ── ticker bar ── */
.nx-ticker {{
    background: linear-gradient(90deg, rgba(11,17,32,0.9), rgba(15,25,50,0.8));
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 10px 16px;
    font-size: 1.08rem;
    color: {CYAN};
    letter-spacing: 0.5px;
    white-space: nowrap;
    overflow-x: auto;
    margin-bottom: 10px;
}}
.nx-ticker b {{ color: #fff; font-size: 1.15rem; }}
.nx-ticker .sep {{ color: #1a2744; margin: 0 6px; }}

/* ── news items ── */
.nx-news-item {{
    border-bottom: 1px solid rgba(26,39,68,0.4);
    padding: 8px 0;
    font-size: 1.08rem;
    transition: background 0.2s;
}}
.nx-news-item:hover {{
    background: rgba(0,229,255,0.03);
}}

/* ── live dot ── */
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.3; }}
}}
.nx-live {{
    display: inline-block;
    width: 6px; height: 6px;
    background: {GREEN};
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
    box-shadow: 0 0 6px {GREEN};
}}

/* ── score bars ── */
.nx-bar-wrap {{
    background: rgba(26,39,68,0.3);
    border-radius: 4px;
    height: 8px;
    width: 100%;
    overflow: hidden;
    margin-top: 2px;
}}
.nx-bar {{
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
}}

/* ── metrics ── */
[data-testid="stMetricValue"] {{
    color: {CYAN} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 2rem !important;
}}
[data-testid="stMetricLabel"] {{ color: {DIM} !important; font-size: 1.05rem !important; }}
[data-testid="stMetricDelta"] {{ font-family: 'JetBrains Mono', monospace !important; }}

/* ── streamlit button ── */
[data-testid="stButton"] > button {{
    background: linear-gradient(135deg, {CYAN}, #0091ea) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    letter-spacing: 1px;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.3s;
}}
[data-testid="stButton"] > button:hover {{
    box-shadow: 0 0 20px rgba(0,229,255,0.3) !important;
    transform: translateY(-1px);
}}

/* ── selectbox ── */
[data-testid="stSelectbox"] label {{ color: {DIM} !important; font-size: 1.05rem; letter-spacing: 1.5px; text-transform: uppercase; }}

/* ── hide chrome ── */
#MainMenu {{visibility:hidden;}} footer {{visibility:hidden;}} .stDeployButton {{display:none;}}

/* ── scrollbar ── */
::-webkit-scrollbar {{ width:5px; }}
::-webkit-scrollbar-track {{ background:{BG}; }}
::-webkit-scrollbar-thumb {{ background: linear-gradient({CYAN}, {PURPLE}); border-radius:4px; }}

/* ── ORACLE AI Chat Interface ── */
@keyframes oracleGlow {{
  0%, 100% {{ box-shadow: 0 0 20px rgba(179,136,255,0.3), 0 0 40px rgba(0,229,255,0.1); }}
  50% {{ box-shadow: 0 0 30px rgba(179,136,255,0.5), 0 0 60px rgba(0,229,255,0.2); }}
}}
@keyframes messageSlide {{
  from {{ opacity:0; transform:translateY(10px); }}
  to {{ opacity:1; transform:translateY(0); }}
}}
@keyframes typingPulse {{
  0%, 100% {{ opacity:0.3; }}
  50% {{ opacity:1; }}
}}
@keyframes oracleFloat {{
  0%, 100% {{ transform: translateY(0); }}
  50% {{ transform: translateY(-3px); }}
}}
.oracle-chat-container {{
    background: linear-gradient(180deg, rgba(11,17,32,0.98) 0%, rgba(8,14,26,0.99) 100%);
    border: 1px solid {BORDER};
    border-radius: 12px;
    overflow: hidden;
    animation: oracleGlow 3s ease-in-out infinite;
}}
.oracle-header {{
    background: linear-gradient(90deg, rgba(179,136,255,0.15), rgba(0,229,255,0.1));
    border-bottom: 1px solid {BORDER};
    padding: 14px 18px;
    display: flex;
    align-items: center;
    gap: 12px;
}}
.oracle-avatar {{
    width: 38px; height: 38px;
    background: linear-gradient(135deg, {PURPLE}, {CYAN});
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    animation: oracleFloat 2s ease-in-out infinite;
    box-shadow: 0 0 15px rgba(179,136,255,0.4);
}}
.oracle-title {{
    font-family: 'Inter', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: {PURPLE};
    letter-spacing: 2px;
    text-transform: uppercase;
}}
.oracle-subtitle {{
    font-size: 0.75rem;
    color: {DIM};
    letter-spacing: 1px;
}}
.oracle-messages {{
    max-height: 400px;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
}}
.oracle-msg {{
    animation: messageSlide 0.3s ease both;
    max-width: 85%;
    padding: 12px 16px;
    border-radius: 12px;
    font-size: 0.95rem;
    line-height: 1.5;
}}
.oracle-msg-user {{
    background: linear-gradient(135deg, rgba(0,229,255,0.15), rgba(0,229,255,0.08));
    border: 1px solid rgba(0,229,255,0.3);
    align-self: flex-end;
    color: #eef2f7;
}}
.oracle-msg-ai {{
    background: linear-gradient(135deg, rgba(179,136,255,0.12), rgba(179,136,255,0.06));
    border: 1px solid rgba(179,136,255,0.25);
    align-self: flex-start;
    color: #c0c8d8;
}}
.oracle-msg-ai::before {{
    content: '';
}}
.oracle-typing {{
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 12px 16px;
    color: {DIM};
    font-size: 0.85rem;
}}
.oracle-typing-dot {{
    width: 6px; height: 6px;
    background: {PURPLE};
    border-radius: 50%;
    animation: typingPulse 1s ease-in-out infinite;
}}
.oracle-typing-dot:nth-child(2) {{ animation-delay: 0.2s; }}
.oracle-typing-dot:nth-child(3) {{ animation-delay: 0.4s; }}
.oracle-input-area {{
    border-top: 1px solid {BORDER};
    padding: 14px;
    background: rgba(5,8,15,0.6);
}}
.oracle-btn {{
    background: linear-gradient(135deg, {PURPLE}, {CYAN}) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    letter-spacing: 1px;
    transition: all 0.3s;
    cursor: pointer;
}}
.oracle-btn:hover {{
    box-shadow: 0 0 25px rgba(179,136,255,0.4) !important;
    transform: scale(1.02);
}}
.oracle-status {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.8rem;
    color: {DIM};
    padding: 6px 0;
}}
.oracle-status-dot {{
    width: 6px; height: 6px;
    background: {GREEN};
    border-radius: 50%;
    box-shadow: 0 0 8px {GREEN};
    animation: pulse 2s infinite;
}}

/* ── Streamlit expander title fix ── */
[data-testid="stExpander"] summary span {{
    letter-spacing: 1.5px !important;
    word-spacing: 4px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    color: {CYAN} !important;
    text-transform: uppercase !important;
}}
[data-testid="stExpander"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    background: linear-gradient(135deg, rgba(11,17,32,0.95), rgba(15,25,50,0.9)) !important;
}}
[data-testid="stExpander"] summary {{
    padding: 12px 16px !important;
}}

/* ── Mobile responsive ── */
@media (max-width: 768px) {{
    .block-container {{ padding: 0.3rem 0.5rem 0 0.5rem !important; }}
    .nx-ticker {{
        flex-direction: column !important;
        gap: 8px !important;
        padding: 10px 12px !important;
        white-space: normal !important;
    }}
    .nx-ticker > div {{
        flex-wrap: wrap !important;
        justify-content: center !important;
        gap: 8px !important;
    }}
    .nx-panel {{
        padding: 10px 10px !important;
    }}
    .nx-panel-title {{
        font-size: 0.85rem !important;
        letter-spacing: 1.5px !important;
    }}
    .nx-row {{
        font-size: 0.9rem !important;
        flex-wrap: wrap !important;
        gap: 4px !important;
    }}
    .nx-label {{ font-size: 0.85rem !important; }}
    .nx-big {{ font-size: 1.6rem !important; }}
    .nx-news-item {{ font-size: 0.9rem !important; }}
    [data-testid="stMetricValue"] {{ font-size: 1.4rem !important; }}
    h1 {{ font-size: 1.2rem !important; letter-spacing: 1px !important; }}
    h2 {{ font-size: 1rem !important; }}
    h3 {{ font-size: 0.9rem !important; }}
    [data-testid="stExpander"] summary span {{
        font-size: 0.8rem !important;
        letter-spacing: 1px !important;
    }}
    .nx-action-long, .nx-action-short, .nx-action-flat {{
        padding: 12px 8px !important;
    }}
    /* Oracle chat floating panel - full width on mobile */
    [data-testid="stVerticalBlock"][style*="position:fixed"] {{
        width: calc(100vw - 20px) !important;
        left: 10px !important;
        right: 10px !important;
        bottom: 10px !important;
        max-height: 70vh !important;
    }}
}}

@media (max-width: 480px) {{
    .block-container {{ padding: 0.2rem 0.3rem 0 0.3rem !important; }}
    .nx-ticker > div:last-child {{
        gap: 10px !important;
    }}
    .nx-panel {{ padding: 8px 8px !important; }}
    .nx-big {{ font-size: 1.3rem !important; }}
    [data-testid="stMetricValue"] {{ font-size: 1.2rem !important; }}
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE — shown before user enters the terminal
# ══════════════════════════════════════════════════════════════════════════════
if 'entered' not in st.session_state:
    st.session_state.entered = False

if st.session_state.entered is False:
    _all_tickers = sorted(set(v['ticker'] for v in ASSETS.values()))

    # Hide sidebar + kill Streamlit padding for full-page landing
    st.markdown(f"""
<style>
[data-testid="stSidebar"]{{display:none;}}
[data-testid="stAppViewContainer"]>div:first-child{{padding-top:0;}}
.block-container{{padding-top:0!important;padding-bottom:0!important;max-width:100%!important;}}
[data-testid="stHeader"]{{display:none;}}
[data-testid="stToolbar"]{{display:none;}}
.stApp>header{{display:none;}}
html,body,[data-testid="stAppViewContainer"]{{overflow-x:hidden;-webkit-overflow-scrolling:touch;}}
/* Mobile landing page fixes */
@media (max-width: 768px) {{
    .trader-scene {{ transform:scale(0.55) !important; transform-origin:top center; height:220px !important; margin-bottom:-60px; }}
    .trader-scene svg {{ max-width:100vw; }}
}}
@media (max-width: 480px) {{
    .trader-scene {{ transform:scale(0.4) !important; height:180px !important; margin-bottom:-80px; }}
}}
</style>""", unsafe_allow_html=True)

    # ── Animated trader scene ────────────────────────────────────────────────
    st.markdown(f"""
<style>
@keyframes tickerScroll {{
  0% {{ transform: translateX(0); }}
  100% {{ transform: translateX(-50%); }}
}}
@keyframes fadeInUp {{
  from {{ opacity:0; transform:translateY(30px); }}
  to {{ opacity:1; transform:translateY(0); }}
}}
@keyframes heroGlow {{
  0%,100% {{ text-shadow:0 0 40px rgba(0,229,255,0.3),0 0 80px rgba(0,229,255,0.1); }}
  50% {{ text-shadow:0 0 60px rgba(0,229,255,0.5),0 0 120px rgba(0,229,255,0.2); }}
}}
@keyframes candleUp {{
  0% {{ height:0; opacity:0; }}
  100% {{ opacity:1; }}
}}
@keyframes screenFlicker {{
  0%,97%,100% {{ opacity:1; }}
  98% {{ opacity:0.85; }}
}}
@keyframes dataFloat {{
  0% {{ transform:translateY(0) translateX(0); opacity:0; }}
  15% {{ opacity:1; }}
  85% {{ opacity:1; }}
  100% {{ transform:translateY(-120px) translateX(20px); opacity:0; }}
}}
@keyframes pricePulse {{
  0%,100% {{ opacity:0.4; }}
  50% {{ opacity:1; }}
}}
@keyframes chartLine {{
  from {{ stroke-dashoffset:600; }}
  to {{ stroke-dashoffset:0; }}
}}
@keyframes scanBeam {{
  0% {{ top:-2px; opacity:0.6; }}
  100% {{ top:100%; opacity:0; }}
}}
@keyframes traderBreathe {{
  0%,100% {{ transform:translateY(0); }}
  50% {{ transform:translateY(-2px); }}
}}
@keyframes keyType {{
  0%,75%,100% {{ opacity:0.1; }}
  80%,90% {{ opacity:0.9; fill:rgba(0,229,255,0.7); filter:drop-shadow(0 0 3px rgba(0,229,255,0.5)); }}
}}
@keyframes cursorBlink {{
  0%,50% {{ opacity:1; }}
  51%,100% {{ opacity:0; }}
}}
@keyframes ripple {{
  0% {{ transform:scale(0.8); opacity:0.6; }}
  100% {{ transform:scale(2.5); opacity:0; }}
}}
@keyframes signalPing {{
  0% {{ r:2; opacity:1; }}
  100% {{ r:12; opacity:0; }}
}}
@keyframes monitor-glow {{
  0%,100% {{ filter:drop-shadow(0 0 8px rgba(0,229,255,0.15)); }}
  50% {{ filter:drop-shadow(0 0 20px rgba(0,229,255,0.35)); }}
}}
.trader-scene {{
  position: relative;
  width: 100%;
  max-width: 1000px;
  height: 50vh;
  min-height: 350px;
  margin: 0 auto;
  animation: fadeInUp 1s ease;
}}
.trader-scene svg {{
  width: 100%;
  height: 100%;
}}
.monitor-group {{
  animation: monitor-glow 4s ease-in-out infinite;
}}
.chart-path {{
  stroke-dasharray: 600;
  stroke-dashoffset: 600;
  animation: chartLine 2.5s ease forwards;
}}
.chart-path-2 {{
  stroke-dasharray: 600;
  stroke-dashoffset: 600;
  animation: chartLine 2.5s ease 0.5s forwards;
}}
.chart-path-3 {{
  stroke-dasharray: 600;
  stroke-dashoffset: 600;
  animation: chartLine 2.5s ease 1s forwards;
}}
.scan-line {{
  animation: scanBeam 3s linear infinite;
}}
.trader-body {{
  animation: traderBreathe 4s ease-in-out infinite;
}}
.data-particle {{
  animation: dataFloat 3s ease-in-out infinite;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  fill: {CYAN};
  opacity: 0;
}}
.data-particle:nth-child(2) {{ animation-delay: 0.5s; }}
.data-particle:nth-child(3) {{ animation-delay: 1.0s; }}
.data-particle:nth-child(4) {{ animation-delay: 1.5s; }}
.data-particle:nth-child(5) {{ animation-delay: 2.0s; }}
.data-particle:nth-child(6) {{ animation-delay: 2.5s; }}
.signal-ping {{
  animation: signalPing 2s ease-out infinite;
}}
.signal-ping-2 {{
  animation: signalPing 2s ease-out 0.7s infinite;
}}
.signal-ping-3 {{
  animation: signalPing 2s ease-out 1.4s infinite;
}}
.candle {{
  animation: candleUp 0.6s ease forwards;
  opacity: 0;
}}
.sim-ticker {{
  overflow:hidden;
  white-space:nowrap;
  background:linear-gradient(90deg,rgba(11,17,32,0.95),rgba(15,25,50,0.85));
  border:1px solid {BORDER};
  border-radius:6px;
  padding:10px 0;
  margin:0 0 6px 0;
}}
.sim-ticker-inner {{
  display:inline-block;
  animation:tickerScroll 25s linear infinite;
  padding-left:100%;
}}
</style>

<div style='text-align:center;padding:16px 20px 0;'>
<div style='font-size:4.8rem;color:{CYAN};font-weight:900;font-family:Inter,sans-serif;letter-spacing:10px;
animation:heroGlow 3s ease-in-out infinite;line-height:1;'>&#9670; NEXUS</div>
<div style='color:{DIM};font-size:1.05rem;margin-top:8px;letter-spacing:5px;font-family:Inter,sans-serif;
animation:fadeInUp 0.8s ease 0.3s forwards;opacity:0;'>CRYPTO PREDICTION MARKET TERMINAL</div>
</div>

<div class='trader-scene'>
<svg viewBox='0 0 900 420' xmlns='http://www.w3.org/2000/svg'>
  <!-- Desk surface -->
  <rect x='100' y='310' width='700' height='6' rx='3' fill='url(#deskGrad)' opacity='0.7'/>
  <defs>
    <linearGradient id='deskGrad' x1='0' y1='0' x2='1' y2='0'>
      <stop offset='0%' stop-color='transparent'/>
      <stop offset='30%' stop-color='{CYAN}' stop-opacity='0.3'/>
      <stop offset='70%' stop-color='{PURPLE}' stop-opacity='0.3'/>
      <stop offset='100%' stop-color='transparent'/>
    </linearGradient>
    <linearGradient id='screenGrad' x1='0' y1='0' x2='0' y2='1'>
      <stop offset='0%' stop-color='rgba(0,229,255,0.08)'/>
      <stop offset='100%' stop-color='rgba(5,8,15,0.95)'/>
    </linearGradient>
    <linearGradient id='chartGrad1' x1='0' y1='0' x2='1' y2='0'>
      <stop offset='0%' stop-color='{CYAN}'/>
      <stop offset='100%' stop-color='{GREEN}'/>
    </linearGradient>
    <linearGradient id='chartGrad2' x1='0' y1='0' x2='1' y2='0'>
      <stop offset='0%' stop-color='{PURPLE}'/>
      <stop offset='100%' stop-color='{CYAN}'/>
    </linearGradient>
    <linearGradient id='chartGrad3' x1='0' y1='0' x2='1' y2='0'>
      <stop offset='0%' stop-color='{AMBER}'/>
      <stop offset='100%' stop-color='{RED}'/>
    </linearGradient>
  </defs>

  <!-- === LEFT MONITOR === -->
  <g class='monitor-group'>
    <rect x='115' y='120' width='220' height='185' rx='6' fill='rgba(5,8,15,0.9)' stroke='{BORDER}' stroke-width='1.5'/>
    <rect x='115' y='120' width='220' height='3' rx='6' fill='{CYAN}' opacity='0.6'/>
    <!-- candlestick chart -->
    <g transform='translate(130,140)'>
      <line x1='0' y1='145' x2='190' y2='145' stroke='{BORDER}' stroke-width='0.5' opacity='0.3'/>
      <line x1='0' y1='110' x2='190' y2='110' stroke='{BORDER}' stroke-width='0.5' opacity='0.2'/>
      <line x1='0' y1='75' x2='190' y2='75' stroke='{BORDER}' stroke-width='0.5' opacity='0.2'/>
      <line x1='0' y1='40' x2='190' y2='40' stroke='{BORDER}' stroke-width='0.5' opacity='0.2'/>
      <!-- candles -->
      <rect class='candle' style='animation-delay:0.3s' x='8' y='60' width='8' height='50' rx='1' fill='{GREEN}' opacity='0.8'/><line x1='12' y1='50' x2='12' y2='120' stroke='{GREEN}' stroke-width='1' opacity='0.5'/>
      <rect class='candle' style='animation-delay:0.45s' x='22' y='45' width='8' height='40' rx='1' fill='{GREEN}' opacity='0.8'/><line x1='26' y1='35' x2='26' y2='95' stroke='{GREEN}' stroke-width='1' opacity='0.5'/>
      <rect class='candle' style='animation-delay:0.6s' x='36' y='70' width='8' height='55' rx='1' fill='{RED}' opacity='0.8'/><line x1='40' y1='55' x2='40' y2='135' stroke='{RED}' stroke-width='1' opacity='0.5'/>
      <rect class='candle' style='animation-delay:0.75s' x='50' y='80' width='8' height='35' rx='1' fill='{RED}' opacity='0.8'/><line x1='54' y1='70' x2='54' y2='125' stroke='{RED}' stroke-width='1' opacity='0.5'/>
      <rect class='candle' style='animation-delay:0.9s' x='64' y='55' width='8' height='50' rx='1' fill='{GREEN}' opacity='0.8'/><line x1='68' y1='40' x2='68' y2='115' stroke='{GREEN}' stroke-width='1' opacity='0.5'/>
      <rect class='candle' style='animation-delay:1.05s' x='78' y='35' width='8' height='45' rx='1' fill='{GREEN}' opacity='0.8'/><line x1='82' y1='25' x2='82' y2='90' stroke='{GREEN}' stroke-width='1' opacity='0.5'/>
      <rect class='candle' style='animation-delay:1.2s' x='92' y='50' width='8' height='30' rx='1' fill='{RED}' opacity='0.8'/><line x1='96' y1='40' x2='96' y2='90' stroke='{RED}' stroke-width='1' opacity='0.5'/>
      <rect class='candle' style='animation-delay:1.35s' x='106' y='30' width='8' height='55' rx='1' fill='{GREEN}' opacity='0.8'/><line x1='110' y1='20' x2='110' y2='95' stroke='{GREEN}' stroke-width='1' opacity='0.5'/>
      <rect class='candle' style='animation-delay:1.5s' x='120' y='25' width='8' height='40' rx='1' fill='{GREEN}' opacity='0.8'/><line x1='124' y1='15' x2='124' y2='75' stroke='{GREEN}' stroke-width='1' opacity='0.5'/>
      <rect class='candle' style='animation-delay:1.65s' x='134' y='40' width='8' height='50' rx='1' fill='{RED}' opacity='0.8'/><line x1='138' y1='30' x2='138' y2='100' stroke='{RED}' stroke-width='1' opacity='0.5'/>
      <rect class='candle' style='animation-delay:1.8s' x='148' y='20' width='8' height='45' rx='1' fill='{GREEN}' opacity='0.8'/><line x1='152' y1='10' x2='152' y2='75' stroke='{GREEN}' stroke-width='1' opacity='0.5'/>
      <rect class='candle' style='animation-delay:1.95s' x='162' y='15' width='8' height='35' rx='1' fill='{GREEN}' opacity='0.8'/><line x1='166' y1='8' x2='166' y2='60' stroke='{GREEN}' stroke-width='1' opacity='0.5'/>
      <rect class='candle' style='animation-delay:2.1s' x='176' y='10' width='8' height='40' rx='1' fill='{GREEN}' opacity='0.8'/><line x1='180' y1='5' x2='180' y2='60' stroke='{GREEN}' stroke-width='1' opacity='0.5'/>
    </g>
    <!-- monitor label -->
    <text x='135' y='138' fill='{CYAN}' font-family='JetBrains Mono,monospace' font-size='9' opacity='0.7'>BTC/USD · 1H</text>
    <text x='275' y='138' fill='{GREEN}' font-family='JetBrains Mono,monospace' font-size='9' text-anchor='end' opacity='0.8'>&#9650; +2.4%</text>
    <!-- scan line -->
    <rect class='scan-line' x='116' y='120' width='218' height='1.5' fill='{CYAN}' opacity='0.15' style='position:relative;'/>
  </g>

  <!-- === CENTER MONITOR (main) === -->
  <g class='monitor-group'>
    <rect x='345' y='90' width='210' height='215' rx='6' fill='rgba(5,8,15,0.9)' stroke='{CYAN}' stroke-width='1' opacity='0.8'/>
    <rect x='345' y='90' width='210' height='3' rx='6' fill='{CYAN}' opacity='0.8'/>
    <!-- line chart -->
    <path class='chart-path' d='M360,260 L375,245 L390,250 L405,230 L420,235 L435,210 L450,220 L465,190 L480,200 L495,170 L510,160 L525,140 L540,130' fill='none' stroke='url(#chartGrad1)' stroke-width='2.5' stroke-linecap='round'/>
    <!-- area fill under -->
    <path d='M360,260 L375,245 L390,250 L405,230 L420,235 L435,210 L450,220 L465,190 L480,200 L495,170 L510,160 L525,140 L540,130 L540,280 L360,280 Z' fill='{CYAN}' opacity='0.04'/>
    <!-- MACD bars at bottom -->
    <rect x='365' y='268' width='6' height='8' fill='{GREEN}' opacity='0.5' rx='1'/>
    <rect x='377' y='270' width='6' height='6' fill='{GREEN}' opacity='0.4' rx='1'/>
    <rect x='389' y='272' width='6' height='4' fill='{RED}' opacity='0.4' rx='1'/>
    <rect x='401' y='270' width='6' height='6' fill='{RED}' opacity='0.5' rx='1'/>
    <rect x='413' y='268' width='6' height='8' fill='{RED}' opacity='0.4' rx='1'/>
    <rect x='425' y='266' width='6' height='10' fill='{GREEN}' opacity='0.5' rx='1'/>
    <rect x='437' y='264' width='6' height='12' fill='{GREEN}' opacity='0.5' rx='1'/>
    <rect x='449' y='266' width='6' height='10' fill='{GREEN}' opacity='0.4' rx='1'/>
    <rect x='461' y='268' width='6' height='8' fill='{GREEN}' opacity='0.5' rx='1'/>
    <rect x='473' y='265' width='6' height='11' fill='{GREEN}' opacity='0.6' rx='1'/>
    <rect x='485' y='262' width='6' height='14' fill='{GREEN}' opacity='0.5' rx='1'/>
    <rect x='497' y='260' width='6' height='16' fill='{GREEN}' opacity='0.6' rx='1'/>
    <!-- live dot -->
    <circle cx='540' cy='130' r='3' fill='{CYAN}'/>
    <circle cx='540' cy='130' r='3' fill='{CYAN}' class='signal-ping' opacity='0.6'/>
    <circle cx='540' cy='130' r='3' fill='{CYAN}' class='signal-ping-2' opacity='0.4'/>
    <!-- labels -->
    <text x='360' y='108' fill='{CYAN}' font-family='JetBrains Mono,monospace' font-size='9.5' font-weight='700'>&#9670; NEXUS FUSION</text>
    <text x='540' y='108' fill='{GREEN}' font-family='JetBrains Mono,monospace' font-size='9' text-anchor='end'>BULLISH 0.72</text>
    <!-- crosshair -->
    <line x1='480' y1='95' x2='480' y2='280' stroke='{CYAN}' stroke-width='0.5' stroke-dasharray='3,3' opacity='0.2'/>
    <line x1='348' y1='200' x2='553' y2='200' stroke='{CYAN}' stroke-width='0.5' stroke-dasharray='3,3' opacity='0.2'/>
    <!-- monitor stand -->
    <rect x='435' y='305' width='30' height='10' rx='2' fill='{BORDER}' opacity='0.4'/>
  </g>

  <!-- === RIGHT MONITOR === -->
  <g class='monitor-group'>
    <rect x='565' y='120' width='220' height='185' rx='6' fill='rgba(5,8,15,0.9)' stroke='{BORDER}' stroke-width='1.5'/>
    <rect x='565' y='120' width='220' height='3' rx='6' fill='{PURPLE}' opacity='0.6'/>
    <!-- multiple line charts (overlaid) -->
    <path class='chart-path-2' d='M580,270 L600,255 L620,260 L640,240 L660,220 L680,230 L700,200 L720,210 L740,185 L760,175' fill='none' stroke='url(#chartGrad2)' stroke-width='2' stroke-linecap='round'/>
    <path class='chart-path-3' d='M580,250 L600,260 L620,245 L640,255 L660,250 L680,240 L700,250 L720,235 L740,230 L760,225' fill='none' stroke='url(#chartGrad3)' stroke-width='1.5' stroke-linecap='round' stroke-dasharray='4,2'/>
    <!-- data rows -->
    <text x='580' y='145' fill='{PURPLE}' font-family='JetBrains Mono,monospace' font-size='8' opacity='0.7'>POLYMARKET</text>
    <text x='580' y='160' fill='{DIM}' font-family='JetBrains Mono,monospace' font-size='7.5'>BTC &gt; 100K    <tspan fill='{GREEN}'>72%</tspan></text>
    <text x='580' y='172' fill='{DIM}' font-family='JetBrains Mono,monospace' font-size='7.5'>ETH &gt; 4K      <tspan fill='{AMBER}'>58%</tspan></text>
    <text x='580' y='184' fill='{DIM}' font-family='JetBrains Mono,monospace' font-size='7.5'>SOL &gt; 200     <tspan fill='{GREEN}'>65%</tspan></text>
    <text x='580' y='196' fill='{DIM}' font-family='JetBrains Mono,monospace' font-size='7.5'>BNB &gt; 700     <tspan fill='{RED}'>41%</tspan></text>
    <!-- RSS feed section -->
    <line x1='575' y1='205' x2='775' y2='205' stroke='{BORDER}' stroke-width='0.5' opacity='0.4'/>
    <text x='580' y='218' fill='{AMBER}' font-family='JetBrains Mono,monospace' font-size='8' opacity='0.7'>RSS / VADER</text>
    <text x='580' y='232' fill='{DIM}' font-family='JetBrains Mono,monospace' font-size='7'>■ CoinDesk: bullish momentum<tspan fill='{GREEN}'> +0.6</tspan></text>
    <text x='580' y='244' fill='{DIM}' font-family='JetBrains Mono,monospace' font-size='7'>■ CoinTelegraph: ETF flows<tspan fill='{GREEN}'> +0.4</tspan></text>
    <text x='580' y='256' fill='{DIM}' font-family='JetBrains Mono,monospace' font-size='7'>■ Reuters: regulation risk<tspan fill='{RED}'> -0.3</tspan></text>
  </g>

  <!-- === TRADER FIGURE === -->
  <g class='trader-body' transform='translate(400,180)'>
    <!-- chair -->
    <ellipse cx='50' cy='155' rx='35' ry='5' fill='{BORDER}' opacity='0.25'/>
    <rect x='25' y='120' width='50' height='40' rx='10' fill='rgba(15,25,50,0.9)' stroke='{BORDER}' stroke-width='1'/>
    <!-- body/torso -->
    <rect x='30' y='80' width='40' height='50' rx='8' fill='rgba(20,35,65,0.95)' stroke='{BORDER}' stroke-width='0.8'/>
    <!-- shoulders -->
    <rect x='20' y='82' width='60' height='15' rx='7' fill='rgba(20,35,65,0.9)' stroke='{BORDER}' stroke-width='0.8'/>
    <!-- head -->
    <ellipse cx='50' cy='65' rx='16' ry='19' fill='rgba(25,40,70,0.95)' stroke='{BORDER}' stroke-width='1'/>
    <!-- headset -->
    <path d='M34,58 Q34,42 50,42 Q66,42 66,58' fill='none' stroke='{CYAN}' stroke-width='2' opacity='0.6'/>
    <circle cx='34' cy='60' r='5' fill='rgba(0,229,255,0.15)' stroke='{CYAN}' stroke-width='1' opacity='0.6'/>
    <circle cx='66' cy='60' r='5' fill='rgba(0,229,255,0.15)' stroke='{CYAN}' stroke-width='1' opacity='0.6'/>
    <!-- mic -->
    <path d='M34,65 Q25,75 30,80' fill='none' stroke='{CYAN}' stroke-width='1.5' opacity='0.4'/>
    <!-- arms reaching to keyboard -->
    <path d='M25,95 Q5,110 -15,112' fill='none' stroke='{BORDER}' stroke-width='5' stroke-linecap='round' opacity='0.7'/>
    <path d='M75,95 Q95,110 115,112' fill='none' stroke='{BORDER}' stroke-width='5' stroke-linecap='round' opacity='0.7'/>
    <!-- keyboard -->
    <rect x='-30' y='110' width='160' height='8' rx='3' fill='rgba(15,25,50,0.8)' stroke='{BORDER}' stroke-width='0.8'/>
    <!-- keyboard keys glow — animated typing -->
    <rect x='-20' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1' style='animation:keyType 1.8s ease infinite 0.0s'/>
    <rect x='-12' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1' style='animation:keyType 1.8s ease infinite 0.35s'/>
    <rect x='-4' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1' style='animation:keyType 1.8s ease infinite 0.15s'/>
    <rect x='4' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1' style='animation:keyType 1.8s ease infinite 0.55s'/>
    <rect x='20' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1' style='animation:keyType 1.8s ease infinite 0.8s'/>
    <rect x='28' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1' style='animation:keyType 1.8s ease infinite 0.25s'/>
    <rect x='50' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1' style='animation:keyType 1.8s ease infinite 1.1s'/>
    <rect x='70' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1' style='animation:keyType 1.8s ease infinite 0.45s'/>
    <rect x='90' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1' style='animation:keyType 1.8s ease infinite 0.7s'/>
    <rect x='100' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1' style='animation:keyType 1.8s ease infinite 1.3s'/>
    <rect x='110' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1' style='animation:keyType 1.8s ease infinite 0.95s'/>
    <!-- blinking cursor on center monitor -->
    <rect x='538' y='132' width='6' height='10' rx='1' fill='{CYAN}' opacity='0.7' style='animation:cursorBlink 1s step-end infinite'/>
    <!-- face glow from screens -->
    <ellipse cx='50' cy='65' rx='14' ry='17' fill='{CYAN}' opacity='0.03'/>
    <!-- visor/glasses -->
    <rect x='38' y='58' width='24' height='8' rx='3' fill='rgba(0,229,255,0.08)' stroke='{CYAN}' stroke-width='0.8' opacity='0.5'/>
  </g>

  <!-- floating data particles -->
  <text class='data-particle' x='160' y='105' style='animation-delay:0.2s'>$87,432</text>
  <text class='data-particle' x='710' y='100' fill='{GREEN}' style='animation-delay:0.8s'>+2.4%</text>
  <text class='data-particle' x='270' y='115' fill='{PURPLE}' style='animation-delay:1.3s'>72% YES</text>
  <text class='data-particle' x='620' y='108' fill='{AMBER}' style='animation-delay:1.8s'>RSI: 62</text>
  <text class='data-particle' x='200' y='95' fill='{GREEN}' style='animation-delay:2.3s'>MACD &#9650;</text>
  <text class='data-particle' x='680' y='95' style='animation-delay:2.8s'>VOL: 1.2B</text>

  <!-- desk reflection/glow -->
  <ellipse cx='450' cy='316' rx='250' ry='8' fill='{CYAN}' opacity='0.03'/>

  <!-- floor ambient -->
  <rect x='50' y='380' width='800' height='1' fill='{BORDER}' opacity='0.1'/>
</svg>
</div>
""", unsafe_allow_html=True)

    # Scrolling ticker tape
    _sim_prices = {
        'BTC': (87432.15, +2.4), 'ETH': (3241.78, -0.8), 'SOL': (178.92, +5.1),
        'BNB': (612.30, +1.2), 'XRP': (2.41, -1.5), 'ADA': (0.72, +3.3),
        'AVAX': (38.15, -2.1), 'DOT': (8.92, +0.6), 'DOGE': (0.187, +4.7),
        'LINK': (18.45, +1.8), 'LTC': (94.20, -0.3), 'UNI': (12.85, +2.9),
    }
    _ticker_html = " &nbsp;&nbsp;&nbsp; ".join(
        f"<span style='color:#eef2f7;font-weight:600;'>{sym}</span> "
        f"<span style='color:{GREEN if chg>0 else RED};'>${p:,.2f} ({chg:+.1f}%)</span>"
        for sym,(p,chg) in _sim_prices.items()
    )
    st.markdown(f"""<div class='sim-ticker'>
<div class='sim-ticker-inner' style='font-family:JetBrains Mono,monospace;font-size:0.88rem;'>
{_ticker_html} &nbsp;&nbsp;&nbsp; {_ticker_html}
</div></div>""", unsafe_allow_html=True)

    # ── THE BIG BUTTON ──
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        if st.button("INITIALIZE TERMINAL", use_container_width=True, type="primary"):
            st.session_state.entered = 'transitioning'
            st.rerun()

    st.markdown(f"""<div style='text-align:center;padding:4px 0 10px;'>
<div style='color:#1a2744;font-size:0.82rem;'>
Rule-based engine · No AI predictions · No financial advice</div>
</div>""", unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# TRANSITION CINEMATIC — full-screen animation before terminal reveal
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.get('entered') == 'transitioning':
    import time as _time

    # Ticker data for flowing banner
    _sim_prices = {
        'BTC': (87432.15, +2.4), 'ETH': (3241.78, -0.8), 'SOL': (178.92, +5.1),
        'BNB': (612.30, +1.2), 'XRP': (2.41, -1.5), 'ADA': (0.72, +3.3),
        'AVAX': (38.15, -2.1), 'DOT': (8.92, +0.6), 'DOGE': (0.187, +4.7),
        'LINK': (18.45, +1.8), 'LTC': (94.20, -0.3), 'UNI': (12.85, +2.9),
    }
    _ticker_html = " &nbsp;&nbsp;&#8226;&nbsp;&nbsp; ".join(
        f"<span style='color:#eef2f7;font-weight:600;'>{sym}</span> "
        f"<span style='color:{GREEN if chg>0 else RED};'>${p:,.2f} ({chg:+.1f}%)</span>"
        for sym,(p,chg) in _sim_prices.items()
    )

    # Kill ALL Streamlit chrome — pure black canvas
    st.markdown(f"""
<style>
[data-testid="stSidebar"]{{display:none;}}
[data-testid="stHeader"]{{display:none;}}
[data-testid="stToolbar"]{{display:none;}}
.stApp>header{{display:none;}}
.block-container{{padding:0!important;max-width:100%!important;}}
[data-testid="stAppViewContainer"]>div:first-child{{padding-top:0;}}
html,body,.stApp,[data-testid="stAppViewContainer"]{{background:#020408!important;overflow:hidden!important;-webkit-overflow-scrolling:touch;}}
.stApp>div>div{{background:#020408!important;}}

/* ── KEYFRAMES ── */
@keyframes tickerFlow {{
  0% {{ transform: translateX(0); }}
  100% {{ transform: translateX(-50%); }}
}}
@keyframes gridPerspective {{
  0% {{ transform:perspective(400px) rotateX(65deg) scale(3); opacity:0; }}
  30% {{ opacity:0.12; }}
  70% {{ transform:perspective(400px) rotateX(25deg) scale(1.2); opacity:0.06; }}
  100% {{ transform:perspective(400px) rotateX(0deg) scale(1); opacity:0; }}
}}
/* Reduce heavy animations on mobile */
@media (max-width: 768px) {{
  .tx-scan-1,.tx-scan-2,.tx-data-col,.tx-particle {{ display:none !important; }}
  .tx-ring {{ animation-duration:2s !important; }}
}}
@keyframes horizonSlash {{
  0% {{ width:0; opacity:0; }}
  20% {{ opacity:1; }}
  60% {{ width:80vw; }}
  100% {{ width:100vw; opacity:0; }}
}}
@keyframes scanBeam {{
  0% {{ top:-5%; opacity:0; }}
  10% {{ opacity:0.7; }}
  90% {{ opacity:0.3; }}
  100% {{ top:105%; opacity:0; }}
}}
@keyframes titleExplode {{
  0% {{ opacity:0; letter-spacing:60px; transform:scale(0.3); }}
  25% {{ opacity:1; letter-spacing:18px; transform:scale(1.1); }}
  40% {{ letter-spacing:10px; transform:scale(1); text-shadow:0 0 80px rgba(0,229,255,0.9),0 0 200px rgba(0,229,255,0.4); }}
  70% {{ opacity:1; text-shadow:0 0 40px rgba(0,229,255,0.6),0 0 100px rgba(0,229,255,0.2); }}
  100% {{ opacity:1; letter-spacing:10px; text-shadow:0 0 30px rgba(0,229,255,0.4); }}
}}
@keyframes subtitleType {{
  0% {{ width:0; border-right-color:{CYAN}; }}
  70% {{ width:100%; border-right-color:{CYAN}; }}
  100% {{ width:100%; border-right-color:transparent; }}
}}
@keyframes bootLine {{
  0% {{ opacity:0; transform:translateX(-20px); }}
  20% {{ opacity:1; transform:translateX(0); }}
  100% {{ opacity:1; }}
}}
@keyframes progressFill {{
  0% {{ width:0; }}
  15% {{ width:8%; }}
  30% {{ width:25%; }}
  50% {{ width:55%; }}
  70% {{ width:78%; }}
  90% {{ width:95%; }}
  100% {{ width:100%; }}
}}
@keyframes ringExpand {{
  0% {{ transform:translate(-50%,-50%) scale(0); opacity:0.5; }}
  50% {{ opacity:0.2; }}
  100% {{ transform:translate(-50%,-50%) scale(4); opacity:0; }}
}}
@keyframes particleDrift {{
  0% {{ transform:translateY(0); opacity:0; }}
  10% {{ opacity:0.8; }}
  100% {{ transform:translateY(-100vh); opacity:0; }}
}}
@keyframes dataRain {{
  0% {{ transform:translateY(-100%); opacity:0; }}
  10% {{ opacity:0.15; }}
  90% {{ opacity:0.15; }}
  100% {{ transform:translateY(100vh); opacity:0; }}
}}
@keyframes statusGlow {{
  0%,100% {{ opacity:0.5; }}
  50% {{ opacity:1; }}
}}
@keyframes lineRevealLR {{
  0% {{ transform:scaleX(0); transform-origin:left; }}
  100% {{ transform:scaleX(1); transform-origin:left; }}
}}

/* ── SCENE ── */
.tx-scene {{
  position:fixed;
  top:0;left:0;right:0;bottom:0;
  z-index:999999;
  background:radial-gradient(ellipse at 50% 40%, #0a1628 0%, #020408 70%);
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  overflow:hidden;
  font-family:'JetBrains Mono',monospace;
}}
.tx-grid {{
  position:absolute;top:0;left:0;right:0;bottom:0;
  background-image:
    linear-gradient(rgba(0,229,255,0.03) 1px,transparent 1px),
    linear-gradient(90deg,rgba(0,229,255,0.03) 1px,transparent 1px);
  background-size:50px 50px;
  animation:gridPerspective 3.5s ease-out forwards;
}}
.tx-horizon {{
  position:absolute;top:50%;left:50%;transform:translateX(-50%);
  height:2px;
  background:linear-gradient(90deg,transparent,{CYAN} 20%,{PURPLE} 50%,{CYAN} 80%,transparent);
  animation:horizonSlash 2.5s ease-out forwards;
  box-shadow:0 0 40px rgba(0,229,255,0.5),0 0 100px rgba(0,229,255,0.2);
}}
.tx-scan {{
  position:absolute;left:0;right:0;height:4px;
  background:linear-gradient(90deg,transparent 5%,rgba(0,229,255,0.6) 50%,transparent 95%);
  animation:scanBeam 2s ease-in-out forwards;
  box-shadow:0 0 30px rgba(0,229,255,0.4),0 -20px 60px rgba(0,229,255,0.1);
}}
.tx-scan-2 {{
  position:absolute;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent 5%,rgba(179,136,255,0.4) 50%,transparent 95%);
  animation:scanBeam 2.5s ease-in-out 0.4s forwards;
  box-shadow:0 0 20px rgba(179,136,255,0.3);
}}
.tx-title {{
  font-family:Inter,sans-serif;font-size:7rem;font-weight:900;
  color:{CYAN};z-index:5;position:relative;
  animation:titleExplode 2.5s cubic-bezier(0.16,1,0.3,1) forwards;
  margin-bottom:8px;
}}
.tx-subtitle {{
  color:{DIM};font-size:0.95rem;letter-spacing:6px;
  z-index:5;position:relative;
  overflow:hidden;white-space:nowrap;
  border-right:2px solid transparent;
  animation:subtitleType 2s steps(40) 0.8s forwards;
  width:0;
}}
.tx-ring {{
  position:absolute;top:48%;left:50%;
  width:80px;height:80px;border-radius:50%;
  border:1px solid {CYAN};
  animation:ringExpand 2s ease-out forwards;
}}
.tx-boot {{
  position:absolute;bottom:22%;left:50%;transform:translateX(-50%);
  z-index:5;text-align:left;width:380px;
}}
.tx-boot-line {{
  font-size:0.78rem;line-height:2.2;opacity:0;
  animation:bootLine 0.4s ease forwards;
}}
.tx-progress-wrap {{
  position:absolute;bottom:15%;left:50%;transform:translateX(-50%);
  width:280px;z-index:5;
}}
.tx-progress-track {{
  width:100%;height:4px;background:rgba(255,255,255,0.04);border-radius:4px;overflow:hidden;
}}
.tx-progress-bar {{
  height:100%;border-radius:4px;
  background:linear-gradient(90deg,{CYAN},{GREEN});
  animation:progressFill 1.5s ease-out forwards;
  box-shadow:0 0 15px rgba(0,229,255,0.5);
}}
.tx-progress-label {{
  display:flex;justify-content:space-between;margin-top:6px;
  font-size:0.7rem;color:{DIM};letter-spacing:1px;
}}
.tx-particle {{
  position:absolute;bottom:0;width:2px;height:2px;border-radius:50%;
  animation:particleDrift 3s ease-out forwards;opacity:0;
}}
.tx-data-col {{
  position:absolute;top:0;width:12px;
  font-size:8px;line-height:1.4;color:{CYAN};text-align:center;
  animation:dataRain 4s linear forwards;opacity:0;
  font-family:'JetBrains Mono',monospace;
}}

/* ── Mobile transition overrides ── */
@media (max-width: 768px) {{
  .tx-title {{ font-size:3rem !important; animation:titleExplode 1.5s ease forwards !important; }}
  .tx-subtitle {{ font-size:0.7rem !important; letter-spacing:3px !important; }}
  .tx-boot {{ width:90vw !important; left:5% !important; transform:none !important; }}
  .tx-boot-line {{ font-size:0.65rem !important; }}
  .tx-progress-wrap {{ width:80vw !important; left:10% !important; transform:none !important; }}
  .tx-scan,.tx-scan-2,.tx-data-col,.tx-particle {{ display:none !important; }}
  .tx-grid {{ background-size:30px 30px !important; }}
  .tx-ring {{ display:none !important; }}
  .tx-horizon {{ box-shadow:none !important; }}
  .await-title {{ font-size:2.5rem !important; letter-spacing:6px !important; }}
  .await-diamond {{ font-size:3rem !important; }}
  .await-sub {{ font-size:0.85rem !important; letter-spacing:2px !important; }}
  .await-hint {{ font-size:0.85rem !important; }}
  .await-status {{ flex-direction:column !important; gap:12px !important; align-items:center !important; }}
  .await-status-item {{ font-size:0.8rem !important; }}
  .await-orbit,.await-orbit2 {{ display:none !important; }}
}}
@media (max-width: 480px) {{
  .tx-title {{ font-size:2.2rem !important; }}
  .tx-boot {{ font-size:0.55rem !important; }}
  .await-title {{ font-size:2rem !important; letter-spacing:4px !important; }}
}}
</style>

<div class='tx-scene'>
  <div class='tx-grid'></div>
  <div class='tx-horizon'></div>
  <div class='tx-scan'></div>
  <div class='tx-scan-2'></div>

  <!-- expanding rings -->
  <div class='tx-ring' style='animation-delay:0.2s;'></div>
  <div class='tx-ring' style='animation-delay:0.5s;border-color:{PURPLE};width:60px;height:60px;'></div>
  <div class='tx-ring' style='animation-delay:0.8s;width:100px;height:100px;'></div>
  <div class='tx-ring' style='animation-delay:1.1s;border-color:{GREEN};width:40px;height:40px;'></div>

  <!-- TITLE -->
  <div class='tx-title'>&#9670; NEXUS</div>
  <div class='tx-subtitle'>INITIALIZING PREDICTION TERMINAL</div>

  <!-- boot sequence -->
  <div class='tx-boot'>
    <div class='tx-boot-line' style='animation-delay:0.2s;color:{GREEN};'>&#9656; OPERATOR AUTHENTICATED</div>
    <div class='tx-boot-line' style='animation-delay:0.4s;color:{GREEN};'>&#9656; YAHOO FINANCE STREAM &#8212;&#8212; CONNECTED</div>
    <div class='tx-boot-line' style='animation-delay:0.6s;color:{GREEN};'>&#9656; POLYMARKET GAMMA API &#8212;&#8212; CONNECTED</div>
    <div class='tx-boot-line' style='animation-delay:0.8s;color:{GREEN};'>&#9656; RSS/VADER NLP ENGINE &#8212;&#8212; LOADED</div>
    <div class='tx-boot-line' style='animation-delay:1.0s;color:{CYAN};'>&#9656; FUSION ENGINE CALIBRATED</div>
    <div class='tx-boot-line' style='animation-delay:1.2s;color:{AMBER};font-weight:700;animation:bootLine 0.3s ease 1.2s forwards,statusGlow 0.4s ease-in-out 1.2s 2;'>&#9656; ALL SYSTEMS GO &#8212;&#8212; ENTERING TERMINAL</div>
  </div>

  <!-- progress bar -->
  <div class='tx-progress-wrap'>
    <div class='tx-progress-track'><div class='tx-progress-bar'></div></div>
    <div class='tx-progress-label'><span>BOOT</span><span style='animation:statusGlow 0.8s ease-in-out infinite;color:{CYAN};'>LOADING</span><span>READY</span></div>
  </div>

  <!-- particles -->
  <div class='tx-particle' style='left:12%;animation-delay:0.1s;background:{CYAN};'></div>
  <div class='tx-particle' style='left:25%;animation-delay:0.4s;background:{GREEN};width:3px;height:3px;'></div>
  <div class='tx-particle' style='left:38%;animation-delay:0.7s;background:{PURPLE};'></div>
  <div class='tx-particle' style='left:50%;animation-delay:0.2s;background:{CYAN};width:3px;height:3px;'></div>
  <div class='tx-particle' style='left:62%;animation-delay:0.9s;background:{AMBER};'></div>
  <div class='tx-particle' style='left:75%;animation-delay:0.5s;background:{CYAN};'></div>
  <div class='tx-particle' style='left:88%;animation-delay:0.3s;background:{GREEN};width:3px;height:3px;'></div>
  <div class='tx-particle' style='left:5%;animation-delay:1.2s;background:{PURPLE};'></div>
  <div class='tx-particle' style='left:95%;animation-delay:0.8s;background:{CYAN};'></div>
  <div class='tx-particle' style='left:42%;animation-delay:1.5s;background:{RED};width:3px;height:3px;'></div>

  <!-- matrix-style data rain columns -->
  <div class='tx-data-col' style='left:8%;animation-delay:0.2s;'>1<br>0<br>1<br>1<br>0<br>0<br>1<br>0<br>1<br>1</div>
  <div class='tx-data-col' style='left:18%;animation-delay:0.8s;'>0<br>1<br>0<br>0<br>1<br>1<br>0<br>1<br>0<br>1</div>
  <div class='tx-data-col' style='left:32%;animation-delay:0.4s;color:{PURPLE};'>1<br>1<br>0<br>1<br>0<br>1<br>1<br>0<br>0<br>1</div>
  <div class='tx-data-col' style='left:55%;animation-delay:1.0s;'>0<br>0<br>1<br>0<br>1<br>1<br>0<br>1<br>1<br>0</div>
  <div class='tx-data-col' style='left:72%;animation-delay:0.6s;color:{GREEN};'>1<br>0<br>1<br>1<br>0<br>0<br>1<br>0<br>1<br>0</div>
  <div class='tx-data-col' style='left:85%;animation-delay:1.2s;'>0<br>1<br>1<br>0<br>1<br>0<br>0<br>1<br>0<br>1</div>
  <div class='tx-data-col' style='left:92%;animation-delay:0.3s;color:{AMBER};'>1<br>0<br>0<br>1<br>1<br>0<br>1<br>0<br>1<br>1</div>

  <!-- Flowing ticker bar at bottom -->
  <div style='position:absolute;bottom:0;left:0;right:0;height:36px;background:linear-gradient(180deg,transparent,rgba(5,8,15,0.95));overflow:hidden;border-top:1px solid rgba(0,229,255,0.15);'>
    <div style='display:flex;animation:tickerFlow 20s linear infinite;white-space:nowrap;padding:8px 0;font-family:JetBrains Mono,monospace;font-size:0.82rem;'>
      {_ticker_html} &nbsp;&nbsp;&#8226;&nbsp;&nbsp; {_ticker_html} &nbsp;&nbsp;&#8226;&nbsp;&nbsp; {_ticker_html}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Boot sound design (Web Audio API) ──
    import streamlit.components.v1 as _boot_snd
    _boot_snd.html("""
<script>
try {
  const W = window.parent || window;
  const A = new (W.AudioContext || W.webkitAudioContext || window.AudioContext || window.webkitAudioContext)();
  A.resume();

  // Deep bass drone (filtered noise) - shorter
  const droneLen = 1.6;
  const nBuf = A.createBuffer(1, A.sampleRate * droneLen, A.sampleRate);
  const nD = nBuf.getChannelData(0);
  for (let i = 0; i < nD.length; i++) nD[i] = Math.random() * 2 - 1;
  const drone = A.createBufferSource();
  drone.buffer = nBuf;
  const droneF = A.createBiquadFilter();
  droneF.type = 'lowpass'; droneF.frequency.value = 90;
  const droneG = A.createGain();
  droneG.gain.setValueAtTime(0, A.currentTime);
  droneG.gain.linearRampToValueAtTime(0.10, A.currentTime + 0.2);
  droneG.gain.linearRampToValueAtTime(0.05, A.currentTime + 1.2);
  droneG.gain.linearRampToValueAtTime(0, A.currentTime + droneLen);
  drone.connect(droneF); droneF.connect(droneG); droneG.connect(A.destination);
  drone.start();

  // Rising synth sweep - faster
  const sweep = A.createOscillator();
  sweep.type = 'sine';
  sweep.frequency.setValueAtTime(80, A.currentTime);
  sweep.frequency.exponentialRampToValueAtTime(800, A.currentTime + 1.4);
  const sweepG = A.createGain();
  sweepG.gain.setValueAtTime(0, A.currentTime);
  sweepG.gain.linearRampToValueAtTime(0.05, A.currentTime + 0.15);
  sweepG.gain.linearRampToValueAtTime(0.02, A.currentTime + 1.0);
  sweepG.gain.linearRampToValueAtTime(0, A.currentTime + 1.5);
  sweep.connect(sweepG); sweepG.connect(A.destination);
  sweep.start(); sweep.stop(A.currentTime + 1.5);

  // Boot-line beeps - faster timing
  const beepTimes = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2];
  const beepFreqs = [440, 523, 587, 659, 784, 1047];
  beepTimes.forEach((t, i) => {
    const o = A.createOscillator();
    o.type = 'triangle';
    o.frequency.value = beepFreqs[i];
    const g = A.createGain();
    g.gain.setValueAtTime(0, A.currentTime + t);
    g.gain.linearRampToValueAtTime(0.06, A.currentTime + t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.001, A.currentTime + t + 0.12);
    o.connect(g); g.connect(A.destination);
    o.start(A.currentTime + t); o.stop(A.currentTime + t + 0.15);
  });

  // Final "ready" chord at 1.4s
  [523, 659, 784].forEach(f => {
    const o = A.createOscillator();
    o.type = 'sine';
    o.frequency.value = f;
    const g = A.createGain();
    g.gain.setValueAtTime(0, A.currentTime + 1.4);
    g.gain.linearRampToValueAtTime(0.06, A.currentTime + 1.45);
    g.gain.exponentialRampToValueAtTime(0.001, A.currentTime + 1.8);
    o.connect(g); g.connect(A.destination);
    o.start(A.currentTime + 1.4); o.stop(A.currentTime + 1.9);
  });
} catch(e) {}
</script>
""", height=0)

    _time.sleep(1.8)
    st.session_state.entered = True
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TERMINAL — revealed after transition completes
# ══════════════════════════════════════════════════════════════════════════════

# ── Sidebar: asset picker ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding:4px 0 8px 0;'>
        <span style='color:{CYAN};font-size:1.3rem;font-weight:800;font-family:Inter,sans-serif;letter-spacing:3px;'>&#9670; NEXUS</span><br>
        <span style='color:{DIM};font-size:0.78rem;letter-spacing:2px;font-family:Inter,sans-serif;'>CRYPTO PREDICTION TERMINAL</span>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    _tickers = sorted(set(v['ticker'] for v in ASSETS.values()))
    selected_asset = st.selectbox("ASSET", _tickers, index=_tickers.index("BTC"))
    analyze_btn = st.button("EXECUTE SCAN", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown(f"""
    <div style='padding-top:4px;'>
        <span class='nx-live'></span>
        <span style='color:{GREEN};font-size:0.85rem;font-weight:600;'>LIVE</span>
        <span style='color:{DIM};font-size:0.8rem;'>&nbsp;&nbsp;{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
    </div>""", unsafe_allow_html=True)
    st.markdown(f"""
<div style='margin-top:12px;padding:10px;background:rgba(0,229,255,0.04);border:1px solid {BORDER};border-radius:6px;'>
<span style='color:{DIM};font-size:0.85rem;letter-spacing:1px;line-height:1.8;'>
Fuses real-time price action, prediction<br>
markets &amp; news sentiment into one signal.
</span>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    
    # ── ORACLE AI Chat Button ────────────────────────────────────────────────
    st.markdown(f"""
<div style='margin-bottom:12px;padding:12px;background:linear-gradient(135deg,rgba(179,136,255,0.08),rgba(0,229,255,0.05));
border:1px solid rgba(179,136,255,0.3);border-radius:8px;'>
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:8px;'>
        <div style='width:28px;height:28px;background:linear-gradient(135deg,{PURPLE},{CYAN});border-radius:50%;
        display:flex;align-items:center;justify-content:center;font-size:0.9rem;box-shadow:0 0 12px rgba(179,136,255,0.4);'>&#9671;</div>
        <span style='color:{PURPLE};font-size:0.95rem;font-weight:700;letter-spacing:2px;font-family:Inter,sans-serif;'>ORACLE AI</span>
    </div>
    <span style='color:{DIM};font-size:0.75rem;line-height:1.5;'>
    ChatGPT-powered market intelligence.<br>Ask anything about your analysis.
    </span>
</div>""", unsafe_allow_html=True)
    oracle_btn = st.button("SUMMON ORACLE", use_container_width=True, key="oracle_btn")
    
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    shutdown_btn = st.button("SHUTDOWN TERMINAL", use_container_width=True)
    
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    info_btn = st.button("MORE INFO", use_container_width=True, key="info_btn")

    # Instant shutdown sound — plays in main frame before rerun
    if shutdown_btn:
        import streamlit.components.v1 as _sd_click
        _sd_click.html("""<script>
try {
  const A = new (window.parent.AudioContext || window.parent.webkitAudioContext ||
                  window.AudioContext || window.webkitAudioContext)();
  A.resume();
  // Immediate power-down thud
  const o = A.createOscillator(); o.type = 'sine';
  o.frequency.setValueAtTime(200, A.currentTime);
  o.frequency.exponentialRampToValueAtTime(30, A.currentTime + 0.4);
  const g = A.createGain();
  g.gain.setValueAtTime(0.15, A.currentTime);
  g.gain.exponentialRampToValueAtTime(0.001, A.currentTime + 0.5);
  o.connect(g); g.connect(A.destination);
  o.start(); o.stop(A.currentTime + 0.5);
  // Store context on parent so shutdown iframe can reuse it
  window.parent._nexusAudio = A;
} catch(e) {}
</script>""", height=0)


# ── Run analysis ─────────────────────────────────────────────────────────────
if 'data' not in st.session_state:
    st.session_state.data = None

# ── More Info Dialog ─────────────────────────────────────────────────────────
@st.dialog("System Architecture & Design", width="large")
def _show_info_dialog():
    import pathlib
    readme_path = pathlib.Path(__file__).parent / "README.md"
    if readme_path.exists():
        content = readme_path.read_text(encoding="utf-8", errors="ignore")
        st.markdown(content)
    else:
        st.warning("README.md not found.")

if info_btn:
    _show_info_dialog()

# ── Oracle AI Session State ──────────────────────────────────────────────────
if 'oracle_active' not in st.session_state:
    st.session_state.oracle_active = False
if 'oracle_messages' not in st.session_state:
    st.session_state.oracle_messages = []
if 'oracle_pending' not in st.session_state:
    st.session_state.oracle_pending = False

# Handle Oracle button click
if oracle_btn:
    st.session_state.oracle_active = not st.session_state.oracle_active

# ── Shutdown ─────────────────────────────────────────────────────────────────
if shutdown_btn:
    st.session_state.entered = 'shutdown'
    st.rerun()

if st.session_state.entered == 'shutdown':
    st.markdown("""<style>
[data-testid="stSidebar"]{display:none;}
.block-container{padding:0!important;max-width:100%!important;}
[data-testid="stAppViewContainer"]>div:first-child{padding-top:0;}
header{display:none!important;}
iframe{border:none!important;}
[data-testid="stAppViewContainer"] iframe[height="700"]{position:fixed!important;top:0!important;left:0!important;width:100vw!important;height:100vh!important;z-index:99999!important;border:none!important;}
</style>""", unsafe_allow_html=True)
    import streamlit.components.v1 as _sd_scroll
    _sd_scroll.html("<script>window.parent.scrollTo(0,0);</script>",height=0)

    import streamlit.components.v1 as _components
    _sd_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700;900&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:{BG};display:flex;align-items:center;justify-content:center;height:100vh;width:100vw;overflow:hidden;font-family:'Inter',sans-serif;}}

/* Windows-style spinning dots */
.ring{{width:80px;height:80px;margin:0 auto 40px;position:relative;}}
.ring .dot{{position:absolute;width:7px;height:7px;background:{CYAN};border-radius:50%;top:0;left:50%;margin-left:-3.5px;transform-origin:3.5px 40px;animation:winSpin 3s cubic-bezier(0.5,0,0.5,1) infinite;opacity:0;}}
.ring .dot:nth-child(1){{animation-delay:0s;}}
.ring .dot:nth-child(2){{animation-delay:0.15s;}}
.ring .dot:nth-child(3){{animation-delay:0.30s;}}
.ring .dot:nth-child(4){{animation-delay:0.45s;}}
.ring .dot:nth-child(5){{animation-delay:0.60s;}}
@keyframes winSpin{{
  0%{{transform:rotate(0deg);opacity:0;}}
  10%{{opacity:1;}}
  80%{{opacity:1;}}
  90%{{opacity:0;}}
  100%{{transform:rotate(720deg);opacity:0;}}
}}

/* Fade/slide in */
@keyframes fadeSlideIn{{
  from{{opacity:0;transform:translateY(16px);}} to{{opacity:1;transform:translateY(0);}}
}}
@keyframes logIn{{
  from{{opacity:0;transform:translateX(-8px);}} to{{opacity:1;transform:translateX(0);}}
}}

/* Progress bar */
@keyframes progressFill{{
  0%{{width:0%;}} 12%{{width:6%;}} 25%{{width:18%;}} 40%{{width:32%;}}
  55%{{width:50%;}} 68%{{width:65%;}} 78%{{width:78%;}} 88%{{width:90%;}}
  95%{{width:96%;}} 100%{{width:100%;}}
}}

/* CRT screen-off at the end */
@keyframes screenOff{{
  0%{{opacity:1;filter:brightness(1);transform:scale(1);}}
  40%{{opacity:1;filter:brightness(1.5);transform:scale(1);}}
  65%{{opacity:1;filter:brightness(3);transform:scaleY(0.006) scaleX(1);}}
  82%{{opacity:0.5;filter:brightness(2);transform:scaleY(0.003) scaleX(0.3);}}
  100%{{opacity:0;filter:brightness(0);transform:scaleY(0) scaleX(0);}}
}}

/* Goodbye fade in */
@keyframes goodbyeFadeIn{{
  from{{opacity:0;transform:scale(0.96);filter:blur(3px);}}
  to{{opacity:1;transform:scale(1);filter:blur(0);}}
}}

/* Scanlines overlay */
@keyframes scanAnim{{0%{{background-position:0 0;}}100%{{background-position:0 4px;}}}}
.scanlines{{
  position:fixed;inset:0;pointer-events:none;z-index:10;
  background:repeating-linear-gradient(0deg,rgba(255,255,255,0.012) 0px,transparent 1px,transparent 3px);
  animation:scanAnim 0.15s steps(1) infinite;
}}
.vignette{{
  position:fixed;inset:0;pointer-events:none;z-index:10;
  background:radial-gradient(ellipse at center,transparent 40%,rgba(0,0,0,0.5) 100%);
}}

/* Layout */
.content{{text-align:center;animation:screenOff 1.2s ease-in 5.8s forwards;transform-origin:center;}}
.title{{font-size:2rem;font-weight:300;color:#c0c8d8;letter-spacing:6px;animation:fadeSlideIn 0.6s ease 0.2s both;}}
.subtitle{{font-size:0.9rem;color:#3a4a6a;letter-spacing:3px;margin-top:10px;animation:fadeSlideIn 0.6s ease 0.5s both;}}

/* Progress bar */
.pbar-wrap{{
  width:300px;height:3px;background:rgba(26,39,68,0.5);border-radius:3px;
  margin:32px auto 0;overflow:hidden;animation:fadeSlideIn 0.5s ease 0.7s both;
}}
.pbar{{
  height:100%;border-radius:3px;width:0%;
  background:linear-gradient(90deg,{CYAN},{PURPLE},{CYAN});
  animation:progressFill 4.5s cubic-bezier(0.4,0,0.2,1) 0.8s forwards;
  box-shadow:0 0 10px rgba(0,229,255,0.3);
}}

/* Log lines */
.log{{
  margin-top:28px;text-align:left;display:inline-block;
  font-family:'JetBrains Mono',monospace;font-size:0.82rem;line-height:2.2;color:#3a4a6a;
}}
.log div{{opacity:0;animation:logIn 0.3s ease forwards;}}
.log .ok{{color:{GREEN};font-weight:600;}}
.log .fin{{color:rgba(90,98,114,0.35);}}

/* Goodbye overlay */
.goodbye{{
  position:fixed;inset:0;z-index:100;background:{BG};
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  opacity:0;pointer-events:none;
  animation:goodbyeFadeIn 1.2s ease 6.8s forwards;
}}
.goodbye-diamond{{
  font-size:3rem;color:{CYAN};
  filter:drop-shadow(0 0 20px rgba(0,229,255,0.5));
  margin-bottom:16px;
}}
.goodbye-text{{
  font-size:1.6rem;color:#c0c8d8;letter-spacing:6px;font-weight:300;
}}
.goodbye-sub{{
  font-size:0.85rem;color:#3a4a6a;letter-spacing:3px;margin-top:14px;
}}
</style></head><body>
<div class="scanlines"></div>
<div class="vignette"></div>
<div class="content">
<div class="ring"><div class="dot"></div><div class="dot"></div><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
<div class="title">Shutting down</div>
<div class="subtitle">NEXUS TERMINAL v3.0</div>
<div class="pbar-wrap"><div class="pbar"></div></div>
<div class="log">
<div style="animation-delay:1.2s"><span class="ok">[OK]</span> Saving session state</div>
<div style="animation-delay:1.8s"><span class="ok">[OK]</span> Closing market feeds</div>
<div style="animation-delay:2.4s"><span class="ok">[OK]</span> Disconnecting Polymarket</div>
<div style="animation-delay:3.0s"><span class="ok">[OK]</span> Flushing signal cache</div>
<div style="animation-delay:3.6s"><span class="ok">[OK]</span> Terminating fusion engine</div>
<div style="animation-delay:4.2s"><span class="fin">--- session closed gracefully</span></div>
</div>
</div>
<div class="goodbye">
<div class="goodbye-diamond">&#9670;</div>
<div class="goodbye-text">Stay sharp out there.</div>
<div class="goodbye-sub">NEXUS TERMINAL &mdash; SESSION COMPLETE</div>
</div>
<script>
try {{
  const W = window.parent || window;
  const A = W._nexusAudio || new (W.AudioContext || W.webkitAudioContext || window.AudioContext || window.webkitAudioContext)();
  A.resume();
  const t0 = A.currentTime;

  // Soft continuous drone — calm low pad
  const hum = A.createOscillator();
  hum.type = 'sine'; hum.frequency.value = 120;
  const hum2 = A.createOscillator();
  hum2.type = 'sine'; hum2.frequency.value = 120.4;
  const humF = A.createBiquadFilter();
  humF.type = 'lowpass'; humF.frequency.value = 180;
  const humG = A.createGain();
  humG.gain.setValueAtTime(0.03, t0);
  humG.gain.setValueAtTime(0.03, t0 + 5.5);
  humG.gain.linearRampToValueAtTime(0, t0 + 7.0);
  hum.connect(humF); hum2.connect(humF);
  humF.connect(humG); humG.connect(A.destination);
  hum.start(); hum2.start();
  hum.stop(t0 + 7.0); hum2.stop(t0 + 7.0);

  // [OK] beeps — softer
  [1.2, 1.8, 2.4, 3.0, 3.6].forEach((t, i) => {{
    const o = A.createOscillator();
    o.type = 'sine';
    o.frequency.value = 600 - i * 40;
    const g = A.createGain();
    g.gain.setValueAtTime(0, t0 + t);
    g.gain.linearRampToValueAtTime(0.03, t0 + t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.001, t0 + t + 0.15);
    o.connect(g); g.connect(A.destination);
    o.start(t0 + t); o.stop(t0 + t + 0.18);
  }});

  // Gentle power-down sweep
  const pd = A.createOscillator();
  pd.type = 'sine';
  pd.frequency.setValueAtTime(300, t0 + 5.3);
  pd.frequency.exponentialRampToValueAtTime(40, t0 + 7.0);
  const pdG = A.createGain();
  pdG.gain.setValueAtTime(0, t0 + 5.3);
  pdG.gain.linearRampToValueAtTime(0.04, t0 + 5.5);
  pdG.gain.exponentialRampToValueAtTime(0.001, t0 + 7.2);
  pd.connect(pdG); pdG.connect(A.destination);
  pd.start(t0 + 5.3); pd.stop(t0 + 7.2);

  // Gentle goodbye chime
  [392, 523, 659].forEach((f, i) => {{
    const o = A.createOscillator();
    o.type = 'sine';
    o.frequency.value = f;
    const g = A.createGain();
    g.gain.setValueAtTime(0, t0 + 7.0 + i * 0.12);
    g.gain.linearRampToValueAtTime(0.06, t0 + 7.05 + i * 0.12);
    g.gain.exponentialRampToValueAtTime(0.001, t0 + 8.0 + i * 0.12);
    o.connect(g); g.connect(A.destination);
    o.start(t0 + 7.0 + i * 0.12); o.stop(t0 + 8.5);
  }});
}} catch(e) {{}}
</script>
</body></html>"""
    _components.html(_sd_html, height=700, scrolling=False)

    import time as _time2
    _time2.sleep(9)
    st.session_state.entered = False
    st.session_state.data = None
    st.rerun()
    st.stop()  # Prevent any content below from rendering during shutdown

import threading, time as _cache_time

_scan_cache = {}
_scan_lock = threading.Lock()
_scan_ttl = 120  # seconds

def _run_scan(ticker, yf_symbol, keywords_tuple, is_btc):
    key = (ticker, yf_symbol, keywords_tuple, is_btc)
    with _scan_lock:
        if key in _scan_cache:
            val, ts = _scan_cache[key]
            if _cache_time.time() - ts < _scan_ttl:
                return val

    from concurrent.futures import ThreadPoolExecutor
    keywords = list(keywords_tuple)
    with ThreadPoolExecutor(max_workers=5) as pool:
        f_price   = pool.submit(technical.fetch_price_data, yf_symbol, "2y")
        f_weekly  = pool.submit(technical.get_weekly_trend, yf_symbol)
        f_btc     = pool.submit(technical.get_btc_trend) if not is_btc else None
        f_pm      = pool.submit(polymarket.analyze, keywords, ticker)
        f_news    = pool.submit(news.analyze, keywords)

    raw_price_df = f_price.result()
    price_df = technical.compute_indicators(raw_price_df.copy() if raw_price_df is not None else None)
    tech_data = technical.compute_signals(price_df)
    weekly_trend = f_weekly.result()
    btc_trend = f_btc.result() if f_btc else "N/A"
    pm_data = f_pm.result()
    news_data = f_news.result()

    decision = fusion.fuse(
        tech_data=tech_data, pm_data=pm_data, news_data=news_data,
        is_btc=is_btc, btc_trend=btc_trend, weekly_trend=weekly_trend,
    )

    # ── Extended analysis: feature lab → fused DataFrame → backtest + simulator
    feature_df = None
    fused_df = None
    bt_results = None
    sim_without_pm = None
    sim_with_pm = None
    scenario_table = None
    pm_return_beta = 0.0
    payoff_grid = None
    hedge_profiles = None
    resolution_table = None

    try:
        feature_df = technical.build_feature_lab(
            raw_price_df.copy() if raw_price_df is not None else None
        )
        if feature_df is not None and len(feature_df) >= 50:
            fused_df = fusion.fuse_dataframe(feature_df, pm_data)
            bt_results = backtest.run_backtest_variants(feature_df, fused_df)
            trade_input = backtest.prepare_trade_input(feature_df, fused_df)
            sim_without_pm = backtest.run_trade_simulation(trade_input, use_pm=False,
                                                           label='Without PM')
            sim_with_pm = backtest.run_trade_simulation(trade_input, use_pm=True,
                                                        label='With PM')
            # Scenario analysis
            base_conf = decision.get('confidence', 50)
            scenario_table = []
            for sc_name in fusion.SCENARIOS:
                adj = fusion.scenario_adjusted_confidence(base_conf, sc_name)
                scenario_table.append({'Scenario': sc_name, 'Adjusted Confidence': round(adj, 1)})
            # PM-Return Beta + Payoff / Hedge / Resolution
            pm_return_beta = backtest.compute_pm_return_beta(feature_df, fused_df)
            payoff_grid = backtest.build_probability_payoff_grid(fused_df, pm_return_beta)
            hedge_profiles = backtest.build_hedge_profiles(fused_df)
            resolution_table = backtest.build_resolution_window_table(fused_df, pm_return_beta)
    except Exception:
        pass  # Extended analysis is optional; don't break the main scan

    result = dict(
        asset_ticker=ticker, tech=tech_data, price_df=price_df,
        pm=pm_data, news=news_data, decision=decision,
        weekly_trend=weekly_trend, btc_trend=btc_trend,
        feature_df=feature_df, fused_df=fused_df,
        backtest=bt_results,
        sim_without_pm=sim_without_pm, sim_with_pm=sim_with_pm,
        scenario_table=scenario_table,
        pm_return_beta=pm_return_beta,
        payoff_grid=payoff_grid,
        hedge_profiles=hedge_profiles,
        resolution_table=resolution_table,
    )
    with _scan_lock:
        _scan_cache[key] = (result, _cache_time.time())
    return result

if analyze_btn:
    asset = resolve_asset(selected_asset.lower())
    is_btc = asset['ticker'] == "BTC"
    result = _run_scan(asset['ticker'], asset['yf_symbol'],
                       tuple(asset['keywords']), is_btc)
    result['asset'] = asset
    st.session_state.data = result

    # ── Scan activation sound ──
    import streamlit.components.v1 as _scan_snd
    _scan_snd.html("""<script>
try {
  const W = window.parent || window;
  const A = new (W.AudioContext || W.webkitAudioContext)();
  A.resume();
  [523,659,784,1047].forEach((f,i) => {
    const o = A.createOscillator(); o.type='sine'; o.frequency.value=f;
    const g = A.createGain();
    g.gain.setValueAtTime(0, A.currentTime + i*0.06);
    g.gain.linearRampToValueAtTime(0.08, A.currentTime + i*0.06 + 0.02);
    g.gain.exponentialRampToValueAtTime(0.001, A.currentTime + i*0.06 + 0.2);
    o.connect(g); g.connect(A.destination);
    o.start(A.currentTime + i*0.06); o.stop(A.currentTime + i*0.06 + 0.25);
  });
} catch(e){}
</script>""", height=0)

# ── Awaiting Scan ────────────────────────────────────────────────────────────
if st.session_state.data is None:
    # Prefetch selected asset in background so it's cached when user clicks
    _pf_key = f"_prefetch_{selected_asset}"
    if not st.session_state.get(_pf_key):
        st.session_state[_pf_key] = True
        def _prefetch():
            try:
                _a = resolve_asset(selected_asset.lower())
                _run_scan(_a['ticker'], _a['yf_symbol'], tuple(_a['keywords']),
                          _a['ticker'] == "BTC")
            except Exception:
                pass
        threading.Thread(target=_prefetch, daemon=True).start()

    st.markdown(f"""
<style>
@keyframes awaitPulse {{
  0%,100% {{ text-shadow:0 0 30px rgba(0,229,255,0.3),0 0 60px rgba(0,229,255,0.1); }}
  50% {{ text-shadow:0 0 60px rgba(0,229,255,0.6),0 0 120px rgba(0,229,255,0.25),0 0 200px rgba(0,229,255,0.1); }}
}}
@keyframes diamondSpin {{
  0% {{ transform:rotate(0deg) scale(1); filter:drop-shadow(0 0 8px rgba(0,229,255,0.4)); }}
  25% {{ transform:rotate(90deg) scale(1.15); filter:drop-shadow(0 0 20px rgba(0,229,255,0.7)); }}
  50% {{ transform:rotate(180deg) scale(1); filter:drop-shadow(0 0 8px rgba(0,229,255,0.4)); }}
  75% {{ transform:rotate(270deg) scale(1.15); filter:drop-shadow(0 0 20px rgba(0,229,255,0.7)); }}
  100% {{ transform:rotate(360deg) scale(1); filter:drop-shadow(0 0 8px rgba(0,229,255,0.4)); }}
}}
@keyframes scanLine {{
  0% {{ left:-100%; }}
  100% {{ left:100%; }}
}}
@keyframes gridPulse {{
  0%,100% {{ opacity:0.03; }}
  50% {{ opacity:0.08; }}
}}
@keyframes textFlicker {{
  0%,95%,100% {{ opacity:1; }}
  96% {{ opacity:0.4; }}
  97% {{ opacity:1; }}
  98% {{ opacity:0.3; }}
}}
@keyframes orbitDot {{
  0% {{ transform:rotate(0deg) translateX(120px) rotate(0deg); }}
  100% {{ transform:rotate(360deg) translateX(120px) rotate(-360deg); }}
}}
@keyframes fadeInUp {{
  from {{ opacity:0; transform:translateY(20px); }}
  to {{ opacity:1; transform:translateY(0); }}
}}
@keyframes barSweep {{
  0% {{ width:0%; }}
  50% {{ width:100%; }}
  100% {{ width:0%; }}
}}
@keyframes statusBlink {{
  0%,100% {{ opacity:1; }}
  50% {{ opacity:0.3; }}
}}
.await-wrap {{
  position:relative;
  text-align:center;
  padding:60px 20px 80px;
  overflow:hidden;
  min-height:70vh;
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:center;
}}
.await-grid {{
  position:absolute;
  inset:0;
  background-image:
    linear-gradient(rgba(0,229,255,0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,229,255,0.05) 1px, transparent 1px);
  background-size:60px 60px;
  animation:gridPulse 4s ease-in-out infinite;
  pointer-events:none;
}}
.await-scanline {{
  position:absolute;
  top:0;bottom:0;width:200px;
  background:linear-gradient(90deg,transparent,rgba(0,229,255,0.06),transparent);
  animation:scanLine 4s linear infinite;
  pointer-events:none;
}}
.await-diamond {{
  display:inline-block;
  font-size:5rem;
  color:{CYAN};
  animation:diamondSpin 8s cubic-bezier(0.4,0,0.2,1) infinite;
  margin-bottom:12px;
}}
.await-title {{
  font-size:5rem;
  color:{CYAN};
  font-weight:900;
  font-family:Inter,sans-serif;
  letter-spacing:14px;
  animation:awaitPulse 3s ease-in-out infinite;
  margin-top:4px;
}}
.await-sub {{
  color:{DIM};
  font-size:1.15rem;
  margin-top:14px;
  letter-spacing:4px;
  font-family:Inter,sans-serif;
  animation:textFlicker 5s infinite, fadeInUp 0.8s 0.3s both;
}}
.await-bar-wrap {{
  width:200px;
  height:2px;
  background:rgba(26,39,68,0.4);
  border-radius:2px;
  margin:28px auto 0;
  overflow:hidden;
  animation:fadeInUp 0.8s 0.5s both;
}}
.await-bar {{
  height:100%;
  background:linear-gradient(90deg,{CYAN},{PURPLE});
  border-radius:2px;
  animation:barSweep 3s ease-in-out infinite;
}}
.await-hint {{
  color:{DIM};
  font-size:1.1rem;
  margin-top:28px;
  letter-spacing:1.5px;
  animation:fadeInUp 0.8s 0.7s both;
}}
.await-orbit {{
  position:absolute;
  width:8px;height:8px;
  border-radius:50%;
  background:{CYAN};
  opacity:0.25;
  box-shadow:0 0 12px {CYAN};
  animation:orbitDot 12s linear infinite;
  top:50%;left:50%;
  margin:-4px 0 0 -4px;
}}
.await-orbit2 {{
  position:absolute;
  width:5px;height:5px;
  border-radius:50%;
  background:{PURPLE};
  opacity:0.2;
  box-shadow:0 0 10px {PURPLE};
  animation:orbitDot 18s linear reverse infinite;
  top:50%;left:50%;
  margin:-2.5px 0 0 -2.5px;
}}
.await-status {{
  display:flex;
  justify-content:center;
  gap:30px;
  margin-top:32px;
  animation:fadeInUp 0.8s 0.9s both;
}}
.await-status-item {{
  display:flex;
  align-items:center;
  gap:8px;
  font-family:JetBrains Mono,monospace;
  font-size:0.95rem;
  color:{DIM};
  letter-spacing:1px;
}}
.await-dot {{
  width:7px;height:7px;
  border-radius:50%;
  animation:statusBlink 2s infinite;
}}
</style>

<div class='await-wrap'>
  <div class='await-grid'></div>
  <div class='await-scanline'></div>
  <div class='await-orbit'></div>
  <div class='await-orbit2'></div>

  <div class='await-diamond'>&#9670;</div>
  <div class='await-title'>NEXUS</div>
  <div class='await-sub'>TERMINAL ONLINE — AWAITING SCAN</div>
  <div class='await-bar-wrap'><div class='await-bar'></div></div>

  <div class='await-hint'>
    ← Select an asset and hit
    <span style='color:{CYAN};font-weight:700;padding:4px 12px;border:1px solid {CYAN};
    border-radius:4px;background:rgba(0,229,255,0.08);margin-left:6px;'>EXECUTE SCAN</span>
  </div>

  <div class='await-status'>
    <div class='await-status-item'><div class='await-dot' style='background:{GREEN};box-shadow:0 0 6px {GREEN};'></div>YAHOO FINANCE</div>
    <div class='await-status-item'><div class='await-dot' style='background:{PURPLE};box-shadow:0 0 6px {PURPLE};animation-delay:0.4s;'></div>POLYMARKET</div>
    <div class='await-status-item'><div class='await-dot' style='background:{AMBER};box-shadow:0 0 6px {AMBER};animation-delay:0.8s;'></div>RSS / VADER</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Idle ambient hum (Web Audio API) ──
    import streamlit.components.v1 as _await_snd
    _await_snd.html(f"""
<script>
try {{
  const W = window.parent || window;
  const A = new (W.AudioContext || W.webkitAudioContext || window.AudioContext || window.webkitAudioContext)();
  A.resume();

  // Soft ambient pad — two detuned sines
  const pad1 = A.createOscillator();
  pad1.type = 'sine'; pad1.frequency.value = 110;
  const pad2 = A.createOscillator();
  pad2.type = 'sine'; pad2.frequency.value = 110.8;
  const padG = A.createGain();
  padG.gain.value = 0.025;
  const padF = A.createBiquadFilter();
  padF.type = 'lowpass'; padF.frequency.value = 200;
  pad1.connect(padF); pad2.connect(padF);
  padF.connect(padG); padG.connect(A.destination);
  pad1.start(); pad2.start();

  // Periodic soft blips every ~3s
  let blipN = 0;
  function blip() {{
    if (blipN > 30) return;
    const o = A.createOscillator();
    o.type = 'sine';
    o.frequency.value = 880 + Math.random() * 200;
    const g = A.createGain();
    g.gain.setValueAtTime(0, A.currentTime);
    g.gain.linearRampToValueAtTime(0.04, A.currentTime + 0.01);
    g.gain.exponentialRampToValueAtTime(0.001, A.currentTime + 0.25);
    o.connect(g); g.connect(A.destination);
    o.start(); o.stop(A.currentTime + 0.3);
    blipN++;
    setTimeout(blip, 2500 + Math.random() * 1500);
  }}
  setTimeout(blip, 800);
}} catch(e) {{}}
</script>
""", height=0)
    
    # ── Oracle Floating Panel (awaiting page) ────────────────────────────────
    if st.session_state.oracle_active:

        @st.fragment
        def _oracle_panel_await():
            st.markdown("<div class='oracle-float-marker' style='display:none'></div>", unsafe_allow_html=True)

            # ── Inject scoped CSS for chat styling ──
            st.markdown(f"""<style>
            @keyframes oraclePulse {{ 0%,100% {{ box-shadow:0 0 8px rgba(179,136,255,0.4); }} 50% {{ box-shadow:0 0 20px rgba(179,136,255,0.8),0 0 40px rgba(0,229,255,0.3); }} }}
            @keyframes oracleGradient {{ 0% {{ background-position:0% 50%; }} 50% {{ background-position:100% 50%; }} 100% {{ background-position:0% 50%; }} }}
            @keyframes dotPulse {{ 0%,80%,100% {{ opacity:0.2;transform:scale(0.8); }} 40% {{ opacity:1;transform:scale(1); }} }}
            .oracle-avatar {{ animation: oraclePulse 3s ease-in-out infinite; }}
            .oracle-header-bar {{ background: linear-gradient(270deg,rgba(179,136,255,0.15),rgba(0,229,255,0.1),rgba(179,136,255,0.15)); background-size:400% 400%; animation: oracleGradient 8s ease infinite; }}
            .oracle-dot {{ display:inline-block;width:6px;height:6px;border-radius:50%;background:{PURPLE};margin:0 2px;animation:dotPulse 1.4s infinite ease-in-out; }}
            .oracle-dot:nth-child(2) {{ animation-delay:0.2s; }}
            .oracle-dot:nth-child(3) {{ animation-delay:0.4s; }}
            .oracle-user-msg {{ background:linear-gradient(135deg,rgba(0,229,255,0.18),rgba(0,229,255,0.06)); border:1px solid rgba(0,229,255,0.25); padding:10px 14px; border-radius:14px 14px 4px 14px; max-width:82%; color:#eef2f7; font-size:0.85rem; line-height:1.5; word-wrap:break-word; }}
            .oracle-ai-msg {{ background:linear-gradient(135deg,rgba(179,136,255,0.1),rgba(12,16,30,0.8)); border:1px solid rgba(179,136,255,0.18); padding:10px 14px; border-radius:4px 14px 14px 14px; max-width:82%; color:#cdd4e0; font-size:0.85rem; line-height:1.6; word-wrap:break-word; }}
            </style>""", unsafe_allow_html=True)

            # ── Header ──
            st.markdown(f"""
<div class='oracle-header-bar' style='padding:12px 16px;display:flex;align-items:center;gap:12px;
border-bottom:1px solid rgba(179,136,255,0.15);margin:0 -12px 0;border-radius:14px 14px 0 0;'>
    <div class='oracle-avatar' style='width:36px;height:36px;background:linear-gradient(135deg,{PURPLE},{CYAN});
    border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0;'>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
    </div>
    <div style='flex:1;'>
        <div style='color:#fff;font-size:0.9rem;font-weight:700;letter-spacing:1.5px;font-family:Inter,sans-serif;'>ORACLE AI</div>
        <div style='color:rgba(179,136,255,0.7);font-size:0.68rem;letter-spacing:0.5px;'>Groq · LLaMA 3.3 70B</div>
    </div>
    <div style='display:flex;align-items:center;gap:6px;padding:4px 10px;background:rgba(0,255,170,0.08);border:1px solid rgba(0,255,170,0.2);border-radius:20px;'>
        <div style='width:6px;height:6px;background:{GREEN};border-radius:50%;box-shadow:0 0 8px {GREEN};'></div>
        <span style='color:{GREEN};font-size:0.68rem;font-weight:600;letter-spacing:1px;'>LIVE</span>
    </div>
</div>""", unsafe_allow_html=True)

            # ── Chat area ──
            chat_container = st.container(height=320)
            with chat_container:
                if not st.session_state.oracle_messages:
                    st.markdown(f"""
<div style='text-align:center;padding:40px 20px 20px;'>
    <div style='width:56px;height:56px;margin:0 auto 16px;background:linear-gradient(135deg,{PURPLE},{CYAN});
    border-radius:50%;display:flex;align-items:center;justify-content:center;box-shadow:0 0 30px rgba(179,136,255,0.3);'>
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.5"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
    </div>
    <div style='color:#fff;font-size:0.95rem;font-weight:600;margin-bottom:4px;'>How can I help?</div>
    <div style='color:{DIM};font-size:0.78rem;line-height:1.5;'>Ask about signals, risk, backtests, or Polymarket data.</div>
</div>""", unsafe_allow_html=True)
                    _suggestions = ["Explain my signal", "Why is caution high?", "Polymarket outlook?"]
                    _scols = st.columns(len(_suggestions))
                    for _si, _sq in enumerate(_suggestions):
                        if _scols[_si].button(_sq, key=f"sug_await_{_si}", use_container_width=True):
                            st.session_state.oracle_messages.append({"role": "user", "content": _sq})
                            st.session_state.oracle_pending = True
                            st.rerun()
                else:
                    for msg in st.session_state.oracle_messages:
                        if msg['role'] == 'user':
                            st.markdown(f"""
<div style='display:flex;justify-content:flex-end;margin:8px 4px;'>
    <div class='oracle-user-msg'>{msg['content']}</div>
</div>""", unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
<div style='display:flex;justify-content:flex-start;margin:8px 4px;gap:8px;'>
    <div style='width:24px;height:24px;min-width:24px;background:linear-gradient(135deg,{PURPLE},{CYAN});border-radius:50%;
    display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:4px;'>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
    </div>
    <div class='oracle-ai-msg'>{msg['content']}</div>
</div>""", unsafe_allow_html=True)

            # ── Float + Scroll JS ──
            import streamlit.components.v1 as _oscroll
            _oscroll.html("""<script>
(function(){
  function floatAndScroll(attempts) {
    try {
      var doc = window.parent.document;
      var marker = doc.querySelector('.oracle-float-marker');
      if (marker) {
        var el = marker;
        for (var i = 0; i < 15; i++) {
          el = el.parentElement;
          if (!el) break;
          var tid = el.getAttribute('data-testid');
          if (tid === 'stVerticalBlock' || tid === 'stVerticalBlockBorderWrapper') {
            if (el.offsetHeight > 100) {
              el.style.cssText = 'position:fixed !important;bottom:20px;right:20px;width:380px;max-height:560px;z-index:99999;background:rgba(6,10,20,0.97);border:1px solid rgba(179,136,255,0.25);border-radius:16px;box-shadow:0 12px 48px rgba(0,0,0,0.7),0 0 40px rgba(179,136,255,0.08),0 0 80px rgba(0,229,255,0.04);backdrop-filter:blur(24px);padding:0 0 8px;overflow-y:auto;overflow-x:hidden;';
              var bw = el.querySelectorAll('[data-testid="stVerticalBlockBorderWrapper"]');
              bw.forEach(function(b){ b.style.border='none'; b.style.background='transparent'; });
              break;
            }
          }
        }
        var sc = (el || marker).querySelectorAll('[data-testid="stScrollableContainer"]');
        if (!sc.length) sc = doc.querySelectorAll('[data-testid="stScrollableContainer"]');
        sc.forEach(function(c){ if(c.scrollHeight > c.clientHeight + 30) c.scrollTop = c.scrollHeight + 9999; });
      }
    } catch(e){}
    if (attempts > 0) {
      requestAnimationFrame(function(){ setTimeout(function(){ floatAndScroll(attempts - 1); }, 150); });
    }
  }
  setTimeout(function(){ floatAndScroll(10); }, 100);
})();
</script>""", height=0)

            if st.session_state.oracle_pending:
                with chat_container:
                    st.markdown(f"""
<div style='display:flex;align-items:center;gap:8px;margin:8px 4px;'>
    <div style='width:24px;height:24px;min-width:24px;background:linear-gradient(135deg,{PURPLE},{CYAN});border-radius:50%;
    display:flex;align-items:center;justify-content:center;flex-shrink:0;'>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
    </div>
    <div class='oracle-ai-msg' style='padding:12px 16px;'>
        <span class='oracle-dot'></span><span class='oracle-dot'></span><span class='oracle-dot'></span>
    </div>
</div>""", unsafe_allow_html=True)
                try:
                    response = _ask_oracle(st.session_state.oracle_messages, st.session_state.data)
                    st.session_state.oracle_messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.session_state.oracle_messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
                st.session_state.oracle_pending = False
                st.rerun()

            user_query = st.chat_input("Ask Oracle anything...", key="oracle_chat_await")
            if user_query:
                st.session_state.oracle_messages.append({"role": "user", "content": user_query})
                st.session_state.oracle_pending = True
                st.rerun()

            if st.session_state.oracle_messages:
                if st.button("🗑 Clear", key="clear_oracle_await", use_container_width=True):
                    st.session_state.oracle_messages = []
                    st.rerun()

        _oracle_panel_await()

    st.stop()

    # ── Guard: no scan data yet ──
if st.session_state.data is None:
    st.stop()

# ── Unpack ───────────────────────────────────────────────────────────────────
if not st.session_state.data:
    st.stop()
d       = st.session_state.data
asset   = d['asset']
tech    = d['tech']
pm      = d['pm']
nws     = d['news']
dec     = d['decision']
pdf     = d['price_df']

# ═════════════════════════════════════════════════════════════════════════════
# SCAN RESULTS — Collapsible Sections Layout
# ═════════════════════════════════════════════════════════════════════════════

# ── Compact Ticker Bar (Always visible) ──────────────────────────────────────
price_str = f"${tech['price']:,.2f}" if tech else "---"
price_clr = GREEN if tech and tech.get('rsi', 50) > 50 else RED
action = dec['action']
clr = GREEN if action == "LONG" else RED if action == "SHORT" else DIM
icon = "&#9650;" if action == "LONG" else "&#9660;" if action == "SHORT" else "&#9724;"
size_pct = dec['position_size'] * 100
conf = dec['confidence']
caut = dec['caution']
fusion_score = max(0, min(100, conf - caut * 0.5 + 25))

# Color for fusion score
f_clr = GREEN if fusion_score >= 65 else AMBER if fusion_score >= 45 else RED

st.markdown(f"""
<div class='nx-ticker' style='display:flex;align-items:center;justify-content:space-between;padding:12px 20px;'>
  <div style='display:flex;align-items:center;gap:16px;'>
    <span class='nx-live'></span>
    <span style='font-size:1.4rem;font-weight:800;color:#eef2f7;'>{asset['ticker']}</span>
    <span style='color:{price_clr};font-weight:700;font-size:1.2rem;'>{price_str}</span>
  </div>
  <div style='display:flex;align-items:center;gap:24px;'>
    <div style='text-align:center;'>
      <span style='color:{clr};font-size:1.6rem;font-weight:900;'>{icon} {action}</span>
    </div>
    <div style='text-align:center;border-left:1px solid {BORDER};padding-left:20px;'>
      <span style='color:{f_clr};font-size:1.4rem;font-weight:800;'>{fusion_score:.0f}</span>
      <span style='color:{DIM};font-size:0.8rem;'> SIGNAL</span>
    </div>
    <div style='text-align:center;border-left:1px solid {BORDER};padding-left:20px;'>
      <span style='color:{CYAN};font-size:1.1rem;font-weight:700;'>{conf:.0f}%</span>
      <span style='color:{DIM};font-size:0.8rem;'> CONF</span>
    </div>
    <div style='color:{DIM};font-size:0.9rem;'>{datetime.now().strftime('%H:%M:%S')}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DECISION DETAILS (Collapsible)
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("DECISION DETAILS", expanded=True):
    cls = "nx-action-long" if action == "LONG" else "nx-action-short" if action == "SHORT" else "nx-action-flat"
    bar_clr = GREEN if size_pct > 60 else AMBER if size_pct > 30 else RED
    zone = dec['risk_zone']
    zclr = GREEN if zone == "Tradeable" else AMBER if zone in ("Cautious", "High Risk") else RED

    # 4-column layout: Action | Gauge | Stats | Breakdown
    col_action, col_gauge, col_stats, col_breakdown = st.columns([1.2, 1.5, 1.3, 1.2])

    with col_action:
        st.markdown(f"""<div class='nx-cascade-1'><div class='{cls}' style='min-height:220px;display:flex;flex-direction:column;justify-content:center;'>
<div style='color:{clr};font-size:1rem;opacity:0.35;'>{icon}</div>
<div style='color:{clr};font-size:2.8rem;font-weight:900;letter-spacing:5px;font-family:Inter,sans-serif;margin:6px 0;'>{action}</div>
<div style='display:flex;justify-content:center;gap:24px;margin-top:12px;'>
<div style='text-align:center;'>
  <span style='color:{bar_clr};font-size:1.6rem;font-weight:800;'>{size_pct:.0f}%</span><br>
  <span style='color:{DIM};font-size:0.7rem;letter-spacing:1.5px;'>POSITION</span>
</div>
<div style='text-align:center;'>
  <span style='color:{zclr};font-size:1.2rem;font-weight:700;'>{zone}</span><br>
  <span style='color:{DIM};font-size:0.7rem;letter-spacing:1.5px;'>RISK ZONE</span>
</div>
</div>
</div></div>""", unsafe_allow_html=True)

# ── Gauge ──
with col_gauge:

    # Arc math: 240° sweep, starts at 150°
    arc_total = 240
    arc_start = 150
    score_angle = arc_start + (fusion_score / 100) * arc_total
    # SVG arc endpoint
    import math
    cx, cy, r_arc = 150, 140, 100
    def _arc_point(angle_deg):
        rad = math.radians(angle_deg)
        return cx + r_arc * math.cos(rad), cy + r_arc * math.sin(rad)

    ax1, ay1 = _arc_point(arc_start)
    ax2, ay2 = _arc_point(arc_start + arc_total)
    nx, ny = _arc_point(score_angle)

    # Color based on score
    if fusion_score >= 65:
        g_color, g_label = GREEN, "BULLISH"
    elif fusion_score >= 45:
        g_color, g_label = AMBER, "NEUTRAL"
    else:
        g_color, g_label = RED, "BEARISH"

    # Needle angle (rotate from center)
    needle_rot = arc_start + (fusion_score / 100) * arc_total

    # Tick marks
    ticks_html = ""
    for i in range(0, 101, 10):
        tick_angle = arc_start + (i / 100) * arc_total
        tx1, ty1 = cx + 88 * math.cos(math.radians(tick_angle)), cy + 88 * math.sin(math.radians(tick_angle))
        tx2, ty2 = cx + 100 * math.cos(math.radians(tick_angle)), cy + 100 * math.sin(math.radians(tick_angle))
        lx, ly = cx + 78 * math.cos(math.radians(tick_angle)), cy + 78 * math.sin(math.radians(tick_angle))
        t_opacity = "0.6" if i % 20 == 0 else "0.25"
        t_width = "2" if i % 20 == 0 else "1"
        ticks_html += f"<line x1='{tx1:.1f}' y1='{ty1:.1f}' x2='{tx2:.1f}' y2='{ty2:.1f}' stroke='{DIM}' stroke-width='{t_width}' opacity='{t_opacity}'/>"
        if i % 20 == 0:
            ticks_html += f"<text x='{lx:.1f}' y='{ly:.1f}' fill='{DIM}' font-size='12' text-anchor='middle' dominant-baseline='middle' font-family='JetBrains Mono,monospace'>{i}</text>"

    # Large-arc flag
    large_arc = 1 if arc_total > 180 else 0

    st.markdown(f"""
<style>
@keyframes gaugeArcDraw {{
  from {{ stroke-dashoffset: 380; }}
  to {{ stroke-dashoffset: {380 - (fusion_score / 100) * 380:.0f}; }}
}}
@keyframes needleSwing {{
  0% {{ transform: rotate({arc_start}deg); }}
  70% {{ transform: rotate({needle_rot + 3}deg); }}
  85% {{ transform: rotate({needle_rot - 1.5}deg); }}
  100% {{ transform: rotate({needle_rot}deg); }}
}}
@keyframes needleIdle {{
  0%   {{ transform: rotate({needle_rot}deg); }}
  25%  {{ transform: rotate({needle_rot + 1.2}deg); }}
  50%  {{ transform: rotate({needle_rot - 0.8}deg); }}
  75%  {{ transform: rotate({needle_rot + 0.6}deg); }}
  100% {{ transform: rotate({needle_rot}deg); }}
}}
@keyframes scoreCount {{
  from {{ opacity:0; transform:translateY(6px); }}
  to {{ opacity:1; transform:translateY(0); }}
}}
@keyframes glowPulse {{
  0%,100% {{ filter:drop-shadow(0 0 6px {g_color}40); }}
  50% {{ filter:drop-shadow(0 0 18px {g_color}80); }}
}}
.gauge-svg {{
  max-width:300px;
  margin:0 auto;
  display:block;
  animation: glowPulse 3s ease-in-out infinite;
}}
.gauge-arc-bg {{
  fill:none;stroke:{BORDER};stroke-width:14;stroke-linecap:round;opacity:0.4;
}}
.gauge-arc-fill {{
  fill:none;stroke:url(#gaugeGrad);stroke-width:14;stroke-linecap:round;
  stroke-dasharray:380;stroke-dashoffset:380;
  animation: gaugeArcDraw 1.8s cubic-bezier(0.25,1,0.5,1) 0.3s forwards;
}}
.gauge-needle {{
  animation: needleSwing 1.8s cubic-bezier(0.25,1,0.5,1) 0.3s forwards;
  transform-box: fill-box;
}}
.gauge-score {{
  animation: scoreCount 0.6s ease 1.5s both;
}}
</style>

<div class='nx-cascade-3'><div class='nx-panel' style='padding:14px 10px 10px;'>
<div class='nx-panel-title'>FUSION SIGNAL</div>
<svg class='gauge-svg' viewBox='0 0 300 195'>
  <defs>
    <linearGradient id='gaugeGrad' x1='0' y1='0' x2='1' y2='0'>
      <stop offset='0%' stop-color='{RED}'/>
      <stop offset='40%' stop-color='{AMBER}'/>
      <stop offset='100%' stop-color='{GREEN}'/>
    </linearGradient>
    <filter id='glow'>
      <feGaussianBlur stdDeviation='3' result='blur'/>
      <feMerge><feMergeNode in='blur'/><feMergeNode in='SourceGraphic'/></feMerge>
    </filter>
  </defs>

  <!-- zone arcs (background) -->
  <path class='gauge-arc-bg' d='M{ax1:.1f},{ay1:.1f} A{r_arc},{r_arc} 0 {large_arc},1 {ax2:.1f},{ay2:.1f}'/>

  <!-- colored fill arc -->
  <path class='gauge-arc-fill' d='M{ax1:.1f},{ay1:.1f} A{r_arc},{r_arc} 0 {large_arc},1 {ax2:.1f},{ay2:.1f}' filter='url(#glow)'/>

  <!-- tick marks -->
  {ticks_html}

  <!-- zone labels -->
  <text x='52' y='178' fill='{RED}' font-size='11' font-family='JetBrains Mono,monospace' font-weight='600' opacity='0.7'>SELL</text>
  <text x='139' y='46' fill='{AMBER}' font-size='11' font-family='JetBrains Mono,monospace' font-weight='600' opacity='0.7'>HOLD</text>
  <text x='222' y='178' fill='{GREEN}' font-size='11' font-family='JetBrains Mono,monospace' font-weight='600' opacity='0.7'>BUY</text>

  <!-- needle -->
  <g style='transform-origin:{cx}px {cy}px;animation:needleSwing 1.8s cubic-bezier(0.25,1,0.5,1) 0.3s forwards, needleIdle 3s ease-in-out 2.2s infinite;'>
    <!-- needle shadow -->
    <polygon points='{cx + 88},{cy} {cx + 6},{cy - 4.5} {cx + 6},{cy + 4.5}' fill='rgba(0,0,0,0.4)'/>
    <!-- needle body -->
    <polygon points='{cx + 85},{cy} {cx + 8},{cy - 4} {cx + 8},{cy + 4}' fill='{g_color}' filter='url(#glow)'/>
    <!-- center hub outer -->
    <circle cx='{cx}' cy='{cy}' r='10' fill='{BG}' stroke='{g_color}' stroke-width='2.5'/>
    <!-- center hub inner glow -->
    <circle cx='{cx}' cy='{cy}' r='5' fill='{g_color}' opacity='0.8'/>
    <circle cx='{cx}' cy='{cy}' r='2.5' fill='#fff' opacity='0.9'/>
  </g>

  <!-- center score -->
  <text class='gauge-score' x='{cx}' y='{cy + 35}' text-anchor='middle' fill='#fff' font-size='38' font-weight='900' font-family='JetBrains Mono,monospace'>{fusion_score:.0f}</text>
  <text class='gauge-score' x='{cx}' y='{cy + 52}' text-anchor='middle' fill='{g_color}' font-size='14' font-weight='700' font-family='Inter,sans-serif' letter-spacing='3'>{g_label}</text>
</svg>
</div>

</div></div>
""", unsafe_allow_html=True)

# ── Stats Column ──
with col_stats:
    pm_agree = "Yes" if dec.get('pm_agreement') else "No"
    pm_agree_clr = GREEN if dec.get('pm_agreement') else DIM
    
    st.markdown(f"""<div class='nx-cascade-3'><div class='nx-panel' style='min-height:230px;'>
<div class='nx-panel-title'>SIGNAL SUMMARY</div>
<div class='nx-row'><span class='nx-label'>Confidence</span><span class='nx-cyan' style='font-size:1.2rem;'>{conf:.0f}%</span></div>
<div class='nx-row'><span class='nx-label'>Caution</span><span class='nx-red' style='font-size:1.2rem;'>{caut:.0f}%</span></div>
<div class='nx-row'><span class='nx-label'>Agreement</span><span class='nx-cyan'>{dec['agreement']['agreement']}</span></div>
<div class='nx-row'><span class='nx-label'>PM Confirms</span><span style='color:{pm_agree_clr};font-weight:700;'>{pm_agree}</span></div>
<div class='nx-row'><span class='nx-label'>Weekly Trend</span><span class='nx-val'>{d['weekly_trend']}</span></div>
<div class='nx-row'><span class='nx-label'>BTC Leader</span><span class='nx-val'>{d['btc_trend']}</span></div>
</div></div>""", unsafe_allow_html=True)

# ── Breakdown Column ──
with col_breakdown:
    tech_s = dec.get('tech_confidence', 0)
    pm_conf_s = dec.get('pm_confirmation', 0)
    pm_qual_s = dec.get('pm_quality', 0)
    pm_confl_s = dec.get('pm_conflict', 0)
    caut_s = dec.get('caution', 0)

    def _score_bar(label, val, color, max_val=100):
        pct = min(abs(val) / max_val * 100, 100) if max_val > 0 else 0
        sign_clr = GREEN if val > 0 else RED if val < 0 else DIM
        return (f"<div class='nx-row'><span class='nx-label'>{label}</span>"
                f"<span style='color:{sign_clr};font-weight:600;font-size:1rem;'>{val:.1f}</span></div>"
                f"<div class='nx-bar-wrap'><div class='nx-bar' style='width:{pct}%;background:{color};'></div></div>")

    st.markdown(f"""<div class='nx-cascade-4'><div class='nx-panel' style='min-height:230px;'>
<div class='nx-panel-title'>FUSION BREAKDOWN</div>
{_score_bar('Tech ×0.55', tech_s, CYAN)}
{_score_bar('PM Conf ×0.25', pm_conf_s, PURPLE)}
{_score_bar('PM Qual ×0.10', pm_qual_s, GREEN)}
{_score_bar('PM Confl ×0.10', pm_confl_s, RED)}
{_score_bar('Caution ×0.25', caut_s, AMBER)}
</div></div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PRICE CHART (Collapsible)
# ═════════════════════════════════════════════════════════════════════════════
with st.expander("PRICE CHART", expanded=True):
    if pdf is not None and len(pdf) > 0:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            vertical_spacing=0.04,
                            row_heights=[0.6, 0.2, 0.2],
                            subplot_titles=(
                                f"{asset['ticker']}  ·  PRICE ACTION  ·  BOLLINGER BANDS",
                                "RSI (14)",
                                "MACD"
                            ))

        # Candles
        fig.add_trace(go.Candlestick(
            x=pdf.index, open=pdf['Open'], high=pdf['High'],
            low=pdf['Low'], close=pdf['Close'], name="Price",
            increasing_line_color=GREEN, increasing_fillcolor=GREEN,
            decreasing_line_color=RED, decreasing_fillcolor=RED,
        ), row=1, col=1)

        # BB
        fig.add_trace(go.Scatter(x=pdf.index, y=pdf['BB_Upper'], name='BB+',
                                 line=dict(color=CYAN, width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=pdf.index, y=pdf['BB_Lower'], name='BB-',
                                 line=dict(color=CYAN, width=1, dash='dot'),
                                 fill='tonexty', fillcolor='rgba(0,229,255,0.04)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=pdf.index, y=pdf['SMA_20'], name='SMA20',
                                 line=dict(color=AMBER, width=1)), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=pdf.index, y=pdf['RSI'], name='RSI',
                                 line=dict(color=PURPLE, width=1.5),
                                 fill='tozeroy', fillcolor='rgba(179,136,255,0.06)'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=RED, line_width=0.8, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=GREEN, line_width=0.8, row=2, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor='rgba(179,136,255,0.03)', line_width=0, row=2, col=1)

        # MACD
        fig.add_trace(go.Scatter(x=pdf.index, y=pdf['MACD'], name='MACD',
                                 line=dict(color=CYAN, width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=pdf.index, y=pdf['MACD_Signal'], name='Signal',
                                 line=dict(color=AMBER, width=1)), row=3, col=1)
        colors = [GREEN if v >= 0 else RED for v in pdf['MACD_Hist']]
        fig.add_trace(go.Bar(x=pdf.index, y=pdf['MACD_Hist'], name='Hist',
                             marker_color=colors, opacity=0.7), row=3, col=1)

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=PANEL,
            height=600, showlegend=False, xaxis_rangeslider_visible=False,
            margin=dict(l=60,r=20,t=40,b=25),
            font=dict(family='JetBrains Mono', color='#5a6272', size=14),
        )
        for ax in ['xaxis','xaxis2','xaxis3']:
            fig.update_layout(**{ax: dict(gridcolor='#111b30', showgrid=True, zeroline=False)})
        for ax in ['yaxis','yaxis2','yaxis3']:
            fig.update_layout(**{ax: dict(gridcolor='#111b30', showgrid=True, zeroline=False)})
        fig.update_annotations(font=dict(color=CYAN, size=15, family='Inter'))

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SIGNAL SOURCES (Collapsible)
# ═════════════════════════════════════════════════════════════════════════════
with st.expander("SIGNAL SOURCES", expanded=False):
    col_tech, col_pm, col_news = st.columns([1.2, 1, 1])

    # ── Technical Analysis (First - most important) ──
    with col_tech:
        if tech:
            sigs = tech.get('signals', {})

            def _sig_badge(val):
                if val > 0: return f"<span style='color:{GREEN};background:rgba(0,255,170,0.1);padding:2px 8px;border-radius:3px;font-size:1rem;font-weight:600;'>+{val}</span>"
                if val < 0: return f"<span style='color:{RED};background:rgba(255,61,90,0.1);padding:2px 8px;border-radius:3px;font-size:1rem;font-weight:600;'>{val}</span>"
                return f"<span style='color:{DIM};background:rgba(90,98,114,0.1);padding:2px 8px;border-radius:3px;font-size:1rem;'>0</span>"

            st.markdown(f"""<div class='nx-panel'>
<div class='nx-panel-title'>TECHNICAL ANALYSIS</div>
<div class='nx-row'><span class='nx-label'>Price</span><span class='nx-val'>${tech['price']:,.2f}</span></div>
<div class='nx-row'><span class='nx-label'>RSI (14)</span><span style='color:{PURPLE};font-weight:600;'>{tech['rsi']:.1f}</span></div>
<div class='nx-row'><span class='nx-label'>MACD Hist</span><span class='nx-val'>{tech.get('macd_hist',0):.4f}</span></div>
<div class='nx-row'><span class='nx-label'>Volatility</span><span class='nx-val'>{tech.get('volatility',0)*100:.2f}%</span></div>
<div class='nx-row'><span class='nx-label'>ATR</span><span class='nx-val'>{tech.get('atr',0):.2f}</span></div>
<div class='nx-row'><span class='nx-label'>Drawdown</span><span class='nx-red'>{tech.get('drawdown',0)*100:.1f}%</span></div>
<div style='border-top:1px solid {BORDER};margin-top:10px;padding-top:10px;'>
<div class='nx-panel-title' style='font-size:0.9rem;margin-bottom:6px;'>SIGNAL VECTORS</div>
<div style='display:grid;grid-template-columns:1fr 1fr;gap:4px 12px;'>
<div class='nx-row'><span class='nx-label'>Trend</span>{_sig_badge(sigs.get('trend',0))}</div>
<div class='nx-row'><span class='nx-label'>MACD</span>{_sig_badge(sigs.get('macd',0))}</div>
<div class='nx-row'><span class='nx-label'>RSI</span>{_sig_badge(sigs.get('rsi',0))}</div>
<div class='nx-row'><span class='nx-label'>Breakout</span>{_sig_badge(sigs.get('breakout',0))}</div>
<div class='nx-row' style='grid-column:span 2;'><span class='nx-label'>Momentum</span>{_sig_badge(sigs.get('momentum',0))}</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

    # ── Polymarket Signals ──
    with col_pm:
        pm_items = ""
        if pm['count'] > 0:
            for m in pm.get('markets', [])[:6]:
                direction = m.get('direction') or polymarket.tag_direction(m['question'], asset['ticker'])
                dclr = GREEN if direction=="bullish" else RED if direction=="bearish" else DIM
                icon = "&#9650;" if direction=="bullish" else "&#9660;" if direction=="bearish" else "&#8226;"
                odds = f"{m['yes_odds']*100:.0f}%" if m.get('yes_odds') else "—"
                q = m['question'][:50] + ('…' if len(m['question'])>50 else '')
                pm_items += f"""<div class='nx-news-item'>
<span style='color:{dclr};font-weight:700;font-size:1rem;'>{icon} {direction.upper()}</span>
<span style='color:#1a2744;'> │ </span>
<span style='color:{CYAN};font-size:1rem;font-weight:600;'>YES {odds}</span><br>
<span style='color:#8899b4;font-size:0.95rem;'>{q}</span>
</div>"""
        else:
            pm_items = f"<span style='color:{DIM};'>No markets found</span>"
        st.markdown(f"""<div class='nx-panel'><div class='nx-panel-title'>POLYMARKET</div>
{pm_items}
</div>""", unsafe_allow_html=True)

    # ── News Feed ──
    with col_news:
        news_items = ""
        if nws['count'] > 0:
            for a in nws.get('articles', [])[:6]:
                sclr = GREEN if a['sentiment']>=0.05 else RED if a['sentiment']<=-0.05 else DIM
                title = a['title'][:48] + ('…' if len(a['title'])>48 else '')
                news_items += f"""<div class='nx-news-item'>
<span style='color:{sclr};font-weight:700;font-size:1rem;'>{a['sentiment']:+.2f}</span>
<span style='color:#1a2744;'> │ </span>
<span style='color:#c0c8d8;font-size:0.95rem;'>{title}</span>
<br><span style='color:#3a4a6a;font-size:0.85rem;'>{a['source']}</span>
</div>"""
        else:
            news_items = f"<span style='color:{DIM};'>No news found</span>"
        st.markdown(f"""<div class='nx-panel'><div class='nx-panel-title'>NEWS SENTIMENT</div>
{news_items}
</div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EXTENDED ANALYSIS (Collapsible)
# ═════════════════════════════════════════════════════════════════════════════
_has_extended = d.get('backtest') is not None

if _has_extended:
    with st.expander("EXTENDED ANALYSIS", expanded=False):
        _tab_bt, _tab_sim, _tab_scen, _tab_diag = st.tabs([
            "📊 BACKTEST", "🔄 SIMULATOR", "📈 SCENARIOS", "🔧 DIAGNOSTICS"
        ])

        # ── Tab 1: Backtest Comparison ───────────────────────────────────────
        with _tab_bt:
            _bt = d['backtest']
            _bt_colors = [CYAN, PURPLE, GREEN]
            _bt_fig = go.Figure()
            for _i, (_name, _data) in enumerate(_bt.items()):
                _net = _data['net'].dropna()
                _bt_fig.add_trace(go.Scatter(
                    x=_net.index, y=_net * 100,
                    name=_name, line=dict(color=_bt_colors[_i % 3], width=2),
                ))
            _bt_fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=PANEL,
                height=380, margin=dict(l=60, r=20, t=30, b=30),
                font=dict(family='JetBrains Mono', color='#5a6272', size=13),
                yaxis_title='Cumulative Return %',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                xaxis=dict(gridcolor='#111b30'), yaxis=dict(gridcolor='#111b30'),
            )
            _bt_fig.add_hline(y=0, line_dash='dash', line_color=DIM, line_width=0.8)
            st.plotly_chart(_bt_fig, use_container_width=True, config={'displayModeBar': False})

            _bt_summary = backtest.summarize_backtests(_bt)
            if not _bt_summary.empty:
                st.dataframe(_bt_summary, use_container_width=True, hide_index=True)

        # ── Tab 2: Trade Simulator ───────────────────────────────────────────
        with _tab_sim:
            _sim_wo = d.get('sim_without_pm')
            _sim_w = d.get('sim_with_pm')
            if _sim_wo and _sim_w:
                _sim_fig = go.Figure()
                if not _sim_wo['equity_df'].empty:
                    _sim_fig.add_trace(go.Scatter(
                        x=_sim_wo['equity_df']['timestamp'],
                        y=_sim_wo['equity_df']['equity'],
                        name='Without PM', line=dict(color=CYAN, width=2),
                    ))
                if not _sim_w['equity_df'].empty:
                    _sim_fig.add_trace(go.Scatter(
                        x=_sim_w['equity_df']['timestamp'],
                        y=_sim_w['equity_df']['equity'],
                        name='With PM', line=dict(color=PURPLE, width=2),
                    ))
                _sim_fig.add_hline(y=backtest.SIM_INITIAL_CASH, line_dash='dash',
                                   line_color=DIM, line_width=0.8,
                                   annotation_text='$10K Start')
                _sim_fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=PANEL,
                    height=380, margin=dict(l=60, r=20, t=30, b=30),
                    font=dict(family='JetBrains Mono', color='#5a6272', size=13),
                    yaxis_title='Equity ($)',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    xaxis=dict(gridcolor='#111b30'), yaxis=dict(gridcolor='#111b30'),
                )
                st.plotly_chart(_sim_fig, use_container_width=True, config={'displayModeBar': False})

                _sim_summary = pd.DataFrame([_sim_wo['summary'], _sim_w['summary']])
                st.dataframe(_sim_summary, use_container_width=True, hide_index=True)

                # Round-trip table (most recent)
                _rt = _sim_w['roundtrip_df']
                if not _rt.empty:
                    st.markdown(f"<div style='color:{CYAN};font-size:0.95rem;font-weight:600;"
                                f"letter-spacing:2px;margin-top:8px;'>RECENT ROUND TRIPS (With PM)</div>",
                                unsafe_allow_html=True)
                    st.dataframe(_rt.tail(10), use_container_width=True, hide_index=True)

        # ── Tab 3: Scenario Analysis ─────────────────────────────────────────
        with _tab_scen:
            _sc = d.get('scenario_table')
            if _sc:
                _sc_df = pd.DataFrame(_sc)
                _base = dec['confidence']

                _sc_fig = go.Figure()
                _sc_colors = []
                for _, _row in _sc_df.iterrows():
                    _v = _row['Adjusted Confidence']
                    _sc_colors.append(GREEN if _v >= 65 else AMBER if _v >= 40 else RED)

                _sc_fig.add_trace(go.Bar(
                    x=_sc_df['Scenario'], y=_sc_df['Adjusted Confidence'],
                    marker_color=_sc_colors, text=_sc_df['Adjusted Confidence'].round(1),
                    textposition='outside', textfont=dict(color='#8899b4', size=13),
                ))
                _sc_fig.add_hline(y=_base, line_dash='dash', line_color=CYAN, line_width=1,
                                  annotation_text=f'Current: {_base:.0f}')
                _sc_fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=PANEL,
                    height=350, margin=dict(l=60, r=20, t=30, b=30),
                    font=dict(family='JetBrains Mono', color='#5a6272', size=13),
                    yaxis_title='Confidence', yaxis_range=[0, 110],
                    xaxis=dict(gridcolor='#111b30'), yaxis=dict(gridcolor='#111b30'),
                )
                st.plotly_chart(_sc_fig, use_container_width=True, config={'displayModeBar': False})
                st.dataframe(_sc_df, use_container_width=True, hide_index=True)

        # ── Tab 4: Fusion Diagnostics ────────────────────────────────────────
        with _tab_diag:
            _tc = dec.get('tech_confidence', 0)
            _pmc = dec.get('pm_confirmation', 0)
            _pmq = dec.get('pm_quality', 0)
            _pmx = dec.get('pm_conflict', 0)
            _cau = dec.get('caution', 0)

            _waterfall_labels = [
                'TechConf ×0.55', 'PMConfirm ×0.25', 'PMQuality ×0.10',
                'PMConflict ×−0.10', 'Caution ×−0.25', 'Final Confidence'
            ]
            _waterfall_vals = [
                0.55 * _tc, 0.25 * _pmc, 0.10 * _pmq,
                -0.10 * _pmx, -0.25 * _cau
            ]
            _final = max(0, min(100, sum(_waterfall_vals)))
            _waterfall_vals.append(_final)

            _wf_colors = [
                GREEN if v > 0 else RED if v < 0 else DIM
                for v in _waterfall_vals[:-1]
            ] + [CYAN]

            _wf_fig = go.Figure(go.Bar(
                x=_waterfall_labels, y=_waterfall_vals,
                marker_color=_wf_colors,
                text=[f'{v:+.1f}' if i < 5 else f'{v:.1f}' for i, v in enumerate(_waterfall_vals)],
                textposition='outside', textfont=dict(color='#8899b4', size=13),
            ))
            _wf_fig.add_hline(y=0, line_color=DIM, line_width=0.8)
            _wf_fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=PANEL,
                height=350, margin=dict(l=60, r=20, t=30, b=60),
                font=dict(family='JetBrains Mono', color='#5a6272', size=13),
                yaxis_title='Contribution to Confidence',
                xaxis=dict(gridcolor='#111b30'), yaxis=dict(gridcolor='#111b30'),
            )
            st.plotly_chart(_wf_fig, use_container_width=True, config={'displayModeBar': False})

            # CautionScore breakdown
            _cd = dec.get('caution_detail', {})
            if _cd:
                _caut_items = [
                    ('VolSpike ×0.24', _cd.get('vol_spike', 0), AMBER),
                    ('Drawdown ×0.18', _cd.get('drawdown_score', 0), RED),
                    ('SpreadStress ×0.16', _cd.get('spread_stress', 0), PURPLE),
                    ('ProbWhipsaw ×0.14', _cd.get('prob_whipsaw', 0), CYAN),
                    ('EventRisk ×0.14', _cd.get('event_risk', 0), GREEN),
                    ('Divergence ×0.14', _cd.get('divergence', 0), DIM),
                ]
                _caut_html = ""
                for _lbl, _val, _clr in _caut_items:
                    _pct = min(_val, 100)
                    _caut_html += (
                        f"<div class='nx-row'><span class='nx-label'>{_lbl}</span>"
                        f"<span style='color:{_clr};font-weight:600;font-size:1.05rem;'>{_val:.1f}</span></div>"
                        f"<div class='nx-bar-wrap'><div class='nx-bar' style='width:{_pct}%;background:{_clr};'></div></div>"
                    )
                st.markdown(f"""<div class='nx-panel' style='margin-top:8px;'>
<div class='nx-panel-title'>CAUTION SCORE BREAKDOWN</div>
{_caut_html}
</div>""", unsafe_allow_html=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(f"""<div style='text-align:center;padding:14px;margin-top:12px;border-top:1px solid {BORDER};'>
<span style='color:#2a3552;font-size:0.95rem;letter-spacing:2px;font-family:Inter,sans-serif;'>
NEXUS TERMINAL v3.0 &nbsp;&#9670;&nbsp; RULE-BASED DECISION ENGINE &nbsp;&#9670;&nbsp; {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</span>
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ORACLE AI — Floating Customer-Service-Style Side Panel
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.oracle_active:

    @st.fragment
    def _oracle_panel_main():
        # Marker for CSS targeting
        st.markdown("<div class='oracle-float-marker' style='display:none'></div>", unsafe_allow_html=True)

        # ── Inject scoped CSS for chat styling ──
        st.markdown(f"""<style>
        @keyframes oraclePulse {{ 0%,100% {{ box-shadow:0 0 8px rgba(179,136,255,0.4); }} 50% {{ box-shadow:0 0 20px rgba(179,136,255,0.8),0 0 40px rgba(0,229,255,0.3); }} }}
        @keyframes oracleGradient {{ 0% {{ background-position:0% 50%; }} 50% {{ background-position:100% 50%; }} 100% {{ background-position:0% 50%; }} }}
        @keyframes dotPulse {{ 0%,80%,100% {{ opacity:0.2;transform:scale(0.8); }} 40% {{ opacity:1;transform:scale(1); }} }}
        .oracle-avatar {{ animation: oraclePulse 3s ease-in-out infinite; }}
        .oracle-header-bar {{ background: linear-gradient(270deg,rgba(179,136,255,0.15),rgba(0,229,255,0.1),rgba(179,136,255,0.15)); background-size:400% 400%; animation: oracleGradient 8s ease infinite; }}
        .oracle-dot {{ display:inline-block;width:6px;height:6px;border-radius:50%;background:{PURPLE};margin:0 2px;animation:dotPulse 1.4s infinite ease-in-out; }}
        .oracle-dot:nth-child(2) {{ animation-delay:0.2s; }}
        .oracle-dot:nth-child(3) {{ animation-delay:0.4s; }}
        .oracle-user-msg {{ background:linear-gradient(135deg,rgba(0,229,255,0.18),rgba(0,229,255,0.06)); border:1px solid rgba(0,229,255,0.25); padding:10px 14px; border-radius:14px 14px 4px 14px; max-width:82%; color:#eef2f7; font-size:0.85rem; line-height:1.5; word-wrap:break-word; }}
        .oracle-ai-msg {{ background:linear-gradient(135deg,rgba(179,136,255,0.1),rgba(12,16,30,0.8)); border:1px solid rgba(179,136,255,0.18); padding:10px 14px; border-radius:4px 14px 14px 14px; max-width:82%; color:#cdd4e0; font-size:0.85rem; line-height:1.6; word-wrap:break-word; }}
        </style>""", unsafe_allow_html=True)

        # ── Header ──
        st.markdown(f"""
<div class='oracle-header-bar' style='padding:12px 16px;display:flex;align-items:center;gap:12px;
border-bottom:1px solid rgba(179,136,255,0.15);margin:0 -12px 0;border-radius:14px 14px 0 0;'>
    <div class='oracle-avatar' style='width:36px;height:36px;background:linear-gradient(135deg,{PURPLE},{CYAN});
    border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0;'>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
    </div>
    <div style='flex:1;'>
        <div style='color:#fff;font-size:0.9rem;font-weight:700;letter-spacing:1.5px;font-family:Inter,sans-serif;'>ORACLE AI</div>
        <div style='color:rgba(179,136,255,0.7);font-size:0.68rem;letter-spacing:0.5px;'>Groq · LLaMA 3.3 70B</div>
    </div>
    <div style='display:flex;align-items:center;gap:6px;padding:4px 10px;background:rgba(0,255,170,0.08);border:1px solid rgba(0,255,170,0.2);border-radius:20px;'>
        <div style='width:6px;height:6px;background:{GREEN};border-radius:50%;box-shadow:0 0 8px {GREEN};'></div>
        <span style='color:{GREEN};font-size:0.68rem;font-weight:600;letter-spacing:1px;'>LIVE</span>
    </div>
</div>""", unsafe_allow_html=True)

        # ── Chat area ──
        chat_container = st.container(height=320)
        with chat_container:
            if not st.session_state.oracle_messages:
                st.markdown(f"""
<div style='text-align:center;padding:40px 20px 20px;'>
    <div style='width:56px;height:56px;margin:0 auto 16px;background:linear-gradient(135deg,{PURPLE},{CYAN});
    border-radius:50%;display:flex;align-items:center;justify-content:center;box-shadow:0 0 30px rgba(179,136,255,0.3);'>
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.5"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
    </div>
    <div style='color:#fff;font-size:0.95rem;font-weight:600;margin-bottom:4px;'>How can I help?</div>
    <div style='color:{DIM};font-size:0.78rem;line-height:1.5;'>Ask about signals, risk, backtests, or Polymarket data.</div>
</div>""", unsafe_allow_html=True)
                _suggestions = ["Explain my signal", "Why is caution high?", "Polymarket outlook?"]
                _scols = st.columns(len(_suggestions))
                for _si, _sq in enumerate(_suggestions):
                    if _scols[_si].button(_sq, key=f"sug_main_{_si}", use_container_width=True):
                        st.session_state.oracle_messages.append({"role": "user", "content": _sq})
                        st.session_state.oracle_pending = True
                        st.rerun()
            else:
                for msg in st.session_state.oracle_messages:
                    if msg['role'] == 'user':
                        st.markdown(f"""
<div style='display:flex;justify-content:flex-end;margin:8px 4px;'>
    <div class='oracle-user-msg'>{msg['content']}</div>
</div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
<div style='display:flex;justify-content:flex-start;margin:8px 4px;gap:8px;'>
    <div style='width:24px;height:24px;min-width:24px;background:linear-gradient(135deg,{PURPLE},{CYAN});border-radius:50%;
    display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:4px;'>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
    </div>
    <div class='oracle-ai-msg'>{msg['content']}</div>
</div>""", unsafe_allow_html=True)

        # ── Float + Scroll JS ──
        import streamlit.components.v1 as _oscroll
        _oscroll.html("""<script>
(function(){
  function floatAndScroll(attempts) {
    try {
      var doc = window.parent.document;
      var marker = doc.querySelector('.oracle-float-marker');
      if (marker) {
        var el = marker;
        for (var i = 0; i < 15; i++) {
          el = el.parentElement;
          if (!el) break;
          var tid = el.getAttribute('data-testid');
          if (tid === 'stVerticalBlock' || tid === 'stVerticalBlockBorderWrapper') {
            if (el.offsetHeight > 100) {
              el.style.cssText = 'position:fixed !important;bottom:20px;right:20px;width:380px;max-height:560px;z-index:99999;background:rgba(6,10,20,0.97);border:1px solid rgba(179,136,255,0.25);border-radius:16px;box-shadow:0 12px 48px rgba(0,0,0,0.7),0 0 40px rgba(179,136,255,0.08),0 0 80px rgba(0,229,255,0.04);backdrop-filter:blur(24px);padding:0 0 8px;overflow-y:auto;overflow-x:hidden;';
              var bw = el.querySelectorAll('[data-testid="stVerticalBlockBorderWrapper"]');
              bw.forEach(function(b){ b.style.border='none'; b.style.background='transparent'; });
              break;
            }
          }
        }
        var sc = (el || marker).querySelectorAll('[data-testid="stScrollableContainer"]');
        if (!sc.length) sc = doc.querySelectorAll('[data-testid="stScrollableContainer"]');
        sc.forEach(function(c){ if(c.scrollHeight > c.clientHeight + 30) c.scrollTop = c.scrollHeight + 9999; });
      }
    } catch(e){}
    if (attempts > 0) {
      requestAnimationFrame(function(){ setTimeout(function(){ floatAndScroll(attempts - 1); }, 150); });
    }
  }
  setTimeout(function(){ floatAndScroll(10); }, 100);
})();
</script>""", height=0)

        # Process pending LLM call
        if st.session_state.oracle_pending:
            with chat_container:
                st.markdown(f"""
<div style='display:flex;align-items:center;gap:8px;margin:8px 4px;'>
    <div style='width:24px;height:24px;min-width:24px;background:linear-gradient(135deg,{PURPLE},{CYAN});border-radius:50%;
    display:flex;align-items:center;justify-content:center;flex-shrink:0;'>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
    </div>
    <div class='oracle-ai-msg' style='padding:12px 16px;'>
        <span class='oracle-dot'></span><span class='oracle-dot'></span><span class='oracle-dot'></span>
    </div>
</div>""", unsafe_allow_html=True)
            try:
                response = _ask_oracle(st.session_state.oracle_messages, st.session_state.data)
                st.session_state.oracle_messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.session_state.oracle_messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
            st.session_state.oracle_pending = False
            st.rerun()

        # Chat input
        user_query = st.chat_input("Ask Oracle anything...", key="oracle_chat_main")
        if user_query:
            st.session_state.oracle_messages.append({"role": "user", "content": user_query})
            st.session_state.oracle_pending = True
            st.rerun()

        if st.session_state.oracle_messages:
            if st.button("🗑 Clear", key="clear_oracle", use_container_width=True):
                st.session_state.oracle_messages = []
                st.rerun()

    _oracle_panel_main()
