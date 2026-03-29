"""
Predictive Market Decision Engine
=================================
A rule-based crypto trading decision system.

Modules:
    config      - Configuration and thresholds
    technical   - Price data and technical indicators
    polymarket  - Prediction market sentiment
    news        - RSS news sentiment
    fusion      - Signal combination
    display     - Formatted output

Usage:
    python main.py
"""

from . import config
from . import technical
from . import polymarket
from . import news
from . import fusion
from . import display
