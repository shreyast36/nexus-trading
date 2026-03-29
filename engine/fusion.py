"""
Fusion Engine
=============
Combines signals from Technical Analysis, Polymarket, and News into a final decision.

Implements the notebook's fusion formula:
  FinalConfidence = 0.55·TechConf + 0.25·PMConfirm + 0.10·PMQuality
                    − 0.10·PMConflict − 0.25·CautionScore

CautionScore (6 components):
  0.24·VolSpike + 0.18·Drawdown + 0.16·SpreadStress
  + 0.14·ProbWhipsaw + 0.14·EventRisk + 0.14·Divergence

Additional features kept from the dashboard engine:
- Signal Agreement scoring (for display)
- Multi-Timeframe Confluence (weekly trend)
- BTC Market Leader Check (altcoin penalty)

Output:
- Final Confidence (0-100)
- Caution Score (0-100) with component breakdown
- Risk Zone (Tradeable/Cautious/High Risk/Avoid)
- Position Size (0%, 25%, 50%, 75%, 100%)
- Recommended Action (LONG/SHORT/NO TRADE)
"""

import numpy as np
import pandas as pd
from config import THRESHOLDS, FUSION_WEIGHTS


# =============================================================================
# UTILITIES
# =============================================================================

def min_max_scale(series):
    """Min-max normalize a Series to 0-1, handling inf/NaN."""
    series = pd.Series(series).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    if series.nunique(dropna=False) <= 1:
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / (series.max() - series.min())


# =============================================================================
# SIGNAL AGREEMENT
# =============================================================================

def compute_agreement(tech_direction: str, pm_sentiment: float, news_sentiment: float) -> dict:
    """
    Score how well the three signal sources agree.
    
    Agreement Levels:
        High Conviction: All 3 agree → +20 confidence boost
        Moderate Conviction: 2 of 3 agree → +10 confidence boost
        Conflicting: Mixed signals → +25 caution penalty
        
    Args:
        tech_direction: "Long", "Short", or "Flat"
        pm_sentiment: Polymarket sentiment (-1 to +1)
        news_sentiment: News sentiment (-1 to +1)
        
    Returns:
        Dict with agreement label, score, and boost/penalty
    """
    # Convert to directional signals (-1, 0, +1)
    tech_dir = 1 if tech_direction == "Long" else (-1 if tech_direction == "Short" else 0)
    pm_dir = 1 if pm_sentiment > 0.05 else (-1 if pm_sentiment < -0.05 else 0)
    news_dir = 1 if news_sentiment > 0.05 else (-1 if news_sentiment < -0.05 else 0)
    
    signals = [tech_dir, pm_dir, news_dir]
    non_zero = [s for s in signals if s != 0]
    
    # All neutral
    if len(non_zero) == 0:
        return {"agreement": "Neutral", "score": 0, "boost": 0, "caution_add": 0}
    
    # Check if all non-zero signals agree
    if len(non_zero) >= 2 and all(s == non_zero[0] for s in non_zero):
        if len(non_zero) == 3:
            return {"agreement": "High Conviction", "score": 3, "boost": 20, "caution_add": 0}
        return {"agreement": "Moderate Conviction", "score": 2, "boost": 10, "caution_add": 0}
    
    # Check for conflict (both +1 and -1 present)
    if 1 in non_zero and -1 in non_zero:
        return {"agreement": "Conflicting", "score": -1, "boost": 0, "caution_add": 25}
    
    return {"agreement": "Mixed", "score": 1, "boost": 0, "caution_add": 0}


# =============================================================================
# CAUTION SCORE
# =============================================================================

def compute_caution(tech_data: dict, pm_data: dict, news_data: dict) -> dict:
    """
    Calculate CautionScore using the notebook's 6-component formula.
    
    Components:
        VolSpike (24%): Volatility vs its 60-day mean
        DrawdownScore (18%): Depth of drawdown from peak
        SpreadStress (16%): PM bid-ask spread tightness
        ProbWhipsaw (14%): PM probability disagreement
        EventRisk (14%): Fraction of PM markets near resolution
        Divergence (14%): Tech vs PM directional disagreement
        
    Returns:
        Dict with total caution score and individual components
    """
    # -------------------------------------------------------------------------
    # VolSpike (0-100): already computed in technical.py
    # -------------------------------------------------------------------------
    vol_spike = 0.0
    if tech_data:
        vol_spike = float(tech_data.get('vol_spike', 0))
    
    # -------------------------------------------------------------------------
    # DrawdownScore (0-100): abs(drawdown) × 200
    # -------------------------------------------------------------------------
    drawdown_score = 0.0
    if tech_data:
        drawdown_score = float(tech_data.get('drawdown_score', 0))
    
    # -------------------------------------------------------------------------
    # SpreadStress (0-100): PM spread × 200
    # -------------------------------------------------------------------------
    spread_stress = 0.0
    if pm_data:
        spread_stress = float(np.clip(pm_data.get('spread_mean', 0) * 200, 0, 100))
    
    # -------------------------------------------------------------------------
    # ProbWhipsaw (0-100): PM dispersion × 300
    # -------------------------------------------------------------------------
    prob_whipsaw = 0.0
    if pm_data:
        prob_whipsaw = float(np.clip(pm_data.get('dispersion', 0) * 300, 0, 100))
    
    # -------------------------------------------------------------------------
    # EventRisk (0-100): Fraction of extreme-odds markets × 100
    # -------------------------------------------------------------------------
    event_risk = 0.0
    if pm_data:
        event_risk = float(np.clip(pm_data.get('event_risk', 0) * 100, 0, 100))
    
    # -------------------------------------------------------------------------
    # Divergence: 50 if tech and PM disagree on direction, else 0
    # -------------------------------------------------------------------------
    divergence = 0.0
    if tech_data and pm_data:
        tech_sign = 1 if tech_data.get('direction') == 'Long' else (-1 if tech_data.get('direction') == 'Short' else 0)
        pm_sign = 1 if pm_data.get('sentiment', 0) > 0 else (-1 if pm_data.get('sentiment', 0) < 0 else 0)
        if tech_sign != 0 and pm_sign != 0 and tech_sign != pm_sign:
            divergence = 50.0
    
    # -------------------------------------------------------------------------
    # Weighted CautionScore (notebook formula)
    # -------------------------------------------------------------------------
    total = float(np.clip(
        0.24 * vol_spike +
        0.18 * drawdown_score +
        0.16 * spread_stress +
        0.14 * prob_whipsaw +
        0.14 * event_risk +
        0.14 * divergence,
        0, 100
    ))
    
    return {
        "total": total,
        "vol_spike": vol_spike,
        "drawdown_score": drawdown_score,
        "spread_stress": spread_stress,
        "prob_whipsaw": prob_whipsaw,
        "event_risk": event_risk,
        "divergence": divergence,
    }


# =============================================================================
# MAIN FUSION FUNCTION
# =============================================================================

def fuse(
    tech_data: dict,
    pm_data: dict,
    news_data: dict,
    is_btc: bool = True,
    btc_trend: str = "Unknown",
    weekly_trend: str = "Unknown"
) -> dict:
    """
    Combine all signals into a final trading decision.
    
    Uses the notebook's fusion formula:
      FinalConfidence = 0.55·TechConf + 0.25·PMConfirm + 0.10·PMQuality
                        − 0.10·PMConflict − 0.25·CautionScore
    
    Process:
        1. Compute TechnicalConfidence from tech_score
        2. Compute PM Confirmation/Conflict/Quality from sentiment vs tech direction
        3. Compute CautionScore (6-component)
        4. Compute signal agreement (for display)
        5. Apply timeframe and BTC adjustments
        6. Calculate FinalConfidence
        7. Determine risk zone, position size, action
    """
    
    # =========================================================================
    # 1. TECHNICAL CONFIDENCE
    # =========================================================================
    
    tech_score = tech_data.get('tech_score', 0) if tech_data else 0
    tech_dir = tech_data.get('direction', 'Flat') if tech_data else 'Flat'
    tech_conf = float(np.clip(abs(tech_score) * 100, 0, 100))
    
    # =========================================================================
    # 2. PM CONFIRMATION / CONFLICT / QUALITY
    # =========================================================================
    
    pm_sent = pm_data.get('sentiment', 0) if pm_data else 0
    pm_qual = pm_data.get('quality', 0) if pm_data else 0
    
    tech_sign = 1 if tech_dir == "Long" else (-1 if tech_dir == "Short" else 0)
    pm_sign = 1 if pm_sent > 0 else (-1 if pm_sent < 0 else 0)
    
    same_dir = (tech_sign == pm_sign) and (tech_sign != 0)
    opp_dir = (tech_sign == -pm_sign) and (tech_sign != 0) and (pm_sign != 0)
    
    pm_confirmation = float(abs(pm_sent) * 100) if same_dir else 0.0
    pm_conflict = float(abs(pm_sent) * 100) if opp_dir else 0.0
    pm_quality = float(np.clip(pm_qual * 100, 0, 100))
    
    # =========================================================================
    # 3. CAUTION SCORE (6-component from notebook)
    # =========================================================================
    
    caution_detail = compute_caution(tech_data, pm_data, news_data)
    caution = caution_detail['total']
    
    # =========================================================================
    # 4. SIGNAL AGREEMENT (kept for display)
    # =========================================================================
    
    news_sent = news_data.get('sentiment', 0) if news_data else 0
    agreement = compute_agreement(tech_dir, pm_sent, news_sent)
    
    # =========================================================================
    # 5. MULTI-TIMEFRAME & BTC ADJUSTMENTS
    # =========================================================================
    
    timeframe_boost = 0
    timeframe_penalty = 0
    
    if weekly_trend != "Unknown":
        if (tech_dir == "Long" and weekly_trend == "Bullish") or \
           (tech_dir == "Short" and weekly_trend == "Bearish"):
            timeframe_boost = 8
        elif (tech_dir == "Long" and weekly_trend == "Bearish") or \
             (tech_dir == "Short" and weekly_trend == "Bullish"):
            timeframe_penalty = 10
    
    btc_penalty = 0
    if not is_btc and btc_trend != "Unknown":
        if tech_dir == "Long" and btc_trend == "Bearish":
            btc_penalty = 12
        elif tech_dir == "Short" and btc_trend == "Bullish":
            btc_penalty = 8
    
    # =========================================================================
    # 6. FINAL CONFIDENCE (notebook formula + dashboard extras)
    # =========================================================================
    
    confidence = float(np.clip(
        FUSION_WEIGHTS['technical'] * tech_conf +
        FUSION_WEIGHTS['pm_confirmation'] * pm_confirmation +
        FUSION_WEIGHTS['pm_quality'] * pm_quality +
        FUSION_WEIGHTS['pm_conflict'] * pm_conflict +
        FUSION_WEIGHTS['caution'] * caution +
        timeframe_boost -
        timeframe_penalty -
        btc_penalty,
        0, 100
    ))
    
    # =========================================================================
    # 7. RISK ZONE
    # =========================================================================
    
    if caution >= THRESHOLDS['caution_high']:
        risk_zone = "Avoid"
    elif caution >= THRESHOLDS['caution_medium']:
        risk_zone = "High Risk"
    elif caution >= THRESHOLDS['caution_low']:
        risk_zone = "Cautious"
    else:
        risk_zone = "Tradeable"
    
    # =========================================================================
    # 8. POSITION SIZE (5-tier from notebook)
    # =========================================================================
    
    if confidence < THRESHOLDS['confidence_trade'] or risk_zone == "Avoid":
        position_size = 0.0
    elif confidence < THRESHOLDS['confidence_quarter']:
        position_size = 0.25
    elif confidence < THRESHOLDS['confidence_half']:
        position_size = 0.50
    elif confidence < THRESHOLDS['confidence_three_quarter']:
        position_size = 0.75
    else:
        position_size = 1.0
    
    # Cap position size if BTC penalty active
    if btc_penalty > 0 and position_size > 0.5:
        position_size = 0.5
    
    # =========================================================================
    # 9. FINAL ACTION
    # =========================================================================
    
    if position_size > 0 and tech_dir in ["Long", "Short"]:
        action = tech_dir.upper()
    else:
        action = "NO TRADE"
    
    # =========================================================================
    # RETURN COMPLETE DECISION
    # =========================================================================
    
    return {
        "confidence": confidence,
        "caution": caution,
        "caution_detail": caution_detail,
        "risk_zone": risk_zone,
        "position_size": position_size,
        "action": action,
        "agreement": agreement,
        "pm_agreement": same_dir,
        "timeframe_boost": timeframe_boost,
        "timeframe_penalty": timeframe_penalty,
        "btc_penalty": btc_penalty,
        # Component scores for dashboard display
        "tech_confidence": tech_conf,
        "pm_confirmation": pm_confirmation,
        "pm_quality": pm_quality,
        "pm_conflict": pm_conflict,
        "tech_score": tech_score,
    }


# =============================================================================
# DATAFRAME-LEVEL FUSION (for backtesting)
# =============================================================================

def fuse_dataframe(feature_df, pm_data):
    """
    Apply fusion logic to every row of the feature DataFrame.
    PM snapshot data is broadcast across all rows.
    Used by the backtest engine and trade simulator.

    Args:
        feature_df: Output of technical.build_feature_lab()
        pm_data: Polymarket aggregate dict (from polymarket.analyze)

    Returns:
        DataFrame with FinalConfidence, CautionScore, RiskZone,
        PositionSize, FinalAction, etc. per row.
    """
    fused = feature_df.copy()

    pm_sent = pm_data.get('sentiment', 0) if pm_data else 0
    pm_qual = pm_data.get('quality', 0) if pm_data else 0
    pm_spread = pm_data.get('spread_mean', 0) if pm_data else 0
    pm_event = pm_data.get('event_risk', 0) if pm_data else 0
    pm_disp = pm_data.get('dispersion', 0) if pm_data else 0

    # Broadcast PM snapshot
    fused['pm_hist_liq_weighted_sentiment'] = pm_sent
    fused['pm_hist_net_sentiment'] = pm_sent
    fused['pm_hist_mean_price'] = 0.5
    fused['pm_hist_market_quality'] = pm_qual

    # CautionScore components
    fused['DrawdownComponent'] = np.clip(fused['Drawdown'].abs() * 200, 0, 100)
    fused['SpreadStress'] = float(np.clip(pm_spread * 200, 0, 100))
    fused['ProbWhipsaw'] = float(np.clip(pm_disp * 300, 0, 100))
    fused['EventRiskScore'] = float(np.clip(pm_event * 100, 0, 100))

    tech_sign = np.where(
        fused['BaseDirection'] == 'Long', 1,
        np.where(fused['BaseDirection'] == 'Short', -1, 0)
    )
    pm_s = 1 if pm_sent > 0 else (-1 if pm_sent < 0 else 0)
    fused['Divergence'] = np.where(
        (tech_sign != 0) & (tech_sign != pm_s) & (pm_s != 0), 50.0, 0.0
    )

    fused['CautionScore'] = np.clip(
        0.24 * fused['VolSpike'].fillna(0) +
        0.18 * fused['DrawdownComponent'] +
        0.16 * fused['SpreadStress'] +
        0.14 * fused['ProbWhipsaw'] +
        0.14 * fused['EventRiskScore'] +
        0.14 * fused['Divergence'],
        0, 100
    )

    # PM Confirmation / Conflict / Quality
    pm_abs = abs(pm_sent) * 100
    same = (tech_sign == pm_s) & (tech_sign != 0)
    opp = (tech_sign == -pm_s) & (tech_sign != 0) & (pm_s != 0)

    fused['PMConfirmationScore'] = np.where(same, pm_abs, 0.0)
    fused['PMConflictPenalty'] = np.where(opp, pm_abs, 0.0)
    fused['PMQualityScore'] = float(np.clip(pm_qual * 100, 0, 100))

    # FinalConfidence
    fused['FinalConfidence'] = np.clip(
        FUSION_WEIGHTS['technical'] * fused['TechnicalConfidence'] +
        FUSION_WEIGHTS['pm_confirmation'] * fused['PMConfirmationScore'] +
        FUSION_WEIGHTS['pm_quality'] * fused['PMQualityScore'] +
        FUSION_WEIGHTS['pm_conflict'] * fused['PMConflictPenalty'] +
        FUSION_WEIGHTS['caution'] * fused['CautionScore'],
        0, 100
    )

    # RiskZone
    fused['RiskZone'] = pd.cut(
        fused['CautionScore'],
        bins=[-np.inf, 30, 55, 75, np.inf],
        labels=['Tradeable', 'Cautious', 'High Risk', 'Avoid']
    )

    # PositionSize (5-tier)
    fused['PositionSize'] = np.select(
        [fused['FinalConfidence'] < 35,
         fused['FinalConfidence'] < 55,
         fused['FinalConfidence'] < 70,
         fused['FinalConfidence'] < 85],
        [0.0, 0.25, 0.50, 0.75],
        default=1.0
    )
    fused.loc[fused['RiskZone'] == 'Avoid', 'PositionSize'] = 0.0

    # PMAgreement
    fused['PMAgreement'] = same

    # FinalAction
    fused['FinalAction'] = np.where(
        (fused['PositionSize'] > 0) &
        (fused['BaseDirection'].isin(['Long', 'Short'])),
        fused['BaseDirection'], 'No Trade'
    )

    return fused


# =============================================================================
# SCENARIO ANALYSIS
# =============================================================================

SCENARIOS = {
    'Neutral':        {'pm_shift': 0,   'caution_adj': 0},
    'Bull Run':       {'pm_shift': 15,  'caution_adj': -10},
    'Flash Crash':    {'pm_shift': -20, 'caution_adj': 25},
    'Regulatory FUD': {'pm_shift': -10, 'caution_adj': 15},
    'ETF Approval':   {'pm_shift': 20,  'caution_adj': -5},
    'Exchange Hack':  {'pm_shift': -15, 'caution_adj': 30},
}


def scenario_adjusted_confidence(base_confidence: float, scenario_name: str) -> float:
    """
    Adjust FinalConfidence under a hypothetical scenario.
    """
    params = SCENARIOS.get(scenario_name, SCENARIOS['Neutral'])
    return max(0, min(100, base_confidence + params['pm_shift'] - params['caution_adj']))
