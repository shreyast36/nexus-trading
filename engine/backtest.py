"""
Backtest & Trade Simulator
==========================
Ports the notebook's 3-variant backtest comparison and 10K risk-managed
trade simulator with and without Polymarket overlay.

Backtest Variants:
  1. Technical Only — trade on tech signals alone
  2. Technical + PM Filter — only enter when PM agrees
  3. Technical + PM Sizing — scale position by PM confidence

Trade Simulator:
  Long-only 10K strategy with stop-loss, take-profit, slippage, fees.
  Two runs: without PM overlay vs with PM overlay (entry filter + exit logic).
"""

import numpy as np
import pandas as pd
from config import THRESHOLDS


# =============================================================================
# SIMULATION PARAMETERS (matching notebook)
# =============================================================================

SIM_INITIAL_CASH = 10_000.0
SIM_RISK_PER_TRADE = 0.015
SIM_MAX_POSITION = 0.08
SIM_STOP_LOSS_PCT = 0.003
SIM_TARGET_PCT = 0.01
SIM_SLIPPAGE_PCT = 0.0005   # 5 bps per side
SIM_FEE_PCT = 0.001          # 10 bps per side
SIM_ROLLING_WINDOW = 70
SIM_PM_MIN_CONFIDENCE = float(THRESHOLDS['confidence_trade'])
SIM_PM_EXIT_CONFIDENCE = max(20.0, SIM_PM_MIN_CONFIDENCE - 10.0)


# =============================================================================
# HELPERS
# =============================================================================

def safe_float(value, default=0.0):
    if pd.isna(value):
        return default
    return float(value)


def position_size_units(cash, entry_price, stop_loss_pct, risk_per_trade):
    """Risk-based position sizing: risk a fixed fraction of cash."""
    risk_amount = cash * risk_per_trade
    stop_distance = entry_price * stop_loss_pct
    return risk_amount / stop_distance if stop_distance > 0 else 0.0


# =============================================================================
# BACKTEST VARIANTS
# =============================================================================

def run_backtest_variants(feature_data, fused_data, fee_bps=5):
    """
    Run 3 backtest variants (cumulative return series).

    Args:
        feature_data: Output of technical.build_feature_lab()
        fused_data: Output of fusion.fuse_dataframe()
        fee_bps: Transaction fee in basis points

    Returns:
        Dict of {variant_name: {'gross': Series, 'net': Series}}
    """
    fee = fee_bps / 10_000
    ret = feature_data['Return_1D'].shift(-1).fillna(0)

    direction = feature_data['BaseDirection']
    tech_signal = pd.Series(
        np.where(direction == 'Long', 1, np.where(direction == 'Short', -1, 0)),
        index=feature_data.index
    )

    position_size = (
        fused_data['PositionSize']
        if 'PositionSize' in fused_data.columns
        else pd.Series(0, index=feature_data.index)
    )
    final_action = (
        fused_data['FinalAction']
        if 'FinalAction' in fused_data.columns
        else pd.Series('No Trade', index=feature_data.index)
    )

    # Variant 1: Technical Only
    tech_net = (tech_signal * ret - fee * tech_signal.diff().abs().fillna(0)).cumsum()

    # Variant 2: Technical + PM Filter
    pm_filter = pd.Series(
        np.where(final_action.isin(['Long', 'Short']), tech_signal, 0),
        index=feature_data.index
    )
    pm_filter_net = (pm_filter * ret - fee * pm_filter.diff().abs().fillna(0)).cumsum()

    # Variant 3: Technical + PM Sizing
    pm_size = tech_signal * position_size
    pm_size_net = (pm_size * ret - fee * pm_size.diff().abs().fillna(0)).cumsum()

    return {
        'Technical Only': {'net': tech_net},
        'Technical + PM Filter': {'net': pm_filter_net},
        'Technical + PM Sizing': {'net': pm_size_net},
    }


def summarize_backtests(variants):
    """Create summary DataFrame from backtest variants."""
    rows = []
    for name, data in variants.items():
        net = data['net'].dropna()
        if len(net) == 0:
            continue
        total_ret = float(net.iloc[-1]) * 100
        peak = net.cummax()
        max_dd = float((net - peak).min()) * 100
        rows.append({
            'Strategy': name,
            'Total Return %': round(total_ret, 2),
            'Max Drawdown %': round(max_dd, 2),
        })
    return pd.DataFrame(rows)


# =============================================================================
# TRADE INPUT PREPARATION
# =============================================================================

def prepare_trade_input(feature_data, fused_data, rolling_window=None):
    """Build combined table for the trade simulator."""
    if rolling_window is None:
        rolling_window = min(SIM_ROLLING_WINDOW, max(5, len(feature_data) - 1))

    ti = feature_data.copy().sort_index()
    for col in ['FinalAction', 'PositionSize', 'FinalConfidence', 'RiskZone',
                'pm_hist_liq_weighted_sentiment', 'pm_hist_net_sentiment']:
        ti[col] = fused_data[col] if col in fused_data.columns else np.nan

    ti['rolling_mean'] = ti['Close'].rolling(rolling_window).mean()
    ti['signal'] = np.where(
        ti['Close'] > ti['rolling_mean'], 1,
        np.where(ti['Close'] < ti['rolling_mean'], -1, 0)
    )
    return ti


# =============================================================================
# ROUND-TRIP TABLE
# =============================================================================

def build_roundtrip_table(trades_df, initial_cash):
    """Compress sequential BUY fills into round trips ending on SELL."""
    roundtrips = []
    open_cost, open_qty, entry_time = 0.0, 0.0, None

    for _, t in trades_df.iterrows():
        if t['side'] == 'BUY':
            if open_qty == 0:
                entry_time = t['timestamp']
            open_cost += t['notional'] + t['fee']
            open_qty += t['qty']
        else:
            proceeds = t['notional'] - t['fee']
            roundtrips.append({
                'entry_time': entry_time,
                'exit_time': t['timestamp'],
                'qty': open_qty,
                'buy_cost': open_cost,
                'sell_proceeds': proceeds,
                'realized_pnl': proceeds - open_cost,
                'exit_reason': t.get('reason', 'SELL'),
            })
            open_cost, open_qty, entry_time = 0.0, 0.0, None

    cols = ['entry_time', 'exit_time', 'qty', 'buy_cost',
            'sell_proceeds', 'realized_pnl', 'exit_reason']
    if not roundtrips:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(roundtrips)
    df['cum_equity'] = initial_cash + df['realized_pnl'].cumsum()
    return df


# =============================================================================
# TRADE SIMULATOR
# =============================================================================

def run_trade_simulation(data, use_pm=False, label='Without PM'):
    """
    10K risk-managed long-only trade simulator.

    Walks bars sequentially, creates fills with slippage/fees,
    and tracks equity curve + round-trip P&L.

    Args:
        data: Output of prepare_trade_input()
        use_pm: Whether to apply PM overlay (entry filter, sizing, exit)
        label: Strategy label for output

    Returns:
        Dict with trades_df, roundtrip_df, equity_df, summary
    """
    cash = float(SIM_INITIAL_CASH)
    position, avg_entry = 0.0, None
    trades, equity_curve = [], []
    skipped = 0

    for ts, row in data.iterrows():
        if pd.isna(row.get('rolling_mean')):
            continue

        price = float(row['Close'])
        bar_high = float(row['High']) if pd.notna(row.get('High')) else price
        bar_low = float(row['Low']) if pd.notna(row.get('Low')) else price
        sig = int(row['signal'])
        equity_now = cash + position * price
        equity_curve.append((ts, equity_now))

        # Base size from risk-per-trade formula
        size = position_size_units(cash, price, SIM_STOP_LOSS_PCT, SIM_RISK_PER_TRADE)

        # Cap total exposure
        max_val = SIM_MAX_POSITION * equity_now
        if abs((position + size) * price) > max_val and price > 0:
            size = max(0, max_val / price - position)

        # PM overlay
        pm_multiplier, pm_entry_ok, pm_exit = 1.0, True, False
        if use_pm:
            pm_multiplier = np.clip(safe_float(row.get('PositionSize', 0), 0), 0, 1)
            pm_action = row.get('FinalAction', 'No Trade')
            pm_conf = safe_float(row.get('FinalConfidence', 0), 0)
            pm_risk = str(row.get('RiskZone', 'Tradeable'))
            pm_sent = safe_float(row.get('pm_hist_liq_weighted_sentiment', 0), 0)

            pm_entry_ok = (
                pm_action == 'Long'
                and pm_conf >= SIM_PM_MIN_CONFIDENCE
                and pm_multiplier > 0
                and pm_risk != 'Avoid'
            )
            size *= pm_multiplier

            if position > 0 and (
                pm_conf < SIM_PM_EXIT_CONFIDENCE
                or pm_risk == 'Avoid'
                or pm_sent < 0
            ):
                pm_exit = True

        # Stop / target checks
        stop_hit = target_hit = False
        if position > 0 and avg_entry is not None:
            if bar_low <= avg_entry * (1 - SIM_STOP_LOSS_PCT):
                stop_hit = True
            elif bar_high >= avg_entry * (1 + SIM_TARGET_PCT):
                target_hit = True

        # ── BUY ──────────────────────────────────────────────────────────
        if sig == 1 and size > 0 and (pm_entry_ok if use_pm else True):
            fill = price * (1 + SIM_SLIPPAGE_PCT)
            max_qty = cash / (fill * (1 + SIM_FEE_PCT)) if fill > 0 else 0
            size = min(size, max_qty)
            if size > 0:
                notional = fill * size
                fee = SIM_FEE_PCT * notional
                cash -= notional + fee
                old_pos = position
                position += size
                avg_entry = (
                    fill if old_pos == 0
                    else (avg_entry * old_pos + fill * size) / position
                )
                trades.append({
                    'timestamp': ts, 'strategy': label, 'side': 'BUY',
                    'reason': 'SIGNAL', 'fill_price': fill, 'qty': size,
                    'notional': notional, 'fee': fee, 'cash_after': cash,
                    'position_after': position, 'pm_multiplier': pm_multiplier,
                })
        elif sig == 1 and use_pm and not pm_entry_ok:
            skipped += 1

        # ── SELL ─────────────────────────────────────────────────────────
        elif position > 0 and (sig == -1 or stop_hit or target_hit
                               or (use_pm and pm_exit)):
            fill = price * (1 - SIM_SLIPPAGE_PCT)
            qty = position
            notional = fill * qty
            fee = SIM_FEE_PCT * notional
            cash += notional - fee
            reason = (
                'STOP' if stop_hit else
                'TARGET' if target_hit else
                'PM_EXIT' if (use_pm and pm_exit) else 'SIGNAL'
            )
            position, avg_entry = 0.0, None
            trades.append({
                'timestamp': ts, 'strategy': label, 'side': 'SELL',
                'reason': reason, 'fill_price': fill, 'qty': qty,
                'notional': notional, 'fee': fee, 'cash_after': cash,
                'position_after': position, 'pm_multiplier': pm_multiplier,
            })

    # Final equity point
    if not data.empty:
        fp = float(data['Close'].iloc[-1])
        equity_curve.append((data.index[-1], cash + position * fp))

    trades_df = (
        pd.DataFrame(trades) if trades
        else pd.DataFrame(columns=[
            'timestamp', 'strategy', 'side', 'reason', 'fill_price',
            'qty', 'notional', 'fee', 'cash_after', 'position_after',
            'pm_multiplier',
        ])
    )
    roundtrip_df = build_roundtrip_table(trades_df, SIM_INITIAL_CASH)
    equity_df = (
        pd.DataFrame(equity_curve, columns=['timestamp', 'equity'])
        .drop_duplicates(subset=['timestamp'], keep='last')
        if equity_curve
        else pd.DataFrame(columns=['timestamp', 'equity'])
    )

    final_eq = float(equity_df['equity'].iloc[-1]) if not equity_df.empty else SIM_INITIAL_CASH
    max_dd = (
        float((equity_df['equity'] / equity_df['equity'].cummax() - 1).min()) * 100
        if not equity_df.empty else 0
    )
    win_rate = (
        float((roundtrip_df['realized_pnl'] > 0).mean()) * 100
        if not roundtrip_df.empty else 0
    )

    return {
        'trades_df': trades_df,
        'roundtrip_df': roundtrip_df,
        'equity_df': equity_df,
        'summary': {
            'Strategy': label,
            'Final Equity': round(final_eq, 2),
            'Total Return %': round((final_eq - SIM_INITIAL_CASH) / SIM_INITIAL_CASH * 100, 2),
            'Max Drawdown %': round(max_dd, 2),
            'Round Trips': int(len(roundtrip_df)),
            'Win Rate %': round(win_rate, 1),
            'Skipped PM Entries': skipped,
        },
    }


# =============================================================================
# ITERATION-2 CONSTANTS
# =============================================================================

ITER2_RESOLUTION_WINDOWS = [3, 7, 14, 30]
ITER2_PROBABILITY_GRID = np.round(np.linspace(0.10, 0.90, 17), 2)
ITER2_PERP_HEDGE_RATIO = 0.50
ITER2_OPTION_FLOOR = -0.08
ITER2_OPTION_PREMIUM_PCT = 0.01
ITER2_PM_HEDGE_CONTRACT_COST = 0.08
ITER2_PM_EVENT_THRESHOLD = -0.05
ITER2_RETURN_GRID = np.linspace(-0.20, 0.20, 81)
ITER2_PORTFOLIO_WEIGHTS = {'BTC': 0.50, 'ETH': 0.30, 'SOL': 0.20}


# =============================================================================
# PM-RETURN BETA
# =============================================================================

def compute_pm_return_beta(feature_df, fused_df):
    """
    Historical correlation between PM probability changes and asset returns.
    beta = cov(asset_return, pm_delta) / var(pm_delta)
    """
    pm_delta = fused_df['pm_hist_mean_price'].diff().replace([np.inf, -np.inf], np.nan)
    asset_ret = feature_df['Return_1D'].replace([np.inf, -np.inf], np.nan)
    if pm_delta.notna().sum() > 5 and pm_delta.var(skipna=True) and pm_delta.var(skipna=True) > 0:
        return float(asset_ret.cov(pm_delta) / pm_delta.var(skipna=True))
    return 0.0


# =============================================================================
# PROBABILITY PAYOFF EXPLORER
# =============================================================================

def build_probability_payoff_grid(fused_df, pm_return_beta,
                                  horizons=None, probability_grid=None):
    """
    Grid of expected P&L across PM-probability outcomes and time horizons.
    """
    if horizons is None:
        horizons = ITER2_RESOLUTION_WINDOWS
    if probability_grid is None:
        probability_grid = ITER2_PROBABILITY_GRID

    latest = fused_df.iloc[-1]
    current_price = float(fused_df['Close'].iloc[-1])
    current_prob = safe_float(latest.get('pm_hist_mean_price', 0.5), 0.5)
    beta = float(pm_return_beta)

    base_units = position_size_units(SIM_INITIAL_CASH, current_price,
                                     SIM_STOP_LOSS_PCT, SIM_RISK_PER_TRADE)
    cap_units = (SIM_MAX_POSITION * SIM_INITIAL_CASH) / current_price if current_price > 0 else 0.0
    base_units = min(base_units, cap_units)
    pm_units = base_units * np.clip(safe_float(latest.get('PositionSize', 0), 0), 0, 1)

    rows = []
    for horizon in horizons:
        time_scale = np.sqrt(max(horizon, 1)) / np.sqrt(30)
        holding_cost = SIM_FEE_PCT * current_price * base_units * 2
        for future_prob in probability_grid:
            prob_delta = future_prob - current_prob
            exp_ret = beta * prob_delta * time_scale
            wo_pnl = base_units * current_price * exp_ret - holding_cost
            w_pnl = pm_units * current_price * exp_ret - SIM_FEE_PCT * current_price * pm_units * 2
            rows.append({
                'HorizonDays': horizon,
                'FuturePMProb': round(future_prob, 2),
                'ProbDelta': round(prob_delta, 3),
                'ImpliedReturn': round(exp_ret, 5),
                'WithoutPM_PnL': round(wo_pnl, 2),
                'WithPM_PnL': round(w_pnl, 2),
            })
    return pd.DataFrame(rows)


# =============================================================================
# HEDGE COMPARISON PROFILES
# =============================================================================

def build_hedge_profiles(fused_df, return_grid=None):
    """
    Side-by-side P&L of spot vs hedged strategies across return outcomes.
    """
    if return_grid is None:
        return_grid = ITER2_RETURN_GRID

    latest = fused_df.iloc[-1]
    current_price = float(fused_df['Close'].iloc[-1])
    base_units = position_size_units(SIM_INITIAL_CASH, current_price,
                                     SIM_STOP_LOSS_PCT, SIM_RISK_PER_TRADE)
    cap_units = (SIM_MAX_POSITION * SIM_INITIAL_CASH) / current_price if current_price > 0 else 0.0
    base_units = min(base_units, cap_units)
    pm_units = base_units * np.clip(safe_float(latest.get('PositionSize', 0), 0), 0, 1)
    current_prob = safe_float(latest.get('pm_hist_mean_price', 0.5), 0.5)

    rows = []
    for ar in return_grid:
        spot = base_units * current_price * ar
        perp = spot - ITER2_PERP_HEDGE_RATIO * base_units * current_price * ar
        opt_floor = max(ar, ITER2_OPTION_FLOOR)
        option = base_units * current_price * opt_floor - ITER2_OPTION_PREMIUM_PCT * base_units * current_price
        bearish_event = 1.0 if ar <= ITER2_PM_EVENT_THRESHOLD else max(0.0, min(1.0, current_prob - ar))
        pm_overlay = spot + pm_units * current_price * max(0.0, bearish_event - ITER2_PM_HEDGE_CONTRACT_COST)
        rows.append({
            'AssetReturn': round(ar, 4),
            'Spot Only': round(spot, 2),
            'Perp Hedge': round(perp, 2),
            'Option Floor': round(option, 2),
            'PM Overlay': round(pm_overlay, 2),
        })
    return pd.DataFrame(rows)


# =============================================================================
# RESOLUTION WINDOW ANALYSIS
# =============================================================================

def build_resolution_window_table(fused_df, pm_return_beta, windows=None):
    """
    How confidence and expected return change as event resolution approaches.
    """
    if windows is None:
        windows = ITER2_RESOLUTION_WINDOWS

    latest = fused_df.iloc[-1]
    current_price = float(fused_df['Close'].iloc[-1])
    beta = float(pm_return_beta)
    current_prob = safe_float(latest.get('pm_hist_mean_price', 0.5), 0.5)

    base_units = position_size_units(SIM_INITIAL_CASH, current_price,
                                     SIM_STOP_LOSS_PCT, SIM_RISK_PER_TRADE)
    cap_units = (SIM_MAX_POSITION * SIM_INITIAL_CASH) / current_price if current_price > 0 else 0.0
    base_units = min(base_units, cap_units)
    pm_units = base_units * np.clip(safe_float(latest.get('PositionSize', 0), 0), 0, 1)

    rows = []
    max_w = max(windows) if windows else 30
    for w in windows:
        time_scale = np.sqrt(max(w, 1)) / np.sqrt(30)
        adj_conf = float(np.clip(
            latest['FinalConfidence'] - (w / max_w) * 8
            + latest['pm_hist_liq_weighted_sentiment'] * 12,
            0, 100,
        ))
        exp_ret = beta * (current_prob - 0.5) * time_scale
        rows.append({
            'Window (days)': w,
            'Adj Confidence': round(adj_conf, 1),
            'Implied Return': round(exp_ret, 5),
            'Without PM PnL': round(base_units * current_price * exp_ret, 2),
            'With PM PnL': round(pm_units * current_price * exp_ret, 2),
            'Size Multiplier': round(np.clip(pm_units / base_units if base_units > 0 else 0, 0, 1), 2),
        })
    return pd.DataFrame(rows)


# =============================================================================
# CSV EXPORT
# =============================================================================

def export_scan_csv(scan_data: dict) -> dict:
    """
    Build a dict of {filename: DataFrame} from scan results for CSV export.
    """
    from pathlib import Path
    exports = {}

    if scan_data.get('feature_df') is not None:
        exports['feature_data'] = scan_data['feature_df']
    if scan_data.get('fused_df') is not None:
        exports['fused_data'] = scan_data['fused_df']
    if scan_data.get('backtest'):
        summary = summarize_backtests(scan_data['backtest'])
        if not summary.empty:
            exports['backtest_summary'] = summary
    for key in ('sim_without_pm', 'sim_with_pm'):
        sim = scan_data.get(key)
        if sim:
            label = key.replace('sim_', '')
            if not sim['trades_df'].empty:
                exports[f'{label}_trades'] = sim['trades_df']
            if not sim['roundtrip_df'].empty:
                exports[f'{label}_roundtrips'] = sim['roundtrip_df']
            if not sim['equity_df'].empty:
                exports[f'{label}_equity'] = sim['equity_df']
            exports[f'{label}_summary'] = pd.DataFrame([sim['summary']])
    if scan_data.get('scenario_table'):
        exports['scenario_analysis'] = pd.DataFrame(scan_data['scenario_table'])
    if scan_data.get('payoff_grid') is not None:
        exports['probability_payoff'] = scan_data['payoff_grid']
    if scan_data.get('hedge_profiles') is not None:
        exports['hedge_profiles'] = scan_data['hedge_profiles']
    if scan_data.get('resolution_table') is not None:
        exports['resolution_windows'] = scan_data['resolution_table']
    return exports
