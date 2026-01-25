#!/usr/bin/env python3
"""
Crypto Strategy Backtest Engine
===============================
A complete backtesting solution for crypto trading strategies.

Features:
- Historical data fetching via CCXT (200+ exchanges)
- Technical indicators via pandas-ta
- Vectorized signal generation
- Portfolio simulation with stop-loss/take-profit
- Interactive Plotly HTML reports
- Runnable Python strategy code generation

Usage:
    python backtest.py --symbol BTC/USDT --timeframe 4h --days 365 \
        --entry "rsi<30,price<sma50" --exit "rsi>70" \
        --stop-loss 5 --take-profit 15 --output report.html
"""

import argparse
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta


# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    days: int = 365,
    exchange_id: str = "binance"
) -> pd.DataFrame:
    """Fetch historical OHLCV data from exchange."""
    
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({'enableRateLimit': True})
    
    # Calculate start time
    since = exchange.parse8601((datetime.utcnow() - timedelta(days=days)).isoformat())
    
    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < 1000:
            break
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    return df


# ============================================================================
# INDICATOR CALCULATION
# ============================================================================

def calculate_indicators(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Calculate all required technical indicators."""
    
    df = df.copy()
    
    # RSI
    df['rsi'] = ta.rsi(df['close'], length=config.get('rsi_period', 14))
    
    # MACD
    macd = ta.macd(
        df['close'],
        fast=config.get('macd_fast', 12),
        slow=config.get('macd_slow', 26),
        signal=config.get('macd_signal', 9)
    )
    if macd is not None:
        df['macd'] = macd.iloc[:, 0]
        df['macd_hist'] = macd.iloc[:, 1]
        df['macd_signal'] = macd.iloc[:, 2]
    
    # Moving Averages
    for period in [9, 21, 50, 100, 200]:
        df[f'sma{period}'] = ta.sma(df['close'], length=period)
        df[f'ema{period}'] = ta.ema(df['close'], length=period)
    
    # Bollinger Bands
    bb = ta.bbands(df['close'], length=config.get('bb_period', 20), std=config.get('bb_std', 2.0))
    if bb is not None:
        df['bb_upper'] = bb.iloc[:, 2]
        df['bb_middle'] = bb.iloc[:, 1]
        df['bb_lower'] = bb.iloc[:, 0]
    
    # ATR
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=config.get('atr_period', 14))
    
    # Volume MA
    df['volume_sma'] = ta.sma(df['volume'], length=20)
    
    return df


# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def parse_conditions(condition_str: str) -> List[Dict]:
    """Parse condition string into list of condition dicts.
    
    Examples:
        "rsi<30,price<sma50" -> [{'indicator': 'rsi', 'op': '<', 'value': 30}, ...]
        "rsi>70" -> [{'indicator': 'rsi', 'op': '>', 'value': 70}]
    """
    conditions = []
    if not condition_str:
        return conditions
    
    for cond in condition_str.split(','):
        cond = cond.strip().lower()
        
        # Match patterns like: rsi<30, price>sma50, macd_crossover
        match = re.match(r'(\w+)(>=|<=|>|<|==|=)(\w+)', cond)
        if match:
            indicator, op, value = match.groups()
            op = '==' if op == '=' else op
            
            # Check if value is numeric or another indicator
            try:
                value = float(value)
                conditions.append({'indicator': indicator, 'op': op, 'value': value})
            except ValueError:
                conditions.append({'indicator': indicator, 'op': op, 'ref': value})
        
        # Special patterns
        elif 'crossover' in cond or 'cross_above' in cond:
            parts = cond.replace('crossover', '').replace('cross_above', '').replace('_', '').strip()
            conditions.append({'type': 'crossover', 'indicator': parts or 'macd'})
        elif 'crossunder' in cond or 'cross_below' in cond:
            parts = cond.replace('crossunder', '').replace('cross_below', '').replace('_', '').strip()
            conditions.append({'type': 'crossunder', 'indicator': parts or 'macd'})
    
    return conditions


def evaluate_condition(df: pd.DataFrame, condition: Dict) -> pd.Series:
    """Evaluate a single condition across the dataframe."""
    
    if condition.get('type') == 'crossover':
        ind = condition['indicator']
        if 'macd' in ind:
            return (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        return pd.Series(False, index=df.index)
    
    if condition.get('type') == 'crossunder':
        ind = condition['indicator']
        if 'macd' in ind:
            return (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        return pd.Series(False, index=df.index)
    
    indicator = condition['indicator']
    op = condition['op']
    
    # Get indicator values
    if indicator == 'price':
        left = df['close']
    elif indicator in df.columns:
        left = df[indicator]
    else:
        return pd.Series(False, index=df.index)
    
    # Get comparison value
    if 'value' in condition:
        right = condition['value']
    elif 'ref' in condition:
        ref = condition['ref']
        if ref in df.columns:
            right = df[ref]
        else:
            return pd.Series(False, index=df.index)
    else:
        return pd.Series(False, index=df.index)
    
    # Evaluate
    if op == '<':
        return left < right
    elif op == '<=':
        return left <= right
    elif op == '>':
        return left > right
    elif op == '>=':
        return left >= right
    elif op == '==':
        return left == right
    
    return pd.Series(False, index=df.index)


def generate_signals(
    df: pd.DataFrame,
    entry_conditions: List[Dict],
    exit_conditions: List[Dict]
) -> pd.DataFrame:
    """Generate trading signals based on conditions."""
    
    df = df.copy()
    
    # Entry: ALL conditions must be true (AND)
    if entry_conditions:
        entry_signals = pd.Series(True, index=df.index)
        for cond in entry_conditions:
            entry_signals &= evaluate_condition(df, cond)
    else:
        entry_signals = pd.Series(False, index=df.index)
    
    # Exit: ANY condition can be true (OR)
    if exit_conditions:
        exit_signals = pd.Series(False, index=df.index)
        for cond in exit_conditions:
            exit_signals |= evaluate_condition(df, cond)
    else:
        exit_signals = pd.Series(False, index=df.index)
    
    df['entry_signal'] = entry_signals.astype(int)
    df['exit_signal'] = exit_signals.astype(int)
    
    return df


# ============================================================================
# PORTFOLIO SIMULATION
# ============================================================================

def simulate_portfolio(
    df: pd.DataFrame,
    initial_capital: float = 10000,
    position_size_pct: float = 10,
    stop_loss_pct: float = 5,
    take_profit_pct: float = 15,
    commission_pct: float = 0.1,
    slippage_pct: float = 0.05
) -> Dict:
    """Simulate portfolio with position management."""
    
    capital = initial_capital
    position = 0.0
    entry_price = 0.0
    trades = []
    equity_curve = []
    current_trade = None
    
    for i, (timestamp, row) in enumerate(df.iterrows()):
        price = row['close']
        
        # Check stop-loss / take-profit if in position
        if position > 0 and entry_price > 0:
            pnl_pct = (price - entry_price) / entry_price * 100
            
            # Stop loss
            if pnl_pct <= -stop_loss_pct:
                exit_price = entry_price * (1 - stop_loss_pct / 100)
                proceeds = position * exit_price * (1 - commission_pct / 100)
                capital += proceeds
                
                if current_trade:
                    current_trade['exit_time'] = timestamp
                    current_trade['exit_price'] = exit_price
                    current_trade['pnl_pct'] = -stop_loss_pct
                    current_trade['pnl_amount'] = proceeds - current_trade['cost']
                    current_trade['exit_reason'] = 'stop_loss'
                    trades.append(current_trade)
                
                position = 0
                entry_price = 0
                current_trade = None
            
            # Take profit
            elif pnl_pct >= take_profit_pct:
                exit_price = entry_price * (1 + take_profit_pct / 100)
                proceeds = position * exit_price * (1 - commission_pct / 100)
                capital += proceeds
                
                if current_trade:
                    current_trade['exit_time'] = timestamp
                    current_trade['exit_price'] = exit_price
                    current_trade['pnl_pct'] = take_profit_pct
                    current_trade['pnl_amount'] = proceeds - current_trade['cost']
                    current_trade['exit_reason'] = 'take_profit'
                    trades.append(current_trade)
                
                position = 0
                entry_price = 0
                current_trade = None
        
        # Process signals
        if row['entry_signal'] == 1 and position == 0:
            # Buy
            position_value = capital * position_size_pct / 100
            actual_price = price * (1 + slippage_pct / 100)
            cost = position_value * (1 + commission_pct / 100)
            
            if cost <= capital:
                position = position_value / actual_price
                entry_price = actual_price
                capital -= cost
                
                current_trade = {
                    'entry_time': timestamp,
                    'entry_price': actual_price,
                    'position_size': position,
                    'cost': cost
                }
        
        elif row['exit_signal'] == 1 and position > 0:
            # Sell on signal
            actual_price = price * (1 - slippage_pct / 100)
            proceeds = position * actual_price * (1 - commission_pct / 100)
            pnl_pct = (actual_price - entry_price) / entry_price * 100
            capital += proceeds
            
            if current_trade:
                current_trade['exit_time'] = timestamp
                current_trade['exit_price'] = actual_price
                current_trade['pnl_pct'] = pnl_pct
                current_trade['pnl_amount'] = proceeds - current_trade['cost']
                current_trade['exit_reason'] = 'signal'
                trades.append(current_trade)
            
            position = 0
            entry_price = 0
            current_trade = None
        
        # Record equity
        equity = capital + (position * price if position > 0 else 0)
        equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'price': price,
            'position': position
        })
    
    # Close remaining position
    if position > 0:
        final_price = df.iloc[-1]['close']
        proceeds = position * final_price * (1 - commission_pct / 100)
        pnl_pct = (final_price - entry_price) / entry_price * 100
        capital += proceeds
        
        if current_trade:
            current_trade['exit_time'] = df.index[-1]
            current_trade['exit_price'] = final_price
            current_trade['pnl_pct'] = pnl_pct
            current_trade['pnl_amount'] = proceeds - current_trade['cost']
            current_trade['exit_reason'] = 'end_of_backtest'
            trades.append(current_trade)
    
    return {
        'trades': trades,
        'equity_curve': equity_curve,
        'final_equity': equity_curve[-1]['equity'] if equity_curve else initial_capital,
        'initial_capital': initial_capital
    }


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

def calculate_metrics(results: Dict, df: pd.DataFrame) -> Dict:
    """Calculate comprehensive performance metrics."""
    
    equity_curve = results['equity_curve']
    trades = results['trades']
    initial_capital = results['initial_capital']
    final_equity = results['final_equity']
    
    # Basic returns
    total_return_pct = (final_equity - initial_capital) / initial_capital * 100
    
    # Trade statistics
    if trades:
        winning = [t for t in trades if t['pnl_pct'] > 0]
        losing = [t for t in trades if t['pnl_pct'] <= 0]
        
        total_trades = len(trades)
        win_rate = len(winning) / total_trades * 100 if total_trades > 0 else 0
        
        avg_win = np.mean([t['pnl_pct'] for t in winning]) if winning else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losing]) if losing else 0
        
        gross_profit = sum(t['pnl_amount'] for t in winning) if winning else 0
        gross_loss = abs(sum(t['pnl_amount'] for t in losing)) if losing else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    else:
        total_trades = win_rate = avg_win = avg_loss = 0
        profit_factor = 0
        winning = losing = []
    
    # Drawdown
    equities = [e['equity'] for e in equity_curve]
    peak = equities[0]
    max_drawdown = 0
    drawdowns = []
    
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        drawdowns.append(dd)
        if dd > max_drawdown:
            max_drawdown = dd
    
    # Sharpe Ratio (annualized, assuming 0% risk-free rate)
    if len(equities) > 1:
        returns = [(equities[i] - equities[i-1]) / equities[i-1] for i in range(1, len(equities))]
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Annualization factor depends on timeframe
        periods_per_year = 365 * 6  # Assume 4h default
        sharpe = (avg_return / std_return) * np.sqrt(periods_per_year) if std_return > 0 else 0
    else:
        sharpe = 0
    
    # Buy and hold comparison
    first_price = df.iloc[0]['close']
    last_price = df.iloc[-1]['close']
    buy_hold_return = (last_price - first_price) / first_price * 100
    
    return {
        'total_return_pct': round(total_return_pct, 2),
        'max_drawdown_pct': round(max_drawdown, 2),
        'sharpe_ratio': round(sharpe, 2),
        'total_trades': total_trades,
        'winning_trades': len(winning),
        'losing_trades': len(losing),
        'win_rate_pct': round(win_rate, 1),
        'avg_win_pct': round(avg_win, 2),
        'avg_loss_pct': round(avg_loss, 2),
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'Inf',
        'final_equity': round(final_equity, 2),
        'initial_capital': initial_capital,
        'buy_hold_return_pct': round(buy_hold_return, 2),
        'drawdowns': drawdowns
    }


# ============================================================================
# HTML REPORT GENERATION
# ============================================================================

def generate_html_report(
    df: pd.DataFrame,
    results: Dict,
    metrics: Dict,
    config: Dict
) -> str:
    """Generate beautiful interactive HTML report."""
    
    # Prepare data for charts
    timestamps = [str(t) for t in df.index.tolist()]
    
    # Equity curve data
    equity_data = results['equity_curve']
    equity_times = [str(e['timestamp']) for e in equity_data]
    equity_values = [e['equity'] for e in equity_data]
    
    # Trade markers
    trades = results['trades']
    buy_times = [str(t['entry_time']) for t in trades]
    buy_prices = [t['entry_price'] for t in trades]
    sell_times = [str(t['exit_time']) for t in trades]
    sell_prices = [t['exit_price'] for t in trades]
    
    # Determine result colors
    return_class = 'positive' if metrics['total_return_pct'] > 0 else 'negative'
    vs_bh = metrics['total_return_pct'] - metrics['buy_hold_return_pct']
    vs_bh_class = 'positive' if vs_bh > 0 else 'negative'
    
    # Format trades for table
    trades_html = ''
    for t in trades[-15:]:  # Last 15 trades
        pnl_class = 'positive' if t['pnl_pct'] > 0 else 'negative'
        trades_html += f'''
        <tr>
            <td>{str(t['entry_time'])[:16]}</td>
            <td>{str(t['exit_time'])[:16]}</td>
            <td>${t['entry_price']:,.2f}</td>
            <td>${t['exit_price']:,.2f}</td>
            <td class="{pnl_class}">{t['pnl_pct']:+.2f}%</td>
            <td class="exit-reason">{t['exit_reason']}</td>
        </tr>'''
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Strategy Backtest Report | {config.get('symbol', 'BTC/USDT')}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-void: #05070a;
            --bg-deep: #0a0e14;
            --bg-surface: #111820;
            --bg-elevated: #1a2332;
            --bg-hover: #243044;
            --text-primary: #e6edf3;
            --text-secondary: #7d8590;
            --text-muted: #484f58;
            --accent-cyan: #00d9ff;
            --accent-green: #00ff9d;
            --accent-red: #ff4757;
            --accent-gold: #ffd93d;
            --accent-purple: #a855f7;
            --gradient-cyan: linear-gradient(135deg, #00d9ff 0%, #0099ff 100%);
            --gradient-green: linear-gradient(135deg, #00ff9d 0%, #00cc7d 100%);
            --gradient-gold: linear-gradient(135deg, #ffd93d 0%, #ff9f43 100%);
            --border-subtle: rgba(255,255,255,0.06);
            --border-accent: rgba(0,217,255,0.3);
            --glow-cyan: 0 0 30px rgba(0,217,255,0.3);
            --glow-green: 0 0 30px rgba(0,255,157,0.3);
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Space Grotesk', -apple-system, sans-serif;
            background: var(--bg-void);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }}
        
        /* Animated background */
        .bg-pattern {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(ellipse at 20% 20%, rgba(0,217,255,0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(168,85,247,0.06) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 50%, rgba(0,255,157,0.04) 0%, transparent 70%);
            pointer-events: none;
            z-index: 0;
        }}
        
        .container {{
            position: relative;
            z-index: 1;
            max-width: 1400px;
            margin: 0 auto;
            padding: 40px 24px;
        }}
        
        /* Header */
        .header {{
            text-align: center;
            padding: 60px 0;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 48px;
        }}
        
        .header-badge {{
            display: inline-block;
            padding: 6px 16px;
            background: var(--bg-elevated);
            border: 1px solid var(--border-accent);
            border-radius: 20px;
            font-size: 0.75rem;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: var(--accent-cyan);
            margin-bottom: 24px;
        }}
        
        .header h1 {{
            font-size: clamp(2.5rem, 5vw, 4rem);
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 16px;
            background: linear-gradient(135deg, var(--text-primary) 0%, var(--accent-cyan) 50%, var(--accent-green) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .header-meta {{
            display: flex;
            justify-content: center;
            gap: 32px;
            flex-wrap: wrap;
            color: var(--text-secondary);
            font-size: 0.95rem;
        }}
        
        .header-meta span {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .header-meta .dot {{
            width: 6px;
            height: 6px;
            background: var(--accent-cyan);
            border-radius: 50%;
        }}
        
        /* Metrics Grid */
        .metrics-hero {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 48px;
        }}
        
        .metric-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 28px;
            text-align: center;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        
        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--gradient-cyan);
            opacity: 0;
            transition: opacity 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-4px);
            border-color: var(--border-accent);
            box-shadow: var(--glow-cyan);
        }}
        
        .metric-card:hover::before {{
            opacity: 1;
        }}
        
        .metric-card.hero {{
            background: linear-gradient(135deg, var(--bg-elevated) 0%, var(--bg-surface) 100%);
            border-color: var(--border-accent);
        }}
        
        .metric-card.hero::before {{
            opacity: 1;
        }}
        
        .metric-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 8px;
            letter-spacing: -0.02em;
        }}
        
        .metric-value.positive {{ color: var(--accent-green); }}
        .metric-value.negative {{ color: var(--accent-red); }}
        .metric-value.neutral {{ color: var(--accent-cyan); }}
        
        .metric-label {{
            color: var(--text-secondary);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .metric-sub {{
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid var(--border-subtle);
            font-size: 0.8rem;
            color: var(--text-muted);
        }}
        
        /* Sections */
        .section {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 32px;
            margin-bottom: 32px;
        }}
        
        .section-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border-subtle);
        }}
        
        .section-icon {{
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--bg-elevated);
            border-radius: 10px;
            font-size: 1.2rem;
        }}
        
        .section h2 {{
            font-size: 1.25rem;
            font-weight: 600;
        }}
        
        /* Strategy Rules */
        .strategy-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 24px;
        }}
        
        .rule-block {{
            background: var(--bg-elevated);
            border-radius: 12px;
            padding: 24px;
        }}
        
        .rule-block h3 {{
            font-size: 0.9rem;
            color: var(--accent-cyan);
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .rule-block ul {{
            list-style: none;
        }}
        
        .rule-block li {{
            padding: 8px 0;
            border-bottom: 1px solid var(--border-subtle);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
        }}
        
        .rule-block li:last-child {{
            border-bottom: none;
        }}
        
        /* Charts */
        .chart-container {{
            background: var(--bg-deep);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 24px;
        }}
        
        /* Trades Table */
        .trades-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        
        .trades-table th {{
            text-align: left;
            padding: 16px;
            background: var(--bg-elevated);
            color: var(--text-secondary);
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.75rem;
        }}
        
        .trades-table th:first-child {{
            border-radius: 8px 0 0 8px;
        }}
        
        .trades-table th:last-child {{
            border-radius: 0 8px 8px 0;
        }}
        
        .trades-table td {{
            padding: 14px 16px;
            border-bottom: 1px solid var(--border-subtle);
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .trades-table tr:hover td {{
            background: var(--bg-hover);
        }}
        
        .trades-table .positive {{ color: var(--accent-green); }}
        .trades-table .negative {{ color: var(--accent-red); }}
        
        .exit-reason {{
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            background: var(--bg-elevated);
        }}
        
        /* Footer */
        .footer {{
            margin-top: 64px;
            padding: 48px;
            background: linear-gradient(135deg, var(--bg-surface) 0%, var(--bg-elevated) 100%);
            border: 1px solid var(--border-accent);
            border-radius: 24px;
            text-align: center;
        }}
        
        .footer-brand {{
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 12px;
            background: var(--gradient-cyan);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .footer-tagline {{
            color: var(--text-secondary);
            margin-bottom: 24px;
            font-size: 1.1rem;
        }}
        
        .footer-cta {{
            display: inline-block;
            padding: 14px 32px;
            background: var(--gradient-cyan);
            color: var(--bg-void);
            font-weight: 600;
            border-radius: 12px;
            text-decoration: none;
            transition: all 0.3s ease;
        }}
        
        .footer-cta:hover {{
            transform: scale(1.05);
            box-shadow: var(--glow-cyan);
        }}
        
        .footer-note {{
            margin-top: 24px;
            color: var(--text-muted);
            font-size: 0.85rem;
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .container {{ padding: 20px 16px; }}
            .header {{ padding: 40px 0; }}
            .header h1 {{ font-size: 2rem; }}
            .metric-value {{ font-size: 1.6rem; }}
            .section {{ padding: 20px; }}
        }}
    </style>
</head>
<body>
    <div class="bg-pattern"></div>
    
    <div class="container">
        <header class="header">
            <div class="header-badge">Backtest Report</div>
            <h1>{config.get('name', 'Trading Strategy')}</h1>
            <div class="header-meta">
                <span><div class="dot"></div>{config.get('symbol', 'BTC/USDT')}</span>
                <span><div class="dot"></div>{config.get('timeframe', '4h')} timeframe</span>
                <span><div class="dot"></div>{config.get('days', 365)} days</span>
                <span><div class="dot"></div>{metrics['total_trades']} trades</span>
            </div>
        </header>
        
        <div class="metrics-hero">
            <div class="metric-card hero">
                <div class="metric-value {return_class}">{metrics['total_return_pct']:+.1f}%</div>
                <div class="metric-label">Total Return</div>
                <div class="metric-sub">vs B&H: <span class="{vs_bh_class}">{vs_bh:+.1f}%</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">-{metrics['max_drawdown_pct']:.1f}%</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="metric-value neutral">{metrics['sharpe_ratio']:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if metrics['win_rate_pct'] > 50 else 'negative'}">{metrics['win_rate_pct']:.0f}%</div>
                <div class="metric-label">Win Rate</div>
                <div class="metric-sub">{metrics['winning_trades']}W / {metrics['losing_trades']}L</div>
            </div>
            <div class="metric-card">
                <div class="metric-value neutral">{metrics['profit_factor']}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
            <div class="metric-card">
                <div class="metric-value neutral">${metrics['final_equity']:,.0f}</div>
                <div class="metric-label">Final Equity</div>
                <div class="metric-sub">from ${metrics['initial_capital']:,.0f}</div>
            </div>
        </div>
        
        <section class="section">
            <div class="section-header">
                <div class="section-icon">📊</div>
                <h2>Equity Curve</h2>
            </div>
            <div class="chart-container" id="equity-chart"></div>
        </section>
        
        <section class="section">
            <div class="section-header">
                <div class="section-icon">📉</div>
                <h2>Drawdown</h2>
            </div>
            <div class="chart-container" id="drawdown-chart"></div>
        </section>
        
        <section class="section">
            <div class="section-header">
                <div class="section-icon">🎯</div>
                <h2>Strategy Rules</h2>
            </div>
            <div class="strategy-grid">
                <div class="rule-block">
                    <h3>📈 Entry Conditions (AND)</h3>
                    <ul>
                        {''.join(f'<li>{c}</li>' for c in config.get('entry_display', ['N/A']))}
                    </ul>
                </div>
                <div class="rule-block">
                    <h3>📉 Exit Conditions (OR)</h3>
                    <ul>
                        {''.join(f'<li>{c}</li>' for c in config.get('exit_display', ['N/A']))}
                    </ul>
                </div>
                <div class="rule-block">
                    <h3>⚙️ Risk Management</h3>
                    <ul>
                        <li>Stop Loss: {config.get('stop_loss', 5)}%</li>
                        <li>Take Profit: {config.get('take_profit', 15)}%</li>
                        <li>Position Size: {config.get('position_size', 10)}%</li>
                        <li>Commission: {config.get('commission', 0.1)}%</li>
                    </ul>
                </div>
            </div>
        </section>
        
        <section class="section">
            <div class="section-header">
                <div class="section-icon">📋</div>
                <h2>Recent Trades</h2>
            </div>
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>Entry</th>
                        <th>Exit</th>
                        <th>Entry Price</th>
                        <th>Exit Price</th>
                        <th>P&L</th>
                        <th>Reason</th>
                    </tr>
                </thead>
                <tbody>
                    {trades_html}
                </tbody>
            </table>
        </section>
        
        <section class="section">
            <div class="section-header">
                <div class="section-icon">📈</div>
                <h2>Price Chart with Signals</h2>
            </div>
            <div class="chart-container" id="price-chart"></div>
        </section>
        
        <footer class="footer">
            <div class="footer-brand">🚀 Crypto Backtest Skill</div>
            <div class="footer-tagline">几分钟验证你的交易策略想法</div>
            <a href="https://github.com/0xrikt/crypto-skills" class="footer-cta" target="_blank">
                ⭐ Star on GitHub
            </a>
            <div class="footer-note">
                截图分享你的回测结果，帮助更多人发现这个工具！<br>
                Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} • Past performance does not guarantee future results
            </div>
        </footer>
    </div>
    
    <script>
        const darkTheme = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#e6edf3', family: 'Space Grotesk' }},
            xaxis: {{ gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' }},
            yaxis: {{ gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' }}
        }};
        
        // Equity Chart
        Plotly.newPlot('equity-chart', [{{
            x: {json.dumps(equity_times)},
            y: {json.dumps(equity_values)},
            type: 'scatter',
            mode: 'lines',
            fill: 'tozeroy',
            fillcolor: 'rgba(0,217,255,0.1)',
            line: {{ color: '#00d9ff', width: 2 }},
            name: 'Portfolio Value'
        }}], {{
            ...darkTheme,
            height: 350,
            margin: {{ t: 20, r: 20, b: 40, l: 60 }},
            yaxis: {{ ...darkTheme.yaxis, title: 'Value ($)', tickformat: '$,.0f' }},
            shapes: [{{
                type: 'line',
                y0: {metrics['initial_capital']},
                y1: {metrics['initial_capital']},
                x0: 0, x1: 1,
                xref: 'paper',
                line: {{ dash: 'dash', color: 'rgba(255,255,255,0.3)', width: 1 }}
            }}]
        }}, {{ responsive: true }});
        
        // Drawdown Chart
        Plotly.newPlot('drawdown-chart', [{{
            x: {json.dumps(equity_times)},
            y: {json.dumps([-d for d in metrics['drawdowns']])},
            type: 'scatter',
            mode: 'lines',
            fill: 'tozeroy',
            fillcolor: 'rgba(255,71,87,0.2)',
            line: {{ color: '#ff4757', width: 1 }},
            name: 'Drawdown'
        }}], {{
            ...darkTheme,
            height: 200,
            margin: {{ t: 20, r: 20, b: 40, l: 60 }},
            yaxis: {{ ...darkTheme.yaxis, title: 'Drawdown %', tickformat: '.1f' }}
        }}, {{ responsive: true }});
        
        // Price Chart
        Plotly.newPlot('price-chart', [
            {{
                x: {json.dumps(timestamps)},
                open: {json.dumps(df['open'].tolist())},
                high: {json.dumps(df['high'].tolist())},
                low: {json.dumps(df['low'].tolist())},
                close: {json.dumps(df['close'].tolist())},
                type: 'candlestick',
                name: 'Price',
                increasing: {{ line: {{ color: '#00ff9d' }} }},
                decreasing: {{ line: {{ color: '#ff4757' }} }}
            }},
            {{
                x: {json.dumps(buy_times)},
                y: {json.dumps(buy_prices)},
                type: 'scatter',
                mode: 'markers',
                name: 'Buy',
                marker: {{ symbol: 'triangle-up', size: 14, color: '#00ff9d' }}
            }},
            {{
                x: {json.dumps(sell_times)},
                y: {json.dumps(sell_prices)},
                type: 'scatter',
                mode: 'markers',
                name: 'Sell',
                marker: {{ symbol: 'triangle-down', size: 14, color: '#ff4757' }}
            }}
        ], {{
            ...darkTheme,
            height: 500,
            margin: {{ t: 20, r: 20, b: 40, l: 60 }},
            xaxis: {{ ...darkTheme.xaxis, rangeslider: {{ visible: false }} }},
            yaxis: {{ ...darkTheme.yaxis, title: 'Price ($)', tickformat: '$,.0f' }},
            legend: {{ orientation: 'h', y: 1.1 }}
        }}, {{ responsive: true }});
    </script>
</body>
</html>'''
    
    return html


# ============================================================================
# CODE GENERATION
# ============================================================================

def generate_strategy_code(config: Dict, df: pd.DataFrame) -> str:
    """Generate runnable Python strategy code."""
    
    code = f'''#!/usr/bin/env python3
"""
{config.get('name', 'Trading Strategy')}
{'=' * len(config.get('name', 'Trading Strategy'))}

Auto-generated by Crypto Backtest Skill
https://github.com/0xrikt/crypto-skills

Asset: {config.get('symbol', 'BTC/USDT')}
Timeframe: {config.get('timeframe', '4h')}

Entry: {config.get('entry_str', 'N/A')}
Exit: {config.get('exit_str', 'N/A')}
Stop Loss: {config.get('stop_loss', 5)}%
Take Profit: {config.get('take_profit', 15)}%
"""

import ccxt
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta


# =============================================================================
# CONFIGURATION
# =============================================================================

SYMBOL = "{config.get('symbol', 'BTC/USDT')}"
TIMEFRAME = "{config.get('timeframe', '4h')}"
EXCHANGE = "binance"

# Risk Management
INITIAL_CAPITAL = {config.get('initial_capital', 10000)}
POSITION_SIZE_PCT = {config.get('position_size', 10)}  # % of portfolio per trade
STOP_LOSS_PCT = {config.get('stop_loss', 5)}
TAKE_PROFIT_PCT = {config.get('take_profit', 15)}
COMMISSION_PCT = {config.get('commission', 0.1)}


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_data(days: int = 365) -> pd.DataFrame:
    """Fetch historical OHLCV data."""
    exchange = getattr(ccxt, EXCHANGE)({{'enableRateLimit': True}})
    since = exchange.parse8601((datetime.utcnow() - timedelta(days=days)).isoformat())
    
    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < 1000:
            break
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df


# =============================================================================
# INDICATORS
# =============================================================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators."""
    df = df.copy()
    
    # RSI
    df['rsi'] = ta.rsi(df['close'], length=14)
    
    # MACD
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    if macd is not None:
        df['macd'] = macd.iloc[:, 0]
        df['macd_signal'] = macd.iloc[:, 2]
    
    # Moving Averages
    df['sma50'] = ta.sma(df['close'], length=50)
    df['ema21'] = ta.ema(df['close'], length=21)
    
    # Bollinger Bands
    bb = ta.bbands(df['close'], length=20, std=2.0)
    if bb is not None:
        df['bb_upper'] = bb.iloc[:, 2]
        df['bb_lower'] = bb.iloc[:, 0]
    
    return df


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate entry and exit signals."""
    df = df.copy()
    
    # Entry conditions: {config.get('entry_str', 'rsi<30')}
    entry = pd.Series(True, index=df.index)
    # TODO: Customize entry conditions
    entry &= df['rsi'] < 30  # Example
    
    # Exit conditions: {config.get('exit_str', 'rsi>70')}
    exit_signal = pd.Series(False, index=df.index)
    # TODO: Customize exit conditions
    exit_signal |= df['rsi'] > 70  # Example
    
    df['entry_signal'] = entry.astype(int)
    df['exit_signal'] = exit_signal.astype(int)
    
    return df


# =============================================================================
# BACKTEST
# =============================================================================

def backtest(df: pd.DataFrame) -> dict:
    """Run backtest simulation."""
    capital = INITIAL_CAPITAL
    position = 0.0
    entry_price = 0.0
    trades = []
    
    for timestamp, row in df.iterrows():
        price = row['close']
        
        # Check stop-loss / take-profit
        if position > 0:
            pnl_pct = (price - entry_price) / entry_price * 100
            
            if pnl_pct <= -STOP_LOSS_PCT:
                proceeds = position * price * (1 - COMMISSION_PCT / 100)
                capital += proceeds
                trades.append({{'pnl_pct': pnl_pct, 'reason': 'stop_loss'}})
                position = 0
            
            elif pnl_pct >= TAKE_PROFIT_PCT:
                proceeds = position * price * (1 - COMMISSION_PCT / 100)
                capital += proceeds
                trades.append({{'pnl_pct': pnl_pct, 'reason': 'take_profit'}})
                position = 0
        
        # Entry
        if row['entry_signal'] == 1 and position == 0:
            position_value = capital * POSITION_SIZE_PCT / 100
            position = position_value / price
            entry_price = price
            capital -= position_value * (1 + COMMISSION_PCT / 100)
        
        # Exit
        elif row['exit_signal'] == 1 and position > 0:
            proceeds = position * price * (1 - COMMISSION_PCT / 100)
            pnl_pct = (price - entry_price) / entry_price * 100
            capital += proceeds
            trades.append({{'pnl_pct': pnl_pct, 'reason': 'signal'}})
            position = 0
    
    # Close remaining
    if position > 0:
        capital += position * df.iloc[-1]['close']
    
    return {{
        'final_capital': capital,
        'return_pct': (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
        'total_trades': len(trades),
        'trades': trades
    }}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"Fetching {{SYMBOL}} data...")
    df = fetch_data(days=365)
    print(f"Got {{len(df)}} candles")
    
    print("Calculating indicators...")
    df = calculate_indicators(df)
    
    print("Generating signals...")
    df = generate_signals(df)
    
    print("Running backtest...")
    results = backtest(df)
    
    print("\\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"Final Capital: ${{results['final_capital']:,.2f}}")
    print(f"Return: {{results['return_pct']:+.2f}}%")
    print(f"Total Trades: {{results['total_trades']}}")
'''
    
    return code


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Crypto Strategy Backtest Engine')
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading pair')
    parser.add_argument('--timeframe', default='4h', help='Candle timeframe')
    parser.add_argument('--days', type=int, default=365, help='Backtest period in days')
    parser.add_argument('--exchange', default='binance', help='Exchange to fetch data from')
    parser.add_argument('--entry', default='rsi<30', help='Entry conditions (comma-separated)')
    parser.add_argument('--exit', default='rsi>70', help='Exit conditions (comma-separated)')
    parser.add_argument('--stop-loss', type=float, default=5, help='Stop loss percentage')
    parser.add_argument('--take-profit', type=float, default=15, help='Take profit percentage')
    parser.add_argument('--position-size', type=float, default=10, help='Position size percentage')
    parser.add_argument('--initial-capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--commission', type=float, default=0.1, help='Commission percentage')
    parser.add_argument('--output', default='report.html', help='Output HTML file')
    parser.add_argument('--name', default='Trading Strategy', help='Strategy name')
    
    args = parser.parse_args()
    
    print(f"🚀 Crypto Backtest Engine")
    print(f"{'='*50}")
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Period: {args.days} days")
    print(f"Entry: {args.entry}")
    print(f"Exit: {args.exit}")
    print()
    
    # Fetch data
    print("📊 Fetching historical data...")
    df = fetch_ohlcv(args.symbol, args.timeframe, args.days, args.exchange)
    print(f"   Got {len(df)} candles")
    
    # Calculate indicators
    print("📈 Calculating indicators...")
    indicator_config = {}
    df = calculate_indicators(df, indicator_config)
    
    # Generate signals
    print("🎯 Generating signals...")
    entry_conditions = parse_conditions(args.entry)
    exit_conditions = parse_conditions(args.exit)
    df = generate_signals(df, entry_conditions, exit_conditions)
    
    entry_count = df['entry_signal'].sum()
    exit_count = df['exit_signal'].sum()
    print(f"   Entry signals: {entry_count}")
    print(f"   Exit signals: {exit_count}")
    
    # Simulate portfolio
    print("💰 Running backtest...")
    results = simulate_portfolio(
        df,
        initial_capital=args.initial_capital,
        position_size_pct=args.position_size,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        commission_pct=args.commission
    )
    
    # Calculate metrics
    metrics = calculate_metrics(results, df)
    
    # Prepare config for report
    config = {
        'name': args.name,
        'symbol': args.symbol,
        'timeframe': args.timeframe,
        'days': args.days,
        'entry_str': args.entry,
        'exit_str': args.exit,
        'entry_display': [c.strip() for c in args.entry.split(',')],
        'exit_display': [c.strip() for c in args.exit.split(',')] + [f'Stop Loss: -{args.stop_loss}%', f'Take Profit: +{args.take_profit}%'],
        'stop_loss': args.stop_loss,
        'take_profit': args.take_profit,
        'position_size': args.position_size,
        'commission': args.commission,
        'initial_capital': args.initial_capital
    }
    
    # Generate HTML report
    print("📄 Generating report...")
    html = generate_html_report(df, results, metrics, config)
    
    output_path = Path(args.output)
    output_path.write_text(html)
    print(f"   Saved: {output_path.absolute()}")
    
    # Generate strategy code
    code_path = output_path.with_suffix('.py')
    code = generate_strategy_code(config, df)
    code_path.write_text(code)
    print(f"   Saved: {code_path.absolute()}")
    
    # Print results
    print()
    print(f"{'='*50}")
    print("📈 BACKTEST RESULTS")
    print(f"{'='*50}")
    print(f"Total Return:    {metrics['total_return_pct']:+.2f}%")
    print(f"Max Drawdown:    -{metrics['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    print(f"Win Rate:        {metrics['win_rate_pct']:.1f}%")
    print(f"Total Trades:    {metrics['total_trades']}")
    print(f"Profit Factor:   {metrics['profit_factor']}")
    print(f"Final Equity:    ${metrics['final_equity']:,.2f}")
    print(f"Buy & Hold:      {metrics['buy_hold_return_pct']:+.2f}%")
    print()
    
    vs_bh = metrics['total_return_pct'] - metrics['buy_hold_return_pct']
    if vs_bh > 0:
        print(f"✅ Strategy beats Buy & Hold by {vs_bh:+.2f}%")
    else:
        print(f"❌ Strategy underperforms Buy & Hold by {vs_bh:.2f}%")


if __name__ == '__main__':
    main()
