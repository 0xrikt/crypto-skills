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
- Multi-language support (en/zh)

Usage:
    python backtest.py --symbol BTC/USDT --timeframe 4h --days 365 \
        --entry "rsi<30,price<sma50" --exit "rsi>70" \
        --stop-loss 5 --take-profit 15 --output report.html --lang en
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
# LANGUAGE LABELS
# ============================================================================

LABELS = {
    'en': {
        'title': 'Strategy Backtest Report',
        'strategy_summary': 'Strategy Summary',
        'strategy_config': 'Strategy Configuration',
        'symbol': 'Symbol',
        'timeframe': 'Timeframe',
        'period': 'Period',
        'backtest_period': 'Backtest Period',
        'initial_capital': 'Initial Capital',
        'entry': 'Entry Conditions',
        'exit': 'Exit Conditions',
        'entry_all': 'Entry (ALL conditions must be met)',
        'exit_any': 'Exit (ANY condition triggers)',
        'stop_loss': 'Stop Loss',
        'take_profit': 'Take Profit',
        'position_size': 'Position Size',
        'commission': 'Commission',
        'risk_management': 'Risk Management',
        'performance_metrics': 'Performance Metrics',
        'total_return': 'Total Return',
        'sharpe_ratio': 'Sharpe Ratio',
        'max_drawdown': 'Max Drawdown',
        'win_rate': 'Win Rate',
        'total_trades': 'Total Trades',
        'profit_factor': 'Profit Factor',
        'avg_trade': 'Avg Trade',
        'best_trade': 'Best Trade',
        'worst_trade': 'Worst Trade',
        'equity_curve': 'Equity Curve',
        'price_signals': 'Price & Signals',
        'trade_pnl': 'Trade P&L Distribution',
        'indicators': 'Indicators',
        'tagline': 'Validate your trading ideas in minutes',
        'share_cta': 'Share your results to help others discover this tool!',
        'generated': 'Generated on',
        'disclaimer': 'Past performance ≠ future results',
        'trade_table_title': 'Trade History',
        'trade_no': '#',
        'trade_entry_date': 'Entry Date',
        'trade_exit_date': 'Exit Date',
        'trade_type': 'Type',
        'trade_entry_price': 'Entry Price',
        'trade_exit_price': 'Exit Price',
        'trade_pnl_label': 'P&L',
        'buy': 'Buy',
        'sell': 'Sell',
        'entry_signal': 'Entry Signal',
        'exit_signal': 'Exit Signal',
        'price': 'Price',
        'equity': 'Equity',
        'days': 'days',
    },
    'zh': {
        'title': '策略回测报告',
        'strategy_summary': '策略摘要',
        'strategy_config': '策略配置',
        'symbol': '交易对',
        'timeframe': '时间周期',
        'period': '回测周期',
        'backtest_period': '回测周期',
        'initial_capital': '初始资金',
        'entry': '入场条件',
        'exit': '出场条件',
        'entry_all': '入场条件（全部满足）',
        'exit_any': '出场条件（任一触发）',
        'stop_loss': '止损',
        'take_profit': '止盈',
        'position_size': '仓位大小',
        'commission': '手续费',
        'risk_management': '风险管理',
        'performance_metrics': '绩效指标',
        'total_return': '总收益',
        'sharpe_ratio': '夏普比率',
        'max_drawdown': '最大回撤',
        'win_rate': '胜率',
        'total_trades': '总交易次数',
        'profit_factor': '盈亏比',
        'avg_trade': '平均交易收益',
        'best_trade': '最佳交易',
        'worst_trade': '最差交易',
        'equity_curve': '资金曲线',
        'price_signals': '价格与信号',
        'trade_pnl': '交易盈亏分布',
        'indicators': '技术指标',
        'tagline': '几分钟验证你的交易策略想法',
        'share_cta': '截图分享你的回测结果，帮助更多人发现这个工具！',
        'generated': '生成时间',
        'disclaimer': '过往表现不代表未来收益',
        'trade_table_title': '交易记录',
        'trade_no': '序号',
        'trade_entry_date': '入场日期',
        'trade_exit_date': '出场日期',
        'trade_type': '类型',
        'trade_entry_price': '入场价格',
        'trade_exit_price': '出场价格',
        'trade_pnl_label': '盈亏',
        'buy': '买入',
        'sell': '卖出',
        'entry_signal': '入场信号',
        'exit_signal': '出场信号',
        'price': '价格',
        'equity': '资金',
        'days': '天',
    }
}


# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    days: int = 365,
    exchange_id: str = "okx"
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
    """Calculate comprehensive technical indicators.
    
    Indicators available after calculation:
    - Momentum: rsi, stoch_k, stoch_d, willr, cci, mfi, roc
    - Trend: sma{9,21,50,100,200}, ema{9,21,50,100,200}, adx, plus_di, minus_di
    - Volatility: bb_upper, bb_middle, bb_lower, bb_width, bb_pct, atr, atr_pct
    - Volume: volume_sma, volume_ratio, obv, obv_sma
    - Price Position: price_pct_from_high, price_pct_from_low, drawdown
    - Derived: price_change, price_pct_change, rsi_change, macd_change
    """
    
    df = df.copy()
    
    # ========== MOMENTUM INDICATORS ==========
    
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
    
    # Stochastic (KDJ)
    stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
    if stoch is not None:
        df['stoch_k'] = stoch.iloc[:, 0]
        df['stoch_d'] = stoch.iloc[:, 1]
    
    # Williams %R
    df['willr'] = ta.willr(df['high'], df['low'], df['close'], length=14)
    
    # CCI (Commodity Channel Index)
    df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20)
    
    # MFI (Money Flow Index) - RSI with volume
    df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
    
    # ROC (Rate of Change)
    df['roc'] = ta.roc(df['close'], length=10)
    df['roc_20'] = ta.roc(df['close'], length=20)
    
    # ========== TREND INDICATORS ==========
    
    # Moving Averages
    for period in [9, 21, 50, 100, 200]:
        df[f'sma{period}'] = ta.sma(df['close'], length=period)
        df[f'ema{period}'] = ta.ema(df['close'], length=period)
    
    # ADX (Average Directional Index) - Trend Strength
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    if adx is not None:
        df['adx'] = adx.iloc[:, 0]
        df['plus_di'] = adx.iloc[:, 1]  # +DI
        df['minus_di'] = adx.iloc[:, 2]  # -DI
    
    # ========== VOLATILITY INDICATORS ==========
    
    # Bollinger Bands
    bb = ta.bbands(df['close'], length=config.get('bb_period', 20), std=config.get('bb_std', 2.0))
    if bb is not None:
        df['bb_lower'] = bb.iloc[:, 0]
        df['bb_middle'] = bb.iloc[:, 1]
        df['bb_upper'] = bb.iloc[:, 2]
        df['bb_width'] = bb.iloc[:, 3] if bb.shape[1] > 3 else (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_pct'] = bb.iloc[:, 4] if bb.shape[1] > 4 else (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ATR
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=config.get('atr_period', 14))
    df['atr_pct'] = df['atr'] / df['close'] * 100  # ATR as percentage of price
    
    # ========== VOLUME INDICATORS ==========
    
    # Volume SMA and ratio
    df['volume_sma'] = ta.sma(df['volume'], length=20)
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # OBV (On-Balance Volume)
    df['obv'] = ta.obv(df['close'], df['volume'])
    df['obv_sma'] = ta.sma(df['obv'], length=20)
    
    # ========== PRICE POSITION INDICATORS ==========
    
    # Rolling High/Low (for drawdown and position calculation)
    for period in [20, 50, 90, 200]:
        df[f'high_{period}'] = df['high'].rolling(window=period).max()
        df[f'low_{period}'] = df['low'].rolling(window=period).min()
    
    # Drawdown from rolling high
    df['drawdown'] = (df['close'] - df['high_90']) / df['high_90'] * 100
    df['drawdown_50'] = (df['close'] - df['high_50']) / df['high_50'] * 100
    
    # Price position relative to range
    df['price_position_90'] = (df['close'] - df['low_90']) / (df['high_90'] - df['low_90'])
    
    # Distance from moving averages (percentage)
    df['dist_sma50'] = (df['close'] - df['sma50']) / df['sma50'] * 100
    df['dist_sma200'] = (df['close'] - df['sma200']) / df['sma200'] * 100
    
    # ========== DERIVED / CHANGE INDICATORS ==========
    
    # Price changes
    df['price_change'] = df['close'].diff()
    df['price_pct_change'] = df['close'].pct_change() * 100
    df['price_change_5'] = df['close'].diff(5)
    df['price_pct_change_5'] = df['close'].pct_change(5) * 100
    
    # Indicator changes (for "turning" detection)
    df['rsi_change'] = df['rsi'].diff()
    df['macd_change'] = df['macd'].diff() if 'macd' in df.columns else None
    df['macd_hist_change'] = df['macd_hist'].diff() if 'macd_hist' in df.columns else None
    
    # Consecutive conditions (count of consecutive up/down days)
    df['consecutive_up'] = df['close'].gt(df['close'].shift(1)).astype(int)
    df['consecutive_up'] = df['consecutive_up'].groupby((df['consecutive_up'] != df['consecutive_up'].shift()).cumsum()).cumsum()
    
    df['consecutive_down'] = df['close'].lt(df['close'].shift(1)).astype(int)
    df['consecutive_down'] = df['consecutive_down'].groupby((df['consecutive_down'] != df['consecutive_down'].shift()).cumsum()).cumsum()
    
    return df


# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def parse_conditions(condition_str: str) -> List[Dict]:
    """Parse condition string into list of condition dicts.
    
    Supported patterns:
    
    1. Simple comparison:
       "rsi<30", "price>sma50", "adx>=25"
    
    2. Percentage-based reference:
       "price<sma200_98pct" - Price below 98% of SMA200
       "price>sma50_105pct" - Price above 105% of SMA50
       "price<bb_lower" - Price below BB lower band
    
    3. Crossover/Crossunder:
       "macd_crossover" - MACD crosses above signal
       "ema9_cross_above_ema21" - EMA9 crosses above EMA21
       "price_crossunder_sma200" - Price crosses below SMA200
    
    4. Consecutive periods:
       "consecutive_up>=3" - 3+ consecutive up days
       "rsi<30_for_3" - RSI below 30 for 3 consecutive periods
    
    5. Change/Turning:
       "rsi_turning_up" - RSI is increasing (change > 0)
       "macd_hist_turning_down" - MACD histogram decreasing
    
    6. Percentile/Position:
       "bb_pct<0.2" - Price in lower 20% of BB range
       "price_position_90<0.3" - Price in lower 30% of 90-day range
    
    7. Distance from MA:
       "dist_sma200<-10" - Price 10% below SMA200
    """
    conditions = []
    if not condition_str:
        return conditions
    
    for cond in condition_str.split(','):
        cond = cond.strip().lower()
        
        # Pattern: indicator_cross_above_indicator2 or indicator_crossover_indicator2
        cross_match = re.match(r'(\w+)_(cross_?(?:over|above))_(\w+)', cond)
        if cross_match:
            ind1, _, ind2 = cross_match.groups()
            conditions.append({'type': 'crossover', 'indicator': ind1, 'ref': ind2})
            continue
        
        cross_under_match = re.match(r'(\w+)_(cross_?(?:under|below))_(\w+)', cond)
        if cross_under_match:
            ind1, _, ind2 = cross_under_match.groups()
            conditions.append({'type': 'crossunder', 'indicator': ind1, 'ref': ind2})
            continue
        
        # Pattern: indicator_turning_up or indicator_turning_down
        turning_match = re.match(r'(\w+)_turning_(up|down)', cond)
        if turning_match:
            indicator, direction = turning_match.groups()
            conditions.append({'type': 'turning', 'indicator': indicator, 'direction': direction})
            continue
        
        # Pattern: indicator<value_for_N (consecutive periods)
        consecutive_match = re.match(r'(\w+)(>=|<=|>|<|==)(\d+(?:\.\d+)?)_for_(\d+)', cond)
        if consecutive_match:
            indicator, op, value, periods = consecutive_match.groups()
            conditions.append({
                'type': 'consecutive',
                'indicator': indicator,
                'op': op,
                'value': float(value),
                'periods': int(periods)
            })
            continue
        
        # Pattern: simple comparison like rsi<30, price>sma50
        # Also supports percentage references: price<sma200_98pct, price>sma50_105pct
        match = re.match(r'(\w+)(>=|<=|>|<|==|=)(\w+(?:\.\d+)?)', cond)
        if match:
            indicator, op, value = match.groups()
            op = '==' if op == '=' else op
            
            # Check for percentage reference pattern: sma200_98pct, ema50_105pct
            pct_match = re.match(r'(\w+)_(\d+)pct', value)
            if pct_match:
                ref_indicator, pct = pct_match.groups()
                conditions.append({
                    'indicator': indicator, 
                    'op': op, 
                    'ref': ref_indicator,
                    'ref_pct': float(pct) / 100.0  # Convert 98 to 0.98
                })
                continue
            
            # Check if value is numeric or another indicator
            try:
                value = float(value)
                conditions.append({'indicator': indicator, 'op': op, 'value': value})
            except ValueError:
                conditions.append({'indicator': indicator, 'op': op, 'ref': value})
            continue
        
        # Legacy patterns for backwards compatibility
        if 'crossover' in cond or 'cross_above' in cond:
            parts = cond.replace('crossover', '').replace('cross_above', '').replace('_', '').strip()
            conditions.append({'type': 'crossover', 'indicator': parts or 'macd', 'ref': 'macd_signal' if not parts or 'macd' in parts else None})
        elif 'crossunder' in cond or 'cross_below' in cond:
            parts = cond.replace('crossunder', '').replace('cross_below', '').replace('_', '').strip()
            conditions.append({'type': 'crossunder', 'indicator': parts or 'macd', 'ref': 'macd_signal' if not parts or 'macd' in parts else None})
    
    return conditions


def evaluate_condition(df: pd.DataFrame, condition: Dict) -> pd.Series:
    """Evaluate a single condition across the dataframe.
    
    Supports:
    - Simple comparisons (<, >, <=, >=, ==)
    - Crossover/Crossunder between any two indicators
    - Turning up/down detection
    - Consecutive periods meeting condition
    """
    
    # Handle crossover between two indicators
    if condition.get('type') == 'crossover':
        ind = condition['indicator']
        ref = condition.get('ref')
        
        # Default ref for MACD
        if 'macd' in ind and ref is None:
            ref = 'macd_signal'
            ind = 'macd'
        
        # Get the two series
        if ind == 'price':
            series1 = df['close']
        elif ind in df.columns:
            series1 = df[ind]
        else:
            return pd.Series(False, index=df.index)
        
        if ref in df.columns:
            series2 = df[ref]
        else:
            return pd.Series(False, index=df.index)
        
        # Crossover: was below/equal, now above
        return (series1 > series2) & (series1.shift(1) <= series2.shift(1))
    
    # Handle crossunder
    if condition.get('type') == 'crossunder':
        ind = condition['indicator']
        ref = condition.get('ref')
        
        if 'macd' in ind and ref is None:
            ref = 'macd_signal'
            ind = 'macd'
        
        if ind == 'price':
            series1 = df['close']
        elif ind in df.columns:
            series1 = df[ind]
        else:
            return pd.Series(False, index=df.index)
        
        if ref in df.columns:
            series2 = df[ref]
        else:
            return pd.Series(False, index=df.index)
        
        return (series1 < series2) & (series1.shift(1) >= series2.shift(1))
    
    # Handle turning up/down
    if condition.get('type') == 'turning':
        ind = condition['indicator']
        direction = condition['direction']
        
        if ind in df.columns:
            series = df[ind]
        elif ind == 'price':
            series = df['close']
        else:
            return pd.Series(False, index=df.index)
        
        change = series.diff()
        
        if direction == 'up':
            # Turning up: current change > 0, previous change <= 0
            return (change > 0) & (change.shift(1) <= 0)
        else:
            # Turning down: current change < 0, previous change >= 0
            return (change < 0) & (change.shift(1) >= 0)
    
    # Handle consecutive periods
    if condition.get('type') == 'consecutive':
        ind = condition['indicator']
        op = condition['op']
        value = condition['value']
        periods = condition['periods']
        
        if ind in df.columns:
            series = df[ind]
        elif ind == 'price':
            series = df['close']
        else:
            return pd.Series(False, index=df.index)
        
        # Evaluate base condition
        if op == '<':
            base_cond = series < value
        elif op == '<=':
            base_cond = series <= value
        elif op == '>':
            base_cond = series > value
        elif op == '>=':
            base_cond = series >= value
        elif op == '==':
            base_cond = series == value
        else:
            return pd.Series(False, index=df.index)
        
        # Check if condition met for N consecutive periods
        # Rolling sum of True values, must equal periods
        consecutive_count = base_cond.astype(int).rolling(window=periods).sum()
        return consecutive_count >= periods
    
    # Standard comparison
    indicator = condition.get('indicator')
    op = condition.get('op')
    
    if not indicator or not op:
        return pd.Series(False, index=df.index)
    
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
            # Apply percentage multiplier if specified (e.g., sma200_98pct -> SMA200 * 0.98)
            if 'ref_pct' in condition:
                right = right * condition['ref_pct']
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
    config: Dict,
    lang: str = 'en'
) -> str:
    """Generate beautiful interactive HTML report.
    
    Args:
        df: DataFrame with OHLCV data and indicators
        results: Backtest results dict
        metrics: Performance metrics dict
        config: Strategy configuration
        lang: Language code ('en' or 'zh')
    """
    L = LABELS.get(lang, LABELS['en'])
    
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
        
        /* Strategy Summary */
        .strategy-summary {{
            border: 2px solid var(--accent-cyan);
            background: linear-gradient(135deg, var(--bg-surface) 0%, rgba(0, 217, 255, 0.05) 100%);
        }}
        
        .strategy-info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
            padding: 20px;
            background: var(--bg-elevated);
            border-radius: 12px;
        }}
        
        .info-item {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        
        .info-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
        }}
        
        .info-value {{
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
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
        
        .rule-block.entry-block {{
            border-left: 3px solid var(--accent-green);
        }}
        
        .rule-block.exit-block {{
            border-left: 3px solid var(--accent-red);
        }}
        
        .rule-block.risk-block {{
            border-left: 3px solid var(--accent-gold);
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
            font-size: 0.9rem;
        }}
        
        .rule-block li code {{
            font-family: 'JetBrains Mono', monospace;
            background: var(--bg-deep);
            padding: 2px 8px;
            border-radius: 4px;
            color: var(--accent-cyan);
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
            <div class="header-badge">{L['title']}</div>
            <h1>{config.get('name', 'Trading Strategy')}</h1>
            <div class="header-meta">
                <span><div class="dot"></div>{config.get('symbol', 'BTC/USDT')}</span>
                <span><div class="dot"></div>{config.get('timeframe', '4h')} {L['timeframe']}</span>
                <span><div class="dot"></div>{config.get('days', 365)} days</span>
                <span><div class="dot"></div>{metrics['total_trades']} {L['total_trades'].lower()}</span>
            </div>
        </header>
        
        <!-- Strategy Summary - The most important section -->
        <section class="section strategy-summary">
            <div class="section-header">
                <div class="section-icon">📋</div>
                <h2>{L['strategy_summary']}</h2>
            </div>
            
            <!-- Basic Info -->
            <div class="strategy-info-grid">
                <div class="info-item">
                    <span class="info-label">{L['symbol']}</span>
                    <span class="info-value">{config.get('symbol', 'BTC/USDT')}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">{L['timeframe']}</span>
                    <span class="info-value">{config.get('timeframe', '4h')}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">{L['backtest_period']}</span>
                    <span class="info-value">{config.get('days', 365)} {L['days']}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">{L['initial_capital']}</span>
                    <span class="info-value">${config.get('initial_capital', 10000):,.0f}</span>
                </div>
            </div>
            
            <!-- Entry/Exit Conditions -->
            <div class="strategy-grid">
                <div class="rule-block entry-block">
                    <h3>📈 {L['entry_all']}</h3>
                    <ul>
                        {''.join(f'<li><code>{c}</code></li>' for c in config.get('entry_display', ['N/A']))}
                    </ul>
                </div>
                <div class="rule-block exit-block">
                    <h3>📉 {L['exit_any']}</h3>
                    <ul>
                        {''.join(f'<li><code>{c}</code></li>' for c in config.get('exit_display', ['N/A']))}
                    </ul>
                </div>
                <div class="rule-block risk-block">
                    <h3>⚙️ {L['risk_management']}</h3>
                    <ul>
                        <li>{L['stop_loss']}: <code>-{config.get('stop_loss', 5)}%</code></li>
                        <li>{L['take_profit']}: <code>+{config.get('take_profit', 15)}%</code></li>
                        <li>{L['position_size']}: <code>{config.get('position_size', 10)}%</code></li>
                        <li>{L['commission']}: <code>{config.get('commission', 0.1)}%</code></li>
                    </ul>
                </div>
            </div>
        </section>
        
        <!-- Performance Metrics -->
        <div class="metrics-hero">
            <div class="metric-card hero">
                <div class="metric-value {return_class}">{metrics['total_return_pct']:+.1f}%</div>
                <div class="metric-label">{L['total_return']}</div>
                <div class="metric-sub">vs B&H: <span class="{vs_bh_class}">{vs_bh:+.1f}%</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">-{metrics['max_drawdown_pct']:.1f}%</div>
                <div class="metric-label">{L['max_drawdown']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-value neutral">{metrics['sharpe_ratio']:.2f}</div>
                <div class="metric-label">{L['sharpe_ratio']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if metrics['win_rate_pct'] > 50 else 'negative'}">{metrics['win_rate_pct']:.0f}%</div>
                <div class="metric-label">{L['win_rate']}</div>
                <div class="metric-sub">{metrics['winning_trades']}W / {metrics['losing_trades']}L</div>
            </div>
            <div class="metric-card">
                <div class="metric-value neutral">{metrics['profit_factor']}</div>
                <div class="metric-label">{L['profit_factor']}</div>
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
                <h2>{L['equity_curve']}</h2>
            </div>
            <div class="chart-container" id="equity-chart"></div>
        </section>
        
        <section class="section">
            <div class="section-header">
                <div class="section-icon">📉</div>
                <h2>{L['max_drawdown']}</h2>
            </div>
            <div class="chart-container" id="drawdown-chart"></div>
        </section>
        
        <section class="section">
            <div class="section-header">
                <div class="section-icon">📋</div>
                <h2>{L['trade_table_title']}</h2>
            </div>
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>{L['trade_entry_date']}</th>
                        <th>{L['trade_exit_date']}</th>
                        <th>{L['trade_entry_price']}</th>
                        <th>{L['trade_exit_price']}</th>
                        <th>{L['trade_pnl_label']}</th>
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
                <h2>{L['price_signals']}</h2>
            </div>
            <div class="chart-container" id="price-chart"></div>
        </section>
        
        <footer class="footer">
            <div class="footer-brand">🚀 Crypto Backtest Skill</div>
            <div class="footer-tagline">{L['tagline']}</div>
            <a href="https://github.com/0xrikt/crypto-skills" class="footer-cta" target="_blank">
                ⭐ Star on GitHub
            </a>
            <div class="footer-note">
                {L['share_cta']}<br>
                {L['generated']} {datetime.now().strftime('%Y-%m-%d %H:%M')} • {L['disclaimer']}
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
    parser.add_argument('--exchange', default='okx', help='Exchange to fetch data from (okx, kucoin, kraken, coinbase, bybit)')
    parser.add_argument('--entry', default='rsi<30', help='Entry conditions (comma-separated)')
    parser.add_argument('--exit', default='rsi>70', help='Exit conditions (comma-separated)')
    parser.add_argument('--stop-loss', type=float, default=5, help='Stop loss percentage')
    parser.add_argument('--take-profit', type=float, default=15, help='Take profit percentage')
    parser.add_argument('--position-size', type=float, default=10, help='Position size percentage')
    parser.add_argument('--initial-capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--commission', type=float, default=0.1, help='Commission percentage')
    parser.add_argument('--output', default='report.html', help='Output HTML file')
    parser.add_argument('--name', default='Trading Strategy', help='Strategy name')
    parser.add_argument('--lang', default='en', choices=['en', 'zh'], help='Report language (en/zh)')
    
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
    html = generate_html_report(df, results, metrics, config, lang=args.lang)
    
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
