#!/usr/bin/env python3
"""
Pair Trading / Relative Strength Strategy Backtest

This strategy compares two assets (e.g., BTC and ETH) and trades based on
the assumption that their trends will eventually align.

Strategy Logic:
- If Asset A significantly outperforms Asset B → Long Asset B (expect catch-up)
- If Asset B significantly outperforms Asset A → Long Asset A (expect catch-up)

SPOT ONLY: All trades are long-only, no shorting, no leverage.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Language labels
LABELS = {
    'en': {
        'title': 'Pair Trading Backtest Report',
        'subtitle': 'Relative Strength Mean Reversion',
        'strategy_summary': 'Strategy Summary',
        'original_strategy_idea': 'Original Strategy Idea',
        'no_description_provided': 'Trade underperforming asset when spread deviates from mean',
        'symbol_a': 'Symbol A',
        'symbol_b': 'Symbol B',
        'timeframe': 'Timeframe',
        'backtest_period': 'Backtest Period',
        'to': 'to',
        'days': 'days',
        'lookback': 'Lookback Period',
        'threshold': 'Entry Threshold',
        'exit_threshold': 'Exit Threshold',
        'initial_capital': 'Initial Capital',
        'position_size': 'Position Size',
        'stop_loss': 'Stop Loss',
        'take_profit': 'Take Profit',
        'commission': 'Commission',
        'strategy_logic': 'Strategy Logic',
        'logic_desc': 'When {a} outperforms {b} by more than {t}%, long {b} (expect catch-up). Vice versa.',
        'exit_logic': 'Exit when spread returns within ±{t}% of mean, or stop-loss/take-profit hit.',
        'performance_metrics': 'Performance Metrics',
        'total_return': 'Total Return',
        'buy_hold_a': 'Buy & Hold {a}',
        'buy_hold_b': 'Buy & Hold {b}',
        'max_drawdown': 'Max Drawdown',
        'sharpe_ratio': 'Sharpe Ratio',
        'total_trades': 'Total Trades',
        'win_rate': 'Win Rate',
        'profit_factor': 'Profit Factor',
        'trades_on_a': 'Trades on {a}',
        'trades_on_b': 'Trades on {b}',
        'price_comparison': 'Price Comparison (Normalized)',
        'relative_spread': 'Relative Performance Spread',
        'equity_curve': 'Portfolio Equity Curve',
        'trade_history': 'Trade History',
        'date': 'Date',
        'action': 'Action',
        'asset': 'Asset',
        'price': 'Price',
        'amount': 'Amount',
        'pnl': 'P&L',
        'buy': 'BUY',
        'sell': 'SELL',
        'tagline': 'Validate your trading ideas in minutes',
        'share_cta': 'Found this useful? Share it with fellow traders!',
        'generated': 'Generated',
        'disclaimer': 'Past performance ≠ future results',
    },
    'zh': {
        'title': '配对交易回测报告',
        'subtitle': '相对强弱均值回归策略',
        'strategy_summary': '策略概览',
        'original_strategy_idea': '原始策略思路',
        'no_description_provided': '当价差偏离均值时，做多表现落后的资产',
        'symbol_a': '资产 A',
        'symbol_b': '资产 B',
        'timeframe': '时间周期',
        'backtest_period': '回测区间',
        'to': '至',
        'days': '天',
        'lookback': '回溯周期',
        'threshold': '入场阈值',
        'exit_threshold': '出场阈值',
        'initial_capital': '初始资金',
        'position_size': '仓位比例',
        'stop_loss': '止损',
        'take_profit': '止盈',
        'commission': '手续费',
        'strategy_logic': '策略逻辑',
        'logic_desc': '当 {a} 相对 {b} 跑赢超过 {t}% 时，做多 {b}（预期追涨）。反之亦然。',
        'exit_logic': '当价差回归至均值 ±{t}% 内，或触发止损/止盈时平仓。',
        'performance_metrics': '绩效指标',
        'total_return': '总收益率',
        'buy_hold_a': '持有 {a}',
        'buy_hold_b': '持有 {b}',
        'max_drawdown': '最大回撤',
        'sharpe_ratio': '夏普比率',
        'total_trades': '总交易次数',
        'win_rate': '胜率',
        'profit_factor': '盈亏比',
        'trades_on_a': '{a} 交易次数',
        'trades_on_b': '{b} 交易次数',
        'price_comparison': '价格对比（标准化）',
        'relative_spread': '相对强弱价差',
        'equity_curve': '账户权益曲线',
        'trade_history': '交易记录',
        'date': '日期',
        'action': '操作',
        'asset': '资产',
        'price': '价格',
        'amount': '数量',
        'pnl': '盈亏',
        'buy': '买入',
        'sell': '卖出',
        'tagline': '几分钟验证你的交易策略想法',
        'share_cta': '觉得有用？分享给其他交易者吧！',
        'generated': '生成时间',
        'disclaimer': '历史表现不代表未来收益',
    }
}


def fetch_data(symbol: str, days: int, timeframe: str = '4h', exchange_id: str = 'okx') -> pd.DataFrame:
    """Fetch OHLCV data from exchange."""
    print(f"📊 Fetching {symbol} data ({days} days, {timeframe})...")
    
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({'enableRateLimit': True})
    
    since = exchange.parse8601((datetime.now(tz=None) - timedelta(days=days)).isoformat())
    
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
    
    print(f"   ✓ Loaded {len(df)} candles for {symbol}")
    return df


def calculate_spread(df_a: pd.DataFrame, df_b: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Calculate relative performance spread between two assets.
    
    Spread = (Return_A - Return_B) over lookback period
    Positive spread = A outperforming B
    Negative spread = B outperforming A
    """
    # Align dataframes on common index
    common_idx = df_a.index.intersection(df_b.index)
    df_a = df_a.loc[common_idx].copy()
    df_b = df_b.loc[common_idx].copy()
    
    # Calculate rolling returns
    df_a['return'] = df_a['close'].pct_change(lookback) * 100
    df_b['return'] = df_b['close'].pct_change(lookback) * 100
    
    # Create combined dataframe
    df = pd.DataFrame(index=common_idx)
    df['price_a'] = df_a['close']
    df['price_b'] = df_b['close']
    df['return_a'] = df_a['return']
    df['return_b'] = df_b['return']
    df['spread'] = df['return_a'] - df['return_b']
    
    # Calculate spread statistics
    df['spread_mean'] = df['spread'].rolling(window=lookback*2).mean()
    df['spread_std'] = df['spread'].rolling(window=lookback*2).std()
    df['spread_zscore'] = (df['spread'] - df['spread_mean']) / df['spread_std']
    
    # Normalize prices for comparison chart
    df['price_a_norm'] = df['price_a'] / df['price_a'].iloc[0] * 100
    df['price_b_norm'] = df['price_b'] / df['price_b'].iloc[0] * 100
    
    return df.dropna()


def generate_signals(df: pd.DataFrame, threshold: float = 10.0, exit_threshold: float = 2.0, only_long: str = 'both') -> pd.DataFrame:
    """
    Generate trading signals based on spread deviation.
    
    - Long B when spread > threshold (A outperforming, expect B to catch up)
    - Long A when spread < -threshold (B outperforming, expect A to catch up)
    - Exit when spread returns to within ±exit_threshold of mean
    
    Args:
        only_long: 'A' = only long A, 'B' = only long B, 'both' = trade both directions
    """
    df = df.copy()
    df['signal'] = 0  # 0 = no position, 1 = long A, 2 = long B
    df['target_asset'] = ''
    
    position = 0  # Current position
    
    for i in range(1, len(df)):
        spread = df['spread'].iloc[i]
        
        if position == 0:
            # No position - check for entry
            if spread > threshold and only_long in ['B', 'both']:
                # A significantly outperforming B → Long B
                position = 2
                df.iloc[i, df.columns.get_loc('signal')] = 2
                df.iloc[i, df.columns.get_loc('target_asset')] = 'B'
            elif spread < -threshold and only_long in ['A', 'both']:
                # B significantly outperforming A → Long A
                position = 1
                df.iloc[i, df.columns.get_loc('signal')] = 1
                df.iloc[i, df.columns.get_loc('target_asset')] = 'A'
        else:
            # Have position - check for exit
            df.iloc[i, df.columns.get_loc('signal')] = position
            
            if abs(spread) < exit_threshold:
                # Spread reverted to mean - exit
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
    
    return df


def run_backtest(df: pd.DataFrame, config: Dict) -> Tuple[Dict, List[Dict], pd.DataFrame]:
    """Run backtest simulation."""
    
    initial_capital = config.get('initial_capital', 10000)
    position_pct = config.get('position_size', 20) / 100
    stop_loss_pct = config.get('stop_loss', 10) / 100
    take_profit_pct = config.get('take_profit', 25) / 100
    commission_pct = config.get('commission', 0.1) / 100
    
    # Track portfolio
    cash = initial_capital
    position_asset = None  # 'A' or 'B'
    position_qty = 0
    entry_price = 0
    
    equity_curve = []
    trades = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        current_signal = row['signal']
        
        # Get current prices
        price_a = row['price_a']
        price_b = row['price_b']
        
        # Calculate current equity
        if position_asset == 'A':
            position_value = position_qty * price_a
        elif position_asset == 'B':
            position_value = position_qty * price_b
        else:
            position_value = 0
        
        current_equity = cash + position_value
        equity_curve.append({
            'timestamp': df.index[i],
            'equity': current_equity,
            'cash': cash,
            'position_value': position_value
        })
        
        # Check stop-loss / take-profit if in position
        if position_asset:
            current_price = price_a if position_asset == 'A' else price_b
            pnl_pct = (current_price - entry_price) / entry_price
            
            if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
                # Exit position
                exit_value = position_qty * current_price * (1 - commission_pct)
                pnl = exit_value - (entry_price * position_qty)
                cash += exit_value
                
                trades.append({
                    'date': df.index[i],
                    'action': 'SELL',
                    'asset': position_asset,
                    'price': current_price,
                    'qty': position_qty,
                    'value': exit_value,
                    'pnl': pnl,
                    'reason': 'stop_loss' if pnl_pct <= -stop_loss_pct else 'take_profit'
                })
                
                position_asset = None
                position_qty = 0
                entry_price = 0
                continue
        
        # Process signals
        if current_signal == 0 and position_asset:
            # Exit signal - close position
            current_price = price_a if position_asset == 'A' else price_b
            exit_value = position_qty * current_price * (1 - commission_pct)
            pnl = exit_value - (entry_price * position_qty)
            cash += exit_value
            
            trades.append({
                'date': df.index[i],
                'action': 'SELL',
                'asset': position_asset,
                'price': current_price,
                'qty': position_qty,
                'value': exit_value,
                'pnl': pnl,
                'reason': 'signal_exit'
            })
            
            position_asset = None
            position_qty = 0
            entry_price = 0
            
        elif current_signal > 0 and not position_asset:
            # Entry signal
            target = 'A' if current_signal == 1 else 'B'
            price = price_a if target == 'A' else price_b
            
            # Calculate position size
            position_value = cash * position_pct
            position_qty = (position_value * (1 - commission_pct)) / price
            entry_price = price
            position_asset = target
            cash -= position_value
            
            trades.append({
                'date': df.index[i],
                'action': 'BUY',
                'asset': target,
                'price': price,
                'qty': position_qty,
                'value': position_value,
                'pnl': 0,
                'reason': 'signal_entry'
            })
    
    # Close any remaining position at end
    if position_asset:
        final_price = df.iloc[-1]['price_a'] if position_asset == 'A' else df.iloc[-1]['price_b']
        exit_value = position_qty * final_price * (1 - commission_pct)
        pnl = exit_value - (entry_price * position_qty)
        cash += exit_value
        
        trades.append({
            'date': df.index[-1],
            'action': 'SELL',
            'asset': position_asset,
            'price': final_price,
            'qty': position_qty,
            'value': exit_value,
            'pnl': pnl,
            'reason': 'end_of_backtest'
        })
    
    # Calculate metrics
    equity_df = pd.DataFrame(equity_curve).set_index('timestamp')
    final_equity = equity_df['equity'].iloc[-1]
    total_return = (final_equity - initial_capital) / initial_capital * 100
    
    # Buy & Hold returns
    bh_return_a = (df['price_a'].iloc[-1] / df['price_a'].iloc[0] - 1) * 100
    bh_return_b = (df['price_b'].iloc[-1] / df['price_b'].iloc[0] - 1) * 100
    
    # Max drawdown
    rolling_max = equity_df['equity'].cummax()
    drawdown = (equity_df['equity'] - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()
    
    # Trade statistics
    if trades:
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        winning_trades = [t for t in sell_trades if t['pnl'] > 0]
        losing_trades = [t for t in sell_trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(sell_trades) * 100 if sell_trades else 0
        total_profit = sum(t['pnl'] for t in winning_trades)
        total_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        trades_a = len([t for t in trades if t['asset'] == 'A' and t['action'] == 'BUY'])
        trades_b = len([t for t in trades if t['asset'] == 'B' and t['action'] == 'BUY'])
    else:
        win_rate = 0
        profit_factor = 0
        trades_a = 0
        trades_b = 0
    
    # Sharpe ratio (annualized)
    returns = equity_df['equity'].pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 6) if returns.std() > 0 else 0  # 6 for 4h timeframe
    
    metrics = {
        'initial_capital': initial_capital,
        'final_equity': final_equity,
        'total_return': total_return,
        'buy_hold_return_a': bh_return_a,
        'buy_hold_return_b': bh_return_b,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'total_trades': len([t for t in trades if t['action'] == 'BUY']),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'trades_a': trades_a,
        'trades_b': trades_b
    }
    
    return metrics, trades, equity_df


def generate_html_report(df: pd.DataFrame, metrics: Dict, trades: List[Dict], 
                         equity_df: pd.DataFrame, config: Dict, lang: str = 'en') -> str:
    """Generate professional HTML report matching single-asset style."""
    
    L = LABELS.get(lang, LABELS['en'])
    
    symbol_a = config.get('symbol_a', 'BTC/USDT')
    symbol_b = config.get('symbol_b', 'ETH/USDT')
    name_a = symbol_a.split('/')[0]
    name_b = symbol_b.split('/')[0]
    
    # Calculate additional metrics
    initial_capital = config.get('initial_capital', 10000)
    final_equity = metrics.get('final_equity', initial_capital)
    total_profit = final_equity - initial_capital
    
    # Calculate drawdowns for chart
    equities = equity_df['equity'].tolist()
    peak = equities[0]
    drawdowns = []
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        drawdowns.append(-dd)
    
    # Prepare trade markers for spread chart and price chart
    buy_times = []
    buy_spreads = []
    buy_prices_norm = []
    sell_times = []
    sell_spreads = []
    sell_prices_norm = []
    
    for t in trades:
        trade_time = t['date'].isoformat()
        # Find closest values
        if t['date'] in df.index:
            spread_val = df.loc[t['date'], 'spread']
            price_norm = df.loc[t['date'], 'price_b_norm']  # ETH normalized price
        else:
            # Find nearest index
            idx = df.index.get_indexer([t['date']], method='nearest')[0]
            spread_val = df.iloc[idx]['spread']
            price_norm = df.iloc[idx]['price_b_norm']
        
        if t['action'] == 'BUY':
            buy_times.append(trade_time)
            buy_spreads.append(spread_val)
            buy_prices_norm.append(price_norm)
        else:
            sell_times.append(trade_time)
            sell_spreads.append(spread_val)
            sell_prices_norm.append(price_norm)
    
    # Timestamps for charts
    timestamps = [ts.isoformat() for ts in df.index]
    equity_timestamps = [ts.isoformat() for ts in equity_df.index]
    
    # Format trades table (paired entry/exit)
    trades_html = ''
    buy_trades = [t for t in trades if t['action'] == 'BUY']
    sell_trades = [t for t in trades if t['action'] == 'SELL']
    
    for i, (buy, sell) in enumerate(zip(buy_trades, sell_trades)):
        pnl_class = 'positive' if sell['pnl'] > 0 else 'negative'
        pnl_pct = (sell['price'] - buy['price']) / buy['price'] * 100
        asset_name = name_b if buy['asset'] == 'B' else name_a
        trades_html += f'''
        <tr>
            <td>{buy['date'].strftime('%Y-%m-%d %H:%M')}</td>
            <td>{sell['date'].strftime('%Y-%m-%d %H:%M')}</td>
            <td><span class="asset-badge">{asset_name}</span></td>
            <td>${buy['price']:,.2f}</td>
            <td>${sell['price']:,.2f}</td>
            <td class="{pnl_class}">{pnl_pct:+.2f}%</td>
            <td class="{pnl_class}">{'+' if sell['pnl'] > 0 else ''}${sell['pnl']:,.2f}</td>
        </tr>'''
    
    # Determine colors
    return_class = 'positive' if metrics['total_return'] > 0 else 'negative'
    profit_class = 'positive' if total_profit > 0 else 'negative'
    
    # Only long info
    only_long = config.get('only_long', 'both')
    if only_long == 'B':
        only_long_text = f"只做多 {name_b}" if lang == 'zh' else f"Long {name_b} only"
    elif only_long == 'A':
        only_long_text = f"只做多 {name_a}" if lang == 'zh' else f"Long {name_a} only"
    else:
        only_long_text = "双向交易" if lang == 'zh' else "Both directions"
    
    html = f'''<!DOCTYPE html>
<html lang="{lang}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{L['title']} | {name_a} vs {name_b}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-void: #f8fafc;
            --bg-deep: #ffffff;
            --bg-surface: #ffffff;
            --bg-elevated: #f1f5f9;
            --bg-hover: #e2e8f0;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --text-muted: #94a3b8;
            --accent-cyan: #0ea5e9;
            --accent-btc: #f7931a;
            --accent-eth: #627eea;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-gold: #f59e0b;
            --accent-purple: #8b5cf6;
            --gradient-pair: linear-gradient(135deg, #f7931a 0%, #627eea 100%);
            --border-subtle: #e2e8f0;
            --border-accent: rgba(14,165,233,0.4);
            --glow-cyan: 0 4px 20px rgba(14,165,233,0.15);
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Space Grotesk', -apple-system, sans-serif;
            background: var(--bg-void);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }}
        
        .bg-pattern {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: 
                radial-gradient(ellipse at 20% 20%, rgba(247,147,26,0.05) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(98,126,234,0.05) 0%, transparent 50%);
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
        
        .header-subtitle {{
            font-size: 0.85rem;
            letter-spacing: 1px;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 12px;
        }}
        
        .header-title {{
            font-size: clamp(1.8rem, 4vw, 2.5rem);
            font-weight: 700;
            margin-bottom: 16px;
            background: var(--gradient-pair);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .header-desc {{
            max-width: 700px;
            margin: 0 auto 20px;
            color: var(--text-secondary);
            font-size: 1.1rem;
            line-height: 1.5;
        }}
        
        .header-meta {{
            display: flex;
            justify-content: center;
            gap: 24px;
            flex-wrap: wrap;
            color: var(--text-secondary);
            font-size: 0.85rem;
        }}
        
        .header-meta span {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        .header-meta .dot {{
            width: 5px;
            height: 5px;
            background: var(--accent-cyan);
            border-radius: 50%;
        }}
        
        /* Strategy Compact Grid - 3x2 */
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
        
        .strategy-compact {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            grid-template-rows: repeat(2, minmax(130px, auto));
            gap: 12px;
        }}
        
        @media (max-width: 900px) {{
            .strategy-compact {{ 
                grid-template-columns: repeat(2, 1fr); 
                grid-template-rows: repeat(3, minmax(130px, auto));
            }}
        }}
        
        @media (max-width: 600px) {{
            .strategy-compact {{ 
                grid-template-columns: 1fr; 
                grid-template-rows: repeat(6, auto);
            }}
        }}
        
        .strategy-block {{
            background: var(--bg-elevated);
            border-radius: 12px;
            padding: 16px;
            display: flex;
            flex-direction: column;
        }}
        
        .strategy-block h4 {{
            font-size: 0.7rem;
            font-weight: 700;
            color: var(--text-muted);
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .param-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 5px 0;
            font-size: 0.85rem;
        }}
        
        .param-row span {{
            color: var(--text-secondary);
            font-size: 0.8rem;
        }}
        
        .param-row code {{
            font-family: 'JetBrains Mono', monospace;
            background: var(--bg-deep);
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            color: var(--accent-cyan);
        }}
        
        .param-row code.green {{ color: var(--accent-green); }}
        .param-row code.red {{ color: var(--accent-red); }}
        .param-row code.btc {{ color: var(--accent-btc); }}
        .param-row code.eth {{ color: var(--accent-eth); }}
        
        /* Metrics Table */
        .metrics-table-container {{
            overflow-x: auto;
            margin-bottom: 32px;
        }}
        
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        
        .metrics-table thead th {{
            background: var(--bg-elevated);
            padding: 16px 20px;
            text-align: center;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.75rem;
            border-bottom: 2px solid var(--border-subtle);
        }}
        
        .metrics-table tbody tr {{
            border-bottom: 1px solid var(--border-subtle);
        }}
        
        .metrics-table tbody tr:hover {{
            background: var(--bg-elevated);
        }}
        
        .metrics-table td {{
            padding: 14px 20px;
        }}
        
        .metrics-table .metric-name {{
            color: var(--text-secondary);
            font-weight: 500;
        }}
        
        .metrics-table .metric-val {{
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            text-align: right;
        }}
        
        .metrics-table .metric-val.positive {{ color: var(--accent-green); }}
        .metrics-table .metric-val.negative {{ color: var(--accent-red); }}
        
        /* Charts */
        .chart-container {{
            background: var(--bg-deep);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 32px;
            min-height: 320px;
            position: relative;
            overflow: hidden;
        }}
        
        .chart-container:last-child {{
            margin-bottom: 0;
        }}
        
        .chart-label {{
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 12px;
            padding-left: 8px;
        }}
        
        .charts-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 32px;
        }}
        
        @media (max-width: 992px) {{
            .charts-row {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .chart-half {{
            min-height: auto;
        }}
        
        .chart-half .chart-container {{
            margin-bottom: 0;
            min-height: 280px;
        }}
        
        .trades-wrapper {{
            margin-top: 40px;
            padding-top: 24px;
            border-top: 1px solid var(--border-subtle);
        }}
        
        /* Trades Table */
        .trades-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        
        .trades-table th {{
            text-align: left;
            padding: 14px 16px;
            background: var(--bg-elevated);
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.75rem;
        }}
        
        .trades-table th:first-child {{ border-radius: 8px 0 0 8px; }}
        .trades-table th:last-child {{ border-radius: 0 8px 8px 0; }}
        
        .trades-table td {{
            padding: 14px 16px;
            border-bottom: 1px solid var(--border-subtle);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }}
        
        .trades-table tr:hover td {{
            background: var(--bg-hover);
        }}
        
        .positive {{ color: var(--accent-green); }}
        .negative {{ color: var(--accent-red); }}
        
        .asset-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 0.75rem;
            font-weight: 600;
            background: linear-gradient(135deg, rgba(98,126,234,0.15) 0%, rgba(98,126,234,0.25) 100%);
            color: var(--accent-eth);
        }}
        
        /* Footer */
        .footer {{
            margin-top: 48px;
            padding: 32px 24px;
            text-align: center;
            border-top: 1px solid var(--border-subtle);
            background: var(--bg-surface);
        }}
        
        .footer-tagline {{
            font-size: 1rem;
            font-weight: 500;
            color: var(--text-primary);
            margin-bottom: 16px;
        }}
        
        .footer-github {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 20px;
            background: var(--bg-elevated);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 0.9rem;
            transition: all 0.2s ease;
        }}
        
        .footer-github:hover {{
            background: var(--bg-hover);
            border-color: var(--border-accent);
            color: var(--text-primary);
        }}
        
        .footer-note {{
            margin-top: 16px;
            color: var(--text-muted);
            font-size: 0.75rem;
        }}
    </style>
</head>
<body>
    <div class="bg-pattern"></div>
    
    <div class="container">
        <header class="header">
            <div class="header-subtitle">{L['title']}</div>
            <h1 class="header-title">{name_a} ↔ {name_b}</h1>
            <p class="header-desc">"{config.get('description', L['no_description_provided'])}"</p>
            <div class="header-meta">
                <span><div class="dot"></div>{config.get('timeframe', '4h')}</span>
                <span><div class="dot"></div>{df.index.min().strftime('%Y-%m-%d')} → {df.index.max().strftime('%Y-%m-%d')}</span>
                <span><div class="dot"></div>{metrics['total_trades']} trades</span>
            </div>
        </header>
        
        <!-- Strategy Summary - Compact 3x2 Grid -->
        <section class="section">
            <div class="section-header">
                <div class="section-icon">📋</div>
                <h2>{L['strategy_summary']}</h2>
            </div>
            
            <div class="strategy-compact">
                <div class="strategy-block">
                    <h4>📊 DATA</h4>
                    <div class="param-row"><span>{L['symbol_a']}</span><code class="btc">{name_a}</code></div>
                    <div class="param-row"><span>{L['symbol_b']}</span><code class="eth">{name_b}</code></div>
                    <div class="param-row"><span>{L['timeframe']}</span><code>{config.get('timeframe', '4h')}</code></div>
                </div>
                <div class="strategy-block">
                    <h4>🟢 ENTRY</h4>
                    <div class="param-row"><span>{L['threshold']}</span><code class="green">spread > {config.get('threshold', 3)}%</code></div>
                    <div class="param-row"><span>Direction</span><code>{only_long_text}</code></div>
                </div>
                <div class="strategy-block">
                    <h4>🔴 EXIT</h4>
                    <div class="param-row"><span>{L['exit_threshold']}</span><code>|spread| &lt; {config.get('exit_threshold', 1)}%</code></div>
                    <div class="param-row"><span>{L['stop_loss']}</span><code class="red">-{config.get('stop_loss', 20)}%</code></div>
                    <div class="param-row"><span>{L['take_profit']}</span><code class="green">+{config.get('take_profit', 20)}%</code></div>
                </div>
                <div class="strategy-block">
                    <h4>💰 CAPITAL</h4>
                    <div class="param-row"><span>{L['initial_capital']}</span><code>${config.get('initial_capital', 10000):,}</code></div>
                    <div class="param-row"><span>{L['position_size']}</span><code>{config.get('position_size', 20)}%</code></div>
                    <div class="param-row"><span>{L['commission']}</span><code>{config.get('commission', 0.1)}%</code></div>
                </div>
                <div class="strategy-block">
                    <h4>📅 PERIOD</h4>
                    <div class="param-row"><span>{'实际天数' if lang == 'zh' else 'Days'}</span><code>{(df.index.max() - df.index.min()).days}</code></div>
                    <div class="param-row"><span>{'开始' if lang == 'zh' else 'Start'}</span><code>{df.index.min().strftime('%m-%d')}</code></div>
                    <div class="param-row"><span>{'结束' if lang == 'zh' else 'End'}</span><code>{df.index.max().strftime('%m-%d')}</code></div>
                </div>
                <div class="strategy-block">
                    <h4>⚙️ EXECUTION</h4>
                    <div class="param-row"><span>Leverage</span><code>1x</code></div>
                    <div class="param-row"><span>Order</span><code>Market</code></div>
                    <div class="param-row"><span>Side</span><code>Long Only</code></div>
                </div>
            </div>
        </section>
        
        <!-- Trade History with Charts -->
        <section class="section">
            <div class="section-header">
                <div class="section-icon">📈</div>
                <h2>{L['trade_history']}</h2>
            </div>
            
            <div class="chart-label">📊 {L['price_comparison']}</div>
            <div class="chart-container" id="price-chart"></div>
            
            <div class="chart-label">📉 {L['relative_spread']}</div>
            <div class="chart-container" id="spread-chart"></div>
            
            <div class="trades-wrapper">
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>{'入场时间' if lang == 'zh' else 'Entry'}</th>
                        <th>{'出场时间' if lang == 'zh' else 'Exit'}</th>
                        <th>{L['asset']}</th>
                        <th>{'入场价' if lang == 'zh' else 'Entry $'}</th>
                        <th>{'出场价' if lang == 'zh' else 'Exit $'}</th>
                        <th>{'收益率' if lang == 'zh' else 'Return'}</th>
                        <th>{L['pnl']}</th>
                    </tr>
                </thead>
                <tbody>
                    {trades_html}
                </tbody>
            </table>
            </div>
        </section>
        
        <!-- Performance Metrics Table -->
        <section class="section">
            <div class="section-header">
                <div class="section-icon">📊</div>
                <h2>{L['performance_metrics']}</h2>
            </div>
            
            <div class="metrics-table-container">
                <table class="metrics-table">
                    <thead>
                        <tr>
                            <th colspan="2">{'收益指标' if lang == 'zh' else 'Returns'}</th>
                            <th colspan="2">{'风险指标' if lang == 'zh' else 'Risk'}</th>
                            <th colspan="2">{'交易统计' if lang == 'zh' else 'Trading'}</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td class="metric-name">{L['total_return']}</td>
                            <td class="metric-val {return_class}">{metrics['total_return']:+.2f}%</td>
                            <td class="metric-name">{L['max_drawdown']}</td>
                            <td class="metric-val negative">{metrics['max_drawdown']:.2f}%</td>
                            <td class="metric-name">{L['total_trades']}</td>
                            <td class="metric-val">{metrics['total_trades']}</td>
                        </tr>
                        <tr>
                            <td class="metric-name">{L['buy_hold_a'].format(a=name_a)}</td>
                            <td class="metric-val" style="color: var(--accent-btc);">{metrics['buy_hold_return_a']:+.2f}%</td>
                            <td class="metric-name">{L['sharpe_ratio']}</td>
                            <td class="metric-val">{metrics['sharpe_ratio']:.2f}</td>
                            <td class="metric-name">{L['win_rate']}</td>
                            <td class="metric-val {'positive' if metrics['win_rate'] > 50 else 'negative'}">{metrics['win_rate']:.0f}%</td>
                        </tr>
                        <tr>
                            <td class="metric-name">{L['buy_hold_b'].format(b=name_b)}</td>
                            <td class="metric-val" style="color: var(--accent-eth);">{metrics['buy_hold_return_b']:+.2f}%</td>
                            <td class="metric-name">{L['profit_factor']}</td>
                            <td class="metric-val">{metrics['profit_factor']:.2f}</td>
                            <td class="metric-name">{L['trades_on_a'].format(a=name_a)}</td>
                            <td class="metric-val">{metrics['trades_a']}</td>
                        </tr>
                        <tr>
                            <td class="metric-name">{'最终权益' if lang == 'zh' else 'Final Equity'}</td>
                            <td class="metric-val">${final_equity:,.2f}</td>
                            <td class="metric-name">{'vs 持有' if lang == 'zh' else 'vs Hold'}</td>
                            <td class="metric-val {return_class}">{metrics['total_return'] - metrics['buy_hold_return_b']:+.2f}%</td>
                            <td class="metric-name">{L['trades_on_b'].format(b=name_b)}</td>
                            <td class="metric-val">{metrics['trades_b']}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </section>
        
        <!-- Charts Row: Equity + Drawdown -->
        <div class="charts-row">
            <section class="section chart-half">
                <div class="section-header">
                    <div class="section-icon">💰</div>
                    <h2>{L['equity_curve']}</h2>
                </div>
                <div class="chart-container" id="equity-chart"></div>
            </section>
            
            <section class="section chart-half">
                <div class="section-header">
                    <div class="section-icon">📉</div>
                    <h2>{L['max_drawdown']}</h2>
                </div>
                <div class="chart-container" id="drawdown-chart"></div>
            </section>
        </div>
        
        <footer class="footer">
            <div class="footer-tagline">{L['tagline']}</div>
            <a href="https://github.com/0xrikt/crypto-skills" target="_blank" class="footer-github">
                <svg height="20" width="20" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
                </svg>
                <span>github.com/0xrikt/crypto-skills</span>
            </a>
            <div class="footer-note">
                {L['generated']} {datetime.now().strftime('%Y-%m-%d %H:%M')} • {L['disclaimer']}
            </div>
        </footer>
    </div>
    
    <script>
        const chartTheme = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#1e293b', family: 'Space Grotesk' }},
            xaxis: {{ gridcolor: 'rgba(0,0,0,0.08)', zerolinecolor: 'rgba(0,0,0,0.15)' }},
            yaxis: {{ gridcolor: 'rgba(0,0,0,0.08)', zerolinecolor: 'rgba(0,0,0,0.15)' }}
        }};
        
        // Spread Chart with Trade Markers
        Plotly.newPlot('spread-chart', [
            {{
                x: {json.dumps(timestamps)},
                y: {json.dumps(df['spread'].tolist())},
                type: 'scatter',
                mode: 'lines',
                name: 'Spread',
                line: {{ color: '#6366f1', width: 1.5 }},
                fill: 'tozeroy',
                fillcolor: 'rgba(99, 102, 241, 0.1)'
            }},
            {{
                x: {json.dumps(buy_times)},
                y: {json.dumps(buy_spreads)},
                type: 'scatter',
                mode: 'markers',
                name: '{"买入" if lang == "zh" else "Buy"}',
                marker: {{ symbol: 'triangle-up', size: 14, color: '#10b981', line: {{ color: '#059669', width: 2 }} }}
            }},
            {{
                x: {json.dumps(sell_times)},
                y: {json.dumps(sell_spreads)},
                type: 'scatter',
                mode: 'markers',
                name: '{"卖出" if lang == "zh" else "Sell"}',
                marker: {{ symbol: 'triangle-down', size: 14, color: '#f59e0b', line: {{ color: '#d97706', width: 2 }} }}
            }}
        ], {{
            ...chartTheme,
            height: 350,
            margin: {{ t: 30, r: 50, b: 50, l: 70 }},
            yaxis: {{ ...chartTheme.yaxis, title: '{name_a} vs {name_b} Spread (%)' }},
            legend: {{ orientation: 'h', y: 1.1 }},
            shapes: [
                {{ type: 'line', y0: {config.get('threshold', 3)}, y1: {config.get('threshold', 3)}, x0: 0, x1: 1, xref: 'paper', line: {{ dash: 'dash', color: '#ef4444', width: 1 }} }},
                {{ type: 'line', y0: -{config.get('threshold', 3)}, y1: -{config.get('threshold', 3)}, x0: 0, x1: 1, xref: 'paper', line: {{ dash: 'dash', color: '#22c55e', width: 1 }} }},
                {{ type: 'line', y0: 0, y1: 0, x0: 0, x1: 1, xref: 'paper', line: {{ color: '#9ca3af', width: 1 }} }}
            ]
        }}, {{ responsive: true }});
        
        // Equity Chart
        Plotly.newPlot('equity-chart', [{{
            x: {json.dumps(equity_timestamps)},
            y: {json.dumps(equities)},
            type: 'scatter',
            mode: 'lines',
            fill: 'tozeroy',
            fillcolor: 'rgba(16, 185, 129, 0.15)',
            line: {{ color: '#10b981', width: 2 }},
            name: 'Portfolio'
        }}], {{
            ...chartTheme,
            height: 280,
            margin: {{ t: 20, r: 20, b: 40, l: 70 }},
            yaxis: {{ ...chartTheme.yaxis, title: 'Equity ($)', tickformat: '$,.0f' }},
            shapes: [{{
                type: 'line',
                y0: {initial_capital}, y1: {initial_capital},
                x0: 0, x1: 1, xref: 'paper',
                line: {{ dash: 'dash', color: 'rgba(0,0,0,0.3)', width: 1 }}
            }}]
        }}, {{ responsive: true }});
        
        // Drawdown Chart
        Plotly.newPlot('drawdown-chart', [{{
            x: {json.dumps(equity_timestamps)},
            y: {json.dumps(drawdowns)},
            type: 'scatter',
            mode: 'lines',
            fill: 'tozeroy',
            fillcolor: 'rgba(239, 68, 68, 0.15)',
            line: {{ color: '#ef4444', width: 1.5 }},
            name: 'Drawdown'
        }}], {{
            ...chartTheme,
            height: 280,
            margin: {{ t: 20, r: 20, b: 40, l: 70 }},
            yaxis: {{ ...chartTheme.yaxis, title: 'Drawdown %', tickformat: '.1f' }}
        }}, {{ responsive: true }});
        
        // Price Comparison Chart with Trade Markers
        Plotly.newPlot('price-chart', [
            {{
                x: {json.dumps(timestamps)},
                y: {json.dumps(df['price_a_norm'].tolist())},
                type: 'scatter',
                mode: 'lines',
                name: '{name_a}',
                line: {{ color: '#f7931a', width: 2 }}
            }},
            {{
                x: {json.dumps(timestamps)},
                y: {json.dumps(df['price_b_norm'].tolist())},
                type: 'scatter',
                mode: 'lines',
                name: '{name_b}',
                line: {{ color: '#627eea', width: 2 }}
            }},
            {{
                x: {json.dumps(buy_times)},
                y: {json.dumps(buy_prices_norm)},
                type: 'scatter',
                mode: 'markers',
                name: '{"买入" if lang == "zh" else "Buy"}',
                marker: {{ symbol: 'triangle-up', size: 12, color: '#10b981', line: {{ color: '#059669', width: 2 }} }}
            }},
            {{
                x: {json.dumps(sell_times)},
                y: {json.dumps(sell_prices_norm)},
                type: 'scatter',
                mode: 'markers',
                name: '{"卖出" if lang == "zh" else "Sell"}',
                marker: {{ symbol: 'triangle-down', size: 12, color: '#f59e0b', line: {{ color: '#d97706', width: 2 }} }}
            }}
        ], {{
            ...chartTheme,
            height: 320,
            margin: {{ t: 30, r: 50, b: 50, l: 70 }},
            yaxis: {{ ...chartTheme.yaxis, title: '{"标准化价格 (基准=100)" if lang == "zh" else "Normalized Price (Base=100)"}' }},
            legend: {{ orientation: 'h', y: 1.12 }}
        }}, {{ responsive: true }});
    </script>
</body>
</html>'''
    
    return html


def main():
    parser = argparse.ArgumentParser(description='Pair Trading Backtest')
    parser.add_argument('--symbol-a', default='BTC/USDT', help='First symbol (default: BTC/USDT)')
    parser.add_argument('--symbol-b', default='ETH/USDT', help='Second symbol (default: ETH/USDT)')
    parser.add_argument('--days', type=int, default=365, help='Backtest period in days')
    parser.add_argument('--timeframe', default='4h', help='Candle timeframe')
    parser.add_argument('--lookback', type=int, default=20, help='Lookback period for spread calculation')
    parser.add_argument('--threshold', type=float, default=10.0, help='Entry threshold (spread %)')
    parser.add_argument('--exit-threshold', type=float, default=2.0, help='Exit threshold (spread %)')
    parser.add_argument('--initial-capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--position-size', type=float, default=20, help='Position size percentage')
    parser.add_argument('--stop-loss', type=float, default=10, help='Stop loss percentage')
    parser.add_argument('--take-profit', type=float, default=25, help='Take profit percentage')
    parser.add_argument('--commission', type=float, default=0.1, help='Commission percentage')
    parser.add_argument('--exchange', default='okx', help='Exchange (default: okx)')
    parser.add_argument('--output', default='pair_trading_report.html', help='Output HTML file')
    parser.add_argument('--lang', default='en', choices=['en', 'zh'], help='Report language')
    parser.add_argument('--description', default='', help='Original strategy description')
    parser.add_argument('--only-long', default='both', choices=['A', 'B', 'both'], 
                        help='Only long specific asset: A, B, or both (default: both)')
    
    args = parser.parse_args()
    
    config = {
        'symbol_a': args.symbol_a,
        'symbol_b': args.symbol_b,
        'days': args.days,
        'timeframe': args.timeframe,
        'lookback': args.lookback,
        'threshold': args.threshold,
        'exit_threshold': args.exit_threshold,
        'initial_capital': args.initial_capital,
        'position_size': args.position_size,
        'stop_loss': args.stop_loss,
        'take_profit': args.take_profit,
        'commission': args.commission,
        'description': args.description,
        'only_long': args.only_long
    }
    
    print(f"\n{'='*60}")
    print(f"  PAIR TRADING BACKTEST")
    print(f"  {args.symbol_a} ↔ {args.symbol_b}")
    print(f"{'='*60}\n")
    
    # Fetch data
    df_a = fetch_data(args.symbol_a, args.days, args.timeframe, args.exchange)
    df_b = fetch_data(args.symbol_b, args.days, args.timeframe, args.exchange)
    
    # Calculate spread
    print("\n📈 Calculating relative spread...")
    df = calculate_spread(df_a, df_b, args.lookback)
    print(f"   ✓ Spread range: {df['spread'].min():.1f}% to {df['spread'].max():.1f}%")
    
    # Generate signals
    print("\n🎯 Generating signals...")
    df = generate_signals(df, args.threshold, args.exit_threshold, args.only_long)
    signal_count = (df['signal'] != df['signal'].shift()).sum()
    only_long_msg = f" (only long {args.only_long})" if args.only_long != 'both' else ""
    print(f"   ✓ Generated {signal_count} signal changes{only_long_msg}")
    
    # Run backtest
    print("\n💰 Running backtest simulation...")
    metrics, trades, equity_df = run_backtest(df, config)
    
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Total Return:    {metrics['total_return']:+.1f}%")
    print(f"  Buy & Hold {args.symbol_a.split('/')[0]}:  {metrics['buy_hold_return_a']:+.1f}%")
    print(f"  Buy & Hold {args.symbol_b.split('/')[0]}:  {metrics['buy_hold_return_b']:+.1f}%")
    print(f"  Max Drawdown:    {metrics['max_drawdown']:.1f}%")
    print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    print(f"  Win Rate:        {metrics['win_rate']:.0f}%")
    print(f"  Total Trades:    {metrics['total_trades']}")
    print(f"{'='*60}\n")
    
    # Generate report
    print("📄 Generating HTML report...")
    html = generate_html_report(df, metrics, trades, equity_df, config, args.lang)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"   ✓ Report saved to: {args.output}")
    print(f"\n✅ Done!\n")


if __name__ == '__main__':
    main()
