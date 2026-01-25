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
        'tagline': 'Democratizing algorithmic crypto trading',
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
        'tagline': '让算法加密交易人人可用',
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


def generate_signals(df: pd.DataFrame, threshold: float = 10.0, exit_threshold: float = 2.0) -> pd.DataFrame:
    """
    Generate trading signals based on spread deviation.
    
    - Long B when spread > threshold (A outperforming, expect B to catch up)
    - Long A when spread < -threshold (B outperforming, expect A to catch up)
    - Exit when spread returns to within ±exit_threshold of mean
    """
    df = df.copy()
    df['signal'] = 0  # 0 = no position, 1 = long A, 2 = long B
    df['target_asset'] = ''
    
    position = 0  # Current position
    
    for i in range(1, len(df)):
        spread = df['spread'].iloc[i]
        
        if position == 0:
            # No position - check for entry
            if spread > threshold:
                # A significantly outperforming B → Long B
                position = 2
                df.iloc[i, df.columns.get_loc('signal')] = 2
                df.iloc[i, df.columns.get_loc('target_asset')] = 'B'
            elif spread < -threshold:
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
    """Generate HTML report."""
    
    L = LABELS.get(lang, LABELS['en'])
    
    symbol_a = config.get('symbol_a', 'BTC/USDT')
    symbol_b = config.get('symbol_b', 'ETH/USDT')
    name_a = symbol_a.split('/')[0]
    name_b = symbol_b.split('/')[0]
    
    # Create charts
    # Convert timestamps to strings for JSON serialization
    timestamps = [ts.isoformat() for ts in df.index]
    equity_timestamps = [ts.isoformat() for ts in equity_df.index]
    
    # 1. Price comparison chart
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=timestamps, y=df['price_a_norm'].tolist(),
        name=name_a, line=dict(color='#f7931a', width=2)
    ))
    fig_price.add_trace(go.Scatter(
        x=timestamps, y=df['price_b_norm'].tolist(),
        name=name_b, line=dict(color='#627eea', width=2)
    ))
    fig_price.update_layout(
        template='plotly_white',
        height=400,
        margin=dict(l=50, r=50, t=30, b=50),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        yaxis_title='Normalized Price (Base=100)'
    )
    
    # 2. Spread chart
    fig_spread = go.Figure()
    fig_spread.add_trace(go.Scatter(
        x=timestamps, y=df['spread'].tolist(),
        name='Spread', line=dict(color='#6366f1', width=1.5),
        fill='tozeroy', fillcolor='rgba(99, 102, 241, 0.1)'
    ))
    
    threshold = config.get('threshold', 10)
    fig_spread.add_hline(y=threshold, line_dash='dash', line_color='#ef4444', 
                         annotation_text=f'+{threshold}%')
    fig_spread.add_hline(y=-threshold, line_dash='dash', line_color='#22c55e',
                         annotation_text=f'-{threshold}%')
    fig_spread.add_hline(y=0, line_color='#9ca3af', line_width=1)
    
    fig_spread.update_layout(
        template='plotly_white',
        height=300,
        margin=dict(l=50, r=50, t=30, b=50),
        yaxis_title=f'{name_a} vs {name_b} Spread (%)'
    )
    
    # 3. Equity curve
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=equity_timestamps, y=equity_df['equity'].tolist(),
        name='Portfolio', line=dict(color='#10b981', width=2),
        fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.1)'
    ))
    fig_equity.update_layout(
        template='plotly_white',
        height=350,
        margin=dict(l=50, r=50, t=30, b=50),
        yaxis_title='Equity ($)'
    )
    
    # Format trades table
    trades_html = ''
    for t in trades[-50:]:  # Last 50 trades
        pnl_class = 'positive' if t['pnl'] > 0 else 'negative' if t['pnl'] < 0 else ''
        action_label = L['buy'] if t['action'] == 'BUY' else L['sell']
        trades_html += f'''
        <tr>
            <td>{t['date'].strftime('%Y-%m-%d %H:%M')}</td>
            <td><span class="badge {'buy' if t['action']=='BUY' else 'sell'}">{action_label}</span></td>
            <td>{t['asset']}</td>
            <td>${t['price']:,.2f}</td>
            <td>{t['qty']:.6f}</td>
            <td class="{pnl_class}">{'+' if t['pnl'] > 0 else ''}${t['pnl']:,.2f}</td>
        </tr>
        '''
    
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
            --bg-void: #ffffff;
            --bg-deep: #f0f2f5;
            --bg-surface: #ffffff;
            --bg-elevated: #e8eaed;
            --bg-hover: #d0d2d6;
            --text-primary: #1a1a1a;
            --text-secondary: #555555;
            --text-muted: #888888;
            --accent-btc: #f7931a;
            --accent-eth: #627eea;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-purple: #6366f1;
            --border-subtle: rgba(0,0,0,0.1);
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Space Grotesk', sans-serif;
            background: var(--bg-deep);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 40px 24px;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 48px;
        }}
        
        .header-badge {{
            display: inline-block;
            padding: 8px 20px;
            background: linear-gradient(135deg, var(--accent-btc) 0%, var(--accent-eth) 100%);
            color: white;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 16px;
            letter-spacing: 0.5px;
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 8px;
            background: linear-gradient(135deg, var(--accent-btc) 0%, var(--accent-eth) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .header-subtitle {{
            color: var(--text-secondary);
            font-size: 1.1rem;
        }}
        
        .header-meta {{
            display: flex;
            justify-content: center;
            gap: 24px;
            margin-top: 16px;
            flex-wrap: wrap;
        }}
        
        .header-meta span {{
            color: var(--text-muted);
            font-size: 0.9rem;
        }}
        
        .strategy-summary {{
            background: var(--bg-surface);
            border: 2px solid var(--accent-purple);
            border-radius: 20px;
            padding: 32px;
            margin-bottom: 48px;
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.1);
        }}
        
        .strategy-summary h2 {{
            font-size: 1.5rem;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .original-idea {{
            background: var(--bg-elevated);
            border-left: 4px solid var(--accent-btc);
            padding: 16px 20px;
            margin-bottom: 24px;
            border-radius: 8px;
            font-style: italic;
            color: var(--text-secondary);
        }}
        
        .original-idea strong {{
            color: var(--text-primary);
            display: block;
            margin-bottom: 8px;
            font-style: normal;
        }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        
        .info-item {{
            background: var(--bg-deep);
            padding: 16px;
            border-radius: 12px;
        }}
        
        .info-label {{
            display: block;
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .info-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-primary);
        }}
        
        .logic-box {{
            background: var(--bg-deep);
            padding: 20px;
            border-radius: 12px;
            border-left: 4px solid var(--accent-purple);
        }}
        
        .logic-box h3 {{
            font-size: 1rem;
            margin-bottom: 12px;
            color: var(--accent-purple);
        }}
        
        .logic-box p {{
            color: var(--text-secondary);
            margin-bottom: 8px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-bottom: 48px;
        }}
        
        .metric-card {{
            background: var(--bg-surface);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 2px 12px rgba(0,0,0,0.05);
            border: 1px solid var(--border-subtle);
        }}
        
        .metric-card.highlight {{
            border: 2px solid var(--accent-green);
            background: linear-gradient(135deg, var(--bg-surface) 0%, rgba(16, 185, 129, 0.05) 100%);
        }}
        
        .metric-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        
        .metric-value.positive {{ color: var(--accent-green); }}
        .metric-value.negative {{ color: var(--accent-red); }}
        
        .metric-label {{
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .section {{
            background: var(--bg-surface);
            border-radius: 20px;
            padding: 32px;
            margin-bottom: 32px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.05);
        }}
        
        .section h2 {{
            font-size: 1.3rem;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .chart-container {{
            width: 100%;
            border-radius: 12px;
            overflow: hidden;
        }}
        
        .trades-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        
        .trades-table th {{
            text-align: left;
            padding: 12px 16px;
            background: var(--bg-deep);
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.5px;
        }}
        
        .trades-table td {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-subtle);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }}
        
        .trades-table tr:hover {{
            background: var(--bg-deep);
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        
        .badge.buy {{ background: rgba(16, 185, 129, 0.15); color: var(--accent-green); }}
        .badge.sell {{ background: rgba(239, 68, 68, 0.15); color: var(--accent-red); }}
        
        .positive {{ color: var(--accent-green); }}
        .negative {{ color: var(--accent-red); }}
        
        .footer {{
            text-align: center;
            padding: 48px 24px;
            margin-top: 48px;
        }}
        
        .footer-brand {{
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        
        .footer-tagline {{
            color: var(--text-muted);
            margin-bottom: 24px;
        }}
        
        .footer-cta {{
            display: inline-block;
            padding: 12px 32px;
            background: linear-gradient(135deg, var(--accent-btc) 0%, var(--accent-eth) 100%);
            color: white;
            text-decoration: none;
            border-radius: 12px;
            font-weight: 600;
            margin-bottom: 24px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .footer-cta:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(247, 147, 26, 0.3);
        }}
        
        .footer-note {{
            font-size: 0.85rem;
            color: var(--text-muted);
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{ font-size: 1.8rem; }}
            .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .info-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <div class="header-badge">⚡ Pair Trading</div>
            <h1>{name_a} ↔ {name_b}</h1>
            <div class="header-subtitle">{L['subtitle']}</div>
            <div class="header-meta">
                <span>📊 {config.get('timeframe', '4h')} {L['timeframe']}</span>
                <span>📅 {config.get('days', 365)} {L['days']}</span>
                <span>🔄 {metrics['total_trades']} {L['total_trades']}</span>
            </div>
        </header>
        
        <!-- Strategy Summary -->
        <section class="strategy-summary">
            <h2>📋 {L['strategy_summary']}</h2>
            
            <div class="original-idea">
                <strong>{L['original_strategy_idea']}</strong>
                "{config.get('description', L['no_description_provided'])}"
            </div>
            
            <div class="info-grid">
                <div class="info-item">
                    <span class="info-label">{L['symbol_a']}</span>
                    <span class="info-value" style="color: var(--accent-btc);">{symbol_a}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">{L['symbol_b']}</span>
                    <span class="info-value" style="color: var(--accent-eth);">{symbol_b}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">{L['backtest_period']}</span>
                    <span class="info-value">{df.index.min().strftime('%Y-%m-%d')} {L['to']} {df.index.max().strftime('%Y-%m-%d')}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">{L['lookback']}</span>
                    <span class="info-value">{config.get('lookback', 20)} {L['timeframe']}s</span>
                </div>
                <div class="info-item">
                    <span class="info-label">{L['threshold']}</span>
                    <span class="info-value">±{config.get('threshold', 10)}%</span>
                </div>
                <div class="info-item">
                    <span class="info-label">{L['initial_capital']}</span>
                    <span class="info-value">${config.get('initial_capital', 10000):,}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">{L['position_size']}</span>
                    <span class="info-value">{config.get('position_size', 20)}%</span>
                </div>
                <div class="info-item">
                    <span class="info-label">{L['stop_loss']} / {L['take_profit']}</span>
                    <span class="info-value">-{config.get('stop_loss', 10)}% / +{config.get('take_profit', 25)}%</span>
                </div>
            </div>
            
            <div class="logic-box">
                <h3>🎯 {L['strategy_logic']}</h3>
                <p>{L['logic_desc'].format(a=name_a, b=name_b, t=config.get('threshold', 10))}</p>
                <p>{L['exit_logic'].format(t=config.get('exit_threshold', 2))}</p>
            </div>
        </section>
        
        <!-- Metrics -->
        <div class="metrics-grid">
            <div class="metric-card highlight">
                <div class="metric-value {'positive' if metrics['total_return'] > 0 else 'negative'}">
                    {'+' if metrics['total_return'] > 0 else ''}{metrics['total_return']:.1f}%
                </div>
                <div class="metric-label">{L['total_return']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color: var(--accent-btc);">
                    {'+' if metrics['buy_hold_return_a'] > 0 else ''}{metrics['buy_hold_return_a']:.1f}%
                </div>
                <div class="metric-label">{L['buy_hold_a'].format(a=name_a)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color: var(--accent-eth);">
                    {'+' if metrics['buy_hold_return_b'] > 0 else ''}{metrics['buy_hold_return_b']:.1f}%
                </div>
                <div class="metric-label">{L['buy_hold_b'].format(b=name_b)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">{metrics['max_drawdown']:.1f}%</div>
                <div class="metric-label">{L['max_drawdown']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['sharpe_ratio']:.2f}</div>
                <div class="metric-label">{L['sharpe_ratio']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['total_trades']}</div>
                <div class="metric-label">{L['total_trades']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['win_rate']:.0f}%</div>
                <div class="metric-label">{L['win_rate']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['profit_factor']:.2f}</div>
                <div class="metric-label">{L['profit_factor']}</div>
            </div>
        </div>
        
        <!-- Price Comparison -->
        <section class="section">
            <h2>📈 {L['price_comparison']}</h2>
            <div class="chart-container" id="price-chart"></div>
        </section>
        
        <!-- Spread Chart -->
        <section class="section">
            <h2>📊 {L['relative_spread']}</h2>
            <div class="chart-container" id="spread-chart"></div>
        </section>
        
        <!-- Equity Curve -->
        <section class="section">
            <h2>💰 {L['equity_curve']}</h2>
            <div class="chart-container" id="equity-chart"></div>
        </section>
        
        <!-- Trade History -->
        <section class="section">
            <h2>📝 {L['trade_history']}</h2>
            <div style="overflow-x: auto;">
                <table class="trades-table">
                    <thead>
                        <tr>
                            <th>{L['date']}</th>
                            <th>{L['action']}</th>
                            <th>{L['asset']}</th>
                            <th>{L['price']}</th>
                            <th>{L['amount']}</th>
                            <th>{L['pnl']}</th>
                        </tr>
                    </thead>
                    <tbody>
                        {trades_html}
                    </tbody>
                </table>
            </div>
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
        Plotly.newPlot('price-chart', {json.dumps(fig_price.to_dict()['data'])}, {json.dumps(fig_price.to_dict()['layout'])});
        Plotly.newPlot('spread-chart', {json.dumps(fig_spread.to_dict()['data'])}, {json.dumps(fig_spread.to_dict()['layout'])});
        Plotly.newPlot('equity-chart', {json.dumps(fig_equity.to_dict()['data'])}, {json.dumps(fig_equity.to_dict()['layout'])});
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
        'description': args.description
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
    df = generate_signals(df, args.threshold, args.exit_threshold)
    signal_count = (df['signal'] != df['signal'].shift()).sum()
    print(f"   ✓ Generated {signal_count} signal changes")
    
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
