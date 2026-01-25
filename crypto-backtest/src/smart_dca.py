#!/usr/bin/env python3
"""
Smart DCA (Dollar Cost Averaging) Backtest
==========================================
Intelligent periodic investment with valuation-based allocation.

Features:
- Multi-factor valuation scoring
- Dynamic allocation based on market state
- Comparison with fixed DCA
- Beautiful HTML report
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta


# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_ohlcv(symbol: str, timeframe: str, days: int, exchange_id: str = "kucoin") -> pd.DataFrame:
    """Fetch historical OHLCV data."""
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({'enableRateLimit': True})
    
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
# VALUATION MODEL
# ============================================================================

def calculate_valuation_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate multi-factor valuation score."""
    df = df.copy()
    
    # 1. RSI - Momentum
    df['rsi'] = ta.rsi(df['close'], length=14)
    
    # 2. SMA(200) - Long-term trend position
    df['sma200'] = ta.sma(df['close'], length=200)
    
    # 3. Bollinger Bands - Statistical deviation
    bb = ta.bbands(df['close'], length=20, std=2.0)
    if bb is not None:
        df['bb_upper'] = bb.iloc[:, 2]
        df['bb_middle'] = bb.iloc[:, 1]
        df['bb_lower'] = bb.iloc[:, 0]
    
    # 4. Drawdown from rolling high
    df['rolling_high_90'] = df['close'].rolling(window=90).max()
    df['drawdown_pct'] = (df['close'] - df['rolling_high_90']) / df['rolling_high_90'] * 100
    
    # 5. MACD - Momentum direction
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    if macd is not None:
        df['macd'] = macd.iloc[:, 0]
        df['macd_signal'] = macd.iloc[:, 2]
        df['macd_hist'] = macd.iloc[:, 1]
    
    # 6. Volume relative to average
    df['volume_sma'] = ta.sma(df['volume'], length=20)
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Calculate valuation score
    df['score'] = 0.0
    
    # RSI score (weight: 1.0)
    df.loc[df['rsi'] < 35, 'score'] += 1.0
    df.loc[df['rsi'] > 70, 'score'] -= 1.0
    
    # Price vs SMA200 (weight: 1.0)
    df.loc[df['close'] < df['sma200'], 'score'] += 1.0
    df.loc[df['close'] > df['sma200'] * 1.3, 'score'] -= 1.0
    
    # Bollinger Band position (weight: 1.0)
    df.loc[df['close'] < df['bb_lower'], 'score'] += 1.0
    df.loc[df['close'] > df['bb_upper'], 'score'] -= 1.0
    
    # Drawdown score (weight: 1.0)
    df.loc[df['drawdown_pct'] < -25, 'score'] += 1.0
    df.loc[df['drawdown_pct'] > -5, 'score'] -= 0.5
    
    # MACD turning (weight: 0.5)
    df['macd_turning_up'] = (df['macd_hist'] > df['macd_hist'].shift(1)) & (df['macd_hist'].shift(1) < 0)
    df['macd_turning_down'] = (df['macd_hist'] < df['macd_hist'].shift(1)) & (df['macd_hist'].shift(1) > 0)
    df.loc[df['macd_turning_up'], 'score'] += 0.5
    df.loc[df['macd_turning_down'], 'score'] -= 0.5
    
    # Classify market state
    df['market_state'] = 'normal'
    df.loc[df['score'] >= 3.0, 'market_state'] = 'extreme_undervalued'
    df.loc[(df['score'] >= 1.5) & (df['score'] < 3.0), 'market_state'] = 'undervalued'
    df.loc[(df['score'] <= -1.5) & (df['score'] > -3.0), 'market_state'] = 'overvalued'
    df.loc[df['score'] <= -3.0, 'market_state'] = 'extreme_overvalued'
    
    return df


def get_allocation_multiplier(score: float) -> float:
    """Get allocation multiplier based on valuation score."""
    if score >= 3.0:
        return 2.0  # Extreme undervaluation: 2x
    elif score >= 1.5:
        return 1.5  # Undervalued: 1.5x
    elif score >= -1.5:
        return 1.0  # Normal: 1x
    elif score >= -3.0:
        return 0.5  # Overvalued: 0.5x
    else:
        return 0.25  # Extreme overvaluation: 0.25x


# ============================================================================
# SMART DCA SIMULATION
# ============================================================================

def simulate_smart_dca(
    df: pd.DataFrame,
    base_amount: float = 200,
    frequency_days: int = 7
) -> dict:
    """Simulate smart DCA with valuation-based allocation."""
    
    # Resample to weekly (or specified frequency)
    # Get one data point per period
    df_weekly = df.resample(f'{frequency_days}D').last().dropna()
    
    # Calculate valuation for each period
    df_with_score = calculate_valuation_score(df)
    
    # Align scores with weekly data
    df_weekly = df_weekly.copy()
    df_weekly['score'] = df_with_score['score'].reindex(df_weekly.index, method='ffill')
    df_weekly['rsi'] = df_with_score['rsi'].reindex(df_weekly.index, method='ffill')
    df_weekly['market_state'] = df_with_score['market_state'].reindex(df_weekly.index, method='ffill')
    
    # Smart DCA simulation
    smart_records = []
    smart_total_invested = 0
    smart_total_btc = 0
    
    # Fixed DCA simulation (for comparison)
    fixed_records = []
    fixed_total_invested = 0
    fixed_total_btc = 0
    
    for timestamp, row in df_weekly.iterrows():
        price = row['close']
        score = row['score'] if pd.notna(row['score']) else 0
        
        # Smart DCA
        multiplier = get_allocation_multiplier(score)
        smart_amount = base_amount * multiplier
        smart_btc_bought = smart_amount / price
        smart_total_invested += smart_amount
        smart_total_btc += smart_btc_bought
        
        smart_records.append({
            'timestamp': timestamp,
            'price': price,
            'score': score,
            'market_state': row['market_state'],
            'multiplier': multiplier,
            'amount_invested': smart_amount,
            'btc_bought': smart_btc_bought,
            'total_invested': smart_total_invested,
            'total_btc': smart_total_btc,
            'portfolio_value': smart_total_btc * price,
            'avg_cost': smart_total_invested / smart_total_btc if smart_total_btc > 0 else 0
        })
        
        # Fixed DCA
        fixed_btc_bought = base_amount / price
        fixed_total_invested += base_amount
        fixed_total_btc += fixed_btc_bought
        
        fixed_records.append({
            'timestamp': timestamp,
            'price': price,
            'amount_invested': base_amount,
            'btc_bought': fixed_btc_bought,
            'total_invested': fixed_total_invested,
            'total_btc': fixed_total_btc,
            'portfolio_value': fixed_total_btc * price,
            'avg_cost': fixed_total_invested / fixed_total_btc if fixed_total_btc > 0 else 0
        })
    
    # Final metrics
    final_price = df_weekly.iloc[-1]['close']
    
    smart_final_value = smart_total_btc * final_price
    smart_return_pct = (smart_final_value - smart_total_invested) / smart_total_invested * 100
    smart_avg_cost = smart_total_invested / smart_total_btc
    
    fixed_final_value = fixed_total_btc * final_price
    fixed_return_pct = (fixed_final_value - fixed_total_invested) / fixed_total_invested * 100
    fixed_avg_cost = fixed_total_invested / fixed_total_btc
    
    return {
        'smart': {
            'records': smart_records,
            'total_invested': smart_total_invested,
            'total_btc': smart_total_btc,
            'final_value': smart_final_value,
            'return_pct': smart_return_pct,
            'avg_cost': smart_avg_cost
        },
        'fixed': {
            'records': fixed_records,
            'total_invested': fixed_total_invested,
            'total_btc': fixed_total_btc,
            'final_value': fixed_final_value,
            'return_pct': fixed_return_pct,
            'avg_cost': fixed_avg_cost
        },
        'comparison': {
            'extra_return_pct': smart_return_pct - fixed_return_pct,
            'cost_savings_pct': (fixed_avg_cost - smart_avg_cost) / fixed_avg_cost * 100,
            'extra_btc': smart_total_btc - fixed_total_btc,
            'investment_diff': smart_total_invested - fixed_total_invested
        }
    }


# ============================================================================
# HTML REPORT
# ============================================================================

def generate_html_report(results: dict, config: dict) -> str:
    """Generate beautiful HTML report for Smart DCA."""
    
    smart = results['smart']
    fixed = results['fixed']
    comp = results['comparison']
    
    # Prepare chart data
    timestamps = [str(r['timestamp']) for r in smart['records']]
    smart_values = [r['portfolio_value'] for r in smart['records']]
    fixed_values = [r['portfolio_value'] for r in fixed['records']]
    smart_invested = [r['total_invested'] for r in smart['records']]
    fixed_invested = [r['total_invested'] for r in fixed['records']]
    prices = [r['price'] for r in smart['records']]
    scores = [r['score'] for r in smart['records']]
    amounts = [r['amount_invested'] for r in smart['records']]
    
    # Market state distribution
    state_counts = {}
    for r in smart['records']:
        state = r['market_state']
        state_counts[state] = state_counts.get(state, 0) + 1
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart DCA Report | {config.get('symbol', 'BTC/USDT')}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-void: #05070a;
            --bg-deep: #0a0e14;
            --bg-surface: #111820;
            --bg-elevated: #1a2332;
            --text-primary: #e6edf3;
            --text-secondary: #7d8590;
            --accent-cyan: #00d9ff;
            --accent-green: #00ff9d;
            --accent-red: #ff4757;
            --accent-gold: #ffd93d;
            --accent-purple: #a855f7;
            --border-subtle: rgba(255,255,255,0.06);
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Space Grotesk', sans-serif;
            background: var(--bg-void);
            color: var(--text-primary);
            line-height: 1.6;
        }}
        
        .bg-pattern {{
            position: fixed;
            inset: 0;
            background: 
                radial-gradient(ellipse at 20% 20%, rgba(0,217,255,0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(168,85,247,0.06) 0%, transparent 50%);
            pointer-events: none;
        }}
        
        .container {{ position: relative; max-width: 1400px; margin: 0 auto; padding: 40px 24px; }}
        
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
            border: 1px solid var(--accent-cyan);
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
            margin-bottom: 16px;
            background: linear-gradient(135deg, var(--text-primary) 0%, var(--accent-gold) 50%, var(--accent-green) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .header-meta {{
            display: flex;
            justify-content: center;
            gap: 32px;
            flex-wrap: wrap;
            color: var(--text-secondary);
        }}
        
        .comparison-hero {{
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 24px;
            margin-bottom: 48px;
            align-items: center;
        }}
        
        .strategy-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 32px;
            text-align: center;
        }}
        
        .strategy-card.winner {{
            border-color: var(--accent-green);
            box-shadow: 0 0 40px rgba(0,255,157,0.15);
        }}
        
        .strategy-card h3 {{
            font-size: 1.1rem;
            color: var(--text-secondary);
            margin-bottom: 16px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .strategy-card .value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        
        .strategy-card .value.positive {{ color: var(--accent-green); }}
        .strategy-card .value.negative {{ color: var(--accent-red); }}
        
        .strategy-card .sub {{ color: var(--text-secondary); font-size: 0.9rem; }}
        
        .vs-badge {{
            background: var(--bg-elevated);
            border-radius: 50%;
            width: 60px;
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            color: var(--accent-purple);
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 48px;
        }}
        
        .metric-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 24px;
            text-align: center;
        }}
        
        .metric-card .value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--accent-cyan);
        }}
        
        .metric-card .label {{
            color: var(--text-secondary);
            font-size: 0.8rem;
            text-transform: uppercase;
            margin-top: 8px;
        }}
        
        .section {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 32px;
            margin-bottom: 32px;
        }}
        
        .section h2 {{
            font-size: 1.25rem;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border-subtle);
        }}
        
        .chart-container {{
            background: var(--bg-deep);
            border-radius: 12px;
            padding: 16px;
        }}
        
        .footer {{
            margin-top: 64px;
            padding: 48px;
            background: linear-gradient(135deg, var(--bg-surface) 0%, var(--bg-elevated) 100%);
            border: 1px solid var(--accent-cyan);
            border-radius: 24px;
            text-align: center;
        }}
        
        .footer-brand {{
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-green));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 12px;
        }}
        
        .footer-cta {{
            display: inline-block;
            padding: 14px 32px;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-green));
            color: var(--bg-void);
            font-weight: 600;
            border-radius: 12px;
            text-decoration: none;
            margin-top: 16px;
        }}
        
        @media (max-width: 768px) {{
            .comparison-hero {{ grid-template-columns: 1fr; }}
            .vs-badge {{ margin: 16px auto; }}
        }}
    </style>
</head>
<body>
    <div class="bg-pattern"></div>
    <div class="container">
        <header class="header">
            <div class="header-badge">Smart DCA Backtest</div>
            <h1>Smart DCA vs Fixed DCA</h1>
            <div class="header-meta">
                <span>📊 {config.get('symbol', 'BTC/USDT')}</span>
                <span>📅 {len(smart['records'])} periods</span>
                <span>💰 Base {config.get('base_amount', 200)} USDT/period</span>
            </div>
        </header>
        
        <div class="comparison-hero">
            <div class="strategy-card {'winner' if comp['extra_return_pct'] > 0 else ''}">
                <h3>🧠 Smart DCA</h3>
                <div class="value {'positive' if smart['return_pct'] > 0 else 'negative'}">{smart['return_pct']:+.1f}%</div>
                <div class="sub">Total Invested ${smart['total_invested']:,.0f}</div>
                <div class="sub">Final Value ${smart['final_value']:,.0f}</div>
                <div class="sub">Avg Cost ${smart['avg_cost']:,.0f}</div>
            </div>
            
            <div class="vs-badge">VS</div>
            
            <div class="strategy-card {'winner' if comp['extra_return_pct'] < 0 else ''}">
                <h3>📊 Fixed DCA</h3>
                <div class="value {'positive' if fixed['return_pct'] > 0 else 'negative'}">{fixed['return_pct']:+.1f}%</div>
                <div class="sub">Total Invested ${fixed['total_invested']:,.0f}</div>
                <div class="sub">Final Value ${fixed['final_value']:,.0f}</div>
                <div class="sub">Avg Cost ${fixed['avg_cost']:,.0f}</div>
            </div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="value" style="color: {'var(--accent-green)' if comp['extra_return_pct'] > 0 else 'var(--accent-red)'}">{comp['extra_return_pct']:+.2f}%</div>
                <div class="label">Smart DCA Alpha</div>
            </div>
            <div class="metric-card">
                <div class="value">{comp['cost_savings_pct']:+.2f}%</div>
                <div class="label">Cost Reduction</div>
            </div>
            <div class="metric-card">
                <div class="value">{smart['total_btc']:.6f}</div>
                <div class="label">Smart DCA BTC</div>
            </div>
            <div class="metric-card">
                <div class="value">{fixed['total_btc']:.6f}</div>
                <div class="label">Fixed DCA BTC</div>
            </div>
        </div>
        
        <section class="section">
            <h2>📈 Portfolio Growth</h2>
            <div class="chart-container" id="value-chart"></div>
        </section>
        
        <section class="section">
            <h2>📊 Valuation Score & Investment</h2>
            <div class="chart-container" id="allocation-chart"></div>
        </section>
        
        <section class="section">
            <h2>💰 BTC Price</h2>
            <div class="chart-container" id="price-chart"></div>
        </section>
        
        <footer class="footer">
            <div class="footer-brand">🚀 Crypto Backtest Skill</div>
            <div style="color: var(--text-secondary);">Validate your trading ideas in minutes</div>
            <a href="https://github.com/0xrikt/crypto-skills" class="footer-cta" target="_blank">⭐ Star on GitHub</a>
            <div style="margin-top: 16px; color: var(--text-secondary); font-size: 0.85rem;">
                Share your results to help others discover this tool!<br>
                Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} • Past performance ≠ future results
            </div>
        </footer>
    </div>
    
    <script>
        const theme = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#e6edf3', family: 'Space Grotesk' }},
            xaxis: {{ gridcolor: 'rgba(255,255,255,0.06)' }},
            yaxis: {{ gridcolor: 'rgba(255,255,255,0.06)' }}
        }};
        
        // Portfolio Value Comparison
        Plotly.newPlot('value-chart', [
            {{
                x: {json.dumps(timestamps)},
                y: {json.dumps(smart_values)},
                type: 'scatter',
                mode: 'lines',
                name: 'Smart DCA',
                line: {{ color: '#00ff9d', width: 2 }},
                fill: 'tozeroy',
                fillcolor: 'rgba(0,255,157,0.1)'
            }},
            {{
                x: {json.dumps(timestamps)},
                y: {json.dumps(fixed_values)},
                type: 'scatter',
                mode: 'lines',
                name: 'Fixed DCA',
                line: {{ color: '#00d9ff', width: 2, dash: 'dash' }}
            }},
            {{
                x: {json.dumps(timestamps)},
                y: {json.dumps(smart_invested)},
                type: 'scatter',
                mode: 'lines',
                name: 'Smart Cost Basis',
                line: {{ color: '#ffd93d', width: 1, dash: 'dot' }}
            }}
        ], {{
            ...theme,
            height: 400,
            margin: {{ t: 20, r: 20, b: 40, l: 60 }},
            yaxis: {{ ...theme.yaxis, title: 'Value ($)', tickformat: '$,.0f' }},
            legend: {{ orientation: 'h', y: 1.1 }}
        }}, {{ responsive: true }});
        
        // Allocation Chart
        Plotly.newPlot('allocation-chart', [
            {{
                x: {json.dumps(timestamps)},
                y: {json.dumps(scores)},
                type: 'scatter',
                mode: 'lines',
                name: 'Valuation Score',
                line: {{ color: '#a855f7', width: 2 }},
                yaxis: 'y'
            }},
            {{
                x: {json.dumps(timestamps)},
                y: {json.dumps(amounts)},
                type: 'bar',
                name: 'Investment',
                marker: {{ 
                    color: {json.dumps(amounts)},
                    colorscale: [[0, '#ff4757'], [0.5, '#ffd93d'], [1, '#00ff9d']]
                }},
                yaxis: 'y2'
            }}
        ], {{
            ...theme,
            height: 350,
            margin: {{ t: 20, r: 60, b: 40, l: 60 }},
            yaxis: {{ ...theme.yaxis, title: 'Valuation Score', side: 'left' }},
            yaxis2: {{ title: 'Investment ($)', side: 'right', overlaying: 'y', tickformat: '$,.0f' }},
            legend: {{ orientation: 'h', y: 1.1 }},
            shapes: [
                {{ type: 'line', y0: 0, y1: 0, x0: 0, x1: 1, xref: 'paper', yref: 'y', line: {{ dash: 'dash', color: 'rgba(255,255,255,0.3)' }} }}
            ]
        }}, {{ responsive: true }});
        
        // Price Chart
        Plotly.newPlot('price-chart', [{{
            x: {json.dumps(timestamps)},
            y: {json.dumps(prices)},
            type: 'scatter',
            mode: 'lines',
            name: 'BTC Price',
            line: {{ color: '#ffd93d', width: 2 }},
            fill: 'tozeroy',
            fillcolor: 'rgba(255,217,61,0.1)'
        }}], {{
            ...theme,
            height: 300,
            margin: {{ t: 20, r: 20, b: 40, l: 60 }},
            yaxis: {{ ...theme.yaxis, title: 'Price ($)', tickformat: '$,.0f' }}
        }}, {{ responsive: true }});
    </script>
</body>
</html>'''
    
    return html


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Smart DCA Backtest')
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading pair')
    parser.add_argument('--days', type=int, default=1095, help='Backtest period (default: 3 years)')
    parser.add_argument('--base-amount', type=float, default=200, help='Base investment per period')
    parser.add_argument('--frequency', type=int, default=7, help='Investment frequency in days')
    parser.add_argument('--output', default='smart_dca_report.html', help='Output HTML file')
    
    args = parser.parse_args()
    
    print(f"🧠 Smart DCA Backtest")
    print(f"{'='*50}")
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.days} days ({args.days // 365} years)")
    print(f"Base Amount: ${args.base_amount}/period")
    print(f"Frequency: Every {args.frequency} days")
    print()
    
    # Fetch data
    print("📊 Fetching historical data...")
    df = fetch_ohlcv(args.symbol, '1d', args.days)
    print(f"   Got {len(df)} daily candles")
    
    # Run simulation
    print("🧮 Running Smart DCA simulation...")
    results = simulate_smart_dca(df, args.base_amount, args.frequency)
    
    smart = results['smart']
    fixed = results['fixed']
    comp = results['comparison']
    
    # Generate report
    print("📄 Generating report...")
    config = {
        'symbol': args.symbol,
        'base_amount': args.base_amount,
        'frequency': args.frequency,
        'days': args.days
    }
    html = generate_html_report(results, config)
    
    output_path = Path(args.output)
    output_path.write_text(html)
    print(f"   Saved: {output_path.absolute()}")
    
    # Print results
    print()
    print(f"{'='*50}")
    print("📈 RESULTS COMPARISON")
    print(f"{'='*50}")
    print()
    print("🧠 Smart DCA:")
    print(f"   Invested:    ${smart['total_invested']:,.0f}")
    print(f"   Total BTC:   {smart['total_btc']:.6f}")
    print(f"   Final Value: ${smart['final_value']:,.0f}")
    print(f"   Return:      {smart['return_pct']:+.2f}%")
    print(f"   Avg Cost:    ${smart['avg_cost']:,.0f}")
    print()
    print("📊 Fixed DCA:")
    print(f"   Invested:    ${fixed['total_invested']:,.0f}")
    print(f"   Total BTC:   {fixed['total_btc']:.6f}")
    print(f"   Final Value: ${fixed['final_value']:,.0f}")
    print(f"   Return:      {fixed['return_pct']:+.2f}%")
    print(f"   Avg Cost:    ${fixed['avg_cost']:,.0f}")
    print()
    print(f"{'='*50}")
    
    if comp['extra_return_pct'] > 0:
        print(f"✅ Smart DCA wins! Alpha: {comp['extra_return_pct']:+.2f}%")
    else:
        print(f"📊 Fixed DCA wins by {abs(comp['extra_return_pct']):.2f}%")
    
    print(f"   Cost reduction: {comp['cost_savings_pct']:.2f}%")


if __name__ == '__main__':
    main()
