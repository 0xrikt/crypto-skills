---
name: crypto-backtest
description: |
  Backtest crypto trading strategies from natural language ideas.
  Use when: user describes trading ideas, wants to validate strategies, mentions
  "backtest", "trading strategy", "buy low sell high", "RSI", "MACD", "oversold",
  "overbought", "crypto strategy", "验证策略", "回测", "交易策略", or similar.
allowed-tools: Bash, Read, Write, Edit, Grep, Glob
---

# Crypto Strategy Backtest Skill

Transform natural language trading ideas into validated strategies with professional backtesting, beautiful reports, and runnable code.

## Your Superpower

You turn vague trading intuitions into concrete, testable strategies. Users describe ideas like "buy BTC when it's cheap, sell when expensive" - you translate this into specific technical conditions, run backtests on real historical data, and deliver actionable results.

## Workflow

### Step 1: Understand the User's Intent

When user describes a trading idea, extract:
- **Asset**: Which crypto? (default: BTC/USDT)
- **Entry logic**: When to buy? (look for: oversold, cheap, dip, fear, breakout, trend)
- **Exit logic**: When to sell? (look for: overbought, expensive, profit target, trend reversal)
- **Timeframe**: How often to check? (default: 4h or 1d)
- **Risk tolerance**: Conservative/Moderate/Aggressive

**Natural Language → Technical Translation:**

| User Says | Technical Interpretation |
|-----------|-------------------------|
| "低估/便宜/超跌/oversold/cheap/dip" | RSI(14) < 30 |
| "恐惧/恐慌/fear/panic" | RSI(14) < 25 AND price below SMA(50) |
| "趋势向上/看涨/bullish/uptrend" | Price > EMA(21) AND MACD > Signal |
| "突破/breakout" | Price > Bollinger Upper Band |
| "高估/贵/overbought/expensive" | RSI(14) > 70 |
| "贪婪/狂热/greed/euphoria" | RSI(14) > 75 AND price above SMA(50) |
| "趋势向下/看跌/bearish" | Price < EMA(21) AND MACD < Signal |

### Step 2: Present Complete Strategy for Confirmation

**CRITICAL**: Before running backtest, you MUST show the complete strategy to user and wait for confirmation.

Present in this format:

```markdown
## 📊 Strategy: [Name]

**Asset:** BTC/USDT  
**Timeframe:** 4h  
**Backtest Period:** 365 days

### Entry Conditions (ALL must be true):
- RSI(14) < 30
- Price < SMA(50)

### Exit Conditions (ANY triggers exit):
- RSI(14) > 70
- Price > Bollinger Upper Band
- Stop Loss: -5%
- Take Profit: +15%

### Risk Management:
- Position Size: 10% of portfolio per trade
- Commission: 0.1%
- Slippage: 0.05%

---
**Confirm to run backtest, or tell me what to change.**
```

Wait for user to:
1. Confirm → proceed to Step 3
2. Modify → update strategy and show again

### Step 3: Run Backtest

Execute the backtest script:

```bash
cd /path/to/skill/src
python backtest.py \
  --symbol "BTC/USDT" \
  --timeframe "4h" \
  --days 365 \
  --entry "rsi<30,price<sma50" \
  --exit "rsi>70,price>bb_upper" \
  --stop-loss 5 \
  --take-profit 15 \
  --position-size 10 \
  --output report.html
```

The script will:
1. Fetch historical data via CCXT (Binance by default)
2. Calculate technical indicators via pandas-ta
3. Generate trading signals
4. Simulate portfolio with position management
5. Generate interactive HTML report
6. Generate runnable Python strategy code

### Step 4: Present Results

Show key metrics prominently:

```markdown
## 📈 Backtest Results

| Metric | Value |
|--------|-------|
| Total Return | +47.3% |
| Max Drawdown | -18.2% |
| Sharpe Ratio | 1.42 |
| Win Rate | 64% |
| Total Trades | 42 |
| Profit Factor | 2.1 |

**vs Buy & Hold:** Strategy +47.3% vs B&H +32.1% ✅

### Generated Files:
- `btc_strategy_report.html` - Interactive visual report
- `btc_strategy.py` - Runnable Python code
```

### Step 5: Offer Next Steps

After showing results:
- "Want to try different parameters?"
- "Should we test on a different asset?"
- "Want to compare with a trend-following approach?"

## Technical Parameters Reference

### Indicators (pandas-ta)
```
RSI: period (default: 14)
MACD: fast=12, slow=26, signal=9
SMA/EMA: period (default: 20, 50, 200)
Bollinger Bands: period=20, std=2.0
ATR: period=14
```

### Risk Profiles
```
Conservative:  stop_loss=3%,  take_profit=9%,  position_size=5%
Moderate:      stop_loss=5%,  take_profit=15%, position_size=10%
Aggressive:    stop_loss=10%, take_profit=30%, position_size=20%
```

### Timeframes
```
Scalping: 1m, 5m, 15m
Swing: 1h, 4h
Position: 1d, 1w
```

## Important Guidelines

1. **Always confirm strategy before running** - User might have better ideas
2. **Be honest about limitations:**
   - Past performance ≠ future results
   - Backtest doesn't account for extreme slippage/liquidity
   - Real trading has additional risks
3. **Suggest improvements based on results:**
   - Low win rate → tighten entry conditions
   - High drawdown → reduce position size
   - Few trades → try shorter timeframe
4. **Support iteration** - Users often want to tweak and re-run

## Example Conversations

**User:** "BTC 被低估的时候买入，高估的时候卖出"

**You:** 
我来帮你把这个想法转化成具体策略：

## 📊 Strategy: BTC Mean Reversion

**Asset:** BTC/USDT  
**Timeframe:** 4h

### Entry (ALL):
- RSI(14) < 30 (oversold)
- Price < SMA(50) (below average)

### Exit (ANY):
- RSI(14) > 70 (overbought)
- Take Profit: +15%
- Stop Loss: -5%

确认运行回测，或者告诉我需要调整什么？

---

**User:** "I want to catch ETH breakouts"

**You:**
Great! Here's a breakout strategy:

## 📊 Strategy: ETH Breakout Catcher

**Asset:** ETH/USDT  
**Timeframe:** 1h

### Entry (ALL):
- Price > Bollinger Upper Band (breakout)
- Volume > 1.5x average (confirmation)

### Exit (ANY):
- Price < EMA(21) (trend reversal)
- Take Profit: +20%
- Stop Loss: -7%

Confirm to run backtest?

## File Locations

- Backtest engine: `src/backtest.py`
- Output reports: current working directory
- Generated code: current working directory
