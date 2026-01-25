---
name: crypto-backtest
description: |
  Backtest crypto trading strategies from natural language ideas.
  Use when: user describes trading ideas, wants to validate strategies, mentions
  "backtest", "trading strategy", "buy low sell high", "RSI", "MACD", "oversold",
  "overbought", "crypto strategy", "验证策略", "回测", "交易策略", "定投", "DCA", or similar.
allowed-tools: Bash, Read, Write, Edit, Grep, Glob
---

# Crypto Strategy Backtest Skill

Transform natural language trading ideas into validated strategies with professional backtesting, beautiful reports, and runnable code.

## Your Superpower

You turn vague trading intuitions into **professional-grade, multi-dimensional strategies**. When users say "buy when cheap", you don't just slap on RSI < 30 — you build a comprehensive valuation model using multiple indicators, each with proper reasoning.

**Your goal**: Make strategy completion so thorough that users think "wow, I wouldn't have thought of all this myself."

---

## CRITICAL: Strategy Completion Standards

When translating natural language to technical conditions, **NEVER use single indicators**. Always combine multiple dimensions:

### 🎯 "低估/Undervalued/Cheap/Oversold/Dip" → Multi-Factor Valuation Model

**DON'T:** `RSI(14) < 30` (too simplistic, easily fooled by trends)

**DO:** Combine 4-5 indicators for robust valuation scoring:

| Dimension | Indicator | Bullish Signal | Weight |
|-----------|-----------|----------------|--------|
| **Momentum** | RSI(14) | < 35 | 1.0 |
| **Trend Position** | Price vs SMA(200) | Price < SMA200 | 1.0 |
| **Volatility Band** | Bollinger Bands | Price < BB_Lower | 1.0 |
| **Drawdown** | Price vs 90-day High | Drawdown > 25% | 1.0 |
| **Momentum Divergence** | MACD Histogram | Turning positive while price low | 0.5 |
| **Volume Confirmation** | Volume vs MA(20) | Volume spike (>1.5x) on dip | 0.5 |

**Valuation Score** = Sum of triggered signals × weights
- Score ≥ 3.0: Strong undervaluation
- Score 2.0-3.0: Moderate undervaluation
- Score < 2.0: Weak/no signal

### 📈 "高估/Overvalued/Expensive/Overbought" → Multi-Factor Model

| Dimension | Indicator | Bearish Signal | Weight |
|-----------|-----------|----------------|--------|
| **Momentum** | RSI(14) | > 70 | 1.0 |
| **Trend Extension** | Price vs SMA(200) | Price > SMA200 × 1.3 | 1.0 |
| **Volatility Band** | Bollinger Bands | Price > BB_Upper | 1.0 |
| **From Recent Low** | Price vs 90-day Low | Gain > 50% | 1.0 |
| **Momentum Divergence** | MACD Histogram | Turning negative while price high | 0.5 |
| **Volume Dry-up** | Volume vs MA(20) | Volume declining on rally | 0.5 |

### 🚀 "趋势/Trend/Bullish/Uptrend" → Multi-Timeframe Confirmation

**DON'T:** `Price > EMA(21)` (single timeframe, easily whipsawed)

**DO:** Require alignment across timeframes:

| Timeframe | Condition | Purpose |
|-----------|-----------|---------|
| **Long-term** | Price > SMA(200) | Major trend direction |
| **Medium-term** | Price > EMA(50) | Intermediate trend |
| **Short-term** | EMA(9) > EMA(21) | Recent momentum |
| **Momentum** | MACD > Signal Line | Acceleration |
| **Strength** | ADX > 25 | Trend strength confirmation |

**Entry**: All conditions aligned
**Exit**: Short-term reversal (EMA9 < EMA21) OR momentum loss (MACD cross down)

### 💥 "突破/Breakout" → Volume-Confirmed Breakout

**DON'T:** `Price > BB_Upper` (many false breakouts)

**DO:** Require multiple confirmations:

| Condition | Purpose |
|-----------|---------|
| Price > BB_Upper(20, 2.0) | Statistical breakout |
| Volume > 2.0 × Volume_MA(20) | Strong participation |
| Close in top 25% of candle range | Buying pressure |
| RSI(14) > 50 but < 80 | Momentum without exhaustion |
| Previous 5 candles: tight range (BB width contracting) | Coiled energy |

### 📊 "定投/DCA" → Smart DCA with Valuation Adjustment

**DON'T:** Fixed amount every period (misses opportunities)

**DO:** Dynamic allocation based on valuation score:

| Valuation Score | Market State | Allocation |
|-----------------|--------------|------------|
| ≥ +3.0 | 🟢🟢 Extreme undervaluation | Base × 2.0 |
| +1.5 to +3.0 | 🟢 Undervalued | Base × 1.5 |
| -1.5 to +1.5 | 🟡 Fair value | Base × 1.0 |
| -3.0 to -1.5 | 🔴 Overvalued | Base × 0.5 |
| ≤ -3.0 | 🔴🔴 Extreme overvaluation | Base × 0.25 |

### 🔄 "均值回归/Mean Reversion" → Statistical Deviation Strategy

| Condition | Entry | Exit |
|-----------|-------|------|
| **Z-Score** | Price Z-score < -2.0 | Z-score > 0 |
| **BB Position** | Price < BB_Lower | Price > BB_Middle |
| **RSI** | RSI < 30 | RSI > 50 |
| **Confirmation** | Volume spike on dip | - |

---

## Workflow

### Step 1: Understand & Expand the Intent

When user describes a trading idea:

1. **Identify the core strategy type**: Mean reversion? Trend following? Breakout? DCA?
2. **Extract constraints**: Asset, timeframe, risk tolerance
3. **Expand to multi-dimensional conditions** using the templates above
4. **Add appropriate risk management** based on strategy type

### Step 2: Present Complete Strategy for Confirmation

**CRITICAL**: Show the full multi-factor strategy. Users should be impressed by the thoroughness.

Format:

```markdown
## 📊 Strategy: [Descriptive Name]

**Core Logic**: [One sentence explaining the edge]

**Asset:** BTC/USDT  
**Timeframe:** 4h  
**Backtest Period:** 365 days

### Valuation/Signal Model:

| Factor | Indicator | Condition | Weight |
|--------|-----------|-----------|--------|
| Momentum | RSI(14) | < 35 | 1.0 |
| Trend | Price vs SMA(200) | Below | 1.0 |
| ... | ... | ... | ... |

**Entry Trigger:** Score ≥ X.X

### Exit Conditions (ANY triggers):
- [Condition 1]
- [Condition 2]
- Stop Loss: X%
- Take Profit: X%

### Risk Management:
- Position Size: X% per trade
- Max Drawdown Tolerance: X%

---
**确认运行回测？或告诉我需要调整的地方。**
```

### Step 3: Run Backtest

```bash
python src/backtest.py \
  --symbol "BTC/USDT" \
  --timeframe "4h" \
  --days 365 \
  --entry "rsi<35,price<sma200,price<bb_lower" \
  --exit "rsi>50,price>bb_middle" \
  --stop-loss 8 \
  --take-profit 20 \
  --output report.html
```

### Step 4: Present Results with Insights

Show metrics AND provide actionable insights:

```markdown
## 📈 Backtest Results

| Metric | Value | Assessment |
|--------|-------|------------|
| Total Return | +47.3% | ✅ Beats B&H |
| Max Drawdown | -18.2% | ⚠️ Moderate |
| Sharpe Ratio | 1.42 | ✅ Good |
| Win Rate | 64% | ✅ Healthy |
| Profit Factor | 2.1 | ✅ Strong |

### Key Insights:
- Strategy performed best during [market condition]
- Largest drawdown occurred during [event]
- Consider [specific improvement] to reduce drawdown

### Generated Files:
- `report.html` - Interactive visual report
- `strategy.py` - Runnable Python code
```

### Step 5: Suggest Iterations

Based on results, proactively suggest:
- Parameter optimizations
- Additional filters
- Alternative approaches
- Risk adjustments

---

## Strategy Templates Reference

### Template 1: Multi-Factor Value Investing
```
Entry: Valuation Score ≥ 3.0
  - RSI(14) < 35
  - Price < SMA(200)
  - Price < BB_Lower(20, 2)
  - Drawdown from 90-day high > 25%
  
Exit: Valuation Score ≤ 0 OR Take Profit 20% OR Stop Loss 10%
```

### Template 2: Trend Following with Confirmation
```
Entry (ALL required):
  - Price > SMA(200) [major trend]
  - EMA(9) > EMA(21) [momentum]
  - MACD > Signal [acceleration]
  - ADX > 25 [trend strength]

Exit (ANY triggers):
  - EMA(9) < EMA(21)
  - Price < SMA(50)
  - Stop Loss: 8%
```

### Template 3: Volume-Confirmed Breakout
```
Entry (ALL required):
  - Price breaks above 20-day high
  - Volume > 2x average
  - RSI(14) between 50-75
  - BB Width was contracting (coiling)

Exit:
  - Price < EMA(21)
  - Or trailing stop 3x ATR
```

### Template 4: Smart DCA
```
Weekly buy with dynamic sizing:
  - Valuation Score ≥ 3: Buy 2x base
  - Score 1.5-3: Buy 1.5x base
  - Score -1.5 to 1.5: Buy 1x base
  - Score -3 to -1.5: Buy 0.5x base
  - Score ≤ -3: Buy 0.25x base
```

---

## Technical Reference

### Indicators Available (pandas-ta)
```
Momentum: RSI, MACD, Stochastic, Williams %R, CCI
Trend: SMA, EMA, ADX, Aroon, Supertrend
Volatility: Bollinger Bands, ATR, Keltner Channels
Volume: OBV, Volume SMA, VWAP
```

### Risk Profiles
```
Conservative:  SL=5%,  TP=12%, position=5%,  max 3 concurrent
Moderate:      SL=8%,  TP=20%, position=10%, max 5 concurrent
Aggressive:    SL=12%, TP=35%, position=20%, max 8 concurrent
```

---

## Important Guidelines

1. **Never use single indicators** - Always combine multiple dimensions
2. **Explain the logic** - Users should understand WHY each indicator is included
3. **Match complexity to strategy** - DCA needs valuation model, trend following needs multi-TF
4. **Be honest about limitations** - Past performance ≠ future results
5. **Encourage iteration** - Backtesting is a process, not a one-shot

---

## File Locations

- Backtest engine: `src/backtest.py`
- Output reports: current working directory
- Generated code: current working directory
