---
name: crypto-backtest
description: |
  Backtest crypto trading strategies from natural language ideas.
  Use when: user describes trading ideas, wants to validate strategies, mentions
  "backtest", "trading strategy", "buy low sell high", "RSI", "MACD", "oversold",
  "overbought", "crypto strategy", "validate strategy", "backtest", "DCA", or similar.
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

### 🎯 "Undervalued/Cheap/Oversold/Dip" → Multi-Factor Valuation Model

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

### 📈 "Overvalued/Expensive/Overbought" → Multi-Factor Model

| Dimension | Indicator | Bearish Signal | Weight |
|-----------|-----------|----------------|--------|
| **Momentum** | RSI(14) | > 70 | 1.0 |
| **Trend Extension** | Price vs SMA(200) | Price > SMA200 × 1.3 | 1.0 |
| **Volatility Band** | Bollinger Bands | Price > BB_Upper | 1.0 |
| **From Recent Low** | Price vs 90-day Low | Gain > 50% | 1.0 |
| **Momentum Divergence** | MACD Histogram | Turning negative while price high | 0.5 |
| **Volume Dry-up** | Volume vs MA(20) | Volume declining on rally | 0.5 |

### 🚀 "Trend/Bullish/Uptrend" → Multi-Timeframe Confirmation

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

### 💥 "Breakout" → Volume-Confirmed Breakout

**DON'T:** `Price > BB_Upper` (many false breakouts)

**DO:** Require multiple confirmations:

| Condition | Purpose |
|-----------|---------|
| Price > BB_Upper(20, 2.0) | Statistical breakout |
| Volume > 2.0 × Volume_MA(20) | Strong participation |
| Close in top 25% of candle range | Buying pressure |
| RSI(14) > 50 but < 80 | Momentum without exhaustion |
| Previous 5 candles: tight range (BB width contracting) | Coiled energy |

### 📊 "DCA" → Smart DCA with Valuation Adjustment

**DON'T:** Fixed amount every period (misses opportunities)

**DO:** Dynamic allocation based on valuation score:

| Valuation Score | Market State | Allocation |
|-----------------|--------------|------------|
| ≥ +3.0 | 🟢🟢 Extreme undervaluation | Base × 2.0 |
| +1.5 to +3.0 | 🟢 Undervalued | Base × 1.5 |
| -1.5 to +1.5 | 🟡 Fair value | Base × 1.0 |
| -3.0 to -1.5 | 🔴 Overvalued | Base × 0.5 |
| ≤ -3.0 | 🔴🔴 Extreme overvaluation | Base × 0.25 |

### 🔄 "Mean Reversion" → Statistical Deviation Strategy

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

**CRITICAL**: Show the full multi-factor strategy in a structured, professional format. Users should be impressed by the thoroughness.

Format (use 5 standard sections: Data, Signal, Capital, Risk, Execution):

```markdown
## 📊 Strategy: [Descriptive Name]

**Core Logic**: [One sentence explaining the edge]

---

### 📈 DATA

| Parameter | Value |
|-----------|-------|
| **Primary Symbol** | BTC/USDT |
| **Timeframe** | 4h |
| **Backtest Period** | 365 days |

**Indicators:**
| Indicator | Parameters |
|-----------|------------|
| RSI | period: 14 |
| SMA | period: 50, 200 |
| Bollinger Bands | period: 20, std_dev: 2 |

---

### 🎯 SIGNAL

**Entry Conditions (ALL must be met):**
| # | Indicator | Operator | Value/Reference |
|---|-----------|----------|-----------------|
| 1 | RSI | < | 35 |
| 2 | Price | < | BB_lower |
| 3 | Price | < | SMA_98% |

**Exit Conditions (ANY triggers exit):**
| # | Indicator | Operator | Value/Reference |
|---|-----------|----------|-----------------|
| 1 | RSI | > | 70 |
| 2 | Price | > | BB_upper |
| 3 | Price | > | SMA_105% |

**Execution Schedule:**
- Frequency: every 4h
- Check times: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC

---

### 💰 CAPITAL

| Parameter | Value |
|-----------|-------|
| Total Capital | $10,000 |
| Allocation per Trade | $200 (fixed) or 10% (percentage) |
| Reserve Ratio | 20% (kept as cash buffer) |
| Max Drawdown Limit | 15% |

---

### ⚠️ RISK

| Parameter | Value |
|-----------|-------|
| Stop Loss | 8% |
| Take Profit | 15% (or trailing) |
| Max Account Risk | 75% |
| Max Positions | 1 |

**Emergency Rules:**
- If account drawdown > 15%: pause trading
- If single trade loss > 8%: close position

---

### ⚙️ EXECUTION

| Parameter | Value |
|-----------|-------|
| Order Type | market |
| Position Side | long_only / long_short |
| Leverage | 1x (spot) or 3x (perp) |
| Position Sizing | fixed_amount: $200 |

---

**✅ Confirm to run backtest? Or let me know what to adjust.**
```

### Key Principles for Strategy Presentation:

1. **Indicator References**: Support percentage-based references
   - `SMA_98%` = 98% of SMA value (buy below)
   - `SMA_105%` = 105% of SMA value (sell above)
   - `BB_lower`, `BB_upper`, `BB_middle` = Bollinger Band levels

2. **Condition Logic**: Always specify clearly
   - Entry: `ALL` conditions must be met (AND logic)
   - Exit: `ANY` condition triggers (OR logic)

3. **Risk Management**: Include emergency rules
   - Account-level risk limits
   - Position-level stop losses
   - Reserve capital buffer

4. **Be Specific**: Use exact numbers, not vague descriptions
   - ❌ "buy when cheap"
   - ✅ "buy when RSI < 35 AND price < SMA_98%"

### Step 3: Run Backtest

**IMPORTANT**: Detect the user's language and pass the `--lang` parameter accordingly:
- If user writes in Chinese → `--lang zh`
- If user writes in English → `--lang en`

This ensures the HTML report text matches the user's language.

```bash
python src/backtest.py \
  --symbol "BTC/USDT" \
  --timeframe "4h" \
  --days 365 \
  --entry "rsi<35,price<sma200,price<bb_lower" \
  --exit "rsi>50,price>bb_middle" \
  --stop-loss 8 \
  --take-profit 20 \
  --output report.html \
  --lang zh  # or --lang en based on user's language
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

### Template 1: Multi-Factor Value Buying

**DATA:**
- Symbol: BTC/USDT | Timeframe: 4h | Period: 365d
- Indicators: RSI(14), SMA(200), BB(20,2), High_90

**SIGNAL:**
| Entry (ALL) | Exit (ANY) |
|-------------|------------|
| RSI < 35 | RSI > 65 |
| Price < SMA200_98pct | Price > SMA200 |
| Price < BB_lower | Price > BB_middle |
| Drawdown > 25% | Stop Loss 10% |

**RISK:** Stop 10% | Take Profit 25% | Position 10%

---

### Template 2: Trend Following with Confirmation

**DATA:**
- Symbol: BTC/USDT | Timeframe: 4h | Period: 365d
- Indicators: SMA(200), EMA(9,21,50), MACD, ADX

**SIGNAL:**
| Entry (ALL) | Exit (ANY) |
|-------------|------------|
| Price > SMA200 | EMA9 < EMA21 |
| EMA9 > EMA21 | Price < SMA50 |
| MACD > MACD_signal | MACD crossunder |
| ADX > 25 | Stop Loss 8% |

**RISK:** Stop 8% | Trailing Stop 3xATR | Position 15%

---

### Template 3: Volume-Confirmed Breakout

**DATA:**
- Symbol: BTC/USDT | Timeframe: 1h | Period: 180d
- Indicators: High_20, Volume_MA(20), RSI(14), BB(20,2)

**SIGNAL:**
| Entry (ALL) | Exit (ANY) |
|-------------|------------|
| Price > High_20 | Price < EMA21 |
| Volume > Volume_MA_200pct | RSI > 80 |
| RSI between 50-75 | Stop Loss 5% |
| BB_width contracting | Take Profit 15% |

**RISK:** Stop 5% | Take Profit 15% | Position 20%

---

### Template 4: Smart DCA

**DATA:**
- Symbol: BTC/USDT | Timeframe: 1d | Frequency: Weekly

**SIGNAL (Valuation-Based Allocation):**
| Score | Market State | Allocation |
|-------|--------------|------------|
| ≥ +3 | 🟢🟢 Strong buy zone | Base × 2.0 |
| +1.5 to +3 | 🟢 Undervalued | Base × 1.5 |
| -1.5 to +1.5 | 🟡 Fair value | Base × 1.0 |
| -3 to -1.5 | 🔴 Overvalued | Base × 0.5 |
| ≤ -3 | 🔴🔴 Extreme caution | Base × 0.25 |

**CAPITAL:** Base $200/week | Reserve 20% | Max DD 15%

---

### Template 5: Example from User (Professional Format)

**DATA:**
```yaml
indicators:
  BB: { period: 20, std_dev: 2 }
  RSI: { period: 14 }
  SMA: { period: 50 }
timeframe: 2h
symbol: BTC-PERP
```

**SIGNAL:**
```yaml
entry_conditions (ALL):
  - RSI < 35
  - Price < BB_lower  
  - Price < SMA_98pct

exit_conditions (ANY):
  - RSI > 70
  - Price > BB_upper
  - Price > SMA_105pct

execution_schedule:
  frequency: 2h
  check_times: [00:00, 02:00, 04:00, ..., 22:00]
```

**CAPITAL:**
```yaml
total_capital: 1000
allocation_per_trade: 200
reserve_ratio: 0.2
max_drawdown_limit: 0.15
```

**RISK:**
```yaml
stop_loss: 8%
take_profit: null (exit by signal)
max_account_risk: 75%
emergency_rules:
  account_risk_threshold: 0.8 → close_all
```

**EXECUTION:**
```yaml
leverage: 3x
order_type: market
position_side: long_only
max_positions: 1
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

## Technical Reference: Available Indicators

### Momentum Indicators
| Indicator | Column Name | Description |
|-----------|-------------|-------------|
| RSI | `rsi` | Relative Strength Index (14) |
| Stochastic %K | `stoch_k` | Stochastic oscillator K line |
| Stochastic %D | `stoch_d` | Stochastic oscillator D line |
| Williams %R | `willr` | Williams %R (14) |
| CCI | `cci` | Commodity Channel Index (20) |
| MFI | `mfi` | Money Flow Index (14) |
| ROC | `roc`, `roc_20` | Rate of Change (10, 20) |
| MACD | `macd`, `macd_signal`, `macd_hist` | MACD line, signal, histogram |

### Trend Indicators
| Indicator | Column Name | Description |
|-----------|-------------|-------------|
| SMA | `sma9`, `sma21`, `sma50`, `sma100`, `sma200` | Simple Moving Averages |
| EMA | `ema9`, `ema21`, `ema50`, `ema100`, `ema200` | Exponential Moving Averages |
| ADX | `adx` | Average Directional Index (trend strength) |
| +DI / -DI | `plus_di`, `minus_di` | Directional Indicators |

### Volatility Indicators
| Indicator | Column Name | Description |
|-----------|-------------|-------------|
| Bollinger Bands | `bb_upper`, `bb_middle`, `bb_lower` | Upper, middle, lower bands |
| BB Width | `bb_width` | Band width (volatility measure) |
| BB %B | `bb_pct` | Price position in BB range (0-1) |
| ATR | `atr` | Average True Range (14) |
| ATR % | `atr_pct` | ATR as % of price |

### Volume Indicators
| Indicator | Column Name | Description |
|-----------|-------------|-------------|
| Volume SMA | `volume_sma` | 20-period volume average |
| Volume Ratio | `volume_ratio` | Current volume / average |
| OBV | `obv`, `obv_sma` | On-Balance Volume |

### Price Position Indicators
| Indicator | Column Name | Description |
|-----------|-------------|-------------|
| Rolling High | `high_20`, `high_50`, `high_90`, `high_200` | N-period high |
| Rolling Low | `low_20`, `low_50`, `low_90`, `low_200` | N-period low |
| Drawdown | `drawdown`, `drawdown_50` | % from rolling high |
| Price Position | `price_position_90` | Position in 90-day range (0-1) |
| Distance from MA | `dist_sma50`, `dist_sma200` | % distance from MA |

### Derived / Change Indicators
| Indicator | Column Name | Description |
|-----------|-------------|-------------|
| Price Change | `price_change`, `price_pct_change` | 1-period change |
| Price Change 5 | `price_change_5`, `price_pct_change_5` | 5-period change |
| RSI Change | `rsi_change` | RSI momentum |
| MACD Change | `macd_change`, `macd_hist_change` | MACD momentum |
| Consecutive Up | `consecutive_up` | Count of consecutive up days |
| Consecutive Down | `consecutive_down` | Count of consecutive down days |

---

## Technical Reference: Condition Syntax

### 1. Simple Comparisons
```
rsi<30              # RSI below 30
price>sma200        # Price above SMA 200
adx>=25             # ADX at least 25
bb_pct<0.2          # Price in lower 20% of BB range
drawdown<-20        # Down 20% from recent high
volume_ratio>2      # Volume 2x above average
```

### 2. Crossover / Crossunder
```
macd_crossover                  # MACD crosses above signal (default)
ema9_cross_above_ema21          # EMA9 crosses above EMA21
price_crossover_sma50           # Price crosses above SMA50
rsi_crossunder_50               # RSI crosses below 50
stoch_k_cross_above_stoch_d     # Stochastic golden cross
```

### 3. Turning Points
```
rsi_turning_up          # RSI starts increasing
macd_hist_turning_down  # MACD histogram starts decreasing
price_turning_up        # Price reversal upward
```

### 4. Consecutive Periods
```
rsi<30_for_3            # RSI below 30 for 3 consecutive periods
price>sma200_for_5      # Price above SMA200 for 5 periods
consecutive_up>=3       # At least 3 consecutive up days
```

### 5. Combined Conditions
Conditions are comma-separated. Entry uses AND logic, Exit uses OR logic.
```
--entry "rsi<35,price<bb_lower,volume_ratio>1.5"
--exit "rsi>70,price>bb_upper"
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
- Smart DCA: `src/smart_dca.py`
- Output reports: current working directory
- Generated code: current working directory

## Smart DCA Usage

For DCA strategies, use the dedicated Smart DCA script:

```bash
python src/smart_dca.py \
  --symbol "BTC/USDT" \
  --days 1095 \
  --base-amount 200 \
  --frequency 7 \
  --output smart_dca_report.html \
  --lang zh  # or --lang en based on user's language
```

## Language Support

**CRITICAL**: Always detect the user's language and pass the appropriate `--lang` parameter:
- User writes in Chinese → `--lang zh` (report in Chinese)
- User writes in English → `--lang en` (report in English)

This ensures the generated HTML report matches the user's language preference.
