# 🚀 Crypto Backtest Skill

**Validate your trading ideas in minutes**

Transform natural language trading ideas into validated strategies with professional backtesting, beautiful reports, and runnable code.

> ⚠️ **Important**: This tool supports **SPOT trading strategies only**. No leverage, no shorting, no futures/perpetual contracts. All strategies are long-only (buy → hold → sell).

## ✨ Features

- **Natural Language Input** - Describe strategies like "buy BTC when oversold, sell when overbought"
- **Automatic Strategy Completion** - AI translates vague ideas into specific technical conditions
- **User Confirmation** - Review and modify the strategy before running
- **Professional Backtesting** - Real historical data from 200+ exchanges via CCXT
- **Beautiful Reports** - Interactive HTML reports with Plotly charts
- **Runnable Code** - Get Python scripts you can run directly

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

### As a Skill (Claude Desktop / AI Agent)

Just describe your trading idea:

```
"I want to buy ETH when it's oversold and there's fear in the market, 
then sell when it becomes overbought"
```

The AI will:
1. Translate this into technical conditions (RSI < 30, etc.)
2. Show you the complete strategy for confirmation
3. Run the backtest on real historical data
4. Generate an interactive HTML report
5. Provide runnable Python code

### Command Line

```bash
python src/backtest.py \
  --symbol BTC/USDT \
  --timeframe 4h \
  --days 365 \
  --entry "rsi<30,price<sma50" \
  --exit "rsi>70" \
  --stop-loss 5 \
  --take-profit 15 \
  --output my_strategy_report.html
```

## 📊 Supported Indicators

### Momentum Indicators
| Indicator | Syntax | Example |
|-----------|--------|---------|
| RSI | `rsi` | `rsi<30`, `rsi>70` |
| MACD | `macd`, `macd_signal`, `macd_hist` | `macd>macd_signal` |
| Stochastic | `stoch_k`, `stoch_d` | `stoch_k<20` |
| Williams %R | `williams_r` | `williams_r<-80` |
| CCI | `cci` | `cci<-100` |
| MFI | `mfi` | `mfi<20` |
| ROC | `roc` | `roc>0` |

### Trend Indicators
| Indicator | Syntax | Example |
|-----------|--------|---------|
| SMA | `sma{period}` | `price>sma50`, `price>sma200` |
| EMA | `ema{period}` | `price<ema21`, `ema9>ema21` |
| ADX | `adx`, `plus_di`, `minus_di` | `adx>25` |

### Volatility Indicators
| Indicator | Syntax | Example |
|-----------|--------|---------|
| Bollinger Bands | `bb_upper`, `bb_lower`, `bb_middle` | `price<bb_lower` |
| BB Width | `bb_width` | `bb_width<0.1` |
| BB %B | `bb_percent_b` | `bb_percent_b<0.2` |
| ATR | `atr`, `atr_pct` | `atr_pct>3` |

### Volume Indicators
| Indicator | Syntax | Example |
|-----------|--------|---------|
| Volume MA | `volume_ma` | `volume>volume_ma` |
| Volume Ratio | `volume_ratio` | `volume_ratio>2` |
| OBV | `obv` | - |

### Price Action
| Indicator | Syntax | Example |
|-----------|--------|---------|
| Rolling High/Low | `high_20`, `low_50`, `high_90` | `price>high_20` |
| Drawdown | `drawdown`, `drawdown_50` | `drawdown<-25` |
| Price Position | `price_position_90` | `price_position_90<0.3` |
| Distance from MA | `dist_sma50`, `dist_sma200` | `dist_sma200<-10` |

### Special Conditions
| Pattern | Syntax | Example |
|---------|--------|---------|
| Crossover | `{ind1}_cross_above_{ind2}` | `ema9_cross_above_ema21` |
| Crossunder | `{ind1}_cross_below_{ind2}` | `price_cross_below_sma200` |
| Turning Up | `{ind}_turning_up` | `rsi_turning_up` |
| Turning Down | `{ind}_turning_down` | `macd_hist_turning_down` |
| Consecutive | `{cond}_for_{n}` | `rsi<30_for_3` |
| Percentage Ref | `{ind}_{pct}pct` | `price<sma200_98pct` |

## 📈 Sample Output

The backtest generates:

1. **HTML Report** - Interactive charts showing:
   - Equity curve
   - Drawdown
   - Price chart with buy/sell signals
   - Performance metrics
   - Trade history

2. **Python Script** - Runnable strategy code that you can:
   - Customize further
   - Run on different assets
   - Deploy to production

## 🎨 Report Preview

The HTML report features:
- Dark theme optimized for trading
- Interactive Plotly charts
- Key metrics at a glance
- Full trade history
- Shareable design

## 📋 CLI Options

```
--symbol        Trading pair (default: BTC/USDT)
--timeframe     Candle timeframe: 1m, 5m, 15m, 1h, 4h, 1d (default: 4h)
--days          Backtest period in days (default: 365)
--exchange      Exchange to fetch data from (default: okx). Options: okx, binance, kucoin, kraken, bybit. Note: OKX ~90 day limit
--entry         Entry conditions, comma-separated (default: rsi<30)
--exit          Exit conditions, comma-separated (default: rsi>70)
--stop-loss     Stop loss percentage (default: 5)
--take-profit   Take profit percentage (default: 15)
--position-size Position size as % of portfolio, NOT dollar amount (default: 10 = 10% per trade)
--initial-capital Starting capital (default: 10000)
--commission    Commission percentage (default: 0.1)
--output        Output HTML file path
--name          Strategy name for the report
--lang          Report language: en or zh (default: en)
```

## 🧠 Smart DCA

For dollar-cost averaging strategies with valuation-based allocation:

```bash
python src/smart_dca.py \
  --symbol BTC/USDT \
  --days 1095 \
  --base-amount 200 \
  --frequency 7 \
  --output smart_dca_report.html \
  --lang zh
```

Smart DCA uses a multi-factor valuation model:
- RSI, SMA(200), Bollinger Bands, Drawdown, MACD
- Adjusts investment amount based on valuation score

## 🔄 Pair Trading / Relative Strength

For strategies that trade based on relative performance between two assets:

```bash
python src/pair_trading.py \
  --symbol-a BTC/USDT \
  --symbol-b ETH/USDT \
  --days 365 \
  --timeframe 4h \
  --lookback 20 \
  --threshold 10 \
  --exit-threshold 2 \
  --output pair_trading_report.html \
  --lang en
```

**Strategy Logic:**
- When BTC significantly outperforms ETH (spread > threshold) → Long ETH (expect catch-up)
- When ETH significantly outperforms BTC (spread < -threshold) → Long BTC (expect catch-up)
- Exit when spread returns to mean (within ±exit-threshold%)

**Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--lookback` | Period for calculating relative performance | 20 |
| `--threshold` | Entry threshold (spread % deviation) | 10 |
| `--exit-threshold` | Exit threshold (spread % to close) | 2 |

**Note:** This is a spot-only, long-only strategy. We go long the underperformer expecting mean reversion.

## ⚠️ Exchange Data Limits

**Different exchanges have different historical data limits!**

| Exchange | Approximate Limit | Notes |
|----------|-------------------|-------|
| **OKX** | ~60-90 days | Default. Good for short-term backtests |
| **KuCoin** | ~200 days | Good alternative for medium-term |
| **Binance** | 1000+ days | Most data, but blocked in some regions |
| **Bybit** | ~200 days | Good alternative |

For backtests > 90 days, use `--exchange kucoin` or `--exchange binance` (if accessible).

## 🛠 Tech Stack

- **Data**: CCXT (200+ exchanges)
- **Indicators**: pandas-ta (130+ indicators)
- **Backtesting**: Custom vectorized engine
- **Visualization**: Plotly
- **Reports**: Self-contained HTML

## ⚠️ Disclaimer

This tool is for educational and research purposes only. Past performance does not guarantee future results. Always do your own research before trading with real money.

## 📄 License

MIT

---

**Like this tool? Star the repo and share your backtest results!** ⭐

[GitHub](https://github.com/0xrikt/crypto-skills) | Created by [@0xrikt](https://github.com/0xrikt)
