# 🚀 Crypto Backtest Skill

**几分钟验证你的交易策略想法**

Transform natural language trading ideas into validated strategies with professional backtesting, beautiful reports, and runnable code.

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

| Indicator | Syntax | Example |
|-----------|--------|---------|
| RSI | `rsi` | `rsi<30`, `rsi>70` |
| MACD | `macd`, `macd_signal` | `macd>macd_signal` |
| SMA | `sma{period}` | `price>sma50` |
| EMA | `ema{period}` | `price<ema21` |
| Bollinger Bands | `bb_upper`, `bb_lower` | `price>bb_upper` |

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
--exchange      Exchange to fetch data from (default: binance)
--entry         Entry conditions, comma-separated (default: rsi<30)
--exit          Exit conditions, comma-separated (default: rsi>70)
--stop-loss     Stop loss percentage (default: 5)
--take-profit   Take profit percentage (default: 15)
--position-size Position size as % of portfolio (default: 10)
--initial-capital Starting capital (default: 10000)
--commission    Commission percentage (default: 0.1)
--output        Output HTML file path
--name          Strategy name for the report
```

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
