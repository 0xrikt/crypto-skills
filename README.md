# 🚀 Crypto Strategy Backtest Skill

**让每个人都能用自然语言验证自己的交易策略想法。**

这是一个为 AI Agent 设计的 Skill，帮助用户：

1. 📝 用自然语言描述交易策略想法
2. ⚙️ 自动生成完整的策略配置（用户可以修改确认）
3. 📊 执行真实历史数据回测
4. 📈 生成精美的可视化报告
5. 💻 输出可直接运行的交易代码

---

## ✨ 特性

- **自然语言输入**: "BTC 被低估的时候买入，高估的时候卖出"
- **策略模板库**: 内置 5+ 经典策略模板
- **真实回测**: 使用 Binance 历史数据
- **交互式确认**: 用户可以在回测前修改策略参数
- **可视化报告**: 资金曲线、回撤分析、交易统计
- **多格式代码输出**: 独立脚本、Freqtrade、实盘 Bot

---

## 🛠️ 安装

```bash
# 克隆项目
git clone https://github.com/your-username/crypto-strategy-backtest-skill.git
cd crypto-strategy-backtest-skill

# 安装依赖
pip install -r requirements.txt
```

### 依赖

- Python 3.10+
- pandas, numpy
- ccxt (交易所数据)
- ta (技术指标)
- plotly (可视化)
- pydantic (数据验证)

---

## 🎯 快速开始

### 方式 1: 作为 Skill 被 AI Agent 调用

这个 Skill 设计为被 AI Agent（如 Claude、GPT）调用。核心流程：

```python
from src.skill import (
    generate_strategy_from_intent,
    run_backtest,
    generate_reports,
    generate_code
)

# 1. 从用户意图生成策略
result = generate_strategy_from_intent(
    intent="BTC 被低估的时候买入 100 USDT，被高估的时候卖出",
    symbol="BTC/USDT",
    timeframe="4h"
)

# 2. 展示策略给用户确认
print(result["strategy_display"])

# 3. 用户确认后，运行回测
backtest_result = run_backtest(result["strategy_config"], days=365)

# 4. 生成报告
report_files = generate_reports(backtest_result["result"])

# 5. 生成代码
code_files = generate_code(result["strategy_config"])
```

### 方式 2: 命令行使用

```bash
# 查看可用策略模板
python -m src.skill --list-templates

# 从意图生成策略（预览）
python -m src.skill --intent "BTC 超卖时买入" --symbol BTC/USDT

# 运行完整流程
python -m src.skill --intent "RSI 低于 30 时买入" --run-backtest --output ./output
```

---

## 📋 可用策略模板

| 模板 | 描述 | 类别 | 风险等级 |
|------|------|------|----------|
| `rsi_oversold` | RSI 超卖反弹策略 | 均值回归 | 中等 |
| `macd_crossover` | MACD 金叉策略 | 趋势跟踪 | 中等 |
| `rsi_macd_combo` | RSI + MACD 组合 | 组合策略 | 中低 |
| `bollinger_bounce` | 布林带下轨反弹 | 均值回归 | 中等 |
| `sma_crossover` | 双均线交叉 | 趋势跟踪 | 低 |

---

## 📊 策略配置格式

策略使用 JSON 格式定义，支持以下配置：

```json
{
  "name": "策略名称",
  "description": "策略描述",
  "symbol": "BTC/USDT",
  "timeframe": "4h",
  
  "entry": {
    "logic": "AND",
    "conditions": [
      {"indicator": "RSI", "params": {"period": 14}, "operator": "<", "value": 30}
    ]
  },
  
  "exit": {
    "stop_loss": {"type": "percent", "value": 3.0},
    "take_profit": {"type": "percent", "value": 6.0},
    "trailing_stop": {"enabled": true, "value": 2.0},
    "conditions": {...}
  },
  
  "position_sizing": {
    "type": "fixed_amount",
    "value": 100
  },
  
  "risk_management": {
    "max_drawdown_percent": 15.0
  }
}
```

### 支持的技术指标

- **RSI**: 相对强弱指数
- **MACD**: 移动平均收敛散度
- **SMA/EMA**: 简单/指数移动平均
- **Bollinger Bands**: 布林带
- **ATR**: 真实波动幅度

### 支持的比较运算符

- `>`, `>=`, `<`, `<=`, `==`
- `cross_above`: 上穿
- `cross_below`: 下穿

---

## 📈 回测报告示例

回测完成后，会生成：

1. **资金曲线图**: 展示资金变化和回撤
2. **交易分析图**: 盈亏分布、累计收益、持仓时间
3. **绩效指标卡**: 夏普比率、胜率、盈利因子等
4. **月度收益热力图**: 按月份展示收益

![报告示例](docs/report_example.png)

---

## 💻 生成的代码格式

### 1. 独立回测脚本 (standalone)

```python
# 可直接运行的 Python 脚本
# 包含数据获取、指标计算、回测逻辑、结果展示
python your_strategy.py
```

### 2. Freqtrade 策略 (freqtrade)

```python
# 可直接用于 Freqtrade 的策略类
freqtrade backtesting --strategy YourStrategy
```

### 3. 实盘 Bot (ccxt_bot)

```python
# 基于 ccxt 的实盘交易 bot
# ⚠️ 请先用小资金测试！
python your_strategy_bot.py
```

---

## 🔌 作为 MCP Server 使用

（开发中）

未来将支持作为 MCP Server 运行，让任何支持 MCP 的 AI 应用都能调用。

---

## ⚠️ 免责声明

- 本工具仅供学习和研究使用
- **回测结果不代表未来收益**
- 实盘交易有风险，请谨慎使用
- 永远不要投入超过你能承受损失的资金

---

## 🤝 贡献

欢迎贡献代码、策略模板、Bug 修复！

```bash
# 开发安装
pip install -e ".[dev]"

# 运行测试
pytest tests/
```

---

## 📜 License

MIT License

---

## 🙏 致谢

- [ccxt](https://github.com/ccxt/ccxt) - 交易所 API
- [ta](https://github.com/bukosabino/ta) - 技术指标
- [Freqtrade](https://github.com/freqtrade/freqtrade) - 量化交易框架
- [Plotly](https://plotly.com/) - 可视化

---

**Made with ❤️ for the crypto community**
