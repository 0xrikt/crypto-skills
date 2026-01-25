# 🤖 AI Agent 使用指南

本文档描述了 AI Agent（如 Claude、GPT）应该如何使用这个 Skill。

---

## 核心流程

```
用户自然语言意图 
    ↓
[Step 1] 调用 generate_strategy_from_intent() 
    ↓
展示策略配置给用户，等待确认/修改
    ↓
[Step 2] 用户确认后，调用 run_backtest()
    ↓
[Step 3] 调用 generate_reports() 生成可视化报告
    ↓
[Step 4] 调用 generate_code() 生成可运行代码
    ↓
展示结果给用户
```

**⚠️ 重要：必须在用户确认策略后才能执行回测！**

---

## Step 1: 从意图生成策略

```python
from src.skill import generate_strategy_from_intent

result = generate_strategy_from_intent(
    intent="BTC 被低估的时候买入 100 USDT",  # 用户的原始描述
    symbol="BTC/USDT",                        # 交易对
    timeframe="4h",                           # 时间周期
    position_size=100.0,                      # 每次交易金额
    initial_capital=10000.0                   # 初始资金
)

# result 包含:
# - strategy_config: 策略配置 (dict)
# - strategy_display: 人类可读的策略展示 (markdown)
# - needs_confirmation: True (必须等用户确认)
```

### Agent 应该展示给用户：

```markdown
## 📊 策略配置预览

**请仔细检查以下策略配置，确认或修改后继续：**

| 配置项 | 值 |
|--------|-----|
| 策略名称 | RSI + MACD 组合策略 (BTC/USDT) |
| 交易对 | BTC/USDT |
| 时间周期 | 4h |
| 入场条件 | RSI(14) < 35 AND MACD(12,26,9) cross_above MACD_SIGNAL(12,26,9) |
| 止损 | 3.0% |
| 止盈 | 6.0% |
| ... | ... |

---

**下一步操作：**
1. ✅ 如果配置正确，请说"确认"或"开始回测"
2. ✏️ 如果需要修改，请直接告诉我要改什么（例如："把止损改成 5%"）
```

---

## Step 2: 处理用户反馈

### 如果用户确认：

```python
from src.skill import run_backtest

backtest_result = run_backtest(
    strategy_config=result["strategy_config"],
    days=365,        # 回测天数
    exchange="binance"
)

# backtest_result 包含:
# - result: BacktestResult 对象
# - summary: 摘要统计 (dict)
# - markdown_report: Markdown 格式报告
```

### 如果用户要修改：

用户可能说：
- "把止损改成 5%"
- "入场条件加上 RSI < 25"
- "换成 MACD 策略"

Agent 应该修改 `strategy_config` 然后重新展示给用户确认。

---

## Step 3: 生成报告

```python
from src.skill import generate_reports

report_files = generate_reports(
    result=backtest_result["result"],
    output_dir="./output"
)

# 返回生成的文件路径
```

---

## Step 4: 生成代码

```python
from src.skill import generate_code

code_files = generate_code(
    strategy_config=result["strategy_config"],
    output_dir="./output"
)

# 返回:
# - files: 文件路径 dict
# - standalone_code: 独立脚本代码
# - freqtrade_code: Freqtrade 策略代码
# - bot_code: 实盘 Bot 代码
```

---

## 完整对话示例

```
用户: "帮我测试一下，BTC 被低估的时候买入 100 USDT，被高估的时候卖出，每6个小时检查一次"

Agent: 
我帮您生成了一个交易策略配置。这个策略使用 RSI + MACD 组合来判断"低估"和"高估"：

## 📊 策略配置预览

| 配置项 | 值 |
|--------|-----|
| 策略名称 | RSI + MACD 组合策略 (BTC/USDT) |
| 交易对 | BTC/USDT |
| 时间周期 | 6h |
| 入场条件 | RSI(14) < 35 AND MACD 金叉 |
| 出场条件 | RSI(14) > 70 OR 止损 3% OR 止盈 6% |
| 仓位 | 固定金额 100 USDT |

请确认这个配置，或告诉我需要修改什么？

---

用户: "RSI 阈值改成 30"

Agent:
好的，我把入场条件的 RSI 阈值从 35 改成了 30：

| 配置项 | 值 |
|--------|-----|
| 入场条件 | RSI(14) < 30 AND MACD 金叉 |
| ... | ... |

确认这个配置吗？

---

用户: "确认，开始回测"

Agent:
正在获取 BTC/USDT 过去一年的 6h K线数据并执行回测...

## 📊 回测结果

| 指标 | 值 |
|------|-----|
| 总收益率 | +15.3% |
| 夏普比率 | 1.25 |
| 最大回撤 | 8.2% |
| 胜率 | 58% |
| 总交易次数 | 42 |

[显示资金曲线图]

我还生成了可运行的代码，您可以：
1. 下载独立回测脚本继续优化
2. 导入到 Freqtrade 进行更详细的回测
3. 使用实盘 Bot 代码（请先用小资金测试！）
```

---

## 策略修改指南

### 常见修改请求及处理方式

| 用户说 | Agent 应该修改的字段 |
|--------|---------------------|
| "止损改成 X%" | `exit.stop_loss.value` |
| "止盈改成 X%" | `exit.take_profit.value` |
| "RSI 阈值改成 X" | `entry.conditions[?].value` |
| "时间周期改成 X" | `timeframe` |
| "仓位改成 X USDT" | `position_sizing.value` |
| "加上移动止损" | `exit.trailing_stop.enabled = True` |
| "换成 XXX 策略" | 重新调用 `generate_strategy_from_intent` |

### 策略配置 JSON 结构

```json
{
  "name": "策略名称",
  "description": "描述",
  "symbol": "BTC/USDT",
  "timeframe": "4h",
  
  "entry": {
    "logic": "AND",  // 或 "OR"
    "conditions": [
      {
        "indicator": "RSI",
        "params": {"period": 14},
        "operator": "<",  // >, >=, <=, ==, cross_above, cross_below
        "value": 30
      }
    ]
  },
  
  "exit": {
    "stop_loss": {"type": "percent", "value": 3.0},
    "take_profit": {"type": "percent", "value": 6.0},
    "trailing_stop": {
      "enabled": false,
      "type": "percent",
      "value": 2.0,
      "activation_percent": 3.0
    },
    "conditions": {...}
  },
  
  "position_sizing": {
    "type": "fixed_amount",  // 或 "percent_equity"
    "value": 100,
    "max_positions": 1
  },
  
  "risk_management": {
    "max_drawdown_percent": 15.0,
    "daily_loss_limit_percent": 5.0,
    "position_limit_percent": 20.0
  },
  
  "initial_capital": 10000.0,
  "commission_percent": 0.1,
  "slippage_percent": 0.05
}
```

---

## 支持的指标

| 指标 | 参数 | 示例 |
|------|------|------|
| RSI | period | RSI(14) |
| MACD | fast_period, slow_period, signal_period | MACD(12,26,9) |
| MACD_SIGNAL | (同上) | |
| SMA | period | SMA(20) |
| EMA | period | EMA(12) |
| BBANDS_UPPER | period, std_dev | BB(20,2.0) 上轨 |
| BBANDS_LOWER | period, std_dev | BB(20,2.0) 下轨 |
| ATR | period | ATR(14) |
| PRICE | - | 当前价格 |
| VOLUME | - | 成交量 |

---

## 错误处理

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| "无法获取市场数据" | 交易对不存在或网络问题 | 检查交易对格式，确保网络连接 |
| "策略配置无效" | JSON 格式错误 | 检查配置格式 |
| "没有交易信号" | 条件太严格 | 放宽入场条件 |

---

## 最佳实践

1. **始终等待用户确认** - 不要自动执行回测
2. **解释策略逻辑** - 用通俗语言解释技术指标
3. **展示风险提示** - 提醒用户回测不代表未来收益
4. **提供修改建议** - 如果策略不合理，主动建议修改
5. **完整展示结果** - 不仅展示收益，也要展示风险指标
