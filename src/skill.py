"""
Crypto Strategy Backtest Skill

Main entry point that orchestrates the full workflow:
1. Parse natural language strategy description
2. Generate structured strategy configuration (for user review)
3. Execute backtest after user confirmation
4. Generate visual reports
5. Output runnable code

This module is designed to be called by AI agents (like Claude, GPT, etc.)
"""

import json
import os
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd

from .strategy_schema import StrategyConfig, EXAMPLE_STRATEGY
from .data_fetcher import DataFetcher, fetch_crypto_data
from .backtest_engine import BacktestEngine, BacktestResult
from .report_generator import ReportGenerator, generate_markdown_report
from .code_generator import CodeGenerator, save_strategy_code


# ============================================================================
# STRATEGY TEMPLATES
# ============================================================================

STRATEGY_TEMPLATES = {
    "rsi_oversold": {
        "name": "RSI 超卖反弹策略",
        "description": "当 RSI 进入超卖区域时买入，超买时卖出",
        "category": "均值回归",
        "risk_level": "中等",
        "suitable_for": "震荡市场",
        "template": {
            "entry": {
                "logic": "AND",
                "conditions": [
                    {"indicator": "RSI", "params": {"period": 14}, "operator": "<", "value": 30}
                ]
            },
            "exit": {
                "stop_loss": {"type": "percent", "value": 3.0},
                "take_profit": {"type": "percent", "value": 6.0},
                "conditions": {
                    "logic": "OR",
                    "conditions": [
                        {"indicator": "RSI", "params": {"period": 14}, "operator": ">", "value": 70}
                    ]
                }
            }
        }
    },
    "macd_crossover": {
        "name": "MACD 金叉策略",
        "description": "MACD 线上穿信号线时买入，下穿时卖出",
        "category": "趋势跟踪",
        "risk_level": "中等",
        "suitable_for": "趋势市场",
        "template": {
            "entry": {
                "logic": "AND",
                "conditions": [
                    {
                        "indicator": "MACD",
                        "params": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                        "operator": "cross_above",
                        "value_indicator": "MACD_SIGNAL",
                        "value_indicator_params": {"fast_period": 12, "slow_period": 26, "signal_period": 9}
                    }
                ]
            },
            "exit": {
                "stop_loss": {"type": "percent", "value": 4.0},
                "take_profit": {"type": "percent", "value": 8.0},
                "conditions": {
                    "logic": "OR",
                    "conditions": [
                        {
                            "indicator": "MACD",
                            "params": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                            "operator": "cross_below",
                            "value_indicator": "MACD_SIGNAL",
                            "value_indicator_params": {"fast_period": 12, "slow_period": 26, "signal_period": 9}
                        }
                    ]
                }
            }
        }
    },
    "rsi_macd_combo": {
        "name": "RSI + MACD 组合策略",
        "description": "RSI 超卖且 MACD 金叉时买入，提高信号可靠性",
        "category": "组合策略",
        "risk_level": "中低",
        "suitable_for": "各类市场",
        "template": {
            "entry": {
                "logic": "AND",
                "conditions": [
                    {"indicator": "RSI", "params": {"period": 14}, "operator": "<", "value": 35},
                    {
                        "indicator": "MACD",
                        "params": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                        "operator": "cross_above",
                        "value_indicator": "MACD_SIGNAL",
                        "value_indicator_params": {"fast_period": 12, "slow_period": 26, "signal_period": 9}
                    }
                ]
            },
            "exit": {
                "stop_loss": {"type": "percent", "value": 3.0},
                "take_profit": {"type": "percent", "value": 6.0},
                "trailing_stop": {"enabled": True, "type": "percent", "value": 2.0, "activation_percent": 3.0},
                "conditions": {
                    "logic": "OR",
                    "conditions": [
                        {"indicator": "RSI", "params": {"period": 14}, "operator": ">", "value": 70}
                    ]
                }
            }
        }
    },
    "bollinger_bounce": {
        "name": "布林带下轨反弹策略",
        "description": "价格触及布林带下轨时买入，触及上轨时卖出",
        "category": "均值回归",
        "risk_level": "中等",
        "suitable_for": "震荡市场",
        "template": {
            "entry": {
                "logic": "AND",
                "conditions": [
                    {
                        "indicator": "PRICE",
                        "params": {},
                        "operator": "<",
                        "value_indicator": "BBANDS_LOWER",
                        "value_indicator_params": {"period": 20, "std_dev": 2.0}
                    }
                ]
            },
            "exit": {
                "stop_loss": {"type": "percent", "value": 3.0},
                "take_profit": {"type": "percent", "value": 5.0},
                "conditions": {
                    "logic": "OR",
                    "conditions": [
                        {
                            "indicator": "PRICE",
                            "params": {},
                            "operator": ">",
                            "value_indicator": "BBANDS_UPPER",
                            "value_indicator_params": {"period": 20, "std_dev": 2.0}
                        }
                    ]
                }
            }
        }
    },
    "sma_crossover": {
        "name": "双均线交叉策略",
        "description": "短期均线上穿长期均线时买入",
        "category": "趋势跟踪",
        "risk_level": "低",
        "suitable_for": "趋势市场",
        "template": {
            "entry": {
                "logic": "AND",
                "conditions": [
                    {
                        "indicator": "EMA",
                        "params": {"period": 12},
                        "operator": "cross_above",
                        "value_indicator": "EMA",
                        "value_indicator_params": {"period": 26}
                    }
                ]
            },
            "exit": {
                "stop_loss": {"type": "percent", "value": 5.0},
                "take_profit": {"type": "percent", "value": 10.0},
                "conditions": {
                    "logic": "OR",
                    "conditions": [
                        {
                            "indicator": "EMA",
                            "params": {"period": 12},
                            "operator": "cross_below",
                            "value_indicator": "EMA",
                            "value_indicator_params": {"period": 26}
                        }
                    ]
                }
            }
        }
    }
}


# ============================================================================
# SKILL FUNCTIONS
# ============================================================================

def list_strategy_templates() -> str:
    """
    List available strategy templates.
    
    Returns:
        Formatted string with template descriptions
    """
    output = "## 📋 可用策略模板\n\n"
    
    for key, template in STRATEGY_TEMPLATES.items():
        output += f"### {template['name']}\n"
        output += f"- **描述**: {template['description']}\n"
        output += f"- **类别**: {template['category']}\n"
        output += f"- **风险等级**: {template['risk_level']}\n"
        output += f"- **适用市场**: {template['suitable_for']}\n"
        output += f"- **模板 ID**: `{key}`\n\n"
    
    return output


def generate_strategy_from_intent(
    intent: str,
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    position_size: float = 100.0,
    initial_capital: float = 10000.0
) -> dict:
    """
    Generate a strategy configuration from natural language intent.
    
    This function is designed to be called by an AI agent. The agent should:
    1. Call this function to generate a strategy
    2. Present the strategy to the user for review
    3. Allow the user to modify the strategy
    4. Proceed with backtesting only after user confirmation
    
    Args:
        intent: Natural language description of the strategy
        symbol: Trading pair (e.g., "BTC/USDT")
        timeframe: Candle timeframe (e.g., "4h", "1d")
        position_size: Amount to trade per position (in quote currency)
        initial_capital: Starting capital for backtest
        
    Returns:
        Dictionary containing:
        - strategy_config: The generated strategy configuration (JSON)
        - strategy_display: Human-readable strategy summary
        - needs_confirmation: Always True - user must confirm before proceeding
    """
    
    # Analyze intent and match to template
    intent_lower = intent.lower()
    
    # Match keywords to templates
    matched_template = None
    
    if "rsi" in intent_lower and "macd" in intent_lower:
        matched_template = "rsi_macd_combo"
    elif "rsi" in intent_lower or "超卖" in intent_lower or "超买" in intent_lower or "低估" in intent_lower or "高估" in intent_lower:
        matched_template = "rsi_oversold"
    elif "macd" in intent_lower or "金叉" in intent_lower or "死叉" in intent_lower:
        matched_template = "macd_crossover"
    elif "布林" in intent_lower or "bollinger" in intent_lower:
        matched_template = "bollinger_bounce"
    elif "均线" in intent_lower or "sma" in intent_lower or "ema" in intent_lower or "交叉" in intent_lower:
        matched_template = "sma_crossover"
    else:
        # Default to RSI + MACD combo as a balanced choice
        matched_template = "rsi_macd_combo"
    
    template = STRATEGY_TEMPLATES[matched_template]
    
    # Build strategy config
    strategy_dict = {
        "name": f"{template['name']} ({symbol})",
        "description": f"基于用户意图: {intent}\n\n{template['description']}",
        "symbol": symbol,
        "timeframe": timeframe,
        **template["template"],
        "position_sizing": {
            "type": "fixed_amount",
            "value": position_size,
            "max_positions": 1
        },
        "risk_management": {
            "max_drawdown_percent": 15.0,
            "daily_loss_limit_percent": 5.0,
            "position_limit_percent": 20.0
        },
        "initial_capital": initial_capital,
        "commission_percent": 0.1,
        "slippage_percent": 0.05
    }
    
    # Validate and create config
    strategy = StrategyConfig(**strategy_dict)
    
    # Generate human-readable display
    display = strategy.to_display_dict()
    
    display_text = f"""
## 📊 策略配置预览

**请仔细检查以下策略配置，确认或修改后继续：**

| 配置项 | 值 |
|--------|-----|
"""
    for key, value in display.items():
        display_text += f"| {key} | {value} |\n"
    
    display_text += f"""
### 策略 JSON（可编辑）

```json
{json.dumps(strategy_dict, indent=2, ensure_ascii=False)}
```

---

**下一步操作：**
1. ✅ 如果配置正确，请说"确认"或"开始回测"
2. ✏️ 如果需要修改，请直接告诉我要改什么（例如："把止损改成 5%"）
3. 📋 如果想用其他模板，请说"换成 XXX 策略"
"""
    
    return {
        "strategy_config": strategy_dict,
        "strategy_display": display_text,
        "matched_template": matched_template,
        "needs_confirmation": True
    }


def run_backtest(
    strategy_config: dict,
    days: int = 365,
    exchange: str = "binance"
) -> dict:
    """
    Run backtest on a strategy configuration.
    
    Args:
        strategy_config: Validated strategy configuration dict
        days: Number of days of historical data to use
        exchange: Exchange to fetch data from
        
    Returns:
        Dictionary containing:
        - result: BacktestResult object
        - summary: Summary statistics
        - markdown_report: Markdown formatted report
    """
    
    # Create strategy config
    strategy = StrategyConfig(**strategy_config)
    
    # Fetch data
    print(f"Fetching {strategy.symbol} data...")
    df = fetch_crypto_data(
        symbol=strategy.symbol,
        timeframe=strategy.timeframe.value,
        days=days,
        exchange=exchange
    )
    
    if df.empty:
        return {
            "error": "无法获取市场数据，请检查交易对和交易所设置",
            "success": False
        }
    
    print(f"Running backtest with {len(df)} candles...")
    
    # Run backtest
    engine = BacktestEngine(strategy, df)
    result = engine.run()
    
    # Generate markdown report
    md_report = generate_markdown_report(result)
    
    return {
        "result": result,
        "summary": result.to_summary_dict(),
        "markdown_report": md_report,
        "success": True
    }


def generate_reports(
    result: BacktestResult,
    output_dir: str = "./output"
) -> dict:
    """
    Generate visual reports from backtest result.
    
    Args:
        result: BacktestResult from run_backtest
        output_dir: Directory to save report files
        
    Returns:
        Dictionary with file paths
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    generator = ReportGenerator(result)
    report_files = generator.generate_full_report(output_dir)
    
    return report_files


def generate_code(
    strategy_config: dict,
    output_dir: str = "./output"
) -> dict:
    """
    Generate runnable code from strategy configuration.
    
    Args:
        strategy_config: Strategy configuration dict
        output_dir: Directory to save code files
        
    Returns:
        Dictionary with file paths and code samples
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    strategy = StrategyConfig(**strategy_config)
    code_files = save_strategy_code(strategy, output_dir)
    
    # Also return the code as strings
    generator = CodeGenerator(strategy)
    
    return {
        "files": code_files,
        "standalone_code": generator.generate("standalone"),
        "freqtrade_code": generator.generate("freqtrade"),
        "bot_code": generator.generate("ccxt_bot")
    }


def full_workflow(
    intent: str,
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    position_size: float = 100.0,
    initial_capital: float = 10000.0,
    backtest_days: int = 365,
    output_dir: str = "./output"
) -> dict:
    """
    Execute the full workflow from intent to reports and code.
    
    ⚠️ NOTE: This function should only be called AFTER user confirms the strategy.
    Use generate_strategy_from_intent() first, present to user, get confirmation,
    then call this function.
    
    Args:
        intent: Natural language strategy description
        symbol: Trading pair
        timeframe: Candle timeframe
        position_size: Position size in quote currency
        initial_capital: Starting capital
        backtest_days: Days of historical data
        output_dir: Output directory
        
    Returns:
        Complete workflow results
    """
    
    print("=" * 60)
    print(f"  Crypto Strategy Backtest Skill")
    print(f"  策略: {intent[:50]}...")
    print("=" * 60)
    
    # Step 1: Generate strategy
    print("\n[1/4] Generating strategy configuration...")
    strategy_result = generate_strategy_from_intent(
        intent, symbol, timeframe, position_size, initial_capital
    )
    strategy_config = strategy_result["strategy_config"]
    
    # Step 2: Run backtest
    print("\n[2/4] Running backtest...")
    backtest_result = run_backtest(strategy_config, backtest_days)
    
    if not backtest_result.get("success"):
        return {
            "error": backtest_result.get("error", "Backtest failed"),
            "success": False
        }
    
    # Step 3: Generate reports
    print("\n[3/4] Generating visual reports...")
    report_files = generate_reports(backtest_result["result"], output_dir)
    
    # Step 4: Generate code
    print("\n[4/4] Generating runnable code...")
    code_files = generate_code(strategy_config, output_dir)
    
    print("\n" + "=" * 60)
    print("  ✅ Workflow completed!")
    print("=" * 60)
    
    return {
        "strategy_config": strategy_config,
        "strategy_display": strategy_result["strategy_display"],
        "backtest_summary": backtest_result["summary"],
        "markdown_report": backtest_result["markdown_report"],
        "report_files": report_files,
        "code_files": code_files["files"],
        "success": True
    }


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface for the skill"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Crypto Strategy Backtest Skill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available templates
  python -m src.skill --list-templates
  
  # Generate strategy from intent
  python -m src.skill --intent "BTC 超卖时买入" --symbol BTC/USDT
  
  # Run full workflow
  python -m src.skill --intent "RSI 低于 30 时买入" --run-backtest
"""
    )
    
    parser.add_argument("--list-templates", action="store_true", help="List available strategy templates")
    parser.add_argument("--intent", type=str, help="Natural language strategy description")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading pair")
    parser.add_argument("--timeframe", type=str, default="4h", help="Candle timeframe")
    parser.add_argument("--position-size", type=float, default=100.0, help="Position size in USDT")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital")
    parser.add_argument("--days", type=int, default=365, help="Backtest days")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--run-backtest", action="store_true", help="Run backtest (requires confirmation)")
    
    args = parser.parse_args()
    
    if args.list_templates:
        print(list_strategy_templates())
        return
    
    if args.intent:
        result = generate_strategy_from_intent(
            args.intent,
            args.symbol,
            args.timeframe,
            args.position_size,
            args.capital
        )
        
        print(result["strategy_display"])
        
        if args.run_backtest:
            confirm = input("\n确认执行回测? (y/n): ")
            if confirm.lower() == 'y':
                workflow_result = full_workflow(
                    args.intent,
                    args.symbol,
                    args.timeframe,
                    args.position_size,
                    args.capital,
                    args.days,
                    args.output
                )
                
                if workflow_result.get("success"):
                    print("\n" + workflow_result["markdown_report"])
                    print(f"\n📁 Reports saved to: {args.output}")
                else:
                    print(f"\n❌ Error: {workflow_result.get('error')}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
