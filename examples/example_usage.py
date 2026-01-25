#!/usr/bin/env python3
"""
Example usage of the Crypto Strategy Backtest Skill

This script demonstrates the recommended workflow:
1. Generate strategy from natural language intent
2. Present to user for confirmation/modification
3. Run backtest
4. Generate reports and code
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.skill import (
    list_strategy_templates,
    generate_strategy_from_intent,
    run_backtest,
    generate_reports,
    generate_code,
    full_workflow
)


def example_1_basic_workflow():
    """
    Basic workflow: intent → strategy → backtest → reports
    """
    print("\n" + "=" * 70)
    print("  Example 1: Basic Workflow")
    print("=" * 70)
    
    # Step 1: User describes their intent
    user_intent = "BTC 被低估的时候就买入 100 USDT，被高估的时候就全部卖出，每四个小时检查一次"
    
    print(f"\n📝 用户意图: {user_intent}\n")
    
    # Step 2: Generate strategy configuration
    result = generate_strategy_from_intent(
        intent=user_intent,
        symbol="BTC/USDT",
        timeframe="4h",
        position_size=100.0,
        initial_capital=10000.0
    )
    
    # Step 3: Present strategy to user for confirmation
    print(result["strategy_display"])
    
    # In a real application, wait for user confirmation here
    # For this example, we'll just proceed
    
    return result["strategy_config"]


def example_2_full_workflow():
    """
    Full workflow with all outputs
    """
    print("\n" + "=" * 70)
    print("  Example 2: Full Workflow with Reports and Code")
    print("=" * 70)
    
    result = full_workflow(
        intent="当 RSI 低于 30 且 MACD 金叉时买入，RSI 高于 70 时卖出",
        symbol="BTC/USDT",
        timeframe="4h",
        position_size=100.0,
        initial_capital=10000.0,
        backtest_days=365,
        output_dir="./output"
    )
    
    if result.get("success"):
        print("\n✅ Workflow completed successfully!")
        print("\n📊 Backtest Summary:")
        for key, value in result["backtest_summary"].items():
            print(f"  {key}: {value}")
        
        print("\n📁 Generated Files:")
        for name, path in result["report_files"]["files"].items():
            print(f"  - {name}: {path}")
        for name, path in result["code_files"].items():
            print(f"  - {name}: {path}")
    else:
        print(f"\n❌ Error: {result.get('error')}")
    
    return result


def example_3_list_templates():
    """
    List available strategy templates
    """
    print("\n" + "=" * 70)
    print("  Example 3: Available Strategy Templates")
    print("=" * 70)
    
    templates = list_strategy_templates()
    print(templates)


def example_4_custom_strategy():
    """
    Create a custom strategy with specific parameters
    """
    print("\n" + "=" * 70)
    print("  Example 4: Custom Strategy Configuration")
    print("=" * 70)
    
    # Define a custom strategy
    custom_strategy = {
        "name": "自定义 RSI 策略",
        "description": "RSI < 25 买入，RSI > 75 卖出，更激进的参数",
        "symbol": "ETH/USDT",
        "timeframe": "1h",
        "entry": {
            "logic": "AND",
            "conditions": [
                {"indicator": "RSI", "params": {"period": 14}, "operator": "<", "value": 25}
            ]
        },
        "exit": {
            "stop_loss": {"type": "percent", "value": 2.5},
            "take_profit": {"type": "percent", "value": 5.0},
            "trailing_stop": {"enabled": True, "type": "percent", "value": 1.5, "activation_percent": 2.0},
            "conditions": {
                "logic": "OR",
                "conditions": [
                    {"indicator": "RSI", "params": {"period": 14}, "operator": ">", "value": 75}
                ]
            }
        },
        "position_sizing": {
            "type": "fixed_amount",
            "value": 50.0,
            "max_positions": 2
        },
        "risk_management": {
            "max_drawdown_percent": 10.0,
            "daily_loss_limit_percent": 3.0,
            "position_limit_percent": 15.0
        },
        "initial_capital": 5000.0,
        "commission_percent": 0.1,
        "slippage_percent": 0.05
    }
    
    print(f"\n📝 Custom Strategy: {custom_strategy['name']}")
    print(f"   Symbol: {custom_strategy['symbol']}")
    print(f"   Timeframe: {custom_strategy['timeframe']}")
    
    # Run backtest with custom strategy
    print("\n🔄 Running backtest...")
    backtest_result = run_backtest(custom_strategy, days=180)
    
    if backtest_result.get("success"):
        print("\n📊 Backtest Results:")
        for key, value in backtest_result["summary"].items():
            print(f"  {key}: {value}")
    else:
        print(f"\n❌ Error: {backtest_result.get('error')}")
    
    return custom_strategy


def example_5_code_generation():
    """
    Generate trading code from a strategy
    """
    print("\n" + "=" * 70)
    print("  Example 5: Code Generation")
    print("=" * 70)
    
    from src.strategy_schema import StrategyConfig, EXAMPLE_STRATEGY
    from src.code_generator import CodeGenerator
    
    # Create strategy
    strategy = StrategyConfig(**EXAMPLE_STRATEGY)
    generator = CodeGenerator(strategy)
    
    # Generate different code formats
    print("\n📄 Generating standalone backtest script...")
    standalone = generator.generate("standalone")
    print(f"   Generated {len(standalone)} characters")
    
    print("\n📄 Generating Freqtrade strategy...")
    freqtrade = generator.generate("freqtrade")
    print(f"   Generated {len(freqtrade)} characters")
    
    print("\n📄 Generating live trading bot...")
    bot = generator.generate("ccxt_bot")
    print(f"   Generated {len(bot)} characters")
    
    # Save to files
    os.makedirs("./output", exist_ok=True)
    
    with open("./output/standalone_example.py", "w") as f:
        f.write(standalone)
    print("\n✅ Saved: ./output/standalone_example.py")
    
    with open("./output/freqtrade_example.py", "w") as f:
        f.write(freqtrade)
    print("✅ Saved: ./output/freqtrade_example.py")
    
    with open("./output/bot_example.py", "w") as f:
        f.write(bot)
    print("✅ Saved: ./output/bot_example.py")


def main():
    """Run all examples"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Example usage of the Crypto Strategy Backtest Skill")
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run a specific example (1-5)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all examples"
    )
    
    args = parser.parse_args()
    
    if args.all or args.example is None:
        # Run all examples
        example_3_list_templates()
        example_1_basic_workflow()
        example_5_code_generation()
        
        # These require network access:
        # example_2_full_workflow()
        # example_4_custom_strategy()
        
        print("\n" + "=" * 70)
        print("  ⚠️  Examples 2 and 4 require network access to fetch market data.")
        print("      Run them separately with --example 2 or --example 4")
        print("=" * 70)
        
    elif args.example == 1:
        example_1_basic_workflow()
    elif args.example == 2:
        example_2_full_workflow()
    elif args.example == 3:
        example_3_list_templates()
    elif args.example == 4:
        example_4_custom_strategy()
    elif args.example == 5:
        example_5_code_generation()


if __name__ == "__main__":
    main()
