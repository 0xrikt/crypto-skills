"""
Crypto Strategy Backtest Skill

A skill that enables AI agents to:
1. Parse natural language trading strategy descriptions
2. Generate structured strategy configurations
3. Execute backtests on historical data
4. Generate visual reports
5. Output runnable trading code

Usage:
    from src.skill import (
        generate_strategy_from_intent,
        run_backtest,
        generate_reports,
        generate_code
    )
"""

from .strategy_schema import StrategyConfig
from .backtest_engine import BacktestEngine, BacktestResult
from .report_generator import ReportGenerator, generate_markdown_report
from .code_generator import CodeGenerator, save_strategy_code
from .data_fetcher import DataFetcher, fetch_crypto_data
from .indicators import TechnicalIndicators
from .skill import (
    generate_strategy_from_intent,
    run_backtest,
    generate_reports,
    generate_code,
    full_workflow,
    list_strategy_templates,
    STRATEGY_TEMPLATES
)

__all__ = [
    # Core classes
    "StrategyConfig",
    "BacktestEngine",
    "BacktestResult",
    "ReportGenerator",
    "CodeGenerator",
    "DataFetcher",
    "TechnicalIndicators",
    
    # Skill functions (main entry points)
    "generate_strategy_from_intent",
    "run_backtest",
    "generate_reports",
    "generate_code",
    "full_workflow",
    "list_strategy_templates",
    
    # Utilities
    "generate_markdown_report",
    "save_strategy_code",
    "fetch_crypto_data",
    "STRATEGY_TEMPLATES",
]

__version__ = "0.1.0"
