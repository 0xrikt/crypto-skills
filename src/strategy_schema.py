"""
Strategy Schema Definition

Defines the structured format for trading strategies.
This is the "contract" between natural language understanding and the backtest engine.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field
from enum import Enum


class Timeframe(str, Enum):
    """Supported trading timeframes"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    H6 = "6h"
    H12 = "12h"
    D1 = "1d"
    W1 = "1w"


class IndicatorType(str, Enum):
    """Supported technical indicators"""
    RSI = "RSI"
    MACD = "MACD"
    MACD_SIGNAL = "MACD_SIGNAL"
    MACD_HIST = "MACD_HIST"
    SMA = "SMA"
    EMA = "EMA"
    BBANDS_UPPER = "BBANDS_UPPER"
    BBANDS_MIDDLE = "BBANDS_MIDDLE"
    BBANDS_LOWER = "BBANDS_LOWER"
    ATR = "ATR"
    VOLUME = "VOLUME"
    PRICE = "PRICE"


class ComparisonOperator(str, Enum):
    """Comparison operators for conditions"""
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "=="
    CROSS_ABOVE = "cross_above"
    CROSS_BELOW = "cross_below"


class LogicOperator(str, Enum):
    """Logic operators for combining conditions"""
    AND = "AND"
    OR = "OR"


class IndicatorParams(BaseModel):
    """Parameters for technical indicators"""
    period: Optional[int] = Field(None, description="Period for RSI, SMA, EMA, ATR")
    fast_period: Optional[int] = Field(None, description="Fast period for MACD")
    slow_period: Optional[int] = Field(None, description="Slow period for MACD")
    signal_period: Optional[int] = Field(None, description="Signal period for MACD")
    std_dev: Optional[float] = Field(None, description="Standard deviation for Bollinger Bands")


class Condition(BaseModel):
    """A single condition in the strategy"""
    indicator: IndicatorType
    params: IndicatorParams = Field(default_factory=IndicatorParams)
    operator: ComparisonOperator
    # Value can be a number or reference to another indicator
    value: Optional[float] = Field(None, description="Numeric threshold")
    value_indicator: Optional[IndicatorType] = Field(None, description="Compare to another indicator")
    value_indicator_params: Optional[IndicatorParams] = Field(None, description="Params for value indicator")


class ConditionGroup(BaseModel):
    """A group of conditions combined with logic operator"""
    logic: LogicOperator = LogicOperator.AND
    conditions: list[Condition]


class StopLoss(BaseModel):
    """Stop loss configuration"""
    type: Literal["percent", "atr", "fixed"] = "percent"
    value: float = Field(..., description="Stop loss value (percent, ATR multiplier, or fixed price)")


class TakeProfit(BaseModel):
    """Take profit configuration"""
    type: Literal["percent", "atr", "fixed", "risk_reward"] = "percent"
    value: float = Field(..., description="Take profit value")


class TrailingStop(BaseModel):
    """Trailing stop configuration"""
    enabled: bool = False
    type: Literal["percent", "atr"] = "percent"
    value: float = Field(1.0, description="Trailing stop distance")
    activation_percent: Optional[float] = Field(None, description="Activate after X% profit")


class PositionSizing(BaseModel):
    """Position sizing configuration"""
    type: Literal["fixed_amount", "percent_equity", "kelly"] = "fixed_amount"
    value: float = Field(..., description="Amount in quote currency or percentage")
    max_positions: int = Field(1, description="Maximum concurrent positions")


class RiskManagement(BaseModel):
    """Risk management rules"""
    max_drawdown_percent: Optional[float] = Field(None, description="Stop trading if drawdown exceeds")
    daily_loss_limit_percent: Optional[float] = Field(None, description="Daily loss limit")
    position_limit_percent: Optional[float] = Field(20.0, description="Max position size as % of equity")


class ExitConfig(BaseModel):
    """Exit strategy configuration"""
    stop_loss: Optional[StopLoss] = None
    take_profit: Optional[TakeProfit] = None
    trailing_stop: Optional[TrailingStop] = None
    conditions: Optional[ConditionGroup] = Field(None, description="Signal-based exit conditions")


class StrategyConfig(BaseModel):
    """
    Complete trading strategy configuration.
    
    This is the core schema that bridges:
    - Natural language understanding (input)
    - Backtest engine (execution)
    - Code generation (output)
    """
    
    # Basic info
    name: str = Field(..., description="Strategy name")
    description: str = Field(..., description="Human-readable description")
    
    # Market settings
    symbol: str = Field("BTC/USDT", description="Trading pair")
    timeframe: Timeframe = Field(Timeframe.H4, description="Candle timeframe")
    
    # Entry conditions
    entry: ConditionGroup = Field(..., description="Entry signal conditions")
    
    # Exit configuration
    exit: ExitConfig = Field(default_factory=ExitConfig, description="Exit strategy")
    
    # Position sizing
    position_sizing: PositionSizing = Field(..., description="How much to trade")
    
    # Risk management
    risk_management: RiskManagement = Field(
        default_factory=RiskManagement,
        description="Risk management rules"
    )
    
    # Backtest settings
    initial_capital: float = Field(10000.0, description="Starting capital in quote currency")
    commission_percent: float = Field(0.1, description="Commission per trade in percent")
    slippage_percent: float = Field(0.05, description="Estimated slippage in percent")

    def to_display_dict(self) -> dict:
        """Convert to a human-readable dictionary for display"""
        return {
            "策略名称": self.name,
            "策略描述": self.description,
            "交易对": self.symbol,
            "时间周期": self.timeframe.value,
            "入场条件": self._format_conditions(self.entry),
            "止损": self._format_stop_loss(),
            "止盈": self._format_take_profit(),
            "移动止损": self._format_trailing_stop(),
            "出场信号": self._format_conditions(self.exit.conditions) if self.exit.conditions else "无",
            "仓位管理": self._format_position_sizing(),
            "风险管理": self._format_risk_management(),
            "初始资金": f"{self.initial_capital} USDT",
            "手续费": f"{self.commission_percent}%",
            "滑点": f"{self.slippage_percent}%",
        }
    
    def _format_conditions(self, cg: Optional[ConditionGroup]) -> str:
        if not cg:
            return "无"
        parts = []
        for c in cg.conditions:
            indicator_str = self._format_indicator(c.indicator, c.params)
            if c.value is not None:
                parts.append(f"{indicator_str} {c.operator.value} {c.value}")
            elif c.value_indicator:
                value_str = self._format_indicator(c.value_indicator, c.value_indicator_params)
                parts.append(f"{indicator_str} {c.operator.value} {value_str}")
        return f" {cg.logic.value} ".join(parts)
    
    def _format_indicator(self, ind: IndicatorType, params: Optional[IndicatorParams]) -> str:
        if not params:
            return ind.value
        if ind in [IndicatorType.RSI, IndicatorType.SMA, IndicatorType.EMA, IndicatorType.ATR]:
            return f"{ind.value}({params.period or 14})"
        if ind in [IndicatorType.MACD, IndicatorType.MACD_SIGNAL, IndicatorType.MACD_HIST]:
            return f"{ind.value}({params.fast_period or 12},{params.slow_period or 26},{params.signal_period or 9})"
        if ind in [IndicatorType.BBANDS_UPPER, IndicatorType.BBANDS_MIDDLE, IndicatorType.BBANDS_LOWER]:
            return f"{ind.value}({params.period or 20},{params.std_dev or 2.0})"
        return ind.value
    
    def _format_stop_loss(self) -> str:
        if not self.exit.stop_loss:
            return "无"
        sl = self.exit.stop_loss
        if sl.type == "percent":
            return f"{sl.value}%"
        elif sl.type == "atr":
            return f"{sl.value}x ATR"
        return f"固定 {sl.value}"
    
    def _format_take_profit(self) -> str:
        if not self.exit.take_profit:
            return "无"
        tp = self.exit.take_profit
        if tp.type == "percent":
            return f"{tp.value}%"
        elif tp.type == "risk_reward":
            return f"风险回报比 1:{tp.value}"
        elif tp.type == "atr":
            return f"{tp.value}x ATR"
        return f"固定 {tp.value}"
    
    def _format_trailing_stop(self) -> str:
        if not self.exit.trailing_stop or not self.exit.trailing_stop.enabled:
            return "无"
        ts = self.exit.trailing_stop
        activation = f", {ts.activation_percent}% 后激活" if ts.activation_percent else ""
        if ts.type == "percent":
            return f"{ts.value}%{activation}"
        return f"{ts.value}x ATR{activation}"
    
    def _format_position_sizing(self) -> str:
        ps = self.position_sizing
        if ps.type == "fixed_amount":
            return f"固定金额 {ps.value} USDT, 最多 {ps.max_positions} 个持仓"
        elif ps.type == "percent_equity":
            return f"资金 {ps.value}%, 最多 {ps.max_positions} 个持仓"
        return f"Kelly 公式, 最多 {ps.max_positions} 个持仓"
    
    def _format_risk_management(self) -> str:
        rm = self.risk_management
        parts = []
        if rm.max_drawdown_percent:
            parts.append(f"最大回撤限制 {rm.max_drawdown_percent}%")
        if rm.daily_loss_limit_percent:
            parts.append(f"日亏损限制 {rm.daily_loss_limit_percent}%")
        if rm.position_limit_percent:
            parts.append(f"单仓位上限 {rm.position_limit_percent}%")
        return ", ".join(parts) if parts else "无"


# Example usage and validation
EXAMPLE_STRATEGY = {
    "name": "RSI 超卖反弹 + MACD 确认",
    "description": "当 RSI 低于 30 且 MACD 金叉时买入，RSI 高于 70 时卖出",
    "symbol": "BTC/USDT",
    "timeframe": "4h",
    "entry": {
        "logic": "AND",
        "conditions": [
            {
                "indicator": "RSI",
                "params": {"period": 14},
                "operator": "<",
                "value": 30
            },
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
                {
                    "indicator": "RSI",
                    "params": {"period": 14},
                    "operator": ">",
                    "value": 70
                }
            ]
        }
    },
    "position_sizing": {
        "type": "fixed_amount",
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


if __name__ == "__main__":
    # Validate example strategy
    strategy = StrategyConfig(**EXAMPLE_STRATEGY)
    print("Strategy validated successfully!")
    print("\n=== 策略配置 ===")
    for key, value in strategy.to_display_dict().items():
        print(f"{key}: {value}")
