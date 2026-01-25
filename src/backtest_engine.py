"""
Backtest Engine

Executes trading strategies on historical data and calculates performance metrics.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from .strategy_schema import (
    StrategyConfig, 
    ConditionGroup, 
    Condition,
    IndicatorType,
    ComparisonOperator
)
from .indicators import TechnicalIndicators, detect_crossover, detect_crossunder


@dataclass
class Trade:
    """Represents a single trade"""
    entry_time: datetime
    entry_price: float
    entry_reason: str
    position_size: float  # In quote currency
    quantity: float  # In base currency
    
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    
    pnl: float = 0.0
    pnl_percent: float = 0.0
    commission: float = 0.0
    
    def close(self, exit_time: datetime, exit_price: float, exit_reason: str, commission_pct: float):
        """Close the trade and calculate PnL"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        
        # Calculate PnL
        gross_pnl = (exit_price - self.entry_price) * self.quantity
        self.commission = self.position_size * commission_pct / 100 * 2  # Entry + Exit
        self.pnl = gross_pnl - self.commission
        self.pnl_percent = (self.pnl / self.position_size) * 100
    
    @property
    def is_open(self) -> bool:
        return self.exit_time is None
    
    @property
    def duration(self) -> Optional[pd.Timedelta]:
        if self.exit_time:
            return self.exit_time - self.entry_time
        return None


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    
    # Capital tracking
    initial_capital: float
    final_capital: float
    
    # Trade list
    trades: list[Trade] = field(default_factory=list)
    
    # Equity curve
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Performance metrics (calculated)
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration: Optional[pd.Timedelta] = None
    
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    avg_trade_duration: Optional[pd.Timedelta] = None
    
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    def calculate_metrics(self):
        """Calculate all performance metrics"""
        if not self.trades:
            return
        
        closed_trades = [t for t in self.trades if not t.is_open]
        if not closed_trades:
            return
        
        self.total_trades = len(closed_trades)
        
        # Win/Loss stats
        wins = [t for t in closed_trades if t.pnl > 0]
        losses = [t for t in closed_trades if t.pnl <= 0]
        
        self.winning_trades = len(wins)
        self.losing_trades = len(losses)
        self.win_rate = len(wins) / len(closed_trades) * 100 if closed_trades else 0
        
        # Average win/loss
        if wins:
            self.avg_win_pct = np.mean([t.pnl_percent for t in wins])
        if losses:
            self.avg_loss_pct = np.mean([t.pnl_percent for t in losses])
        
        # Profit factor
        total_wins = sum(t.pnl for t in wins) if wins else 0
        total_losses = abs(sum(t.pnl for t in losses)) if losses else 0
        self.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Average trade duration
        durations = [t.duration for t in closed_trades if t.duration]
        if durations:
            self.avg_trade_duration = pd.Timedelta(np.mean([d.total_seconds() for d in durations]), unit='s')
        
        # Return calculations
        self.total_return_pct = (self.final_capital - self.initial_capital) / self.initial_capital * 100
        
        # Annualized return
        if not self.equity_curve.empty:
            days = (self.end_date - self.start_date).days
            if days > 0:
                self.annualized_return_pct = ((1 + self.total_return_pct / 100) ** (365 / days) - 1) * 100
        
        # Sharpe and Sortino ratios
        if not self.equity_curve.empty and len(self.equity_curve) > 1:
            returns = self.equity_curve["equity"].pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                # Annualized Sharpe (assuming 4h candles, ~2190 periods per year)
                periods_per_year = 365 * 24 / 4  # Approximate
                self.sharpe_ratio = returns.mean() / returns.std() * np.sqrt(periods_per_year)
                
                # Sortino (only downside deviation)
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0 and downside_returns.std() > 0:
                    self.sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(periods_per_year)
        
        # Max drawdown
        if not self.equity_curve.empty:
            equity = self.equity_curve["equity"]
            rolling_max = equity.expanding().max()
            drawdown = (equity - rolling_max) / rolling_max * 100
            self.max_drawdown_pct = abs(drawdown.min())
            
            # Drawdown duration
            in_drawdown = drawdown < 0
            if in_drawdown.any():
                drawdown_groups = (~in_drawdown).cumsum()
                drawdown_lengths = in_drawdown.groupby(drawdown_groups).sum()
                if len(drawdown_lengths) > 0:
                    max_length = drawdown_lengths.max()
                    # Convert to timedelta based on timeframe
                    self.max_drawdown_duration = pd.Timedelta(hours=max_length * 4)  # Assuming 4h
    
    def to_summary_dict(self) -> dict:
        """Convert to summary dictionary"""
        return {
            "策略名称": self.strategy_name,
            "交易对": self.symbol,
            "时间周期": self.timeframe,
            "回测期间": f"{self.start_date.date()} 至 {self.end_date.date()}",
            "初始资金": f"${self.initial_capital:,.2f}",
            "最终资金": f"${self.final_capital:,.2f}",
            "总收益率": f"{self.total_return_pct:+.2f}%",
            "年化收益率": f"{self.annualized_return_pct:+.2f}%",
            "夏普比率": f"{self.sharpe_ratio:.2f}",
            "索提诺比率": f"{self.sortino_ratio:.2f}",
            "最大回撤": f"{self.max_drawdown_pct:.2f}%",
            "总交易次数": self.total_trades,
            "胜率": f"{self.win_rate:.1f}%",
            "盈利因子": f"{self.profit_factor:.2f}" if self.profit_factor != float('inf') else "∞",
            "平均盈利": f"{self.avg_win_pct:+.2f}%",
            "平均亏损": f"{self.avg_loss_pct:+.2f}%",
            "平均持仓时间": str(self.avg_trade_duration) if self.avg_trade_duration else "N/A",
        }


class BacktestEngine:
    """
    Execute backtests on trading strategies.
    
    This is the core engine that simulates strategy execution on historical data.
    """
    
    def __init__(self, strategy: StrategyConfig, data: pd.DataFrame):
        """
        Initialize backtest engine.
        
        Args:
            strategy: Strategy configuration
            data: OHLCV DataFrame with columns: timestamp, open, high, low, close, volume
        """
        self.strategy = strategy
        self.data = data.copy()
        
        # Add technical indicators
        self._prepare_data()
        
        # State
        self.capital = strategy.initial_capital
        self.position: Optional[Trade] = None
        self.trades: list[Trade] = []
        self.equity_history: list[dict] = []
    
    def _prepare_data(self):
        """Add all required technical indicators to the data"""
        indicators = TechnicalIndicators(self.data)
        
        # Collect all required indicators from strategy
        required_indicators = self._collect_required_indicators()
        
        # Add indicators based on what's needed
        for ind_type, params in required_indicators:
            self._add_indicator(indicators, ind_type, params)
    
    def _collect_required_indicators(self) -> list[tuple]:
        """Collect all indicators needed by the strategy"""
        indicators = []
        
        # From entry conditions
        for cond in self.strategy.entry.conditions:
            indicators.append((cond.indicator, cond.params))
            if cond.value_indicator:
                indicators.append((cond.value_indicator, cond.value_indicator_params))
        
        # From exit conditions
        if self.strategy.exit.conditions:
            for cond in self.strategy.exit.conditions.conditions:
                indicators.append((cond.indicator, cond.params))
                if cond.value_indicator:
                    indicators.append((cond.value_indicator, cond.value_indicator_params))
        
        return indicators
    
    def _add_indicator(self, indicators: TechnicalIndicators, ind_type: IndicatorType, params):
        """Add a specific indicator to the data"""
        if params is None:
            params = type('obj', (object,), {'period': 14, 'fast_period': 12, 'slow_period': 26, 'signal_period': 9, 'std_dev': 2.0})()
        
        col_name = self._get_indicator_column_name(ind_type, params)
        
        if col_name in self.data.columns:
            return  # Already added
        
        if ind_type == IndicatorType.RSI:
            self.data[col_name] = indicators.rsi(params.period or 14)
        
        elif ind_type in [IndicatorType.MACD, IndicatorType.MACD_SIGNAL, IndicatorType.MACD_HIST]:
            fast = params.fast_period or 12
            slow = params.slow_period or 26
            signal = params.signal_period or 9
            macd, macd_signal, macd_hist = indicators.macd(fast, slow, signal)
            self.data[f"MACD_{fast}_{slow}_{signal}"] = macd
            self.data[f"MACD_SIGNAL_{fast}_{slow}_{signal}"] = macd_signal
            self.data[f"MACD_HIST_{fast}_{slow}_{signal}"] = macd_hist
        
        elif ind_type == IndicatorType.SMA:
            self.data[col_name] = indicators.sma(params.period or 20)
        
        elif ind_type == IndicatorType.EMA:
            self.data[col_name] = indicators.ema(params.period or 20)
        
        elif ind_type in [IndicatorType.BBANDS_UPPER, IndicatorType.BBANDS_MIDDLE, IndicatorType.BBANDS_LOWER]:
            period = params.period or 20
            std = params.std_dev or 2.0
            upper, middle, lower = indicators.bollinger_bands(period, std)
            self.data[f"BB_UPPER_{period}"] = upper
            self.data[f"BB_MIDDLE_{period}"] = middle
            self.data[f"BB_LOWER_{period}"] = lower
        
        elif ind_type == IndicatorType.ATR:
            self.data[col_name] = indicators.atr(params.period or 14)
    
    def _get_indicator_column_name(self, ind_type: IndicatorType, params) -> str:
        """Get the column name for an indicator"""
        if params is None:
            params = type('obj', (object,), {'period': 14, 'fast_period': 12, 'slow_period': 26, 'signal_period': 9, 'std_dev': 2.0})()
        
        if ind_type == IndicatorType.RSI:
            return f"RSI_{params.period or 14}"
        elif ind_type == IndicatorType.MACD:
            return f"MACD_{params.fast_period or 12}_{params.slow_period or 26}_{params.signal_period or 9}"
        elif ind_type == IndicatorType.MACD_SIGNAL:
            return f"MACD_SIGNAL_{params.fast_period or 12}_{params.slow_period or 26}_{params.signal_period or 9}"
        elif ind_type == IndicatorType.MACD_HIST:
            return f"MACD_HIST_{params.fast_period or 12}_{params.slow_period or 26}_{params.signal_period or 9}"
        elif ind_type == IndicatorType.SMA:
            return f"SMA_{params.period or 20}"
        elif ind_type == IndicatorType.EMA:
            return f"EMA_{params.period or 20}"
        elif ind_type == IndicatorType.BBANDS_UPPER:
            return f"BB_UPPER_{params.period or 20}"
        elif ind_type == IndicatorType.BBANDS_MIDDLE:
            return f"BB_MIDDLE_{params.period or 20}"
        elif ind_type == IndicatorType.BBANDS_LOWER:
            return f"BB_LOWER_{params.period or 20}"
        elif ind_type == IndicatorType.ATR:
            return f"ATR_{params.period or 14}"
        elif ind_type == IndicatorType.PRICE:
            return "close"
        elif ind_type == IndicatorType.VOLUME:
            return "volume"
        return ind_type.value
    
    def _evaluate_condition(self, condition: Condition, row: pd.Series, prev_row: Optional[pd.Series]) -> bool:
        """Evaluate a single condition"""
        col_name = self._get_indicator_column_name(condition.indicator, condition.params)
        
        if col_name not in row.index:
            return False
        
        current_value = row[col_name]
        
        if pd.isna(current_value):
            return False
        
        # Get comparison value
        if condition.value is not None:
            compare_value = condition.value
        elif condition.value_indicator:
            compare_col = self._get_indicator_column_name(condition.value_indicator, condition.value_indicator_params)
            if compare_col not in row.index:
                return False
            compare_value = row[compare_col]
        else:
            return False
        
        if pd.isna(compare_value):
            return False
        
        # Evaluate operator
        op = condition.operator
        
        if op == ComparisonOperator.GT:
            return current_value > compare_value
        elif op == ComparisonOperator.GTE:
            return current_value >= compare_value
        elif op == ComparisonOperator.LT:
            return current_value < compare_value
        elif op == ComparisonOperator.LTE:
            return current_value <= compare_value
        elif op == ComparisonOperator.EQ:
            return current_value == compare_value
        elif op == ComparisonOperator.CROSS_ABOVE:
            if prev_row is None:
                return False
            prev_val = prev_row.get(col_name)
            if condition.value_indicator:
                prev_compare = prev_row.get(self._get_indicator_column_name(condition.value_indicator, condition.value_indicator_params))
            else:
                prev_compare = condition.value
            if pd.isna(prev_val) or pd.isna(prev_compare):
                return False
            return prev_val <= prev_compare and current_value > compare_value
        elif op == ComparisonOperator.CROSS_BELOW:
            if prev_row is None:
                return False
            prev_val = prev_row.get(col_name)
            if condition.value_indicator:
                prev_compare = prev_row.get(self._get_indicator_column_name(condition.value_indicator, condition.value_indicator_params))
            else:
                prev_compare = condition.value
            if pd.isna(prev_val) or pd.isna(prev_compare):
                return False
            return prev_val >= prev_compare and current_value < compare_value
        
        return False
    
    def _evaluate_condition_group(self, group: ConditionGroup, row: pd.Series, prev_row: Optional[pd.Series]) -> bool:
        """Evaluate a group of conditions"""
        results = [self._evaluate_condition(c, row, prev_row) for c in group.conditions]
        
        if group.logic == "AND":
            return all(results)
        else:  # OR
            return any(results)
    
    def _check_stop_loss(self, row: pd.Series) -> bool:
        """Check if stop loss is triggered"""
        if not self.position or not self.strategy.exit.stop_loss:
            return False
        
        sl = self.strategy.exit.stop_loss
        entry_price = self.position.entry_price
        current_low = row["low"]
        
        if sl.type == "percent":
            stop_price = entry_price * (1 - sl.value / 100)
        elif sl.type == "atr":
            atr_col = f"ATR_{self.strategy.exit.stop_loss.value if hasattr(self.strategy.exit.stop_loss, 'atr_period') else 14}"
            if atr_col in row.index:
                stop_price = entry_price - row[atr_col] * sl.value
            else:
                return False
        else:  # fixed
            stop_price = entry_price - sl.value
        
        return current_low <= stop_price
    
    def _check_take_profit(self, row: pd.Series) -> bool:
        """Check if take profit is triggered"""
        if not self.position or not self.strategy.exit.take_profit:
            return False
        
        tp = self.strategy.exit.take_profit
        entry_price = self.position.entry_price
        current_high = row["high"]
        
        if tp.type == "percent":
            target_price = entry_price * (1 + tp.value / 100)
        elif tp.type == "risk_reward":
            if self.strategy.exit.stop_loss:
                sl = self.strategy.exit.stop_loss
                if sl.type == "percent":
                    risk = entry_price * sl.value / 100
                    target_price = entry_price + risk * tp.value
                else:
                    return False
            else:
                return False
        else:  # fixed
            target_price = entry_price + tp.value
        
        return current_high >= target_price
    
    def _get_exit_price(self, row: pd.Series, reason: str) -> float:
        """Get the exit price based on reason"""
        if reason == "止损":
            sl = self.strategy.exit.stop_loss
            if sl.type == "percent":
                return self.position.entry_price * (1 - sl.value / 100)
        elif reason == "止盈":
            tp = self.strategy.exit.take_profit
            if tp.type == "percent":
                return self.position.entry_price * (1 + tp.value / 100)
        
        return row["close"]
    
    def run(self) -> BacktestResult:
        """
        Run the backtest.
        
        Returns:
            BacktestResult with all metrics and trade history
        """
        self.capital = self.strategy.initial_capital
        self.position = None
        self.trades = []
        self.equity_history = []
        
        prev_row = None
        
        for idx, row in self.data.iterrows():
            timestamp = row["timestamp"]
            
            # Skip rows with NaN values in indicators
            if row.isna().any():
                prev_row = row
                continue
            
            # Check exits first if we have a position
            if self.position:
                exit_reason = None
                
                # Check stop loss
                if self._check_stop_loss(row):
                    exit_reason = "止损"
                # Check take profit
                elif self._check_take_profit(row):
                    exit_reason = "止盈"
                # Check signal-based exit
                elif self.strategy.exit.conditions:
                    if self._evaluate_condition_group(self.strategy.exit.conditions, row, prev_row):
                        exit_reason = "信号出场"
                
                if exit_reason:
                    exit_price = self._get_exit_price(row, exit_reason)
                    self.position.close(
                        timestamp, 
                        exit_price, 
                        exit_reason,
                        self.strategy.commission_percent
                    )
                    self.capital += self.position.pnl
                    self.trades.append(self.position)
                    self.position = None
            
            # Check entry if no position
            if not self.position:
                if self._evaluate_condition_group(self.strategy.entry, row, prev_row):
                    # Calculate position size
                    ps = self.strategy.position_sizing
                    if ps.type == "fixed_amount":
                        position_size = min(ps.value, self.capital * 0.95)  # Leave some buffer
                    elif ps.type == "percent_equity":
                        position_size = self.capital * ps.value / 100
                    else:
                        position_size = self.capital * 0.1  # Default 10%
                    
                    entry_price = row["close"] * (1 + self.strategy.slippage_percent / 100)
                    quantity = position_size / entry_price
                    
                    self.position = Trade(
                        entry_time=timestamp,
                        entry_price=entry_price,
                        entry_reason="信号入场",
                        position_size=position_size,
                        quantity=quantity
                    )
            
            # Record equity
            current_equity = self.capital
            if self.position:
                unrealized_pnl = (row["close"] - self.position.entry_price) * self.position.quantity
                current_equity += unrealized_pnl
            
            self.equity_history.append({
                "timestamp": timestamp,
                "equity": current_equity,
                "capital": self.capital,
                "in_position": self.position is not None
            })
            
            prev_row = row
        
        # Close any open position at the end
        if self.position:
            last_row = self.data.iloc[-1]
            self.position.close(
                last_row["timestamp"],
                last_row["close"],
                "回测结束",
                self.strategy.commission_percent
            )
            self.capital += self.position.pnl
            self.trades.append(self.position)
        
        # Build result
        result = BacktestResult(
            strategy_name=self.strategy.name,
            symbol=self.strategy.symbol,
            timeframe=self.strategy.timeframe.value,
            start_date=self.data["timestamp"].min(),
            end_date=self.data["timestamp"].max(),
            initial_capital=self.strategy.initial_capital,
            final_capital=self.capital,
            trades=self.trades,
            equity_curve=pd.DataFrame(self.equity_history)
        )
        
        result.calculate_metrics()
        
        return result


if __name__ == "__main__":
    # Test with sample data
    from .strategy_schema import EXAMPLE_STRATEGY
    
    # Create sample data
    import numpy as np
    np.random.seed(42)
    n = 500
    
    # Simulate price with trend and noise
    trend = np.linspace(0, 50, n)
    noise = np.cumsum(np.random.randn(n) * 2)
    close = 40000 + trend + noise
    
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h"),
        "open": close + np.random.randn(n) * 100,
        "high": close + np.abs(np.random.randn(n) * 200),
        "low": close - np.abs(np.random.randn(n) * 200),
        "close": close,
        "volume": np.random.randint(100, 1000, n) * 1e6
    })
    
    # Create strategy and run backtest
    strategy = StrategyConfig(**EXAMPLE_STRATEGY)
    engine = BacktestEngine(strategy, df)
    result = engine.run()
    
    print("\n=== 回测结果 ===")
    for key, value in result.to_summary_dict().items():
        print(f"{key}: {value}")
    
    print(f"\n交易记录 (前5笔):")
    for trade in result.trades[:5]:
        print(f"  {trade.entry_time} -> {trade.exit_time}: {trade.pnl_percent:+.2f}% ({trade.exit_reason})")
