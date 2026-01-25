"""
Technical Indicators Module

Calculates technical indicators used in trading strategies.
Uses the 'ta' library for standard indicator calculations.
"""

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange


class TechnicalIndicators:
    """Calculate technical indicators on OHLCV data"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV DataFrame.
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume
        """
        self.df = df.copy()
        self._validate_data()
    
    def _validate_data(self):
        """Ensure required columns exist"""
        required = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def rsi(self, period: int = 14) -> pd.Series:
        """
        Relative Strength Index
        
        Args:
            period: RSI period (default 14)
            
        Returns:
            RSI values (0-100)
        """
        indicator = RSIIndicator(close=self.df["close"], window=period)
        return indicator.rsi()
    
    def macd(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        indicator = MACD(
            close=self.df["close"],
            window_fast=fast_period,
            window_slow=slow_period,
            window_sign=signal_period
        )
        return indicator.macd(), indicator.macd_signal(), indicator.macd_diff()
    
    def sma(self, period: int = 20) -> pd.Series:
        """
        Simple Moving Average
        
        Args:
            period: SMA period
            
        Returns:
            SMA values
        """
        indicator = SMAIndicator(close=self.df["close"], window=period)
        return indicator.sma_indicator()
    
    def ema(self, period: int = 20) -> pd.Series:
        """
        Exponential Moving Average
        
        Args:
            period: EMA period
            
        Returns:
            EMA values
        """
        indicator = EMAIndicator(close=self.df["close"], window=period)
        return indicator.ema_indicator()
    
    def bollinger_bands(
        self,
        period: int = 20,
        std_dev: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands
        
        Args:
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        indicator = BollingerBands(
            close=self.df["close"],
            window=period,
            window_dev=std_dev
        )
        return (
            indicator.bollinger_hband(),
            indicator.bollinger_mavg(),
            indicator.bollinger_lband()
        )
    
    def atr(self, period: int = 14) -> pd.Series:
        """
        Average True Range
        
        Args:
            period: ATR period
            
        Returns:
            ATR values
        """
        indicator = AverageTrueRange(
            high=self.df["high"],
            low=self.df["low"],
            close=self.df["close"],
            window=period
        )
        return indicator.average_true_range()
    
    def add_all_indicators(
        self,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        sma_periods: list[int] = None,
        ema_periods: list[int] = None,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14
    ) -> pd.DataFrame:
        """
        Add all common indicators to the DataFrame.
        
        Returns:
            DataFrame with indicator columns added
        """
        if sma_periods is None:
            sma_periods = [20, 50, 200]
        if ema_periods is None:
            ema_periods = [12, 26]
        
        result = self.df.copy()
        
        # RSI
        result[f"RSI_{rsi_period}"] = self.rsi(rsi_period)
        
        # MACD
        macd_line, signal_line, histogram = self.macd(macd_fast, macd_slow, macd_signal)
        result[f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"] = macd_line
        result[f"MACD_SIGNAL_{macd_fast}_{macd_slow}_{macd_signal}"] = signal_line
        result[f"MACD_HIST_{macd_fast}_{macd_slow}_{macd_signal}"] = histogram
        
        # SMAs
        for period in sma_periods:
            result[f"SMA_{period}"] = self.sma(period)
        
        # EMAs
        for period in ema_periods:
            result[f"EMA_{period}"] = self.ema(period)
        
        # Bollinger Bands
        upper, middle, lower = self.bollinger_bands(bb_period, bb_std)
        result[f"BB_UPPER_{bb_period}"] = upper
        result[f"BB_MIDDLE_{bb_period}"] = middle
        result[f"BB_LOWER_{bb_period}"] = lower
        
        # ATR
        result[f"ATR_{atr_period}"] = self.atr(atr_period)
        
        return result


def detect_crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Detect when series1 crosses above series2.
    
    Returns:
        Boolean series, True where crossover occurs
    """
    prev_below = series1.shift(1) < series2.shift(1)
    curr_above = series1 > series2
    return prev_below & curr_above


def detect_crossunder(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Detect when series1 crosses below series2.
    
    Returns:
        Boolean series, True where crossunder occurs
    """
    prev_above = series1.shift(1) > series2.shift(1)
    curr_below = series1 < series2
    return prev_above & curr_below


if __name__ == "__main__":
    # Test with sample data
    import numpy as np
    
    # Create sample OHLCV data
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n))
    
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h"),
        "open": close + np.random.randn(n) * 0.5,
        "high": close + np.abs(np.random.randn(n)),
        "low": close - np.abs(np.random.randn(n)),
        "close": close,
        "volume": np.random.randint(1000, 10000, n)
    })
    
    # Calculate indicators
    indicators = TechnicalIndicators(df)
    result = indicators.add_all_indicators()
    
    print("Columns added:")
    print([col for col in result.columns if col not in df.columns])
    print("\nSample data:")
    print(result[["close", "RSI_14", "MACD_12_26_9", "SMA_20"]].tail())
