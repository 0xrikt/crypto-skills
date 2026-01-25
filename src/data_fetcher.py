"""
Market Data Fetcher

Fetches historical OHLCV data from cryptocurrency exchanges.
Uses ccxt library for exchange-agnostic data access.
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import time


class DataFetcher:
    """Fetches historical market data from crypto exchanges"""
    
    # Timeframe to milliseconds mapping
    TIMEFRAME_MS = {
        "1m": 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "6h": 6 * 60 * 60 * 1000,
        "12h": 12 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
        "1w": 7 * 24 * 60 * 60 * 1000,
    }
    
    def __init__(self, exchange_id: str = "binance"):
        """
        Initialize data fetcher.
        
        Args:
            exchange_id: Exchange to fetch data from (default: binance)
        """
        self.exchange_id = exchange_id
        self.exchange = getattr(ccxt, exchange_id)({
            "enableRateLimit": True,
        })
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (candlestick) data.
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Candle timeframe (e.g., "4h", "1d")
            start_date: Start date for data
            end_date: End date for data
            limit: Maximum candles per request
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if end_date is None:
            end_date = datetime.now()
        
        if start_date is None:
            # Default to 1 year of data
            start_date = end_date - timedelta(days=365)
        
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)
        
        all_candles = []
        current_ts = start_ts
        
        print(f"Fetching {symbol} {timeframe} data from {start_date.date()} to {end_date.date()}...")
        
        while current_ts < end_ts:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=current_ts,
                    limit=limit
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                
                # Move to next batch
                last_ts = candles[-1][0]
                if last_ts <= current_ts:
                    break
                current_ts = last_ts + self.TIMEFRAME_MS.get(timeframe, 60000)
                
                # Progress indicator
                progress = (current_ts - start_ts) / (end_ts - start_ts) * 100
                print(f"  Progress: {min(progress, 100):.1f}%", end="\r")
                
                # Rate limiting
                time.sleep(self.exchange.rateLimit / 1000)
                
            except Exception as e:
                print(f"\nError fetching data: {e}")
                break
        
        print(f"\nFetched {len(all_candles)} candles")
        
        if not all_candles:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        # Filter to requested date range
        df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]
        
        # Remove duplicates and sort
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        
        return df
    
    def get_available_symbols(self) -> list[str]:
        """Get list of available trading pairs"""
        self.exchange.load_markets()
        return list(self.exchange.symbols)
    
    def get_symbol_info(self, symbol: str) -> dict:
        """Get information about a trading pair"""
        self.exchange.load_markets()
        return self.exchange.markets.get(symbol, {})


# Convenience function for quick data fetching
def fetch_crypto_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    days: int = 365,
    exchange: str = "binance"
) -> pd.DataFrame:
    """
    Quick function to fetch crypto OHLCV data.
    
    Args:
        symbol: Trading pair
        timeframe: Candle timeframe
        days: Number of days of historical data
        exchange: Exchange to use
        
    Returns:
        DataFrame with OHLCV data
    """
    fetcher = DataFetcher(exchange)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    return fetcher.fetch_ohlcv(symbol, timeframe, start_date, end_date)


if __name__ == "__main__":
    # Test data fetching
    df = fetch_crypto_data("BTC/USDT", "4h", days=30)
    print(f"\nData shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nLast 5 rows:")
    print(df.tail())
