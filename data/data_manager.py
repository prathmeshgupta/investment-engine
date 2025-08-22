"""Data manager for coordinating data providers and caching."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, date, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .data_provider import DataProvider
from .yahoo_provider import YahooDataProvider
from .data_cache import DataCache


class DataManager:
    """Manages data providers and coordinates data retrieval."""
    
    def __init__(self, 
                 primary_provider: Optional[DataProvider] = None,
                 cache_enabled: bool = True,
                 cache_ttl: int = 3600):
        """Initialize data manager.
        
        Args:
            primary_provider: Primary data provider
            cache_enabled: Whether to enable caching
            cache_ttl: Cache time-to-live in seconds
        """
        self.primary_provider = primary_provider or YahooDataProvider()
        self.fallback_providers = []
        
        self.cache_enabled = cache_enabled
        self.cache = DataCache(ttl=cache_ttl) if cache_enabled else None
        
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def add_fallback_provider(self, provider: DataProvider) -> None:
        """Add a fallback data provider.
        
        Args:
            provider: Data provider to add as fallback
        """
        self.fallback_providers.append(provider)
    
    def get_price_data(self,
                      symbols: Union[str, List[str]],
                      start_date: date,
                      end_date: date,
                      frequency: str = 'daily') -> pd.DataFrame:
        """Get price data with caching and fallback support.
        
        Args:
            symbols: Symbol or list of symbols
            start_date: Start date for data
            end_date: End date for data
            frequency: Data frequency
            
        Returns:
            DataFrame with price data
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        # Check cache first
        cache_key = f"prices_{'-'.join(symbols)}_{start_date}_{end_date}_{frequency}"
        
        if self.cache_enabled and self.cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Try primary provider
        try:
            data = self.primary_provider.get_price_data(symbols, start_date, end_date, frequency)
            
            if not data.empty:
                # Cache the result
                if self.cache_enabled and self.cache:
                    self.cache.set(cache_key, data)
                return data
        
        except Exception as e:
            print(f"Primary provider failed: {e}")
        
        # Try fallback providers
        for provider in self.fallback_providers:
            try:
                data = provider.get_price_data(symbols, start_date, end_date, frequency)
                if not data.empty:
                    if self.cache_enabled and self.cache:
                        self.cache.set(cache_key, data)
                    return data
            except Exception as e:
                print(f"Fallback provider failed: {e}")
                continue
        
        # Return empty DataFrame if all providers fail
        return pd.DataFrame()
    
    def get_returns_data(self,
                        symbols: Union[str, List[str]],
                        start_date: date,
                        end_date: date,
                        frequency: str = 'daily') -> pd.DataFrame:
        """Get returns data with caching and fallback support.
        
        Args:
            symbols: Symbol or list of symbols
            start_date: Start date for data
            end_date: End date for data
            frequency: Data frequency
            
        Returns:
            DataFrame with returns data
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        cache_key = f"returns_{'-'.join(symbols)}_{start_date}_{end_date}_{frequency}"
        
        if self.cache_enabled and self.cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Get price data and calculate returns
        price_data = self.get_price_data(symbols, start_date, end_date, frequency)
        
        if price_data.empty:
            return pd.DataFrame()
        
        returns_data = price_data.pct_change().dropna()
        
        # Cache the result
        if self.cache_enabled and self.cache:
            self.cache.set(cache_key, returns_data)
        
        return returns_data
    
    def get_fundamental_data(self,
                           symbols: Union[str, List[str]],
                           metrics: List[str],
                           start_date: Optional[date] = None,
                           end_date: Optional[date] = None) -> Dict[str, pd.DataFrame]:
        """Get fundamental data with caching and fallback support.
        
        Args:
            symbols: Symbol or list of symbols
            metrics: List of fundamental metrics
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary of DataFrames with fundamental data
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        cache_key = f"fundamentals_{'-'.join(symbols)}_{'-'.join(metrics)}_{start_date}_{end_date}"
        
        if self.cache_enabled and self.cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Try primary provider
        try:
            data = self.primary_provider.get_fundamental_data(symbols, metrics, start_date, end_date)
            
            if data:
                if self.cache_enabled and self.cache:
                    self.cache.set(cache_key, data)
                return data
        
        except Exception as e:
            print(f"Primary provider failed for fundamentals: {e}")
        
        # Try fallback providers
        for provider in self.fallback_providers:
            try:
                data = provider.get_fundamental_data(symbols, metrics, start_date, end_date)
                if data:
                    if self.cache_enabled and self.cache:
                        self.cache.set(cache_key, data)
                    return data
            except Exception as e:
                continue
        
        return {}
    
    def get_market_cap_data(self,
                          symbols: Union[str, List[str]],
                          start_date: date,
                          end_date: date) -> pd.DataFrame:
        """Get market cap data with caching and fallback support.
        
        Args:
            symbols: Symbol or list of symbols
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with market cap data
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        cache_key = f"market_cap_{'-'.join(symbols)}_{start_date}_{end_date}"
        
        if self.cache_enabled and self.cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Try primary provider
        try:
            data = self.primary_provider.get_market_cap_data(symbols, start_date, end_date)
            
            if not data.empty:
                if self.cache_enabled and self.cache:
                    self.cache.set(cache_key, data)
                return data
        
        except Exception as e:
            print(f"Primary provider failed for market cap: {e}")
        
        # Try fallback providers
        for provider in self.fallback_providers:
            try:
                data = provider.get_market_cap_data(symbols, start_date, end_date)
                if not data.empty:
                    if self.cache_enabled and self.cache:
                        self.cache.set(cache_key, data)
                    return data
            except Exception as e:
                continue
        
        return pd.DataFrame()
    
    def get_batch_data(self,
                      symbols: List[str],
                      start_date: date,
                      end_date: date,
                      data_types: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Get multiple data types in batch for efficiency.
        
        Args:
            symbols: List of symbols
            start_date: Start date for data
            end_date: End date for data
            data_types: List of data types to fetch
            
        Returns:
            Dictionary with different data types
        """
        if data_types is None:
            data_types = ['prices', 'returns', 'market_cap']
        
        results = {}
        
        # Use thread pool for parallel data fetching
        futures = {}
        
        if 'prices' in data_types:
            futures['prices'] = self.executor.submit(
                self.get_price_data, symbols, start_date, end_date
            )
        
        if 'returns' in data_types:
            futures['returns'] = self.executor.submit(
                self.get_returns_data, symbols, start_date, end_date
            )
        
        if 'market_cap' in data_types:
            futures['market_cap'] = self.executor.submit(
                self.get_market_cap_data, symbols, start_date, end_date
            )
        
        if 'fundamentals' in data_types:
            fundamental_metrics = ['market_cap', 'pe_ratio', 'pb_ratio', 'roe', 'debt_to_equity']
            futures['fundamentals'] = self.executor.submit(
                self.get_fundamental_data, symbols, fundamental_metrics
            )
        
        # Collect results
        for data_type, future in futures.items():
            try:
                results[data_type] = future.result(timeout=30)
            except Exception as e:
                print(f"Error fetching {data_type}: {e}")
                results[data_type] = pd.DataFrame() if data_type != 'fundamentals' else {}
        
        return results
    
    def validate_data_quality(self, data: pd.DataFrame, symbol: str = None) -> Dict[str, Any]:
        """Validate data quality and return quality metrics.
        
        Args:
            data: Data to validate
            symbol: Symbol name for context
            
        Returns:
            Dictionary with quality metrics
        """
        if data.empty:
            return {
                'valid': False,
                'issues': ['Empty dataset'],
                'completeness': 0.0,
                'symbol': symbol
            }
        
        issues = []
        
        # Check for missing values
        missing_pct = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        if missing_pct > 0.1:  # More than 10% missing
            issues.append(f'High missing data: {missing_pct:.1%}')
        
        # Check for outliers (returns > 50% or < -50%)
        if data.select_dtypes(include=[np.number]).shape[1] > 0:
            numeric_data = data.select_dtypes(include=[np.number])
            
            # For returns data, check for extreme values
            if 'return' in str(data.columns).lower() or data.abs().max().max() < 1:
                extreme_values = (numeric_data.abs() > 0.5).sum().sum()
                if extreme_values > 0:
                    issues.append(f'Extreme values detected: {extreme_values}')
        
        # Check for constant values
        constant_cols = []
        for col in data.select_dtypes(include=[np.number]).columns:
            if data[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            issues.append(f'Constant columns: {constant_cols}')
        
        # Check data continuity (gaps in time series)
        if isinstance(data.index, pd.DatetimeIndex):
            expected_days = (data.index[-1] - data.index[0]).days
            actual_days = len(data)
            continuity = actual_days / max(expected_days, 1)
            
            if continuity < 0.8:  # Less than 80% continuity
                issues.append(f'Low data continuity: {continuity:.1%}')
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'completeness': 1 - missing_pct,
            'continuity': continuity if isinstance(data.index, pd.DatetimeIndex) else 1.0,
            'symbol': symbol,
            'shape': data.shape
        }
    
    def get_universe_data(self,
                         universe_name: str,
                         start_date: date,
                         end_date: date) -> Dict[str, pd.DataFrame]:
        """Get data for predefined investment universes.
        
        Args:
            universe_name: Name of the universe
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary with universe data
        """
        universes = {
            'sp500': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'JNJ', 'V'],
            'tech_giants': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'NFLX', 'ADBE', 'CRM', 'ORCL'],
            'sectors': ['XLK', 'XLF', 'XLV', 'XLI', 'XLE', 'XLP', 'XLY', 'XLU', 'XLB', 'XLRE'],
            'global_markets': ['SPY', 'EFA', 'EEM', 'VTI', 'VXUS', 'BND', 'BNDX', 'GLD', 'VNQ', 'DBC']
        }
        
        if universe_name not in universes:
            raise ValueError(f"Unknown universe: {universe_name}")
        
        symbols = universes[universe_name]
        
        return self.get_batch_data(symbols, start_date, end_date)
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        if self.cache:
            self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.cache:
            return self.cache.get_stats()
        return {'enabled': False}
