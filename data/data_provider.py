"""Abstract base class for data providers."""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date


class DataProvider(ABC):
    """Abstract base class for market data providers."""
    
    @abstractmethod
    def get_price_data(self,
                      symbols: List[str],
                      start_date: date,
                      end_date: date,
                      frequency: str = 'daily') -> pd.DataFrame:
        """Get price data for symbols.
        
        Args:
            symbols: List of asset symbols
            start_date: Start date for data
            end_date: End date for data
            frequency: Data frequency ('daily', 'weekly', 'monthly')
            
        Returns:
            DataFrame with price data
        """
        pass
    
    @abstractmethod
    def get_returns_data(self,
                        symbols: List[str],
                        start_date: date,
                        end_date: date,
                        frequency: str = 'daily') -> pd.DataFrame:
        """Get returns data for symbols.
        
        Args:
            symbols: List of asset symbols
            start_date: Start date for data
            end_date: End date for data
            frequency: Data frequency
            
        Returns:
            DataFrame with returns data
        """
        pass
    
    @abstractmethod
    def get_fundamental_data(self,
                           symbols: List[str],
                           metrics: List[str],
                           start_date: Optional[date] = None,
                           end_date: Optional[date] = None) -> Dict[str, pd.DataFrame]:
        """Get fundamental data for symbols.
        
        Args:
            symbols: List of asset symbols
            metrics: List of fundamental metrics
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary of DataFrames with fundamental data
        """
        pass
    
    @abstractmethod
    def get_market_cap_data(self,
                          symbols: List[str],
                          start_date: date,
                          end_date: date) -> pd.DataFrame:
        """Get market capitalization data.
        
        Args:
            symbols: List of asset symbols
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with market cap data
        """
        pass
    
    @abstractmethod
    def search_symbols(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """Search for symbols matching query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of symbol information dictionaries
        """
        pass
    
    @abstractmethod
    def get_asset_info(self, symbol: str) -> Dict[str, Any]:
        """Get detailed asset information.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Dictionary with asset information
        """
        pass
    
    def validate_symbols(self, symbols: List[str]) -> List[str]:
        """Validate and filter valid symbols.
        
        Args:
            symbols: List of symbols to validate
            
        Returns:
            List of valid symbols
        """
        valid_symbols = []
        for symbol in symbols:
            try:
                info = self.get_asset_info(symbol)
                if info:
                    valid_symbols.append(symbol)
            except Exception:
                continue
        return valid_symbols
    
    def get_available_date_range(self, symbol: str) -> Tuple[date, date]:
        """Get available date range for a symbol.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Tuple of (start_date, end_date)
        """
        try:
            # Try to get a small sample to determine range
            sample_data = self.get_price_data([symbol], 
                                            date(2000, 1, 1), 
                                            date.today())
            if not sample_data.empty:
                return sample_data.index[0].date(), sample_data.index[-1].date()
        except Exception:
            pass
        
        return date.today(), date.today()
