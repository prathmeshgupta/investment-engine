"""Yahoo Finance data provider implementation."""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date, timedelta
import warnings

from .data_provider import DataProvider


class YahooDataProvider(DataProvider):
    """Yahoo Finance data provider."""
    
    def __init__(self):
        """Initialize Yahoo Finance provider."""
        self.session = None
        warnings.filterwarnings('ignore', category=FutureWarning)
    
    def get_price_data(self,
                      symbols: List[str],
                      start_date: date,
                      end_date: date,
                      frequency: str = 'daily') -> pd.DataFrame:
        """Get price data from Yahoo Finance.
        
        Args:
            symbols: List of asset symbols
            start_date: Start date for data
            end_date: End date for data
            frequency: Data frequency ('daily', 'weekly', 'monthly')
            
        Returns:
            DataFrame with adjusted close prices
        """
        # Map frequency to yfinance interval
        interval_map = {
            'daily': '1d',
            'weekly': '1wk',
            'monthly': '1mo'
        }
        interval = interval_map.get(frequency, '1d')
        
        try:
            # Download data for all symbols
            tickers = yf.Tickers(' '.join(symbols))
            data = tickers.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                return pd.DataFrame()
            
            # Extract adjusted close prices
            if 'Adj Close' in data.columns.levels[0]:
                price_data = data['Adj Close']
            else:
                price_data = data['Close']
            
            # Clean data
            price_data = price_data.dropna(how='all')
            
            return price_data
            
        except Exception as e:
            print(f"Error fetching price data: {e}")
            return pd.DataFrame()
    
    def get_returns_data(self,
                        symbols: List[str],
                        start_date: date,
                        end_date: date,
                        frequency: str = 'daily') -> pd.DataFrame:
        """Get returns data from Yahoo Finance.
        
        Args:
            symbols: List of asset symbols
            start_date: Start date for data
            end_date: End date for data
            frequency: Data frequency
            
        Returns:
            DataFrame with returns data
        """
        price_data = self.get_price_data(symbols, start_date, end_date, frequency)
        
        if price_data.empty:
            return pd.DataFrame()
        
        # Calculate returns
        returns_data = price_data.pct_change().dropna()
        
        return returns_data
    
    def get_fundamental_data(self,
                           symbols: List[str],
                           metrics: List[str],
                           start_date: Optional[date] = None,
                           end_date: Optional[date] = None) -> Dict[str, pd.DataFrame]:
        """Get fundamental data from Yahoo Finance.
        
        Args:
            symbols: List of asset symbols
            metrics: List of fundamental metrics
            start_date: Start date for data (not used for Yahoo)
            end_date: End date for data (not used for Yahoo)
            
        Returns:
            Dictionary of DataFrames with fundamental data
        """
        fundamental_data = {}
        
        # Yahoo Finance fundamental metrics mapping
        yahoo_metrics = {
            'market_cap': 'marketCap',
            'enterprise_value': 'enterpriseValue',
            'pe_ratio': 'trailingPE',
            'pb_ratio': 'priceToBook',
            'debt_to_equity': 'debtToEquity',
            'roe': 'returnOnEquity',
            'roa': 'returnOnAssets',
            'current_ratio': 'currentRatio',
            'gross_margin': 'grossMargins',
            'operating_margin': 'operatingMargins',
            'profit_margin': 'profitMargins',
            'revenue_growth': 'revenueGrowth',
            'earnings_growth': 'earningsGrowth'
        }
        
        for metric in metrics:
            if metric in yahoo_metrics:
                metric_data = pd.DataFrame(index=[date.today()], columns=symbols)
                
                for symbol in symbols:
                    try:
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        
                        yahoo_key = yahoo_metrics[metric]
                        if yahoo_key in info:
                            metric_data.loc[date.today(), symbol] = info[yahoo_key]
                    
                    except Exception as e:
                        print(f"Error fetching {metric} for {symbol}: {e}")
                        continue
                
                fundamental_data[metric] = metric_data.dropna(axis=1, how='all')
        
        return fundamental_data
    
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
        # Get price and shares outstanding data
        price_data = self.get_price_data(symbols, start_date, end_date)
        
        if price_data.empty:
            return pd.DataFrame()
        
        # Get shares outstanding for each symbol
        market_cap_data = pd.DataFrame(index=price_data.index, columns=symbols)
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                shares_outstanding = info.get('sharesOutstanding', None)
                if shares_outstanding and symbol in price_data.columns:
                    # Calculate market cap = price * shares outstanding
                    market_caps = price_data[symbol] * shares_outstanding
                    market_cap_data[symbol] = market_caps
            
            except Exception as e:
                print(f"Error calculating market cap for {symbol}: {e}")
                continue
        
        return market_cap_data.dropna(axis=1, how='all')
    
    def search_symbols(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """Search for symbols using Yahoo Finance.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of symbol information dictionaries
        """
        # Yahoo Finance doesn't have a direct search API
        # This is a simplified implementation
        results = []
        
        # Try the query as a direct symbol
        try:
            ticker = yf.Ticker(query.upper())
            info = ticker.info
            
            if info and 'symbol' in info:
                results.append({
                    'symbol': info.get('symbol', query.upper()),
                    'name': info.get('longName', info.get('shortName', '')),
                    'exchange': info.get('exchange', ''),
                    'sector': info.get('sector', ''),
                    'industry': info.get('industry', '')
                })
        
        except Exception:
            pass
        
        return results[:limit]
    
    def get_asset_info(self, symbol: str) -> Dict[str, Any]:
        """Get detailed asset information.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Dictionary with asset information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract relevant information
            asset_info = {
                'symbol': info.get('symbol', symbol),
                'name': info.get('longName', info.get('shortName', '')),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'country': info.get('country', ''),
                'exchange': info.get('exchange', ''),
                'currency': info.get('currency', 'USD'),
                'market_cap': info.get('marketCap', None),
                'enterprise_value': info.get('enterpriseValue', None),
                'pe_ratio': info.get('trailingPE', None),
                'pb_ratio': info.get('priceToBook', None),
                'dividend_yield': info.get('dividendYield', None),
                'beta': info.get('beta', None),
                'description': info.get('longBusinessSummary', '')
            }
            
            return asset_info
        
        except Exception as e:
            print(f"Error fetching info for {symbol}: {e}")
            return {}
    
    def get_dividend_data(self,
                         symbols: List[str],
                         start_date: date,
                         end_date: date) -> pd.DataFrame:
        """Get dividend data for symbols.
        
        Args:
            symbols: List of asset symbols
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with dividend data
        """
        dividend_data = pd.DataFrame()
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                dividends = ticker.dividends
                
                # Filter by date range
                dividends = dividends[(dividends.index >= pd.Timestamp(start_date)) & 
                                    (dividends.index <= pd.Timestamp(end_date))]
                
                if not dividends.empty:
                    dividend_data[symbol] = dividends
            
            except Exception as e:
                print(f"Error fetching dividends for {symbol}: {e}")
                continue
        
        return dividend_data.fillna(0)
    
    def get_splits_data(self,
                       symbols: List[str],
                       start_date: date,
                       end_date: date) -> pd.DataFrame:
        """Get stock splits data for symbols.
        
        Args:
            symbols: List of asset symbols
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with splits data
        """
        splits_data = pd.DataFrame()
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                splits = ticker.splits
                
                # Filter by date range
                splits = splits[(splits.index >= pd.Timestamp(start_date)) & 
                              (splits.index <= pd.Timestamp(end_date))]
                
                if not splits.empty:
                    splits_data[symbol] = splits
            
            except Exception as e:
                print(f"Error fetching splits for {symbol}: {e}")
                continue
        
        return splits_data.fillna(1)  # Fill with 1 (no split)
    
    def get_options_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get options data for a symbol.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Dictionary with calls and puts DataFrames
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            expirations = ticker.options
            
            if not expirations:
                return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}
            
            # Get options for the nearest expiration
            nearest_expiry = expirations[0]
            options_chain = ticker.option_chain(nearest_expiry)
            
            return {
                'calls': options_chain.calls,
                'puts': options_chain.puts,
                'expiration': nearest_expiry
            }
        
        except Exception as e:
            print(f"Error fetching options for {symbol}: {e}")
            return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}
