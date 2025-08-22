"""Volatility factor model implementation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from core.enums import FactorType


class VolatilityModel:
    """Low volatility factor model implementation."""
    
    def __init__(self, 
                 lookback_period: int = 252,
                 min_observations: int = 126):
        """Initialize volatility model.
        
        Args:
            lookback_period: Period to calculate volatility (252 = 1 year)
            min_observations: Minimum observations required
        """
        self.lookback_period = lookback_period
        self.min_observations = min_observations
        self.volatility_scores = None
        self.factor_returns = None
    
    def calculate_volatility_scores(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility scores for assets.
        
        Args:
            returns: Daily returns DataFrame (dates x assets)
            
        Returns:
            DataFrame with volatility scores (annualized)
        """
        volatility_scores = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for i in range(self.lookback_period, len(returns)):
            date = returns.index[i]
            
            # Calculate rolling volatility
            period_returns = returns.iloc[i-self.lookback_period:i]
            
            # Calculate annualized volatility for each asset
            vol_scores = period_returns.std() * np.sqrt(252)
            volatility_scores.loc[date] = vol_scores
        
        self.volatility_scores = volatility_scores.dropna(how='all')
        return self.volatility_scores
    
    def construct_volatility_factor(self, 
                                  returns: pd.DataFrame,
                                  market_caps: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Construct low volatility factor returns.
        
        Args:
            returns: Daily returns DataFrame
            market_caps: Market capitalization DataFrame (optional)
            
        Returns:
            DataFrame with volatility factor returns
        """
        if self.volatility_scores is None:
            self.calculate_volatility_scores(returns)
        
        factor_returns = pd.DataFrame(index=returns.index, columns=['LowVol'])
        
        for date in self.volatility_scores.index:
            if date not in returns.index:
                continue
            
            vol_scores = self.volatility_scores.loc[date].dropna()
            if len(vol_scores) < 20:  # Minimum number of assets
                continue
            
            # Sort by volatility (ascending - low vol first)
            sorted_vol = vol_scores.sort_values(ascending=True)
            
            # Create low and high volatility portfolios (bottom/top 30%)
            n_assets = len(sorted_vol)
            n_low_vol = max(1, int(0.3 * n_assets))
            n_high_vol = max(1, int(0.3 * n_assets))
            
            low_vol_assets = sorted_vol.head(n_low_vol).index
            high_vol_assets = sorted_vol.tail(n_high_vol).index
            
            # Get next period return
            next_date_idx = returns.index.get_loc(date) + 1
            if next_date_idx >= len(returns):
                continue
            
            next_date = returns.index[next_date_idx]
            next_returns = returns.loc[next_date]
            
            # Calculate portfolio returns
            if market_caps is not None and date in market_caps.index:
                # Value-weighted portfolios
                low_vol_caps = market_caps.loc[date, low_vol_assets].dropna()
                high_vol_caps = market_caps.loc[date, high_vol_assets].dropna()
                
                if len(low_vol_caps) > 0 and len(high_vol_caps) > 0:
                    low_vol_weights = low_vol_caps / low_vol_caps.sum()
                    high_vol_weights = high_vol_caps / high_vol_caps.sum()
                    
                    low_vol_return = (next_returns[low_vol_weights.index] * low_vol_weights).sum()
                    high_vol_return = (next_returns[high_vol_weights.index] * high_vol_weights).sum()
                else:
                    continue
            else:
                # Equal-weighted portfolios
                low_vol_return = next_returns[low_vol_assets].mean()
                high_vol_return = next_returns[high_vol_assets].mean()
            
            # Low volatility factor = Low Vol - High Vol
            factor_returns.loc[next_date, 'LowVol'] = low_vol_return - high_vol_return
        
        self.factor_returns = factor_returns.dropna()
        return self.factor_returns
    
    def calculate_risk_adjusted_volatility(self, 
                                         returns: pd.DataFrame,
                                         market_returns: pd.Series) -> pd.DataFrame:
        """Calculate beta-adjusted volatility scores.
        
        Args:
            returns: Asset returns DataFrame
            market_returns: Market returns series
            
        Returns:
            DataFrame with beta-adjusted volatility scores
        """
        if self.volatility_scores is None:
            self.calculate_volatility_scores(returns)
        
        # Calculate rolling betas
        betas = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for i in range(self.lookback_period, len(returns)):
            date = returns.index[i]
            
            period_returns = returns.iloc[i-self.lookback_period:i]
            period_market = market_returns.iloc[i-self.lookback_period:i]
            
            # Align dates
            common_dates = period_returns.index.intersection(period_market.index)
            if len(common_dates) < self.min_observations:
                continue
            
            period_returns_aligned = period_returns.loc[common_dates]
            period_market_aligned = period_market.loc[common_dates]
            
            # Calculate beta for each asset
            market_var = period_market_aligned.var()
            if market_var > 0:
                for asset in returns.columns:
                    asset_returns = period_returns_aligned[asset].dropna()
                    if len(asset_returns) >= self.min_observations:
                        covariance = asset_returns.cov(period_market_aligned.loc[asset_returns.index])
                        betas.loc[date, asset] = covariance / market_var
        
        # Calculate idiosyncratic volatility (total vol - systematic vol)
        idiosyncratic_vol = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for date in self.volatility_scores.index:
            if date in betas.index:
                total_vol = self.volatility_scores.loc[date]
                beta_values = betas.loc[date]
                
                # Market volatility
                market_vol = market_returns.rolling(self.lookback_period).std().loc[date] * np.sqrt(252)
                
                # Systematic volatility = beta * market_vol
                systematic_vol = beta_values.abs() * market_vol
                
                # Idiosyncratic volatility
                idio_vol = np.sqrt(np.maximum(0, total_vol**2 - systematic_vol**2))
                idiosyncratic_vol.loc[date] = idio_vol
        
        return idiosyncratic_vol.dropna(how='all')
    
    def calculate_downside_volatility(self, 
                                    returns: pd.DataFrame,
                                    threshold: float = 0.0) -> pd.DataFrame:
        """Calculate downside volatility (semi-deviation).
        
        Args:
            returns: Returns DataFrame
            threshold: Threshold for downside calculation (default 0%)
            
        Returns:
            DataFrame with downside volatility scores
        """
        downside_vol = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for i in range(self.lookback_period, len(returns)):
            date = returns.index[i]
            
            period_returns = returns.iloc[i-self.lookback_period:i]
            
            # Calculate downside deviation for each asset
            downside_scores = pd.Series(index=returns.columns, dtype=float)
            
            for asset in returns.columns:
                asset_returns = period_returns[asset].dropna()
                if len(asset_returns) >= self.min_observations:
                    # Only consider returns below threshold
                    downside_returns = asset_returns[asset_returns < threshold]
                    if len(downside_returns) > 0:
                        downside_var = ((downside_returns - threshold) ** 2).mean()
                        downside_scores[asset] = np.sqrt(downside_var) * np.sqrt(252)
                    else:
                        downside_scores[asset] = 0.0
            
            downside_vol.loc[date] = downside_scores
        
        return downside_vol.dropna(how='all')
    
    def calculate_volatility_of_volatility(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility of volatility (vol-of-vol).
        
        Args:
            returns: Returns DataFrame
            
        Returns:
            DataFrame with vol-of-vol scores
        """
        # First calculate rolling volatility with shorter window
        short_window = 21  # 1 month
        rolling_vol = returns.rolling(window=short_window).std() * np.sqrt(252)
        
        # Then calculate volatility of the volatility series
        vol_of_vol = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for i in range(self.lookback_period, len(rolling_vol)):
            date = rolling_vol.index[i]
            
            period_vol = rolling_vol.iloc[i-self.lookback_period:i]
            
            # Calculate volatility of volatility for each asset
            volvol_scores = period_vol.std()
            vol_of_vol.loc[date] = volvol_scores
        
        return vol_of_vol.dropna(how='all')
