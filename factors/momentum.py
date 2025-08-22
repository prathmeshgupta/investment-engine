"""Momentum factor model implementation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from core.enums import FactorType


class MomentumModel:
    """Momentum factor model implementation."""
    
    def __init__(self, 
                 lookback_period: int = 252,
                 skip_period: int = 21,
                 holding_period: int = 21):
        """Initialize momentum model.
        
        Args:
            lookback_period: Period to calculate momentum (252 = 1 year)
            skip_period: Period to skip to avoid microstructure effects (21 = 1 month)
            holding_period: Holding period for momentum portfolios
        """
        self.lookback_period = lookback_period
        self.skip_period = skip_period
        self.holding_period = holding_period
        self.momentum_scores = None
        self.factor_returns = None
    
    def calculate_momentum_scores(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum scores for assets.
        
        Args:
            returns: Daily returns DataFrame (dates x assets)
            
        Returns:
            DataFrame with momentum scores
        """
        # Calculate cumulative returns over lookback period, skipping recent period
        momentum_scores = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for i in range(self.lookback_period + self.skip_period, len(returns)):
            date = returns.index[i]
            
            # Calculate momentum from t-lookback-skip to t-skip
            start_idx = i - self.lookback_period - self.skip_period
            end_idx = i - self.skip_period
            
            period_returns = returns.iloc[start_idx:end_idx]
            
            # Cumulative return over the period
            cum_returns = (1 + period_returns).prod() - 1
            momentum_scores.loc[date] = cum_returns
        
        self.momentum_scores = momentum_scores.dropna(how='all')
        return self.momentum_scores
    
    def construct_momentum_factor(self, 
                                returns: pd.DataFrame,
                                market_caps: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Construct momentum factor returns.
        
        Args:
            returns: Daily returns DataFrame
            market_caps: Market capitalization DataFrame (optional, for value-weighting)
            
        Returns:
            DataFrame with momentum factor returns
        """
        if self.momentum_scores is None:
            self.calculate_momentum_scores(returns)
        
        factor_returns = pd.DataFrame(index=returns.index, columns=['MOM'])
        
        for date in self.momentum_scores.index:
            if date not in returns.index:
                continue
            
            scores = self.momentum_scores.loc[date].dropna()
            if len(scores) < 20:  # Minimum number of assets
                continue
            
            # Sort by momentum scores
            sorted_scores = scores.sort_values(ascending=False)
            
            # Create winner and loser portfolios (top/bottom 30%)
            n_assets = len(sorted_scores)
            n_winners = max(1, int(0.3 * n_assets))
            n_losers = max(1, int(0.3 * n_assets))
            
            winners = sorted_scores.head(n_winners).index
            losers = sorted_scores.tail(n_losers).index
            
            # Get returns for the holding period
            holding_start = returns.index.get_loc(date)
            holding_end = min(holding_start + self.holding_period, len(returns))
            
            if holding_end <= holding_start:
                continue
            
            holding_returns = returns.iloc[holding_start:holding_end]
            
            # Calculate portfolio returns
            if market_caps is not None and date in market_caps.index:
                # Value-weighted portfolios
                winner_caps = market_caps.loc[date, winners].dropna()
                loser_caps = market_caps.loc[date, losers].dropna()
                
                if len(winner_caps) > 0 and len(loser_caps) > 0:
                    winner_weights = winner_caps / winner_caps.sum()
                    loser_weights = loser_caps / loser_caps.sum()
                    
                    winner_returns = (holding_returns[winner_weights.index] * winner_weights).sum(axis=1)
                    loser_returns = (holding_returns[loser_weights.index] * loser_weights).sum(axis=1)
                else:
                    continue
            else:
                # Equal-weighted portfolios
                winner_returns = holding_returns[winners].mean(axis=1)
                loser_returns = holding_returns[losers].mean(axis=1)
            
            # Momentum factor = Winners - Losers
            mom_returns = winner_returns - loser_returns
            
            for ret_date in mom_returns.index:
                if ret_date in factor_returns.index:
                    factor_returns.loc[ret_date, 'MOM'] = mom_returns[ret_date]
        
        self.factor_returns = factor_returns.dropna()
        return self.factor_returns
    
    def calculate_cross_sectional_momentum(self, 
                                         returns: pd.DataFrame,
                                         window: int = 21) -> pd.DataFrame:
        """Calculate cross-sectional momentum scores.
        
        Args:
            returns: Returns DataFrame
            window: Rolling window for cross-sectional ranking
            
        Returns:
            Cross-sectional momentum scores
        """
        cs_momentum = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for i in range(self.lookback_period, len(returns)):
            date = returns.index[i]
            
            # Calculate momentum scores
            start_idx = i - self.lookback_period
            period_returns = returns.iloc[start_idx:i]
            cum_returns = (1 + period_returns).prod() - 1
            
            # Cross-sectional ranking (percentile scores)
            scores = cum_returns.rank(pct=True)
            cs_momentum.loc[date] = scores
        
        return cs_momentum.dropna(how='all')
    
    def calculate_time_series_momentum(self, 
                                     returns: pd.DataFrame,
                                     lookback: int = 252) -> pd.DataFrame:
        """Calculate time-series momentum signals.
        
        Args:
            returns: Returns DataFrame
            lookback: Lookback period for momentum calculation
            
        Returns:
            Time-series momentum signals (-1, 0, 1)
        """
        ts_momentum = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for i in range(lookback, len(returns)):
            date = returns.index[i]
            
            # Calculate momentum for each asset
            for asset in returns.columns:
                asset_returns = returns[asset].iloc[i-lookback:i]
                
                if asset_returns.isna().sum() > lookback * 0.2:  # Too many missing values
                    continue
                
                # Calculate cumulative return
                cum_return = (1 + asset_returns.dropna()).prod() - 1
                
                # Generate signal
                if cum_return > 0:
                    ts_momentum.loc[date, asset] = 1  # Long signal
                elif cum_return < 0:
                    ts_momentum.loc[date, asset] = -1  # Short signal
                else:
                    ts_momentum.loc[date, asset] = 0  # Neutral
        
        return ts_momentum.dropna(how='all')
    
    def calculate_risk_adjusted_momentum(self, 
                                       returns: pd.DataFrame,
                                       volatility_window: int = 63) -> pd.DataFrame:
        """Calculate risk-adjusted momentum scores.
        
        Args:
            returns: Returns DataFrame
            volatility_window: Window for volatility calculation
            
        Returns:
            Risk-adjusted momentum scores
        """
        if self.momentum_scores is None:
            self.calculate_momentum_scores(returns)
        
        # Calculate rolling volatility
        volatility = returns.rolling(window=volatility_window).std() * np.sqrt(252)
        
        # Risk-adjusted momentum = momentum / volatility
        risk_adj_momentum = self.momentum_scores / volatility
        
        return risk_adj_momentum.replace([np.inf, -np.inf], np.nan).dropna(how='all')
