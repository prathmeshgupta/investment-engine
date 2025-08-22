"""Quality factor model implementation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from core.enums import FactorType


class QualityModel:
    """Quality factor model implementation."""
    
    def __init__(self):
        """Initialize quality model."""
        self.quality_scores = None
        self.factor_returns = None
    
    def calculate_quality_scores(self, 
                               fundamentals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate composite quality scores.
        
        Args:
            fundamentals: Dictionary containing fundamental data DataFrames:
                - 'roe': Return on Equity
                - 'roa': Return on Assets  
                - 'debt_to_equity': Debt to Equity ratio
                - 'current_ratio': Current ratio
                - 'gross_margin': Gross profit margin
                - 'earnings_stability': Earnings stability measure
                
        Returns:
            DataFrame with quality scores
        """
        # Validate required fundamentals
        required_metrics = ['roe', 'roa', 'debt_to_equity', 'current_ratio', 'gross_margin']
        for metric in required_metrics:
            if metric not in fundamentals:
                raise ValueError(f"Missing required fundamental metric: {metric}")
        
        # Get common dates and assets
        dates = None
        assets = None
        
        for metric_name, data in fundamentals.items():
            if dates is None:
                dates = data.index
                assets = data.columns
            else:
                dates = dates.intersection(data.index)
                assets = assets.intersection(data.columns)
        
        quality_scores = pd.DataFrame(index=dates, columns=assets)
        
        for date in dates:
            date_scores = pd.DataFrame(index=assets)
            
            # Profitability metrics (higher is better)
            if 'roe' in fundamentals and date in fundamentals['roe'].index:
                roe = fundamentals['roe'].loc[date]
                date_scores['roe_score'] = self._percentile_rank(roe)
            
            if 'roa' in fundamentals and date in fundamentals['roa'].index:
                roa = fundamentals['roa'].loc[date]
                date_scores['roa_score'] = self._percentile_rank(roa)
            
            if 'gross_margin' in fundamentals and date in fundamentals['gross_margin'].index:
                gross_margin = fundamentals['gross_margin'].loc[date]
                date_scores['gross_margin_score'] = self._percentile_rank(gross_margin)
            
            # Financial strength metrics
            if 'debt_to_equity' in fundamentals and date in fundamentals['debt_to_equity'].index:
                debt_to_equity = fundamentals['debt_to_equity'].loc[date]
                # Lower debt is better, so invert the ranking
                date_scores['debt_score'] = self._percentile_rank(-debt_to_equity)
            
            if 'current_ratio' in fundamentals and date in fundamentals['current_ratio'].index:
                current_ratio = fundamentals['current_ratio'].loc[date]
                date_scores['liquidity_score'] = self._percentile_rank(current_ratio)
            
            # Earnings stability (if available)
            if 'earnings_stability' in fundamentals and date in fundamentals['earnings_stability'].index:
                earnings_stability = fundamentals['earnings_stability'].loc[date]
                date_scores['stability_score'] = self._percentile_rank(earnings_stability)
            
            # Calculate composite quality score
            score_columns = [col for col in date_scores.columns if col.endswith('_score')]
            if score_columns:
                composite_score = date_scores[score_columns].mean(axis=1)
                quality_scores.loc[date] = composite_score
        
        self.quality_scores = quality_scores.dropna(how='all')
        return self.quality_scores
    
    def _percentile_rank(self, series: pd.Series) -> pd.Series:
        """Calculate percentile ranks for a series."""
        return series.rank(pct=True)
    
    def construct_quality_factor(self, 
                                returns: pd.DataFrame,
                                market_caps: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Construct quality factor returns.
        
        Args:
            returns: Daily returns DataFrame
            market_caps: Market capitalization DataFrame (optional)
            
        Returns:
            DataFrame with quality factor returns
        """
        if self.quality_scores is None:
            raise ValueError("Quality scores not calculated. Call calculate_quality_scores first.")
        
        factor_returns = pd.DataFrame(index=returns.index, columns=['QMJ'])
        
        for date in self.quality_scores.index:
            if date not in returns.index:
                continue
            
            scores = self.quality_scores.loc[date].dropna()
            if len(scores) < 20:  # Minimum number of assets
                continue
            
            # Sort by quality scores
            sorted_scores = scores.sort_values(ascending=False)
            
            # Create high and low quality portfolios (top/bottom 30%)
            n_assets = len(sorted_scores)
            n_high = max(1, int(0.3 * n_assets))
            n_low = max(1, int(0.3 * n_assets))
            
            high_quality = sorted_scores.head(n_high).index
            low_quality = sorted_scores.tail(n_low).index
            
            # Get next period return
            next_date_idx = returns.index.get_loc(date) + 1
            if next_date_idx >= len(returns):
                continue
            
            next_date = returns.index[next_date_idx]
            next_returns = returns.loc[next_date]
            
            # Calculate portfolio returns
            if market_caps is not None and date in market_caps.index:
                # Value-weighted portfolios
                high_caps = market_caps.loc[date, high_quality].dropna()
                low_caps = market_caps.loc[date, low_quality].dropna()
                
                if len(high_caps) > 0 and len(low_caps) > 0:
                    high_weights = high_caps / high_caps.sum()
                    low_weights = low_caps / low_caps.sum()
                    
                    high_return = (next_returns[high_weights.index] * high_weights).sum()
                    low_return = (next_returns[low_weights.index] * low_weights).sum()
                else:
                    continue
            else:
                # Equal-weighted portfolios
                high_return = next_returns[high_quality].mean()
                low_return = next_returns[low_quality].mean()
            
            # Quality factor = High Quality - Low Quality
            factor_returns.loc[next_date, 'QMJ'] = high_return - low_return
        
        self.factor_returns = factor_returns.dropna()
        return self.factor_returns
    
    def calculate_piotroski_score(self, fundamentals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate Piotroski F-Score for quality assessment.
        
        Args:
            fundamentals: Dictionary with fundamental data
            
        Returns:
            DataFrame with Piotroski F-Scores (0-9)
        """
        required_metrics = [
            'net_income', 'operating_cash_flow', 'roa', 'long_term_debt',
            'current_ratio', 'shares_outstanding', 'gross_margin', 'asset_turnover'
        ]
        
        # Get common dates and assets
        dates = None
        assets = None
        
        for metric in required_metrics:
            if metric in fundamentals:
                if dates is None:
                    dates = fundamentals[metric].index
                    assets = fundamentals[metric].columns
                else:
                    dates = dates.intersection(fundamentals[metric].index)
                    assets = assets.intersection(fundamentals[metric].columns)
        
        piotroski_scores = pd.DataFrame(index=dates, columns=assets)
        
        for date in dates:
            scores = pd.DataFrame(index=assets)
            
            # Profitability criteria (4 points)
            if 'net_income' in fundamentals:
                ni = fundamentals['net_income'].loc[date]
                scores['positive_ni'] = (ni > 0).astype(int)
            
            if 'roa' in fundamentals:
                roa = fundamentals['roa'].loc[date]
                scores['positive_roa'] = (roa > 0).astype(int)
            
            if 'operating_cash_flow' in fundamentals:
                ocf = fundamentals['operating_cash_flow'].loc[date]
                scores['positive_ocf'] = (ocf > 0).astype(int)
                
                # Quality of earnings
                if 'net_income' in fundamentals:
                    ni = fundamentals['net_income'].loc[date]
                    scores['quality_earnings'] = (ocf > ni).astype(int)
            
            # Leverage, liquidity and source of funds criteria (3 points)
            if 'long_term_debt' in fundamentals and date in fundamentals['long_term_debt'].index:
                # Check if we have previous period data
                prev_dates = [d for d in fundamentals['long_term_debt'].index if d < date]
                if prev_dates:
                    prev_date = max(prev_dates)
                    ltd_current = fundamentals['long_term_debt'].loc[date]
                    ltd_prev = fundamentals['long_term_debt'].loc[prev_date]
                    scores['decreasing_debt'] = (ltd_current < ltd_prev).astype(int)
            
            if 'current_ratio' in fundamentals:
                # Similar logic for current ratio improvement
                prev_dates = [d for d in fundamentals['current_ratio'].index if d < date]
                if prev_dates:
                    prev_date = max(prev_dates)
                    cr_current = fundamentals['current_ratio'].loc[date]
                    cr_prev = fundamentals['current_ratio'].loc[prev_date]
                    scores['improving_liquidity'] = (cr_current > cr_prev).astype(int)
            
            if 'shares_outstanding' in fundamentals:
                # No dilution
                prev_dates = [d for d in fundamentals['shares_outstanding'].index if d < date]
                if prev_dates:
                    prev_date = max(prev_dates)
                    shares_current = fundamentals['shares_outstanding'].loc[date]
                    shares_prev = fundamentals['shares_outstanding'].loc[prev_date]
                    scores['no_dilution'] = (shares_current <= shares_prev).astype(int)
            
            # Operating efficiency criteria (2 points)
            if 'gross_margin' in fundamentals:
                prev_dates = [d for d in fundamentals['gross_margin'].index if d < date]
                if prev_dates:
                    prev_date = max(prev_dates)
                    gm_current = fundamentals['gross_margin'].loc[date]
                    gm_prev = fundamentals['gross_margin'].loc[prev_date]
                    scores['improving_margin'] = (gm_current > gm_prev).astype(int)
            
            if 'asset_turnover' in fundamentals:
                prev_dates = [d for d in fundamentals['asset_turnover'].index if d < date]
                if prev_dates:
                    prev_date = max(prev_dates)
                    at_current = fundamentals['asset_turnover'].loc[date]
                    at_prev = fundamentals['asset_turnover'].loc[prev_date]
                    scores['improving_turnover'] = (at_current > at_prev).astype(int)
            
            # Calculate total Piotroski score
            score_columns = [col for col in scores.columns]
            if score_columns:
                total_score = scores[score_columns].sum(axis=1)
                piotroski_scores.loc[date] = total_score
        
        return piotroski_scores.dropna(how='all')
