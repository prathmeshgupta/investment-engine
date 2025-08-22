"""Factor engine for coordinating multi-factor models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date

from .fama_french import FamaFrenchModel
from .momentum import MomentumModel
from .quality import QualityModel
from .volatility import VolatilityModel
from core.enums import FactorType


class FactorEngine:
    """Coordinates multiple factor models and combines factor exposures."""
    
    def __init__(self):
        """Initialize factor engine."""
        self.models = {
            'fama_french_3': FamaFrenchModel('3factor'),
            'fama_french_5': FamaFrenchModel('5factor'),
            'momentum': MomentumModel(),
            'quality': QualityModel(),
            'volatility': VolatilityModel()
        }
        self.factor_returns = pd.DataFrame()
        self.factor_loadings = pd.DataFrame()
        self.combined_scores = pd.DataFrame()
    
    def fama_french_3factor(self, returns: pd.DataFrame, market_caps: pd.DataFrame) -> pd.DataFrame:
        """Calculate Fama-French 3-factor returns."""
        return self.models['fama_french_3'].calculate_factors(returns, market_caps)
    
    def calculate_all_factors(self,
                            returns: pd.DataFrame,
                            market_caps: pd.DataFrame,
                            fundamentals: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """Calculate all factor returns.
        
        Args:
            returns: Asset returns DataFrame
            market_caps: Market capitalization DataFrame
            fundamentals: Fundamental data for quality factors
            
        Returns:
            Combined factor returns DataFrame
        """
        all_factor_returns = pd.DataFrame(index=returns.index)
        
        # Fama-French 3-factor
        try:
            # For demo purposes, create synthetic fundamental data
            if fundamentals is None:
                fundamentals = self._create_synthetic_fundamentals(returns, market_caps)
            
            ff3_returns = self.models['fama_french_3'].construct_factors(
                returns, market_caps, fundamentals.get('book_to_market', pd.DataFrame())
            )
            all_factor_returns = all_factor_returns.join(ff3_returns, how='outer')
        except Exception as e:
            print(f"Error calculating Fama-French 3-factor: {e}")
        
        # Momentum factor
        try:
            mom_returns = self.models['momentum'].construct_momentum_factor(returns, market_caps)
            all_factor_returns = all_factor_returns.join(mom_returns, how='outer')
        except Exception as e:
            print(f"Error calculating momentum factor: {e}")
        
        # Quality factor
        try:
            if fundamentals:
                quality_scores = self.models['quality'].calculate_quality_scores(fundamentals)
                quality_returns = self.models['quality'].construct_quality_factor(returns, market_caps)
                all_factor_returns = all_factor_returns.join(quality_returns, how='outer')
        except Exception as e:
            print(f"Error calculating quality factor: {e}")
        
        # Volatility factor
        try:
            vol_returns = self.models['volatility'].construct_volatility_factor(returns, market_caps)
            all_factor_returns = all_factor_returns.join(vol_returns, how='outer')
        except Exception as e:
            print(f"Error calculating volatility factor: {e}")
        
        self.factor_returns = all_factor_returns.dropna(how='all')
        return self.factor_returns
    
    def _create_synthetic_fundamentals(self, 
                                     returns: pd.DataFrame, 
                                     market_caps: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create synthetic fundamental data for demonstration."""
        np.random.seed(42)  # For reproducibility
        
        fundamentals = {}
        
        # Book-to-market (random walk with mean reversion)
        btm = pd.DataFrame(index=market_caps.index, columns=market_caps.columns)
        initial_btm = np.random.lognormal(0, 0.5, len(market_caps.columns))
        
        for i, date in enumerate(market_caps.index):
            if i == 0:
                btm.loc[date] = initial_btm
            else:
                # Mean reversion with noise
                prev_btm = btm.iloc[i-1].values
                btm.loc[date] = prev_btm * 0.95 + np.random.normal(0, 0.1, len(prev_btm))
                btm.loc[date] = np.maximum(btm.loc[date], 0.1)  # Floor at 0.1
        
        fundamentals['book_to_market'] = btm
        
        # ROE (correlated with returns)
        roe = pd.DataFrame(index=market_caps.index, columns=market_caps.columns)
        for col in returns.columns:
            if col in market_caps.columns:
                # ROE somewhat correlated with past returns
                past_returns = returns[col].rolling(252).mean().fillna(0)
                noise = np.random.normal(0, 0.05, len(past_returns))
                roe[col] = 0.1 + past_returns * 0.5 + noise
        
        fundamentals['roe'] = roe
        
        # ROA (similar to ROE but lower)
        fundamentals['roa'] = roe * 0.6
        
        # Debt to equity
        debt_to_equity = pd.DataFrame(index=market_caps.index, columns=market_caps.columns)
        for col in market_caps.columns:
            debt_to_equity[col] = np.random.lognormal(0, 0.3, len(market_caps.index))
        
        fundamentals['debt_to_equity'] = debt_to_equity
        
        # Current ratio
        current_ratio = pd.DataFrame(index=market_caps.index, columns=market_caps.columns)
        for col in market_caps.columns:
            current_ratio[col] = np.random.lognormal(0.5, 0.2, len(market_caps.index))
        
        fundamentals['current_ratio'] = current_ratio
        
        # Gross margin
        gross_margin = pd.DataFrame(index=market_caps.index, columns=market_caps.columns)
        for col in market_caps.columns:
            gross_margin[col] = np.random.beta(2, 2, len(market_caps.index)) * 0.5 + 0.1
        
        fundamentals['gross_margin'] = gross_margin
        
        return fundamentals
    
    def calculate_factor_loadings(self,
                                asset_returns: pd.DataFrame,
                                risk_free_rate: Optional[pd.Series] = None) -> pd.DataFrame:
        """Calculate factor loadings for all assets.
        
        Args:
            asset_returns: Asset returns DataFrame
            risk_free_rate: Risk-free rate series
            
        Returns:
            Factor loadings DataFrame
        """
        if self.factor_returns.empty:
            raise ValueError("Factor returns not calculated. Call calculate_all_factors first.")
        
        if risk_free_rate is None:
            # Use synthetic risk-free rate
            risk_free_rate = pd.Series(0.02/252, index=asset_returns.index)
        
        # Align dates
        common_dates = asset_returns.index.intersection(self.factor_returns.index).intersection(risk_free_rate.index)
        
        if len(common_dates) < 60:
            raise ValueError("Insufficient overlapping data for factor loading calculation.")
        
        asset_excess = asset_returns.loc[common_dates].subtract(risk_free_rate.loc[common_dates], axis=0)
        factor_excess = self.factor_returns.loc[common_dates].copy()
        
        # Subtract risk-free rate from market factor if present
        if 'MKT' in factor_excess.columns:
            factor_excess['MKT'] = factor_excess['MKT'] - risk_free_rate.loc[common_dates]
        
        loadings = pd.DataFrame(index=asset_returns.columns, columns=factor_excess.columns)
        
        from sklearn.linear_model import LinearRegression
        
        for asset in asset_returns.columns:
            y = asset_excess[asset].dropna()
            X = factor_excess.loc[y.index].dropna()
            
            if len(y) < 30:  # Minimum observations
                continue
            
            # Align data
            common_idx = y.index.intersection(X.index)
            if len(common_idx) < 30:
                continue
            
            y_aligned = y.loc[common_idx]
            X_aligned = X.loc[common_idx]
            
            # Remove any remaining NaN values
            mask = ~(y_aligned.isna() | X_aligned.isna().any(axis=1))
            y_final = y_aligned[mask]
            X_final = X_aligned[mask]
            
            if len(y_final) < 30:
                continue
            
            # Run regression
            try:
                reg = LinearRegression().fit(X_final, y_final)
                loadings.loc[asset, X_final.columns] = reg.coef_
            except Exception as e:
                print(f"Error calculating loadings for {asset}: {e}")
                continue
        
        self.factor_loadings = loadings.astype(float)
        return self.factor_loadings
    
    def calculate_factor_premiums(self, 
                                lookback_period: int = 252) -> pd.Series:
        """Calculate expected factor premiums.
        
        Args:
            lookback_period: Lookback period for premium calculation
            
        Returns:
            Expected factor premiums
        """
        if self.factor_returns.empty:
            raise ValueError("Factor returns not calculated.")
        
        # Use rolling mean as expected premium
        recent_returns = self.factor_returns.tail(lookback_period)
        premiums = recent_returns.mean() * 252  # Annualize
        
        return premiums
    
    def get_factor_exposures(self, 
                           portfolio_weights: pd.Series) -> pd.Series:
        """Calculate portfolio factor exposures.
        
        Args:
            portfolio_weights: Portfolio weights
            
        Returns:
            Portfolio factor exposures
        """
        if self.factor_loadings.empty:
            raise ValueError("Factor loadings not calculated.")
        
        # Align assets
        common_assets = portfolio_weights.index.intersection(self.factor_loadings.index)
        
        if len(common_assets) == 0:
            raise ValueError("No common assets between portfolio and factor loadings.")
        
        weights_aligned = portfolio_weights.loc[common_assets]
        loadings_aligned = self.factor_loadings.loc[common_assets]
        
        # Calculate weighted average exposures
        exposures = (weights_aligned.values.reshape(-1, 1) * loadings_aligned.values).sum(axis=0)
        
        return pd.Series(exposures, index=loadings_aligned.columns)
    
    def calculate_factor_attribution(self,
                                   portfolio_returns: pd.Series,
                                   portfolio_weights: pd.DataFrame) -> pd.DataFrame:
        """Calculate factor attribution for portfolio returns.
        
        Args:
            portfolio_returns: Portfolio returns series
            portfolio_weights: Portfolio weights over time
            
        Returns:
            Factor attribution DataFrame
        """
        if self.factor_returns.empty or self.factor_loadings.empty:
            raise ValueError("Factor returns and loadings must be calculated first.")
        
        attribution = pd.DataFrame(index=portfolio_returns.index, columns=self.factor_returns.columns)
        
        for date in portfolio_returns.index:
            if date in portfolio_weights.index and date in self.factor_returns.index:
                weights = portfolio_weights.loc[date].dropna()
                factor_rets = self.factor_returns.loc[date]
                
                # Get factor exposures for this date
                try:
                    exposures = self.get_factor_exposures(weights)
                    
                    # Factor contribution = exposure * factor return
                    for factor in factor_rets.index:
                        if factor in exposures.index:
                            attribution.loc[date, factor] = exposures[factor] * factor_rets[factor]
                except Exception:
                    continue
        
        return attribution.dropna(how='all')
