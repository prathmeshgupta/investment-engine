"""Fama-French factor models implementation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LinearRegression
from core.enums import FactorType


class FamaFrenchModel:
    """Fama-French 3-factor and 5-factor model implementation."""
    
    def __init__(self, model_type: str = "5factor"):
        """Initialize Fama-French model.
        
        Args:
            model_type: "3factor" or "5factor"
        """
        self.model_type = model_type
        self.factors = self._get_factor_list()
        self.factor_returns = None
        self.risk_free_rate = None
        
    def _get_factor_list(self) -> List[FactorType]:
        """Get list of factors for the model."""
        if self.model_type == "3factor":
            return [FactorType.MARKET, FactorType.SIZE, FactorType.VALUE]
        elif self.model_type == "5factor":
            return [
                FactorType.MARKET, FactorType.SIZE, FactorType.VALUE,
                FactorType.PROFITABILITY, FactorType.INVESTMENT
            ]
        else:
            raise ValueError("model_type must be '3factor' or '5factor'")
    
    def calculate_factors(self, 
                         returns: pd.DataFrame,
                         market_caps: pd.DataFrame,
                         book_to_market: Optional[pd.DataFrame] = None,
                         profitability: Optional[pd.DataFrame] = None,
                         investment: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculate Fama-French factors from stock data."""
        return self.construct_factors(returns, market_caps, book_to_market, profitability, investment)
    
    def construct_factors(self, 
                         returns: pd.DataFrame,
                         market_caps: pd.DataFrame,
                         book_to_market: Optional[pd.DataFrame] = None,
                         profitability: Optional[pd.DataFrame] = None,
                         investment: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Construct Fama-French factors from stock data.
        
        Args:
            returns: Stock returns DataFrame (dates x stocks)
            market_caps: Market capitalization DataFrame
            book_to_market: Book-to-market ratio DataFrame
            profitability: Profitability measure DataFrame (for 5-factor)
            investment: Investment measure DataFrame (for 5-factor)
            
        Returns:
            DataFrame with factor returns
        """
        dates = returns.index
        factor_returns = pd.DataFrame(index=dates)
        
        for date in dates:
            # Create synthetic book-to-market ratios if not provided
            if book_to_market is None:
                book_to_market = pd.DataFrame(
                    np.random.uniform(0.5, 2.0, market_caps.shape),
                    index=market_caps.index,
                    columns=market_caps.columns
                )
            
            # Ensure we have valid data
            if returns.empty or market_caps.empty:
                return pd.DataFrame(index=returns.index, columns=['MKT', 'SMB', 'HML'])
            
            # Get data for this date
            ret = returns.loc[date].dropna()
            mcap = market_caps.loc[date].dropna()
            btm = book_to_market.loc[date].dropna()
            
            # Align data
            common_stocks = ret.index.intersection(mcap.index).intersection(btm.index)
            if len(common_stocks) < 10:
                continue
                
            ret_aligned = ret[common_stocks]
            mcap_aligned = mcap[common_stocks]
            btm_aligned = btm[common_stocks]
            
            # Market factor (value-weighted market return)
            market_weights = mcap_aligned / mcap_aligned.sum()
            market_return = (ret_aligned * market_weights).sum()
            factor_returns.loc[date, 'MKT'] = market_return
            
            # Size and value factors
            smb, hml = self._construct_size_value_factors(
                ret_aligned, mcap_aligned, btm_aligned
            )
            factor_returns.loc[date, 'SMB'] = smb
            factor_returns.loc[date, 'HML'] = hml
            
            # 5-factor model additional factors
            if self.model_type == "5factor" and profitability is not None and investment is not None:
                if date in profitability.index and date in investment.index:
                    prof_aligned = profitability.loc[date][common_stocks].dropna()
                    inv_aligned = investment.loc[date][common_stocks].dropna()
                    
                    # Further align data
                    final_stocks = ret_aligned.index.intersection(prof_aligned.index).intersection(inv_aligned.index)
                    if len(final_stocks) >= 10:
                        rmw, cma = self._construct_profitability_investment_factors(
                            ret_aligned[final_stocks],
                            mcap_aligned[final_stocks],
                            btm_aligned[final_stocks],
                            prof_aligned[final_stocks],
                            inv_aligned[final_stocks]
                        )
                        factor_returns.loc[date, 'RMW'] = rmw
                        factor_returns.loc[date, 'CMA'] = cma
        
        self.factor_returns = factor_returns.dropna()
        return self.factor_returns
    
    def _construct_size_value_factors(self,
                                    returns: pd.Series,
                                    market_caps: pd.Series,
                                    book_to_market: pd.Series) -> Tuple[float, float]:
        """Construct SMB and HML factors."""
        # Size breakpoint (median)
        size_median = market_caps.median()
        
        # Value breakpoints (30th and 70th percentiles)
        btm_30 = book_to_market.quantile(0.3)
        btm_70 = book_to_market.quantile(0.7)
        
        # Create portfolios
        portfolios = {}
        
        # Small/Low, Small/Medium, Small/High
        small_low = (market_caps <= size_median) & (book_to_market <= btm_30)
        small_med = (market_caps <= size_median) & (book_to_market > btm_30) & (book_to_market <= btm_70)
        small_high = (market_caps <= size_median) & (book_to_market > btm_70)
        
        # Big/Low, Big/Medium, Big/High
        big_low = (market_caps > size_median) & (book_to_market <= btm_30)
        big_med = (market_caps > size_median) & (book_to_market > btm_30) & (book_to_market <= btm_70)
        big_high = (market_caps > size_median) & (book_to_market > btm_70)
        
        # Calculate value-weighted returns for each portfolio
        def vw_return(mask):
            if mask.sum() == 0:
                return 0
            weights = market_caps[mask] / market_caps[mask].sum()
            return (returns[mask] * weights).sum()
        
        sl_ret = vw_return(small_low)
        sm_ret = vw_return(small_med)
        sh_ret = vw_return(small_high)
        bl_ret = vw_return(big_low)
        bm_ret = vw_return(big_med)
        bh_ret = vw_return(big_high)
        
        # SMB = (Small High + Small Medium + Small Low)/3 - (Big High + Big Medium + Big Low)/3
        smb = (sh_ret + sm_ret + sl_ret) / 3 - (bh_ret + bm_ret + bl_ret) / 3
        
        # HML = (Small High + Big High)/2 - (Small Low + Big Low)/2
        hml = (sh_ret + bh_ret) / 2 - (sl_ret + bl_ret) / 2
        
        return smb, hml
    
    def _construct_profitability_investment_factors(self,
                                                  returns: pd.Series,
                                                  market_caps: pd.Series,
                                                  book_to_market: pd.Series,
                                                  profitability: pd.Series,
                                                  investment: pd.Series) -> Tuple[float, float]:
        """Construct RMW and CMA factors for 5-factor model."""
        # Size breakpoint
        size_median = market_caps.median()
        
        # Profitability breakpoints
        prof_30 = profitability.quantile(0.3)
        prof_70 = profitability.quantile(0.7)
        
        # Investment breakpoints
        inv_30 = investment.quantile(0.3)
        inv_70 = investment.quantile(0.7)
        
        # Create 2x3 portfolios for profitability
        small_weak = (market_caps <= size_median) & (profitability <= prof_30)
        small_neutral_p = (market_caps <= size_median) & (profitability > prof_30) & (profitability <= prof_70)
        small_robust = (market_caps <= size_median) & (profitability > prof_70)
        
        big_weak = (market_caps > size_median) & (profitability <= prof_30)
        big_neutral_p = (market_caps > size_median) & (profitability > prof_30) & (profitability <= prof_70)
        big_robust = (market_caps > size_median) & (profitability > prof_70)
        
        # Create 2x3 portfolios for investment
        small_conservative = (market_caps <= size_median) & (investment <= inv_30)
        small_neutral_i = (market_caps <= size_median) & (investment > inv_30) & (investment <= inv_70)
        small_aggressive = (market_caps <= size_median) & (investment > inv_70)
        
        big_conservative = (market_caps > size_median) & (investment <= inv_30)
        big_neutral_i = (market_caps > size_median) & (investment > inv_30) & (investment <= inv_70)
        big_aggressive = (market_caps > size_median) & (investment > inv_70)
        
        def vw_return(mask):
            if mask.sum() == 0:
                return 0
            weights = market_caps[mask] / market_caps[mask].sum()
            return (returns[mask] * weights).sum()
        
        # RMW = (Small Robust + Big Robust)/2 - (Small Weak + Big Weak)/2
        rmw = (vw_return(small_robust) + vw_return(big_robust)) / 2 - \
              (vw_return(small_weak) + vw_return(big_weak)) / 2
        
        # CMA = (Small Conservative + Big Conservative)/2 - (Small Aggressive + Big Aggressive)/2
        cma = (vw_return(small_conservative) + vw_return(big_conservative)) / 2 - \
              (vw_return(small_aggressive) + vw_return(big_aggressive)) / 2
        
        return rmw, cma
    
    def calculate_factor_loadings(self, 
                                asset_returns: pd.DataFrame,
                                risk_free_rate: pd.Series) -> pd.DataFrame:
        """Calculate factor loadings (betas) for assets.
        
        Args:
            asset_returns: Asset returns DataFrame
            risk_free_rate: Risk-free rate series
            
        Returns:
            DataFrame with factor loadings
        """
        if self.factor_returns is None:
            raise ValueError("Factor returns not constructed. Call construct_factors first.")
        
        # Align dates
        common_dates = asset_returns.index.intersection(self.factor_returns.index).intersection(risk_free_rate.index)
        
        asset_excess = asset_returns.loc[common_dates].subtract(risk_free_rate.loc[common_dates], axis=0)
        factor_excess = self.factor_returns.loc[common_dates].copy()
        
        # Subtract risk-free rate from market factor
        if 'MKT' in factor_excess.columns:
            factor_excess['MKT'] = factor_excess['MKT'] - risk_free_rate.loc[common_dates]
        
        loadings = pd.DataFrame(index=asset_returns.columns, columns=factor_excess.columns)
        
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
            
            # Run regression
            reg = LinearRegression().fit(X_aligned, y_aligned)
            loadings.loc[asset, X_aligned.columns] = reg.coef_
        
        return loadings.astype(float)
    
    def calculate_expected_returns(self, 
                                 factor_loadings: pd.DataFrame,
                                 factor_premiums: pd.Series) -> pd.Series:
        """Calculate expected returns using factor model.
        
        Args:
            factor_loadings: Factor loadings DataFrame
            factor_premiums: Expected factor premiums
            
        Returns:
            Expected returns series
        """
        # Align factors
        common_factors = factor_loadings.columns.intersection(factor_premiums.index)
        
        expected_returns = pd.Series(index=factor_loadings.index, dtype=float)
        
        for asset in factor_loadings.index:
            loadings = factor_loadings.loc[asset, common_factors].dropna()
            premiums = factor_premiums.loc[loadings.index]
            expected_returns[asset] = (loadings * premiums).sum()
        
        return expected_returns
