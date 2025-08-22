"""Value at Risk (VaR) and Conditional VaR calculations."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.mixture import GaussianMixture


class VaRCalculator:
    """Value at Risk and Conditional VaR calculator."""
    
    def __init__(self):
        """Initialize VaR calculator."""
        self.methods = ['historical', 'parametric', 'monte_carlo', 'cornish_fisher']
    
    def calculate_var(self,
                     returns: pd.Series,
                     confidence_levels: List[float] = [0.95],
                     method: str = 'historical',
                     lookback_window: Optional[int] = None) -> Dict[str, float]:
        """Calculate Value at Risk using specified method.
        
        Args:
            returns: Return series
            confidence_levels: Confidence levels for VaR calculation
            method: VaR calculation method
            lookback_window: Lookback window for calculation
            
        Returns:
            Dictionary with VaR and CVaR metrics
        """
        if method not in self.methods:
            raise ValueError(f"Method must be one of {self.methods}")
        
        # Use lookback window if specified
        if lookback_window is not None:
            returns = returns.tail(lookback_window)
        
        returns_clean = returns.dropna()
        if len(returns_clean) < 30:
            raise ValueError("Insufficient data for VaR calculation")
        
        results = {}
        
        for confidence_level in confidence_levels:
            alpha = 1 - confidence_level
            
            if method == 'historical':
                var, cvar = self._historical_var(returns_clean, alpha)
            elif method == 'parametric':
                var, cvar = self._parametric_var(returns_clean, alpha)
            elif method == 'monte_carlo':
                var, cvar = self._monte_carlo_var(returns_clean, alpha)
            elif method == 'cornish_fisher':
                var, cvar = self._cornish_fisher_var(returns_clean, alpha)
            
            # Store results
            conf_str = f"{int(confidence_level * 100)}"
            results[f'var_{conf_str}'] = abs(var)  # VaR as positive number
            results[f'cvar_{conf_str}'] = abs(cvar)  # CVaR as positive number
        
        return results
    
    def _historical_var(self, returns: pd.Series, alpha: float) -> Tuple[float, float]:
        """Calculate historical VaR and CVaR."""
        sorted_returns = returns.sort_values()
        
        # VaR is the alpha-quantile
        var_index = int(alpha * len(sorted_returns))
        var = sorted_returns.iloc[var_index]
        
        # CVaR is the mean of returns below VaR
        cvar = sorted_returns.iloc[:var_index].mean()
        
        return var, cvar
    
    def _parametric_var(self, returns: pd.Series, alpha: float) -> Tuple[float, float]:
        """Calculate parametric VaR and CVaR assuming normal distribution."""
        mu = returns.mean()
        sigma = returns.std()
        
        # VaR using normal distribution
        var = mu + sigma * stats.norm.ppf(alpha)
        
        # CVaR for normal distribution
        phi = stats.norm.pdf(stats.norm.ppf(alpha))
        cvar = mu - sigma * phi / alpha
        
        return var, cvar
    
    def _monte_carlo_var(self, returns: pd.Series, alpha: float, n_simulations: int = 10000) -> Tuple[float, float]:
        """Calculate Monte Carlo VaR and CVaR."""
        mu = returns.mean()
        sigma = returns.std()
        
        # Generate random returns
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(mu, sigma, n_simulations)
        
        # Calculate VaR and CVaR from simulated returns
        sorted_sim = np.sort(simulated_returns)
        var_index = int(alpha * n_simulations)
        
        var = sorted_sim[var_index]
        cvar = sorted_sim[:var_index].mean()
        
        return var, cvar
    
    def _cornish_fisher_var(self, returns: pd.Series, alpha: float) -> Tuple[float, float]:
        """Calculate VaR using Cornish-Fisher expansion for non-normal distributions."""
        mu = returns.mean()
        sigma = returns.std()
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Standard normal quantile
        z_alpha = stats.norm.ppf(alpha)
        
        # Cornish-Fisher adjustment
        cf_adjustment = (z_alpha**2 - 1) * skewness / 6 + \
                       (z_alpha**3 - 3*z_alpha) * kurtosis / 24 - \
                       (2*z_alpha**3 - 5*z_alpha) * skewness**2 / 36
        
        adjusted_quantile = z_alpha + cf_adjustment
        
        # VaR with Cornish-Fisher adjustment
        var = mu + sigma * adjusted_quantile
        
        # For CVaR, use historical method as approximation
        _, cvar = self._historical_var(returns, alpha)
        
        return var, cvar
    
    def calculate_component_var(self,
                              returns: pd.DataFrame,
                              weights: pd.Series,
                              confidence_level: float = 0.95) -> pd.Series:
        """Calculate component VaR for portfolio assets.
        
        Args:
            returns: Asset returns DataFrame
            weights: Portfolio weights
            confidence_level: Confidence level for VaR
            
        Returns:
            Component VaR for each asset
        """
        # Align data
        common_assets = returns.columns.intersection(weights.index)
        returns_aligned = returns[common_assets]
        weights_aligned = weights.reindex(common_assets, fill_value=0)
        
        # Portfolio returns
        portfolio_returns = (returns_aligned * weights_aligned).sum(axis=1)
        
        # Portfolio VaR
        portfolio_var = self.calculate_var(portfolio_returns, [confidence_level])
        portfolio_var_value = portfolio_var[f'var_{int(confidence_level * 100)}']
        
        # Calculate marginal VaR for each asset
        component_vars = pd.Series(index=common_assets, dtype=float)
        
        for asset in common_assets:
            # Calculate correlation with portfolio
            asset_returns = returns_aligned[asset]
            correlation = asset_returns.corr(portfolio_returns)
            
            # Asset volatility
            asset_vol = asset_returns.std()
            
            # Portfolio volatility
            portfolio_vol = portfolio_returns.std()
            
            # Marginal VaR = correlation * asset_vol / portfolio_vol * portfolio_VaR
            if portfolio_vol > 0:
                marginal_var = correlation * asset_vol / portfolio_vol * portfolio_var_value
                component_var = weights_aligned[asset] * marginal_var
            else:
                component_var = 0
            
            component_vars[asset] = component_var
        
        return component_vars
    
    def calculate_incremental_var(self,
                                returns: pd.DataFrame,
                                current_weights: pd.Series,
                                new_weights: pd.Series,
                                confidence_level: float = 0.95) -> float:
        """Calculate incremental VaR from portfolio change.
        
        Args:
            returns: Asset returns DataFrame
            current_weights: Current portfolio weights
            new_weights: New portfolio weights
            confidence_level: Confidence level for VaR
            
        Returns:
            Incremental VaR
        """
        # Calculate VaR for current portfolio
        common_assets = returns.columns.intersection(current_weights.index)
        returns_aligned = returns[common_assets]
        current_weights_aligned = current_weights.reindex(common_assets, fill_value=0)
        
        current_portfolio_returns = (returns_aligned * current_weights_aligned).sum(axis=1)
        current_var = self.calculate_var(current_portfolio_returns, [confidence_level])
        current_var_value = current_var[f'var_{int(confidence_level * 100)}']
        
        # Calculate VaR for new portfolio
        new_weights_aligned = new_weights.reindex(common_assets, fill_value=0)
        new_portfolio_returns = (returns_aligned * new_weights_aligned).sum(axis=1)
        new_var = self.calculate_var(new_portfolio_returns, [confidence_level])
        new_var_value = new_var[f'var_{int(confidence_level * 100)}']
        
        # Incremental VaR
        incremental_var = new_var_value - current_var_value
        
        return incremental_var
    
    def backtest_var(self,
                    returns: pd.Series,
                    var_estimates: pd.Series,
                    confidence_level: float = 0.95) -> Dict[str, float]:
        """Backtest VaR model performance.
        
        Args:
            returns: Actual returns
            var_estimates: VaR estimates
            confidence_level: Confidence level used for VaR
            
        Returns:
            Backtesting statistics
        """
        # Align data
        common_dates = returns.index.intersection(var_estimates.index)
        actual_returns = returns.loc[common_dates]
        var_values = var_estimates.loc[common_dates]
        
        # VaR violations (returns worse than VaR)
        violations = actual_returns < -var_values
        violation_rate = violations.mean()
        expected_violation_rate = 1 - confidence_level
        
        # Number of violations
        n_violations = violations.sum()
        n_observations = len(violations)
        
        # Kupiec test (likelihood ratio test)
        if n_violations > 0:
            likelihood_ratio = -2 * np.log(
                (expected_violation_rate ** n_violations) * 
                ((1 - expected_violation_rate) ** (n_observations - n_violations))
            ) + 2 * np.log(
                (violation_rate ** n_violations) * 
                ((1 - violation_rate) ** (n_observations - n_violations))
            )
            kupiec_p_value = 1 - stats.chi2.cdf(likelihood_ratio, 1)
        else:
            kupiec_p_value = 0
        
        return {
            'violation_rate': violation_rate,
            'expected_violation_rate': expected_violation_rate,
            'n_violations': n_violations,
            'n_observations': n_observations,
            'kupiec_test_statistic': likelihood_ratio if n_violations > 0 else 0,
            'kupiec_p_value': kupiec_p_value,
            'model_accurate': abs(violation_rate - expected_violation_rate) < 0.01
        }
