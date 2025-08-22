"""Risk parity portfolio optimization."""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize


class RiskParityOptimizer:
    """Risk parity portfolio optimization."""
    
    def __init__(self):
        """Initialize risk parity optimizer."""
        self.covariance_matrix = None
        self.optimal_weights = None
    
    def set_inputs(self, covariance_matrix: pd.DataFrame) -> None:
        """Set optimization inputs.
        
        Args:
            covariance_matrix: Covariance matrix of returns
        """
        if not covariance_matrix.index.equals(covariance_matrix.columns):
            raise ValueError("Covariance matrix must be square")
        
        self.covariance_matrix = covariance_matrix
    
    def optimize_portfolio(self,
                          target_risk_contributions: Optional[pd.Series] = None,
                          min_weights: Optional[Dict[str, float]] = None,
                          max_weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """Optimize portfolio using risk parity approach.
        
        Args:
            target_risk_contributions: Target risk contributions (default: equal)
            min_weights: Minimum weights for assets
            max_weights: Maximum weights for assets
            
        Returns:
            Optimal portfolio weights
        """
        if self.covariance_matrix is None:
            raise ValueError("Must set covariance matrix before optimization")
        
        n_assets = len(self.covariance_matrix)
        assets = self.covariance_matrix.index
        
        # Default to equal risk contributions
        if target_risk_contributions is None:
            target_risk_contributions = pd.Series(1.0 / n_assets, index=assets)
        
        # Initial guess (inverse volatility weights)
        volatilities = np.sqrt(np.diag(self.covariance_matrix.values))
        inv_vol_weights = (1 / volatilities) / (1 / volatilities).sum()
        x0 = inv_vol_weights
        
        # Bounds
        bounds = []
        for i, asset in enumerate(assets):
            min_w = min_weights.get(asset, 1e-6) if min_weights else 1e-6
            max_w = max_weights.get(asset, 1.0) if max_weights else 1.0
            bounds.append((min_w, max_w))
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Objective function: minimize sum of squared deviations from target risk contributions
        def objective(weights):
            # Calculate risk contributions
            portfolio_vol = np.sqrt(weights.T @ self.covariance_matrix.values @ weights)
            marginal_contrib = self.covariance_matrix.values @ weights / portfolio_vol
            risk_contrib = weights * marginal_contrib / portfolio_vol
            
            # Target risk contributions
            target_contrib = target_risk_contributions.values
            
            # Sum of squared deviations
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = pd.Series(result.x, index=assets)
            optimal_weights[optimal_weights < 1e-6] = 0
            optimal_weights = optimal_weights / optimal_weights.sum()
            
            self.optimal_weights = optimal_weights
            return optimal_weights
        else:
            # Fallback to inverse volatility
            return self._inverse_volatility_weights(min_weights, max_weights)
    
    def _inverse_volatility_weights(self,
                                  min_weights: Optional[Dict[str, float]] = None,
                                  max_weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """Calculate inverse volatility weights as fallback."""
        assets = self.covariance_matrix.index
        volatilities = np.sqrt(np.diag(self.covariance_matrix.values))
        
        # Inverse volatility weights
        inv_vol = 1 / volatilities
        weights = inv_vol / inv_vol.sum()
        
        # Apply constraints
        weights_series = pd.Series(weights, index=assets)
        
        if min_weights:
            for asset, min_w in min_weights.items():
                if asset in weights_series.index:
                    weights_series[asset] = max(weights_series[asset], min_w)
        
        if max_weights:
            for asset, max_w in max_weights.items():
                if asset in weights_series.index:
                    weights_series[asset] = min(weights_series[asset], max_w)
        
        # Renormalize
        weights_series = weights_series / weights_series.sum()
        
        self.optimal_weights = weights_series
        return weights_series
    
    def calculate_risk_contributions(self, weights: pd.Series) -> pd.Series:
        """Calculate risk contributions for given weights.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Risk contributions for each asset
        """
        if self.covariance_matrix is None:
            raise ValueError("Must set covariance matrix")
        
        # Align weights
        aligned_weights = weights.reindex(self.covariance_matrix.index, fill_value=0)
        w = aligned_weights.values
        
        # Portfolio volatility
        portfolio_var = w.T @ self.covariance_matrix.values @ w
        portfolio_vol = np.sqrt(portfolio_var)
        
        # Marginal risk contributions
        marginal_contrib = self.covariance_matrix.values @ w / portfolio_vol
        
        # Risk contributions
        risk_contrib = w * marginal_contrib / portfolio_vol
        
        return pd.Series(risk_contrib, index=self.covariance_matrix.index)
    
    def get_equal_risk_contribution_portfolio(self,
                                            min_weights: Optional[Dict[str, float]] = None,
                                            max_weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """Get equal risk contribution portfolio.
        
        Args:
            min_weights: Minimum weights for assets
            max_weights: Maximum weights for assets
            
        Returns:
            Equal risk contribution portfolio weights
        """
        return self.optimize_portfolio(min_weights=min_weights, max_weights=max_weights)
    
    def get_inverse_volatility_portfolio(self,
                                       min_weights: Optional[Dict[str, float]] = None,
                                       max_weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """Get inverse volatility weighted portfolio.
        
        Args:
            min_weights: Minimum weights for assets
            max_weights: Maximum weights for assets
            
        Returns:
            Inverse volatility portfolio weights
        """
        return self._inverse_volatility_weights(min_weights, max_weights)
