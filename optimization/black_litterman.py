"""Black-Litterman portfolio optimization."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import linalg


class BlackLittermanOptimizer:
    """Black-Litterman portfolio optimization."""
    
    def __init__(self, risk_aversion: float = 3.0):
        """Initialize Black-Litterman optimizer.
        
        Args:
            risk_aversion: Risk aversion parameter
        """
        self.risk_aversion = risk_aversion
        self.covariance_matrix = None
        self.market_caps = None
        self.optimal_weights = None
        
    def set_inputs(self, 
                   covariance_matrix: pd.DataFrame,
                   market_caps: pd.Series) -> None:
        """Set optimization inputs.
        
        Args:
            covariance_matrix: Covariance matrix of returns
            market_caps: Market capitalizations for assets
        """
        if not covariance_matrix.index.equals(covariance_matrix.columns):
            raise ValueError("Covariance matrix must be square")
        
        self.covariance_matrix = covariance_matrix
        self.market_caps = market_caps.reindex(covariance_matrix.index, fill_value=0)
    
    def calculate_implied_returns(self) -> pd.Series:
        """Calculate implied equilibrium returns from market capitalization weights.
        
        Returns:
            Implied equilibrium returns
        """
        if self.covariance_matrix is None or self.market_caps is None:
            raise ValueError("Must set inputs before calculating implied returns")
        
        # Market capitalization weights
        w_market = self.market_caps / self.market_caps.sum()
        
        # Implied returns: μ = λ * Σ * w_market
        implied_returns = self.risk_aversion * self.covariance_matrix @ w_market
        
        return implied_returns
    
    def optimize_portfolio(self,
                          views_matrix: Optional[pd.DataFrame] = None,
                          views_returns: Optional[pd.Series] = None,
                          views_uncertainty: Optional[pd.DataFrame] = None,
                          tau: float = 0.025,
                          min_weights: Optional[Dict[str, float]] = None,
                          max_weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """Optimize portfolio using Black-Litterman model.
        
        Args:
            views_matrix: Matrix P linking views to assets
            views_returns: Vector Q of view returns
            views_uncertainty: Uncertainty matrix Ω for views
            tau: Scaling factor for uncertainty of prior
            min_weights: Minimum weights for assets
            max_weights: Maximum weights for assets
            
        Returns:
            Optimal portfolio weights
        """
        if self.covariance_matrix is None or self.market_caps is None:
            raise ValueError("Must set inputs before optimization")
        
        # Calculate implied returns
        mu_implied = self.calculate_implied_returns()
        
        # Prior covariance scaled by tau
        sigma_prior = tau * self.covariance_matrix
        
        if views_matrix is not None and views_returns is not None:
            # Black-Litterman with views
            P = views_matrix.values
            Q = views_returns.values
            
            # Default uncertainty matrix if not provided
            if views_uncertainty is None:
                # Diagonal matrix with variances proportional to view confidence
                omega_diag = np.diag(P @ self.covariance_matrix.values @ P.T) * tau
                Omega = np.diag(omega_diag)
            else:
                Omega = views_uncertainty.values
            
            # Black-Litterman formula
            # μ_BL = [(τΣ)^-1 + P'Ω^-1P]^-1 * [(τΣ)^-1 * μ + P'Ω^-1 * Q]
            sigma_prior_inv = linalg.inv(sigma_prior.values)
            omega_inv = linalg.inv(Omega)
            
            # New expected returns
            M1 = sigma_prior_inv + P.T @ omega_inv @ P
            M2 = sigma_prior_inv @ mu_implied.values + P.T @ omega_inv @ Q
            mu_bl = linalg.solve(M1, M2)
            
            # New covariance matrix
            sigma_bl = linalg.inv(M1)
            
            mu_bl_series = pd.Series(mu_bl, index=self.covariance_matrix.index)
            sigma_bl_df = pd.DataFrame(sigma_bl, 
                                     index=self.covariance_matrix.index,
                                     columns=self.covariance_matrix.columns)
        else:
            # No views - use implied returns
            mu_bl_series = mu_implied
            sigma_bl_df = self.covariance_matrix
        
        # Optimize portfolio with Black-Litterman inputs
        from .mean_variance import MeanVarianceOptimizer
        
        mv_optimizer = MeanVarianceOptimizer()
        mv_optimizer.set_inputs(mu_bl_series, sigma_bl_df)
        
        optimal_weights = mv_optimizer.optimize_portfolio(
            min_weights=min_weights,
            max_weights=max_weights
        )
        
        self.optimal_weights = optimal_weights
        return optimal_weights
    
    def create_view(self, 
                   asset_views: Dict[str, float],
                   relative_views: Optional[Dict[Tuple[str, str], float]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Create views matrix and returns vector.
        
        Args:
            asset_views: Dictionary of absolute views {asset: expected_return}
            relative_views: Dictionary of relative views {(asset1, asset2): outperformance}
            
        Returns:
            Tuple of (views_matrix, views_returns)
        """
        assets = self.covariance_matrix.index
        views = []
        returns = []
        
        # Absolute views
        for asset, expected_return in asset_views.items():
            if asset in assets:
                view_vector = pd.Series(0.0, index=assets)
                view_vector[asset] = 1.0
                views.append(view_vector)
                returns.append(expected_return)
        
        # Relative views
        if relative_views:
            for (asset1, asset2), outperformance in relative_views.items():
                if asset1 in assets and asset2 in assets:
                    view_vector = pd.Series(0.0, index=assets)
                    view_vector[asset1] = 1.0
                    view_vector[asset2] = -1.0
                    views.append(view_vector)
                    returns.append(outperformance)
        
        if not views:
            raise ValueError("No valid views provided")
        
        views_matrix = pd.DataFrame(views, columns=assets)
        views_returns = pd.Series(returns)
        
        return views_matrix, views_returns
    
    def calculate_view_uncertainty(self, 
                                 views_matrix: pd.DataFrame,
                                 confidence_levels: Optional[List[float]] = None) -> pd.DataFrame:
        """Calculate uncertainty matrix for views.
        
        Args:
            views_matrix: Matrix P linking views to assets
            confidence_levels: Confidence levels for each view (0-1)
            
        Returns:
            Uncertainty matrix Ω
        """
        n_views = len(views_matrix)
        
        if confidence_levels is None:
            confidence_levels = [0.5] * n_views  # Default moderate confidence
        
        # Calculate view variances based on confidence
        P = views_matrix.values
        view_variances = []
        
        for i, confidence in enumerate(confidence_levels):
            # View variance = (1/confidence - 1) * P * Σ * P'
            view_var = (1/confidence - 1) * P[i:i+1] @ self.covariance_matrix.values @ P[i:i+1].T
            view_variances.append(view_var[0, 0])
        
        # Create diagonal uncertainty matrix
        omega = np.diag(view_variances)
        
        return pd.DataFrame(omega)
    
    def get_market_portfolio(self) -> pd.Series:
        """Get market capitalization weighted portfolio.
        
        Returns:
            Market cap weighted portfolio
        """
        if self.market_caps is None:
            raise ValueError("Must set market caps")
        
        market_weights = self.market_caps / self.market_caps.sum()
        return market_weights
