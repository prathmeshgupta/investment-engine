"""Position sizing algorithms for risk management."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class PositionSizingMethod(Enum):
    """Position sizing methods."""
    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_TARGET = "volatility_target"
    RISK_PARITY = "risk_parity"
    MAX_LOSS = "max_loss"


class PositionSizer:
    """Position sizing engine for risk-based allocation."""
    
    def __init__(self):
        """Initialize position sizer."""
        self.sizing_methods = {
            PositionSizingMethod.FIXED_FRACTIONAL: self._fixed_fractional_sizing,
            PositionSizingMethod.KELLY_CRITERION: self._kelly_criterion_sizing,
            PositionSizingMethod.VOLATILITY_TARGET: self._volatility_target_sizing,
            PositionSizingMethod.RISK_PARITY: self._risk_parity_sizing,
            PositionSizingMethod.MAX_LOSS: self._max_loss_sizing
        }
    
    def kelly_criterion(self, expected_returns: pd.Series, volatilities: pd.Series, 
                       correlations: pd.DataFrame) -> pd.Series:
        """Calculate Kelly criterion position sizes."""
        return self._kelly_criterion_sizing(expected_returns, volatilities, correlations)
    
    def calculate_position_sizes(self,
                               expected_returns: pd.Series,
                               volatilities: pd.Series,
                               correlations: pd.DataFrame,
                               method: PositionSizingMethod = PositionSizingMethod.VOLATILITY_TARGET,
                               portfolio_value: float = 1000000,
                               **kwargs) -> pd.Series:
        """Calculate position sizes using specified method.
        
        Args:
            expected_returns: Expected returns for assets
            volatilities: Asset volatilities
            correlations: Correlation matrix
            method: Position sizing method
            portfolio_value: Total portfolio value
            **kwargs: Method-specific parameters
            
        Returns:
            Position sizes (dollar amounts)
        """
        if method not in self.sizing_methods:
            raise ValueError(f"Unsupported method: {method}")
        
        # Align data
        common_assets = expected_returns.index.intersection(volatilities.index)
        if correlations is not None:
            common_assets = common_assets.intersection(correlations.index)
        
        expected_returns_aligned = expected_returns.reindex(common_assets)
        volatilities_aligned = volatilities.reindex(common_assets)
        correlations_aligned = correlations.reindex(common_assets, columns=common_assets) if correlations is not None else None
        
        # Calculate weights using specified method
        weights = self.sizing_methods[method](
            expected_returns_aligned,
            volatilities_aligned,
            correlations_aligned,
            **kwargs
        )
        
        # Convert to dollar positions
        position_sizes = weights * portfolio_value
        
        return position_sizes
    
    def _fixed_fractional_sizing(self,
                               expected_returns: pd.Series,
                               volatilities: pd.Series,
                               correlations: Optional[pd.DataFrame],
                               fraction: float = 0.02) -> pd.Series:
        """Fixed fractional position sizing.
        
        Args:
            fraction: Fixed fraction of portfolio per position
        """
        n_assets = len(expected_returns)
        weights = pd.Series(fraction, index=expected_returns.index)
        
        # Normalize if total exceeds 1
        total_weight = weights.sum()
        if total_weight > 1:
            weights = weights / total_weight
        
        return weights
    
    def _kelly_criterion_sizing(self,
                              expected_returns: pd.Series,
                              volatilities: pd.Series,
                              correlations: Optional[pd.DataFrame],
                              risk_free_rate: float = 0.02) -> pd.Series:
        """Kelly criterion position sizing.
        
        Args:
            risk_free_rate: Risk-free rate for Kelly calculation
        """
        # Kelly fraction = (expected_return - risk_free_rate) / variance
        excess_returns = expected_returns - risk_free_rate / 252  # Daily risk-free rate
        variances = volatilities ** 2
        
        kelly_fractions = excess_returns / variances
        
        # Cap Kelly fractions to prevent over-leverage
        max_kelly = 0.25  # Maximum 25% per asset
        kelly_fractions = kelly_fractions.clip(0, max_kelly)
        
        # Normalize weights
        total_weight = kelly_fractions.sum()
        if total_weight > 1:
            weights = kelly_fractions / total_weight
        else:
            weights = kelly_fractions
        
        return weights
    
    def _volatility_target_sizing(self,
                                expected_returns: pd.Series,
                                volatilities: pd.Series,
                                correlations: Optional[pd.DataFrame],
                                target_volatility: float = 0.1) -> pd.Series:
        """Volatility target position sizing.
        
        Args:
            target_volatility: Target portfolio volatility
        """
        # Inverse volatility weights
        inv_vol_weights = (1 / volatilities) / (1 / volatilities).sum()
        
        if correlations is not None:
            # Adjust for correlations
            cov_matrix = np.outer(volatilities, volatilities) * correlations.values
            
            # Portfolio volatility with current weights
            portfolio_vol = np.sqrt(inv_vol_weights.values.T @ cov_matrix @ inv_vol_weights.values)
            
            # Scale weights to achieve target volatility
            scaling_factor = target_volatility / portfolio_vol
            weights = inv_vol_weights * scaling_factor
        else:
            # Simple inverse volatility without correlation adjustment
            weights = inv_vol_weights
        
        # Ensure weights don't exceed 1
        weights = weights.clip(0, 1)
        total_weight = weights.sum()
        if total_weight > 1:
            weights = weights / total_weight
        
        return weights
    
    def _risk_parity_sizing(self,
                          expected_returns: pd.Series,
                          volatilities: pd.Series,
                          correlations: Optional[pd.DataFrame],
                          **kwargs) -> pd.Series:
        """Risk parity position sizing."""
        if correlations is None:
            # Simple inverse volatility
            weights = (1 / volatilities) / (1 / volatilities).sum()
        else:
            # Use optimization for true risk parity
            from scipy.optimize import minimize
            
            n_assets = len(expected_returns)
            
            # Covariance matrix
            cov_matrix = np.outer(volatilities, volatilities) * correlations.values
            
            def risk_parity_objective(weights):
                portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
                marginal_contrib = cov_matrix @ weights / portfolio_vol
                risk_contrib = weights * marginal_contrib / portfolio_vol
                target_contrib = 1 / n_assets
                return np.sum((risk_contrib - target_contrib) ** 2)
            
            # Constraints and bounds
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            bounds = [(0.001, 0.5) for _ in range(n_assets)]
            
            # Initial guess
            x0 = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(risk_parity_objective, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = pd.Series(result.x, index=expected_returns.index)
            else:
                # Fallback to inverse volatility
                weights = (1 / volatilities) / (1 / volatilities).sum()
        
        return weights
    
    def _max_loss_sizing(self,
                       expected_returns: pd.Series,
                       volatilities: pd.Series,
                       correlations: Optional[pd.DataFrame],
                       max_loss_per_position: float = 0.02,
                       confidence_level: float = 0.95) -> pd.Series:
        """Maximum loss position sizing.
        
        Args:
            max_loss_per_position: Maximum loss per position as fraction of portfolio
            confidence_level: Confidence level for loss calculation
        """
        from scipy import stats
        
        # Calculate VaR for each asset
        z_score = stats.norm.ppf(1 - confidence_level)
        asset_vars = volatilities * abs(z_score)  # VaR as positive number
        
        # Position size = max_loss / VaR
        position_fractions = max_loss_per_position / asset_vars
        
        # Cap maximum position size
        max_position = 0.2  # Maximum 20% per asset
        position_fractions = position_fractions.clip(0, max_position)
        
        # Normalize if total exceeds 1
        total_weight = position_fractions.sum()
        if total_weight > 1:
            weights = position_fractions / total_weight
        else:
            weights = position_fractions
        
        return weights
    
    def calculate_optimal_leverage(self,
                                 expected_returns: pd.Series,
                                 volatilities: pd.Series,
                                 correlations: pd.DataFrame,
                                 risk_free_rate: float = 0.02,
                                 max_leverage: float = 2.0) -> float:
        """Calculate optimal portfolio leverage.
        
        Args:
            expected_returns: Expected returns
            volatilities: Asset volatilities
            correlations: Correlation matrix
            risk_free_rate: Risk-free rate
            max_leverage: Maximum allowed leverage
            
        Returns:
            Optimal leverage ratio
        """
        # Create covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlations.values
        
        # Equal weight portfolio for simplicity
        n_assets = len(expected_returns)
        weights = np.ones(n_assets) / n_assets
        
        # Portfolio metrics
        portfolio_return = weights @ expected_returns.values
        portfolio_variance = weights.T @ cov_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # Optimal leverage using Kelly criterion for portfolio
        excess_return = portfolio_return - risk_free_rate / 252
        optimal_leverage = excess_return / portfolio_variance
        
        # Cap leverage
        optimal_leverage = min(optimal_leverage, max_leverage)
        optimal_leverage = max(optimal_leverage, 0.1)  # Minimum 10% allocation
        
        return optimal_leverage
    
    def stress_test_positions(self,
                            position_sizes: pd.Series,
                            stress_scenarios: Dict[str, pd.Series],
                            portfolio_value: float) -> Dict[str, Dict[str, float]]:
        """Stress test position sizes under different scenarios.
        
        Args:
            position_sizes: Current position sizes
            stress_scenarios: Dictionary of stress scenarios {name: returns}
            portfolio_value: Total portfolio value
            
        Returns:
            Stress test results
        """
        results = {}
        
        # Convert position sizes to weights
        weights = position_sizes / portfolio_value
        
        for scenario_name, scenario_returns in stress_scenarios.items():
            # Align data
            common_assets = weights.index.intersection(scenario_returns.index)
            weights_aligned = weights.reindex(common_assets, fill_value=0)
            returns_aligned = scenario_returns.reindex(common_assets, fill_value=0)
            
            # Calculate portfolio return under stress
            portfolio_return = (weights_aligned * returns_aligned).sum()
            portfolio_loss = portfolio_value * portfolio_return
            
            # Individual position losses
            position_losses = position_sizes.reindex(common_assets, fill_value=0) * returns_aligned
            
            results[scenario_name] = {
                'portfolio_return': portfolio_return,
                'portfolio_loss': portfolio_loss,
                'max_position_loss': position_losses.min(),
                'worst_asset': position_losses.idxmin(),
                'positions_at_risk': (position_losses < -0.05 * portfolio_value).sum()
            }
        
        return results
