"""Main optimization engine coordinating different optimization methods."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from .mean_variance import MeanVarianceOptimizer
from .risk_parity import RiskParityOptimizer
from .black_litterman import BlackLittermanOptimizer
from core.enums import OptimizationMethod


class OptimizerEngine:
    """Main optimization engine that coordinates different optimization methods."""
    
    def __init__(self):
        """Initialize optimizer engine."""
        self.optimizers = {
            OptimizationMethod.MEAN_VARIANCE: MeanVarianceOptimizer(),
            OptimizationMethod.RISK_PARITY: RiskParityOptimizer(),
            OptimizationMethod.BLACK_LITTERMAN: BlackLittermanOptimizer(),
            OptimizationMethod.MINIMUM_VARIANCE: MeanVarianceOptimizer(),
            OptimizationMethod.EQUAL_WEIGHT: None  # Special case
        }
        
        self.current_method = None
        self.optimal_weights = None
        self.optimization_results = {}
    
    def optimize_portfolio(self,
                          method: OptimizationMethod,
                          expected_returns: Optional[pd.Series] = None,
                          covariance_matrix: Optional[pd.DataFrame] = None,
                          market_caps: Optional[pd.Series] = None,
                          constraints: Optional[Dict[str, Any]] = None,
                          views: Optional[Dict[str, Any]] = None) -> pd.Series:
        """Optimize portfolio using specified method.
        
        Args:
            method: Optimization method to use
            expected_returns: Expected returns for assets
            covariance_matrix: Covariance matrix of returns
            market_caps: Market capitalizations (for Black-Litterman)
            constraints: Optimization constraints
            views: Views for Black-Litterman
            
        Returns:
            Optimal portfolio weights
        """
        self.current_method = method
        
        # Default constraints
        if constraints is None:
            constraints = {}
        
        # Handle equal weight case
        if method == OptimizationMethod.EQUAL_WEIGHT:
            return self._equal_weight_portfolio(covariance_matrix.index if covariance_matrix is not None 
                                              else expected_returns.index)
        
        # Get optimizer
        optimizer = self.optimizers[method]
        
        # Set inputs based on method
        if method in [OptimizationMethod.MEAN_VARIANCE, OptimizationMethod.MINIMUM_VARIANCE]:
            if expected_returns is None or covariance_matrix is None:
                raise ValueError("Mean-variance optimization requires expected returns and covariance matrix")
            
            optimizer.set_inputs(expected_returns, covariance_matrix)
            
            if method == OptimizationMethod.MINIMUM_VARIANCE:
                # For minimum variance, set target return to minimum
                min_return = expected_returns.min()
                optimal_weights = optimizer.optimize_portfolio(
                    target_return=min_return,
                    min_weights=constraints.get('min_weights'),
                    max_weights=constraints.get('max_weights'),
                    max_turnover=constraints.get('max_turnover'),
                    current_weights=constraints.get('current_weights'),
                    transaction_costs=constraints.get('transaction_costs')
                )
            else:
                optimal_weights = optimizer.optimize_portfolio(
                    target_return=constraints.get('target_return'),
                    target_risk=constraints.get('target_risk'),
                    min_weights=constraints.get('min_weights'),
                    max_weights=constraints.get('max_weights'),
                    max_turnover=constraints.get('max_turnover'),
                    current_weights=constraints.get('current_weights'),
                    transaction_costs=constraints.get('transaction_costs')
                )
        
        elif method == OptimizationMethod.RISK_PARITY:
            if covariance_matrix is None:
                raise ValueError("Risk parity optimization requires covariance matrix")
            
            optimizer.set_inputs(covariance_matrix)
            optimal_weights = optimizer.optimize_portfolio(
                target_risk_contributions=constraints.get('target_risk_contributions'),
                min_weights=constraints.get('min_weights'),
                max_weights=constraints.get('max_weights')
            )
        
        elif method == OptimizationMethod.BLACK_LITTERMAN:
            if covariance_matrix is None or market_caps is None:
                raise ValueError("Black-Litterman optimization requires covariance matrix and market caps")
            
            optimizer.set_inputs(covariance_matrix, market_caps)
            
            # Handle views
            views_matrix = None
            views_returns = None
            views_uncertainty = None
            
            if views:
                if 'asset_views' in views or 'relative_views' in views:
                    views_matrix, views_returns = optimizer.create_view(
                        views.get('asset_views', {}),
                        views.get('relative_views', {})
                    )
                    
                    if 'confidence_levels' in views:
                        views_uncertainty = optimizer.calculate_view_uncertainty(
                            views_matrix, views['confidence_levels']
                        )
            
            optimal_weights = optimizer.optimize_portfolio(
                views_matrix=views_matrix,
                views_returns=views_returns,
                views_uncertainty=views_uncertainty,
                tau=constraints.get('tau', 0.025),
                min_weights=constraints.get('min_weights'),
                max_weights=constraints.get('max_weights')
            )
        
        else:
            raise ValueError(f"Unsupported optimization method: {method}")
        
        self.optimal_weights = optimal_weights
        
        # Store optimization results
        self.optimization_results = {
            'method': method,
            'weights': optimal_weights,
            'timestamp': datetime.now(),
            'constraints': constraints
        }
        
        return optimal_weights
    
    def _equal_weight_portfolio(self, assets: pd.Index) -> pd.Series:
        """Create equal weight portfolio."""
        n_assets = len(assets)
        equal_weights = pd.Series(1.0 / n_assets, index=assets)
        
        self.optimal_weights = equal_weights
        return equal_weights
    
    def compare_methods(self,
                       methods: List[OptimizationMethod],
                       expected_returns: pd.Series,
                       covariance_matrix: pd.DataFrame,
                       market_caps: Optional[pd.Series] = None,
                       constraints: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Compare different optimization methods.
        
        Args:
            methods: List of optimization methods to compare
            expected_returns: Expected returns for assets
            covariance_matrix: Covariance matrix of returns
            market_caps: Market capitalizations
            constraints: Optimization constraints
            
        Returns:
            Comparison DataFrame with portfolio metrics
        """
        results = []
        
        for method in methods:
            try:
                weights = self.optimize_portfolio(
                    method=method,
                    expected_returns=expected_returns,
                    covariance_matrix=covariance_matrix,
                    market_caps=market_caps,
                    constraints=constraints
                )
                
                # Calculate portfolio metrics
                portfolio_return = (weights * expected_returns).sum()
                portfolio_risk = np.sqrt(weights.T @ covariance_matrix @ weights)
                sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
                
                # Concentration metrics
                herfindahl_index = (weights ** 2).sum()
                effective_assets = 1 / herfindahl_index
                max_weight = weights.max()
                
                results.append({
                    'method': method.value,
                    'expected_return': portfolio_return,
                    'volatility': portfolio_risk,
                    'sharpe_ratio': sharpe_ratio,
                    'max_weight': max_weight,
                    'effective_assets': effective_assets,
                    'herfindahl_index': herfindahl_index
                })
                
            except Exception as e:
                print(f"Error optimizing with {method.value}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def calculate_portfolio_metrics(self, 
                                  weights: pd.Series,
                                  expected_returns: pd.Series,
                                  covariance_matrix: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics.
        
        Args:
            weights: Portfolio weights
            expected_returns: Expected returns
            covariance_matrix: Covariance matrix
            
        Returns:
            Dictionary of portfolio metrics
        """
        # Align data
        common_assets = weights.index.intersection(expected_returns.index).intersection(covariance_matrix.index)
        
        w = weights.reindex(common_assets, fill_value=0)
        mu = expected_returns.reindex(common_assets)
        sigma = covariance_matrix.reindex(common_assets, columns=common_assets)
        
        # Basic metrics
        portfolio_return = (w * mu).sum()
        portfolio_variance = w.T @ sigma @ w
        portfolio_risk = np.sqrt(portfolio_variance)
        
        # Risk-adjusted metrics
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        # Concentration metrics
        herfindahl_index = (w ** 2).sum()
        effective_assets = 1 / herfindahl_index if herfindahl_index > 0 else 0
        max_weight = w.max()
        min_weight = w[w > 0].min() if (w > 0).any() else 0
        
        # Diversification ratio
        weighted_avg_vol = (w * np.sqrt(np.diag(sigma))).sum()
        diversification_ratio = weighted_avg_vol / portfolio_risk if portfolio_risk > 0 else 0
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_risk,
            'variance': portfolio_variance,
            'sharpe_ratio': sharpe_ratio,
            'max_weight': max_weight,
            'min_weight': min_weight,
            'effective_assets': effective_assets,
            'herfindahl_index': herfindahl_index,
            'diversification_ratio': diversification_ratio,
            'number_of_positions': (w > 1e-6).sum()
        }
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get history of optimization results.
        
        Returns:
            List of optimization results
        """
        return [self.optimization_results] if self.optimization_results else []
    
    def rebalance_portfolio(self,
                          current_weights: pd.Series,
                          target_weights: pd.Series,
                          rebalance_threshold: float = 0.05,
                          transaction_costs: Optional[Dict[str, float]] = None) -> Tuple[pd.Series, Dict[str, Any]]:
        """Calculate rebalancing trades.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            rebalance_threshold: Minimum weight change to trigger rebalancing
            transaction_costs: Transaction costs per asset
            
        Returns:
            Tuple of (new_weights, rebalancing_info)
        """
        # Align weights
        all_assets = current_weights.index.union(target_weights.index)
        current_aligned = current_weights.reindex(all_assets, fill_value=0)
        target_aligned = target_weights.reindex(all_assets, fill_value=0)
        
        # Calculate weight changes
        weight_changes = target_aligned - current_aligned
        
        # Apply rebalancing threshold
        significant_changes = np.abs(weight_changes) >= rebalance_threshold
        
        if not significant_changes.any():
            # No rebalancing needed
            return current_aligned, {
                'rebalancing_needed': False,
                'total_turnover': 0,
                'transaction_costs': 0,
                'trades': pd.Series(dtype=float)
            }
        
        # Calculate trades
        trades = weight_changes.copy()
        trades[~significant_changes] = 0
        
        # Calculate transaction costs
        total_tc = 0
        if transaction_costs:
            for asset, trade in trades.items():
                if asset in transaction_costs and abs(trade) > 0:
                    total_tc += abs(trade) * transaction_costs[asset]
        
        # New weights after rebalancing
        new_weights = current_aligned + trades
        new_weights = new_weights / new_weights.sum()  # Renormalize
        
        # Calculate turnover
        total_turnover = np.abs(trades).sum()
        
        rebalancing_info = {
            'rebalancing_needed': True,
            'total_turnover': total_turnover,
            'transaction_costs': total_tc,
            'trades': trades[trades != 0],
            'assets_rebalanced': significant_changes.sum()
        }
        
        return new_weights, rebalancing_info
