"""Mean-variance portfolio optimization."""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import minimize
from core.enums import OptimizationMethod


class MeanVarianceOptimizer:
    """Mean-variance portfolio optimization using Modern Portfolio Theory."""
    
    def __init__(self):
        """Initialize mean-variance optimizer."""
        self.expected_returns = None
        self.covariance_matrix = None
        self.optimal_weights = None
        
    def set_inputs(self, 
                   expected_returns: pd.Series,
                   covariance_matrix: pd.DataFrame) -> None:
        """Set optimization inputs.
        
        Args:
            expected_returns: Expected returns for assets
            covariance_matrix: Covariance matrix of returns
        """
        # Validate inputs
        if not expected_returns.index.equals(covariance_matrix.index):
            raise ValueError("Expected returns and covariance matrix indices must match")
        
        if not covariance_matrix.index.equals(covariance_matrix.columns):
            raise ValueError("Covariance matrix must be square")
        
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
    
    def optimize_portfolio(self,
                          target_return: Optional[float] = None,
                          target_risk: Optional[float] = None,
                          min_weights: Optional[Dict[str, float]] = None,
                          max_weights: Optional[Dict[str, float]] = None,
                          max_turnover: Optional[float] = None,
                          current_weights: Optional[pd.Series] = None,
                          transaction_costs: Optional[Dict[str, float]] = None) -> pd.Series:
        """Optimize portfolio using mean-variance optimization.
        
        Args:
            target_return: Target portfolio return
            target_risk: Target portfolio risk (volatility)
            min_weights: Minimum weights for assets
            max_weights: Maximum weights for assets
            max_turnover: Maximum portfolio turnover
            current_weights: Current portfolio weights
            transaction_costs: Transaction costs per asset
            
        Returns:
            Optimal portfolio weights
        """
        if self.expected_returns is None or self.covariance_matrix is None:
            raise ValueError("Must set inputs before optimization")
        
        n_assets = len(self.expected_returns)
        assets = self.expected_returns.index
        
        # Define optimization variables
        w = cp.Variable(n_assets)
        
        # Objective function
        portfolio_return = self.expected_returns.values @ w
        portfolio_risk = cp.quad_form(w, self.covariance_matrix.values)
        
        # Base constraints
        constraints = [cp.sum(w) == 1]  # Weights sum to 1
        
        # Weight bounds
        if min_weights is not None:
            for i, asset in enumerate(assets):
                if asset in min_weights:
                    constraints.append(w[i] >= min_weights[asset])
        else:
            constraints.append(w >= 0)  # Long-only by default
        
        if max_weights is not None:
            for i, asset in enumerate(assets):
                if asset in max_weights:
                    constraints.append(w[i] <= max_weights[asset])
        
        # Turnover constraint
        if max_turnover is not None and current_weights is not None:
            current_w = np.zeros(n_assets)
            for i, asset in enumerate(assets):
                if asset in current_weights.index:
                    current_w[i] = current_weights[asset]
            
            turnover = cp.norm(w - current_w, 1)
            constraints.append(turnover <= max_turnover)
        
        # Target return or risk constraint
        if target_return is not None:
            constraints.append(portfolio_return >= target_return)
            # Minimize risk for given return
            objective = cp.Minimize(portfolio_risk)
        elif target_risk is not None:
            constraints.append(cp.sqrt(portfolio_risk) <= target_risk)
            # Maximize return for given risk
            objective = cp.Maximize(portfolio_return)
        else:
            # Default: maximize Sharpe ratio (approximate)
            # Use risk aversion parameter
            risk_aversion = 1.0
            objective = cp.Maximize(portfolio_return - 0.5 * risk_aversion * portfolio_risk)
        
        # Add transaction costs if provided
        if transaction_costs is not None and current_weights is not None:
            current_w = np.zeros(n_assets)
            tc_vector = np.zeros(n_assets)
            
            for i, asset in enumerate(assets):
                if asset in current_weights.index:
                    current_w[i] = current_weights[asset]
                if asset in transaction_costs:
                    tc_vector[i] = transaction_costs[asset]
            
            # Transaction cost = sum of |weight_change| * cost
            trade_amounts = cp.abs(w - current_w)
            total_tc = tc_vector @ trade_amounts
            
            if target_return is not None:
                objective = cp.Minimize(portfolio_risk + total_tc)
            elif target_risk is not None:
                objective = cp.Maximize(portfolio_return - total_tc)
            else:
                objective = cp.Maximize(portfolio_return - 0.5 * risk_aversion * portfolio_risk - total_tc)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS)
            
            if problem.status not in ["infeasible", "unbounded"]:
                optimal_weights = pd.Series(w.value, index=assets)
                # Clean up small weights
                optimal_weights[np.abs(optimal_weights) < 1e-6] = 0
                # Renormalize
                optimal_weights = optimal_weights / optimal_weights.sum()
                
                self.optimal_weights = optimal_weights
                return optimal_weights
            else:
                raise ValueError(f"Optimization failed with status: {problem.status}")
                
        except Exception as e:
            # Fallback to scipy optimization
            return self._scipy_fallback(target_return, target_risk, min_weights, max_weights)
    
    def _scipy_fallback(self,
                       target_return: Optional[float] = None,
                       target_risk: Optional[float] = None,
                       min_weights: Optional[Dict[str, float]] = None,
                       max_weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """Fallback optimization using scipy."""
        n_assets = len(self.expected_returns)
        assets = self.expected_returns.index
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Bounds
        bounds = []
        for i, asset in enumerate(assets):
            min_w = min_weights.get(asset, 0) if min_weights else 0
            max_w = max_weights.get(asset, 1) if max_weights else 1
            bounds.append((min_w, max_w))
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if target_return is not None:
            constraints.append({
                'type': 'ineq', 
                'fun': lambda x: self.expected_returns.values @ x - target_return
            })
        
        # Objective function
        def objective(x):
            if target_return is not None:
                # Minimize risk
                return x.T @ self.covariance_matrix.values @ x
            else:
                # Maximize Sharpe ratio (minimize negative Sharpe)
                ret = self.expected_returns.values @ x
                risk = np.sqrt(x.T @ self.covariance_matrix.values @ x)
                return -ret / risk if risk > 0 else 1e6
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = pd.Series(result.x, index=assets)
            optimal_weights[np.abs(optimal_weights) < 1e-6] = 0
            optimal_weights = optimal_weights / optimal_weights.sum()
            
            self.optimal_weights = optimal_weights
            return optimal_weights
        else:
            raise ValueError("Optimization failed in scipy fallback")
    
    def calculate_efficient_frontier(self,
                                   n_points: int = 50,
                                   min_weights: Optional[Dict[str, float]] = None,
                                   max_weights: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate efficient frontier.
        
        Args:
            n_points: Number of points on the frontier
            min_weights: Minimum weights for assets
            max_weights: Maximum weights for assets
            
        Returns:
            Tuple of (risks, returns) arrays
        """
        if self.expected_returns is None or self.covariance_matrix is None:
            raise ValueError("Must set inputs before calculating efficient frontier")
        
        # Calculate range of target returns
        min_return = self.expected_returns.min()
        max_return = self.expected_returns.max()
        target_returns = np.linspace(min_return, max_return, n_points)
        
        risks = []
        returns = []
        
        for target_ret in target_returns:
            try:
                weights = self.optimize_portfolio(
                    target_return=target_ret,
                    min_weights=min_weights,
                    max_weights=max_weights
                )
                
                portfolio_return = (weights * self.expected_returns).sum()
                portfolio_risk = np.sqrt(weights.T @ self.covariance_matrix @ weights)
                
                risks.append(portfolio_risk)
                returns.append(portfolio_return)
                
            except Exception:
                continue
        
        return np.array(risks), np.array(returns)
    
    def calculate_portfolio_metrics(self, weights: pd.Series) -> Dict[str, float]:
        """Calculate portfolio metrics for given weights.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Dictionary of portfolio metrics
        """
        if self.expected_returns is None or self.covariance_matrix is None:
            raise ValueError("Must set inputs before calculating metrics")
        
        # Align weights with expected returns
        aligned_weights = weights.reindex(self.expected_returns.index, fill_value=0)
        
        portfolio_return = (aligned_weights * self.expected_returns).sum()
        portfolio_variance = aligned_weights.T @ self.covariance_matrix @ aligned_weights
        portfolio_risk = np.sqrt(portfolio_variance)
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_risk,
            'variance': portfolio_variance,
            'sharpe_ratio': sharpe_ratio
        }
    
    def get_maximum_sharpe_portfolio(self,
                                   min_weights: Optional[Dict[str, float]] = None,
                                   max_weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """Get maximum Sharpe ratio portfolio.
        
        Args:
            min_weights: Minimum weights for assets
            max_weights: Maximum weights for assets
            
        Returns:
            Optimal portfolio weights
        """
        return self.optimize_portfolio(min_weights=min_weights, max_weights=max_weights)
    
    def get_minimum_variance_portfolio(self,
                                     min_weights: Optional[Dict[str, float]] = None,
                                     max_weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """Get minimum variance portfolio.
        
        Args:
            min_weights: Minimum weights for assets
            max_weights: Maximum weights for assets
            
        Returns:
            Minimum variance portfolio weights
        """
        # Set target return to minimum to focus on risk minimization
        min_return = self.expected_returns.min()
        return self.optimize_portfolio(
            target_return=min_return,
            min_weights=min_weights,
            max_weights=max_weights
        )
