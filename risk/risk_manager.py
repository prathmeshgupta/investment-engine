"""Main risk management engine."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

from .var_calculator import VaRCalculator
from .position_sizer import PositionSizer
from .drawdown_monitor import DrawdownMonitor
from core.models import Portfolio, PerformanceMetrics
from core.enums import RiskMeasure


class RiskManager:
    """Main risk management engine."""
    
    def __init__(self, 
                 max_portfolio_var: float = 0.05,
                 max_individual_weight: float = 0.1,
                 max_sector_weight: float = 0.3,
                 max_drawdown: float = 0.2):
        """Initialize risk manager.
        
        Args:
            max_portfolio_var: Maximum portfolio VaR (5% default)
            max_individual_weight: Maximum individual asset weight
            max_sector_weight: Maximum sector weight
            max_drawdown: Maximum drawdown threshold
        """
        self.max_portfolio_var = max_portfolio_var
        self.max_individual_weight = max_individual_weight
        self.max_sector_weight = max_sector_weight
        self.max_drawdown = max_drawdown
        
        self.var_calculator = VaRCalculator()
        self.position_sizer = PositionSizer()
        self.drawdown_monitor = DrawdownMonitor()
        
        self.risk_alerts = []
        self.risk_metrics_history = []
    
    def assess_portfolio_risk(self,
                            portfolio: Portfolio,
                            returns_history: pd.DataFrame,
                            benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Comprehensive portfolio risk assessment.
        
        Args:
            portfolio: Portfolio to assess
            returns_history: Historical returns data
            benchmark_returns: Benchmark returns for relative metrics
            
        Returns:
            Dictionary of risk metrics and alerts
        """
        risk_assessment = {
            'timestamp': datetime.now(),
            'portfolio_value': float(portfolio.total_value),
            'alerts': [],
            'metrics': {}
        }
        
        # Get portfolio weights and returns
        weights = pd.Series(portfolio.weights)
        
        # Align returns with portfolio assets
        portfolio_assets = list(portfolio.positions.keys())
        available_returns = returns_history.reindex(columns=portfolio_assets).dropna(axis=1)
        
        if available_returns.empty:
            risk_assessment['alerts'].append({
                'type': 'data_warning',
                'message': 'No historical returns data available for risk assessment'
            })
            return risk_assessment
        
        # Calculate portfolio returns
        aligned_weights = weights.reindex(available_returns.columns, fill_value=0)
        portfolio_returns = (available_returns * aligned_weights).sum(axis=1)
        
        # VaR and CVaR calculations
        var_metrics = self.var_calculator.calculate_var(
            portfolio_returns, 
            confidence_levels=[0.95, 0.99]
        )
        risk_assessment['metrics'].update(var_metrics)
        
        # Check VaR limits
        if var_metrics.get('var_95', 0) > self.max_portfolio_var:
            risk_assessment['alerts'].append({
                'type': 'var_breach',
                'message': f"Portfolio VaR (95%) {var_metrics['var_95']:.3f} exceeds limit {self.max_portfolio_var:.3f}",
                'severity': 'high'
            })
        
        # Concentration risk checks
        concentration_metrics = self._assess_concentration_risk(portfolio)
        risk_assessment['metrics'].update(concentration_metrics)
        
        # Add concentration alerts
        risk_assessment['alerts'].extend(concentration_metrics.get('alerts', []))
        
        # Drawdown analysis
        drawdown_metrics = self.drawdown_monitor.calculate_drawdowns(portfolio_returns)
        risk_assessment['metrics'].update(drawdown_metrics)
        
        # Check drawdown limits
        current_drawdown = drawdown_metrics.get('current_drawdown', 0)
        if abs(current_drawdown) > self.max_drawdown:
            risk_assessment['alerts'].append({
                'type': 'drawdown_breach',
                'message': f"Current drawdown {current_drawdown:.3f} exceeds limit {self.max_drawdown:.3f}",
                'severity': 'critical'
            })
        
        # Volatility and correlation analysis
        volatility_metrics = self._calculate_volatility_metrics(available_returns, aligned_weights)
        risk_assessment['metrics'].update(volatility_metrics)
        
        # Benchmark relative metrics
        if benchmark_returns is not None:
            relative_metrics = self._calculate_relative_risk_metrics(
                portfolio_returns, benchmark_returns
            )
            risk_assessment['metrics'].update(relative_metrics)
        
        # Store risk metrics history
        self.risk_metrics_history.append(risk_assessment)
        
        # Update alerts
        self.risk_alerts.extend(risk_assessment['alerts'])
        
        return risk_assessment
    
    def _assess_concentration_risk(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Assess concentration risk in the portfolio."""
        metrics = {'alerts': []}
        
        weights = pd.Series(portfolio.weights)
        
        # Individual asset concentration
        max_weight = weights.max()
        metrics['max_individual_weight'] = max_weight
        
        if max_weight > self.max_individual_weight:
            max_asset = weights.idxmax()
            metrics['alerts'].append({
                'type': 'concentration_breach',
                'message': f"Asset {max_asset} weight {max_weight:.3f} exceeds limit {self.max_individual_weight:.3f}",
                'severity': 'medium'
            })
        
        # Sector concentration
        sector_weights = {}
        for symbol, position in portfolio.positions.items():
            sector = position.asset.sector or 'Unknown'
            weight = weights.get(symbol, 0)
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        if sector_weights:
            max_sector_weight = max(sector_weights.values())
            max_sector = max(sector_weights, key=sector_weights.get)
            
            metrics['max_sector_weight'] = max_sector_weight
            metrics['sector_weights'] = sector_weights
            
            if max_sector_weight > self.max_sector_weight:
                metrics['alerts'].append({
                    'type': 'sector_concentration_breach',
                    'message': f"Sector {max_sector} weight {max_sector_weight:.3f} exceeds limit {self.max_sector_weight:.3f}",
                    'severity': 'medium'
                })
        
        # Herfindahl-Hirschman Index
        hhi = (weights ** 2).sum()
        metrics['herfindahl_index'] = hhi
        metrics['effective_assets'] = 1 / hhi if hhi > 0 else 0
        
        return metrics
    
    def _calculate_volatility_metrics(self, 
                                    returns: pd.DataFrame, 
                                    weights: pd.Series) -> Dict[str, float]:
        """Calculate volatility-related risk metrics."""
        # Portfolio volatility
        cov_matrix = returns.cov() * 252  # Annualized
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        
        # Individual asset volatilities
        asset_vols = returns.std() * np.sqrt(252)
        weighted_avg_vol = (weights * asset_vols).sum()
        
        # Diversification ratio
        diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 0
        
        return {
            'portfolio_volatility': portfolio_vol,
            'weighted_avg_volatility': weighted_avg_vol,
            'diversification_ratio': diversification_ratio
        }
    
    def _calculate_relative_risk_metrics(self, 
                                       portfolio_returns: pd.Series,
                                       benchmark_returns: pd.Series) -> Dict[str, float]:
        """Calculate risk metrics relative to benchmark."""
        # Align returns
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        port_ret = portfolio_returns.loc[common_dates]
        bench_ret = benchmark_returns.loc[common_dates]
        
        if len(common_dates) < 30:
            return {}
        
        # Tracking error
        active_returns = port_ret - bench_ret
        tracking_error = active_returns.std() * np.sqrt(252)
        
        # Beta
        covariance = np.cov(port_ret, bench_ret)[0, 1]
        benchmark_var = bench_ret.var()
        beta = covariance / benchmark_var if benchmark_var > 0 else 1
        
        # Information ratio
        active_return = active_returns.mean() * 252
        information_ratio = active_return / tracking_error if tracking_error > 0 else 0
        
        return {
            'tracking_error': tracking_error,
            'beta': beta,
            'information_ratio': information_ratio,
            'active_return': active_return
        }
    
    def check_risk_limits(self, 
                         proposed_weights: pd.Series,
                         returns_history: pd.DataFrame) -> Dict[str, Any]:
        """Check if proposed weights violate risk limits.
        
        Args:
            proposed_weights: Proposed portfolio weights
            returns_history: Historical returns data
            
        Returns:
            Risk check results with violations
        """
        violations = []
        
        # Individual weight limits
        max_weight = proposed_weights.max()
        if max_weight > self.max_individual_weight:
            violations.append({
                'type': 'individual_weight_limit',
                'asset': proposed_weights.idxmax(),
                'current': max_weight,
                'limit': self.max_individual_weight
            })
        
        # Portfolio VaR limit (if returns available)
        if not returns_history.empty:
            # Calculate portfolio returns with proposed weights
            available_returns = returns_history.reindex(columns=proposed_weights.index).dropna(axis=1)
            if not available_returns.empty:
                aligned_weights = proposed_weights.reindex(available_returns.columns, fill_value=0)
                portfolio_returns = (available_returns * aligned_weights).sum(axis=1)
                
                var_95 = self.var_calculator.calculate_var(portfolio_returns)['var_95']
                if var_95 > self.max_portfolio_var:
                    violations.append({
                        'type': 'portfolio_var_limit',
                        'current': var_95,
                        'limit': self.max_portfolio_var
                    })
        
        return {
            'passed': len(violations) == 0,
            'violations': violations
        }
    
    def suggest_risk_adjustments(self, 
                               portfolio: Portfolio,
                               returns_history: pd.DataFrame) -> Dict[str, Any]:
        """Suggest adjustments to reduce portfolio risk.
        
        Args:
            portfolio: Current portfolio
            returns_history: Historical returns data
            
        Returns:
            Risk adjustment suggestions
        """
        suggestions = []
        current_weights = pd.Series(portfolio.weights)
        
        # Check for concentration issues
        max_weight = current_weights.max()
        if max_weight > self.max_individual_weight:
            max_asset = current_weights.idxmax()
            target_weight = self.max_individual_weight
            reduction = max_weight - target_weight
            
            suggestions.append({
                'type': 'reduce_concentration',
                'asset': max_asset,
                'current_weight': max_weight,
                'suggested_weight': target_weight,
                'reduction': reduction,
                'priority': 'high'
            })
        
        # Suggest diversification improvements
        hhi = (current_weights ** 2).sum()
        if hhi > 0.2:  # Highly concentrated
            suggestions.append({
                'type': 'increase_diversification',
                'current_hhi': hhi,
                'effective_assets': 1 / hhi,
                'recommendation': 'Consider adding more assets or rebalancing to reduce concentration',
                'priority': 'medium'
            })
        
        return {
            'suggestions': suggestions,
            'risk_score': self._calculate_risk_score(portfolio, returns_history)
        }
    
    def _calculate_risk_score(self, 
                            portfolio: Portfolio,
                            returns_history: pd.DataFrame) -> float:
        """Calculate overall risk score (0-100, higher = riskier)."""
        score = 0
        weights = pd.Series(portfolio.weights)
        
        # Concentration penalty
        hhi = (weights ** 2).sum()
        concentration_score = min(hhi * 100, 50)
        score += concentration_score
        
        # Volatility penalty
        if not returns_history.empty:
            available_returns = returns_history.reindex(columns=list(portfolio.positions.keys())).dropna(axis=1)
            if not available_returns.empty:
                aligned_weights = weights.reindex(available_returns.columns, fill_value=0)
                portfolio_returns = (available_returns * aligned_weights).sum(axis=1)
                vol = portfolio_returns.std() * np.sqrt(252)
                vol_score = min(vol * 100, 30)
                score += vol_score
        
        # Drawdown penalty
        if hasattr(self, 'current_drawdown'):
            drawdown_score = min(abs(self.current_drawdown) * 100, 20)
            score += drawdown_score
        
        return min(score, 100)
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report.
        
        Returns:
            Risk report with current status and history
        """
        return {
            'current_alerts': [alert for alert in self.risk_alerts if alert.get('severity') in ['high', 'critical']],
            'total_alerts': len(self.risk_alerts),
            'recent_metrics': self.risk_metrics_history[-5:] if self.risk_metrics_history else [],
            'risk_limits': {
                'max_portfolio_var': self.max_portfolio_var,
                'max_individual_weight': self.max_individual_weight,
                'max_sector_weight': self.max_sector_weight,
                'max_drawdown': self.max_drawdown
            }
        }
