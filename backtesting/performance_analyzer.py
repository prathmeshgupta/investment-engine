"""Performance analysis and metrics calculation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from scipy import stats

from core.models import PerformanceMetrics


class PerformanceAnalyzer:
    """Analyze portfolio performance and calculate metrics."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize performance analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_metrics(self,
                         portfolio_returns: pd.Series,
                         benchmark_returns: Optional[pd.Series] = None,
                         periods_per_year: int = 252) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics.
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            periods_per_year: Number of periods per year
            
        Returns:
            PerformanceMetrics object
        """
        if portfolio_returns.empty:
            raise ValueError("Portfolio returns cannot be empty")
        
        # Basic return metrics
        total_return = (1 + portfolio_returns).prod() - 1
        n_periods = len(portfolio_returns)
        annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
        
        # Risk metrics
        volatility = portfolio_returns.std() * np.sqrt(periods_per_year)
        
        # Risk-adjusted metrics
        excess_returns = portfolio_returns - self.risk_free_rate / periods_per_year
        sharpe_ratio = excess_returns.mean() / portfolio_returns.std() * np.sqrt(periods_per_year) if portfolio_returns.std() > 0 else 0
        
        # Drawdown metrics
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        # VaR and CVaR
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Benchmark relative metrics
        alpha = None
        beta = None
        tracking_error = None
        information_ratio = None
        
        if benchmark_returns is not None:
            # Align returns
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_dates) > 30:
                port_aligned = portfolio_returns.loc[common_dates]
                bench_aligned = benchmark_returns.loc[common_dates]
                
                # Beta calculation
                covariance = np.cov(port_aligned, bench_aligned)[0, 1]
                benchmark_variance = bench_aligned.var()
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
                
                # Alpha calculation
                benchmark_return = (1 + bench_aligned).prod() ** (periods_per_year / len(bench_aligned)) - 1
                alpha = annualized_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
                
                # Tracking error
                active_returns = port_aligned - bench_aligned
                tracking_error = active_returns.std() * np.sqrt(periods_per_year)
                
                # Information ratio
                active_return = active_returns.mean() * periods_per_year
                information_ratio = active_return / tracking_error if tracking_error > 0 else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
            alpha=alpha,
            beta=beta,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            start_date=portfolio_returns.index[0].date(),
            end_date=portfolio_returns.index[-1].date()
        )
    
    def calculate_rolling_metrics(self,
                                portfolio_returns: pd.Series,
                                window: int = 252,
                                metrics: List[str] = None) -> pd.DataFrame:
        """Calculate rolling performance metrics.
        
        Args:
            portfolio_returns: Portfolio return series
            window: Rolling window size
            metrics: List of metrics to calculate
            
        Returns:
            DataFrame with rolling metrics
        """
        if metrics is None:
            metrics = ['return', 'volatility', 'sharpe_ratio', 'max_drawdown']
        
        rolling_metrics = pd.DataFrame(index=portfolio_returns.index)
        
        for metric in metrics:
            if metric == 'return':
                rolling_metrics[metric] = portfolio_returns.rolling(window).apply(
                    lambda x: (1 + x).prod() ** (252 / len(x)) - 1
                )
            elif metric == 'volatility':
                rolling_metrics[metric] = portfolio_returns.rolling(window).std() * np.sqrt(252)
            elif metric == 'sharpe_ratio':
                rolling_metrics[metric] = portfolio_returns.rolling(window).apply(
                    lambda x: (x.mean() - self.risk_free_rate / 252) / x.std() * np.sqrt(252)
                )
            elif metric == 'max_drawdown':
                rolling_metrics[metric] = portfolio_returns.rolling(window).apply(
                    self._calculate_rolling_max_drawdown
                )
        
        return rolling_metrics.dropna()
    
    def _calculate_rolling_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown for a rolling window."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        return drawdowns.min()
    
    def calculate_attribution(self,
                            portfolio_returns: pd.Series,
                            factor_returns: pd.DataFrame,
                            portfolio_exposures: pd.DataFrame) -> pd.DataFrame:
        """Calculate factor attribution for portfolio returns.
        
        Args:
            portfolio_returns: Portfolio return series
            factor_returns: Factor return DataFrame
            portfolio_exposures: Portfolio factor exposures over time
            
        Returns:
            Attribution DataFrame
        """
        # Align dates
        common_dates = portfolio_returns.index.intersection(factor_returns.index).intersection(portfolio_exposures.index)
        
        attribution = pd.DataFrame(index=common_dates, columns=factor_returns.columns)
        
        for date in common_dates:
            exposures = portfolio_exposures.loc[date]
            factor_rets = factor_returns.loc[date]
            
            # Factor contribution = exposure * factor return
            for factor in factor_returns.columns:
                if factor in exposures.index:
                    attribution.loc[date, factor] = exposures[factor] * factor_rets[factor]
        
        return attribution.dropna()
    
    def calculate_style_analysis(self,
                               portfolio_returns: pd.Series,
                               style_returns: pd.DataFrame,
                               window: int = 36) -> pd.DataFrame:
        """Perform rolling style analysis.
        
        Args:
            portfolio_returns: Portfolio return series
            style_returns: Style benchmark returns
            window: Rolling window for analysis (months)
            
        Returns:
            Style exposures over time
        """
        from sklearn.linear_model import LinearRegression
        
        # Convert window to daily periods (approximate)
        daily_window = window * 21
        
        style_exposures = pd.DataFrame(index=portfolio_returns.index, columns=style_returns.columns)
        
        for i in range(daily_window, len(portfolio_returns)):
            end_date = portfolio_returns.index[i]
            start_idx = i - daily_window
            
            # Get window data
            port_window = portfolio_returns.iloc[start_idx:i]
            style_window = style_returns.iloc[start_idx:i]
            
            # Align data
            common_dates = port_window.index.intersection(style_window.index)
            if len(common_dates) < 20:
                continue
            
            y = port_window.loc[common_dates].values
            X = style_window.loc[common_dates].values
            
            # Fit regression with non-negative constraints
            reg = LinearRegression(positive=True).fit(X, y)
            
            # Normalize exposures to sum to 1
            exposures = reg.coef_
            exposures = exposures / exposures.sum() if exposures.sum() > 0 else exposures
            
            style_exposures.loc[end_date] = exposures
        
        return style_exposures.dropna()
    
    def calculate_risk_adjusted_returns(self,
                                      portfolio_returns: pd.Series,
                                      benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calculate various risk-adjusted return metrics.
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            
        Returns:
            Dictionary of risk-adjusted metrics
        """
        metrics = {}
        
        # Basic metrics
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        excess_return = annual_return - self.risk_free_rate
        metrics['sharpe_ratio'] = excess_return / annual_vol if annual_vol > 0 else 0
        
        # Sortino ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        metrics['sortino_ratio'] = excess_return / downside_vol if downside_vol > 0 else 0
        
        # Calmar ratio
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_dd = abs(drawdowns.min())
        metrics['calmar_ratio'] = annual_return / max_dd if max_dd > 0 else 0
        
        # Omega ratio
        threshold = self.risk_free_rate / 252
        gains = portfolio_returns[portfolio_returns > threshold] - threshold
        losses = threshold - portfolio_returns[portfolio_returns <= threshold]
        metrics['omega_ratio'] = gains.sum() / losses.sum() if losses.sum() > 0 else np.inf
        
        if benchmark_returns is not None:
            # Align returns
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_dates) > 30:
                port_aligned = portfolio_returns.loc[common_dates]
                bench_aligned = benchmark_returns.loc[common_dates]
                
                # Information ratio
                active_returns = port_aligned - bench_aligned
                tracking_error = active_returns.std() * np.sqrt(252)
                active_return = active_returns.mean() * 252
                metrics['information_ratio'] = active_return / tracking_error if tracking_error > 0 else 0
                
                # Treynor ratio
                covariance = np.cov(port_aligned, bench_aligned)[0, 1]
                benchmark_var = bench_aligned.var()
                beta = covariance / benchmark_var if benchmark_var > 0 else 1
                metrics['treynor_ratio'] = excess_return / beta if beta != 0 else 0
        
        return metrics
    
    def generate_performance_report(self,
                                  portfolio_returns: pd.Series,
                                  benchmark_returns: Optional[pd.Series] = None,
                                  strategy_name: str = "Portfolio") -> Dict[str, Any]:
        """Generate comprehensive performance report.
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            strategy_name: Name of the strategy
            
        Returns:
            Performance report dictionary
        """
        # Calculate main metrics
        metrics = self.calculate_metrics(portfolio_returns, benchmark_returns)
        
        # Calculate risk-adjusted metrics
        risk_adjusted = self.calculate_risk_adjusted_returns(portfolio_returns, benchmark_returns)
        
        # Monthly/yearly statistics
        monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        yearly_returns = portfolio_returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        
        report = {
            'strategy_name': strategy_name,
            'period': {
                'start_date': portfolio_returns.index[0].strftime('%Y-%m-%d'),
                'end_date': portfolio_returns.index[-1].strftime('%Y-%m-%d'),
                'total_days': len(portfolio_returns)
            },
            'returns': {
                'total_return': metrics.total_return,
                'annualized_return': metrics.annualized_return,
                'best_month': monthly_returns.max(),
                'worst_month': monthly_returns.min(),
                'positive_months': (monthly_returns > 0).sum(),
                'negative_months': (monthly_returns < 0).sum(),
                'best_year': yearly_returns.max() if len(yearly_returns) > 0 else None,
                'worst_year': yearly_returns.min() if len(yearly_returns) > 0 else None
            },
            'risk': {
                'volatility': metrics.volatility,
                'max_drawdown': metrics.max_drawdown,
                'var_95': metrics.var_95,
                'cvar_95': metrics.cvar_95,
                'skewness': portfolio_returns.skew(),
                'kurtosis': portfolio_returns.kurtosis()
            },
            'risk_adjusted': risk_adjusted,
            'benchmark_relative': {
                'alpha': metrics.alpha,
                'beta': metrics.beta,
                'tracking_error': metrics.tracking_error,
                'information_ratio': metrics.information_ratio
            } if benchmark_returns is not None else None
        }
        
        return report
