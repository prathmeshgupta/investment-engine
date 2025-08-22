"""Advanced Strategy Analyzer with Research-Backed Metrics."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from backtesting.performance_analyzer import PerformanceAnalyzer


@dataclass
class FactorExposure:
    """Factor exposure analysis results."""
    factor_name: str
    exposure: float
    t_statistic: float
    p_value: float
    r_squared: float
    significance: str


@dataclass
class StrategyInsights:
    """Actionable strategy insights."""
    overall_rating: str
    key_strengths: List[str]
    areas_for_improvement: List[str]
    risk_warnings: List[str]
    recommendations: List[str]
    factor_tilts: List[str]


class AdvancedStrategyAnalyzer:
    """Advanced strategy analyzer using research-backed metrics."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.performance_analyzer = PerformanceAnalyzer()
    
    def analyze_strategy_comprehensive(self, 
                                    returns: pd.Series,
                                    benchmark_returns: pd.Series,
                                    factor_returns: Optional[pd.DataFrame] = None,
                                    holdings: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Comprehensive strategy analysis using academic metrics.
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            factor_returns: Factor return data (SMB, HML, RMW, CMA, etc.)
            holdings: Portfolio holdings data
            
        Returns:
            Comprehensive analysis results
        """
        results = {}
        
        # Basic performance metrics
        results['basic_metrics'] = self._calculate_basic_metrics(returns, benchmark_returns)
        
        # Risk-adjusted performance metrics
        results['risk_metrics'] = self._calculate_risk_adjusted_metrics(returns, benchmark_returns)
        
        # Factor exposure analysis
        if factor_returns is not None:
            results['factor_analysis'] = self._analyze_factor_exposures(returns, factor_returns)
        
        # Regime analysis
        results['regime_analysis'] = self._analyze_performance_regimes(returns, benchmark_returns)
        
        # Drawdown analysis
        results['drawdown_analysis'] = self._analyze_drawdowns(returns)
        
        # Tail risk analysis
        results['tail_risk'] = self._analyze_tail_risk(returns)
        
        # Performance attribution
        if holdings is not None:
            results['attribution'] = self._performance_attribution(returns, holdings)
        
        # Strategy insights and recommendations
        results['insights'] = self._generate_strategy_insights(results)
        
        return results
    
    def _calculate_basic_metrics(self, returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
        """Calculate basic performance metrics."""
        excess_returns = returns - benchmark_returns
        
        metrics = {
            'total_return': (1 + returns).prod() - 1,
            'annualized_return': (1 + returns.mean()) ** 252 - 1,
            'volatility': returns.std() * np.sqrt(252),
            'benchmark_return': (1 + benchmark_returns).prod() - 1,
            'excess_return': excess_returns.mean() * 252,
            'tracking_error': excess_returns.std() * np.sqrt(252),
            'hit_rate': (excess_returns > 0).mean(),
            'up_capture': self._calculate_capture_ratio(returns, benchmark_returns, up=True),
            'down_capture': self._calculate_capture_ratio(returns, benchmark_returns, up=False),
            'batting_average': (returns > benchmark_returns).mean()
        }
        
        return metrics
    
    def _calculate_risk_adjusted_metrics(self, returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics based on academic research."""
        excess_returns = returns - benchmark_returns
        risk_free_rate = 0.02 / 252  # Assume 2% annual risk-free rate
        
        # Sharpe Ratio
        sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std() if returns.std() > 0 else 0
        
        # Information Ratio (Grinold & Kahn)
        information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        # Treynor Ratio
        beta = self._calculate_beta(returns, benchmark_returns)
        treynor_ratio = (returns.mean() - risk_free_rate) / beta if beta != 0 else 0
        
        # Jensen's Alpha
        jensen_alpha = returns.mean() - (risk_free_rate + beta * (benchmark_returns.mean() - risk_free_rate))
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else returns.std()
        sortino_ratio = (returns.mean() - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar Ratio
        max_drawdown = self._calculate_max_drawdown(returns)
        calmar_ratio = (returns.mean() * 252) / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Omega Ratio (Keating & Shadwick)
        threshold = 0
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns <= threshold].sum())
        omega_ratio = gains / losses if losses > 0 else np.inf
        
        # Conditional Sharpe Ratio (Agarwal & Naik)
        conditional_sharpe = self._calculate_conditional_sharpe(returns, benchmark_returns)
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'information_ratio': information_ratio,
            'treynor_ratio': treynor_ratio,
            'jensen_alpha': jensen_alpha * 252,  # Annualized
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'omega_ratio': omega_ratio,
            'conditional_sharpe': conditional_sharpe,
            'beta': beta,
            'alpha_t_stat': self._calculate_alpha_t_stat(returns, benchmark_returns)
        }
    
    def _analyze_factor_exposures(self, returns: pd.Series, factor_returns: pd.DataFrame) -> Dict[str, FactorExposure]:
        """Analyze factor exposures using Fama-French methodology."""
        factor_exposures = {}
        
        # Align data
        common_dates = returns.index.intersection(factor_returns.index)
        returns_aligned = returns.loc[common_dates]
        
        for factor in factor_returns.columns:
            factor_data = factor_returns[factor].loc[common_dates]
            
            # Run regression: R_p - R_f = α + β * Factor + ε
            X = np.column_stack([np.ones(len(factor_data)), factor_data.values])
            y = returns_aligned.values
            
            try:
                # OLS regression
                coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                alpha, beta = coeffs[0], coeffs[1]
                
                # Calculate statistics
                y_pred = X @ coeffs
                residuals = y - y_pred
                mse = np.mean(residuals ** 2)
                
                # T-statistic for beta
                x_var = np.var(factor_data.values)
                se_beta = np.sqrt(mse / (len(factor_data) * x_var)) if x_var > 0 else 0
                t_stat = beta / se_beta if se_beta > 0 else 0
                
                # P-value (two-tailed)
                from scipy import stats
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(factor_data) - 2))
                
                # R-squared
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Significance level
                if p_value < 0.01:
                    significance = "Highly Significant"
                elif p_value < 0.05:
                    significance = "Significant"
                elif p_value < 0.10:
                    significance = "Marginally Significant"
                else:
                    significance = "Not Significant"
                
                factor_exposures[factor] = FactorExposure(
                    factor_name=factor,
                    exposure=beta,
                    t_statistic=t_stat,
                    p_value=p_value,
                    r_squared=r_squared,
                    significance=significance
                )
                
            except Exception:
                # Handle singular matrix or other errors
                factor_exposures[factor] = FactorExposure(
                    factor_name=factor,
                    exposure=0.0,
                    t_statistic=0.0,
                    p_value=1.0,
                    r_squared=0.0,
                    significance="Unable to Calculate"
                )
        
        return factor_exposures
    
    def _analyze_performance_regimes(self, returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, Any]:
        """Analyze performance across different market regimes."""
        # Define market regimes based on benchmark performance
        benchmark_rolling_vol = benchmark_returns.rolling(60).std()
        high_vol_threshold = benchmark_rolling_vol.quantile(0.75)
        low_vol_threshold = benchmark_rolling_vol.quantile(0.25)
        
        # Bull/Bear markets based on rolling returns
        benchmark_rolling_ret = benchmark_returns.rolling(60).mean()
        bull_threshold = benchmark_rolling_ret.quantile(0.6)
        bear_threshold = benchmark_rolling_ret.quantile(0.4)
        
        regimes = {
            'high_volatility': benchmark_rolling_vol > high_vol_threshold,
            'low_volatility': benchmark_rolling_vol < low_vol_threshold,
            'bull_market': benchmark_rolling_ret > bull_threshold,
            'bear_market': benchmark_rolling_ret < bear_threshold,
            'crisis_periods': benchmark_returns < benchmark_returns.quantile(0.05)
        }
        
        regime_performance = {}
        for regime_name, regime_mask in regimes.items():
            if regime_mask.sum() > 10:  # Minimum observations
                regime_returns = returns[regime_mask]
                regime_benchmark = benchmark_returns[regime_mask]
                
                regime_performance[regime_name] = {
                    'strategy_return': regime_returns.mean() * 252,
                    'benchmark_return': regime_benchmark.mean() * 252,
                    'excess_return': (regime_returns - regime_benchmark).mean() * 252,
                    'volatility': regime_returns.std() * np.sqrt(252),
                    'sharpe_ratio': (regime_returns.mean() - 0.02/252) / regime_returns.std() if regime_returns.std() > 0 else 0,
                    'hit_rate': (regime_returns > regime_benchmark).mean(),
                    'observations': len(regime_returns)
                }
        
        return regime_performance
    
    def _analyze_drawdowns(self, returns: pd.Series) -> Dict[str, Any]:
        """Comprehensive drawdown analysis."""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        
        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        
        for date, dd in drawdowns.items():
            if dd < -0.01 and not in_drawdown:  # Start of drawdown (>1%)
                in_drawdown = True
                start_date = date
            elif dd >= -0.001 and in_drawdown:  # End of drawdown
                in_drawdown = False
                if start_date:
                    period_drawdowns = drawdowns[start_date:date]
                    drawdown_periods.append({
                        'start': start_date,
                        'end': date,
                        'duration': len(period_drawdowns),
                        'max_drawdown': period_drawdowns.min(),
                        'recovery_time': len(period_drawdowns)
                    })
        
        # Drawdown statistics
        max_drawdown = drawdowns.min()
        avg_drawdown = drawdowns[drawdowns < 0].mean() if (drawdowns < 0).any() else 0
        
        # Ulcer Index (Martin & McCann)
        ulcer_index = np.sqrt(np.mean(drawdowns ** 2))
        
        # Pain Index
        pain_index = abs(drawdowns[drawdowns < 0].sum()) / len(drawdowns) if (drawdowns < 0).any() else 0
        
        return {
            'max_drawdown': max_drawdown,
            'average_drawdown': avg_drawdown,
            'ulcer_index': ulcer_index,
            'pain_index': pain_index,
            'drawdown_periods': len(drawdown_periods),
            'longest_drawdown': max([p['duration'] for p in drawdown_periods]) if drawdown_periods else 0,
            'deepest_drawdown_period': min(drawdown_periods, key=lambda x: x['max_drawdown']) if drawdown_periods else None
        }
    
    def _analyze_tail_risk(self, returns: pd.Series) -> Dict[str, float]:
        """Analyze tail risk characteristics."""
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Conditional Value at Risk (CVaR/Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Skewness and Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Maximum loss
        max_loss = returns.min()
        
        # Tail ratio (95th percentile / 5th percentile)
        tail_ratio = np.percentile(returns, 95) / abs(np.percentile(returns, 5)) if np.percentile(returns, 5) != 0 else 0
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'max_loss': max_loss,
            'tail_ratio': tail_ratio
        }
    
    def _performance_attribution(self, returns: pd.Series, holdings: pd.DataFrame) -> Dict[str, Any]:
        """Brinson-Hood-Beebower performance attribution."""
        # Simplified attribution analysis
        # In practice, this would require detailed sector/security level data
        
        attribution = {
            'asset_allocation_effect': 0.0,  # Would calculate from sector weights vs benchmark
            'security_selection_effect': 0.0,  # Would calculate from security performance
            'interaction_effect': 0.0,  # Cross-product of allocation and selection
            'total_active_return': returns.mean() * 252
        }
        
        return attribution
    
    def _generate_strategy_insights(self, analysis_results: Dict[str, Any]) -> StrategyInsights:
        """Generate actionable insights based on analysis results."""
        basic_metrics = analysis_results['basic_metrics']
        risk_metrics = analysis_results['risk_metrics']
        
        # Overall rating
        sharpe_ratio = risk_metrics['sharpe_ratio']
        information_ratio = risk_metrics['information_ratio']
        
        if sharpe_ratio > 1.5 and information_ratio > 0.5:
            overall_rating = "Excellent"
        elif sharpe_ratio > 1.0 and information_ratio > 0.3:
            overall_rating = "Good"
        elif sharpe_ratio > 0.5 and information_ratio > 0.1:
            overall_rating = "Fair"
        else:
            overall_rating = "Poor"
        
        # Key strengths
        strengths = []
        if risk_metrics['sharpe_ratio'] > 1.0:
            strengths.append("Strong risk-adjusted returns (Sharpe > 1.0)")
        if risk_metrics['information_ratio'] > 0.5:
            strengths.append("Excellent active management (IR > 0.5)")
        if basic_metrics['hit_rate'] > 0.6:
            strengths.append("Consistent outperformance (Hit rate > 60%)")
        if risk_metrics['calmar_ratio'] > 1.0:
            strengths.append("Strong drawdown-adjusted returns")
        
        # Areas for improvement
        improvements = []
        if risk_metrics['sharpe_ratio'] < 0.5:
            improvements.append("Low risk-adjusted returns - consider reducing volatility")
        if basic_metrics['tracking_error'] > 0.1:
            improvements.append("High tracking error - consider tighter risk controls")
        if 'drawdown_analysis' in analysis_results and analysis_results['drawdown_analysis']['max_drawdown'] < -0.2:
            improvements.append("Large maximum drawdown - implement better risk management")
        
        # Risk warnings
        warnings = []
        if 'tail_risk' in analysis_results:
            tail_risk = analysis_results['tail_risk']
            if tail_risk['skewness'] < -0.5:
                warnings.append("Negative skewness - strategy prone to large losses")
            if tail_risk['kurtosis'] > 3:
                warnings.append("High kurtosis - fat tail risk present")
        
        # Recommendations
        recommendations = []
        if risk_metrics['information_ratio'] < 0.2:
            recommendations.append("Consider enhancing alpha generation through better factor timing")
        if basic_metrics['down_capture'] > 1.1:
            recommendations.append("Implement downside protection mechanisms")
        if risk_metrics['beta'] > 1.2:
            recommendations.append("Consider reducing market exposure for better risk control")
        
        # Factor tilts (if factor analysis available)
        factor_tilts = []
        if 'factor_analysis' in analysis_results:
            for factor_name, exposure in analysis_results['factor_analysis'].items():
                if abs(exposure.exposure) > 0.2 and exposure.significance in ["Significant", "Highly Significant"]:
                    tilt_direction = "positive" if exposure.exposure > 0 else "negative"
                    factor_tilts.append(f"Strong {tilt_direction} {factor_name} factor exposure")
        
        return StrategyInsights(
            overall_rating=overall_rating,
            key_strengths=strengths,
            areas_for_improvement=improvements,
            risk_warnings=warnings,
            recommendations=recommendations,
            factor_tilts=factor_tilts
        )
    
    # Helper methods
    def _calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta coefficient."""
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        return covariance / benchmark_variance if benchmark_variance > 0 else 0
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_capture_ratio(self, returns: pd.Series, benchmark_returns: pd.Series, up: bool = True) -> float:
        """Calculate up/down capture ratios."""
        if up:
            mask = benchmark_returns > 0
        else:
            mask = benchmark_returns < 0
        
        if mask.sum() == 0:
            return 0
        
        strategy_avg = returns[mask].mean()
        benchmark_avg = benchmark_returns[mask].mean()
        
        return strategy_avg / benchmark_avg if benchmark_avg != 0 else 0
    
    def _calculate_conditional_sharpe(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate conditional Sharpe ratio during market stress."""
        stress_periods = benchmark_returns < benchmark_returns.quantile(0.1)
        if stress_periods.sum() < 10:
            return 0
        
        stress_returns = returns[stress_periods]
        return (stress_returns.mean() - 0.02/252) / stress_returns.std() if stress_returns.std() > 0 else 0
    
    def _calculate_alpha_t_stat(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate t-statistic for Jensen's alpha."""
        excess_returns = returns - benchmark_returns
        alpha = excess_returns.mean()
        alpha_std = excess_returns.std()
        n = len(excess_returns)
        
        if alpha_std > 0 and n > 1:
            t_stat = alpha * np.sqrt(n) / alpha_std
            return t_stat
        return 0
