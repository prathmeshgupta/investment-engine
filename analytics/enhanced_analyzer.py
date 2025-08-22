"""
Enhanced Academic Research-Based Performance Analyzer
Comprehensive evaluation metrics based on latest financial literature
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class EnhancedPerformanceAnalyzer:
    """Advanced performance analyzer with academic research-based metrics."""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        self.academic_benchmarks = self._initialize_academic_benchmarks()
        self.factor_models = self._initialize_factor_models()
    
    def _initialize_academic_benchmarks(self) -> Dict[str, Dict]:
        """Initialize academic performance benchmarks from literature."""
        return {
            'sharpe_ratio': {
                'excellent': 1.0,
                'good': 0.5,
                'average': 0.3,
                'poor': 0.0,
                'source': 'Sharpe (1966), Lo (2002)'
            },
            'information_ratio': {
                'excellent': 0.5,
                'good': 0.25,
                'average': 0.15,
                'poor': 0.0,
                'source': 'Grinold & Kahn (2000)'
            },
            'sortino_ratio': {
                'excellent': 1.5,
                'good': 1.0,
                'average': 0.5,
                'poor': 0.0,
                'source': 'Sortino & Price (1994)'
            },
            'calmar_ratio': {
                'excellent': 1.0,
                'good': 0.5,
                'average': 0.3,
                'poor': 0.1,
                'source': 'Young (1991)'
            },
            'omega_ratio': {
                'excellent': 1.5,
                'good': 1.2,
                'average': 1.1,
                'poor': 1.0,
                'source': 'Keating & Shadwick (2002)'
            }
        }
    
    def _initialize_factor_models(self) -> Dict[str, Dict]:
        """Initialize factor model specifications from academic literature."""
        return {
            'fama_french_3': {
                'factors': ['market_excess', 'smb', 'hml'],
                'source': 'Fama & French (1993)',
                'description': 'Three-factor model with market, size, and value'
            },
            'fama_french_5': {
                'factors': ['market_excess', 'smb', 'hml', 'rmw', 'cma'],
                'source': 'Fama & French (2015)',
                'description': 'Five-factor model adding profitability and investment'
            },
            'carhart_4': {
                'factors': ['market_excess', 'smb', 'hml', 'momentum'],
                'source': 'Carhart (1997)',
                'description': 'Four-factor model adding momentum to FF3'
            },
            'q_factor': {
                'factors': ['market_excess', 'size', 'investment', 'profitability'],
                'source': 'Hou, Xue & Zhang (2015)',
                'description': 'Q-factor model alternative to Fama-French'
            }
        }
    
    def analyze_comprehensive_performance(self, returns: pd.Series, 
                                        benchmark_returns: pd.Series = None,
                                        factor_returns: pd.DataFrame = None) -> Dict[str, Any]:
        """Comprehensive performance analysis with academic metrics."""
        
        analysis = {
            'basic_metrics': self._calculate_basic_metrics(returns, benchmark_returns),
            'risk_metrics': self._calculate_risk_metrics(returns, benchmark_returns),
            'tail_risk_metrics': self._calculate_tail_risk_metrics(returns),
            'factor_analysis': self._perform_factor_analysis(returns, factor_returns),
            'regime_analysis': self._analyze_market_regimes(returns),
            'attribution_analysis': self._perform_attribution_analysis(returns, benchmark_returns, factor_returns),
            'academic_insights': self._generate_academic_insights(returns, benchmark_returns, factor_returns),
            'implementation_metrics': self._calculate_implementation_metrics(returns),
            'behavioral_metrics': self._calculate_behavioral_metrics(returns)
        }
        
        return analysis
    
    def _calculate_basic_metrics(self, returns: pd.Series, 
                               benchmark_returns: pd.Series = None) -> Dict[str, float]:
        """Calculate basic performance metrics."""
        
        metrics = {}
        
        # Return metrics
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annualized_return'] = (1 + returns.mean()) ** 252 - 1
        metrics['volatility'] = returns.std() * np.sqrt(252)
        metrics['excess_return'] = metrics['annualized_return'] - self.risk_free_rate
        
        # Risk-adjusted metrics
        if metrics['volatility'] > 0:
            metrics['sharpe_ratio'] = metrics['excess_return'] / metrics['volatility']
        else:
            metrics['sharpe_ratio'] = 0
        
        # Benchmark-relative metrics
        if benchmark_returns is not None:
            active_returns = returns - benchmark_returns
            metrics['active_return'] = active_returns.mean() * 252
            metrics['tracking_error'] = active_returns.std() * np.sqrt(252)
            if metrics['tracking_error'] > 0:
                metrics['information_ratio'] = metrics['active_return'] / metrics['tracking_error']
            else:
                metrics['information_ratio'] = 0
            
            # Beta calculation
            if len(benchmark_returns) > 1:
                covariance = np.cov(returns, benchmark_returns)[0, 1]
                benchmark_variance = np.var(benchmark_returns)
                if benchmark_variance > 0:
                    metrics['beta'] = covariance / benchmark_variance
                else:
                    metrics['beta'] = 1.0
                
                # Jensen's Alpha
                expected_return = self.risk_free_rate + metrics['beta'] * (benchmark_returns.mean() * 252 - self.risk_free_rate)
                metrics['jensen_alpha'] = metrics['annualized_return'] - expected_return
        
        return metrics
    
    def _calculate_risk_metrics(self, returns: pd.Series, 
                              benchmark_returns: pd.Series = None) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        
        metrics = {}
        
        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        
        metrics['max_drawdown'] = drawdowns.min()
        metrics['average_drawdown'] = drawdowns[drawdowns < 0].mean() if len(drawdowns[drawdowns < 0]) > 0 else 0
        
        # Recovery analysis
        drawdown_periods = self._identify_drawdown_periods(drawdowns)
        if drawdown_periods:
            recovery_times = [period['recovery_time'] for period in drawdown_periods if period['recovery_time'] is not None]
            metrics['average_recovery_time'] = np.mean(recovery_times) if recovery_times else np.nan
            metrics['max_recovery_time'] = np.max(recovery_times) if recovery_times else np.nan
        
        # Downside risk metrics
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            metrics['downside_deviation'] = downside_returns.std() * np.sqrt(252)
            metrics['sortino_ratio'] = (returns.mean() * 252 - self.risk_free_rate) / metrics['downside_deviation']
        else:
            metrics['downside_deviation'] = 0
            metrics['sortino_ratio'] = np.inf
        
        # Calmar ratio
        if abs(metrics['max_drawdown']) > 0:
            metrics['calmar_ratio'] = (returns.mean() * 252) / abs(metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = np.inf
        
        # Ulcer Index (Peter Martin & Byron McCann, 1989)
        squared_drawdowns = drawdowns ** 2
        metrics['ulcer_index'] = np.sqrt(squared_drawdowns.mean())
        
        # Pain Index
        metrics['pain_index'] = abs(drawdowns).mean()
        
        return metrics
    
    def _calculate_tail_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate tail risk and extreme value metrics."""
        
        metrics = {}
        
        # Value at Risk (VaR)
        metrics['var_95'] = np.percentile(returns, 5)
        metrics['var_99'] = np.percentile(returns, 1)
        metrics['var_99_9'] = np.percentile(returns, 0.1)
        
        # Conditional Value at Risk (Expected Shortfall)
        metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
        metrics['cvar_99'] = returns[returns <= metrics['var_99']].mean()
        
        # Maximum Loss
        metrics['maximum_loss'] = returns.min()
        
        # Tail Ratio (Eling & Schuhmacher, 2007)
        right_tail = returns[returns >= np.percentile(returns, 95)]
        left_tail = returns[returns <= np.percentile(returns, 5)]
        if len(left_tail) > 0 and abs(left_tail.mean()) > 0:
            metrics['tail_ratio'] = right_tail.mean() / abs(left_tail.mean())
        else:
            metrics['tail_ratio'] = np.inf
        
        # Skewness and Kurtosis
        metrics['skewness'] = stats.skew(returns)
        metrics['kurtosis'] = stats.kurtosis(returns)
        metrics['excess_kurtosis'] = metrics['kurtosis'] - 3
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        metrics['jarque_bera_stat'] = jb_stat
        metrics['jarque_bera_pvalue'] = jb_pvalue
        metrics['is_normal_distribution'] = jb_pvalue > 0.05
        
        return metrics
    
    def _perform_factor_analysis(self, returns: pd.Series, 
                               factor_returns: pd.DataFrame = None) -> Dict[str, Any]:
        """Perform comprehensive factor analysis."""
        
        if factor_returns is None:
            return {'error': 'No factor returns provided'}
        
        analysis = {}
        
        # Single factor regressions
        single_factor_results = {}
        for factor in factor_returns.columns:
            if len(factor_returns[factor].dropna()) > 10:  # Minimum observations
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    factor_returns[factor].fillna(0), returns.fillna(0)
                )
                
                single_factor_results[factor] = {
                    'beta': slope,
                    'alpha': intercept * 252,  # Annualized
                    'r_squared': r_value ** 2,
                    'p_value': p_value,
                    'std_error': std_err,
                    'significant': p_value < 0.05
                }
        
        analysis['single_factor_analysis'] = single_factor_results
        
        # Multi-factor regression
        if len(factor_returns.columns) > 1:
            # Prepare data
            y = returns.fillna(0).values
            X = factor_returns.fillna(0).values
            
            # Fit regression
            reg = LinearRegression().fit(X, y)
            
            # Calculate statistics
            y_pred = reg.predict(X)
            residuals = y - y_pred
            
            analysis['multi_factor_analysis'] = {
                'alpha': reg.intercept_ * 252,  # Annualized
                'factor_loadings': dict(zip(factor_returns.columns, reg.coef_)),
                'r_squared': reg.score(X, y),
                'residual_volatility': np.std(residuals) * np.sqrt(252),
                'tracking_error_to_factors': np.std(residuals) * np.sqrt(252)
            }
            
            # Factor contribution analysis
            factor_contributions = {}
            for i, factor in enumerate(factor_returns.columns):
                contribution = reg.coef_[i] * factor_returns.iloc[:, i].mean() * 252
                factor_contributions[factor] = contribution
            
            analysis['factor_contributions'] = factor_contributions
        
        return analysis
    
    def _analyze_market_regimes(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze performance across different market regimes."""
        
        # Simple regime identification based on volatility
        rolling_vol = returns.rolling(20).std()
        vol_threshold = rolling_vol.quantile(0.7)
        
        high_vol_periods = rolling_vol > vol_threshold
        low_vol_periods = rolling_vol <= vol_threshold
        
        regime_analysis = {}
        
        # High volatility regime
        high_vol_returns = returns[high_vol_periods]
        if len(high_vol_returns) > 0:
            regime_analysis['high_volatility'] = {
                'count': len(high_vol_returns),
                'percentage': len(high_vol_returns) / len(returns),
                'mean_return': high_vol_returns.mean() * 252,
                'volatility': high_vol_returns.std() * np.sqrt(252),
                'sharpe_ratio': (high_vol_returns.mean() * 252 - self.risk_free_rate) / (high_vol_returns.std() * np.sqrt(252)) if high_vol_returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(high_vol_returns)
            }
        
        # Low volatility regime
        low_vol_returns = returns[low_vol_periods]
        if len(low_vol_returns) > 0:
            regime_analysis['low_volatility'] = {
                'count': len(low_vol_returns),
                'percentage': len(low_vol_returns) / len(returns),
                'mean_return': low_vol_returns.mean() * 252,
                'volatility': low_vol_returns.std() * np.sqrt(252),
                'sharpe_ratio': (low_vol_returns.mean() * 252 - self.risk_free_rate) / (low_vol_returns.std() * np.sqrt(252)) if low_vol_returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(low_vol_returns)
            }
        
        return regime_analysis
    
    def _perform_attribution_analysis(self, returns: pd.Series, 
                                    benchmark_returns: pd.Series = None,
                                    factor_returns: pd.DataFrame = None) -> Dict[str, Any]:
        """Perform return attribution analysis."""
        
        attribution = {}
        
        if benchmark_returns is not None:
            # Brinson attribution (simplified)
            total_return = (1 + returns).prod() - 1
            benchmark_return = (1 + benchmark_returns).prod() - 1
            
            attribution['total_active_return'] = total_return - benchmark_return
            attribution['benchmark_return'] = benchmark_return
            attribution['portfolio_return'] = total_return
        
        if factor_returns is not None and len(factor_returns.columns) > 1:
            # Factor-based attribution
            y = returns.fillna(0).values
            X = factor_returns.fillna(0).values
            
            reg = LinearRegression().fit(X, y)
            
            factor_attribution = {}
            for i, factor in enumerate(factor_returns.columns):
                factor_return = factor_returns.iloc[:, i].mean() * 252
                factor_loading = reg.coef_[i]
                contribution = factor_loading * factor_return
                factor_attribution[factor] = contribution
            
            attribution['factor_attribution'] = factor_attribution
            attribution['alpha_contribution'] = reg.intercept_ * 252
        
        return attribution
    
    def _generate_academic_insights(self, returns: pd.Series, 
                                  benchmark_returns: pd.Series = None,
                                  factor_returns: pd.DataFrame = None) -> Dict[str, Any]:
        """Generate insights based on academic research."""
        
        insights = {
            'performance_evaluation': [],
            'risk_assessment': [],
            'factor_insights': [],
            'implementation_notes': [],
            'academic_references': []
        }
        
        # Calculate key metrics for insights
        sharpe_ratio = (returns.mean() * 252 - self.risk_free_rate) / (returns.std() * np.sqrt(252))
        max_dd = self._calculate_max_drawdown(returns)
        
        # Performance insights
        if sharpe_ratio > 1.0:
            insights['performance_evaluation'].append(
                "Excellent risk-adjusted performance (Sharpe > 1.0) - consistent with top-quartile institutional managers (Sharpe, 1994)"
            )
        elif sharpe_ratio > 0.5:
            insights['performance_evaluation'].append(
                "Good risk-adjusted performance - above median institutional performance (CEM Benchmarking, 2020)"
            )
        else:
            insights['performance_evaluation'].append(
                "Below-average risk-adjusted performance - consider factor exposure review (Fama & French, 2010)"
            )
        
        # Risk insights
        if abs(max_dd) < 0.10:
            insights['risk_assessment'].append(
                "Low maximum drawdown (<10%) indicates strong risk management (Calmar, 1991)"
            )
        elif abs(max_dd) > 0.20:
            insights['risk_assessment'].append(
                "High maximum drawdown (>20%) suggests need for enhanced risk controls (Magdon-Ismail & Atiya, 2004)"
            )
        
        # Factor insights
        if factor_returns is not None:
            # Check for momentum exposure
            if 'momentum' in factor_returns.columns:
                momentum_corr = returns.corr(factor_returns['momentum'])
                if abs(momentum_corr) > 0.3:
                    insights['factor_insights'].append(
                        f"Significant momentum exposure (corr={momentum_corr:.2f}) - consistent with Jegadeesh & Titman (1993) findings"
                    )
        
        # Implementation insights
        volatility = returns.std() * np.sqrt(252)
        if volatility > 0.20:
            insights['implementation_notes'].append(
                "High volatility (>20%) may indicate need for position sizing adjustments (Kelly, 1956)"
            )
        
        # Academic references
        insights['academic_references'] = [
            "Sharpe, W.F. (1994). The Sharpe Ratio. Journal of Portfolio Management",
            "Fama, E.F. & French, K.R. (2010). Luck versus Skill in the Cross-Section of Mutual Fund Returns",
            "Jegadeesh, N. & Titman, S. (1993). Returns to Buying Winners and Selling Losers",
            "Calmar, T. (1991). The Calmar Ratio: A Smoother Tool. Futures Magazine"
        ]
        
        return insights
    
    def _calculate_implementation_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate implementation and capacity-related metrics."""
        
        metrics = {}
        
        # Turnover estimation (simplified)
        # In practice, this would require position data
        volatility = returns.std()
        metrics['estimated_turnover'] = volatility * 10  # Rough approximation
        
        # Capacity indicators
        metrics['return_consistency'] = 1 - (returns.std() / abs(returns.mean())) if returns.mean() != 0 else 0
        
        # Implementation shortfall estimation
        # Simplified - would need transaction cost data
        metrics['estimated_implementation_shortfall'] = volatility * 0.01  # 1% of volatility as rough estimate
        
        return metrics
    
    def _calculate_behavioral_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate behavioral finance-related metrics."""
        
        metrics = {}
        
        # Loss aversion metrics
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(gains) > 0 and len(losses) > 0:
            metrics['gain_loss_ratio'] = gains.mean() / abs(losses.mean())
            metrics['win_rate'] = len(gains) / len(returns)
        
        # Streaks analysis
        returns_sign = np.sign(returns)
        streaks = []
        current_streak = 1
        
        for i in range(1, len(returns_sign)):
            if returns_sign.iloc[i] == returns_sign.iloc[i-1]:
                current_streak += 1
            else:
                streaks.append(current_streak)
                current_streak = 1
        streaks.append(current_streak)
        
        metrics['max_winning_streak'] = max([s for i, s in enumerate(streaks) if returns_sign.iloc[i] > 0], default=0)
        metrics['max_losing_streak'] = max([s for i, s in enumerate(streaks) if returns_sign.iloc[i] < 0], default=0)
        
        return metrics
    
    def _identify_drawdown_periods(self, drawdowns: pd.Series) -> List[Dict]:
        """Identify individual drawdown periods."""
        
        periods = []
        in_drawdown = False
        start_date = None
        peak_value = 0
        trough_value = 0
        
        for date, dd in drawdowns.items():
            if dd < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                start_date = date
                peak_value = 0
                trough_value = dd
            elif dd < 0 and in_drawdown:
                # Continue drawdown
                trough_value = min(trough_value, dd)
            elif dd >= 0 and in_drawdown:
                # End of drawdown
                in_drawdown = False
                recovery_time = (date - start_date).days if start_date else None
                
                periods.append({
                    'start_date': start_date,
                    'end_date': date,
                    'peak_to_trough': trough_value,
                    'recovery_time': recovery_time
                })
        
        return periods
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown for a return series."""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        return drawdowns.min()

# Initialize enhanced analyzer
enhanced_analyzer = EnhancedPerformanceAnalyzer()
