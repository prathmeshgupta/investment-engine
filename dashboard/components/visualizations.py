"""
Advanced Visualization Components for Investment Engine Dashboard
Enhanced charts and graphs with academic research-based analytics
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class AdvancedVisualizationManager:
    """Manages advanced visualization components with academic-grade analytics."""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'success': '#38a169',
            'warning': '#d69e2e',
            'danger': '#e53e3e',
            'info': '#3182ce',
            'light': '#f7fafc',
            'dark': '#2d3748'
        }
        self.academic_metrics = self._initialize_academic_metrics()
    
    def _initialize_academic_metrics(self) -> Dict[str, Dict]:
        """Initialize academic performance metrics definitions."""
        return {
            'sharpe_ratio': {
                'name': 'Sharpe Ratio',
                'formula': '(Return - Risk_Free_Rate) / Volatility',
                'interpretation': 'Risk-adjusted return per unit of total risk',
                'benchmark': {'excellent': 1.0, 'good': 0.5, 'poor': 0.0}
            },
            'information_ratio': {
                'name': 'Information Ratio',
                'formula': 'Active_Return / Tracking_Error',
                'interpretation': 'Active return per unit of active risk',
                'benchmark': {'excellent': 0.5, 'good': 0.25, 'poor': 0.0}
            },
            'sortino_ratio': {
                'name': 'Sortino Ratio',
                'formula': '(Return - Target) / Downside_Deviation',
                'interpretation': 'Return per unit of downside risk',
                'benchmark': {'excellent': 1.5, 'good': 1.0, 'poor': 0.5}
            },
            'calmar_ratio': {
                'name': 'Calmar Ratio',
                'formula': 'Annual_Return / Max_Drawdown',
                'interpretation': 'Annual return per unit of maximum drawdown',
                'benchmark': {'excellent': 1.0, 'good': 0.5, 'poor': 0.2}
            },
            'omega_ratio': {
                'name': 'Omega Ratio',
                'formula': 'Probability_Weighted_Gains / Probability_Weighted_Losses',
                'interpretation': 'Ratio of gains to losses above/below threshold',
                'benchmark': {'excellent': 1.5, 'good': 1.2, 'poor': 1.0}
            }
        }
    
    def create_enhanced_performance_dashboard(self, returns: pd.Series, 
                                            benchmark_returns: pd.Series,
                                            factor_returns: pd.DataFrame = None) -> go.Figure:
        """Create comprehensive performance dashboard with multiple panels."""
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(returns, benchmark_returns, factor_returns)
        
        # Create subplot structure
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Cumulative Returns', 'Rolling Sharpe Ratio', 'Drawdown Analysis',
                'Return Distribution', 'Risk-Return Scatter', 'Factor Attribution',
                'Rolling Correlation', 'Performance Attribution', 'Risk Decomposition'
            ],
            specs=[
                [{"secondary_y": True}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"type": "pie"}]
            ]
        )
        
        # 1. Cumulative Returns
        cum_returns = (1 + returns).cumprod()
        cum_benchmark = (1 + benchmark_returns).cumprod()
        
        fig.add_trace(
            go.Scatter(x=returns.index, y=cum_returns, name='Portfolio', 
                      line=dict(color=self.color_palette['primary'], width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=benchmark_returns.index, y=cum_benchmark, name='Benchmark',
                      line=dict(color=self.color_palette['secondary'], width=2)),
            row=1, col=1
        )
        
        # 2. Rolling Sharpe Ratio
        rolling_sharpe = self._calculate_rolling_sharpe(returns, window=60)
        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, name='Rolling Sharpe',
                      line=dict(color=self.color_palette['success'], width=2)),
            row=1, col=2
        )
        
        # 3. Drawdown Analysis
        drawdowns = self._calculate_drawdowns(returns)
        fig.add_trace(
            go.Scatter(x=drawdowns.index, y=drawdowns, fill='tonexty', name='Drawdown',
                      line=dict(color=self.color_palette['danger'], width=1)),
            row=1, col=3
        )
        
        # 4. Return Distribution
        fig.add_trace(
            go.Histogram(x=returns, nbinsx=50, name='Return Distribution',
                        marker_color=self.color_palette['info'], opacity=0.7),
            row=2, col=1
        )
        
        # 5. Risk-Return Scatter (if factor returns available)
        if factor_returns is not None:
            risk_return_data = self._prepare_risk_return_scatter(returns, factor_returns)
            fig.add_trace(
                go.Scatter(x=risk_return_data['risk'], y=risk_return_data['return'],
                          mode='markers', name='Risk-Return',
                          marker=dict(size=10, color=self.color_palette['warning'])),
                row=2, col=2
            )
        
        # 6. Factor Attribution
        if factor_returns is not None:
            attribution = self._calculate_factor_attribution(returns, factor_returns)
            fig.add_trace(
                go.Bar(x=list(attribution.keys()), y=list(attribution.values()),
                      name='Factor Attribution', marker_color=self.color_palette['primary']),
                row=2, col=3
            )
        
        # 7. Rolling Correlation
        rolling_corr = returns.rolling(60).corr(benchmark_returns)
        fig.add_trace(
            go.Scatter(x=rolling_corr.index, y=rolling_corr, name='Rolling Correlation',
                      line=dict(color=self.color_palette['secondary'], width=2)),
            row=3, col=1
        )
        
        # 8. Performance Attribution
        attribution_data = self._calculate_performance_attribution(returns, benchmark_returns)
        fig.add_trace(
            go.Waterfall(
                x=list(attribution_data.keys()),
                y=list(attribution_data.values()),
                name='Performance Attribution'
            ),
            row=3, col=2
        )
        
        # 9. Risk Decomposition (Pie Chart)
        risk_decomp = self._calculate_risk_decomposition(returns, factor_returns)
        fig.add_trace(
            go.Pie(labels=list(risk_decomp.keys()), values=list(risk_decomp.values()),
                  name='Risk Decomposition'),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Comprehensive Performance Analytics Dashboard",
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def create_factor_heatmap(self, factor_returns: pd.DataFrame, 
                             portfolio_returns: pd.Series) -> go.Figure:
        """Create factor correlation and exposure heatmap."""
        
        # Calculate factor correlations
        factor_corr = factor_returns.corr()
        
        # Calculate portfolio factor exposures
        exposures = {}
        for factor in factor_returns.columns:
            beta, _, _, _, _ = stats.linregress(factor_returns[factor], portfolio_returns)
            exposures[factor] = beta
        
        # Create subplot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Factor Correlations', 'Portfolio Factor Exposures'],
            specs=[[{"type": "heatmap"}, {"type": "bar"}]]
        )
        
        # Factor correlation heatmap
        fig.add_trace(
            go.Heatmap(
                z=factor_corr.values,
                x=factor_corr.columns,
                y=factor_corr.index,
                colorscale='RdBu',
                zmid=0,
                text=np.round(factor_corr.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                name='Correlations'
            ),
            row=1, col=1
        )
        
        # Portfolio exposures
        fig.add_trace(
            go.Bar(
                x=list(exposures.keys()),
                y=list(exposures.values()),
                marker_color=[self.color_palette['success'] if v > 0 else self.color_palette['danger'] 
                             for v in exposures.values()],
                name='Exposures'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500,
            title_text="Factor Analysis Dashboard"
        )
        
        return fig
    
    def create_regime_analysis_chart(self, returns: pd.Series, 
                                   market_data: pd.DataFrame = None) -> go.Figure:
        """Create regime analysis visualization using Hidden Markov Models concept."""
        
        # Simple regime identification using volatility clustering
        rolling_vol = returns.rolling(20).std()
        vol_threshold = rolling_vol.quantile(0.7)
        
        regimes = pd.Series(index=returns.index, dtype=str)
        regimes[rolling_vol <= vol_threshold] = 'Low Volatility'
        regimes[rolling_vol > vol_threshold] = 'High Volatility'
        
        # Calculate regime-specific statistics
        regime_stats = {}
        for regime in regimes.unique():
            if pd.notna(regime):
                regime_returns = returns[regimes == regime]
                regime_stats[regime] = {
                    'mean': regime_returns.mean() * 252,
                    'vol': regime_returns.std() * np.sqrt(252),
                    'sharpe': (regime_returns.mean() * 252) / (regime_returns.std() * np.sqrt(252)),
                    'count': len(regime_returns)
                }
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Returns by Regime', 'Regime Statistics', 
                          'Regime Transitions', 'Performance Comparison'],
            specs=[[{"secondary_y": True}, {"type": "table"}],
                   [{"secondary_y": False}, {"type": "bar"}]]
        )
        
        # Returns colored by regime
        colors = [self.color_palette['success'] if r == 'Low Volatility' 
                 else self.color_palette['danger'] for r in regimes]
        
        fig.add_trace(
            go.Scatter(x=returns.index, y=returns, mode='markers',
                      marker=dict(color=colors, size=4),
                      name='Returns by Regime'),
            row=1, col=1
        )
        
        # Add volatility on secondary axis
        fig.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol, 
                      line=dict(color='gray', width=1),
                      name='Rolling Volatility'),
            row=1, col=1, secondary_y=True
        )
        
        # Regime statistics table
        table_data = []
        for regime, stats in regime_stats.items():
            table_data.append([regime, f"{stats['mean']:.2%}", f"{stats['vol']:.2%}", 
                             f"{stats['sharpe']:.2f}", str(stats['count'])])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Regime', 'Annual Return', 'Volatility', 'Sharpe', 'Observations']),
                cells=dict(values=list(zip(*table_data)))
            ),
            row=1, col=2
        )
        
        # Regime transitions
        regime_changes = (regimes != regimes.shift()).cumsum()
        fig.add_trace(
            go.Scatter(x=returns.index, y=regime_changes, 
                      line=dict(color=self.color_palette['info'], width=2),
                      name='Regime Changes'),
            row=2, col=1
        )
        
        # Performance comparison
        regime_names = list(regime_stats.keys())
        regime_sharpes = [regime_stats[r]['sharpe'] for r in regime_names]
        
        fig.add_trace(
            go.Bar(x=regime_names, y=regime_sharpes,
                  marker_color=[self.color_palette['success'], self.color_palette['danger']],
                  name='Sharpe by Regime'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Market Regime Analysis")
        return fig
    
    def create_tail_risk_analysis(self, returns: pd.Series) -> go.Figure:
        """Create comprehensive tail risk analysis visualization."""
        
        # Calculate tail risk metrics
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Extreme value analysis
        from scipy import stats as scipy_stats
        
        # Fit Generalized Extreme Value distribution
        gev_params = scipy_stats.genextreme.fit(returns)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Return Distribution with VaR', 'Q-Q Plot vs Normal',
                          'Tail Risk Metrics', 'Extreme Value Distribution Fit'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "bar"}, {"secondary_y": False}]]
        )
        
        # 1. Distribution with VaR lines
        fig.add_trace(
            go.Histogram(x=returns, nbinsx=50, name='Returns',
                        marker_color=self.color_palette['info'], opacity=0.7),
            row=1, col=1
        )
        
        # Add VaR lines
        fig.add_vline(x=var_95, line_dash="dash", line_color=self.color_palette['warning'],
                     annotation_text="VaR 95%", row=1, col=1)
        fig.add_vline(x=var_99, line_dash="dash", line_color=self.color_palette['danger'],
                     annotation_text="VaR 99%", row=1, col=1)
        
        # 2. Q-Q Plot
        theoretical_quantiles = scipy_stats.norm.ppf(np.linspace(0.01, 0.99, len(returns)))
        sample_quantiles = np.sort(returns)
        
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sample_quantiles,
                      mode='markers', name='Q-Q Plot',
                      marker=dict(color=self.color_palette['primary'], size=4)),
            row=1, col=2
        )
        
        # Add diagonal line
        min_val, max_val = min(theoretical_quantiles.min(), sample_quantiles.min()), \
                          max(theoretical_quantiles.max(), sample_quantiles.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', name='Normal Line',
                      line=dict(color='red', dash='dash')),
            row=1, col=2
        )
        
        # 3. Tail Risk Metrics
        tail_metrics = {
            'VaR 95%': var_95,
            'VaR 99%': var_99,
            'CVaR 95%': cvar_95,
            'CVaR 99%': cvar_99
        }
        
        fig.add_trace(
            go.Bar(x=list(tail_metrics.keys()), y=list(tail_metrics.values()),
                  marker_color=self.color_palette['danger'],
                  name='Tail Risk Metrics'),
            row=2, col=1
        )
        
        # 4. Extreme Value Distribution
        x_range = np.linspace(returns.min(), returns.max(), 100)
        gev_pdf = scipy_stats.genextreme.pdf(x_range, *gev_params)
        
        fig.add_trace(
            go.Scatter(x=x_range, y=gev_pdf, mode='lines',
                      name='GEV Fit', line=dict(color=self.color_palette['success'], width=2)),
            row=2, col=2
        )
        
        # Add empirical density
        hist, bin_edges = np.histogram(returns, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        fig.add_trace(
            go.Scatter(x=bin_centers, y=hist, mode='lines',
                      name='Empirical', line=dict(color=self.color_palette['info'], width=2)),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Tail Risk Analysis")
        return fig
    
    def _calculate_comprehensive_metrics(self, returns: pd.Series, 
                                       benchmark_returns: pd.Series,
                                       factor_returns: pd.DataFrame = None) -> Dict:
        """Calculate comprehensive performance metrics."""
        
        metrics = {}
        
        # Basic metrics
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annualized_return'] = (1 + returns.mean()) ** 252 - 1
        metrics['volatility'] = returns.std() * np.sqrt(252)
        metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility']
        
        # Risk metrics
        metrics['max_drawdown'] = self._calculate_max_drawdown(returns)
        metrics['var_95'] = np.percentile(returns, 5)
        metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
        
        # Relative metrics
        active_returns = returns - benchmark_returns
        metrics['tracking_error'] = active_returns.std() * np.sqrt(252)
        metrics['information_ratio'] = (active_returns.mean() * 252) / metrics['tracking_error']
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            metrics['downside_deviation'] = downside_returns.std() * np.sqrt(252)
            metrics['sortino_ratio'] = metrics['annualized_return'] / metrics['downside_deviation']
        
        return metrics
    
    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int = 60) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        rolling_mean = returns.rolling(window).mean() * 252
        rolling_std = returns.rolling(window).std() * np.sqrt(252)
        return rolling_mean / rolling_std
    
    def _calculate_drawdowns(self, returns: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        return (cumulative - running_max) / running_max
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        drawdowns = self._calculate_drawdowns(returns)
        return drawdowns.min()
    
    def _prepare_risk_return_scatter(self, returns: pd.Series, 
                                   factor_returns: pd.DataFrame) -> Dict:
        """Prepare risk-return scatter data."""
        data = {'risk': [], 'return': []}
        
        for factor in factor_returns.columns:
            factor_ret = factor_returns[factor].mean() * 252
            factor_risk = factor_returns[factor].std() * np.sqrt(252)
            data['return'].append(factor_ret)
            data['risk'].append(factor_risk)
        
        # Add portfolio point
        port_ret = returns.mean() * 252
        port_risk = returns.std() * np.sqrt(252)
        data['return'].append(port_ret)
        data['risk'].append(port_risk)
        
        return data
    
    def _calculate_factor_attribution(self, returns: pd.Series, 
                                    factor_returns: pd.DataFrame) -> Dict:
        """Calculate factor attribution using regression."""
        attribution = {}
        
        if factor_returns is not None:
            # Run multiple regression
            from sklearn.linear_model import LinearRegression
            
            X = factor_returns.values
            y = returns.values
            
            reg = LinearRegression().fit(X, y)
            
            for i, factor in enumerate(factor_returns.columns):
                attribution[factor] = reg.coef_[i] * factor_returns.iloc[:, i].mean() * 252
        
        return attribution
    
    def _calculate_performance_attribution(self, returns: pd.Series, 
                                         benchmark_returns: pd.Series) -> Dict:
        """Calculate performance attribution components."""
        
        total_return = (1 + returns).prod() - 1
        benchmark_return = (1 + benchmark_returns).prod() - 1
        active_return = total_return - benchmark_return
        
        return {
            'Benchmark': benchmark_return,
            'Active Return': active_return,
            'Total Return': total_return
        }
    
    def _calculate_risk_decomposition(self, returns: pd.Series, 
                                    factor_returns: pd.DataFrame = None) -> Dict:
        """Calculate risk decomposition."""
        
        total_var = returns.var()
        
        if factor_returns is not None:
            # Simple factor risk decomposition
            risk_decomp = {}
            remaining_var = total_var
            
            for factor in factor_returns.columns:
                factor_var = factor_returns[factor].var()
                contribution = min(factor_var / total_var, remaining_var / total_var)
                risk_decomp[factor] = contribution
                remaining_var -= contribution * total_var
            
            if remaining_var > 0:
                risk_decomp['Idiosyncratic'] = remaining_var / total_var
        else:
            # Simple decomposition
            risk_decomp = {
                'Systematic': 0.7,
                'Idiosyncratic': 0.3
            }
        
        return risk_decomp

# Initialize visualization manager
visualization_manager = AdvancedVisualizationManager()
