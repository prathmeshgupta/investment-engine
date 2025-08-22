"""Visualization utilities for charts and graphs."""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class Visualizer:
    """Creates visualizations for the investment dashboard."""
    
    def __init__(self):
        """Initialize visualizer with default styling."""
        self.color_palette = px.colors.qualitative.Set3
        self.template = "plotly_white"
    
    def create_performance_chart(self, 
                               portfolio_data: pd.DataFrame,
                               benchmark_data: Optional[pd.DataFrame] = None) -> go.Figure:
        """Create portfolio performance chart.
        
        Args:
            portfolio_data: Portfolio performance data
            benchmark_data: Benchmark performance data
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Portfolio performance line
        fig.add_trace(go.Scatter(
            x=portfolio_data.index,
            y=portfolio_data['value'] if 'value' in portfolio_data.columns else portfolio_data.iloc[:, 0],
            mode='lines',
            name='Portfolio',
            line=dict(color='#3498db', width=2)
        ))
        
        # Benchmark if provided
        if benchmark_data is not None:
            fig.add_trace(go.Scatter(
                x=benchmark_data.index,
                y=benchmark_data.iloc[:, 0],
                mode='lines',
                name='Benchmark',
                line=dict(color='#e74c3c', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title="Portfolio Performance Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            template=self.template,
            hovermode='x unified'
        )
        
        return fig
    
    def create_allocation_chart(self, holdings_data: pd.DataFrame) -> go.Figure:
        """Create portfolio allocation pie chart.
        
        Args:
            holdings_data: Holdings data with symbols and weights
            
        Returns:
            Plotly figure
        """
        # Determine columns for symbols and values
        symbol_col = 'Symbol' if 'Symbol' in holdings_data.columns else holdings_data.columns[0]
        value_col = 'Weight' if 'Weight' in holdings_data.columns else 'Market Value'
        
        if value_col not in holdings_data.columns:
            value_col = holdings_data.columns[1]
        
        fig = go.Figure(data=[go.Pie(
            labels=holdings_data[symbol_col],
            values=holdings_data[value_col],
            hole=0.3,
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title="Portfolio Allocation",
            template=self.template,
            showlegend=True
        )
        
        return fig
    
    def create_returns_distribution(self, returns_data: pd.Series) -> go.Figure:
        """Create returns distribution histogram.
        
        Args:
            returns_data: Returns series
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=returns_data,
            nbinsx=50,
            name='Returns',
            opacity=0.7,
            marker_color='#3498db'
        ))
        
        # Normal distribution overlay
        mean_return = returns_data.mean()
        std_return = returns_data.std()
        x_range = np.linspace(returns_data.min(), returns_data.max(), 100)
        normal_dist = (1 / (std_return * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean_return) / std_return) ** 2)
        
        # Scale normal distribution to match histogram
        hist_max = len(returns_data) * (returns_data.max() - returns_data.min()) / 50
        normal_dist_scaled = normal_dist * hist_max / normal_dist.max()
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal_dist_scaled,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='#e74c3c', width=2)
        ))
        
        fig.update_layout(
            title="Returns Distribution",
            xaxis_title="Returns",
            yaxis_title="Frequency",
            template=self.template
        )
        
        return fig
    
    def create_drawdown_chart(self, portfolio_data: pd.DataFrame) -> go.Figure:
        """Create drawdown chart.
        
        Args:
            portfolio_data: Portfolio data with values or returns
            
        Returns:
            Plotly figure
        """
        # Calculate drawdowns
        if 'value' in portfolio_data.columns:
            values = portfolio_data['value']
        else:
            values = portfolio_data.iloc[:, 0]
        
        running_max = values.expanding().max()
        drawdowns = (values - running_max) / running_max
        
        fig = go.Figure()
        
        # Drawdown area chart
        fig.add_trace(go.Scatter(
            x=portfolio_data.index,
            y=drawdowns,
            fill='tonexty',
            mode='lines',
            name='Drawdown',
            line=dict(color='#e74c3c'),
            fillcolor='rgba(231, 76, 60, 0.3)'
        ))
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        fig.update_layout(
            title="Portfolio Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template=self.template,
            yaxis=dict(tickformat='.1%')
        )
        
        return fig
    
    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap.
        
        Args:
            correlation_matrix: Correlation matrix
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Asset Correlation Matrix",
            template=self.template,
            width=600,
            height=600
        )
        
        return fig
    
    def create_rolling_metrics_chart(self, 
                                   portfolio_data: pd.DataFrame,
                                   window: int = 252) -> go.Figure:
        """Create rolling metrics chart.
        
        Args:
            portfolio_data: Portfolio data
            window: Rolling window size
            
        Returns:
            Plotly figure
        """
        # Calculate rolling metrics
        if 'returns' in portfolio_data.columns:
            returns = portfolio_data['returns']
        else:
            returns = portfolio_data.pct_change().iloc[:, 0]
        
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
        
        fig = go.Figure()
        
        # Rolling volatility
        fig.add_trace(go.Scatter(
            x=portfolio_data.index,
            y=rolling_vol,
            mode='lines',
            name='Rolling Volatility',
            line=dict(color='#3498db'),
            yaxis='y'
        ))
        
        # Rolling Sharpe ratio
        fig.add_trace(go.Scatter(
            x=portfolio_data.index,
            y=rolling_sharpe,
            mode='lines',
            name='Rolling Sharpe Ratio',
            line=dict(color='#e74c3c'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title=f"Rolling Metrics ({window}-day window)",
            xaxis_title="Date",
            template=self.template,
            yaxis=dict(title="Volatility", side="left"),
            yaxis2=dict(title="Sharpe Ratio", side="right", overlaying="y")
        )
        
        return fig
    
    def create_var_chart(self, var_data: Dict[str, float]) -> go.Figure:
        """Create VaR chart.
        
        Args:
            var_data: VaR data with confidence levels
            
        Returns:
            Plotly figure
        """
        confidence_levels = []
        var_values = []
        
        for key, value in var_data.items():
            if 'var_' in key:
                conf_level = int(key.split('_')[1])
                confidence_levels.append(conf_level)
                var_values.append(value * 100)  # Convert to percentage
        
        fig = go.Figure(data=[
            go.Bar(
                x=[f"{cl}%" for cl in confidence_levels],
                y=var_values,
                marker_color='#e74c3c',
                text=[f"{v:.2f}%" for v in var_values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Value at Risk (VaR)",
            xaxis_title="Confidence Level",
            yaxis_title="VaR (%)",
            template=self.template
        )
        
        return fig
    
    def create_factor_exposure_chart(self, exposures: pd.DataFrame) -> go.Figure:
        """Create factor exposure chart.
        
        Args:
            exposures: Factor exposures data
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        for factor in exposures.columns:
            fig.add_trace(go.Scatter(
                x=exposures.index,
                y=exposures[factor],
                mode='lines',
                name=factor,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Factor Exposures Over Time",
            xaxis_title="Date",
            yaxis_title="Exposure",
            template=self.template,
            hovermode='x unified'
        )
        
        return fig
    
    def create_efficient_frontier(self, 
                                returns: np.ndarray,
                                risks: np.ndarray,
                                optimal_point: Optional[Tuple[float, float]] = None) -> go.Figure:
        """Create efficient frontier chart.
        
        Args:
            returns: Expected returns array
            risks: Risk (volatility) array
            optimal_point: Optimal portfolio (risk, return) point
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Efficient frontier
        fig.add_trace(go.Scatter(
            x=risks * 100,  # Convert to percentage
            y=returns * 100,  # Convert to percentage
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='#3498db', width=3)
        ))
        
        # Optimal portfolio point
        if optimal_point:
            fig.add_trace(go.Scatter(
                x=[optimal_point[0] * 100],
                y=[optimal_point[1] * 100],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(color='#e74c3c', size=10, symbol='star')
            ))
        
        fig.update_layout(
            title="Efficient Frontier",
            xaxis_title="Risk (Volatility %)",
            yaxis_title="Expected Return (%)",
            template=self.template
        )
        
        return fig
    
    def create_trade_execution_chart(self, execution_data: List[Dict[str, Any]]) -> go.Figure:
        """Create trade execution quality chart.
        
        Args:
            execution_data: List of execution records
            
        Returns:
            Plotly figure
        """
        if not execution_data:
            return go.Figure().add_annotation(text="No execution data available")
        
        # Extract data
        timestamps = [record.get('timestamp', datetime.now()) for record in execution_data]
        slippages = [abs(record.get('slippage', 0)) * 10000 for record in execution_data]  # Convert to bps
        costs = [record.get('cost', 0) for record in execution_data]
        
        fig = go.Figure()
        
        # Slippage over time
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=slippages,
            mode='markers+lines',
            name='Slippage (bps)',
            line=dict(color='#3498db'),
            yaxis='y'
        ))
        
        # Transaction costs
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=costs,
            mode='markers+lines',
            name='Transaction Cost ($)',
            line=dict(color='#e74c3c'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Trade Execution Quality",
            xaxis_title="Time",
            template=self.template,
            yaxis=dict(title="Slippage (bps)", side="left"),
            yaxis2=dict(title="Cost ($)", side="right", overlaying="y")
        )
        
        return fig
    
    def create_performance_attribution_chart(self, attribution_data: pd.DataFrame) -> go.Figure:
        """Create performance attribution chart.
        
        Args:
            attribution_data: Attribution data by factor/sector
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Stacked bar chart for attribution
        for column in attribution_data.columns:
            fig.add_trace(go.Bar(
                x=attribution_data.index,
                y=attribution_data[column],
                name=column
            ))
        
        fig.update_layout(
            title="Performance Attribution",
            xaxis_title="Period",
            yaxis_title="Attribution (%)",
            template=self.template,
            barmode='relative'
        )
        
        return fig
