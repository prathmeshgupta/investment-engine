"""Report generation utilities for investment analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
from jinja2 import Template
import base64
import io

from backtesting.performance_analyzer import PerformanceAnalyzer
from .visualizer import Visualizer


class ReportGenerator:
    """Generates comprehensive investment reports."""
    
    def __init__(self):
        """Initialize report generator."""
        self.visualizer = Visualizer()
        self.performance_analyzer = PerformanceAnalyzer()
    
    def generate_portfolio_report(self,
                                portfolio_data: pd.DataFrame,
                                holdings_data: pd.DataFrame,
                                benchmark_data: Optional[pd.DataFrame] = None,
                                report_title: str = "Portfolio Performance Report") -> Dict[str, Any]:
        """Generate comprehensive portfolio report.
        
        Args:
            portfolio_data: Portfolio performance data
            holdings_data: Current holdings data
            benchmark_data: Benchmark comparison data
            report_title: Report title
            
        Returns:
            Report data dictionary
        """
        # Calculate performance metrics
        if 'returns' in portfolio_data.columns:
            returns = portfolio_data['returns'].dropna()
        else:
            returns = portfolio_data.pct_change().iloc[:, 0].dropna()
        
        benchmark_returns = None
        if benchmark_data is not None:
            benchmark_returns = benchmark_data.pct_change().iloc[:, 0].dropna()
        
        performance_metrics = self.performance_analyzer.calculate_metrics(
            returns, benchmark_returns
        )
        
        # Generate visualizations
        charts = self._generate_report_charts(portfolio_data, holdings_data, benchmark_data)
        
        # Portfolio summary
        current_value = portfolio_data.iloc[-1, 0] if not portfolio_data.empty else 0
        initial_value = portfolio_data.iloc[0, 0] if not portfolio_data.empty else 0
        total_return = (current_value - initial_value) / initial_value if initial_value > 0 else 0
        
        portfolio_summary = {
            'current_value': current_value,
            'initial_value': initial_value,
            'total_return': total_return,
            'period_start': portfolio_data.index[0] if not portfolio_data.empty else datetime.now(),
            'period_end': portfolio_data.index[-1] if not portfolio_data.empty else datetime.now(),
            'number_of_holdings': len(holdings_data)
        }
        
        # Risk analysis
        risk_analysis = self._analyze_portfolio_risk(returns, holdings_data)
        
        # Holdings analysis
        holdings_analysis = self._analyze_holdings(holdings_data)
        
        return {
            'report_title': report_title,
            'generation_date': datetime.now(),
            'portfolio_summary': portfolio_summary,
            'performance_metrics': performance_metrics.__dict__,
            'risk_analysis': risk_analysis,
            'holdings_analysis': holdings_analysis,
            'charts': charts
        }
    
    def _generate_report_charts(self,
                              portfolio_data: pd.DataFrame,
                              holdings_data: pd.DataFrame,
                              benchmark_data: Optional[pd.DataFrame] = None) -> Dict[str, str]:
        """Generate charts for the report.
        
        Returns:
            Dictionary of chart names to base64 encoded images
        """
        charts = {}
        
        # Performance chart
        perf_fig = self.visualizer.create_performance_chart(portfolio_data, benchmark_data)
        charts['performance'] = self._fig_to_base64(perf_fig)
        
        # Allocation chart
        allocation_fig = self.visualizer.create_allocation_chart(holdings_data)
        charts['allocation'] = self._fig_to_base64(allocation_fig)
        
        # Returns distribution
        if 'returns' in portfolio_data.columns:
            returns = portfolio_data['returns'].dropna()
        else:
            returns = portfolio_data.pct_change().iloc[:, 0].dropna()
        
        dist_fig = self.visualizer.create_returns_distribution(returns)
        charts['returns_distribution'] = self._fig_to_base64(dist_fig)
        
        # Drawdown chart
        dd_fig = self.visualizer.create_drawdown_chart(portfolio_data)
        charts['drawdown'] = self._fig_to_base64(dd_fig)
        
        return charts
    
    def _fig_to_base64(self, fig) -> str:
        """Convert plotly figure to base64 string."""
        img_bytes = fig.to_image(format="png", width=800, height=500)
        img_base64 = base64.b64encode(img_bytes).decode()
        return f"data:image/png;base64,{img_base64}"
    
    def _analyze_portfolio_risk(self,
                              returns: pd.Series,
                              holdings_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze portfolio risk metrics."""
        # Basic risk metrics
        annual_vol = returns.std() * np.sqrt(252)
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # VaR calculations
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Concentration risk
        if 'Weight' in holdings_data.columns:
            weights = holdings_data['Weight']
            hhi = (weights ** 2).sum()
            effective_assets = 1 / hhi if hhi > 0 else 0
            max_weight = weights.max()
        else:
            hhi = 0
            effective_assets = 0
            max_weight = 0
        
        return {
            'annual_volatility': annual_vol,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95': abs(var_95),
            'var_99': abs(var_99),
            'cvar_95': abs(cvar_95),
            'herfindahl_index': hhi,
            'effective_number_assets': effective_assets,
            'max_position_weight': max_weight
        }
    
    def _analyze_holdings(self, holdings_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current holdings."""
        analysis = {}
        
        if 'Market Value' in holdings_data.columns:
            total_value = holdings_data['Market Value'].sum()
            analysis['total_market_value'] = total_value
            analysis['largest_holding'] = holdings_data.loc[holdings_data['Market Value'].idxmax(), 'Symbol']
            analysis['largest_holding_value'] = holdings_data['Market Value'].max()
            analysis['smallest_holding'] = holdings_data.loc[holdings_data['Market Value'].idxmin(), 'Symbol']
            analysis['smallest_holding_value'] = holdings_data['Market Value'].min()
        
        if 'P&L' in holdings_data.columns:
            total_pnl = holdings_data['P&L'].sum()
            winners = (holdings_data['P&L'] > 0).sum()
            losers = (holdings_data['P&L'] < 0).sum()
            
            analysis['total_unrealized_pnl'] = total_pnl
            analysis['winning_positions'] = winners
            analysis['losing_positions'] = losers
            analysis['win_rate'] = winners / len(holdings_data) if len(holdings_data) > 0 else 0
        
        return analysis
    
    def generate_risk_report(self,
                           portfolio_data: pd.DataFrame,
                           risk_metrics: Dict[str, Any],
                           var_data: Dict[str, float]) -> Dict[str, Any]:
        """Generate risk-focused report.
        
        Args:
            portfolio_data: Portfolio performance data
            risk_metrics: Risk metrics dictionary
            var_data: VaR analysis data
            
        Returns:
            Risk report data
        """
        # Calculate additional risk metrics
        if 'returns' in portfolio_data.columns:
            returns = portfolio_data['returns'].dropna()
        else:
            returns = portfolio_data.pct_change().iloc[:, 0].dropna()
        
        # Rolling risk metrics
        rolling_vol = returns.rolling(252).std() * np.sqrt(252)
        rolling_var = returns.rolling(252).quantile(0.05)
        
        # Stress test scenarios
        stress_scenarios = self._generate_stress_scenarios(returns)
        
        # Risk decomposition
        risk_decomposition = self._decompose_risk(returns)
        
        return {
            'report_title': 'Portfolio Risk Analysis Report',
            'generation_date': datetime.now(),
            'current_risk_metrics': risk_metrics,
            'var_analysis': var_data,
            'rolling_metrics': {
                'volatility': rolling_vol.iloc[-1] if not rolling_vol.empty else 0,
                'var_95': abs(rolling_var.iloc[-1]) if not rolling_var.empty else 0
            },
            'stress_scenarios': stress_scenarios,
            'risk_decomposition': risk_decomposition
        }
    
    def _generate_stress_scenarios(self, returns: pd.Series) -> Dict[str, float]:
        """Generate stress test scenarios."""
        scenarios = {}
        
        # Historical worst periods
        scenarios['worst_day'] = returns.min()
        scenarios['worst_week'] = returns.rolling(5).sum().min()
        scenarios['worst_month'] = returns.rolling(21).sum().min()
        
        # Percentile scenarios
        scenarios['5th_percentile'] = np.percentile(returns, 5)
        scenarios['1st_percentile'] = np.percentile(returns, 1)
        
        # Volatility scenarios
        current_vol = returns.std()
        scenarios['2x_volatility'] = -2 * current_vol
        scenarios['3x_volatility'] = -3 * current_vol
        
        return scenarios
    
    def _decompose_risk(self, returns: pd.Series) -> Dict[str, Any]:
        """Decompose portfolio risk into components."""
        # Simple risk decomposition
        total_var = returns.var()
        
        # Time-based decomposition (simplified)
        daily_var = returns.var()
        weekly_var = returns.rolling(5).sum().var() / 5
        monthly_var = returns.rolling(21).sum().var() / 21
        
        return {
            'total_variance': total_var,
            'daily_component': daily_var / total_var if total_var > 0 else 0,
            'weekly_component': weekly_var / total_var if total_var > 0 else 0,
            'monthly_component': monthly_var / total_var if total_var > 0 else 0
        }
    
    def generate_execution_report(self,
                                execution_history: List[Dict[str, Any]],
                                trade_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate execution quality report.
        
        Args:
            execution_history: Strategy execution history
            trade_data: Individual trade execution data
            
        Returns:
            Execution report data
        """
        # Execution statistics
        total_executions = len(execution_history)
        successful_executions = sum(1 for ex in execution_history if ex.get('status') == 'completed')
        
        # Trade statistics
        total_trades = len(trade_data)
        successful_trades = sum(1 for trade in trade_data if trade.get('status') == 'filled')
        
        # Cost analysis
        total_costs = sum(trade.get('cost', 0) for trade in trade_data)
        avg_cost_per_trade = total_costs / total_trades if total_trades > 0 else 0
        
        # Slippage analysis
        slippages = [abs(trade.get('slippage', 0)) for trade in trade_data if 'slippage' in trade]
        avg_slippage = np.mean(slippages) if slippages else 0
        max_slippage = max(slippages) if slippages else 0
        
        return {
            'report_title': 'Execution Quality Report',
            'generation_date': datetime.now(),
            'execution_summary': {
                'total_executions': total_executions,
                'successful_executions': successful_executions,
                'success_rate': successful_executions / total_executions if total_executions > 0 else 0
            },
            'trade_summary': {
                'total_trades': total_trades,
                'successful_trades': successful_trades,
                'fill_rate': successful_trades / total_trades if total_trades > 0 else 0
            },
            'cost_analysis': {
                'total_transaction_costs': total_costs,
                'average_cost_per_trade': avg_cost_per_trade,
                'cost_as_percentage': 0  # Would need portfolio value to calculate
            },
            'slippage_analysis': {
                'average_slippage_bps': avg_slippage * 10000,
                'maximum_slippage_bps': max_slippage * 10000,
                'slippage_volatility': np.std(slippages) * 10000 if slippages else 0
            }
        }
    
    def export_to_html(self, report_data: Dict[str, Any], output_path: str) -> None:
        """Export report to HTML format.
        
        Args:
            report_data: Report data dictionary
            output_path: Output file path
        """
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ report_title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                .header { text-align: center; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 20px; }
                .section { margin: 30px 0; }
                .metric { display: inline-block; margin: 10px; padding: 15px; background-color: #ecf0f1; border-radius: 5px; }
                .chart { text-align: center; margin: 20px 0; }
                .chart img { max-width: 100%; height: auto; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #3498db; color: white; }
                .positive { color: #27ae60; }
                .negative { color: #e74c3c; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ report_title }}</h1>
                <p>Generated on {{ generation_date.strftime('%Y-%m-%d %H:%M:%S') }}</p>
            </div>
            
            {% if portfolio_summary %}
            <div class="section">
                <h2>Portfolio Summary</h2>
                <div class="metric">
                    <h4>Current Value</h4>
                    <h3>${{ "{:,.2f}".format(portfolio_summary.current_value) }}</h3>
                </div>
                <div class="metric">
                    <h4>Total Return</h4>
                    <h3 class="{{ 'positive' if portfolio_summary.total_return > 0 else 'negative' }}">
                        {{ "{:.2%}".format(portfolio_summary.total_return) }}
                    </h3>
                </div>
                <div class="metric">
                    <h4>Number of Holdings</h4>
                    <h3>{{ portfolio_summary.number_of_holdings }}</h3>
                </div>
            </div>
            {% endif %}
            
            {% if performance_metrics %}
            <div class="section">
                <h2>Performance Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Annualized Return</td><td>{{ "{:.2%}".format(performance_metrics.annualized_return) }}</td></tr>
                    <tr><td>Volatility</td><td>{{ "{:.2%}".format(performance_metrics.volatility) }}</td></tr>
                    <tr><td>Sharpe Ratio</td><td>{{ "{:.2f}".format(performance_metrics.sharpe_ratio) }}</td></tr>
                    <tr><td>Maximum Drawdown</td><td>{{ "{:.2%}".format(performance_metrics.max_drawdown) }}</td></tr>
                    <tr><td>Calmar Ratio</td><td>{{ "{:.2f}".format(performance_metrics.calmar_ratio) }}</td></tr>
                </table>
            </div>
            {% endif %}
            
            {% if charts %}
            <div class="section">
                <h2>Charts</h2>
                {% for chart_name, chart_data in charts.items() %}
                <div class="chart">
                    <h3>{{ chart_name.replace('_', ' ').title() }}</h3>
                    <img src="{{ chart_data }}" alt="{{ chart_name }}">
                </div>
                {% endfor %}
            </div>
            {% endif %}
            
            {% if risk_analysis %}
            <div class="section">
                <h2>Risk Analysis</h2>
                <table>
                    <tr><th>Risk Metric</th><th>Value</th></tr>
                    <tr><td>Annual Volatility</td><td>{{ "{:.2%}".format(risk_analysis.annual_volatility) }}</td></tr>
                    <tr><td>VaR (95%)</td><td>{{ "{:.2%}".format(risk_analysis.var_95) }}</td></tr>
                    <tr><td>CVaR (95%)</td><td>{{ "{:.2%}".format(risk_analysis.cvar_95) }}</td></tr>
                    <tr><td>Skewness</td><td>{{ "{:.2f}".format(risk_analysis.skewness) }}</td></tr>
                    <tr><td>Kurtosis</td><td>{{ "{:.2f}".format(risk_analysis.kurtosis) }}</td></tr>
                </table>
            </div>
            {% endif %}
        </body>
        </html>
        """
        
        template = Template(html_template)
        html_content = template.render(**report_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def export_to_json(self, report_data: Dict[str, Any], output_path: str) -> None:
        """Export report to JSON format.
        
        Args:
            report_data: Report data dictionary
            output_path: Output file path
        """
        # Convert datetime objects to strings for JSON serialization
        json_data = self._prepare_for_json(report_data)
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
    
    def _prepare_for_json(self, data: Any) -> Any:
        """Prepare data for JSON serialization."""
        if isinstance(data, dict):
            return {key: self._prepare_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        elif isinstance(data, (datetime, pd.Timestamp)):
            return data.isoformat()
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        elif pd.isna(data):
            return None
        else:
            return data
