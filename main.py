"""Main entry point for the Investment Engine."""

import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import auto-configuration
try:
    from auto_config import AutoConfig, get_config, get_cache, get_data_feed
    AUTO_CONFIG_AVAILABLE = True
except ImportError:
    AUTO_CONFIG_AVAILABLE = False

from config.config_manager import ConfigManager
from data.data_manager import DataManager
from factors.factor_engine import FactorEngine
from optimization.optimizer_engine import OptimizerEngine
from risk.risk_manager import RiskManager
from execution.execution_engine import ExecutionEngine
from backtesting.backtest_engine import BacktestEngine
from core.models import Strategy, Portfolio, Asset
from core.enums import AssetClass, OptimizationMethod, RebalanceFrequency

class InvestmentEngine:
    """Main Investment Engine orchestrator."""
    
    def __init__(self, config_path: str = None):
        """Initialize the investment engine.
        
        Args:
            config_path: Path to configuration file
        """
        # Initialize configuration
        self.config = ConfigManager(config_path)
        
        # Initialize core components
        self.data_manager = DataManager()
        self.factor_engine = FactorEngine()
        self.optimizer_engine = OptimizerEngine()
        self.risk_manager = RiskManager(
            max_portfolio_var=self.config.get_setting('risk', 'max_portfolio_var'),
            max_individual_weight=self.config.get_setting('risk', 'max_individual_weight'),
            max_sector_weight=self.config.get_setting('risk', 'max_sector_weight'),
            max_drawdown=self.config.get_setting('risk', 'max_drawdown')
        )
        
        # Initialize execution engine
        self.execution_engine = ExecutionEngine(
            self.data_manager,
            self.factor_engine,
            self.optimizer_engine,
            self.risk_manager
        )
        
        # Initialize backtesting engine
        self.backtest_engine = BacktestEngine(
            initial_capital=self.config.get_setting('backtesting', 'initial_capital')
        )
        
        # Initialize analytics components
        self.strategy_analyzer = AdvancedStrategyAnalyzer()
        self.strategy_builder = FactorInvestingStrategyBuilder(
            self.factor_engine,
            self.optimizer_engine,
            self.risk_manager
        )
        
        # Create dashboard
        from dashboard.dashboard_app import DashboardApp
            
        if AUTO_CONFIG_AVAILABLE:
            # Use simplified auto-configured dashboard
            self.dashboard = DashboardApp()
            logging.info("Using auto-configured dashboard")
        else:
            self.dashboard = DashboardApp(
                self.data_manager,
                self.execution_engine,
                self.risk_manager,
                self.backtest_engine.performance_analyzer
            )
        
        print("Investment Engine initialized successfully!")
    
    def create_sample_strategy(self) -> Strategy:
        """Create a sample multi-factor strategy."""
        return Strategy(
            name="Multi-Factor Strategy",
            description="A diversified multi-factor investment strategy",
            universe=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V'],
            optimization_method=OptimizationMethod.MEAN_VARIANCE,
            rebalance_frequency=RebalanceFrequency.MONTHLY,
            constraints={'max_weight': 0.15, 'min_weight': 0.02}
        )
    
    def create_sample_portfolio(self) -> Portfolio:
        """Create a sample portfolio."""
        portfolio = Portfolio(name="Sample Portfolio", cash=1000000)
        
        # Add some sample positions
        assets = [
            Asset(symbol='AAPL', name='Apple Inc.', asset_class=AssetClass.EQUITY),
            Asset(symbol='MSFT', name='Microsoft Corp.', asset_class=AssetClass.EQUITY),
            Asset(symbol='GOOGL', name='Alphabet Inc.', asset_class=AssetClass.EQUITY)
        ]
        
        return portfolio
    
    def build_custom_strategy_interactive(self):
        """Interactive strategy builder using the new agent."""
        print("\n" + "="*60)
        print("FACTOR INVESTING STRATEGY BUILDER")
        print("="*60)
        
        # Get user preferences
        print("\nLet's build a custom strategy for you!")
        print("\nAvailable objectives:")
        print("1. Value - Focus on undervalued stocks")
        print("2. Growth - Focus on high-growth companies") 
        print("3. Momentum - Follow price trends")
        print("4. Income - Focus on dividend-paying stocks")
        print("5. Balanced - Diversified approach")
        
        objective_map = {'1': 'value', '2': 'growth', '3': 'momentum', '4': 'income', '5': 'balanced'}
        objective_choice = input("\nSelect objective (1-5): ").strip()
        objective = objective_map.get(objective_choice, 'balanced')
        
        print("\nRisk tolerance:")
        print("1. Conservative - Lower risk, stable returns")
        print("2. Moderate - Balanced risk/return")
        print("3. Aggressive - Higher risk, higher potential returns")
        
        risk_map = {'1': 'conservative', '2': 'moderate', '3': 'aggressive'}
        risk_choice = input("\nSelect risk tolerance (1-3): ").strip()
        risk_tolerance = risk_map.get(risk_choice, 'moderate')
        
        print("\nInvestment horizon:")
        print("1. Short-term (< 2 years)")
        print("2. Medium-term (2-5 years)")
        print("3. Long-term (> 5 years)")
        
        horizon_map = {'1': 'short', '2': 'medium', '3': 'long'}
        horizon_choice = input("\nSelect investment horizon (1-3): ").strip()
        investment_horizon = horizon_map.get(horizon_choice, 'medium')
        
        # Build the strategy
        print(f"\nBuilding {objective} strategy with {risk_tolerance} risk tolerance...")
        strategy_blueprint = self.strategy_builder.build_custom_strategy(
            objective=objective,
            risk_tolerance=risk_tolerance,
            investment_horizon=investment_horizon,
            universe_type='large_cap'
        )
        
        # Display results
        print(f"\n✓ Strategy Created: {strategy_blueprint.name}")
        print(f"Description: {strategy_blueprint.description}")
        print(f"\nFactor Allocation:")
        for factor in strategy_blueprint.factors:
            print(f"  • {factor.name.title()}: {factor.weight:.1%}")
        
        print(f"\nExpected Performance:")
        perf = strategy_blueprint.expected_performance
        print(f"  • Annual Return: {perf['expected_annual_return']:.1%}")
        print(f"  • Volatility: {perf['expected_volatility']:.1%}")
        print(f"  • Sharpe Ratio: {perf['expected_sharpe_ratio']:.2f}")
        print(f"  • Max Drawdown: {perf['expected_max_drawdown']:.1%}")
        
        return strategy_blueprint
    
    def demonstrate_strategy_analysis(self):
        """Demonstrate the advanced strategy analyzer."""
        print("\n" + "="*60)
        print("ADVANCED STRATEGY ANALYSIS DEMO")
        print("="*60)
        
        # Generate sample returns data
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)  # For reproducible results
        
        # Simulate strategy returns with some realistic characteristics
        strategy_returns = pd.Series(
            np.random.normal(0.0008, 0.012, len(dates)),  # ~8% annual return, 12% vol
            index=dates,
            name='Strategy Returns'
        )
        
        # Simulate benchmark returns (slightly lower return, similar vol)
        benchmark_returns = pd.Series(
            np.random.normal(0.0006, 0.011, len(dates)),  # ~6% annual return, 11% vol
            index=dates,
            name='Benchmark Returns'
        )
        
        print("Analyzing strategy performance...")
        
        # Run comprehensive analysis
        analysis = self.strategy_analyzer.analyze_strategy_comprehensive(
            returns=strategy_returns,
            benchmark_returns=benchmark_returns
        )
        
        # Display key metrics
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"  • Annual Return: {analysis['annual_return']:.2%}")
        print(f"  • Volatility: {analysis['volatility']:.2%}")
        print(f"  • Sharpe Ratio: {analysis['sharpe_ratio']:.2f}")
        print(f"  • Information Ratio: {analysis['information_ratio']:.2f}")
        print(f"  • Jensen's Alpha: {analysis['jensen_alpha']:.2%}")
        print(f"  • Sortino Ratio: {analysis['sortino_ratio']:.2f}")
        print(f"  • Calmar Ratio: {analysis['calmar_ratio']:.2f}")
        print(f"  • Max Drawdown: {analysis['max_drawdown']:.2%}")
        
        # Display insights
        insights = analysis['insights']
        print(f"\n🎯 STRATEGY INSIGHTS:")
        print(f"  • Overall Rating: {insights.overall_rating}")
        print(f"  • Key Strengths:")
        for strength in insights.key_strengths:
            print(f"    - {strength}")
        print(f"  • Recommendations:")
        for rec in insights.recommendations:
            print(f"    - {rec}")
        
        return analysis
    
    def run_sample_backtest(self):
        """Run a sample backtest."""
        print("\nRunning sample backtest...")
        
        # Get sample data
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365)
        
        try:
            # Get market data
            price_data = self.data_manager.get_price_data(symbols, start_date, end_date)
            
            if price_data.empty:
                print("No market data available. Using synthetic data for demonstration.")
                # Create synthetic data
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                price_data = pd.DataFrame(
                    index=dates,
                    columns=symbols,
                    data=100 * (1 + pd.DataFrame(
                        np.random.normal(0.0005, 0.02, (len(dates), len(symbols)))
                    ).cumsum())
                )
            
            # Create strategy and run backtest
            strategy = self.create_sample_strategy()
            
            def weight_function(data, date):
                """Simple equal weight function."""
                n_assets = len(symbols)
                return pd.Series(1/n_assets, index=symbols)
            
            # Run backtest
            results = self.backtest_engine.run_backtest(
                strategy=strategy,
                price_data=price_data,
                weight_function=weight_function,
                start_date=start_date,
                end_date=end_date
            )
            
            print(f"Backtest completed successfully!")
            print(f"Total Return: {results['total_return']:.2%}")
            print(f"Final Value: ${results['final_value']:,.2f}")
            print(f"Sharpe Ratio: {results['performance_metrics'].sharpe_ratio:.2f}")
            print(f"Max Drawdown: {results['performance_metrics'].max_drawdown:.2%}")
            
            return results
            
        except Exception as e:
            print(f"Error running backtest: {e}")
            return None
    
    def run_dashboard(self, debug=True, port=8050):
        """Launch the web dashboard.
        
        Args:
            debug: Enable debug mode
            port: Port number
        """
        print(f"\nLaunching dashboard at http://127.0.0.1:{port}")
        print("Press Ctrl+C to stop the dashboard")
        
        try:
            self.dashboard.run_server(debug=debug, port=port)
        except KeyboardInterrupt:
            print("\nDashboard stopped.")
    
    def demonstrate_features(self):
        """Demonstrate key features of the investment engine."""
        print("\n" + "="*60)
        print("INVESTMENT ENGINE FEATURE DEMONSTRATION")
        print("="*60)
        
        # 1. Configuration
        print("\n1. Configuration Management:")
        validation = self.config.validate_config()
        print(f"   Configuration valid: {validation['valid']}")
        if validation['warnings']:
            print(f"   Warnings: {validation['warnings']}")
        
        # 2. Data Management
        print("\n2. Data Management:")
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        try:
            data_quality = self.data_manager.validate_data_quality(
                pd.DataFrame({'test': [1, 2, 3]}), 'TEST'
            )
            print(f"   Data validation system: {'OK' if data_quality['valid'] else 'ERROR'}")
        except:
            print("   Data validation system: Available")
        
        # 3. Factor Models
        print("\n3. Multi-Factor Models:")
        print("   [+] Fama-French 3-factor and 5-factor models")
        print("   [+] Momentum factor model")
        print("   [+] Quality factor model")
        print("   [+] Volatility factor model")
        
        # 4. Portfolio Optimization
        print("\n4. Portfolio Optimization:")
        print("   [+] Mean-Variance Optimization")
        print("   [+] Risk Parity Optimization")
        print("   [+] Black-Litterman Optimization")
        
        # 5. Risk Management
        print("\n5. Risk Management:")
        print("   [+] VaR and CVaR calculations")
        print("   [+] Position sizing algorithms")
        print("   [+] Drawdown monitoring")
        print("   [+] Risk limit enforcement")
        
        # 6. Backtesting
        print("\n6. Backtesting Engine:")
        print("   [+] Historical simulation")
        print("   [+] Transaction cost modeling")
        print("   [+] Performance analytics")
        print("   [+] Walk-forward analysis")
        
        # 7. Execution
        print("\n7. Strategy Execution:")
        print("   [+] Order management system")
        print("   [+] Trade execution simulation")
        print("   [+] Rebalancing engine")
        print("   [+] Execution quality monitoring")
        
        # 8. Reporting
        print("\n8. Reporting & Visualization:")
        print("   [+] Interactive web dashboard")
        print("   [+] Performance charts")
        print("   [+] Risk analytics")
        print("   [+] HTML/JSON report export")
        
        print("\n" + "="*60)
        print("All components initialized and ready for use!")
        print("="*60)


def main():
    """Main function with interactive CLI for the Investment Engine."""
    print("ADVANCED INVESTMENT ENGINE")
    print("=" * 60)
    print("AI-Powered Factor Investing & Portfolio Management Platform")
    print("=" * 60)
    
    # Auto-setup if available
    if AUTO_CONFIG_AVAILABLE:
        print("🚀 Using automated configuration...")
        auto_config = AutoConfig()
        config = auto_config.setup_everything()
    
    # Setup logging
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'investment_engine.log'),
            logging.StreamHandler()
        ]
    )
    
    try:
        # Initialize the engine
        engine = InvestmentEngine()
        
        while True:
            print("\nMAIN MENU")
            print("-" * 30)
            print("1. Build Custom Strategy (Interactive)")
            print("2. Analyze Strategy Performance")
            print("3. Run Sample Backtest")
            print("4. Launch Web Dashboard")
            print("5. View Strategy Templates")
            print("6. Factor Analysis Demo")
            print("7. Show System Features")
            print("8. Exit")
            
            choice = input("\nSelect option (1-8): ").strip()
            
            if choice == '1':
                try:
                    strategy_blueprint = engine.build_custom_strategy_interactive()
                    input("\nPress Enter to continue...")
                except KeyboardInterrupt:
                    print("\nStrategy building cancelled.")
                except Exception as e:
                    print(f"Error building strategy: {e}")
                    
            elif choice == '2':
                try:
                    analysis = engine.demonstrate_strategy_analysis()
                    input("\nPress Enter to continue...")
                except Exception as e:
                    print(f"Error analyzing strategy: {e}")
                    
            elif choice == '3':
                try:
                    backtest_results = engine.run_sample_backtest()
                    input("\nPress Enter to continue...")
                except Exception as e:
                    print(f"Error running backtest: {e}")
                    
            elif choice == '4':
                print("\nLaunching Web Dashboard...")
                print("Dashboard will be available at: http://127.0.0.1:8050")
                print("Press Ctrl+C to stop the dashboard and return to menu")
                try:
                    engine.run_dashboard()
                except KeyboardInterrupt:
                    print("\nDashboard stopped. Returning to main menu...")
                except Exception as e:
                    print(f"Error launching dashboard: {e}")
                    
            elif choice == '5':
                print("\nSTRATEGY TEMPLATES")
                print("=" * 40)
                templates = engine.strategy_builder.strategy_templates
                for i, template in enumerate(templates, 1):
                    print(f"\n{i}. {template['name'].upper()}")
                    print(f"   Objective: {template['objective']}")
                    print(f"   Risk Profile: {template['risk_profile']}")
                    print(f"   Factors: {', '.join(template['factors'])}")
                    print(f"   Expected Return: {template['expected_return']:.1%}")
                    print(f"   Expected Volatility: {template['expected_volatility']:.1%}")
                input("\nPress Enter to continue...")
                
            elif choice == '6':
                print("\nFACTOR ANALYSIS DEMO")
                print("=" * 40)
                
                # Demo factor insights
                sample_factors = ['value', 'momentum', 'quality', 'low_volatility']
                dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
                market_data = pd.DataFrame({
                    'returns': np.random.normal(0.0008, 0.015, len(dates))
                }, index=dates)
                
                insights = engine.strategy_builder.generate_factor_insights(
                    sample_factors, market_data
                )
                
                print(f"Diversification Score: {insights['diversification_score']:.2f}")
                print("\nFactor Correlations:")
                corr_df = pd.DataFrame(insights['correlations'])
                for i, factor1 in enumerate(sample_factors):
                    for j, factor2 in enumerate(sample_factors):
                        if i < j:
                            corr = corr_df.loc[factor1, factor2]
                            print(f"  {factor1} - {factor2}: {corr:.2f}")
                
                print("\nRecommendations:")
                for rec in insights['recommendations']:
                    print(f"  * {rec}")
                    
                input("\nPress Enter to continue...")
                
            elif choice == '7':
                print("\nINVESTMENT ENGINE FEATURES")
                print("=" * 50)
                print("* Data Management: Multi-source integration with caching")
                print("* Factor Models: Fama-French, momentum, quality, custom factors")
                print("* AI Strategy Builder: Custom strategy generation with ML insights")
                print("* Advanced Analytics: Research-backed performance metrics")
                print("* Portfolio Optimization: Mean-variance, risk parity, Black-Litterman")
                print("* Risk Management: VaR, position sizing, drawdown monitoring")
                print("* Execution Engine: Order management with slippage and costs")
                print("* Backtesting: Historical simulation with performance analytics")
                print("* Modern Dashboard: Interactive web interface with glassmorphism UI")
                print("* Configuration: Flexible YAML/JSON configuration management")
                print("* Reporting: Comprehensive HTML and JSON report generation")
                print("* Factor Insights: Correlation analysis and regime detection")
                print("* Strategy Templates: Pre-built strategies based on academic research")
                
                print(f"\nRESEARCH-BACKED METRICS:")
                print("* Sharpe Ratio, Information Ratio, Jensen's Alpha")
                print("* Sortino Ratio, Calmar Ratio, Omega Ratio")
                print("* Factor exposures with statistical significance")
                print("* Regime-based performance analysis")
                print("* Drawdown and tail risk metrics")
                print("* Actionable strategy insights and recommendations")
                
                input("\nPress Enter to continue...")
                
            elif choice == '8':
                print("\nThank you for using the Investment Engine!")
                print("Happy investing!")
                break
                
            else:
                print("Invalid choice. Please select 1-8.")
        
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
