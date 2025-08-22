"""Demonstration of the Factor Investing Strategy Builder Agent."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from analytics.strategy_builder import FactorInvestingStrategyBuilder
from pprint import pprint


def demo_custom_strategy_building():
    """Demonstrate custom strategy building capabilities."""
    print("=" * 60)
    print("FACTOR INVESTING STRATEGY BUILDER AGENT DEMO")
    print("=" * 60)
    
    # Initialize the strategy builder
    builder = FactorInvestingStrategyBuilder()
    
    print("\n1. BUILDING CUSTOM STRATEGIES")
    print("-" * 40)
    
    # Demo different investment objectives
    objectives = [
        {
            'objective': 'value',
            'risk_tolerance': 'conservative',
            'investment_horizon': 'long',
            'universe_type': 'large_cap'
        },
        {
            'objective': 'growth',
            'risk_tolerance': 'aggressive',
            'investment_horizon': 'medium',
            'universe_type': 'all_cap'
        },
        {
            'objective': 'balanced',
            'risk_tolerance': 'moderate',
            'investment_horizon': 'medium',
            'universe_type': 'large_cap'
        }
    ]
    
    strategies = []
    for params in objectives:
        print(f"\nBuilding {params['objective']} strategy...")
        strategy = builder.build_custom_strategy(**params)
        strategies.append(strategy)
        
        print(f"Strategy Name: {strategy.name}")
        print(f"Description: {strategy.description}")
        print(f"Factors: {[f.name for f in strategy.factors]}")
        print(f"Factor Weights: {[f'{f.name}: {f.weight:.2f}' for f in strategy.factors]}")
        print(f"Expected Return: {strategy.expected_performance['expected_annual_return']:.2%}")
        print(f"Expected Volatility: {strategy.expected_performance['expected_volatility']:.2%}")
        print(f"Expected Sharpe: {strategy.expected_performance['expected_sharpe_ratio']:.2f}")
    
    return strategies


def demo_template_recommendations():
    """Demonstrate template-based strategy recommendations."""
    print("\n\n2. TEMPLATE-BASED RECOMMENDATIONS")
    print("-" * 40)
    
    builder = FactorInvestingStrategyBuilder()
    
    # Sample user profiles
    user_profiles = [
        {
            'risk_tolerance': 'conservative',
            'investment_horizon': 'long',
            'objectives': ['value', 'income'],
            'expected_return': 0.08,
            'max_volatility': 0.12
        },
        {
            'risk_tolerance': 'aggressive',
            'investment_horizon': 'short',
            'objectives': ['growth', 'momentum'],
            'expected_return': 0.15,
            'max_volatility': 0.20
        }
    ]
    
    for i, profile in enumerate(user_profiles, 1):
        print(f"\nUser Profile {i}:")
        print(f"Risk Tolerance: {profile['risk_tolerance']}")
        print(f"Objectives: {profile['objectives']}")
        print(f"Expected Return: {profile['expected_return']:.1%}")
        
        recommendations = builder.recommend_strategy_from_template(profile)
        
        print(f"\nRecommended Strategies ({len(recommendations)} found):")
        for j, strategy in enumerate(recommendations, 1):
            print(f"  {j}. {strategy.name}")
            print(f"     Expected Return: {strategy.expected_performance['expected_annual_return']:.2%}")
            print(f"     Expected Vol: {strategy.expected_performance['expected_volatility']:.2%}")
            print(f"     Factors: {[f.name for f in strategy.factors]}")


def demo_factor_insights():
    """Demonstrate factor analysis and insights."""
    print("\n\n3. FACTOR INSIGHTS AND ANALYSIS")
    print("-" * 40)
    
    builder = FactorInvestingStrategyBuilder()
    
    # Generate sample market data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    market_data = pd.DataFrame({
        'returns': np.random.normal(0.0008, 0.015, len(dates))
    }, index=dates)
    
    # Analyze different factor combinations
    factor_sets = [
        ['value', 'momentum', 'quality'],
        ['momentum', 'low_volatility', 'size'],
        ['value', 'quality', 'profitability', 'low_volatility']
    ]
    
    for i, factors in enumerate(factor_sets, 1):
        print(f"\nFactor Set {i}: {factors}")
        insights = builder.generate_factor_insights(factors, market_data)
        
        print(f"Diversification Score: {insights['diversification_score']:.2f}")
        print("Factor Correlations:")
        corr_matrix = pd.DataFrame(insights['correlations'])
        for factor1 in factors:
            for factor2 in factors:
                if factor1 < factor2:  # Avoid duplicates
                    corr = corr_matrix.loc[factor1, factor2]
                    print(f"  {factor1} - {factor2}: {corr:.2f}")
        
        print("Recommendations:")
        for rec in insights['recommendations']:
            print(f"  • {rec}")


def demo_strategy_optimization():
    """Demonstrate strategy optimization capabilities."""
    print("\n\n4. STRATEGY OPTIMIZATION")
    print("-" * 40)
    
    builder = FactorInvestingStrategyBuilder()
    
    # Create a base strategy
    base_strategy = builder.build_custom_strategy(
        objective='balanced',
        risk_tolerance='moderate',
        investment_horizon='medium'
    )
    
    print(f"Original Strategy: {base_strategy.name}")
    print(f"Original Factors: {[f'{f.name}: {f.weight:.2f}' for f in base_strategy.factors]}")
    
    # Generate sample performance data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    performance_data = pd.DataFrame({
        'portfolio_value': np.cumprod(1 + np.random.normal(0.0008, 0.012, len(dates)))
    }, index=dates)
    
    # Optimize the strategy
    optimized_strategy = builder.optimize_existing_strategy(
        base_strategy, 
        performance_data,
        market_regime='normal'
    )
    
    print(f"\nOptimized Strategy: {optimized_strategy.name}")
    print(f"Optimized Factors: {[f'{f.name}: {f.weight:.2f}' for f in optimized_strategy.factors]}")
    
    # Compare expected performance
    print("\nPerformance Comparison:")
    print(f"Original Expected Return: {base_strategy.expected_performance['expected_annual_return']:.2%}")
    print(f"Optimized Expected Return: {optimized_strategy.expected_performance['expected_annual_return']:.2%}")
    print(f"Original Expected Vol: {base_strategy.expected_performance['expected_volatility']:.2%}")
    print(f"Optimized Expected Vol: {optimized_strategy.expected_performance['expected_volatility']:.2%}")


def demo_factor_research_insights():
    """Demonstrate research-backed factor insights."""
    print("\n\n5. ACADEMIC RESEARCH INSIGHTS")
    print("-" * 40)
    
    builder = FactorInvestingStrategyBuilder()
    
    print("Factor Templates Based on Academic Research:")
    print("-" * 50)
    
    for factor_name, template in builder.factor_templates.items():
        print(f"\n{factor_name.upper()} FACTOR:")
        print(f"  Research Basis: {template['description']}")
        print(f"  Expected Return: {template['expected_return']:.1%}")
        print(f"  Volatility: {template['volatility']:.1%}")
        print(f"  Sharpe Ratio: {template['sharpe_ratio']:.2f}")
        print(f"  Best Regimes: {', '.join(template['best_regimes'])}")
        print(f"  Market Correlation: {template['correlation_with_market']:.2f}")
        print(f"  Suggested Rebalancing: {template['rebalance_frequency']}")
    
    print("\n\nStrategy Templates:")
    print("-" * 20)
    
    for template in builder.strategy_templates:
        print(f"\n{template['name'].upper()}:")
        print(f"  Objective: {template['objective']}")
        print(f"  Risk Profile: {template['risk_profile']}")
        print(f"  Factors: {template['factors']}")
        print(f"  Expected Return: {template['expected_return']:.1%}")
        print(f"  Expected Volatility: {template['expected_volatility']:.1%}")
        print(f"  Max Drawdown: {template['max_drawdown']:.1%}")


def main():
    """Run the complete demonstration."""
    try:
        # Run all demonstrations
        strategies = demo_custom_strategy_building()
        demo_template_recommendations()
        demo_factor_insights()
        demo_strategy_optimization()
        demo_factor_research_insights()
        
        print("\n\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("\nThe Factor Investing Strategy Builder Agent provides:")
        print("* Custom strategy building based on objectives and risk tolerance")
        print("* Template-based recommendations for different investor profiles")
        print("* Factor analysis and correlation insights")
        print("* Strategy optimization based on performance data")
        print("* Research-backed factor templates from academic literature")
        print("* Actionable insights and recommendations")
        
        print(f"\nBuilt {len(strategies)} custom strategies successfully!")
        print("Ready for integration with the main investment engine.")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
