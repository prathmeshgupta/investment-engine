"""Factor Investing Strategy Builder Agent."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from core.models import Strategy, Asset
from core.enums import AssetClass, OptimizationMethod, RebalanceFrequency
from factors.factor_engine import FactorEngine
from optimization.optimizer_engine import OptimizerEngine
from risk.risk_manager import RiskManager


@dataclass
class FactorConfig:
    """Configuration for individual factors."""
    name: str
    weight: float
    lookback_period: int
    rebalance_frequency: str
    enabled: bool = True
    min_weight: float = 0.0
    max_weight: float = 1.0


@dataclass
class StrategyBlueprint:
    """Complete strategy blueprint."""
    name: str
    description: str
    factors: List[FactorConfig]
    universe: List[str]
    optimization_method: str
    risk_constraints: Dict[str, float]
    rebalance_frequency: str
    expected_performance: Dict[str, float]
    risk_profile: str


class FactorInvestingStrategyBuilder:
    """AI-powered factor investing strategy builder."""
    
    def __init__(self, factor_engine: FactorEngine = None, 
                 optimizer_engine: OptimizerEngine = None,
                 risk_manager: RiskManager = None):
        """Initialize the strategy builder."""
        self.factor_engine = factor_engine or FactorEngine()
        self.optimizer_engine = optimizer_engine or OptimizerEngine()
        self.risk_manager = risk_manager or RiskManager()
        
        # Pre-defined factor templates based on academic research
        self.factor_templates = self._initialize_factor_templates()
        self.strategy_templates = self._initialize_strategy_templates()
    
    def build_custom_strategy(self, 
                            objective: str,
                            risk_tolerance: str,
                            investment_horizon: str,
                            universe_type: str = "large_cap",
                            factor_preferences: Optional[List[str]] = None,
                            constraints: Optional[Dict[str, float]] = None) -> StrategyBlueprint:
        """Build a custom factor investing strategy based on user preferences.
        
        Args:
            objective: Investment objective ('growth', 'value', 'income', 'balanced', 'momentum')
            risk_tolerance: Risk tolerance ('conservative', 'moderate', 'aggressive')
            investment_horizon: Investment horizon ('short', 'medium', 'long')
            universe_type: Universe type ('large_cap', 'small_cap', 'all_cap', 'international')
            factor_preferences: Preferred factors to emphasize
            constraints: Additional constraints
            
        Returns:
            Complete strategy blueprint
        """
        # Select appropriate factors based on objective and research
        selected_factors = self._select_factors_for_objective(
            objective, risk_tolerance, factor_preferences
        )
        
        # Determine universe
        universe = self._get_universe_for_type(universe_type)
        
        # Set optimization method based on risk tolerance
        optimization_method = self._get_optimization_method(risk_tolerance)
        
        # Configure risk constraints
        risk_constraints = self._configure_risk_constraints(
            risk_tolerance, constraints
        )
        
        # Set rebalancing frequency based on horizon and factors
        rebalance_freq = self._determine_rebalance_frequency(
            investment_horizon, selected_factors
        )
        
        # Generate strategy name and description
        strategy_name = self._generate_strategy_name(objective, selected_factors)
        description = self._generate_strategy_description(
            objective, selected_factors, risk_tolerance
        )
        
        # Estimate expected performance
        expected_performance = self._estimate_performance(
            selected_factors, risk_tolerance
        )
        
        return StrategyBlueprint(
            name=strategy_name,
            description=description,
            factors=selected_factors,
            universe=universe,
            optimization_method=optimization_method,
            risk_constraints=risk_constraints,
            rebalance_frequency=rebalance_freq,
            expected_performance=expected_performance,
            risk_profile=risk_tolerance
        )
    
    def recommend_strategy_from_template(self, 
                                       user_profile: Dict[str, Any]) -> List[StrategyBlueprint]:
        """Recommend strategies from templates based on user profile.
        
        Args:
            user_profile: User investment profile
            
        Returns:
            List of recommended strategy blueprints
        """
        recommendations = []
        
        risk_tolerance = user_profile.get('risk_tolerance', 'moderate')
        investment_horizon = user_profile.get('investment_horizon', 'medium')
        objectives = user_profile.get('objectives', ['balanced'])
        
        # Score each template based on user profile
        template_scores = []
        for template in self.strategy_templates:
            score = self._score_template_fit(template, user_profile)
            template_scores.append((template, score))
        
        # Sort by score and return top 3
        template_scores.sort(key=lambda x: x[1], reverse=True)
        
        for template, score in template_scores[:3]:
            if score > 0.6:  # Minimum fit threshold
                # Customize template for user
                customized = self._customize_template(template, user_profile)
                recommendations.append(customized)
        
        return recommendations
    
    def optimize_existing_strategy(self, 
                                 strategy: StrategyBlueprint,
                                 performance_data: pd.DataFrame,
                                 market_regime: str = "normal") -> StrategyBlueprint:
        """Optimize an existing strategy based on performance data.
        
        Args:
            strategy: Current strategy blueprint
            performance_data: Historical performance data
            market_regime: Current market regime
            
        Returns:
            Optimized strategy blueprint
        """
        # Analyze current performance
        returns = performance_data.pct_change().dropna()
        
        # Identify underperforming factors
        factor_performance = self._analyze_factor_performance(
            strategy.factors, returns
        )
        
        # Suggest factor weight adjustments
        optimized_factors = self._optimize_factor_weights(
            strategy.factors, factor_performance, market_regime
        )
        
        # Update risk constraints if needed
        updated_constraints = self._update_risk_constraints(
            strategy.risk_constraints, returns
        )
        
        # Create optimized strategy
        optimized_strategy = StrategyBlueprint(
            name=f"{strategy.name} (Optimized)",
            description=f"{strategy.description} - Optimized for current market conditions",
            factors=optimized_factors,
            universe=strategy.universe,
            optimization_method=strategy.optimization_method,
            risk_constraints=updated_constraints,
            rebalance_frequency=strategy.rebalance_frequency,
            expected_performance=self._estimate_performance(optimized_factors, strategy.risk_profile),
            risk_profile=strategy.risk_profile
        )
        
        return optimized_strategy
    
    def generate_factor_insights(self, 
                               factors: List[str],
                               market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate insights about factor performance and correlations.
        
        Args:
            factors: List of factor names
            market_data: Market data for analysis
            
        Returns:
            Factor insights and recommendations
        """
        insights = {}
        
        # Calculate factor returns (simplified)
        factor_returns = {}
        for factor in factors:
            if factor in self.factor_templates:
                # Simulate factor returns based on template characteristics
                template = self.factor_templates[factor]
                factor_returns[factor] = self._simulate_factor_returns(
                    template, market_data
                )
        
        # Analyze correlations
        factor_df = pd.DataFrame(factor_returns)
        correlation_matrix = factor_df.corr()
        
        # Identify factor regimes
        regime_analysis = self._analyze_factor_regimes(factor_df)
        
        # Generate recommendations
        recommendations = self._generate_factor_recommendations(
            factor_df, correlation_matrix, regime_analysis
        )
        
        insights = {
            'factor_returns': factor_returns,
            'correlations': correlation_matrix.to_dict(),
            'regime_analysis': regime_analysis,
            'recommendations': recommendations,
            'diversification_score': self._calculate_diversification_score(correlation_matrix)
        }
        
        return insights
    
    def _initialize_factor_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize factor templates based on academic research."""
        return {
            'value': {
                'description': 'Value factor (HML) - Fama & French',
                'expected_return': 0.04,
                'volatility': 0.12,
                'sharpe_ratio': 0.33,
                'best_regimes': ['recession', 'recovery'],
                'correlation_with_market': 0.1,
                'lookback_period': 252,
                'rebalance_frequency': 'quarterly'
            },
            'momentum': {
                'description': 'Momentum factor - Jegadeesh & Titman',
                'expected_return': 0.08,
                'volatility': 0.16,
                'sharpe_ratio': 0.50,
                'best_regimes': ['expansion', 'bull_market'],
                'correlation_with_market': 0.3,
                'lookback_period': 126,
                'rebalance_frequency': 'monthly'
            },
            'quality': {
                'description': 'Quality factor (RMW) - Fama & French',
                'expected_return': 0.03,
                'volatility': 0.08,
                'sharpe_ratio': 0.38,
                'best_regimes': ['recession', 'bear_market'],
                'correlation_with_market': -0.1,
                'lookback_period': 252,
                'rebalance_frequency': 'quarterly'
            },
            'low_volatility': {
                'description': 'Low volatility anomaly - Baker & Haugen',
                'expected_return': 0.02,
                'volatility': 0.10,
                'sharpe_ratio': 0.20,
                'best_regimes': ['recession', 'high_volatility'],
                'correlation_with_market': -0.2,
                'lookback_period': 252,
                'rebalance_frequency': 'quarterly'
            },
            'size': {
                'description': 'Size factor (SMB) - Fama & French',
                'expected_return': 0.02,
                'volatility': 0.14,
                'sharpe_ratio': 0.14,
                'best_regimes': ['recovery', 'expansion'],
                'correlation_with_market': 0.2,
                'lookback_period': 252,
                'rebalance_frequency': 'quarterly'
            },
            'profitability': {
                'description': 'Profitability factor - Novy-Marx',
                'expected_return': 0.03,
                'volatility': 0.09,
                'sharpe_ratio': 0.33,
                'best_regimes': ['expansion', 'normal'],
                'correlation_with_market': 0.1,
                'lookback_period': 252,
                'rebalance_frequency': 'quarterly'
            }
        }
    
    def _initialize_strategy_templates(self) -> List[Dict[str, Any]]:
        """Initialize pre-built strategy templates."""
        return [
            {
                'name': 'Conservative Value',
                'objective': 'value',
                'risk_profile': 'conservative',
                'factors': ['value', 'quality', 'low_volatility'],
                'factor_weights': [0.4, 0.4, 0.2],
                'expected_return': 0.08,
                'expected_volatility': 0.12,
                'max_drawdown': 0.15
            },
            {
                'name': 'Momentum Growth',
                'objective': 'growth',
                'risk_profile': 'aggressive',
                'factors': ['momentum', 'quality', 'profitability'],
                'factor_weights': [0.5, 0.3, 0.2],
                'expected_return': 0.12,
                'expected_volatility': 0.18,
                'max_drawdown': 0.25
            },
            {
                'name': 'Balanced Multi-Factor',
                'objective': 'balanced',
                'risk_profile': 'moderate',
                'factors': ['value', 'momentum', 'quality', 'low_volatility'],
                'factor_weights': [0.25, 0.25, 0.25, 0.25],
                'expected_return': 0.10,
                'expected_volatility': 0.15,
                'max_drawdown': 0.20
            },
            {
                'name': 'Quality Defensive',
                'objective': 'income',
                'risk_profile': 'conservative',
                'factors': ['quality', 'low_volatility', 'profitability'],
                'factor_weights': [0.4, 0.4, 0.2],
                'expected_return': 0.07,
                'expected_volatility': 0.10,
                'max_drawdown': 0.12
            }
        ]
    
    def _select_factors_for_objective(self, 
                                    objective: str, 
                                    risk_tolerance: str,
                                    preferences: Optional[List[str]] = None) -> List[FactorConfig]:
        """Select factors based on investment objective."""
        factor_configs = []
        
        # Base factor selection by objective
        if objective == 'value':
            base_factors = [('value', 0.5), ('quality', 0.3), ('profitability', 0.2)]
        elif objective == 'growth':
            base_factors = [('momentum', 0.4), ('quality', 0.3), ('profitability', 0.3)]
        elif objective == 'momentum':
            base_factors = [('momentum', 0.6), ('quality', 0.2), ('size', 0.2)]
        elif objective == 'income':
            base_factors = [('quality', 0.4), ('low_volatility', 0.4), ('profitability', 0.2)]
        else:  # balanced
            base_factors = [('value', 0.25), ('momentum', 0.25), ('quality', 0.25), ('low_volatility', 0.25)]
        
        # Adjust weights based on risk tolerance
        risk_multipliers = {
            'conservative': {'momentum': 0.7, 'low_volatility': 1.3, 'quality': 1.2},
            'moderate': {'momentum': 1.0, 'low_volatility': 1.0, 'quality': 1.0},
            'aggressive': {'momentum': 1.4, 'low_volatility': 0.6, 'quality': 0.9}
        }
        
        multiplier = risk_multipliers.get(risk_tolerance, risk_multipliers['moderate'])
        
        # Create factor configs
        total_weight = 0
        for factor_name, weight in base_factors:
            adjusted_weight = weight * multiplier.get(factor_name, 1.0)
            total_weight += adjusted_weight
            
            template = self.factor_templates[factor_name]
            config = FactorConfig(
                name=factor_name,
                weight=adjusted_weight,
                lookback_period=template['lookback_period'],
                rebalance_frequency=template['rebalance_frequency'],
                enabled=True
            )
            factor_configs.append(config)
        
        # Normalize weights
        for config in factor_configs:
            config.weight /= total_weight
        
        return factor_configs
    
    def _get_universe_for_type(self, universe_type: str) -> List[str]:
        """Get asset universe based on type."""
        universes = {
            'large_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V'],
            'small_cap': ['IWM', 'VTI', 'VTEB', 'VEA', 'VWO', 'BND', 'VXUS', 'VNQ', 'GLD', 'TLT'],
            'all_cap': ['VTI', 'VXUS', 'BND', 'VNQ', 'VWO', 'GLD', 'TLT', 'VTEB', 'VEA', 'IWM'],
            'international': ['VEA', 'VWO', 'VXUS', 'EFA', 'EEM', 'IEFA', 'IEMG', 'VGK', 'VPL', 'VSS']
        }
        return universes.get(universe_type, universes['large_cap'])
    
    def _get_optimization_method(self, risk_tolerance: str) -> str:
        """Get optimization method based on risk tolerance."""
        methods = {
            'conservative': 'risk_parity',
            'moderate': 'mean_variance',
            'aggressive': 'mean_variance'
        }
        return methods.get(risk_tolerance, 'mean_variance')
    
    def _configure_risk_constraints(self, 
                                  risk_tolerance: str,
                                  additional_constraints: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Configure risk constraints based on tolerance."""
        base_constraints = {
            'conservative': {
                'max_individual_weight': 0.05,
                'max_sector_weight': 0.15,
                'max_portfolio_var': 0.02,
                'max_drawdown': 0.10,
                'min_diversification': 20
            },
            'moderate': {
                'max_individual_weight': 0.08,
                'max_sector_weight': 0.25,
                'max_portfolio_var': 0.04,
                'max_drawdown': 0.15,
                'min_diversification': 15
            },
            'aggressive': {
                'max_individual_weight': 0.12,
                'max_sector_weight': 0.35,
                'max_portfolio_var': 0.06,
                'max_drawdown': 0.25,
                'min_diversification': 10
            }
        }
        
        constraints = base_constraints.get(risk_tolerance, base_constraints['moderate'])
        
        if additional_constraints:
            constraints.update(additional_constraints)
        
        return constraints
    
    def _determine_rebalance_frequency(self, 
                                     horizon: str, 
                                     factors: List[FactorConfig]) -> str:
        """Determine optimal rebalancing frequency."""
        # Consider factor characteristics and investment horizon
        momentum_weight = sum(f.weight for f in factors if f.name == 'momentum')
        
        if horizon == 'short':
            return 'monthly' if momentum_weight > 0.3 else 'quarterly'
        elif horizon == 'medium':
            return 'quarterly'
        else:  # long
            return 'semi_annually' if momentum_weight < 0.2 else 'quarterly'
    
    def _generate_strategy_name(self, objective: str, factors: List[FactorConfig]) -> str:
        """Generate a descriptive strategy name."""
        primary_factor = max(factors, key=lambda f: f.weight).name
        return f"{objective.title()} {primary_factor.title()} Multi-Factor Strategy"
    
    def _generate_strategy_description(self, 
                                     objective: str,
                                     factors: List[FactorConfig],
                                     risk_tolerance: str) -> str:
        """Generate strategy description."""
        factor_names = [f.name for f in factors]
        primary_factor = max(factors, key=lambda f: f.weight).name
        
        description = f"A {risk_tolerance} {objective} strategy that combines "
        description += f"{', '.join(factor_names[:-1])} and {factor_names[-1]} factors. "
        description += f"The strategy emphasizes {primary_factor} factors while maintaining "
        description += f"diversification across multiple factor exposures."
        
        return description
    
    def _estimate_performance(self, 
                            factors: List[FactorConfig],
                            risk_tolerance: str) -> Dict[str, float]:
        """Estimate expected performance based on factor research."""
        expected_return = 0
        expected_vol = 0
        
        for factor in factors:
            template = self.factor_templates[factor.name]
            expected_return += factor.weight * template['expected_return']
            expected_vol += (factor.weight ** 2) * (template['volatility'] ** 2)
        
        # Add correlation effects (simplified)
        expected_vol = np.sqrt(expected_vol) * 0.8  # Assume some diversification benefit
        
        # Adjust for risk tolerance
        risk_adjustments = {
            'conservative': {'return': 0.8, 'vol': 0.7},
            'moderate': {'return': 1.0, 'vol': 1.0},
            'aggressive': {'return': 1.2, 'vol': 1.3}
        }
        
        adj = risk_adjustments.get(risk_tolerance, risk_adjustments['moderate'])
        expected_return *= adj['return']
        expected_vol *= adj['vol']
        
        return {
            'expected_annual_return': expected_return,
            'expected_volatility': expected_vol,
            'expected_sharpe_ratio': expected_return / expected_vol if expected_vol > 0 else 0,
            'expected_max_drawdown': expected_vol * 1.5  # Rule of thumb
        }
    
    def _score_template_fit(self, template: Dict[str, Any], user_profile: Dict[str, Any]) -> float:
        """Score how well a template fits the user profile."""
        score = 0.0
        
        # Risk profile match
        if template['risk_profile'] == user_profile.get('risk_tolerance', 'moderate'):
            score += 0.4
        
        # Objective match
        objectives = user_profile.get('objectives', ['balanced'])
        if template['objective'] in objectives:
            score += 0.3
        
        # Return expectations
        expected_return = user_profile.get('expected_return', 0.08)
        return_diff = abs(template['expected_return'] - expected_return)
        score += max(0, 0.2 - return_diff * 2)  # Penalty for return mismatch
        
        # Risk tolerance
        risk_tolerance = user_profile.get('max_volatility', 0.15)
        vol_diff = abs(template['expected_volatility'] - risk_tolerance)
        score += max(0, 0.1 - vol_diff)
        
        return min(1.0, score)
    
    def _customize_template(self, template: Dict[str, Any], user_profile: Dict[str, Any]) -> StrategyBlueprint:
        """Customize a template for specific user profile."""
        # Convert template to StrategyBlueprint format
        factors = []
        for i, factor_name in enumerate(template['factors']):
            weight = template['factor_weights'][i]
            template_info = self.factor_templates[factor_name]
            
            config = FactorConfig(
                name=factor_name,
                weight=weight,
                lookback_period=template_info['lookback_period'],
                rebalance_frequency=template_info['rebalance_frequency']
            )
            factors.append(config)
        
        # Customize based on user preferences
        universe_type = user_profile.get('universe_preference', 'large_cap')
        universe = self._get_universe_for_type(universe_type)
        
        risk_constraints = self._configure_risk_constraints(
            template['risk_profile'],
            user_profile.get('additional_constraints')
        )
        
        return StrategyBlueprint(
            name=template['name'],
            description=f"Customized {template['name']} strategy",
            factors=factors,
            universe=universe,
            optimization_method=self._get_optimization_method(template['risk_profile']),
            risk_constraints=risk_constraints,
            rebalance_frequency='quarterly',
            expected_performance={
                'expected_annual_return': template['expected_return'],
                'expected_volatility': template['expected_volatility'],
                'expected_max_drawdown': template['max_drawdown']
            },
            risk_profile=template['risk_profile']
        )
    
    # Additional helper methods for optimization and analysis
    def _analyze_factor_performance(self, factors: List[FactorConfig], returns: pd.Series) -> Dict[str, float]:
        """Analyze individual factor performance."""
        # Simplified factor performance analysis
        performance = {}
        for factor in factors:
            # In practice, this would analyze actual factor returns
            template = self.factor_templates[factor.name]
            performance[factor.name] = template['sharpe_ratio']
        return performance
    
    def _optimize_factor_weights(self, 
                                factors: List[FactorConfig],
                                performance: Dict[str, float],
                                market_regime: str) -> List[FactorConfig]:
        """Optimize factor weights based on performance and regime."""
        # Simplified optimization - in practice would use more sophisticated methods
        optimized_factors = []
        
        for factor in factors:
            new_weight = factor.weight
            perf = performance.get(factor.name, 0)
            
            # Adjust based on performance
            if perf > 0.5:  # Good performance
                new_weight *= 1.1
            elif perf < 0.2:  # Poor performance
                new_weight *= 0.9
            
            # Adjust based on market regime
            template = self.factor_templates[factor.name]
            if market_regime in template.get('best_regimes', []):
                new_weight *= 1.05
            
            optimized_factor = FactorConfig(
                name=factor.name,
                weight=new_weight,
                lookback_period=factor.lookback_period,
                rebalance_frequency=factor.rebalance_frequency,
                enabled=factor.enabled
            )
            optimized_factors.append(optimized_factor)
        
        # Normalize weights
        total_weight = sum(f.weight for f in optimized_factors)
        for factor in optimized_factors:
            factor.weight /= total_weight
        
        return optimized_factors
    
    def _update_risk_constraints(self, 
                               current_constraints: Dict[str, float],
                               returns: pd.Series) -> Dict[str, float]:
        """Update risk constraints based on recent performance."""
        updated = current_constraints.copy()
        
        # Analyze recent volatility
        recent_vol = returns.tail(60).std() * np.sqrt(252)
        
        # Convert to scalar if it's a Series
        if hasattr(recent_vol, 'iloc'):
            recent_vol = recent_vol.iloc[0] if len(recent_vol) > 0 else recent_vol
        
        # Adjust VaR constraint if volatility has changed significantly
        if recent_vol > 0.2:  # High volatility
            updated['max_portfolio_var'] *= 0.8  # Tighten constraint
        elif recent_vol < 0.1:  # Low volatility
            updated['max_portfolio_var'] *= 1.1  # Relax slightly
        
        return updated
    
    def _simulate_factor_returns(self, template: Dict[str, Any], market_data: pd.DataFrame) -> pd.Series:
        """Simulate factor returns based on template characteristics."""
        # Simplified simulation - in practice would use actual factor data
        n_periods = len(market_data)
        returns = np.random.normal(
            template['expected_return'] / 252,
            template['volatility'] / np.sqrt(252),
            n_periods
        )
        return pd.Series(returns, index=market_data.index)
    
    def _analyze_factor_regimes(self, factor_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze factor performance across different regimes."""
        # Simplified regime analysis
        regimes = {}
        
        for factor in factor_df.columns:
            factor_returns = factor_df[factor]
            
            regimes[factor] = {
                'bull_market_performance': factor_returns[factor_returns > 0].mean(),
                'bear_market_performance': factor_returns[factor_returns < 0].mean(),
                'volatility_regime_performance': {
                    'high_vol': factor_returns[factor_returns.rolling(20).std() > factor_returns.std()].mean(),
                    'low_vol': factor_returns[factor_returns.rolling(20).std() <= factor_returns.std()].mean()
                }
            }
        
        return regimes
    
    def _generate_factor_recommendations(self, 
                                       factor_df: pd.DataFrame,
                                       correlations: pd.DataFrame,
                                       regime_analysis: Dict[str, Any]) -> List[str]:
        """Generate factor-based recommendations."""
        recommendations = []
        
        # Check for high correlations
        high_corr_pairs = []
        for i in range(len(correlations.columns)):
            for j in range(i+1, len(correlations.columns)):
                corr = correlations.iloc[i, j]
                if abs(corr) > 0.7:
                    high_corr_pairs.append((correlations.columns[i], correlations.columns[j], corr))
        
        if high_corr_pairs:
            recommendations.append(f"High correlation detected between factors: {high_corr_pairs[0][0]} and {high_corr_pairs[0][1]} ({high_corr_pairs[0][2]:.2f})")
        
        # Performance-based recommendations
        factor_performance = factor_df.mean().sort_values(ascending=False)
        best_factor = factor_performance.index[0]
        worst_factor = factor_performance.index[-1]
        
        recommendations.append(f"Best performing factor: {best_factor}")
        recommendations.append(f"Consider reducing exposure to: {worst_factor}")
        
        return recommendations
    
    def _calculate_diversification_score(self, correlation_matrix: pd.DataFrame) -> float:
        """Calculate diversification score based on correlations."""
        # Average absolute correlation as inverse diversification measure
        avg_abs_corr = correlation_matrix.abs().mean().mean()
        return max(0, 1 - avg_abs_corr)  # Higher score = better diversification
