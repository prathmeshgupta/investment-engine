"""Transaction cost modeling for backtesting."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class CostModel(Enum):
    """Transaction cost model types."""
    FIXED = "fixed"
    PROPORTIONAL = "proportional"
    TIERED = "tiered"
    MARKET_IMPACT = "market_impact"


class TransactionCostModel:
    """Model transaction costs for backtesting."""
    
    def __init__(self, 
                 model_type: CostModel = CostModel.PROPORTIONAL,
                 base_cost: float = 0.001):
        """Initialize transaction cost model.
        
        Args:
            model_type: Type of cost model
            base_cost: Base cost parameter
        """
        self.model_type = model_type
        self.base_cost = base_cost
        
        # Default cost parameters by asset class
        self.cost_parameters = {
            'equity': {'spread': 0.0005, 'commission': 0.0005, 'impact': 0.0001},
            'bond': {'spread': 0.001, 'commission': 0.0002, 'impact': 0.0002},
            'commodity': {'spread': 0.002, 'commission': 0.001, 'impact': 0.0005},
            'currency': {'spread': 0.0001, 'commission': 0.0001, 'impact': 0.00005}
        }
    
    def calculate_cost(self,
                      symbol: str,
                      trade_value: float,
                      trade_type: str,
                      asset_class: str = 'equity',
                      volume_participation: float = 0.1) -> float:
        """Calculate transaction cost for a trade.
        
        Args:
            symbol: Asset symbol
            trade_value: Trade value in dollars
            trade_type: 'buy' or 'sell'
            asset_class: Asset class for cost parameters
            volume_participation: Participation rate in daily volume
            
        Returns:
            Transaction cost in dollars
        """
        if self.model_type == CostModel.FIXED:
            return self._fixed_cost(trade_value)
        elif self.model_type == CostModel.PROPORTIONAL:
            return self._proportional_cost(trade_value, asset_class)
        elif self.model_type == CostModel.TIERED:
            return self._tiered_cost(trade_value, asset_class)
        elif self.model_type == CostModel.MARKET_IMPACT:
            return self._market_impact_cost(trade_value, volume_participation, asset_class)
        else:
            return 0.0
    
    def _fixed_cost(self, trade_value: float) -> float:
        """Fixed cost per trade."""
        return self.base_cost
    
    def _proportional_cost(self, trade_value: float, asset_class: str) -> float:
        """Proportional cost based on trade value."""
        params = self.cost_parameters.get(asset_class, self.cost_parameters['equity'])
        
        # Bid-ask spread cost
        spread_cost = trade_value * params['spread'] / 2
        
        # Commission cost
        commission_cost = trade_value * params['commission']
        
        return spread_cost + commission_cost
    
    def _tiered_cost(self, trade_value: float, asset_class: str) -> float:
        """Tiered cost structure based on trade size."""
        params = self.cost_parameters.get(asset_class, self.cost_parameters['equity'])
        
        # Define tiers (trade value thresholds)
        tiers = [
            (10000, params['commission'] * 2),      # Small trades: higher rate
            (100000, params['commission'] * 1.5),   # Medium trades: medium rate
            (1000000, params['commission']),        # Large trades: base rate
            (float('inf'), params['commission'] * 0.5)  # Very large trades: lower rate
        ]
        
        # Find applicable tier
        cost_rate = params['commission']
        for threshold, rate in tiers:
            if trade_value <= threshold:
                cost_rate = rate
                break
        
        # Add spread cost
        spread_cost = trade_value * params['spread'] / 2
        commission_cost = trade_value * cost_rate
        
        return spread_cost + commission_cost
    
    def _market_impact_cost(self, 
                          trade_value: float, 
                          volume_participation: float,
                          asset_class: str) -> float:
        """Market impact cost model."""
        params = self.cost_parameters.get(asset_class, self.cost_parameters['equity'])
        
        # Base costs
        spread_cost = trade_value * params['spread'] / 2
        commission_cost = trade_value * params['commission']
        
        # Market impact (square root law)
        # Impact increases with square root of participation rate
        impact_cost = trade_value * params['impact'] * np.sqrt(volume_participation)
        
        return spread_cost + commission_cost + impact_cost
    
    def calculate_portfolio_costs(self,
                                trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate total costs for a portfolio of trades.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Cost breakdown dictionary
        """
        total_cost = 0.0
        cost_breakdown = {
            'total_cost': 0.0,
            'spread_cost': 0.0,
            'commission_cost': 0.0,
            'impact_cost': 0.0,
            'trade_count': len(trades)
        }
        
        for trade in trades:
            trade_cost = self.calculate_cost(
                symbol=trade.get('symbol', ''),
                trade_value=trade.get('value', 0),
                trade_type=trade.get('type', 'buy'),
                asset_class=trade.get('asset_class', 'equity'),
                volume_participation=trade.get('volume_participation', 0.1)
            )
            
            total_cost += trade_cost
        
        cost_breakdown['total_cost'] = total_cost
        cost_breakdown['avg_cost_per_trade'] = total_cost / len(trades) if trades else 0
        cost_breakdown['cost_as_pct_of_value'] = total_cost / sum(t.get('value', 0) for t in trades) if trades else 0
        
        return cost_breakdown
    
    def estimate_annual_costs(self,
                            portfolio_value: float,
                            turnover_rate: float,
                            rebalance_frequency: int = 12) -> Dict[str, float]:
        """Estimate annual transaction costs.
        
        Args:
            portfolio_value: Portfolio value
            turnover_rate: Annual portfolio turnover rate
            rebalance_frequency: Number of rebalances per year
            
        Returns:
            Annual cost estimates
        """
        # Annual trading volume
        annual_volume = portfolio_value * turnover_rate
        
        # Average trade size
        avg_trade_size = annual_volume / rebalance_frequency if rebalance_frequency > 0 else annual_volume
        
        # Estimate costs
        sample_cost = self.calculate_cost('SAMPLE', avg_trade_size, 'buy')
        annual_cost = sample_cost * rebalance_frequency
        
        return {
            'annual_cost': annual_cost,
            'cost_as_pct_of_portfolio': annual_cost / portfolio_value,
            'cost_per_rebalance': sample_cost,
            'estimated_trades_per_year': rebalance_frequency,
            'annual_volume': annual_volume
        }
    
    def optimize_trade_execution(self,
                               target_trades: Dict[str, float],
                               current_positions: Dict[str, float],
                               min_trade_size: float = 1000) -> Dict[str, float]:
        """Optimize trade execution to minimize costs.
        
        Args:
            target_trades: Target trade amounts by symbol
            current_positions: Current position sizes
            min_trade_size: Minimum trade size threshold
            
        Returns:
            Optimized trade amounts
        """
        optimized_trades = {}
        
        for symbol, target_amount in target_trades.items():
            current_amount = current_positions.get(symbol, 0)
            trade_amount = target_amount - current_amount
            
            # Skip small trades to reduce cost impact
            if abs(trade_amount) < min_trade_size:
                continue
            
            # Round to minimize odd lot costs
            if abs(trade_amount) > 10000:
                # Round to nearest $100 for large trades
                trade_amount = round(trade_amount / 100) * 100
            elif abs(trade_amount) > 1000:
                # Round to nearest $10 for medium trades
                trade_amount = round(trade_amount / 10) * 10
            
            if abs(trade_amount) >= min_trade_size:
                optimized_trades[symbol] = trade_amount
        
        return optimized_trades
    
    def calculate_implementation_shortfall(self,
                                         intended_trades: Dict[str, float],
                                         actual_trades: Dict[str, float],
                                         price_moves: Dict[str, float]) -> Dict[str, float]:
        """Calculate implementation shortfall.
        
        Args:
            intended_trades: Originally intended trades
            actual_trades: Actually executed trades
            price_moves: Price moves during execution period
            
        Returns:
            Implementation shortfall metrics
        """
        total_shortfall = 0.0
        total_intended_value = 0.0
        
        for symbol in intended_trades:
            intended_amount = intended_trades.get(symbol, 0)
            actual_amount = actual_trades.get(symbol, 0)
            price_move = price_moves.get(symbol, 0)
            
            # Shortfall from incomplete execution
            execution_shortfall = (intended_amount - actual_amount) * price_move
            total_shortfall += execution_shortfall
            total_intended_value += abs(intended_amount)
        
        return {
            'total_shortfall': total_shortfall,
            'shortfall_bps': (total_shortfall / total_intended_value * 10000) if total_intended_value > 0 else 0,
            'execution_rate': sum(abs(actual_trades.get(s, 0)) for s in intended_trades) / total_intended_value if total_intended_value > 0 else 0
        }
