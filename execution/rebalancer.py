"""Portfolio rebalancing engine."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from decimal import Decimal

from core.models import Portfolio, Asset, Position, Transaction
from core.enums import AssetClass
from backtesting.transaction_cost_model import TransactionCostModel


class Rebalancer:
    """Portfolio rebalancing engine."""
    
    def __init__(self, 
                 min_trade_threshold: float = 0.01,
                 max_turnover: float = 1.0):
        """Initialize rebalancer.
        
        Args:
            min_trade_threshold: Minimum trade size as fraction of portfolio
            max_turnover: Maximum portfolio turnover per rebalance
        """
        self.min_trade_threshold = min_trade_threshold
        self.max_turnover = max_turnover
        self.transaction_cost_model = TransactionCostModel()
    
    def rebalance_portfolio(self,
                          portfolio: Portfolio,
                          target_weights: pd.Series,
                          current_prices: pd.Series,
                          transaction_cost_model: Optional[TransactionCostModel] = None) -> Dict[str, Any]:
        """Rebalance portfolio to target weights.
        
        Args:
            portfolio: Current portfolio
            target_weights: Target asset weights
            current_prices: Current asset prices
            transaction_cost_model: Transaction cost model
            
        Returns:
            Rebalancing result with trades and metrics
        """
        cost_model = transaction_cost_model or self.transaction_cost_model
        
        # Get current portfolio state
        current_weights = pd.Series(portfolio.weights)
        total_value = float(portfolio.total_value)
        
        # Align target weights with available assets
        all_assets = set(current_weights.index) | set(target_weights.index)
        current_weights = current_weights.reindex(all_assets, fill_value=0)
        target_weights = target_weights.reindex(all_assets, fill_value=0)
        
        # Calculate weight differences
        weight_diff = target_weights - current_weights
        
        # Calculate target dollar amounts
        target_amounts = target_weights * total_value
        current_amounts = current_weights * total_value
        trade_amounts = target_amounts - current_amounts
        
        # Generate trades
        trades = []
        total_turnover = 0
        estimated_costs = 0
        
        for symbol in all_assets:
            trade_amount = trade_amounts[symbol]
            
            # Skip small trades
            if abs(trade_amount) < self.min_trade_threshold * total_value:
                continue
            
            # Get current price
            if symbol in current_prices.index and current_prices[symbol] > 0:
                price = current_prices[symbol]
                quantity = trade_amount / price
                
                # Create asset if not exists
                if symbol in portfolio.positions:
                    asset = portfolio.positions[symbol].asset
                else:
                    asset = Asset(symbol=symbol, name=symbol, asset_class=AssetClass.EQUITY)
                
                # Estimate transaction cost
                trade_cost = cost_model.calculate_cost(
                    symbol=symbol,
                    trade_value=abs(trade_amount),
                    trade_type='buy' if trade_amount > 0 else 'sell'
                )
                
                trade = {
                    'symbol': symbol,
                    'asset': asset,
                    'current_weight': current_weights[symbol],
                    'target_weight': target_weights[symbol],
                    'weight_change': weight_diff[symbol],
                    'trade_amount': trade_amount,
                    'quantity': quantity,
                    'price': price,
                    'estimated_cost': trade_cost
                }
                
                trades.append(trade)
                total_turnover += abs(trade_amount)
                estimated_costs += trade_cost
        
        # Check turnover constraint
        turnover_rate = total_turnover / total_value
        if turnover_rate > self.max_turnover:
            # Scale down trades to meet turnover constraint
            scale_factor = self.max_turnover / turnover_rate
            trades = self._scale_trades(trades, scale_factor)
            total_turnover *= scale_factor
            estimated_costs *= scale_factor
        
        # Optimize trade execution order
        optimized_trades = self._optimize_trade_order(trades)
        
        return {
            'trades': optimized_trades,
            'total_turnover': total_turnover,
            'turnover_rate': total_turnover / total_value,
            'estimated_costs': estimated_costs,
            'cost_as_pct': estimated_costs / total_value,
            'trades_count': len(optimized_trades),
            'target_weights': target_weights.to_dict(),
            'current_weights': current_weights.to_dict()
        }
    
    def _scale_trades(self, trades: List[Dict[str, Any]], scale_factor: float) -> List[Dict[str, Any]]:
        """Scale trades to meet constraints."""
        scaled_trades = []
        
        for trade in trades:
            scaled_trade = trade.copy()
            scaled_trade['trade_amount'] *= scale_factor
            scaled_trade['quantity'] *= scale_factor
            scaled_trade['estimated_cost'] *= scale_factor
            
            # Recalculate weight change
            scaled_trade['weight_change'] *= scale_factor
            
            scaled_trades.append(scaled_trade)
        
        return scaled_trades
    
    def _optimize_trade_order(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize trade execution order."""
        # Sort trades by priority:
        # 1. Sells first (to free up cash)
        # 2. Larger trades first (better execution)
        # 3. Lower cost trades first
        
        def trade_priority(trade):
            is_sell = trade['trade_amount'] < 0
            trade_size = abs(trade['trade_amount'])
            cost_ratio = trade['estimated_cost'] / abs(trade['trade_amount']) if trade['trade_amount'] != 0 else 0
            
            # Priority score (lower is better)
            priority = (
                0 if is_sell else 1,  # Sells first
                -trade_size,  # Larger trades first
                cost_ratio  # Lower cost ratio first
            )
            
            return priority
        
        return sorted(trades, key=trade_priority)
    
    def calculate_rebalancing_need(self,
                                 portfolio: Portfolio,
                                 target_weights: pd.Series,
                                 threshold: float = 0.05) -> Dict[str, Any]:
        """Calculate if rebalancing is needed.
        
        Args:
            portfolio: Current portfolio
            target_weights: Target weights
            threshold: Rebalancing threshold
            
        Returns:
            Rebalancing need assessment
        """
        current_weights = pd.Series(portfolio.weights)
        
        # Align weights
        all_assets = set(current_weights.index) | set(target_weights.index)
        current_weights = current_weights.reindex(all_assets, fill_value=0)
        target_weights = target_weights.reindex(all_assets, fill_value=0)
        
        # Calculate deviations
        weight_deviations = abs(target_weights - current_weights)
        max_deviation = weight_deviations.max()
        total_deviation = weight_deviations.sum()
        
        # Assets that need rebalancing
        assets_to_rebalance = weight_deviations[weight_deviations > threshold].index.tolist()
        
        needs_rebalancing = max_deviation > threshold
        
        return {
            'needs_rebalancing': needs_rebalancing,
            'max_deviation': max_deviation,
            'total_deviation': total_deviation,
            'threshold': threshold,
            'assets_to_rebalance': assets_to_rebalance,
            'deviation_by_asset': weight_deviations.to_dict()
        }
    
    def simulate_rebalancing(self,
                           portfolio: Portfolio,
                           target_weights: pd.Series,
                           current_prices: pd.Series,
                           scenarios: List[Dict[str, float]]) -> Dict[str, Any]:
        """Simulate rebalancing under different scenarios.
        
        Args:
            portfolio: Current portfolio
            target_weights: Target weights
            current_prices: Current prices
            scenarios: List of price change scenarios
            
        Returns:
            Simulation results
        """
        base_rebalance = self.rebalance_portfolio(portfolio, target_weights, current_prices)
        
        scenario_results = []
        
        for i, scenario in enumerate(scenarios):
            # Apply scenario price changes
            scenario_prices = current_prices.copy()
            for symbol, price_change in scenario.items():
                if symbol in scenario_prices.index:
                    scenario_prices[symbol] *= (1 + price_change)
            
            # Simulate rebalancing with scenario prices
            scenario_result = self.rebalance_portfolio(portfolio, target_weights, scenario_prices)
            
            scenario_results.append({
                'scenario_id': i,
                'price_changes': scenario,
                'turnover_rate': scenario_result['turnover_rate'],
                'estimated_costs': scenario_result['estimated_costs'],
                'trades_count': scenario_result['trades_count']
            })
        
        return {
            'base_case': base_rebalance,
            'scenarios': scenario_results,
            'avg_turnover': np.mean([s['turnover_rate'] for s in scenario_results]),
            'avg_costs': np.mean([s['estimated_costs'] for s in scenario_results]),
            'max_turnover': max([s['turnover_rate'] for s in scenario_results]),
            'min_turnover': min([s['turnover_rate'] for s in scenario_results])
        }
    
    def get_rebalancing_schedule(self,
                               strategy_frequency: str,
                               start_date: datetime,
                               end_date: datetime) -> List[datetime]:
        """Generate rebalancing schedule.
        
        Args:
            strategy_frequency: Rebalancing frequency
            start_date: Start date
            end_date: End date
            
        Returns:
            List of rebalancing dates
        """
        schedule = []
        current_date = start_date
        
        if strategy_frequency == 'daily':
            while current_date <= end_date:
                schedule.append(current_date)
                current_date += pd.Timedelta(days=1)
        
        elif strategy_frequency == 'weekly':
            # Rebalance on Mondays
            while current_date <= end_date:
                if current_date.weekday() == 0:  # Monday
                    schedule.append(current_date)
                current_date += pd.Timedelta(days=1)
        
        elif strategy_frequency == 'monthly':
            while current_date <= end_date:
                if current_date.day == 1:  # First day of month
                    schedule.append(current_date)
                current_date += pd.Timedelta(days=1)
        
        elif strategy_frequency == 'quarterly':
            while current_date <= end_date:
                if current_date.day == 1 and current_date.month in [1, 4, 7, 10]:
                    schedule.append(current_date)
                current_date += pd.Timedelta(days=1)
        
        return schedule
    
    def analyze_rebalancing_impact(self,
                                 historical_returns: pd.DataFrame,
                                 target_weights: pd.Series,
                                 rebalancing_frequency: str) -> Dict[str, Any]:
        """Analyze the impact of rebalancing frequency.
        
        Args:
            historical_returns: Historical returns data
            target_weights: Target portfolio weights
            rebalancing_frequency: Rebalancing frequency
            
        Returns:
            Rebalancing impact analysis
        """
        # Simulate portfolio with and without rebalancing
        dates = historical_returns.index
        
        # Portfolio without rebalancing (buy and hold)
        bnh_weights = target_weights.copy()
        bnh_values = []
        
        # Portfolio with rebalancing
        rebal_weights = target_weights.copy()
        rebal_values = []
        
        # Get rebalancing dates
        rebal_dates = set(self.get_rebalancing_schedule(
            rebalancing_frequency, dates[0], dates[-1]
        ))
        
        total_rebalancing_costs = 0
        rebalancing_count = 0
        
        for i, date in enumerate(dates):
            if i == 0:
                bnh_values.append(1.0)
                rebal_values.append(1.0)
                continue
            
            # Calculate returns
            returns = historical_returns.loc[date]
            
            # Buy and hold portfolio
            bnh_return = (bnh_weights * returns).sum()
            bnh_values.append(bnh_values[-1] * (1 + bnh_return))
            
            # Update buy and hold weights (drift with returns)
            bnh_weights = bnh_weights * (1 + returns)
            bnh_weights = bnh_weights / bnh_weights.sum()
            
            # Rebalanced portfolio
            rebal_return = (rebal_weights * returns).sum()
            rebal_values.append(rebal_values[-1] * (1 + rebal_return))
            
            # Check if rebalancing date
            if date in rebal_dates:
                # Calculate rebalancing cost
                current_portfolio_value = rebal_values[-1]
                weight_diff = abs(target_weights - rebal_weights).sum()
                rebalancing_cost = weight_diff * 0.001  # Assume 0.1% cost
                
                total_rebalancing_costs += rebalancing_cost * current_portfolio_value
                rebalancing_count += 1
                
                # Reset to target weights
                rebal_weights = target_weights.copy()
            else:
                # Update rebalanced weights (drift with returns)
                rebal_weights = rebal_weights * (1 + returns)
                rebal_weights = rebal_weights / rebal_weights.sum()
        
        # Calculate metrics
        bnh_total_return = bnh_values[-1] - 1
        rebal_total_return = rebal_values[-1] - 1
        
        bnh_vol = pd.Series(bnh_values).pct_change().std() * np.sqrt(252)
        rebal_vol = pd.Series(rebal_values).pct_change().std() * np.sqrt(252)
        
        return {
            'buy_and_hold_return': bnh_total_return,
            'rebalanced_return': rebal_total_return,
            'rebalancing_benefit': rebal_total_return - bnh_total_return,
            'buy_and_hold_volatility': bnh_vol,
            'rebalanced_volatility': rebal_vol,
            'total_rebalancing_costs': total_rebalancing_costs,
            'rebalancing_count': rebalancing_count,
            'avg_cost_per_rebalance': total_rebalancing_costs / rebalancing_count if rebalancing_count > 0 else 0,
            'net_benefit': (rebal_total_return - bnh_total_return) - total_rebalancing_costs
        }
