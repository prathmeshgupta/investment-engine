"""Trade execution engine for order fulfillment."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from decimal import Decimal
import time
import random

from core.models import Order, Asset, Transaction
from core.enums import OrderType, OrderStatus


class TradeExecutor:
    """Executes trades and manages order fulfillment."""
    
    def __init__(self, 
                 execution_delay: float = 0.1,
                 slippage_model: str = 'linear',
                 max_slippage: float = 0.001):
        """Initialize trade executor.
        
        Args:
            execution_delay: Simulated execution delay in seconds
            slippage_model: Slippage model ('linear', 'square_root', 'fixed')
            max_slippage: Maximum slippage as fraction of price
        """
        self.execution_delay = execution_delay
        self.slippage_model = slippage_model
        self.max_slippage = max_slippage
        
        # Market simulation parameters
        self.market_hours = {'start': 9.5, 'end': 16}  # 9:30 AM to 4:00 PM
        self.volatility_multiplier = 1.0
        
    def execute_order(self, order_id: str, order: Order = None) -> Dict[str, Any]:
        """Execute an order.
        
        Args:
            order_id: Order ID
            order: Order object (if not using order manager)
            
        Returns:
            Execution result
        """
        if order is None:
            return {'status': 'failed', 'reason': 'No order provided'}
        
        try:
            # Simulate execution delay
            time.sleep(self.execution_delay)
            
            # Check market hours (simplified)
            if not self._is_market_open():
                return {
                    'status': 'failed',
                    'reason': 'Market is closed',
                    'order_id': order_id
                }
            
            # Calculate execution price with slippage
            execution_price = self._calculate_execution_price(order)
            
            # Determine fill quantity (could be partial)
            fill_quantity = self._determine_fill_quantity(order)
            
            # Calculate transaction cost
            transaction_cost = self._calculate_transaction_cost(order, fill_quantity, execution_price)
            
            # Determine execution status
            if abs(fill_quantity) >= abs(order.quantity):
                status = 'filled'
            elif abs(fill_quantity) > 0:
                status = 'partial_fill'
            else:
                status = 'failed'
                return {
                    'status': status,
                    'reason': 'Unable to fill order',
                    'order_id': order_id
                }
            
            return {
                'status': status,
                'order_id': order_id,
                'fill_quantity': float(fill_quantity),
                'execution_price': float(execution_price),
                'cost': float(transaction_cost),
                'timestamp': datetime.now(),
                'slippage': float(execution_price - (order.price or execution_price)) / float(order.price or execution_price) if order.price else 0
            }
        
        except Exception as e:
            return {
                'status': 'failed',
                'reason': str(e),
                'order_id': order_id
            }
    
    def _is_market_open(self) -> bool:
        """Check if market is open (simplified simulation)."""
        current_time = datetime.now()
        current_hour = current_time.hour + current_time.minute / 60
        
        # Simple check for weekdays during market hours
        if current_time.weekday() >= 5:  # Weekend
            return False
        
        return self.market_hours['start'] <= current_hour <= self.market_hours['end']
    
    def _calculate_execution_price(self, order: Order) -> Decimal:
        """Calculate execution price including slippage."""
        base_price = order.price or Decimal('100')  # Default price if not specified
        
        if order.order_type == OrderType.MARKET:
            # Market orders have slippage
            slippage = self._calculate_slippage(order)
            
            # Apply slippage based on order direction
            if order.quantity > 0:  # Buy order
                execution_price = base_price * (1 + slippage)
            else:  # Sell order
                execution_price = base_price * (1 - slippage)
            
            return execution_price
        
        elif order.order_type == OrderType.LIMIT:
            # Limit orders execute at limit price or better
            # Simulate some probability of execution
            if random.random() < 0.8:  # 80% chance of execution
                return order.price
            else:
                # Order not filled
                return Decimal('0')
        
        else:
            return base_price
    
    def _calculate_slippage(self, order: Order) -> Decimal:
        """Calculate slippage based on order size and model."""
        order_size = abs(float(order.quantity))
        
        if self.slippage_model == 'fixed':
            slippage = self.max_slippage
        
        elif self.slippage_model == 'linear':
            # Linear slippage based on order size
            size_factor = min(order_size / 10000, 1.0)  # Normalize by $10k
            slippage = self.max_slippage * size_factor
        
        elif self.slippage_model == 'square_root':
            # Square root slippage (common in market impact models)
            size_factor = min(order_size / 10000, 1.0)
            slippage = self.max_slippage * np.sqrt(size_factor)
        
        else:
            slippage = 0
        
        # Add random component
        random_factor = random.uniform(0.5, 1.5)
        slippage *= random_factor
        
        return Decimal(str(min(slippage, self.max_slippage)))
    
    def _determine_fill_quantity(self, order: Order) -> Decimal:
        """Determine how much of the order gets filled."""
        # Simulate partial fills based on order type and market conditions
        
        if order.order_type == OrderType.MARKET:
            # Market orders usually get filled completely
            if random.random() < 0.95:  # 95% chance of complete fill
                return order.quantity
            else:
                # Partial fill
                fill_ratio = random.uniform(0.7, 0.95)
                return order.quantity * Decimal(str(fill_ratio))
        
        elif order.order_type == OrderType.LIMIT:
            # Limit orders have lower fill probability
            if random.random() < 0.6:  # 60% chance of any fill
                if random.random() < 0.8:  # 80% of fills are complete
                    return order.quantity
                else:
                    # Partial fill
                    fill_ratio = random.uniform(0.3, 0.8)
                    return order.quantity * Decimal(str(fill_ratio))
            else:
                return Decimal('0')  # No fill
        
        else:
            return order.quantity
    
    def _calculate_transaction_cost(self, 
                                  order: Order, 
                                  fill_quantity: Decimal, 
                                  execution_price: Decimal) -> Decimal:
        """Calculate transaction costs."""
        trade_value = abs(fill_quantity * execution_price)
        
        # Simple cost model: 0.1% of trade value
        commission = trade_value * Decimal('0.001')
        
        # Add fixed fee for small trades
        if trade_value < 1000:
            commission += Decimal('1.0')
        
        return commission
    
    def batch_execute_orders(self, orders: List[Tuple[str, Order]]) -> List[Dict[str, Any]]:
        """Execute multiple orders in batch.
        
        Args:
            orders: List of (order_id, order) tuples
            
        Returns:
            List of execution results
        """
        results = []
        
        for order_id, order in orders:
            result = self.execute_order(order_id, order)
            results.append(result)
            
            # Small delay between executions
            time.sleep(0.01)
        
        return results
    
    def simulate_market_impact(self, 
                             orders: List[Order],
                             market_data: pd.DataFrame) -> Dict[str, Any]:
        """Simulate market impact of large orders.
        
        Args:
            orders: List of orders to simulate
            market_data: Historical market data
            
        Returns:
            Market impact simulation results
        """
        total_impact = 0
        order_impacts = []
        
        for order in orders:
            symbol = order.asset.symbol
            order_value = abs(float(order.quantity * (order.price or Decimal('100'))))
            
            # Get average daily volume from market data
            if symbol in market_data.columns:
                avg_volume = market_data[symbol].rolling(20).mean().iloc[-1] * 100  # Assume price * volume
                participation_rate = order_value / avg_volume if avg_volume > 0 else 0.1
            else:
                participation_rate = 0.05  # Default 5% participation
            
            # Market impact using square root law
            impact = 0.1 * np.sqrt(participation_rate)  # 10 basis points * sqrt(participation)
            impact = min(impact, 0.02)  # Cap at 2%
            
            order_impact = {
                'symbol': symbol,
                'order_value': order_value,
                'participation_rate': participation_rate,
                'estimated_impact': impact,
                'impact_cost': order_value * impact
            }
            
            order_impacts.append(order_impact)
            total_impact += order_impact['impact_cost']
        
        return {
            'total_impact_cost': total_impact,
            'avg_impact_bps': np.mean([o['estimated_impact'] * 10000 for o in order_impacts]),
            'max_impact_bps': max([o['estimated_impact'] * 10000 for o in order_impacts]) if order_impacts else 0,
            'order_impacts': order_impacts
        }
    
    def get_execution_quality_metrics(self, 
                                    executions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate execution quality metrics.
        
        Args:
            executions: List of execution results
            
        Returns:
            Execution quality metrics
        """
        if not executions:
            return {}
        
        successful_executions = [e for e in executions if e['status'] in ['filled', 'partial_fill']]
        
        if not successful_executions:
            return {
                'fill_rate': 0,
                'avg_slippage_bps': 0,
                'avg_execution_time': 0
            }
        
        # Fill rate
        fill_rate = len(successful_executions) / len(executions)
        
        # Average slippage
        slippages = [abs(e.get('slippage', 0)) * 10000 for e in successful_executions]  # Convert to basis points
        avg_slippage_bps = np.mean(slippages) if slippages else 0
        
        # Execution time (would need timestamps in real implementation)
        avg_execution_time = self.execution_delay  # Simplified
        
        # Cost analysis
        total_costs = sum(e.get('cost', 0) for e in successful_executions)
        total_value = sum(abs(e.get('fill_quantity', 0) * e.get('execution_price', 0)) for e in successful_executions)
        avg_cost_bps = (total_costs / total_value * 10000) if total_value > 0 else 0
        
        return {
            'total_executions': len(executions),
            'successful_executions': len(successful_executions),
            'fill_rate': fill_rate,
            'avg_slippage_bps': avg_slippage_bps,
            'max_slippage_bps': max(slippages) if slippages else 0,
            'avg_execution_time_seconds': avg_execution_time,
            'total_transaction_costs': total_costs,
            'avg_cost_bps': avg_cost_bps
        }
    
    def optimize_execution_schedule(self, 
                                  large_order: Order,
                                  time_horizon_minutes: int = 60) -> List[Dict[str, Any]]:
        """Optimize execution schedule for large orders.
        
        Args:
            large_order: Large order to split
            time_horizon_minutes: Time horizon for execution
            
        Returns:
            Optimized execution schedule
        """
        total_quantity = abs(float(large_order.quantity))
        
        # Simple TWAP (Time-Weighted Average Price) strategy
        num_slices = min(10, max(2, time_horizon_minutes // 5))  # 2-10 slices, 5 min minimum per slice
        slice_size = total_quantity / num_slices
        slice_interval = time_horizon_minutes / num_slices
        
        execution_schedule = []
        current_time = datetime.now()
        
        for i in range(num_slices):
            execution_time = current_time + timedelta(minutes=i * slice_interval)
            
            # Adjust slice size (larger slices when volatility is lower)
            volatility_adjustment = random.uniform(0.8, 1.2)  # Simulate volatility
            adjusted_slice_size = slice_size * volatility_adjustment
            
            # Ensure we don't exceed total quantity
            if i == num_slices - 1:  # Last slice
                remaining_quantity = total_quantity - sum(s['quantity'] for s in execution_schedule)
                adjusted_slice_size = remaining_quantity
            
            slice_order = {
                'execution_time': execution_time,
                'quantity': adjusted_slice_size,
                'order_type': large_order.order_type,
                'estimated_impact': self._estimate_slice_impact(adjusted_slice_size, total_quantity)
            }
            
            execution_schedule.append(slice_order)
        
        return execution_schedule
    
    def _estimate_slice_impact(self, slice_size: float, total_size: float) -> float:
        """Estimate market impact for an order slice."""
        participation_rate = slice_size / 100000  # Assume $100k average volume
        impact = 0.1 * np.sqrt(participation_rate)  # Square root impact model
        return min(impact, 0.01)  # Cap at 1%
