"""Main execution engine for strategy implementation."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
from concurrent.futures import ThreadPoolExecutor

from core.models import Portfolio, Strategy, Order, Asset, Position, Transaction
from core.enums import OrderType, OrderStatus, RebalanceFrequency
from data.data_manager import DataManager
from factors.factor_engine import FactorEngine
from optimization.optimizer_engine import OptimizerEngine
from risk.risk_manager import RiskManager
from .rebalancer import Rebalancer
from .order_manager import OrderManager
from .trade_executor import TradeExecutor


class ExecutionEngine:
    """Main execution engine coordinating strategy implementation."""
    
    def __init__(self,
                 data_manager: DataManager,
                 factor_engine: FactorEngine,
                 optimizer_engine: OptimizerEngine,
                 risk_manager: RiskManager):
        """Initialize execution engine.
        
        Args:
            data_manager: Data management system
            factor_engine: Factor model engine
            optimizer_engine: Portfolio optimization engine
            risk_manager: Risk management system
        """
        self.data_manager = data_manager
        self.factor_engine = factor_engine
        self.optimizer_engine = optimizer_engine
        self.risk_manager = risk_manager
        
        self.rebalancer = Rebalancer()
        self.order_manager = OrderManager()
        self.trade_executor = TradeExecutor()
        
        self.active_strategies = {}
        self.execution_history = []
        self.performance_tracking = {}
        
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def register_strategy(self, strategy: Strategy, portfolio: Portfolio) -> str:
        """Register a strategy for execution.
        
        Args:
            strategy: Strategy configuration
            portfolio: Associated portfolio
            
        Returns:
            Strategy execution ID
        """
        execution_id = f"{strategy.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.active_strategies[execution_id] = {
            'strategy': strategy,
            'portfolio': portfolio,
            'status': 'registered',
            'last_rebalance': None,
            'next_rebalance': self._calculate_next_rebalance(strategy),
            'execution_count': 0,
            'alerts': []
        }
        
        return execution_id
    
    def execute_strategy(self,
                        execution_id: str,
                        force_rebalance: bool = False) -> Dict[str, Any]:
        """Execute a registered strategy.
        
        Args:
            execution_id: Strategy execution ID
            force_rebalance: Force rebalancing regardless of schedule
            
        Returns:
            Execution results
        """
        if execution_id not in self.active_strategies:
            raise ValueError(f"Strategy {execution_id} not registered")
        
        strategy_info = self.active_strategies[execution_id]
        strategy = strategy_info['strategy']
        portfolio = strategy_info['portfolio']
        
        # Check if rebalancing is needed
        current_time = datetime.now()
        needs_rebalance = (
            force_rebalance or 
            strategy_info['next_rebalance'] is None or
            current_time >= strategy_info['next_rebalance']
        )
        
        if not needs_rebalance:
            return {
                'execution_id': execution_id,
                'status': 'skipped',
                'reason': 'Not time for rebalancing',
                'next_rebalance': strategy_info['next_rebalance']
            }
        
        try:
            # Update strategy status
            strategy_info['status'] = 'executing'
            
            # Get market data
            symbols = list(strategy.universe)
            end_date = current_time.date()
            start_date = end_date - timedelta(days=365)  # 1 year lookback
            
            market_data = self.data_manager.get_batch_data(
                symbols, start_date, end_date
            )
            
            if not market_data.get('returns', pd.DataFrame()).empty:
                returns_data = market_data['returns']
                price_data = market_data.get('prices', pd.DataFrame())
                
                # Calculate factor exposures and expected returns
                factor_results = self.factor_engine.calculate_combined_factors(
                    returns_data, symbols
                )
                
                # Optimize portfolio
                optimization_result = self.optimizer_engine.optimize_portfolio(
                    expected_returns=factor_results['expected_returns'],
                    covariance_matrix=factor_results['covariance_matrix'],
                    method=strategy.optimization_method,
                    constraints=strategy.constraints
                )
                
                target_weights = optimization_result['weights']
                
                # Risk check
                risk_check = self.risk_manager.check_risk_limits(
                    target_weights, returns_data
                )
                
                if not risk_check['passed']:
                    strategy_info['alerts'].extend([
                        f"Risk limit violation: {violation}" 
                        for violation in risk_check['violations']
                    ])
                    
                    # Apply risk adjustments
                    risk_suggestions = self.risk_manager.suggest_risk_adjustments(
                        portfolio, returns_data
                    )
                    
                    if risk_suggestions['suggestions']:
                        # Implement risk adjustments
                        target_weights = self._apply_risk_adjustments(
                            target_weights, risk_suggestions
                        )
                
                # Execute rebalancing
                rebalance_result = self.rebalancer.rebalance_portfolio(
                    portfolio=portfolio,
                    target_weights=target_weights,
                    current_prices=price_data.iloc[-1] if not price_data.empty else pd.Series(),
                    transaction_cost_model=None  # Use default
                )
                
                # Execute trades
                if rebalance_result['trades']:
                    execution_result = self._execute_trades(
                        rebalance_result['trades'], execution_id
                    )
                else:
                    execution_result = {'trades_executed': 0, 'total_cost': 0}
                
                # Update strategy info
                strategy_info['last_rebalance'] = current_time
                strategy_info['next_rebalance'] = self._calculate_next_rebalance(strategy)
                strategy_info['execution_count'] += 1
                strategy_info['status'] = 'completed'
                
                # Record execution history
                execution_record = {
                    'execution_id': execution_id,
                    'timestamp': current_time,
                    'target_weights': target_weights.to_dict(),
                    'trades': rebalance_result['trades'],
                    'execution_result': execution_result,
                    'risk_metrics': risk_check,
                    'optimization_result': optimization_result
                }
                
                self.execution_history.append(execution_record)
                
                return {
                    'execution_id': execution_id,
                    'status': 'completed',
                    'timestamp': current_time,
                    'trades_executed': execution_result['trades_executed'],
                    'total_cost': execution_result['total_cost'],
                    'target_weights': target_weights.to_dict(),
                    'next_rebalance': strategy_info['next_rebalance'],
                    'alerts': strategy_info['alerts']
                }
            
            else:
                strategy_info['status'] = 'failed'
                return {
                    'execution_id': execution_id,
                    'status': 'failed',
                    'reason': 'No market data available'
                }
        
        except Exception as e:
            strategy_info['status'] = 'failed'
            strategy_info['alerts'].append(f"Execution error: {str(e)}")
            
            return {
                'execution_id': execution_id,
                'status': 'failed',
                'reason': str(e),
                'timestamp': current_time
            }
    
    def _calculate_next_rebalance(self, strategy: Strategy) -> datetime:
        """Calculate next rebalancing date."""
        current_time = datetime.now()
        
        if strategy.rebalance_frequency == RebalanceFrequency.DAILY:
            return current_time + timedelta(days=1)
        elif strategy.rebalance_frequency == RebalanceFrequency.WEEKLY:
            days_ahead = 7 - current_time.weekday()
            return current_time + timedelta(days=days_ahead)
        elif strategy.rebalance_frequency == RebalanceFrequency.MONTHLY:
            if current_time.month == 12:
                next_month = current_time.replace(year=current_time.year + 1, month=1, day=1)
            else:
                next_month = current_time.replace(month=current_time.month + 1, day=1)
            return next_month
        elif strategy.rebalance_frequency == RebalanceFrequency.QUARTERLY:
            current_quarter = (current_time.month - 1) // 3 + 1
            if current_quarter == 4:
                next_quarter_start = current_time.replace(year=current_time.year + 1, month=1, day=1)
            else:
                next_quarter_month = current_quarter * 3 + 1
                next_quarter_start = current_time.replace(month=next_quarter_month, day=1)
            return next_quarter_start
        elif strategy.rebalance_frequency == RebalanceFrequency.ANNUALLY:
            return current_time.replace(year=current_time.year + 1, month=1, day=1)
        
        return current_time + timedelta(days=30)  # Default to monthly
    
    def _apply_risk_adjustments(self,
                              target_weights: pd.Series,
                              risk_suggestions: Dict[str, Any]) -> pd.Series:
        """Apply risk management adjustments to target weights."""
        adjusted_weights = target_weights.copy()
        
        for suggestion in risk_suggestions['suggestions']:
            if suggestion['type'] == 'reduce_concentration':
                asset = suggestion['asset']
                target_weight = suggestion['suggested_weight']
                
                if asset in adjusted_weights:
                    adjusted_weights[asset] = min(adjusted_weights[asset], target_weight)
        
        # Renormalize weights
        total_weight = adjusted_weights.sum()
        if total_weight > 0:
            adjusted_weights = adjusted_weights / total_weight
        
        return adjusted_weights
    
    def _execute_trades(self,
                       trades: List[Dict[str, Any]],
                       execution_id: str) -> Dict[str, Any]:
        """Execute trades through trade executor."""
        executed_trades = 0
        total_cost = 0.0
        failed_trades = []
        
        for trade in trades:
            try:
                # Create order
                order = Order(
                    asset=trade['asset'],
                    quantity=Decimal(str(trade['quantity'])),
                    order_type=OrderType.MARKET,
                    price=Decimal(str(trade['price'])) if 'price' in trade else None
                )
                
                # Submit order
                order_id = self.order_manager.submit_order(order)
                
                # Execute order (simplified - in practice would integrate with broker)
                execution_result = self.trade_executor.execute_order(order_id)
                
                if execution_result['status'] == 'filled':
                    executed_trades += 1
                    total_cost += execution_result.get('cost', 0)
                else:
                    failed_trades.append({
                        'trade': trade,
                        'reason': execution_result.get('reason', 'Unknown')
                    })
            
            except Exception as e:
                failed_trades.append({
                    'trade': trade,
                    'reason': str(e)
                })
        
        return {
            'trades_executed': executed_trades,
            'total_cost': total_cost,
            'failed_trades': failed_trades
        }
    
    def run_continuous_execution(self, check_interval: int = 300) -> None:
        """Run continuous execution loop.
        
        Args:
            check_interval: Check interval in seconds
        """
        async def execution_loop():
            while True:
                try:
                    # Check all active strategies
                    for execution_id in list(self.active_strategies.keys()):
                        strategy_info = self.active_strategies[execution_id]
                        
                        if strategy_info['status'] in ['registered', 'completed']:
                            # Execute strategy if needed
                            result = self.execute_strategy(execution_id)
                            
                            if result['status'] == 'failed':
                                print(f"Strategy {execution_id} failed: {result.get('reason', 'Unknown')}")
                    
                    # Wait for next check
                    await asyncio.sleep(check_interval)
                
                except Exception as e:
                    print(f"Error in execution loop: {e}")
                    await asyncio.sleep(check_interval)
        
        # Run the execution loop
        asyncio.run(execution_loop())
    
    def get_strategy_status(self, execution_id: str) -> Dict[str, Any]:
        """Get status of a strategy execution.
        
        Args:
            execution_id: Strategy execution ID
            
        Returns:
            Strategy status information
        """
        if execution_id not in self.active_strategies:
            return {'error': 'Strategy not found'}
        
        strategy_info = self.active_strategies[execution_id]
        
        return {
            'execution_id': execution_id,
            'strategy_name': strategy_info['strategy'].name,
            'status': strategy_info['status'],
            'last_rebalance': strategy_info['last_rebalance'],
            'next_rebalance': strategy_info['next_rebalance'],
            'execution_count': strategy_info['execution_count'],
            'alerts': strategy_info['alerts']
        }
    
    def get_execution_history(self, 
                            execution_id: Optional[str] = None,
                            limit: int = 50) -> List[Dict[str, Any]]:
        """Get execution history.
        
        Args:
            execution_id: Filter by execution ID
            limit: Maximum number of records
            
        Returns:
            List of execution records
        """
        history = self.execution_history
        
        if execution_id:
            history = [record for record in history if record['execution_id'] == execution_id]
        
        # Sort by timestamp (most recent first)
        history = sorted(history, key=lambda x: x['timestamp'], reverse=True)
        
        return history[:limit]
    
    def stop_strategy(self, execution_id: str) -> Dict[str, Any]:
        """Stop a strategy execution.
        
        Args:
            execution_id: Strategy execution ID
            
        Returns:
            Stop result
        """
        if execution_id not in self.active_strategies:
            return {'error': 'Strategy not found'}
        
        strategy_info = self.active_strategies[execution_id]
        strategy_info['status'] = 'stopped'
        
        return {
            'execution_id': execution_id,
            'status': 'stopped',
            'timestamp': datetime.now()
        }
    
    def get_performance_summary(self, execution_id: str) -> Dict[str, Any]:
        """Get performance summary for a strategy.
        
        Args:
            execution_id: Strategy execution ID
            
        Returns:
            Performance summary
        """
        if execution_id not in self.active_strategies:
            return {'error': 'Strategy not found'}
        
        strategy_info = self.active_strategies[execution_id]
        portfolio = strategy_info['portfolio']
        
        # Get execution history for this strategy
        strategy_history = [
            record for record in self.execution_history 
            if record['execution_id'] == execution_id
        ]
        
        if not strategy_history:
            return {'error': 'No execution history found'}
        
        # Calculate basic performance metrics
        total_executions = len(strategy_history)
        successful_executions = sum(1 for record in strategy_history 
                                  if record.get('execution_result', {}).get('trades_executed', 0) > 0)
        
        total_costs = sum(record.get('execution_result', {}).get('total_cost', 0) 
                         for record in strategy_history)
        
        return {
            'execution_id': execution_id,
            'strategy_name': strategy_info['strategy'].name,
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'success_rate': successful_executions / total_executions if total_executions > 0 else 0,
            'total_transaction_costs': total_costs,
            'current_portfolio_value': float(portfolio.total_value),
            'last_execution': strategy_history[0]['timestamp'] if strategy_history else None
        }
