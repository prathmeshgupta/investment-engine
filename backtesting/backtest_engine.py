"""Backtesting engine for strategy evaluation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from decimal import Decimal

from core.models import Portfolio, Asset, Position, Transaction, Strategy
from core.enums import AssetClass, RebalanceFrequency
from .transaction_cost_model import TransactionCostModel
from .performance_analyzer import PerformanceAnalyzer


class BacktestEngine:
    """Backtesting engine for investment strategies."""
    
    def __init__(self, 
                 initial_capital: float = 1000000,
                 transaction_cost_model: Optional[TransactionCostModel] = None):
        """Initialize backtest engine.
        
        Args:
            initial_capital: Initial portfolio capital
            transaction_cost_model: Transaction cost model
        """
        self.initial_capital = initial_capital
        self.transaction_cost_model = transaction_cost_model or TransactionCostModel()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Backtest state
        self.portfolio_history = []
        self.transaction_history = []
        self.performance_history = []
        self.rebalance_dates = []
        
    def run_backtest(self,
                    strategy: Strategy,
                    price_data: pd.DataFrame,
                    weight_function: Callable,
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Run backtest for a strategy.
        
        Args:
            strategy: Strategy configuration
            price_data: Price data DataFrame (dates x assets)
            weight_function: Function that returns portfolio weights
            start_date: Backtest start date
            end_date: Backtest end date
            benchmark_returns: Benchmark returns for comparison
            
        Returns:
            Backtest results dictionary
        """
        # Initialize backtest
        self._initialize_backtest()
        
        # Filter data by date range - convert date objects to pandas Timestamp for comparison
        if start_date:
            if hasattr(start_date, 'year'):  # It's a date object
                start_date = pd.Timestamp(start_date)
            price_data = price_data[price_data.index >= start_date]
        if end_date:
            if hasattr(end_date, 'year'):  # It's a date object
                end_date = pd.Timestamp(end_date)
            price_data = price_data[price_data.index <= end_date]
        
        if price_data.empty:
            raise ValueError("No price data available for specified date range")
        
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        # Get rebalancing dates
        rebalance_dates = self._get_rebalance_dates(
            price_data.index, strategy.rebalance_frequency
        )
        
        # Initialize portfolio
        current_portfolio = self._initialize_portfolio(strategy, price_data.iloc[0])
        portfolio_values = []
        
        # Run backtest day by day
        for date in price_data.index:
            # Update portfolio values with price changes
            if date in returns.index:
                current_portfolio = self._update_portfolio_values(
                    current_portfolio, returns.loc[date], price_data.loc[date]
                )
            
            # Check if rebalancing is needed - convert to comparable types
            if any(pd.Timestamp(date) == pd.Timestamp(rd) for rd in rebalance_dates):
                # Get new target weights
                lookback_data = self._get_lookback_data(price_data, date, 252)
                target_weights = weight_function(lookback_data, date)
                
                # Rebalance portfolio
                current_portfolio, transactions = self._rebalance_portfolio(
                    current_portfolio, target_weights, price_data.loc[date], date
                )
                
                self.transaction_history.extend(transactions)
                self.rebalance_dates.append(date)
            
            # Record portfolio state
            portfolio_value = float(current_portfolio.total_value)
            portfolio_values.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': float(current_portfolio.cash),
                'positions': dict(current_portfolio.weights)
            })
        
        # Store results
        self.portfolio_history = portfolio_values
        
        # Calculate performance metrics
        portfolio_returns = self._calculate_portfolio_returns(portfolio_values)
        performance_metrics = self.performance_analyzer.calculate_metrics(
            portfolio_returns, benchmark_returns
        )
        
        # Compile results
        results = {
            'strategy': strategy,
            'start_date': price_data.index[0],
            'end_date': price_data.index[-1],
            'initial_capital': self.initial_capital,
            'final_value': portfolio_values[-1]['portfolio_value'],
            'total_return': (portfolio_values[-1]['portfolio_value'] / self.initial_capital) - 1,
            'portfolio_history': portfolio_values,
            'transaction_history': self.transaction_history,
            'rebalance_dates': self.rebalance_dates,
            'performance_metrics': performance_metrics,
            'transaction_costs': self._calculate_total_transaction_costs()
        }
        
        return results
    
    def _initialize_backtest(self) -> None:
        """Initialize backtest state."""
        self.portfolio_history = []
        self.transaction_history = []
        self.performance_history = []
        self.rebalance_dates = []
    
    def _initialize_portfolio(self, strategy: Strategy, initial_prices: pd.Series) -> Portfolio:
        """Initialize portfolio with cash."""
        portfolio = Portfolio(
            name=f"{strategy.name}_backtest",
            cash=Decimal(str(self.initial_capital))
        )
        return portfolio
    
    def _get_rebalance_dates(self, 
                           date_index: pd.DatetimeIndex,
                           frequency: RebalanceFrequency) -> List[datetime]:
        """Get rebalancing dates based on frequency."""
        rebalance_dates = []
        
        if frequency == RebalanceFrequency.DAILY:
            rebalance_dates = date_index.tolist()
        elif frequency == RebalanceFrequency.WEEKLY:
            # Rebalance on Mondays
            rebalance_dates = [d for d in date_index if d.weekday() == 0]
        elif frequency == RebalanceFrequency.MONTHLY:
            # Rebalance on first trading day of month
            ser = pd.Series(date_index, index=date_index)
            monthly_dates = ser.groupby([date_index.year, date_index.month]).first()
            rebalance_dates = monthly_dates.tolist()
        elif frequency == RebalanceFrequency.QUARTERLY:
            # Rebalance quarterly
            ser = pd.Series(date_index, index=date_index)
            quarterly_dates = ser.groupby([date_index.year, date_index.quarter]).first()
            rebalance_dates = quarterly_dates.tolist()
        elif frequency == RebalanceFrequency.ANNUALLY:
            # Rebalance annually
            ser = pd.Series(date_index, index=date_index)
            yearly_dates = ser.groupby(date_index.year).first()
            rebalance_dates = yearly_dates.tolist()
        
        return rebalance_dates
    
    def _get_lookback_data(self, 
                          price_data: pd.DataFrame,
                          current_date: datetime,
                          lookback_days: int) -> pd.DataFrame:
        """Get lookback data for strategy calculation."""
        end_idx = price_data.index.get_loc(current_date)
        start_idx = max(0, end_idx - lookback_days)
        
        return price_data.iloc[start_idx:end_idx]
    
    def _update_portfolio_values(self,
                               portfolio: Portfolio,
                               returns: pd.Series,
                               prices: pd.Series) -> Portfolio:
        """Update portfolio values based on price changes."""
        for symbol, position in portfolio.positions.items():
            if symbol in returns.index:
                # Update market value based on return
                mv = float(position.market_value)
                r = float(returns[symbol])
                new_market_value = mv * (1.0 + r)
                position.market_value = Decimal(str(new_market_value))
                position.last_updated = datetime.now()
        
        return portfolio
    
    def _rebalance_portfolio(self,
                           portfolio: Portfolio,
                           target_weights: pd.Series,
                           prices: pd.Series,
                           date: datetime) -> Tuple[Portfolio, List[Transaction]]:
        """Rebalance portfolio to target weights."""
        transactions = []
        total_value = float(portfolio.total_value)
        
        # Calculate target positions
        target_positions = {}
        for symbol, weight in target_weights.items():
            if symbol in prices.index and weight > 1e-6:  # Minimum weight threshold
                target_value = total_value * weight
                if symbol in prices.index:
                    price = prices[symbol]
                    if price > 0:
                        target_quantity = target_value / price
                        target_positions[symbol] = target_quantity
        
        # Calculate trades needed
        current_positions = {pos.asset.symbol: float(pos.quantity) 
                           for pos in portfolio.positions.values()}
        
        # Execute trades
        for symbol, target_qty in target_positions.items():
            current_qty = current_positions.get(symbol, 0)
            trade_qty = target_qty - current_qty
            
            if abs(trade_qty) > 1e-6:  # Minimum trade threshold
                price = prices[symbol]
                
                # Create asset if not exists
                if symbol not in [pos.asset.symbol for pos in portfolio.positions.values()]:
                    asset = Asset(symbol=symbol, name=symbol, asset_class=AssetClass.EQUITY)
                else:
                    asset = next(pos.asset for pos in portfolio.positions.values() 
                               if pos.asset.symbol == symbol)
                
                # Calculate transaction costs
                trade_value = abs(trade_qty * price)
                transaction_cost = self.transaction_cost_model.calculate_cost(
                    symbol, trade_value, 'buy' if trade_qty > 0 else 'sell'
                )
                
                # Create transaction
                transaction = Transaction(
                    asset=asset,
                    quantity=Decimal(str(trade_qty)),
                    price=Decimal(str(price)),
                    transaction_type='buy' if trade_qty > 0 else 'sell',
                    timestamp=date,
                    fees=Decimal(str(transaction_cost))
                )
                transactions.append(transaction)
                
                # Update portfolio
                if symbol in current_positions:
                    # Update existing position
                    position = next(pos for pos in portfolio.positions.values() 
                                  if pos.asset.symbol == symbol)
                    position.quantity = Decimal(str(target_qty))
                    position.market_value = Decimal(str(target_qty * price))
                    position.last_updated = date
                else:
                    # Create new position
                    position = Position(
                        asset=asset,
                        quantity=Decimal(str(target_qty)),
                        market_value=Decimal(str(target_qty * price)),
                        cost_basis=Decimal(str(target_qty * price))
                    )
                    portfolio.add_position(position)
                
                # Update cash
                cash_impact = -trade_qty * price - transaction_cost
                portfolio.cash += Decimal(str(cash_impact))
        
        # Remove positions with zero quantity
        symbols_to_remove = []
        for symbol, position in portfolio.positions.items():
            if float(position.quantity) < 1e-6:
                symbols_to_remove.append(symbol)
        
        for symbol in symbols_to_remove:
            portfolio.remove_position(symbol)
        
        return portfolio, transactions
    
    def _calculate_portfolio_returns(self, portfolio_history: List[Dict]) -> pd.Series:
        """Calculate portfolio returns from history."""
        values = [entry['portfolio_value'] for entry in portfolio_history]
        dates = [entry['date'] for entry in portfolio_history]
        
        portfolio_series = pd.Series(values, index=dates)
        returns = portfolio_series.pct_change().dropna()
        
        return returns
    
    def _calculate_total_transaction_costs(self) -> float:
        """Calculate total transaction costs."""
        return sum(float(transaction.fees) for transaction in self.transaction_history)
    
    def run_walk_forward_analysis(self,
                                strategy: Strategy,
                                price_data: pd.DataFrame,
                                weight_function: Callable,
                                train_window: int = 252,
                                test_window: int = 63) -> Dict[str, Any]:
        """Run walk-forward analysis.
        
        Args:
            strategy: Strategy configuration
            price_data: Price data DataFrame
            weight_function: Weight calculation function
            train_window: Training window size
            test_window: Test window size
            
        Returns:
            Walk-forward analysis results
        """
        results = []
        dates = price_data.index
        
        for i in range(train_window, len(dates) - test_window, test_window):
            # Training period
            train_start = i - train_window
            train_end = i
            train_data = price_data.iloc[train_start:train_end]
            
            # Test period
            test_start = i
            test_end = min(i + test_window, len(dates))
            test_data = price_data.iloc[test_start:test_end]
            
            # Run backtest on test period
            test_result = self.run_backtest(
                strategy=strategy,
                price_data=test_data,
                weight_function=weight_function,
                start_date=test_data.index[0],
                end_date=test_data.index[-1]
            )
            
            results.append({
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'test_return': test_result['total_return'],
                'test_metrics': test_result['performance_metrics']
            })
        
        # Aggregate results
        total_returns = [r['test_return'] for r in results]
        avg_return = np.mean(total_returns)
        return_volatility = np.std(total_returns)
        sharpe_ratio = avg_return / return_volatility if return_volatility > 0 else 0
        
        return {
            'periods': results,
            'summary': {
                'average_return': avg_return,
                'return_volatility': return_volatility,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': sum(1 for r in total_returns if r > 0) / len(total_returns),
                'best_period': max(total_returns),
                'worst_period': min(total_returns)
            }
        }
