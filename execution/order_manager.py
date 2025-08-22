"""Order management system for trade execution."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from decimal import Decimal
import uuid
from enum import Enum

from core.models import Order, Asset, Transaction
from core.enums import OrderType, OrderStatus


class OrderPriority(Enum):
    """Order priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class OrderManager:
    """Manages order lifecycle and execution queue."""
    
    def __init__(self):
        """Initialize order manager."""
        self.orders = {}  # order_id -> order
        self.order_queue = []  # List of pending orders
        self.execution_history = []
        self.order_book = {}  # symbol -> list of orders
        
    def submit_order(self, 
                    order: Order,
                    priority: OrderPriority = OrderPriority.NORMAL) -> str:
        """Submit an order for execution.
        
        Args:
            order: Order to submit
            priority: Order priority
            
        Returns:
            Order ID
        """
        order_id = str(uuid.uuid4())
        
        # Set order metadata
        order_entry = {
            'order_id': order_id,
            'order': order,
            'priority': priority,
            'submitted_at': datetime.now(),
            'status': OrderStatus.PENDING,
            'attempts': 0,
            'last_attempt': None,
            'error_message': None
        }
        
        self.orders[order_id] = order_entry
        
        # Add to queue based on priority
        self._add_to_queue(order_entry)
        
        # Add to order book
        symbol = order.asset.symbol
        if symbol not in self.order_book:
            self.order_book[symbol] = []
        self.order_book[symbol].append(order_entry)
        
        return order_id
    
    def _add_to_queue(self, order_entry: Dict[str, Any]) -> None:
        """Add order to execution queue based on priority."""
        # Insert order in queue based on priority
        inserted = False
        for i, existing_order in enumerate(self.order_queue):
            if order_entry['priority'].value > existing_order['priority'].value:
                self.order_queue.insert(i, order_entry)
                inserted = True
                break
        
        if not inserted:
            self.order_queue.append(order_entry)
    
    def get_next_order(self) -> Optional[Dict[str, Any]]:
        """Get next order from queue for execution.
        
        Returns:
            Next order entry or None if queue is empty
        """
        # Filter out cancelled or filled orders
        while self.order_queue:
            order_entry = self.order_queue[0]
            if order_entry['status'] in [OrderStatus.PENDING, OrderStatus.PARTIAL_FILL]:
                return self.order_queue.pop(0)
            else:
                self.order_queue.pop(0)
        
        return None
    
    def update_order_status(self, 
                          order_id: str, 
                          status: OrderStatus,
                          filled_quantity: Optional[Decimal] = None,
                          fill_price: Optional[Decimal] = None,
                          error_message: Optional[str] = None) -> None:
        """Update order status.
        
        Args:
            order_id: Order ID
            status: New status
            filled_quantity: Filled quantity (for partial fills)
            fill_price: Fill price
            error_message: Error message if failed
        """
        if order_id not in self.orders:
            return
        
        order_entry = self.orders[order_id]
        order_entry['status'] = status
        order_entry['last_attempt'] = datetime.now()
        
        if error_message:
            order_entry['error_message'] = error_message
        
        if filled_quantity is not None:
            order_entry['filled_quantity'] = filled_quantity
        
        if fill_price is not None:
            order_entry['fill_price'] = fill_price
        
        # Create transaction record for fills
        if status in [OrderStatus.FILLED, OrderStatus.PARTIAL_FILL] and filled_quantity:
            transaction = Transaction(
                asset=order_entry['order'].asset,
                quantity=filled_quantity,
                price=fill_price or order_entry['order'].price,
                transaction_type='buy' if filled_quantity > 0 else 'sell',
                timestamp=datetime.now(),
                fees=Decimal('0')  # Would be calculated based on broker
            )
            
            self.execution_history.append({
                'order_id': order_id,
                'transaction': transaction,
                'timestamp': datetime.now()
            })
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        if order_id not in self.orders:
            return False
        
        order_entry = self.orders[order_id]
        
        if order_entry['status'] in [OrderStatus.PENDING, OrderStatus.PARTIAL_FILL]:
            order_entry['status'] = OrderStatus.CANCELLED
            order_entry['last_attempt'] = datetime.now()
            return True
        
        return False
    
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order status.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status information
        """
        if order_id not in self.orders:
            return None
        
        order_entry = self.orders[order_id]
        order = order_entry['order']
        
        return {
            'order_id': order_id,
            'symbol': order.asset.symbol,
            'quantity': float(order.quantity),
            'order_type': order.order_type.value,
            'price': float(order.price) if order.price else None,
            'status': order_entry['status'].value,
            'submitted_at': order_entry['submitted_at'],
            'last_attempt': order_entry['last_attempt'],
            'attempts': order_entry['attempts'],
            'filled_quantity': float(order_entry.get('filled_quantity', 0)),
            'fill_price': float(order_entry.get('fill_price', 0)) if order_entry.get('fill_price') else None,
            'error_message': order_entry.get('error_message')
        }
    
    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get pending orders.
        
        Args:
            symbol: Filter by symbol
            
        Returns:
            List of pending orders
        """
        pending_orders = []
        
        for order_id, order_entry in self.orders.items():
            if order_entry['status'] in [OrderStatus.PENDING, OrderStatus.PARTIAL_FILL]:
                if symbol is None or order_entry['order'].asset.symbol == symbol:
                    pending_orders.append(self.get_order_status(order_id))
        
        return pending_orders
    
    def get_order_book_summary(self, symbol: str) -> Dict[str, Any]:
        """Get order book summary for a symbol.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Order book summary
        """
        if symbol not in self.order_book:
            return {
                'symbol': symbol,
                'total_orders': 0,
                'pending_orders': 0,
                'buy_orders': 0,
                'sell_orders': 0,
                'total_buy_quantity': 0,
                'total_sell_quantity': 0
            }
        
        orders = self.order_book[symbol]
        
        total_orders = len(orders)
        pending_orders = sum(1 for o in orders if o['status'] in [OrderStatus.PENDING, OrderStatus.PARTIAL_FILL])
        
        buy_orders = 0
        sell_orders = 0
        total_buy_quantity = 0
        total_sell_quantity = 0
        
        for order_entry in orders:
            order = order_entry['order']
            quantity = float(order.quantity)
            
            if quantity > 0:
                buy_orders += 1
                total_buy_quantity += quantity
            else:
                sell_orders += 1
                total_sell_quantity += abs(quantity)
        
        return {
            'symbol': symbol,
            'total_orders': total_orders,
            'pending_orders': pending_orders,
            'buy_orders': buy_orders,
            'sell_orders': sell_orders,
            'total_buy_quantity': total_buy_quantity,
            'total_sell_quantity': total_sell_quantity
        }
    
    def batch_submit_orders(self, orders: List[Tuple[Order, OrderPriority]]) -> List[str]:
        """Submit multiple orders in batch.
        
        Args:
            orders: List of (order, priority) tuples
            
        Returns:
            List of order IDs
        """
        order_ids = []
        
        for order, priority in orders:
            order_id = self.submit_order(order, priority)
            order_ids.append(order_id)
        
        return order_ids
    
    def get_execution_statistics(self, 
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get execution statistics.
        
        Args:
            start_date: Start date for statistics
            end_date: End date for statistics
            
        Returns:
            Execution statistics
        """
        # Filter orders by date range
        filtered_orders = []
        for order_entry in self.orders.values():
            submitted_at = order_entry['submitted_at']
            if start_date and submitted_at < start_date:
                continue
            if end_date and submitted_at > end_date:
                continue
            filtered_orders.append(order_entry)
        
        if not filtered_orders:
            return {
                'total_orders': 0,
                'filled_orders': 0,
                'cancelled_orders': 0,
                'failed_orders': 0,
                'fill_rate': 0,
                'avg_execution_time': 0
            }
        
        total_orders = len(filtered_orders)
        filled_orders = sum(1 for o in filtered_orders if o['status'] == OrderStatus.FILLED)
        cancelled_orders = sum(1 for o in filtered_orders if o['status'] == OrderStatus.CANCELLED)
        failed_orders = sum(1 for o in filtered_orders if o['status'] == OrderStatus.FAILED)
        
        # Calculate average execution time for filled orders
        execution_times = []
        for order_entry in filtered_orders:
            if order_entry['status'] == OrderStatus.FILLED and order_entry['last_attempt']:
                execution_time = (order_entry['last_attempt'] - order_entry['submitted_at']).total_seconds()
                execution_times.append(execution_time)
        
        avg_execution_time = np.mean(execution_times) if execution_times else 0
        
        return {
            'total_orders': total_orders,
            'filled_orders': filled_orders,
            'cancelled_orders': cancelled_orders,
            'failed_orders': failed_orders,
            'pending_orders': total_orders - filled_orders - cancelled_orders - failed_orders,
            'fill_rate': filled_orders / total_orders if total_orders > 0 else 0,
            'cancel_rate': cancelled_orders / total_orders if total_orders > 0 else 0,
            'fail_rate': failed_orders / total_orders if total_orders > 0 else 0,
            'avg_execution_time_seconds': avg_execution_time
        }
    
    def cleanup_old_orders(self, days_old: int = 30) -> int:
        """Clean up old completed orders.
        
        Args:
            days_old: Remove orders older than this many days
            
        Returns:
            Number of orders removed
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        orders_to_remove = []
        
        for order_id, order_entry in self.orders.items():
            if (order_entry['submitted_at'] < cutoff_date and 
                order_entry['status'] in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.FAILED]):
                orders_to_remove.append(order_id)
        
        # Remove old orders
        for order_id in orders_to_remove:
            del self.orders[order_id]
            
            # Remove from order book
            order_entry = self.orders.get(order_id)
            if order_entry:
                symbol = order_entry['order'].asset.symbol
                if symbol in self.order_book:
                    self.order_book[symbol] = [
                        o for o in self.order_book[symbol] 
                        if o['order_id'] != order_id
                    ]
        
        return len(orders_to_remove)
