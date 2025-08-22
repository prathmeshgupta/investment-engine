"""Centralized Error Handling and Logging System."""

import logging
import traceback
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict, Callable
from functools import wraps
import json
import sqlite3
from enum import Enum
import warnings
import asyncio


class ErrorSeverity(Enum):
    """Error severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ErrorCategory(Enum):
    """Error categories for classification."""
    DATA = "DATA"
    CALCULATION = "CALCULATION"
    NETWORK = "NETWORK"
    DATABASE = "DATABASE"
    CONFIGURATION = "CONFIGURATION"
    VALIDATION = "VALIDATION"
    AUTHENTICATION = "AUTHENTICATION"
    SYSTEM = "SYSTEM"
    USER_INPUT = "USER_INPUT"
    INTEGRATION = "INTEGRATION"


class ErrorHandler:
    """Centralized error handling with logging and recovery strategies."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        self.db_path = Path("data/error_logs.db")
        self.db_path.parent.mkdir(exist_ok=True)
        
        self.loggers = {}
        self.error_counts = {}
        self.recovery_strategies = {}
        
        self._setup_database()
        self._setup_main_logger()
        self._initialized = True
    
    def _setup_database(self):
        """Setup error logging database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS error_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                severity TEXT,
                category TEXT,
                module TEXT,
                function TEXT,
                error_type TEXT,
                error_message TEXT,
                stack_trace TEXT,
                user_id TEXT,
                session_id TEXT,
                context TEXT,
                resolved BOOLEAN DEFAULT FALSE
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS error_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                category TEXT,
                severity TEXT,
                count INTEGER,
                UNIQUE(date, category, severity)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _setup_main_logger(self):
        """Setup main application logger."""
        logger = logging.getLogger('InvestmentEngine')
        logger.setLevel(logging.DEBUG)
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            self.log_dir / 'investment_engine.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        self.loggers['main'] = logger
    
    def get_logger(self, module_name: str) -> logging.Logger:
        """Get or create a logger for a specific module."""
        if module_name not in self.loggers:
            logger = logging.getLogger(f'InvestmentEngine.{module_name}')
            logger.setLevel(logging.DEBUG)
            
            # Inherit handlers from main logger
            for handler in self.loggers['main'].handlers:
                logger.addHandler(handler)
            
            self.loggers[module_name] = logger
        
        return self.loggers[module_name]
    
    def log_error(self, 
                  error: Exception,
                  severity: ErrorSeverity = ErrorSeverity.ERROR,
                  category: ErrorCategory = ErrorCategory.SYSTEM,
                  module: str = None,
                  function: str = None,
                  context: Dict[str, Any] = None,
                  user_id: str = None,
                  session_id: str = None):
        """Log an error with full context."""
        
        # Get caller info if not provided
        if not module or not function:
            frame = sys._getframe(1)
            module = module or frame.f_code.co_filename
            function = function or frame.f_code.co_name
        
        # Format error details
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()
        
        # Log to file
        logger = self.get_logger(module)
        log_message = f"{category.value} - {error_type}: {error_message}"
        
        if severity == ErrorSeverity.DEBUG:
            logger.debug(log_message)
        elif severity == ErrorSeverity.INFO:
            logger.info(log_message)
        elif severity == ErrorSeverity.WARNING:
            logger.warning(log_message)
        elif severity == ErrorSeverity.ERROR:
            logger.error(log_message)
        elif severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        
        # Save to database
        self._save_to_database(
            severity=severity.value,
            category=category.value,
            module=module,
            function=function,
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            user_id=user_id,
            session_id=session_id,
            context=json.dumps(context) if context else None
        )
        
        # Update metrics
        self._update_error_metrics(category.value, severity.value)
        
        # Check if recovery strategy exists
        if category in self.recovery_strategies:
            self._attempt_recovery(error, category)
    
    def _save_to_database(self, **kwargs):
        """Save error to database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO error_logs 
                (severity, category, module, function, error_type, 
                 error_message, stack_trace, user_id, session_id, context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                kwargs.get('severity'),
                kwargs.get('category'),
                kwargs.get('module'),
                kwargs.get('function'),
                kwargs.get('error_type'),
                kwargs.get('error_message'),
                kwargs.get('stack_trace'),
                kwargs.get('user_id'),
                kwargs.get('session_id'),
                kwargs.get('context')
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            # Fallback to file logging if database fails
            self.loggers['main'].error(f"Failed to save error to database: {e}")
    
    def _update_error_metrics(self, category: str, severity: str):
        """Update error metrics."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            today = datetime.now().date()
            
            cursor.execute("""
                INSERT OR REPLACE INTO error_metrics (date, category, severity, count)
                VALUES (?, ?, ?, 
                    COALESCE((SELECT count + 1 FROM error_metrics 
                              WHERE date = ? AND category = ? AND severity = ?), 1))
            """, (today, category, severity, today, category, severity))
            
            conn.commit()
            conn.close()
        except Exception as e:
            self.loggers['main'].error(f"Failed to update error metrics: {e}")
    
    def register_recovery_strategy(self, 
                                 category: ErrorCategory,
                                 strategy: Callable[[Exception], None]):
        """Register a recovery strategy for an error category."""
        self.recovery_strategies[category] = strategy
    
    def _attempt_recovery(self, error: Exception, category: ErrorCategory):
        """Attempt to recover from an error."""
        try:
            strategy = self.recovery_strategies[category]
            strategy(error)
            self.loggers['main'].info(f"Recovery attempted for {category.value} error")
        except Exception as e:
            self.loggers['main'].error(f"Recovery failed: {e}")
    
    def get_error_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get error statistics for the last N days."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        from datetime import timedelta
        start_date = datetime.now().date() - timedelta(days=days)
        
        # Get error counts by category
        cursor.execute("""
            SELECT category, COUNT(*) as count
            FROM error_logs
            WHERE timestamp >= ?
            GROUP BY category
        """, (start_date,))
        
        by_category = dict(cursor.fetchall())
        
        # Get error counts by severity
        cursor.execute("""
            SELECT severity, COUNT(*) as count
            FROM error_logs
            WHERE timestamp >= ?
            GROUP BY severity
        """, (start_date,))
        
        by_severity = dict(cursor.fetchall())
        
        # Get recent errors
        cursor.execute("""
            SELECT timestamp, severity, category, error_message
            FROM error_logs
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT 10
        """, (start_date,))
        
        recent_errors = cursor.fetchall()
        
        conn.close()
        
        return {
            'by_category': by_category,
            'by_severity': by_severity,
            'recent_errors': recent_errors,
            'total_errors': sum(by_category.values())
        }


def with_error_handling(category: ErrorCategory = ErrorCategory.SYSTEM,
                        severity: ErrorSeverity = ErrorSeverity.ERROR,
                        default_return: Any = None):
    """Decorator for automatic error handling."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler = ErrorHandler()
                handler.log_error(
                    error=e,
                    severity=severity,
                    category=category,
                    module=func.__module__,
                    function=func.__name__,
                    context={'args': str(args), 'kwargs': str(kwargs)}
                )
                
                if default_return is not None:
                    return default_return
                raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                handler = ErrorHandler()
                handler.log_error(
                    error=e,
                    severity=severity,
                    category=category,
                    module=func.__module__,
                    function=func.__name__,
                    context={'args': str(args), 'kwargs': str(kwargs)}
                )
                
                if default_return is not None:
                    return default_return
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def validate_input(validation_rules: Dict[str, Callable]):
    """Decorator for input validation."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate kwargs
            for param_name, validator in validation_rules.items():
                if param_name in kwargs:
                    value = kwargs[param_name]
                    if not validator(value):
                        raise ValueError(f"Invalid value for {param_name}: {value}")
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


class SafeExecutor:
    """Context manager for safe execution with error handling."""
    
    def __init__(self, 
                 operation_name: str,
                 category: ErrorCategory = ErrorCategory.SYSTEM,
                 fallback_value: Any = None):
        self.operation_name = operation_name
        self.category = category
        self.fallback_value = fallback_value
        self.handler = ErrorHandler()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self.handler.log_error(
                error=exc_val,
                category=self.category,
                context={'operation': self.operation_name}
            )
            
            if self.fallback_value is not None:
                return True  # Suppress exception
        
        return False


# Example recovery strategies
def data_error_recovery(error: Exception):
    """Recovery strategy for data errors."""
    # Clear cache, retry data fetch, use fallback data source
    print(f"Attempting data recovery: {error}")
    # Implementation would go here


def network_error_recovery(error: Exception):
    """Recovery strategy for network errors."""
    # Retry with exponential backoff, use cached data
    print(f"Attempting network recovery: {error}")
    # Implementation would go here


# Initialize error handler and register default strategies
def initialize_error_handling():
    """Initialize error handling system."""
    handler = ErrorHandler()
    
    # Register recovery strategies
    handler.register_recovery_strategy(ErrorCategory.DATA, data_error_recovery)
    handler.register_recovery_strategy(ErrorCategory.NETWORK, network_error_recovery)
    
    # Set up global exception handler
    def global_exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        handler.log_error(
            error=exc_value,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SYSTEM,
            context={'uncaught': True}
        )
    
    sys.excepthook = global_exception_handler
    
    return handler
