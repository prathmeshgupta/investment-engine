"""Configuration management system."""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

from .settings import Settings


class ConfigManager:
    """Manages configuration settings for the investment engine."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or self._get_default_config_path()
        self.settings = Settings()
        self._load_config()
        
        # Setup logging
        self._setup_logging()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        return os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    def _load_config(self) -> None:
        """Load configuration from file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)
                
                # Update settings with loaded configuration
                self._update_settings(config_data)
                
            except Exception as e:
                print(f"Error loading configuration: {e}")
                print("Using default settings")
        else:
            # Create default configuration file
            self.save_config()
    
    def _update_settings(self, config_data: Dict[str, Any]) -> None:
        """Update settings object with configuration data."""
        for section, values in config_data.items():
            if hasattr(self.settings, section):
                section_obj = getattr(self.settings, section)
                if hasattr(section_obj, '__dict__'):
                    for key, value in values.items():
                        if hasattr(section_obj, key):
                            setattr(section_obj, key, value)
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        config_data = asdict(self.settings)
        
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_data, f, indent=2, default=str)
                    
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def get_setting(self, section: str, key: str, default: Any = None) -> Any:
        """Get a specific setting value.
        
        Args:
            section: Configuration section
            key: Setting key
            default: Default value if not found
            
        Returns:
            Setting value
        """
        try:
            section_obj = getattr(self.settings, section)
            return getattr(section_obj, key, default)
        except AttributeError:
            return default
    
    def set_setting(self, section: str, key: str, value: Any) -> None:
        """Set a specific setting value.
        
        Args:
            section: Configuration section
            key: Setting key
            value: Setting value
        """
        try:
            section_obj = getattr(self.settings, section)
            setattr(section_obj, key, value)
        except AttributeError:
            print(f"Invalid configuration section or key: {section}.{key}")
    
    def update_settings(self, updates: Dict[str, Dict[str, Any]]) -> None:
        """Update multiple settings.
        
        Args:
            updates: Dictionary of section -> {key: value} updates
        """
        for section, settings_dict in updates.items():
            for key, value in settings_dict.items():
                self.set_setting(section, key, value)
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return {
            'url': self.settings.database.url,
            'pool_size': self.settings.database.pool_size,
            'max_overflow': self.settings.database.max_overflow,
            'pool_timeout': self.settings.database.pool_timeout
        }
    
    def get_data_provider_config(self) -> Dict[str, Any]:
        """Get data provider configuration."""
        return {
            'primary_provider': self.settings.data.primary_provider,
            'cache_enabled': self.settings.data.cache_enabled,
            'cache_ttl': self.settings.data.cache_ttl,
            'max_retries': self.settings.data.max_retries,
            'timeout': self.settings.data.timeout
        }
    
    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration."""
        return {
            'max_portfolio_var': self.settings.risk.max_portfolio_var,
            'max_individual_weight': self.settings.risk.max_individual_weight,
            'max_sector_weight': self.settings.risk.max_sector_weight,
            'max_drawdown': self.settings.risk.max_drawdown,
            'var_confidence_level': self.settings.risk.var_confidence_level
        }
    
    def get_execution_config(self) -> Dict[str, Any]:
        """Get execution configuration."""
        return {
            'default_slippage': self.settings.execution.default_slippage,
            'max_order_size': self.settings.execution.max_order_size,
            'execution_delay': self.settings.execution.execution_delay,
            'retry_attempts': self.settings.execution.retry_attempts
        }
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration settings.
        
        Returns:
            Validation results
        """
        errors = []
        warnings = []
        
        # Validate risk settings
        if self.settings.risk.max_portfolio_var <= 0 or self.settings.risk.max_portfolio_var > 1:
            errors.append("max_portfolio_var must be between 0 and 1")
        
        if self.settings.risk.max_individual_weight <= 0 or self.settings.risk.max_individual_weight > 1:
            errors.append("max_individual_weight must be between 0 and 1")
        
        if self.settings.risk.var_confidence_level <= 0 or self.settings.risk.var_confidence_level >= 1:
            errors.append("var_confidence_level must be between 0 and 1")
        
        # Validate execution settings
        if self.settings.execution.default_slippage < 0:
            errors.append("default_slippage cannot be negative")
        
        if self.settings.execution.max_order_size <= 0:
            errors.append("max_order_size must be positive")
        
        # Validate data settings
        if self.settings.data.cache_ttl <= 0:
            warnings.append("cache_ttl should be positive for effective caching")
        
        if self.settings.data.max_retries <= 0:
            warnings.append("max_retries should be positive for reliability")
        
        # Validate logging settings
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.settings.logging.level not in valid_levels:
            errors.append(f"logging level must be one of {valid_levels}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = getattr(logging, self.settings.logging.level.upper())
        
        logging.basicConfig(
            level=log_level,
            format=self.settings.logging.format,
            handlers=[
                logging.FileHandler(self.settings.logging.file_path),
                logging.StreamHandler()
            ]
        )
        
        # Set specific logger levels
        if self.settings.logging.disable_third_party:
            logging.getLogger('urllib3').setLevel(logging.WARNING)
            logging.getLogger('requests').setLevel(logging.WARNING)
            logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    def get_environment_config(self) -> Dict[str, str]:
        """Get environment-specific configuration."""
        env_vars = {}
        
        # Database URL from environment
        if 'DATABASE_URL' in os.environ:
            env_vars['database_url'] = os.environ['DATABASE_URL']
        
        # API keys from environment
        api_keys = ['ALPHA_VANTAGE_API_KEY', 'QUANDL_API_KEY', 'IEX_API_KEY']
        for key in api_keys:
            if key in os.environ:
                env_vars[key.lower()] = os.environ[key]
        
        # Other environment variables
        env_mappings = {
            'INVESTMENT_ENGINE_ENV': 'environment',
            'INVESTMENT_ENGINE_DEBUG': 'debug',
            'INVESTMENT_ENGINE_LOG_LEVEL': 'log_level'
        }
        
        for env_key, config_key in env_mappings.items():
            if env_key in os.environ:
                env_vars[config_key] = os.environ[env_key]
        
        return env_vars
    
    def export_config(self, output_path: str, format: str = 'yaml') -> None:
        """Export configuration to file.
        
        Args:
            output_path: Output file path
            format: Export format ('yaml' or 'json')
        """
        config_data = asdict(self.settings)
        
        try:
            with open(output_path, 'w') as f:
                if format.lower() == 'yaml':
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_data, f, indent=2, default=str)
                    
            print(f"Configuration exported to {output_path}")
            
        except Exception as e:
            print(f"Error exporting configuration: {e}")
    
    def create_sample_config(self, output_path: str) -> None:
        """Create a sample configuration file with comments.
        
        Args:
            output_path: Output file path
        """
        sample_config = """
# Investment Engine Configuration File
# This file contains all configurable settings for the investment engine

# Database Configuration
database:
  url: "sqlite:///investment_engine.db"  # Database connection URL
  pool_size: 10                          # Connection pool size
  max_overflow: 20                       # Maximum overflow connections
  pool_timeout: 30                       # Pool timeout in seconds

# Data Provider Configuration
data:
  primary_provider: "yahoo"              # Primary data provider (yahoo, alpha_vantage, etc.)
  cache_enabled: true                    # Enable data caching
  cache_ttl: 3600                       # Cache time-to-live in seconds
  max_retries: 3                        # Maximum retry attempts
  timeout: 30                           # Request timeout in seconds

# Risk Management Configuration
risk:
  max_portfolio_var: 0.05               # Maximum portfolio VaR (5%)
  max_individual_weight: 0.1            # Maximum individual asset weight (10%)
  max_sector_weight: 0.3                # Maximum sector weight (30%)
  max_drawdown: 0.2                     # Maximum drawdown threshold (20%)
  var_confidence_level: 0.95            # VaR confidence level (95%)

# Execution Configuration
execution:
  default_slippage: 0.001               # Default slippage assumption (0.1%)
  max_order_size: 1000000               # Maximum order size in dollars
  execution_delay: 0.1                  # Simulated execution delay in seconds
  retry_attempts: 3                     # Number of retry attempts for failed orders

# Optimization Configuration
optimization:
  default_method: "mean_variance"       # Default optimization method
  max_iterations: 1000                  # Maximum optimization iterations
  tolerance: 1e-6                       # Convergence tolerance
  regularization: 0.01                  # Regularization parameter

# Backtesting Configuration
backtesting:
  initial_capital: 1000000              # Initial capital for backtesting
  transaction_cost_model: "proportional" # Transaction cost model
  benchmark_symbol: "SPY"               # Default benchmark symbol

# Logging Configuration
logging:
  level: "INFO"                         # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  file_path: "logs/investment_engine.log" # Log file path
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  disable_third_party: true             # Disable verbose third-party logging

# Dashboard Configuration
dashboard:
  host: "127.0.0.1"                    # Dashboard host
  port: 8050                           # Dashboard port
  debug: false                         # Enable debug mode
  auto_refresh_interval: 30            # Auto-refresh interval in seconds
"""
        
        try:
            with open(output_path, 'w') as f:
                f.write(sample_config)
            print(f"Sample configuration created at {output_path}")
        except Exception as e:
            print(f"Error creating sample configuration: {e}")


# Global configuration instance
config_manager = ConfigManager()
