"""Settings data classes for configuration management."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DatabaseSettings:
    """Database configuration settings."""
    url: str = "sqlite:///investment_engine.db"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30


@dataclass
class DataSettings:
    """Data provider configuration settings."""
    primary_provider: str = "yahoo"
    cache_enabled: bool = True
    cache_ttl: int = 3600
    max_retries: int = 3
    timeout: int = 30


@dataclass
class RiskSettings:
    """Risk management configuration settings."""
    max_portfolio_var: float = 0.05
    max_individual_weight: float = 0.1
    max_sector_weight: float = 0.3
    max_drawdown: float = 0.2
    var_confidence_level: float = 0.95


@dataclass
class ExecutionSettings:
    """Execution configuration settings."""
    default_slippage: float = 0.001
    max_order_size: float = 1000000
    execution_delay: float = 0.1
    retry_attempts: int = 3


@dataclass
class OptimizationSettings:
    """Optimization configuration settings."""
    default_method: str = "mean_variance"
    max_iterations: int = 1000
    tolerance: float = 1e-6
    regularization: float = 0.01


@dataclass
class BacktestingSettings:
    """Backtesting configuration settings."""
    initial_capital: float = 1000000
    transaction_cost_model: str = "proportional"
    benchmark_symbol: str = "SPY"


@dataclass
class LoggingSettings:
    """Logging configuration settings."""
    level: str = "INFO"
    file_path: str = "logs/investment_engine.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    disable_third_party: bool = True


@dataclass
class DashboardSettings:
    """Dashboard configuration settings."""
    host: str = "127.0.0.1"
    port: int = 8050
    debug: bool = False
    auto_refresh_interval: int = 30


@dataclass
class Settings:
    """Main settings container."""
    database: DatabaseSettings = None
    data: DataSettings = None
    risk: RiskSettings = None
    execution: ExecutionSettings = None
    optimization: OptimizationSettings = None
    backtesting: BacktestingSettings = None
    logging: LoggingSettings = None
    dashboard: DashboardSettings = None
    
    def __post_init__(self):
        """Initialize nested settings objects."""
        if self.database is None:
            self.database = DatabaseSettings()
        if self.data is None:
            self.data = DataSettings()
        if self.risk is None:
            self.risk = RiskSettings()
        if self.execution is None:
            self.execution = ExecutionSettings()
        if self.optimization is None:
            self.optimization = OptimizationSettings()
        if self.backtesting is None:
            self.backtesting = BacktestingSettings()
        if self.logging is None:
            self.logging = LoggingSettings()
        if self.dashboard is None:
            self.dashboard = DashboardSettings()
