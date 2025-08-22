"""Core data models for the investment engine."""

from datetime import datetime, date
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
import numpy as np
import pandas as pd

from .enums import AssetClass, FactorType, OptimizationMethod, RiskMeasure, RebalanceFrequency, OrderType, OrderStatus


@dataclass
class Asset:
    """Represents a financial asset."""
    symbol: str
    name: str
    asset_class: AssetClass
    currency: str = "USD"
    exchange: Optional[str] = None
    sector: Optional[str] = None
    country: Optional[str] = None
    market_cap: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.symbol = self.symbol.upper()


@dataclass
class Position:
    """Represents a position in an asset."""
    asset: Asset
    quantity: Decimal
    market_value: Decimal
    cost_basis: Decimal
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def weight(self) -> float:
        """Position weight (to be calculated by portfolio)."""
        return 0.0
    
    @property
    def unrealized_pnl(self) -> Decimal:
        """Unrealized profit/loss."""
        return self.market_value - self.cost_basis


@dataclass
class Transaction:
    """Represents a transaction."""
    asset: Asset
    quantity: Decimal
    price: Decimal
    transaction_type: str  # 'buy' or 'sell'
    timestamp: datetime = field(default_factory=datetime.now)
    fees: Decimal = Decimal('0')
    order_id: Optional[str] = None
    
    @property
    def total_value(self) -> Decimal:
        """Total transaction value including fees."""
        return abs(self.quantity * self.price) + self.fees


class Portfolio(BaseModel):
    """Portfolio model with positions and performance tracking."""
    
    name: str
    positions: Dict[str, Position] = Field(default_factory=dict)
    cash: Decimal = Field(default=Decimal('0'))
    base_currency: str = "USD"
    inception_date: date = Field(default_factory=date.today)
    benchmark: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def total_value(self) -> Decimal:
        """Total portfolio value."""
        return sum(pos.market_value for pos in self.positions.values()) + self.cash
    
    @property
    def weights(self) -> Dict[str, float]:
        """Asset weights in the portfolio."""
        total = float(self.total_value)
        if total == 0:
            return {}
        return {
            symbol: float(pos.market_value) / total 
            for symbol, pos in self.positions.items()
        }
    
    def add_position(self, position: Position) -> None:
        """Add or update a position."""
        self.positions[position.asset.symbol] = position
    
    def remove_position(self, symbol: str) -> None:
        """Remove a position."""
        self.positions.pop(symbol, None)
    
    def get_asset_classes(self) -> Dict[AssetClass, float]:
        """Get allocation by asset class."""
        allocation = {}
        total = float(self.total_value)
        
        for pos in self.positions.values():
            asset_class = pos.asset.asset_class
            weight = float(pos.market_value) / total if total > 0 else 0
            allocation[asset_class] = allocation.get(asset_class, 0) + weight
            
        return allocation


class FactorModel(BaseModel):
    """Multi-factor model specification."""
    
    name: str
    factors: List[FactorType]
    lookback_period: int = 252  # trading days
    min_observations: int = 60
    
    def calculate_exposures(self, returns: pd.DataFrame, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate factor exposures for assets."""
        # This would implement factor regression
        # Placeholder for now
        return pd.DataFrame()


class Strategy(BaseModel):
    """Investment strategy configuration."""
    
    name: str
    description: str
    universe: List[str]  # Asset symbols
    factor_model: Optional[FactorModel] = None
    optimization_method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE
    risk_measure: RiskMeasure = RiskMeasure.VARIANCE
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    
    # Constraints
    max_weight: float = 0.1
    min_weight: float = 0.0
    max_turnover: Optional[float] = None
    max_concentration: Optional[float] = None
    
    # Risk parameters
    target_volatility: Optional[float] = None
    max_drawdown: Optional[float] = None
    var_confidence: float = 0.05
    
    # Transaction costs
    transaction_cost: float = 0.001  # 10 bps
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class Order(BaseModel):
    """Order for execution."""
    
    asset: Asset
    quantity: Decimal
    order_type: OrderType
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    filled_quantity: Decimal = Decimal('0')
    filled_price: Optional[Decimal] = None
    
    class Config:
        arbitrary_types_allowed = True


class PerformanceMetrics(BaseModel):
    """Portfolio performance metrics."""
    
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    sortino_ratio: float
    var_95: float
    cvar_95: float
    
    # Relative metrics (vs benchmark)
    alpha: Optional[float] = None
    beta: Optional[float] = None
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None
    
    start_date: date
    end_date: date
    
    @validator('sharpe_ratio', 'calmar_ratio', 'sortino_ratio')
    def validate_ratios(cls, v):
        """Validate ratio values."""
        if v is not None and (np.isnan(v) or np.isinf(v)):
            return 0.0
        return v
