"""Enumerations for the investment engine."""

from enum import Enum, auto


class AssetClass(Enum):
    """Asset class categories."""
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    REAL_ESTATE = "real_estate"
    CURRENCY = "currency"
    ALTERNATIVE = "alternative"
    CASH = "cash"


class FactorType(Enum):
    """Factor model types."""
    MARKET = "market"
    SIZE = "size"
    VALUE = "value"
    PROFITABILITY = "profitability"
    INVESTMENT = "investment"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    VOLATILITY = "volatility"
    CARRY = "carry"
    MEAN_REVERSION = "mean_reversion"


class OptimizationMethod(Enum):
    """Portfolio optimization methods."""
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_DIVERSIFICATION = "maximum_diversification"
    EQUAL_WEIGHT = "equal_weight"


class RiskMeasure(Enum):
    """Risk measurement methods."""
    VARIANCE = "variance"
    VALUE_AT_RISK = "value_at_risk"
    CONDITIONAL_VAR = "conditional_var"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    SEMI_VARIANCE = "semi_variance"
    TRACKING_ERROR = "tracking_error"


class RebalanceFrequency(Enum):
    """Portfolio rebalancing frequency."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUALLY = "semi_annually"
    ANNUALLY = "annually"


class OrderType(Enum):
    """Order types for execution."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order execution status."""
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
