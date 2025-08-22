# Multi-Factor Multi-Asset Investment Strategy Engine

A comprehensive investment management system that implements multi-factor models for portfolio optimization, risk management, and strategy execution across multiple asset classes.

## Features

- **Multi-Factor Models**: Fama-French 3/5-factor, momentum, quality, volatility factors
- **Multi-Asset Support**: Equities, bonds, commodities, REITs, currencies
- **Portfolio Optimization**: Mean-variance, risk parity, Black-Litterman models
- **Risk Management**: VaR, CVaR, drawdown controls, position sizing
- **Backtesting Engine**: Historical performance analysis with transaction costs
- **Real-time Execution**: Automated rebalancing and order management
- **Performance Analytics**: Comprehensive reporting and visualization

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main.py
```

3. Access the dashboard at `http://localhost:8000`

## Architecture

- `core/` - Core data models and business logic
- `factors/` - Multi-factor model implementations
- `optimization/` - Portfolio optimization algorithms
- `risk/` - Risk management and position sizing
- `backtesting/` - Historical testing framework
- `execution/` - Strategy execution and rebalancing
- `data/` - Market data integration
- `api/` - REST API endpoints
- `dashboard/` - Web-based visualization interface

## Configuration

Edit `config/settings.yaml` to customize:
- Asset universes and factor models
- Risk constraints and position limits
- Rebalancing frequency and execution parameters
- Data sources and API keys
