"""
Automatic configuration generator - no manual setup required.
"""

import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime
import secrets
import logging

logger = logging.getLogger(__name__)

class AutoConfig:
    """Automatically generates all configuration without user input."""
    
    def __init__(self):
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
    def setup_everything(self):
        """One-click setup for everything."""
        print("[*] Starting automated setup...")
        
        # Generate config
        config = self.generate_config()
        
        # Save config
        self.save_config(config)
        
        # Setup database
        self.setup_database()
        
        # Create .env file
        self.create_env_file()
        
        print("[OK] Setup complete! No manual configuration needed.")
        return config
    
    def generate_config(self):
        """Generate complete configuration automatically."""
        config = {
            "app": {
                "name": "Investment Engine",
                "version": "1.0.0",
                "environment": "production",
                "debug": False,
                "secret_key": secrets.token_urlsafe(32)
            },
            "database": {
                "type": "sqlite",
                "path": str(self.data_dir / "investment_engine.db"),
                "backup_path": str(self.data_dir / "backup.db")
            },
            "cache": {
                "type": "memory",
                "max_size": 1000,
                "ttl": 3600
            },
            "market_data": {
                "primary_source": "yfinance",
                "fallback_sources": ["yahoo"],
                "update_interval": 60,
                "batch_size": 50,
                "retry_attempts": 3,
                "timeout": 30
            },
            "dashboard": {
                "host": "127.0.0.1",
                "port": 8050,
                "auto_open_browser": True,
                "refresh_interval": 5000
            },
            "security": {
                "enable_auth": False,  # Simplified for local use
                "session_timeout": 86400,
                "encryption_key": secrets.token_urlsafe(32)
            },
            "performance": {
                "max_workers": 4,
                "cache_enabled": True,
                "lazy_loading": True,
                "compression": True
            },
            "features": {
                "real_time_data": True,
                "backtesting": True,
                "paper_trading": True,
                "ai_assistant": True,
                "export_enabled": True
            },
            "paths": {
                "data": str(self.data_dir),
                "logs": str(self.logs_dir),
                "cache": str(self.data_dir / "cache"),
                "exports": str(self.data_dir / "exports")
            }
        }
        
        return config
    
    def save_config(self, config):
        """Save configuration to file."""
        config_file = self.config_dir / "auto_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[*] Configuration saved to {config_file}")
        
    def setup_database(self):
        """Setup SQLite database with tables."""
        db_path = self.data_dir / "investment_engine.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create tables
        tables = [
            """
            CREATE TABLE IF NOT EXISTS portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                config TEXT,
                performance TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT,
                parameters TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                performance_metrics TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER,
                symbol TEXT,
                action TEXT,
                quantity REAL,
                price REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS market_data_cache (
                symbol TEXT,
                date DATE,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        for table in tables:
            cursor.execute(table)
        
        conn.commit()
        conn.close()
        print(f"[*] Database setup complete at {db_path}")
        
    def create_env_file(self):
        """Create .env file with auto-generated values."""
        env_content = f"""# Auto-generated configuration
APP_ENV=production
SECRET_KEY={secrets.token_urlsafe(32)}
DATABASE_PATH=data/investment_engine.db
CACHE_TYPE=memory
DASHBOARD_PORT=8050
AUTO_SETUP=true
CREATED_AT={datetime.now().isoformat()}
"""
        
        env_file = Path(".env")
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"[*] Environment file created at {env_file}")


class SimplifiedDataFeed:
    """Simplified data feed using only yfinance (no API keys needed)."""
    
    def __init__(self):
        import yfinance as yf
        self.yf = yf
        self.cache = {}
        
    def get_quote(self, symbol):
        """Get quote using yfinance."""
        try:
            ticker = self.yf.Ticker(symbol)
            info = ticker.info
            return {
                'symbol': symbol,
                'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'change': info.get('regularMarketChange', 0),
                'change_percent': info.get('regularMarketChangePercent', 0),
                'volume': info.get('volume', 0),
                'timestamp': datetime.now().isoformat()
            }
        except:
            return {'symbol': symbol, 'price': 0, 'error': 'Data unavailable'}
    
    def get_history(self, symbol, period='1mo'):
        """Get historical data."""
        try:
            ticker = self.yf.Ticker(symbol)
            return ticker.history(period=period)
        except:
            return None


class InMemoryCache:
    """Simple in-memory cache to replace Redis."""
    
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
        
    def set(self, key, value, ttl=3600):
        """Set value with TTL."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            oldest = min(self.access_times.items(), key=lambda x: x[1])
            del self.cache[oldest[0]]
            del self.access_times[oldest[0]]
        
        self.cache[key] = {
            'value': value,
            'expires': datetime.now().timestamp() + ttl
        }
        self.access_times[key] = datetime.now().timestamp()
        
    def get(self, key):
        """Get value if not expired."""
        if key in self.cache:
            item = self.cache[key]
            if item['expires'] > datetime.now().timestamp():
                self.access_times[key] = datetime.now().timestamp()
                return item['value']
            else:
                del self.cache[key]
                del self.access_times[key]
        return None
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_times.clear()


# Auto-initialize on import
_auto_config = AutoConfig()
_cache = InMemoryCache()
_data_feed = SimplifiedDataFeed()

def get_config():
    """Get auto-generated configuration."""
    config_file = Path("config/auto_config.json")
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    else:
        return _auto_config.setup_everything()

def get_cache():
    """Get cache instance."""
    return _cache

def get_data_feed():
    """Get data feed instance."""
    return _data_feed
