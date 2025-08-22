"""
Production configuration settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ProductionConfig:
    """Production environment configuration."""
    
    # Base settings
    ENV = 'production'
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.getenv('SECRET_KEY', 'change-this-in-production')
    
    # Database
    DATABASE_CONFIG = {
        'url': os.getenv('DATABASE_URL', 'postgresql://localhost/investment_engine'),
        'pool_size': 20,
        'max_overflow': 40,
        'pool_timeout': 30,
        'pool_recycle': 3600,
        'echo': False
    }
    
    # Redis Cache
    REDIS_CONFIG = {
        'url': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
        'decode_responses': True,
        'max_connections': 50,
        'socket_connect_timeout': 5,
        'socket_keepalive': True,
        'socket_keepalive_options': {}
    }
    
    # API Keys
    MARKET_DATA_APIS = {
        'alpha_vantage': {
            'api_key': os.getenv('ALPHA_VANTAGE_API_KEY'),
            'rate_limit': 5,  # requests per minute
            'timeout': 30
        },
        'finnhub': {
            'api_key': os.getenv('FINNHUB_API_KEY'),
            'rate_limit': 60,
            'timeout': 30
        },
        'iex_cloud': {
            'api_key': os.getenv('IEX_CLOUD_API_KEY'),
            'rate_limit': 100,
            'timeout': 30
        },
        'polygon': {
            'api_key': os.getenv('POLYGON_API_KEY'),
            'rate_limit': 5,
            'timeout': 30
        }
    }
    
    # Broker Configuration
    BROKER_CONFIG = {
        'alpaca': {
            'api_key': os.getenv('ALPACA_API_KEY'),
            'secret_key': os.getenv('ALPACA_SECRET_KEY'),
            'base_url': os.getenv('ALPACA_BASE_URL', 'https://api.alpaca.markets'),
            'paper_trading': False
        }
    }
    
    # WebSocket Settings
    WEBSOCKET_CONFIG = {
        'host': os.getenv('WS_HOST', '0.0.0.0'),
        'port': int(os.getenv('WS_PORT', 8765)),
        'ping_interval': 20,
        'ping_timeout': 10,
        'max_connections': 1000,
        'max_message_size': 1024 * 1024  # 1MB
    }
    
    # Dashboard Settings
    DASHBOARD_CONFIG = {
        'host': os.getenv('DASH_HOST', '0.0.0.0'),
        'port': int(os.getenv('DASH_PORT', 8050)),
        'debug': False,
        'dev_tools_ui': False,
        'dev_tools_props_check': False,
        'compress': True,
        'serve_locally': True
    }
    
    # Security
    SECURITY_CONFIG = {
        'jwt_secret_key': os.getenv('JWT_SECRET_KEY'),
        'jwt_algorithm': os.getenv('JWT_ALGORITHM', 'HS256'),
        'jwt_expiration_hours': int(os.getenv('JWT_EXPIRATION_HOURS', 24)),
        'encryption_key': os.getenv('ENCRYPTION_KEY'),
        'ssl_enabled': True,
        'cors_origins': ['https://yourdomain.com'],
        'rate_limit': '100/hour',
        'max_request_size': 10 * 1024 * 1024  # 10MB
    }
    
    # Logging
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'default',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'detailed',
                'filename': 'logs/app.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 10
            },
            'error_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filename': 'logs/error.log',
                'maxBytes': 10485760,
                'backupCount': 10
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console', 'file', 'error_file']
        }
    }
    
    # Monitoring
    MONITORING_CONFIG = {
        'sentry': {
            'dsn': os.getenv('SENTRY_DSN'),
            'traces_sample_rate': 0.1,
            'profiles_sample_rate': 0.1
        },
        'datadog': {
            'api_key': os.getenv('DATADOG_API_KEY'),
            'app_key': os.getenv('DATADOG_APP_KEY')
        },
        'new_relic': {
            'license_key': os.getenv('NEW_RELIC_LICENSE_KEY'),
            'app_name': 'Investment Engine'
        }
    }
    
    # Cloud Storage
    AWS_CONFIG = {
        'access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
        'secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
        'region': os.getenv('AWS_REGION', 'us-east-1'),
        's3_bucket': os.getenv('AWS_S3_BUCKET', 'investment-engine-data'),
        'cloudwatch_enabled': True
    }
    
    # Email Configuration
    EMAIL_CONFIG = {
        'smtp_host': os.getenv('SMTP_HOST', 'smtp.gmail.com'),
        'smtp_port': int(os.getenv('SMTP_PORT', 587)),
        'smtp_username': os.getenv('SMTP_USERNAME'),
        'smtp_password': os.getenv('SMTP_PASSWORD'),
        'use_tls': True,
        'default_sender': os.getenv('SMTP_USERNAME'),
        'alert_recipients': os.getenv('ALERT_EMAIL_TO', '').split(',')
    }
    
    # Performance Settings
    PERFORMANCE_CONFIG = {
        'worker_processes': 4,
        'worker_connections': 1000,
        'keepalive_timeout': 65,
        'request_timeout': 60,
        'max_request_line': 4094,
        'max_request_fields': 100,
        'max_request_field_size': 8190
    }


# Development configuration
class DevelopmentConfig(ProductionConfig):
    """Development environment configuration."""
    
    ENV = 'development'
    DEBUG = True
    
    DATABASE_CONFIG = {
        'url': 'sqlite:///investment_engine.db',
        'echo': True
    }
    
    REDIS_CONFIG = {
        'url': 'redis://localhost:6379/0',
        'decode_responses': True
    }
    
    DASHBOARD_CONFIG = {
        'host': '127.0.0.1',
        'port': 8050,
        'debug': True,
        'dev_tools_ui': True,
        'dev_tools_props_check': True
    }
    
    SECURITY_CONFIG = {
        **ProductionConfig.SECURITY_CONFIG,
        'ssl_enabled': False,
        'cors_origins': ['*']
    }


# Testing configuration
class TestingConfig(ProductionConfig):
    """Testing environment configuration."""
    
    ENV = 'testing'
    TESTING = True
    DEBUG = True
    
    DATABASE_CONFIG = {
        'url': 'sqlite:///:memory:',
        'echo': False
    }
    
    REDIS_CONFIG = {
        'url': 'redis://localhost:6379/1',
        'decode_responses': True
    }


# Configuration factory
def get_config(env=None):
    """Get configuration based on environment."""
    if env is None:
        env = os.getenv('APP_ENV', 'development')
    
    configs = {
        'production': ProductionConfig,
        'development': DevelopmentConfig,
        'testing': TestingConfig
    }
    
    return configs.get(env, DevelopmentConfig)
