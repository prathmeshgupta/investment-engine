"""Performance Optimization and Scalability Module."""

import asyncio
import functools
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import hashlib
import redis
import pymongo
from dataclasses import dataclass
import psutil
import numba
from numba import jit, prange
import cython
import dask.dataframe as dd
from joblib import Parallel, delayed, Memory
from functools import lru_cache, cached_property
import gc
import warnings

warnings.filterwarnings('ignore')


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds
    enable_parallel: bool = True
    max_workers: int = mp.cpu_count()
    chunk_size: int = 10000
    memory_limit: float = 0.8  # Use max 80% of available memory
    enable_gpu: bool = False
    enable_distributed: bool = False
    redis_host: str = 'localhost'
    redis_port: int = 6379
    mongodb_uri: str = 'mongodb://localhost:27017/'


class CacheManager:
    """Advanced caching system with multiple backends."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.memory_cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
        
        # Initialize Redis connection if available
        self.redis_client = None
        if config.enable_caching:
            try:
                self.redis_client = redis.Redis(
                    host=config.redis_host,
                    port=config.redis_port,
                    decode_responses=True
                )
                self.redis_client.ping()
            except:
                self.redis_client = None
        
        # Initialize joblib memory for disk caching
        self.disk_cache = Memory('cache', verbose=0)
    
    def _generate_key(self, func_name: str, *args, **kwargs) -> str:
        """Generate cache key from function and arguments."""
        key_data = f"{func_name}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        # Try memory cache first
        if key in self.memory_cache:
            self.cache_stats['hits'] += 1
            return self.memory_cache[key]['value']
        
        # Try Redis if available
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    self.cache_stats['hits'] += 1
                    return pickle.loads(value.encode('latin-1'))
            except:
                pass
        
        self.cache_stats['misses'] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        ttl = ttl or self.config.cache_ttl
        
        # Store in memory cache
        self.memory_cache[key] = {
            'value': value,
            'timestamp': time.time(),
            'ttl': ttl
        }
        
        # Store in Redis if available
        if self.redis_client:
            try:
                self.redis_client.setex(
                    key, ttl, 
                    pickle.dumps(value).decode('latin-1')
                )
            except:
                pass
    
    def clear_expired(self):
        """Clear expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, data in self.memory_cache.items():
            if current_time - data['timestamp'] > data['ttl']:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]


def cached_result(ttl: int = 3600):
    """Decorator for caching function results."""
    def decorator(func):
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = hashlib.md5(
                f"{func.__name__}:{args}:{kwargs}".encode()
            ).hexdigest()
            
            # Check cache
            if key in cache:
                value, timestamp = cache[key]
                if time.time() - timestamp < ttl:
                    return value
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            cache[key] = (result, time.time())
            
            return result
        
        return wrapper
    
    return decorator


class ParallelProcessor:
    """Parallel processing utilities for performance optimization."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=config.max_workers)
    
    async def parallel_map_async(self, func: Callable, items: List[Any]) -> List[Any]:
        """Asynchronously map function over items in parallel."""
        loop = asyncio.get_event_loop()
        
        # Use thread pool for I/O bound tasks
        futures = [
            loop.run_in_executor(self.thread_pool, func, item)
            for item in items
        ]
        
        return await asyncio.gather(*futures)
    
    def parallel_map(self, func: Callable, items: List[Any], 
                    use_processes: bool = False) -> List[Any]:
        """Map function over items in parallel."""
        if use_processes:
            with self.process_pool as pool:
                return list(pool.map(func, items))
        else:
            with self.thread_pool as pool:
                return list(pool.map(func, items))
    
    def parallel_dataframe_apply(self, df: pd.DataFrame, 
                                func: Callable, axis: int = 0) -> pd.DataFrame:
        """Apply function to dataframe in parallel."""
        if len(df) < self.config.chunk_size:
            return df.apply(func, axis=axis)
        
        # Split dataframe into chunks
        n_chunks = min(self.config.max_workers, len(df) // self.config.chunk_size + 1)
        chunks = np.array_split(df, n_chunks)
        
        # Process chunks in parallel
        results = Parallel(n_jobs=self.config.max_workers)(
            delayed(lambda x: x.apply(func, axis=axis))(chunk) 
            for chunk in chunks
        )
        
        return pd.concat(results)


class MemoryOptimizer:
    """Memory optimization utilities."""
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize dataframe memory usage."""
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype('category')
        
        return df
    
    @staticmethod
    def monitor_memory():
        """Monitor current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'percent': process.memory_percent(),
            'available': psutil.virtual_memory().available / 1024 / 1024 / 1024  # GB
        }
    
    @staticmethod
    def garbage_collect():
        """Force garbage collection."""
        gc.collect()


@jit(nopython=True, parallel=True)
def fast_portfolio_optimization(returns: np.ndarray, 
                               risk_free_rate: float = 0.02) -> np.ndarray:
    """Fast portfolio optimization using Numba JIT compilation."""
    n_assets = returns.shape[1]
    mean_returns = np.mean(returns, axis=0)
    cov_matrix = np.cov(returns.T)
    
    # Generate random weights
    n_portfolios = 10000
    weights = np.random.random((n_portfolios, n_assets))
    weights = weights / np.sum(weights, axis=1).reshape(-1, 1)
    
    # Calculate portfolio metrics
    portfolio_returns = np.zeros(n_portfolios)
    portfolio_volatility = np.zeros(n_portfolios)
    
    for i in prange(n_portfolios):
        w = weights[i]
        portfolio_returns[i] = np.dot(w, mean_returns)
        portfolio_volatility[i] = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
    
    # Calculate Sharpe ratio
    sharpe_ratio = (portfolio_returns - risk_free_rate) / portfolio_volatility
    
    # Find optimal portfolio
    optimal_idx = np.argmax(sharpe_ratio)
    
    return weights[optimal_idx]


class DatabaseOptimizer:
    """Database query and connection optimization."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.connection_pool = {}
        
        # MongoDB connection pool
        if config.enable_distributed:
            self.mongo_client = pymongo.MongoClient(
                config.mongodb_uri,
                maxPoolSize=50,
                minPoolSize=10
            )
    
    def batch_insert(self, data: List[Dict], collection_name: str, 
                    batch_size: int = 1000):
        """Batch insert data into database."""
        if not self.config.enable_distributed:
            return
        
        db = self.mongo_client['investment_engine']
        collection = db[collection_name]
        
        # Insert in batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            collection.insert_many(batch)
    
    def create_indexes(self, collection_name: str, indexes: List[str]):
        """Create database indexes for faster queries."""
        if not self.config.enable_distributed:
            return
        
        db = self.mongo_client['investment_engine']
        collection = db[collection_name]
        
        for index in indexes:
            collection.create_index(index)


class QueryOptimizer:
    """SQL and data query optimization."""
    
    @staticmethod
    def optimize_pandas_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Optimize pandas query execution."""
        # Use query method for complex conditions
        if 'and' in query or 'or' in query:
            return df.query(query)
        
        # Use boolean indexing for simple conditions
        return df[eval(f"df.{query}")]
    
    @staticmethod
    @lru_cache(maxsize=128)
    def cached_aggregation(df_hash: int, group_by: tuple, 
                          agg_func: str) -> pd.DataFrame:
        """Cached aggregation results."""
        # This is a placeholder - actual implementation would reconstruct df from hash
        pass


class LazyLoader:
    """Lazy loading for large datasets."""
    
    def __init__(self, file_path: str, chunk_size: int = 10000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self._data = None
    
    @property
    def data(self):
        """Load data on first access."""
        if self._data is None:
            self._data = pd.read_csv(self.file_path)
        return self._data
    
    def iter_chunks(self):
        """Iterate over data in chunks."""
        return pd.read_csv(self.file_path, chunksize=self.chunk_size)


class PerformanceMonitor:
    """Monitor and profile application performance."""
    
    def __init__(self):
        self.metrics = []
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function execution."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            self.metrics.append({
                'function': func.__name__,
                'execution_time': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'timestamp': datetime.now()
            })
            
            return result
        
        return wrapper
    
    def get_performance_report(self) -> pd.DataFrame:
        """Generate performance report."""
        if not self.metrics:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.metrics)
        
        # Aggregate by function
        report = df.groupby('function').agg({
            'execution_time': ['mean', 'max', 'min', 'count'],
            'memory_delta': ['mean', 'max']
        }).round(3)
        
        return report


class PerformanceOptimizer:
    """Wrapper to orchestrate performance optimizations for the dashboard.

    Exposes a simple optimize_execution() method used by the UI callbacks.
    Internally leverages fast portfolio optimization and shared utilities.
    """

    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self.cache = CacheManager(self.config)
        self.parallel = ParallelProcessor(self.config)
        self.monitor = PerformanceMonitor()

    def optimize_execution(self) -> Dict[str, Any]:
        """Run a quick optimization routine and return a summary dict.

        - Builds a small synthetic returns matrix
        - Runs fast_portfolio_optimization (Numba-accelerated)
        - Returns the resulting weights and basic metadata
        """
        # Profile this method for visibility in performance report
        @self.monitor.profile_function
        def _run() -> Dict[str, Any]:
            np.random.seed(42)
            n_days = 252
            n_assets = 5
            returns = np.random.normal(0.001, 0.02, size=(n_days, n_assets))

            weights = fast_portfolio_optimization(returns)
            summary = {
                'optimal_weights': weights.tolist(),
                'assets': n_assets,
                'days': n_days,
                'timestamp': datetime.now().isoformat()
            }
            return summary

        result = _run()
        return result


# Singleton instance
_performance_config = PerformanceConfig()
_cache_manager = CacheManager(_performance_config)
_parallel_processor = ParallelProcessor(_performance_config)
_performance_monitor = PerformanceMonitor()


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    return _cache_manager


def get_parallel_processor() -> ParallelProcessor:
    """Get global parallel processor instance."""
    return _parallel_processor


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    return _performance_monitor
