"""Data caching implementation for performance optimization."""

import pandas as pd
import pickle
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import os
import tempfile


class DataCache:
    """In-memory and disk-based data cache."""
    
    def __init__(self, 
                 ttl: int = 3600,
                 max_memory_items: int = 100,
                 use_disk_cache: bool = True):
        """Initialize data cache.
        
        Args:
            ttl: Time-to-live in seconds
            max_memory_items: Maximum items in memory cache
            use_disk_cache: Whether to use disk-based cache
        """
        self.ttl = ttl
        self.max_memory_items = max_memory_items
        self.use_disk_cache = use_disk_cache
        
        # Memory cache
        self.memory_cache = {}
        self.access_times = {}
        
        # Disk cache directory
        if use_disk_cache:
            self.cache_dir = os.path.join(tempfile.gettempdir(), 'investment_engine_cache')
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'disk_reads': 0,
            'disk_writes': 0
        }
    
    def _generate_key_hash(self, key: str) -> str:
        """Generate hash for cache key."""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _is_expired(self, timestamp: datetime) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() - timestamp > timedelta(seconds=self.ttl)
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached item or None if not found/expired
        """
        # Check memory cache first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if not self._is_expired(entry['timestamp']):
                self.access_times[key] = datetime.now()
                self.stats['hits'] += 1
                return entry['data']
            else:
                # Remove expired entry
                del self.memory_cache[key]
                if key in self.access_times:
                    del self.access_times[key]
        
        # Check disk cache
        if self.use_disk_cache:
            disk_data = self._get_from_disk(key)
            if disk_data is not None:
                # Move to memory cache
                self._set_memory(key, disk_data)
                self.stats['hits'] += 1
                self.stats['disk_reads'] += 1
                return disk_data
        
        self.stats['misses'] += 1
        return None
    
    def set(self, key: str, data: Any) -> None:
        """Set item in cache.
        
        Args:
            key: Cache key
            data: Data to cache
        """
        # Set in memory cache
        self._set_memory(key, data)
        
        # Set in disk cache
        if self.use_disk_cache:
            self._set_disk(key, data)
    
    def _set_memory(self, key: str, data: Any) -> None:
        """Set item in memory cache."""
        # Check if we need to evict items
        if len(self.memory_cache) >= self.max_memory_items:
            self._evict_lru()
        
        self.memory_cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }
        self.access_times[key] = datetime.now()
    
    def _set_disk(self, key: str, data: Any) -> None:
        """Set item in disk cache."""
        try:
            key_hash = self._generate_key_hash(key)
            file_path = os.path.join(self.cache_dir, f"{key_hash}.pkl")
            
            cache_entry = {
                'data': data,
                'timestamp': datetime.now(),
                'key': key
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(cache_entry, f)
            
            self.stats['disk_writes'] += 1
            
        except Exception as e:
            print(f"Error writing to disk cache: {e}")
    
    def _get_from_disk(self, key: str) -> Optional[Any]:
        """Get item from disk cache."""
        try:
            key_hash = self._generate_key_hash(key)
            file_path = os.path.join(self.cache_dir, f"{key_hash}.pkl")
            
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, 'rb') as f:
                cache_entry = pickle.load(f)
            
            # Check if expired
            if self._is_expired(cache_entry['timestamp']):
                os.remove(file_path)
                return None
            
            return cache_entry['data']
            
        except Exception as e:
            print(f"Error reading from disk cache: {e}")
            return None
    
    def _evict_lru(self) -> None:
        """Evict least recently used item from memory cache."""
        if not self.access_times:
            return
        
        # Find least recently used key
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Remove from caches
        if lru_key in self.memory_cache:
            del self.memory_cache[lru_key]
        if lru_key in self.access_times:
            del self.access_times[lru_key]
        
        self.stats['evictions'] += 1
    
    def clear(self) -> None:
        """Clear all cache data."""
        # Clear memory cache
        self.memory_cache.clear()
        self.access_times.clear()
        
        # Clear disk cache
        if self.use_disk_cache and os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    try:
                        os.remove(os.path.join(self.cache_dir, filename))
                    except Exception:
                        pass
    
    def cleanup_expired(self) -> int:
        """Clean up expired cache entries.
        
        Returns:
            Number of expired entries removed
        """
        removed_count = 0
        
        # Clean memory cache
        expired_keys = []
        for key, entry in self.memory_cache.items():
            if self._is_expired(entry['timestamp']):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
            if key in self.access_times:
                del self.access_times[key]
            removed_count += 1
        
        # Clean disk cache
        if self.use_disk_cache and os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    file_path = os.path.join(self.cache_dir, filename)
                    try:
                        with open(file_path, 'rb') as f:
                            cache_entry = pickle.load(f)
                        
                        if self._is_expired(cache_entry['timestamp']):
                            os.remove(file_path)
                            removed_count += 1
                    
                    except Exception:
                        # Remove corrupted files
                        try:
                            os.remove(file_path)
                            removed_count += 1
                        except Exception:
                            pass
        
        return removed_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        hit_rate = self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) if (self.stats['hits'] + self.stats['misses']) > 0 else 0
        
        return {
            'memory_items': len(self.memory_cache),
            'max_memory_items': self.max_memory_items,
            'ttl_seconds': self.ttl,
            'hit_rate': hit_rate,
            'total_hits': self.stats['hits'],
            'total_misses': self.stats['misses'],
            'evictions': self.stats['evictions'],
            'disk_reads': self.stats['disk_reads'],
            'disk_writes': self.stats['disk_writes'],
            'disk_cache_enabled': self.use_disk_cache
        }
    
    def get_size_info(self) -> Dict[str, Any]:
        """Get cache size information.
        
        Returns:
            Dictionary with size information
        """
        memory_size = 0
        disk_size = 0
        
        # Estimate memory size
        try:
            memory_size = sum(len(pickle.dumps(entry)) for entry in self.memory_cache.values())
        except Exception:
            memory_size = 0
        
        # Calculate disk size
        if self.use_disk_cache and os.path.exists(self.cache_dir):
            try:
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.pkl'):
                        file_path = os.path.join(self.cache_dir, filename)
                        disk_size += os.path.getsize(file_path)
            except Exception:
                disk_size = 0
        
        return {
            'memory_size_bytes': memory_size,
            'disk_size_bytes': disk_size,
            'total_size_bytes': memory_size + disk_size,
            'memory_size_mb': memory_size / (1024 * 1024),
            'disk_size_mb': disk_size / (1024 * 1024)
        }
