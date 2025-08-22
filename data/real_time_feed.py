"""
Real-time market data feed manager with WebSocket support.
"""

import asyncio
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import yfinance as yf
import pandas as pd
import numpy as np
from collections import deque
import websocket
import requests
import redis
import pickle
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketDataFeed:
    """Manages real-time market data feeds from multiple sources."""
    
    def __init__(self, cache_enabled=True):
        self.subscribers = {}
        self.data_cache = {}
        self.price_history = {}
        self.cache_enabled = cache_enabled
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Initialize Redis cache if enabled
        if cache_enabled:
            try:
                self.redis_client = redis.Redis(
                    host='localhost',
                    port=6379,
                    decode_responses=False,
                    socket_connect_timeout=5
                )
                self.redis_client.ping()
                logger.info("Redis cache connected successfully")
            except:
                logger.warning("Redis not available, using in-memory cache")
                self.redis_client = None
        else:
            self.redis_client = None
        
        # WebSocket connections
        self.ws_connections = {}
        self.running = False
        
    def subscribe(self, symbol: str, callback: Callable):
        """Subscribe to real-time updates for a symbol."""
        if symbol not in self.subscribers:
            self.subscribers[symbol] = []
        self.subscribers[symbol].append(callback)
        
        # Initialize price history
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=1000)
        
        logger.info(f"Subscribed to {symbol}")
        
    def unsubscribe(self, symbol: str, callback: Callable):
        """Unsubscribe from symbol updates."""
        if symbol in self.subscribers:
            self.subscribers[symbol].remove(callback)
            if not self.subscribers[symbol]:
                del self.subscribers[symbol]
                
    def get_realtime_quote(self, symbol: str) -> Dict:
        """Get real-time quote for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            quote = {
                'symbol': symbol,
                'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'bid': info.get('bid', 0),
                'ask': info.get('ask', 0),
                'volume': info.get('volume', 0),
                'timestamp': datetime.now().isoformat(),
                'change': info.get('regularMarketChange', 0),
                'change_percent': info.get('regularMarketChangePercent', 0),
                'open': info.get('regularMarketOpen', 0),
                'high': info.get('regularMarketDayHigh', 0),
                'low': info.get('regularMarketDayLow', 0),
                'previous_close': info.get('regularMarketPreviousClose', 0)
            }
            
            # Cache the quote
            self._cache_data(f"quote:{symbol}", quote, expire=60)
            
            # Update price history
            if symbol in self.price_history:
                self.price_history[symbol].append({
                    'price': quote['price'],
                    'timestamp': quote['timestamp']
                })
            
            return quote
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return self._get_cached_quote(symbol)
    
    def get_batch_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get quotes for multiple symbols efficiently."""
        quotes = {}
        
        # Use thread pool for parallel fetching
        futures = []
        for symbol in symbols:
            future = self.executor.submit(self.get_realtime_quote, symbol)
            futures.append((symbol, future))
        
        for symbol, future in futures:
            try:
                quotes[symbol] = future.result(timeout=5)
            except Exception as e:
                logger.error(f"Error getting quote for {symbol}: {e}")
                quotes[symbol] = self._get_cached_quote(symbol)
        
        return quotes
    
    def get_historical_data(self, symbol: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
        """Get historical data for a symbol."""
        cache_key = f"history:{symbol}:{period}:{interval}"
        
        # Check cache first
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            # Cache the data
            self._cache_data(cache_key, df, expire=3600)  # Cache for 1 hour
            
            return df
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def start_websocket_feed(self, symbols: List[str], source: str = "finnhub"):
        """Start WebSocket feed for real-time data."""
        self.running = True
        
        if source == "finnhub":
            threading.Thread(target=self._finnhub_websocket, args=(symbols,), daemon=True).start()
        elif source == "alpaca":
            threading.Thread(target=self._alpaca_websocket, args=(symbols,), daemon=True).start()
        else:
            logger.warning(f"Unknown WebSocket source: {source}")
    
    def _finnhub_websocket(self, symbols: List[str]):
        """Connect to Finnhub WebSocket for real-time data."""
        # Note: Requires Finnhub API key
        api_key = "YOUR_FINNHUB_API_KEY"  # Replace with actual key
        
        def on_message(ws, message):
            data = json.loads(message)
            if data['type'] == 'trade':
                for trade in data['data']:
                    symbol = trade['s']
                    if symbol in self.subscribers:
                        quote = {
                            'symbol': symbol,
                            'price': trade['p'],
                            'volume': trade['v'],
                            'timestamp': datetime.fromtimestamp(trade['t']/1000).isoformat()
                        }
                        self._notify_subscribers(symbol, quote)
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
        
        def on_close(ws):
            logger.info("WebSocket connection closed")
        
        def on_open(ws):
            for symbol in symbols:
                ws.send(json.dumps({'type': 'subscribe', 'symbol': symbol}))
        
        ws_url = f"wss://ws.finnhub.io?token={api_key}"
        ws = websocket.WebSocketApp(ws_url,
                                    on_message=on_message,
                                    on_error=on_error,
                                    on_close=on_close,
                                    on_open=on_open)
        
        ws.run_forever()
    
    def _alpaca_websocket(self, symbols: List[str]):
        """Connect to Alpaca WebSocket for real-time data."""
        # Note: Requires Alpaca API credentials
        pass  # Implementation would go here
    
    def _notify_subscribers(self, symbol: str, data: Dict):
        """Notify all subscribers of new data."""
        if symbol in self.subscribers:
            for callback in self.subscribers[symbol]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in subscriber callback: {e}")
    
    def _cache_data(self, key: str, data: Any, expire: int = 300):
        """Cache data with expiration."""
        if self.redis_client:
            try:
                serialized = pickle.dumps(data)
                self.redis_client.setex(key, expire, serialized)
            except Exception as e:
                logger.error(f"Error caching data: {e}")
                self.data_cache[key] = data
        else:
            self.data_cache[key] = data
    
    def _get_cached_data(self, key: str) -> Optional[Any]:
        """Get cached data."""
        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    return pickle.loads(data)
            except Exception as e:
                logger.error(f"Error getting cached data: {e}")
        
        return self.data_cache.get(key)
    
    def _get_cached_quote(self, symbol: str) -> Dict:
        """Get cached quote or return empty quote."""
        cached = self._get_cached_data(f"quote:{symbol}")
        if cached:
            return cached
        
        return {
            'symbol': symbol,
            'price': 0,
            'timestamp': datetime.now().isoformat(),
            'error': 'No data available'
        }
    
    def stop(self):
        """Stop all feeds and cleanup."""
        self.running = False
        self.executor.shutdown(wait=True)
        
        # Close WebSocket connections
        for ws in self.ws_connections.values():
            ws.close()
        
        logger.info("Market data feed stopped")


class DataBackfiller:
    """Handles historical data backfilling and synchronization."""
    
    def __init__(self, feed: MarketDataFeed):
        self.feed = feed
        self.backfill_queue = deque()
        self.processing = False
        
    def schedule_backfill(self, symbol: str, start_date: str, end_date: str):
        """Schedule a backfill job."""
        job = {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'status': 'pending'
        }
        self.backfill_queue.append(job)
        
        if not self.processing:
            threading.Thread(target=self._process_backfill_queue, daemon=True).start()
    
    def _process_backfill_queue(self):
        """Process backfill jobs from the queue."""
        self.processing = True
        
        while self.backfill_queue:
            job = self.backfill_queue.popleft()
            try:
                self._backfill_data(job)
                job['status'] = 'completed'
            except Exception as e:
                logger.error(f"Backfill failed for {job['symbol']}: {e}")
                job['status'] = 'failed'
                job['error'] = str(e)
        
        self.processing = False
    
    def _backfill_data(self, job: Dict):
        """Perform the actual backfill."""
        symbol = job['symbol']
        start_date = job['start_date']
        end_date = job['end_date']
        
        logger.info(f"Backfilling {symbol} from {start_date} to {end_date}")
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        # Store in cache
        cache_key = f"backfill:{symbol}:{start_date}:{end_date}"
        self.feed._cache_data(cache_key, df, expire=86400)  # Cache for 24 hours
        
        logger.info(f"Backfilled {len(df)} records for {symbol}")
        
        return df


class StreamingDataPipeline:
    """Manages streaming data pipeline with buffering and processing."""
    
    def __init__(self, feed: MarketDataFeed):
        self.feed = feed
        self.buffers = {}
        self.processors = []
        self.running = False
        
    def add_processor(self, processor: Callable):
        """Add a data processor to the pipeline."""
        self.processors.append(processor)
    
    def start_streaming(self, symbols: List[str], buffer_size: int = 100):
        """Start streaming data pipeline."""
        self.running = True
        
        # Initialize buffers
        for symbol in symbols:
            self.buffers[symbol] = deque(maxlen=buffer_size)
        
        # Subscribe to feed updates
        for symbol in symbols:
            self.feed.subscribe(symbol, lambda data: self._buffer_data(symbol, data))
        
        # Start processing thread
        threading.Thread(target=self._process_buffers, daemon=True).start()
        
        logger.info(f"Started streaming pipeline for {symbols}")
    
    def _buffer_data(self, symbol: str, data: Dict):
        """Buffer incoming data."""
        if symbol in self.buffers:
            self.buffers[symbol].append(data)
    
    def _process_buffers(self):
        """Process buffered data."""
        while self.running:
            for symbol, buffer in self.buffers.items():
                if len(buffer) >= 10:  # Process when buffer has enough data
                    batch = list(buffer)
                    buffer.clear()
                    
                    # Apply processors
                    for processor in self.processors:
                        try:
                            processor(symbol, batch)
                        except Exception as e:
                            logger.error(f"Error in processor: {e}")
            
            asyncio.run(asyncio.sleep(1))
    
    def stop(self):
        """Stop the streaming pipeline."""
        self.running = False
        logger.info("Streaming pipeline stopped")


# Example usage
if __name__ == "__main__":
    # Initialize feed
    feed = MarketDataFeed(cache_enabled=True)
    
    # Get real-time quotes
    symbols = ["AAPL", "GOOGL", "MSFT"]
    quotes = feed.get_batch_quotes(symbols)
    for symbol, quote in quotes.items():
        print(f"{symbol}: ${quote['price']}")
    
    # Get historical data
    historical = feed.get_historical_data("AAPL", period="1mo")
    print(f"Historical data shape: {historical.shape}")
    
    # Set up backfiller
    backfiller = DataBackfiller(feed)
    backfiller.schedule_backfill("AAPL", "2024-01-01", "2024-01-31")
    
    # Set up streaming pipeline
    pipeline = StreamingDataPipeline(feed)
    
    def process_batch(symbol, batch):
        print(f"Processing {len(batch)} records for {symbol}")
    
    pipeline.add_processor(process_batch)
    pipeline.start_streaming(symbols)
