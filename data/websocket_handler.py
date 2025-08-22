"""
WebSocket handler for real-time dashboard updates.
"""

import json
import asyncio
from typing import Dict, List, Set
from datetime import datetime
import logging
from dash import callback_context
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

logger = logging.getLogger(__name__)


class WebSocketHandler:
    """Handles WebSocket connections for dashboard real-time updates."""
    
    def __init__(self):
        self.connections: Set = set()
        self.subscriptions: Dict[str, Set] = {}
        self.data_streams: Dict = {}
        self.update_callbacks = {}
        
    async def handle_connection(self, websocket, path):
        """Handle new WebSocket connection."""
        self.connections.add(websocket)
        try:
            async for message in websocket:
                await self.process_message(websocket, message)
        finally:
            self.connections.remove(websocket)
            await self.cleanup_subscriptions(websocket)
    
    async def process_message(self, websocket, message):
        """Process incoming WebSocket message."""
        try:
            data = json.loads(message)
            action = data.get('action')
            
            if action == 'subscribe':
                await self.handle_subscribe(websocket, data)
            elif action == 'unsubscribe':
                await self.handle_unsubscribe(websocket, data)
            elif action == 'ping':
                await websocket.send(json.dumps({'type': 'pong'}))
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await websocket.send(json.dumps({
                'type': 'error',
                'message': str(e)
            }))
    
    async def handle_subscribe(self, websocket, data):
        """Handle subscription request."""
        symbols = data.get('symbols', [])
        stream_type = data.get('type', 'price')
        
        for symbol in symbols:
            key = f"{stream_type}:{symbol}"
            if key not in self.subscriptions:
                self.subscriptions[key] = set()
            self.subscriptions[key].add(websocket)
        
        await websocket.send(json.dumps({
            'type': 'subscribed',
            'symbols': symbols,
            'stream_type': stream_type
        }))
    
    async def handle_unsubscribe(self, websocket, data):
        """Handle unsubscription request."""
        symbols = data.get('symbols', [])
        stream_type = data.get('type', 'price')
        
        for symbol in symbols:
            key = f"{stream_type}:{symbol}"
            if key in self.subscriptions:
                self.subscriptions[key].discard(websocket)
    
    async def broadcast_update(self, stream_key: str, data: Dict):
        """Broadcast update to all subscribed connections."""
        if stream_key in self.subscriptions:
            message = json.dumps({
                'type': 'update',
                'stream': stream_key,
                'data': data,
                'timestamp': datetime.now().isoformat()
            })
            
            # Send to all subscribed connections
            tasks = []
            for websocket in self.subscriptions[stream_key]:
                tasks.append(websocket.send(message))
            
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def cleanup_subscriptions(self, websocket):
        """Clean up subscriptions for disconnected client."""
        for subscribers in self.subscriptions.values():
            subscribers.discard(websocket)
    
    def create_dash_component(self):
        """Create Dash component for WebSocket integration."""
        return dcc.Interval(
            id='ws-update-interval',
            interval=1000,  # Update every second
            n_intervals=0
        )
    
    def register_dash_callbacks(self, app, feed_manager):
        """Register Dash callbacks for real-time updates."""
        
        @app.callback(
            Output('live-price-graph', 'figure'),
            [Input('ws-update-interval', 'n_intervals')],
            [State('symbol-dropdown', 'value')]
        )
        def update_price_graph(n, selected_symbols):
            if not selected_symbols:
                return go.Figure()
            
            fig = go.Figure()
            
            for symbol in selected_symbols:
                if symbol in feed_manager.price_history:
                    history = list(feed_manager.price_history[symbol])
                    if history:
                        times = [h['timestamp'] for h in history]
                        prices = [h['price'] for h in history]
                        
                        fig.add_trace(go.Scatter(
                            x=times,
                            y=prices,
                            name=symbol,
                            mode='lines'
                        ))
            
            fig.update_layout(
                title='Real-Time Price Chart',
                xaxis_title='Time',
                yaxis_title='Price ($)',
                hovermode='x unified'
            )
            
            return fig
        
        @app.callback(
            Output('market-heatmap', 'figure'),
            [Input('ws-update-interval', 'n_intervals')]
        )
        def update_heatmap(n):
            # Get latest quotes for tracked symbols
            symbols = list(feed_manager.subscribers.keys())
            if not symbols:
                return go.Figure()
            
            quotes = feed_manager.get_batch_quotes(symbols[:20])  # Limit to 20 symbols
            
            # Prepare data for heatmap
            changes = []
            labels = []
            
            for symbol, quote in quotes.items():
                change_pct = quote.get('change_percent', 0)
                changes.append(change_pct)
                labels.append(f"{symbol}<br>{change_pct:.2f}%")
            
            # Create treemap
            fig = go.Figure(go.Treemap(
                labels=labels,
                values=[abs(c) + 1 for c in changes],  # Size by absolute change
                parents=[""] * len(labels),
                marker=dict(
                    colorscale='RdYlGn',
                    cmid=0,
                    colorbar=dict(title="Change %"),
                    cmin=-5,
                    cmax=5,
                    showscale=True,
                    line=dict(width=2)
                ),
                text=[f"{c:.2f}%" for c in changes],
                textposition="middle center",
                hovertemplate='<b>%{label}</b><br>Change: %{text}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Market Heatmap',
                height=500
            )
            
            return fig


class DashboardWebSocketIntegration:
    """Integrates WebSocket with Dash dashboard."""
    
    def __init__(self, ws_handler: WebSocketHandler):
        self.ws_handler = ws_handler
        self.active_streams = {}
        
    def create_live_components(self):
        """Create live dashboard components."""
        return html.Div([
            # WebSocket status indicator
            html.Div(id='ws-status', children=[
                html.Span('●', style={'color': 'green', 'fontSize': '20px'}),
                html.Span(' Connected', style={'marginLeft': '5px'})
            ]),
            
            # Symbol selector for live tracking
            dcc.Dropdown(
                id='live-symbols',
                options=[
                    {'label': 'Apple (AAPL)', 'value': 'AAPL'},
                    {'label': 'Google (GOOGL)', 'value': 'GOOGL'},
                    {'label': 'Microsoft (MSFT)', 'value': 'MSFT'},
                    {'label': 'Amazon (AMZN)', 'value': 'AMZN'},
                    {'label': 'Tesla (TSLA)', 'value': 'TSLA'}
                ],
                value=['AAPL', 'GOOGL'],
                multi=True,
                placeholder='Select symbols to track'
            ),
            
            # Live price ticker
            html.Div(id='price-ticker', className='ticker-container'),
            
            # Live charts container
            html.Div([
                dcc.Graph(id='live-price-graph'),
                dcc.Graph(id='live-volume-graph')
            ], className='live-charts'),
            
            # Market depth/order book
            html.Div(id='order-book', className='order-book-container'),
            
            # WebSocket update interval
            dcc.Interval(
                id='ws-update-interval',
                interval=1000,
                n_intervals=0
            )
        ])
    
    def create_ticker_component(self, quotes: Dict):
        """Create price ticker component."""
        ticker_items = []
        
        for symbol, quote in quotes.items():
            change = quote.get('change', 0)
            change_pct = quote.get('change_percent', 0)
            color = 'green' if change >= 0 else 'red'
            arrow = '▲' if change >= 0 else '▼'
            
            ticker_items.append(
                html.Div([
                    html.Span(symbol, className='ticker-symbol'),
                    html.Span(f"${quote['price']:.2f}", className='ticker-price'),
                    html.Span(
                        f"{arrow} {abs(change):.2f} ({abs(change_pct):.2f}%)",
                        style={'color': color},
                        className='ticker-change'
                    )
                ], className='ticker-item')
            )
        
        return html.Div(ticker_items, className='ticker-scroll')


# Import for HTML components
from dash import html
