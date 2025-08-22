"""Main dashboard application using Dash/Plotly."""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import base64

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use auto-configuration to avoid manual setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auto_config import get_config, get_cache, get_data_feed

try:
    from analytics.strategy_analyzer import AdvancedStrategyAnalyzer
    from analytics.enhanced_analyzer import enhanced_analyzer
    from dashboard.components.tutorial import tutorial_manager, TutorialCallbacks
    from dashboard.components.chatbot import ChatbotManager
    from dashboard.components.research import ResearchManager
    from dashboard.components.visualizations import AdvancedVisualizationManager
except ImportError as e:
    print(f"Some components not available: {e}")
    # Continue with basic functionality
from dashboard.components.usability import (
    KeyboardShortcuts, QuickStartMode, UserFeedback,
    DataValidation, NavigationEnhancer, UndoRedoManager, SettingsManager
)
from dashboard.simplified_interface import SimplifiedDashboard, OnboardingWizard
from core.security import security_manager, audit_logger
from core.architecture import orchestrator


class DashboardApp:
    """Main dashboard application for investment engine."""
    
    def __init__(self, 
                 data_manager=None,
                 execution_engine=None,
                 risk_manager=None,
                 performance_analyzer=None):
        """Initialize dashboard app.
        
        Args:
            data_manager: Data management system
            execution_engine: Execution engine
            risk_manager: Risk management system
            performance_analyzer: Performance analyzer
        """
        self.data_manager = data_manager
        self.execution_engine = execution_engine
        self.risk_manager = risk_manager
        self.performance_analyzer = performance_analyzer
        
        # Initialize components
        try:
            from dashboard.report_generator import ReportGenerator
            self.report_generator = ReportGenerator()
        except ImportError:
            self.report_generator = None
        
        self.strategy_analyzer = AdvancedStrategyAnalyzer()
        
        # Initialize usability components
        self.keyboard_shortcuts = KeyboardShortcuts()
        self.quick_start = QuickStartMode()
        self.user_feedback = UserFeedback()
        self.data_validator = DataValidation()
        self.nav_enhancer = NavigationEnhancer()
        self.undo_redo = UndoRedoManager()
        self.settings_manager = SettingsManager()
        self.simplified_dash = SimplifiedDashboard()
        self.onboarding = OnboardingWizard()
        
        # Auto-configure everything
        self.config = get_config()
        self.cache = get_cache()
        self.data_feed = get_data_feed()
        
        # Initialize Dash app with custom CSS
        assets_folder = os.path.join(os.path.dirname(__file__), 'assets')
        self.app = dash.Dash(__name__, 
                           external_stylesheets=[dbc.themes.BOOTSTRAP],
                           assets_folder=assets_folder)
        
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = html.Div([
            # Include custom CSS
            html.Link(rel='stylesheet', href='/assets/custom.css'),
            # Header
            html.Div([
                html.Div([
                    html.I(className="fas fa-chart-line", style={'fontSize': '2rem', 'marginRight': '1rem', 'color': '#667eea'}),
                    html.H1("Quantitative Investment Engine", className="header-title"),
                    html.P("Advanced Multi-Factor Portfolio Management & Risk Analytics", className="header-subtitle")
                ], className="header-container fade-in")
            ]),
            
            # Main content area with tabs
            html.Div([
                self.keyboard_shortcuts.get_component(),
                self.keyboard_shortcuts.get_help_modal(),
                html.Div(id="breadcrumbs-container", className="mb-3"),
                dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            dbc.ButtonGroup([
                                dbc.Button("Beginner", id="mode-beginner", color="success", outline=True),
                                dbc.Button("Standard", id="mode-standard", color="primary"),
                                dbc.Button("Advanced", id="mode-advanced", color="danger", outline=True),
                            ], className="mb-3")
                        ], className="text-end")
                    ]),
                    dbc.Tabs([
                        dbc.Tab(label="Quick Start", tab_id="quickstart-tab"),
                        dbc.Tab(label="Portfolio Overview", tab_id="portfolio-tab"),
                        dbc.Tab(label="Performance Analytics", tab_id="performance-tab"),
                        dbc.Tab(label="Risk Management", tab_id="risk-tab"),
                        dbc.Tab(label="Strategy Builder", tab_id="strategy-tab"),
                        dbc.Tab(label="Market Data", tab_id="market-tab"),
                        dbc.Tab(label="Research Hub", tab_id="research-tab"),
                        dbc.Tab(label="Settings", tab_id="settings-tab"),
                    ], id="main-tabs", active_tab="quickstart-tab"),
                ]),
                html.Div(id="tab-content", className="mt-4"),
                html.Div(id="toast-container", style={"position": "fixed", "top": 80, "right": 20, "zIndex": 9999}),
                dbc.Modal([
                    dbc.ModalBody([
                        html.Div(id="progress-content")
                    ])
                ], id="progress-modal", is_open=False, backdrop="static", centered=True)
            ], className="main-content"),
            
            # Tutorial system
            tutorial_manager.create_tutorial_modal(),
            tutorial_manager.create_help_button(),
            
            # Status bar
            html.Div([
                html.Div([
                    html.Span(className="status-indicator status-online"),
                    html.Span("System Online", style={'marginRight': '2rem'}),
                    html.Span(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}", id="last-updated"),
                    html.Span(" | ", style={'margin': '0 1rem'}),
                    html.Span("Real-time Data Feed Active", style={'color': '#38a169'})
                ], style={'textAlign': 'center', 'padding': '1rem', 'background': 'rgba(255,255,255,0.1)', 'backdropFilter': 'blur(10px)'})
            ]),
            
            # Refresh interval
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # Update every 30 seconds
                n_intervals=0
            )
        ])
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        # Register tutorial callbacks
        TutorialCallbacks.register_callbacks(self.app, tutorial_manager)
        
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('main-tabs', 'active_tab')],
            prevent_initial_call=False
        )
        def update_tab_content(active_tab):
            """Update content based on selected tab."""
            try:
                if active_tab == 'quickstart-tab':
                    return self.quick_start.get_component()
                elif active_tab == 'portfolio-tab':
                    return self._create_portfolio_tab()
                elif active_tab == 'performance-tab':
                    return self._create_performance_tab()
                elif active_tab == 'risk-tab':
                    return self._create_risk_tab()
                elif active_tab == 'strategy-tab':
                    return self._create_strategy_tab()
                elif active_tab == 'market-tab':
                    return self._create_market_tab()
                elif active_tab == 'research-tab':
                    return ResearchManager().create_research_tab()
                elif active_tab == 'settings-tab':
                    return self._create_settings_tab()
                else:
                    return html.Div("Tab content not implemented yet.")
            except Exception as e:
                return html.Div([
                    html.H3("Error Loading Tab"),
                    html.P(f"Error: {str(e)}"),
                    html.P("Please refresh the page or try again.")
                ], style={'color': 'red', 'padding': '20px'})
        
        # Keyboard shortcuts callback
        @self.app.callback(
            Output('shortcuts-modal', 'is_open'),
            [Input('show-shortcuts', 'n_clicks'),
             Input('close-shortcuts', 'n_clicks')],
            [State('shortcuts-modal', 'is_open')],
            prevent_initial_call=True
        )
        def toggle_shortcuts_modal(show_clicks, close_clicks, is_open):
            ctx = callback_context
            if not ctx.triggered:
                return is_open
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if trigger_id == 'show-shortcuts':
                return True
            elif trigger_id == 'close-shortcuts':
                return False
            return is_open
        
        # Mode selector callbacks
        @self.app.callback(
            [Output('mode-beginner', 'color'),
             Output('mode-standard', 'color'),
             Output('mode-advanced', 'color')],
            [Input('mode-beginner', 'n_clicks'),
             Input('mode-standard', 'n_clicks'),
             Input('mode-advanced', 'n_clicks')],
            prevent_initial_call=True
        )
        def update_mode_selection(beginner_clicks, standard_clicks, advanced_clicks):
            ctx = callback_context
            if not ctx.triggered:
                return 'outline-success', 'primary', 'outline-danger'
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if trigger_id == 'mode-beginner':
                return 'success', 'outline-primary', 'outline-danger'
            elif trigger_id == 'mode-standard':
                return 'outline-success', 'primary', 'outline-danger'
            elif trigger_id == 'mode-advanced':
                return 'outline-success', 'outline-primary', 'danger'
            
            return 'outline-success', 'primary', 'outline-danger'
        
        # Quick Start strategy selection callbacks
        for strategy_key in QuickStartMode.PRESET_STRATEGIES.keys():
            @self.app.callback(
                Output('toast-container', 'children', allow_duplicate=True),
                [Input(f'select-{strategy_key}', 'n_clicks')],
                prevent_initial_call=True
            )
            def select_strategy(n_clicks, key=strategy_key):
                if n_clicks:
                    strategy = QuickStartMode.PRESET_STRATEGIES[key]
                    return self.user_feedback.create_toast(
                        f"Selected {strategy['name']} strategy. Configuring your portfolio...",
                        "Strategy Selected",
                        "success"
                    )
                return dash.no_update
        
        # Settings export/import callbacks
        @self.app.callback(
            Output('download-settings', 'data'),
            [Input('export-settings-btn', 'n_clicks')],
            prevent_initial_call=True
        )
        def export_settings(n_clicks):
            if n_clicks:
                # Gather current settings
                current_settings = {
                    'portfolio_config': {
                        'target_value': 100000,
                        'rebalance_frequency': 'monthly'
                    },
                    'risk_parameters': {
                        'max_position_size': 0.1,
                        'stop_loss': 0.05,
                        'max_drawdown': 0.15
                    },
                    'ui_preferences': {
                        'theme': 'light',
                        'chart_type': 'candlestick'
                    }
                }
                
                settings_json = self.settings_manager.export_settings(current_settings)
                return dict(content=settings_json, filename="investment_settings.json")
        
        @self.app.callback(
            Output('import-status', 'children'),
            [Input('upload-settings', 'contents')],
            [State('upload-settings', 'filename')],
            prevent_initial_call=True
        )
        def import_settings(contents, filename):
            if contents:
                try:
                    content_type, content_string = contents.split(',')
                    decoded = base64.b64decode(content_string).decode('utf-8')
                    
                    result = self.settings_manager.import_settings(decoded)
                    
                    if result['success']:
                        return dbc.Alert(
                            f"Settings imported successfully from {filename}",
                            color="success",
                            dismissable=True
                        )
                    else:
                        return dbc.Alert(
                            f"Error importing settings: {result['error']}",
                            color="danger",
                            dismissable=True
                        )
                except Exception as e:
                    return dbc.Alert(
                        f"Failed to import settings: {str(e)}",
                        color="danger",
                        dismissable=True
                    )
            return dash.no_update
    
    def _create_portfolio_tab(self):
        """Create portfolio overview tab."""
        portfolio_data, holdings_data = self.create_sample_data()
        
        return html.Div([
            # Key metrics cards
            html.Div([
                html.Div([
                    html.P("Portfolio Value", className="metric-title"),
                    html.H2(f"${portfolio_data['portfolio_value'].iloc[-1]:,.0f}", className="metric-value"),
                    html.Span("+2.4%", className="metric-change metric-positive")
                ], className="metric-card fade-in", style={'width': '23%', 'display': 'inline-block', 'margin': '1%'}),
                
                html.Div([
                    html.P("Daily P&L", className="metric-title"),
                    html.H2(f"${(portfolio_data['portfolio_value'].iloc[-1] - portfolio_data['portfolio_value'].iloc[-2]):,.0f}", className="metric-value"),
                    html.Span("+0.8%", className="metric-change metric-positive")
                ], className="metric-card fade-in", style={'width': '23%', 'display': 'inline-block', 'margin': '1%'}),
                
                html.Div([
                    html.P("Sharpe Ratio", className="metric-title"),
                    html.H2("1.42", className="metric-value"),
                    html.Span("Excellent", className="metric-change metric-positive")
                ], className="metric-card fade-in", style={'width': '23%', 'display': 'inline-block', 'margin': '1%'}),
                
                html.Div([
                    html.P("Max Drawdown", className="metric-title"),
                    html.H2("-3.2%", className="metric-value"),
                    html.Span("Low Risk", className="metric-change metric-positive")
                ], className="metric-card fade-in", style={'width': '23%', 'display': 'inline-block', 'margin': '1%'})
            ], style={'textAlign': 'center', 'margin': '2rem 0'}),
            
            # Portfolio allocation chart
            html.Div([
                html.Div([
                    html.H4("Asset Allocation"),
                    dcc.Graph(id="allocation-pie-chart")
                ], className="six columns"),
                
                html.Div([
                    html.H4("Sector Allocation"),
                    dcc.Graph(id="sector-pie-chart")
                ], className="six columns")
            ], className="row"),
            
            # Holdings table
            html.Div([
                html.H4("Current Holdings"),
                html.Div(id="holdings-table")
            ])
        ])
    
    def _create_performance_tab(self):
        """Create advanced performance analytics tab with enhanced academic metrics."""
        portfolio_data, holdings_data = self.create_sample_data()
        
        # Extract returns data safely
        returns = portfolio_data['returns'].dropna()
        
        # Create benchmark returns with numeric index
        benchmark_returns = pd.Series(
            np.random.normal(0.0003, 0.015, len(returns)), 
            index=range(len(returns))
        )
        
        # Ensure returns also has numeric index
        returns = pd.Series(returns.values, index=range(len(returns)))
        
        # Run enhanced comprehensive analysis
        analysis = enhanced_analyzer.analyze_comprehensive_performance(returns, benchmark_returns)
        
        # Create enhanced visualizations
        enhanced_dashboard = AdvancedVisualizationManager().create_enhanced_performance_dashboard(
            returns, benchmark_returns
        )
        
        return html.Div([
            # Enhanced Performance Dashboard
            html.Div([
                html.H3("📊 Enhanced Performance Analytics", className="chart-title"),
                dcc.Graph(figure=enhanced_dashboard, id="enhanced-performance-dashboard")
            ]),
            
            # Academic Insights Section
            html.Div([
                html.H3("🎓 Academic Research Insights", className="chart-title"),
                html.Div([
                    html.Div([
                        html.H5("Performance Evaluation"),
                        html.Ul([html.Li(insight) for insight in analysis['academic_insights']['performance_evaluation']])
                    ], className="chart-container", style={'width': '48%', 'display': 'inline-block', 'margin': '1%'}),
                    
                    html.Div([
                        html.H5("Risk Assessment"),
                        html.Ul([html.Li(insight) for insight in analysis['academic_insights']['risk_assessment']])
                    ], className="chart-container", style={'width': '48%', 'display': 'inline-block', 'margin': '1%'})
                ])
            ]),
            
            # Advanced Performance Metrics
            html.Div([
                html.H3("📈 Comprehensive Metrics", className="chart-title"),
                html.Div([
                    # Risk-Adjusted Returns
                    html.Div([
                        html.P("Sharpe Ratio", className="metric-title"),
                        html.H2(f"{analysis['basic_metrics']['sharpe_ratio']:.3f}", className="metric-value"),
                        html.Span("Risk-Adjusted Return", className="metric-change metric-positive")
                    ], className="metric-card", style={'width': '18%', 'display': 'inline-block', 'margin': '1%'}),
                    
                    html.Div([
                        html.P("Information Ratio", className="metric-title"),
                        html.H2(f"{analysis['basic_metrics'].get('information_ratio', 0):.3f}", className="metric-value"),
                        html.Span("Active Management", className="metric-change metric-positive")
                    ], className="metric-card", style={'width': '18%', 'display': 'inline-block', 'margin': '1%'}),
                    
                    html.Div([
                        html.P("Sortino Ratio", className="metric-title"),
                        html.H2(f"{analysis['risk_metrics'].get('sortino_ratio', 0):.3f}", className="metric-value"),
                        html.Span("Downside Risk", className="metric-change metric-positive")
                    ], className="metric-card", style={'width': '18%', 'display': 'inline-block', 'margin': '1%'}),
                    
                    html.Div([
                        html.P("Max Drawdown", className="metric-title"),
                        html.H2(f"{analysis['risk_metrics'].get('max_drawdown', 0):.2%}", className="metric-value"),
                        html.Span("Peak Decline", className="metric-change metric-negative")
                    ], className="metric-card", style={'width': '18%', 'display': 'inline-block', 'margin': '1%'}),
                    
                    html.Div([
                        html.P("Jensen's Alpha", className="metric-title"),
                        html.H2(f"{analysis['basic_metrics'].get('jensen_alpha', 0):.2%}", className="metric-value"),
                        html.Span("Excess Return", className="metric-change metric-positive")
                    ], className="metric-card", style={'width': '18%', 'display': 'inline-block', 'margin': '1%'})
                ], style={'textAlign': 'center', 'margin': '2rem 0'})
            ])
        ])
    
    def _create_risk_tab(self):
        """Create risk management tab."""
        return html.Div([
            html.H2("Risk Management"),
            
            # Risk alerts
            html.Div([
                html.H4("Risk Alerts"),
                html.Div(id="risk-alerts", style={'backgroundColor': '#fff3cd', 'padding': '10px', 'margin': '10px'})
            ]),
            
            # Risk metrics
            html.Div([
                html.Div([
                    html.H4("VaR Analysis"),
                    dcc.Graph(id="var-chart")
                ], className="six columns"),
                
                html.Div([
                    html.H4("Risk Contribution"),
                    dcc.Graph(id="risk-contribution-chart")
                ], className="six columns")
            ], className="row"),
            
            html.Div([
                html.Div([
                    html.H4("Correlation Matrix"),
                    dcc.Graph(id="correlation-heatmap")
                ], className="six columns"),
                
                html.Div([
                    html.H4("Risk Metrics"),
                    html.Div(id="risk-metrics-table")
                ], className="six columns")
            ], className="row")
        ])
    
    def _create_execution_tab(self):
        """Create strategy execution tab."""
        return html.Div([
            html.H2("Strategy Execution"),
            
            # Strategy controls
            html.Div([
                html.Div([
                    html.Label("Select Strategy:"),
                    dcc.Dropdown(
                        id='strategy-dropdown',
                        options=[
                            {'label': 'Multi-Factor Strategy', 'value': 'multi_factor'},
                            {'label': 'Risk Parity', 'value': 'risk_parity'},
                            {'label': 'Momentum Strategy', 'value': 'momentum'}
                        ],
                        value='multi_factor'
                    )
                ], className="four columns"),
                
                html.Div([
                    html.Button('Execute Strategy', id='execute-button', 
                              style={'backgroundColor': '#3498db', 'color': 'white', 'padding': '10px 20px', 'margin': '10px'})
                ], className="four columns"),
                
                html.Div([
                    html.Button('Stop Strategy', id='stop-button',
                              style={'backgroundColor': '#e74c3c', 'color': 'white', 'padding': '10px 20px', 'margin': '10px'})
                ], className="four columns")
            ], className="row"),
            
            # Execution status
            html.Div([
                html.H4("Execution Status"),
                html.Div(id="execution-status")
            ]),
            
            # Trade history
            html.Div([
                html.H4("Recent Trades"),
                html.Div(id="trade-history-table")
            ]),
            
            # Execution metrics
            html.Div([
                html.Div([
                    html.H4("Execution Quality"),
                    dcc.Graph(id="execution-quality-chart")
                ], className="six columns"),
                
                html.Div([
                    html.H4("Transaction Costs"),
                    dcc.Graph(id="transaction-costs-chart")
                ], className="six columns")
            ], className="row")
        ])
    
    def _create_settings_tab(self):
        """Create settings management tab."""
        return html.Div([
            html.H3("⚙️ Settings Management", className="mb-4"),
            
            # Export/Import Settings
            self.settings_manager.get_export_component(),
            
            html.Hr(className="my-4"),
            
            # Keyboard Shortcuts Reference
            html.Div([
                html.H4("⌨️ Keyboard Shortcuts"),
                html.P("Use these shortcuts to navigate the dashboard quickly:"),
                dbc.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Shortcut"),
                            html.Th("Action")
                        ])
                    ]),
                    html.Tbody([
                        html.Tr([html.Td("Alt + P"), html.Td("Navigate to Portfolio")]),
                        html.Tr([html.Td("Alt + R"), html.Td("Navigate to Risk Management")]),
                        html.Tr([html.Td("Alt + S"), html.Td("Navigate to Strategy Builder")]),
                        html.Tr([html.Td("Alt + H"), html.Td("Show Help")]),
                        html.Tr([html.Td("Ctrl + Z"), html.Td("Undo Last Action")]),
                        html.Tr([html.Td("Ctrl + Y"), html.Td("Redo Last Action")]),
                        html.Tr([html.Td("Ctrl + E"), html.Td("Export Data")]),
                    ])
                ], striped=True, hover=True, responsive=True, className="mt-3")
            ], className="mb-4"),
            
            html.Hr(className="my-4"),
            
            # UI Preferences
            html.Div([
                html.H4("🎨 UI Preferences"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Dashboard Mode"),
                        dbc.RadioItems(
                            id="ui-mode-preference",
                            options=[
                                {"label": "Beginner", "value": "beginner"},
                                {"label": "Standard", "value": "standard"},
                                {"label": "Advanced", "value": "advanced"},
                            ],
                            value="standard",
                            inline=True
                        )
                    ], md=6),
                    dbc.Col([
                        dbc.Label("Theme"),
                        dbc.RadioItems(
                            id="theme-preference",
                            options=[
                                {"label": "Light", "value": "light"},
                                {"label": "Dark", "value": "dark"},
                                {"label": "Auto", "value": "auto"},
                            ],
                            value="light",
                            inline=True
                        )
                    ], md=6)
                ])
            ])
        ])
    
    def _create_market_tab(self):
        """Create market data tab."""
        return html.Div([
            html.H2("Market Data"),
            
            # Symbol input
            html.Div([
                html.Label("Enter Symbols (comma-separated):"),
                dcc.Input(
                    id='symbol-input',
                    type='text',
                    value='AAPL,MSFT,GOOGL,AMZN,TSLA',
                    style={'width': '300px', 'margin': '10px'}
                ),
                html.Button('Update', id='update-market-data-button',
                          style={'backgroundColor': '#2ecc71', 'color': 'white', 'padding': '10px 20px', 'margin': '10px'})
            ]),
            
            # Market charts
            html.Div([
                html.Div([
                    html.H4("Price Chart"),
                    dcc.Graph(id="price-chart")
                ], className="twelve columns")
            ], className="row"),
            
            html.Div([
                html.Div([
                    html.H4("Returns Distribution"),
                    dcc.Graph(id="returns-distribution")
                ], className="six columns"),
                
                html.Div([
                    html.H4("Volatility Analysis"),
                    dcc.Graph(id="volatility-chart")
                ], className="six columns")
            ], className="row"),
            
            # Market data table
            html.Div([
                html.H4("Market Data Summary"),
                html.Div(id="market-data-table")
            ])
        ])
    
    def create_sample_data(self):
        """Create sample data for demonstration."""
        try:
            # Sample portfolio data
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            np.random.seed(42)
            
            # Generate portfolio values
            initial_value = 1000000
            returns = np.random.normal(0.0008, 0.015, len(dates))
            portfolio_values = initial_value * np.exp(np.cumsum(returns))
            
            portfolio_data = pd.DataFrame({
                'date': dates,
                'portfolio_value': portfolio_values.astype(float),
                'returns': np.concatenate([[0], np.diff(portfolio_values) / portfolio_values[:-1]]).astype(float)
            })
            
            # Ensure no timestamp columns are accidentally accessed
            portfolio_data = portfolio_data.reset_index(drop=True)
            
            # Sample holdings
            holdings_data = pd.DataFrame({
                'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
                'Shares': [1000, 800, 500, 300, 200],
                'Price': [150.0, 300.0, 2500.0, 3000.0, 800.0],
                'Market Value': [150000, 240000, 1250000, 900000, 160000],
                'Weight': [0.15, 0.24, 0.125, 0.09, 0.16],
                'P&L': [5000, -2000, 15000, -5000, 8000]
            })
            
            return portfolio_data, holdings_data
        except Exception as e:
            print(f"Error creating sample data: {e}")
            # Return minimal fallback data
            portfolio_data = pd.DataFrame({
                'date': [pd.Timestamp('2023-01-01')],
                'portfolio_value': [1000000.0],
                'returns': [0.0]
            })
            portfolio_data = portfolio_data.reset_index(drop=True)
            holdings_data = pd.DataFrame({
                'Symbol': ['AAPL'],
                'Shares': [1000],
                'Price': [150.0],
                'Market Value': [150000],
                'Weight': [1.0],
                'P&L': [0]
            })
            return portfolio_data, holdings_data
    
    def _create_rolling_metrics_chart(self, returns: pd.Series, benchmark_returns: pd.Series):
        """Create rolling performance metrics chart."""
        try:
            # Use smaller rolling window for shorter data
            window = min(30, len(returns) // 3)
            if window < 5:
                window = 5
            
            # Calculate rolling Sharpe ratio with numeric index
            rolling_sharpe = (returns.rolling(window).mean() - 0.02/252) / returns.rolling(window).std()
            rolling_alpha = (returns - benchmark_returns).rolling(window).mean() * 252
            
            # Create numeric x-axis
            x_axis = list(range(len(rolling_sharpe)))
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=rolling_sharpe.values,
                mode='lines',
                name='Rolling Sharpe Ratio',
                line=dict(color='#667eea', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=rolling_alpha.values,
                mode='lines',
                name='Rolling Alpha',
                line=dict(color='#38a169', width=2),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title=f'Rolling Performance Metrics ({window}-day)',
                xaxis_title='Time Period',
                yaxis_title='Sharpe Ratio',
                yaxis2=dict(title='Alpha', overlaying='y', side='right'),
                template='plotly_white',
                height=400
            )
            
            return fig
        except Exception as e:
            # Return empty chart on error
            fig = go.Figure()
            fig.add_annotation(
                text=f"Chart unavailable: {str(e)[:50]}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def _create_drawdown_chart(self, returns: pd.Series):
        """Create drawdown analysis chart."""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdowns.index,
            y=drawdowns.values * 100,
            mode='lines',
            name='Drawdown %',
            fill='tonexty',
            line=dict(color='#e53e3e', width=1),
            fillcolor='rgba(229, 62, 62, 0.3)'
        ))
        
        fig.update_layout(
            title='Portfolio Drawdown Analysis',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def run_server(self, debug=True, port=8050):
        """Run the dashboard server.
        
        Args:
            debug: Enable debug mode
            port: Port number
        """
        self.app.run(debug=debug, port=port, host='127.0.0.1')
    
    def generate_static_report(self, output_path: str = "portfolio_report.html"):
        """Generate static HTML report.
        
        Args:
            output_path: Output file path
        """
        # Create sample visualizations
        portfolio_data, holdings_data = self.create_sample_data()
        
        # Generate charts
        performance_fig = self.visualizer.create_performance_chart(portfolio_data)
        allocation_fig = self.visualizer.create_allocation_chart(holdings_data)
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Investment Portfolio Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #2c3e50; }}
                .section {{ margin: 30px 0; }}
                .chart {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Investment Portfolio Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Portfolio Performance</h2>
                <div class="chart" id="performance-chart"></div>
            </div>
            
            <div class="section">
                <h2>Asset Allocation</h2>
                <div class="chart" id="allocation-chart"></div>
            </div>
            
            <script>
                Plotly.newPlot('performance-chart', {performance_fig.to_json()});
                Plotly.newPlot('allocation-chart', {allocation_fig.to_json()});
            </script>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Static report generated: {output_path}")


# Example usage
if __name__ == "__main__":
    dashboard = DashboardApp()
    dashboard.run_server(debug=True)
