"""
Interactive Tutorial System for Investment Engine Dashboard
Provides guided tours and contextual help for users
"""

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from typing import Dict, List, Any

class TutorialManager:
    """Manages interactive tutorials and guided tours for the dashboard."""
    
    def __init__(self):
        self.tutorials = self._initialize_tutorials()
        self.current_step = 0
        self.current_tutorial = None
    
    def _initialize_tutorials(self) -> Dict[str, Dict]:
        """Initialize tutorial content for different dashboard sections."""
        return {
            "welcome": {
                "title": "Welcome to Investment Engine Pro",
                "description": "Professional-grade quantitative investment platform",
                "steps": [
                    {
                        "title": "Dashboard Overview",
                        "content": "Navigate through 6 main tabs: Portfolio, Performance, Risk, Strategy, Market Data, and Research.",
                        "target": "#main-tabs",
                        "position": "bottom"
                    },
                    {
                        "title": "Portfolio Analytics",
                        "content": "Monitor real-time portfolio metrics, holdings, and performance indicators.",
                        "target": ".metric-card",
                        "position": "top"
                    },
                    {
                        "title": "Advanced Analytics",
                        "content": "Access sophisticated risk-adjusted metrics based on academic research.",
                        "target": "#performance-tab",
                        "position": "right"
                    }
                ]
            },
            "portfolio": {
                "title": "Portfolio Management",
                "description": "Comprehensive portfolio tracking and analysis",
                "steps": [
                    {
                        "title": "Key Metrics",
                        "content": "Monitor portfolio value, daily P&L, Sharpe ratio, and maximum drawdown.",
                        "target": ".metric-card",
                        "position": "bottom"
                    },
                    {
                        "title": "Asset Allocation",
                        "content": "Visualize portfolio allocation across assets and sectors.",
                        "target": "#allocation-pie-chart",
                        "position": "left"
                    },
                    {
                        "title": "Holdings Analysis",
                        "content": "Detailed breakdown of individual positions and their performance.",
                        "target": "#holdings-table",
                        "position": "top"
                    }
                ]
            },
            "performance": {
                "title": "Performance Analytics",
                "description": "Academic-grade performance evaluation metrics",
                "steps": [
                    {
                        "title": "Risk-Adjusted Returns",
                        "content": "Sharpe, Information, and Sortino ratios for comprehensive performance assessment.",
                        "target": ".metric-card",
                        "position": "bottom"
                    },
                    {
                        "title": "Factor Attribution",
                        "content": "Analyze performance attribution using Fama-French and custom factors.",
                        "target": "#factor-attribution-chart",
                        "position": "right"
                    },
                    {
                        "title": "Rolling Metrics",
                        "content": "Time-varying performance metrics to identify consistency patterns.",
                        "target": "#rolling-metrics-chart",
                        "position": "top"
                    }
                ]
            },
            "research": {
                "title": "Academic Research Hub",
                "description": "Latest factor investing research and insights",
                "steps": [
                    {
                        "title": "Recent Papers",
                        "content": "Top-cited factor investing papers from the past 1, 3, 6, and 12 months.",
                        "target": "#recent-papers",
                        "position": "bottom"
                    },
                    {
                        "title": "Factor Evolution Map",
                        "content": "Interactive mind map showing how factor investing landscape is evolving.",
                        "target": "#factor-mindmap",
                        "position": "left"
                    },
                    {
                        "title": "Implementation Insights",
                        "content": "Practical applications of academic findings to portfolio construction.",
                        "target": "#implementation-insights",
                        "position": "top"
                    }
                ]
            }
        }
    
    def create_tutorial_modal(self) -> html.Div:
        """Create the tutorial modal component."""
        return dbc.Modal([
            dbc.ModalHeader([
                dbc.ModalTitle(id="tutorial-modal-title"),
                html.Button("×", id="tutorial-close", className="btn-close")
            ]),
            dbc.ModalBody([
                html.Div(id="tutorial-content"),
                html.Div([
                    dbc.Progress(id="tutorial-progress", value=0, className="mb-3"),
                    html.Div([
                        dbc.Button("Previous", id="tutorial-prev", color="secondary", size="sm", disabled=True),
                        dbc.Button("Next", id="tutorial-next", color="primary", size="sm", className="ms-2"),
                        dbc.Button("Skip Tutorial", id="tutorial-skip", color="outline-secondary", size="sm", className="ms-auto")
                    ], className="d-flex justify-content-between")
                ])
            ])
        ], id="tutorial-modal", is_open=False, size="lg")
    
    def create_help_button(self) -> html.Div:
        """Create floating help button."""
        return html.Div([
            dbc.Button([
                html.I(className="fas fa-question-circle"),
                " Help"
            ], id="help-button", color="info", size="sm", className="position-fixed", 
            style={"bottom": "20px", "right": "20px", "z-index": "1050"})
        ])
    
    def create_contextual_tooltips(self) -> html.Div:
        """Create contextual tooltips for dashboard elements."""
        tooltips = []
        
        tooltip_configs = [
            {"target": "portfolio-value", "content": "Total market value of all holdings"},
            {"target": "sharpe-ratio", "content": "Risk-adjusted return measure (return per unit of risk)"},
            {"target": "max-drawdown", "content": "Largest peak-to-trough decline in portfolio value"},
            {"target": "information-ratio", "content": "Active return per unit of tracking error"},
            {"target": "sortino-ratio", "content": "Return per unit of downside deviation"},
            {"target": "calmar-ratio", "content": "Annual return divided by maximum drawdown"},
            {"target": "jensen-alpha", "content": "Risk-adjusted excess return over benchmark"}
        ]
        
        for config in tooltip_configs:
            tooltips.append(
                dbc.Tooltip(
                    config["content"],
                    target=config["target"],
                    placement="top"
                )
            )
        
        return html.Div(tooltips)

class TutorialCallbacks:
    """Handles tutorial-related callbacks."""
    
    @staticmethod
    def register_callbacks(app: dash.Dash, tutorial_manager: TutorialManager):
        """Register all tutorial callbacks."""
        
        @app.callback(
            [Output("tutorial-modal", "is_open"),
             Output("tutorial-modal-title", "children"),
             Output("tutorial-content", "children"),
             Output("tutorial-progress", "value")],
            [Input("help-button", "n_clicks"),
             Input("tutorial-next", "n_clicks"),
             Input("tutorial-prev", "n_clicks"),
             Input("tutorial-close", "n_clicks"),
             Input("tutorial-skip", "n_clicks")],
            [State("tutorial-modal", "is_open"),
             State("main-tabs", "value")]
        )
        def handle_tutorial_navigation(help_clicks, next_clicks, prev_clicks, 
                                     close_clicks, skip_clicks, is_open, active_tab):
            """Handle tutorial modal and navigation."""
            ctx = dash.callback_context
            if not ctx.triggered:
                return False, "", "", 0
            
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            if trigger_id == "help-button" and help_clicks:
                # Start tutorial based on current tab
                tutorial_key = "welcome"
                if active_tab == "portfolio-tab":
                    tutorial_key = "portfolio"
                elif active_tab == "performance-tab":
                    tutorial_key = "performance"
                elif active_tab == "research-tab":
                    tutorial_key = "research"
                
                tutorial = tutorial_manager.tutorials[tutorial_key]
                tutorial_manager.current_tutorial = tutorial_key
                tutorial_manager.current_step = 0
                
                step = tutorial["steps"][0]
                progress = (1 / len(tutorial["steps"])) * 100
                
                return True, tutorial["title"], _create_step_content(step), progress
            
            elif trigger_id in ["tutorial-close", "tutorial-skip"]:
                return False, "", "", 0
            
            elif trigger_id == "tutorial-next" and tutorial_manager.current_tutorial:
                tutorial = tutorial_manager.tutorials[tutorial_manager.current_tutorial]
                tutorial_manager.current_step = min(
                    tutorial_manager.current_step + 1, 
                    len(tutorial["steps"]) - 1
                )
                
                step = tutorial["steps"][tutorial_manager.current_step]
                progress = ((tutorial_manager.current_step + 1) / len(tutorial["steps"])) * 100
                
                return True, tutorial["title"], _create_step_content(step), progress
            
            elif trigger_id == "tutorial-prev" and tutorial_manager.current_tutorial:
                tutorial = tutorial_manager.tutorials[tutorial_manager.current_tutorial]
                tutorial_manager.current_step = max(tutorial_manager.current_step - 1, 0)
                
                step = tutorial["steps"][tutorial_manager.current_step]
                progress = ((tutorial_manager.current_step + 1) / len(tutorial["steps"])) * 100
                
                return True, tutorial["title"], _create_step_content(step), progress
            
            return is_open, "", "", 0

def _create_step_content(step: Dict[str, str]) -> html.Div:
    """Create content for a tutorial step."""
    return html.Div([
        html.H5(step["title"], className="mb-3"),
        html.P(step["content"], className="text-muted"),
        html.Hr(),
        html.Small(f"Target: {step['target']}", className="text-info")
    ])

# Initialize tutorial system
tutorial_manager = TutorialManager()
