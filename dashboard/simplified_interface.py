"""
Simplified interface for beginners and improved user experience.
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class SimplifiedDashboard:
    """Simplified dashboard interface for better usability."""
    
    def __init__(self):
        self.presets = self._load_presets()
        
    def _load_presets(self):
        """Load preset configurations."""
        return {
            'beginner': {
                'name': 'Beginner Mode',
                'features': ['portfolio_overview', 'simple_charts', 'guided_help'],
                'complexity': 1
            },
            'intermediate': {
                'name': 'Standard Mode',
                'features': ['portfolio_overview', 'performance', 'risk', 'basic_strategy'],
                'complexity': 2
            },
            'advanced': {
                'name': 'Advanced Mode',
                'features': ['all'],
                'complexity': 3
            }
        }
    
    def create_layout(self, mode: str = 'beginner'):
        """Creates simplified layout based on user mode."""
        preset = self.presets.get(mode, self.presets['beginner'])
        
        return dbc.Container([
            # Header with mode selector
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H1("Investment Engine", className="d-inline-block me-3"),
                        dbc.Badge(preset['name'], color="info", className="align-middle")
                    ])
                ], md=8),
                dbc.Col([
                    self._create_mode_selector()
                ], md=4, className="text-end")
            ], className="mb-4"),
            
            # Main content area
            html.Div(id="main-content", children=[
                self._create_content_by_mode(mode)
            ]),
            
            # Floating help button
            html.Div([
                dbc.Button([
                    html.I(className="fas fa-question-circle")
                ], id="help-btn", color="primary", 
                className="rounded-circle",
                style={
                    'position': 'fixed',
                    'bottom': '20px',
                    'right': '20px',
                    'width': '60px',
                    'height': '60px',
                    'zIndex': 1000
                })
            ]),
            
            # Notifications area
            html.Div(id="notifications", style={'position': 'fixed', 'top': '10px', 'right': '10px', 'zIndex': 9999})
        ], fluid=True, className="py-3")
    
    def _create_mode_selector(self):
        """Creates the interface mode selector."""
        return dbc.ButtonGroup([
            dbc.Button(
                preset['name'],
                id=f"mode-{mode}",
                color="primary" if mode == 'beginner' else "outline-primary",
                size="sm"
            ) for mode, preset in self.presets.items()
        ])
    
    def _create_content_by_mode(self, mode: str):
        """Creates content based on selected mode."""
        if mode == 'beginner':
            return self._create_beginner_content()
        elif mode == 'intermediate':
            return self._create_intermediate_content()
        else:
            return self._create_advanced_content()
    
    def _create_beginner_content(self):
        """Creates simplified beginner interface."""
        return dbc.Row([
            # Left panel - Quick actions
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-bolt me-2"),
                        "Quick Actions"
                    ]),
                    dbc.CardBody([
                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                dbc.Button([
                                    html.I(className="fas fa-plus-circle me-2"),
                                    "Add Investment"
                                ], color="success", className="w-100 mb-2"),
                                dbc.Button([
                                    html.I(className="fas fa-chart-line me-2"),
                                    "View Performance"
                                ], color="info", className="w-100 mb-2"),
                                dbc.Button([
                                    html.I(className="fas fa-balance-scale me-2"),
                                    "Check Balance"
                                ], color="warning", className="w-100 mb-2"),
                                dbc.Button([
                                    html.I(className="fas fa-download me-2"),
                                    "Export Report"
                                ], color="secondary", className="w-100")
                            ])
                        ])
                    ])
                ])
            ], md=3),
            
            # Center - Portfolio overview
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-wallet me-2"),
                        "Your Portfolio"
                    ]),
                    dbc.CardBody([
                        # Summary stats
                        dbc.Row([
                            dbc.Col([
                                self._create_stat_card("Total Value", "$125,430", "success", "+5.2%")
                            ], md=4),
                            dbc.Col([
                                self._create_stat_card("Today's Change", "+$1,240", "success", "+0.99%")
                            ], md=4),
                            dbc.Col([
                                self._create_stat_card("Total Gain", "+$25,430", "success", "+25.4%")
                            ], md=4)
                        ], className="mb-4"),
                        
                        # Simple pie chart
                        html.Div([
                            html.H5("Asset Allocation", className="mb-3"),
                            dcc.Graph(
                                figure=self._create_simple_pie_chart(),
                                config={'displayModeBar': False},
                                style={'height': '300px'}
                            )
                        ])
                    ])
                ])
            ], md=6),
            
            # Right panel - Help & tips
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-lightbulb me-2"),
                        "Tips & Guidance"
                    ]),
                    dbc.CardBody([
                        dbc.Alert([
                            html.H6("💡 Tip of the Day"),
                            html.P("Diversifying your portfolio across different asset classes can help reduce risk.", className="mb-0")
                        ], color="info"),
                        
                        html.H6("Getting Started", className="mt-3"),
                        dbc.Checklist(
                            options=[
                                {"label": "Set up your portfolio", "value": 1},
                                {"label": "Add your first investment", "value": 2},
                                {"label": "Review risk settings", "value": 3},
                                {"label": "Enable notifications", "value": 4}
                            ],
                            value=[1],
                            id="onboarding-checklist"
                        ),
                        
                        dbc.Button([
                            html.I(className="fas fa-graduation-cap me-2"),
                            "Start Tutorial"
                        ], color="primary", className="w-100 mt-3")
                    ])
                ])
            ], md=3)
        ])
    
    def _create_intermediate_content(self):
        """Creates intermediate interface with more features."""
        return dbc.Tabs([
            dbc.Tab(label="Dashboard", tab_id="dash-tab"),
            dbc.Tab(label="Portfolio", tab_id="portfolio-tab"),
            dbc.Tab(label="Performance", tab_id="performance-tab"),
            dbc.Tab(label="Risk Analysis", tab_id="risk-tab"),
            dbc.Tab(label="Strategies", tab_id="strategy-tab")
        ], id="main-tabs", active_tab="dash-tab")
    
    def _create_advanced_content(self):
        """Creates advanced interface with all features."""
        return html.Div([
            dbc.Tabs([
                dbc.Tab(label="Dashboard", tab_id="dash-tab"),
                dbc.Tab(label="Portfolio", tab_id="portfolio-tab"),
                dbc.Tab(label="Performance", tab_id="performance-tab"),
                dbc.Tab(label="Risk Analysis", tab_id="risk-tab"),
                dbc.Tab(label="Strategies", tab_id="strategy-tab"),
                dbc.Tab(label="Market Data", tab_id="market-tab"),
                dbc.Tab(label="Research", tab_id="research-tab"),
                dbc.Tab(label="AI Assistant", tab_id="ai-tab")
            ], id="main-tabs", active_tab="dash-tab"),
            html.Div(id="tab-content", className="mt-3")
        ])
    
    def _create_stat_card(self, title: str, value: str, color: str, change: str):
        """Creates a statistic card."""
        return dbc.Card([
            dbc.CardBody([
                html.H6(title, className="text-muted mb-2"),
                html.H4(value, className="mb-1"),
                dbc.Badge(change, color=color)
            ])
        ], className="text-center")
    
    def _create_simple_pie_chart(self):
        """Creates a simple pie chart for portfolio allocation."""
        fig = go.Figure(data=[go.Pie(
            labels=['Stocks', 'Bonds', 'Real Estate', 'Cash'],
            values=[45, 30, 15, 10],
            hole=.3,
            marker_colors=['#4CAF50', '#2196F3', '#FF9800', '#9E9E9E']
        )])
        
        fig.update_layout(
            showlegend=True,
            margin=dict(t=0, b=0, l=0, r=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig


class OnboardingWizard:
    """Step-by-step onboarding for new users."""
    
    def __init__(self):
        self.steps = [
            {
                'id': 'welcome',
                'title': 'Welcome to Investment Engine',
                'content': 'Let\'s get you started with a few simple steps.'
            },
            {
                'id': 'profile',
                'title': 'Create Your Profile',
                'content': 'Tell us about your investment goals and risk tolerance.'
            },
            {
                'id': 'portfolio',
                'title': 'Set Up Portfolio',
                'content': 'Choose a preset strategy or create your own.'
            },
            {
                'id': 'complete',
                'title': 'You\'re All Set!',
                'content': 'Your dashboard is ready. Start investing smarter today.'
            }
        ]
        self.current_step = 0
    
    def create_wizard(self):
        """Creates the onboarding wizard interface."""
        step = self.steps[self.current_step]
        
        return dbc.Modal([
            dbc.ModalHeader([
                html.H4(step['title']),
                dbc.Progress(
                    value=(self.current_step + 1) * 25,
                    className="mb-0 mt-2",
                    style={'height': '5px'}
                )
            ]),
            dbc.ModalBody([
                self._create_step_content(step['id'])
            ]),
            dbc.ModalFooter([
                dbc.Button(
                    "Previous",
                    id="wizard-prev",
                    disabled=self.current_step == 0,
                    color="secondary"
                ),
                dbc.Button(
                    "Next" if self.current_step < len(self.steps) - 1 else "Finish",
                    id="wizard-next",
                    color="primary"
                )
            ])
        ], id="onboarding-modal", size="lg", backdrop="static")
    
    def _create_step_content(self, step_id: str):
        """Creates content for each wizard step."""
        if step_id == 'welcome':
            return html.Div([
                html.Div([
                    html.I(className="fas fa-chart-line fa-5x text-primary mb-3")
                ], className="text-center"),
                html.P(
                    "Welcome to your personal investment management platform. "
                    "We'll help you build and manage your portfolio with confidence.",
                    className="lead text-center"
                )
            ])
        
        elif step_id == 'profile':
            return html.Div([
                dbc.Form([
                    dbc.FormGroup([
                        dbc.Label("Investment Experience"),
                        dbc.RadioItems(
                            options=[
                                {"label": "Beginner", "value": "beginner"},
                                {"label": "Intermediate", "value": "intermediate"},
                                {"label": "Advanced", "value": "advanced"}
                            ],
                            value="beginner",
                            id="experience-level"
                        )
                    ]),
                    dbc.FormGroup([
                        dbc.Label("Risk Tolerance"),
                        dbc.RadioItems(
                            options=[
                                {"label": "Conservative", "value": "conservative"},
                                {"label": "Moderate", "value": "moderate"},
                                {"label": "Aggressive", "value": "aggressive"}
                            ],
                            value="moderate",
                            id="risk-tolerance"
                        )
                    ]),
                    dbc.FormGroup([
                        dbc.Label("Investment Goal"),
                        dbc.Input(
                            placeholder="e.g., Retirement, Home purchase, Education",
                            id="investment-goal"
                        )
                    ])
                ])
            ])
        
        elif step_id == 'portfolio':
            return html.Div([
                html.H5("Choose a Starting Strategy", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Conservative"),
                                html.P("Low risk, steady returns", className="small"),
                                dbc.Button("Select", color="outline-primary", size="sm", block=True)
                            ])
                        ])
                    ], md=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Balanced"),
                                html.P("Mixed growth and income", className="small"),
                                dbc.Button("Select", color="outline-primary", size="sm", block=True)
                            ])
                        ])
                    ], md=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Growth"),
                                html.P("Higher risk, higher returns", className="small"),
                                dbc.Button("Select", color="outline-primary", size="sm", block=True)
                            ])
                        ])
                    ], md=4)
                ])
            ])
        
        else:  # complete
            return html.Div([
                html.Div([
                    html.I(className="fas fa-check-circle fa-5x text-success mb-3")
                ], className="text-center"),
                html.H5("Setup Complete!", className="text-center mb-3"),
                html.P(
                    "Your dashboard is ready. You can always adjust settings later.",
                    className="text-center"
                )
            ])
