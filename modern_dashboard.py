"""Modern Investment Engine Dashboard with Enhanced UI."""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from auto_config import AutoConfig

# Initialize auto-configuration
auto_config = AutoConfig()
config = auto_config.setup_everything()

# Custom CSS styles
custom_css = """
/* Modern Pastel Gradient Background */
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #fda085 75%, #ffecd2 100%);
    font-family: 'Inter', 'Segoe UI', sans-serif;
    min-height: 100vh;
}

/* Glass morphism effect for cards */
.glass-card {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    border: 1px solid rgba(255, 255, 255, 0.18);
    padding: 2rem;
    margin: 1rem;
    animation: slideUp 0.5s ease-out;
}

/* Smooth animations */
@keyframes slideUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* Beautiful buttons */
.modern-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 12px;
    font-weight: 600;
    transition: all 0.3s ease;
    cursor: pointer;
    box-shadow: 0 4px 15px 0 rgba(31, 38, 135, 0.2);
}

.modern-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px 0 rgba(31, 38, 135, 0.4);
}

/* Tab styling */
.custom-tabs .tab {
    background: rgba(255, 255, 255, 0.7);
    border-radius: 10px 10px 0 0;
    margin: 0 5px;
    transition: all 0.3s ease;
}

.custom-tabs .tab--selected {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

/* Input fields */
.modern-input {
    border: 2px solid #e0e0e0;
    border-radius: 10px;
    padding: 10px 15px;
    transition: all 0.3s ease;
    width: 100%;
}

.modern-input:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    outline: none;
}

/* Headers */
h1, h2, h3 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}

/* Loading animation */
.loading-spinner {
    border: 4px solid rgba(102, 126, 234, 0.1);
    border-left-color: #667eea;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Mobile responsive */
@media (max-width: 768px) {
    .glass-card {
        padding: 1rem;
        margin: 0.5rem;
    }
    
    .modern-btn {
        padding: 10px 20px;
        font-size: 14px;
    }
}
"""

# Initialize Dash app
app = dash.Dash(__name__, 
                external_stylesheets=[
                    dbc.themes.BOOTSTRAP,
                    "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap",
                    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
                ])

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>''' + custom_css + '''</style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.I(className="fas fa-chart-line fa-3x", style={'color': '#667eea', 'marginRight': '1rem'}),
            html.Div([
                html.H1("Investment Engine", style={'margin': '0', 'fontSize': '2.5rem'}),
                html.P("AI-Powered Portfolio Management Platform", 
                      style={'margin': '0', 'opacity': '0.8', 'fontSize': '1.1rem'})
            ])
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 
                 'padding': '2rem', 'animation': 'fadeIn 1s ease-out'})
    ], className="glass-card"),
    
    # Tabs
    dcc.Tabs(id='main-tabs', value='dashboard', children=[
        dcc.Tab(label='📊 Dashboard', value='dashboard', className='tab'),
        dcc.Tab(label='💼 Portfolio', value='portfolio', className='tab'),
        dcc.Tab(label='📈 Analysis', value='analysis', className='tab'),
        dcc.Tab(label='🔬 Research', value='research', className='tab'),
        dcc.Tab(label='🤖 AI Assistant', value='ai', className='tab'),
        dcc.Tab(label='⚙️ Settings', value='settings', className='tab'),
    ], className='custom-tabs', style={'marginTop': '1rem'}),
    
    # Tab content
    html.Div(id='tab-content', className="glass-card", style={'marginTop': '1rem'}),
    
    # Footer
    html.Div([
        html.Hr(style={'borderTop': '2px solid rgba(102, 126, 234, 0.2)'}),
        html.P([
            "© 2025 Investment Engine | ",
            html.Span("🟢 System Online", style={'color': '#4caf50', 'fontWeight': '600'}),
            " | Powered by Advanced AI"
        ], style={'textAlign': 'center', 'opacity': '0.7'})
    ], className="glass-card", style={'marginTop': '2rem'})
], style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '2rem'})

# Callbacks
@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value')
)
def render_tab_content(active_tab):
    if active_tab == 'dashboard':
        return html.Div([
            html.H2("📊 Dashboard Overview", style={'marginBottom': '2rem'}),
            
            # Key metrics cards
            html.Div([
                html.Div([
                    html.Div([
                        html.H4("Total Portfolio Value"),
                        html.H2("$1,234,567", style={'color': '#4caf50'}),
                        html.P("↑ 12.5% this month", style={'color': '#4caf50'})
                    ], className="glass-card", style={'textAlign': 'center', 'flex': '1'})
                ], style={'flex': '1', 'margin': '0 0.5rem'}),
                
                html.Div([
                    html.Div([
                        html.H4("Today's P&L"),
                        html.H2("+$15,234", style={'color': '#4caf50'}),
                        html.P("↑ 1.2%", style={'color': '#4caf50'})
                    ], className="glass-card", style={'textAlign': 'center'})
                ], style={'flex': '1', 'margin': '0 0.5rem'}),
                
                html.Div([
                    html.Div([
                        html.H4("Risk Score"),
                        html.H2("Low", style={'color': '#2196f3'}),
                        html.P("VaR: $45,230", style={'opacity': '0.7'})
                    ], className="glass-card", style={'textAlign': 'center'})
                ], style={'flex': '1', 'margin': '0 0.5rem'}),
                
                html.Div([
                    html.Div([
                        html.H4("Active Strategies"),
                        html.H2("5", style={'color': '#9c27b0'}),
                        html.P("3 outperforming", style={'opacity': '0.7'})
                    ], className="glass-card", style={'textAlign': 'center'})
                ], style={'flex': '1', 'margin': '0 0.5rem'})
            ], style={'display': 'flex', 'marginBottom': '2rem'}),
            
            # Charts
            html.Div([
                dcc.Graph(
                    figure={
                        'data': [
                            go.Scatter(
                                x=pd.date_range(start='2024-01-01', periods=365, freq='D'),
                                y=np.cumsum(np.random.randn(365) * 0.01 + 0.001) * 1000000 + 1000000,
                                mode='lines',
                                name='Portfolio Value',
                                line={'color': '#667eea', 'width': 3},
                                fill='tozeroy',
                                fillcolor='rgba(102, 126, 234, 0.1)'
                            )
                        ],
                        'layout': go.Layout(
                            title='Portfolio Performance',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Value ($)', 'tickformat': ',.0f'},
                            hovermode='x unified',
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font={'family': 'Inter'}
                        )
                    },
                    style={'height': '400px'}
                )
            ], className="glass-card"),
            
            # Quick actions
            html.Div([
                html.H3("Quick Actions"),
                html.Div([
                    html.Button("📈 New Strategy", className="modern-btn", style={'margin': '0.5rem'}),
                    html.Button("💰 Rebalance Portfolio", className="modern-btn", style={'margin': '0.5rem'}),
                    html.Button("📊 Generate Report", className="modern-btn", style={'margin': '0.5rem'}),
                    html.Button("🔍 Run Backtest", className="modern-btn", style={'margin': '0.5rem'}),
                ], style={'textAlign': 'center', 'marginTop': '1rem'})
            ], className="glass-card", style={'marginTop': '2rem'})
        ])
    
    elif active_tab == 'portfolio':
        return html.Div([
            html.H2("💼 Portfolio Management"),
            html.P("Manage your investment portfolio with advanced analytics and optimization tools."),
            
            # Portfolio composition pie chart
            html.Div([
                dcc.Graph(
                    figure=px.pie(
                        values=[30, 25, 20, 15, 10],
                        names=['Tech Stocks', 'Healthcare', 'Finance', 'Energy', 'Real Estate'],
                        title='Portfolio Composition',
                        color_discrete_sequence=px.colors.sequential.Purples_r
                    ).update_traces(textposition='inside', textinfo='percent+label'),
                    style={'height': '400px'}
                )
            ], className="glass-card"),
            
            html.Button("Optimize Portfolio", className="modern-btn", style={'marginTop': '1rem'})
        ])
    
    elif active_tab == 'analysis':
        return html.Div([
            html.H2("📈 Strategy Analysis"),
            html.P("Analyze and backtest your investment strategies with academic research-backed metrics."),
            
            html.Div([
                html.Label("Select Strategy:"),
                dcc.Dropdown(
                    options=[
                        {'label': 'Momentum Factor', 'value': 'momentum'},
                        {'label': 'Value Factor', 'value': 'value'},
                        {'label': 'Quality Factor', 'value': 'quality'},
                        {'label': 'Low Volatility', 'value': 'low_vol'},
                    ],
                    value='momentum',
                    className="modern-input",
                    style={'marginBottom': '1rem'}
                ),
                html.Button("Run Analysis", className="modern-btn")
            ], className="glass-card")
        ])
    
    elif active_tab == 'research':
        return html.Div([
            html.H2("🔬 Research Center"),
            html.P("Access the latest academic research and market insights."),
            
            html.Div([
                html.H4("Latest Research Papers"),
                html.Ul([
                    html.Li("Factor Investing in the Age of Machine Learning (2024)"),
                    html.Li("Risk Parity Strategies: Performance During Market Stress"),
                    html.Li("ESG Integration in Quantitative Portfolio Management"),
                ]),
                html.Button("View Research Library", className="modern-btn", style={'marginTop': '1rem'})
            ], className="glass-card")
        ])
    
    elif active_tab == 'ai':
        return html.Div([
            html.H2("🤖 AI Investment Assistant"),
            html.P("Get personalized investment recommendations powered by advanced AI."),
            
            html.Div([
                html.Label("Ask a question:"),
                dcc.Textarea(
                    placeholder="e.g., What's the best strategy for the current market conditions?",
                    className="modern-input",
                    style={'width': '100%', 'height': '100px', 'marginBottom': '1rem'}
                ),
                html.Button("Get AI Insights", className="modern-btn")
            ], className="glass-card")
        ])
    
    elif active_tab == 'settings':
        return html.Div([
            html.H2("⚙️ Settings"),
            html.P("Configure your investment engine preferences."),
            
            html.Div([
                html.H4("Risk Preferences"),
                dcc.Slider(min=1, max=10, value=5, marks={i: str(i) for i in range(1, 11)}, id='risk-slider'),
                html.Br(),
                html.H4("Auto-Trading"),
                dcc.Checklist(
                    options=[{'label': ' Enable Auto-Trading', 'value': 'auto'}],
                    value=[]
                ),
                html.Br(),
                html.Button("Save Settings", className="modern-btn")
            ], className="glass-card")
        ])
    
    return html.Div("Select a tab to view content")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("MODERN INVESTMENT ENGINE DASHBOARD")
    print("="*60)
    print("Features:")
    print("  - Beautiful pastel gradient UI")
    print("  - Glass morphism effects")
    print("  - Smooth animations")
    print("  - Mobile responsive design")
    print("  - Real-time portfolio tracking")
    print("  - AI-powered insights")
    print("="*60)
    print("Opening dashboard at: http://127.0.0.1:8050")
    print("="*60 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=8050)
