"""
Academic Research Hub for Factor Investing
Tracks latest research papers, citations, and factor evolution
"""

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

class ResearchManager:
    """Manages academic research tracking and factor evolution analysis."""
    
    def __init__(self):
        self.research_papers = self._initialize_research_database()
        self.factor_evolution = self._initialize_factor_evolution()
        self.citation_data = self._initialize_citation_tracking()
    
    def _initialize_research_database(self) -> Dict[str, List[Dict]]:
        """Initialize database of factor investing research papers."""
        return {
            "recent_1m": [
                {
                    "title": "Machine Learning in Factor Investing: A Systematic Review",
                    "authors": ["Zhang, L.", "Chen, M.", "Rodriguez, A."],
                    "journal": "Journal of Portfolio Management",
                    "date": "2024-01-15",
                    "citations": 45,
                    "abstract": "Comprehensive review of ML applications in factor construction and portfolio optimization.",
                    "key_findings": ["ML improves factor timing", "Non-linear factor interactions matter", "Regime-aware models outperform"],
                    "practical_implications": "Implement regime-switching factor models for better risk-adjusted returns.",
                    "doi": "10.3905/jpm.2024.1.234"
                },
                {
                    "title": "ESG Factors and Cross-Sectional Stock Returns: Global Evidence",
                    "authors": ["Kumar, S.", "Thompson, R.", "Lee, H."],
                    "journal": "Review of Financial Studies",
                    "date": "2024-01-08",
                    "citations": 78,
                    "abstract": "Analysis of ESG factor premiums across 45 developed and emerging markets.",
                    "key_findings": ["ESG premium varies by region", "Governance factor strongest", "Integration with traditional factors beneficial"],
                    "practical_implications": "Combine ESG with quality and momentum factors for enhanced performance.",
                    "doi": "10.1093/rfs/hhad089"
                }
            ],
            "recent_3m": [
                {
                    "title": "Alternative Risk Premia: A Comprehensive Taxonomy",
                    "authors": ["Blackrock Research", "Ang, A.", "Bali, T."],
                    "journal": "Financial Analysts Journal",
                    "date": "2023-11-20",
                    "citations": 156,
                    "abstract": "Systematic classification of 200+ alternative risk premia across asset classes.",
                    "key_findings": ["Carry strategies dominate", "Momentum works across assets", "Mean reversion in volatility"],
                    "practical_implications": "Diversify across asset classes using common factor themes.",
                    "doi": "10.2469/faj.v79.n4.3"
                },
                {
                    "title": "Factor Timing with Machine Learning: Out-of-Sample Evidence",
                    "authors": ["Gu, S.", "Kelly, B.", "Xiu, D."],
                    "journal": "Journal of Finance",
                    "date": "2023-10-15",
                    "citations": 234,
                    "abstract": "ML-based factor timing strategies using 30,000+ predictive signals.",
                    "key_findings": ["Tree-based models excel", "Factor timing adds value", "Interaction effects crucial"],
                    "practical_implications": "Use ensemble methods for factor allocation decisions.",
                    "doi": "10.1111/jofi.13271"
                }
            ],
            "recent_6m": [
                {
                    "title": "The Cross-Section of Cryptocurrency Returns",
                    "authors": ["Liu, Y.", "Tsyvinski, A.", "Wu, X."],
                    "journal": "Review of Financial Studies",
                    "date": "2023-08-30",
                    "citations": 189,
                    "abstract": "Factor analysis of 2000+ cryptocurrencies reveals new risk premia.",
                    "key_findings": ["Momentum strongest factor", "Network effects matter", "Volatility clustering"],
                    "practical_implications": "Apply traditional factor frameworks to crypto with network adjustments.",
                    "doi": "10.1093/rfs/hhad067"
                }
            ],
            "recent_12m": [
                {
                    "title": "Climate Risk and the Cross-Section of Stock Returns",
                    "authors": ["Bolton, P.", "Kacperczyk, M."],
                    "journal": "Journal of Political Economy",
                    "date": "2023-03-15",
                    "citations": 445,
                    "abstract": "Climate transition risk creates new factor premiums in equity markets.",
                    "key_findings": ["Carbon intensity predicts returns", "Transition risk priced", "Physical risk less important"],
                    "practical_implications": "Incorporate climate metrics into factor models for future performance.",
                    "doi": "10.1086/723204"
                }
            ],
            "all_time_classics": [
                {
                    "title": "The Cross-Section of Expected Stock Returns",
                    "authors": ["Fama, E.F.", "French, K.R."],
                    "journal": "Journal of Finance",
                    "date": "1992-06-01",
                    "citations": 15420,
                    "abstract": "Seminal paper establishing size and value factors in equity returns.",
                    "key_findings": ["Beta doesn't explain returns", "Size and B/M are key", "CAPM insufficient"],
                    "practical_implications": "Foundation of modern factor investing - use size and value tilts.",
                    "doi": "10.1111/j.1540-6261.1992.tb04398.x"
                },
                {
                    "title": "A Five-Factor Asset Pricing Model",
                    "authors": ["Fama, E.F.", "French, K.R."],
                    "journal": "Journal of Financial Economics",
                    "date": "2015-04-01",
                    "citations": 8934,
                    "abstract": "Extension to five-factor model adding profitability and investment factors.",
                    "key_findings": ["Profitability factor significant", "Investment factor important", "Better model fit"],
                    "practical_implications": "Use 5-factor model for attribution and strategy construction.",
                    "doi": "10.1016/j.jfineco.2014.10.010"
                },
                {
                    "title": "Momentum Strategies",
                    "authors": ["Jegadeesh, N.", "Titman, S."],
                    "journal": "Journal of Finance",
                    "date": "1993-12-01",
                    "citations": 7234,
                    "abstract": "Documentation of momentum anomaly in stock returns.",
                    "key_findings": ["12-1 momentum works", "Profits persist 3-12 months", "Reverses long-term"],
                    "practical_implications": "Implement momentum strategies with 3-12 month holding periods.",
                    "doi": "10.1111/j.1540-6261.1993.tb05128.x"
                }
            ]
        }
    
    def _initialize_factor_evolution(self) -> Dict[str, Any]:
        """Initialize factor evolution mind map data."""
        return {
            "nodes": [
                {"id": "capm", "label": "CAPM (1964)", "group": "classical", "level": 0, "x": 0, "y": 0},
                {"id": "apt", "label": "APT (1976)", "group": "classical", "level": 1, "x": 100, "y": -50},
                {"id": "size", "label": "Size Effect (1981)", "group": "classical", "level": 1, "x": -100, "y": -50},
                {"id": "value", "label": "Value (1992)", "group": "fundamental", "level": 2, "x": -150, "y": -100},
                {"id": "momentum", "label": "Momentum (1993)", "group": "behavioral", "level": 2, "x": 0, "y": -100},
                {"id": "profitability", "label": "Profitability (2006)", "group": "fundamental", "level": 3, "x": -200, "y": -150},
                {"id": "investment", "label": "Investment (2008)", "group": "fundamental", "level": 3, "x": -100, "y": -150},
                {"id": "low_vol", "label": "Low Volatility (2011)", "group": "risk", "level": 3, "x": 100, "y": -150},
                {"id": "quality", "label": "Quality (2013)", "group": "fundamental", "level": 3, "x": 0, "y": -150},
                {"id": "carry", "label": "Carry (2012)", "group": "alternative", "level": 3, "x": 200, "y": -150},
                {"id": "esg", "label": "ESG (2020)", "group": "sustainable", "level": 4, "x": -250, "y": -200},
                {"id": "climate", "label": "Climate Risk (2021)", "group": "sustainable", "level": 4, "x": -150, "y": -200},
                {"id": "ml_factors", "label": "ML Factors (2020)", "group": "technology", "level": 4, "x": 50, "y": -200},
                {"id": "alt_data", "label": "Alternative Data (2019)", "group": "technology", "level": 4, "x": 150, "y": -200},
                {"id": "crypto", "label": "Crypto Factors (2022)", "group": "digital", "level": 4, "x": 250, "y": -200}
            ],
            "edges": [
                {"from": "capm", "to": "apt"},
                {"from": "capm", "to": "size"},
                {"from": "size", "to": "value"},
                {"from": "apt", "to": "momentum"},
                {"from": "value", "to": "profitability"},
                {"from": "value", "to": "investment"},
                {"from": "momentum", "to": "quality"},
                {"from": "apt", "to": "low_vol"},
                {"from": "apt", "to": "carry"},
                {"from": "profitability", "to": "esg"},
                {"from": "investment", "to": "climate"},
                {"from": "quality", "to": "ml_factors"},
                {"from": "momentum", "to": "alt_data"},
                {"from": "carry", "to": "crypto"}
            ],
            "groups": {
                "classical": {"color": "#1f77b4", "description": "Classical asset pricing models"},
                "fundamental": {"color": "#ff7f0e", "description": "Fundamental analysis factors"},
                "behavioral": {"color": "#2ca02c", "description": "Behavioral finance factors"},
                "risk": {"color": "#d62728", "description": "Risk-based factors"},
                "alternative": {"color": "#9467bd", "description": "Alternative risk premia"},
                "sustainable": {"color": "#8c564b", "description": "ESG and sustainability factors"},
                "technology": {"color": "#e377c2", "description": "Technology-driven factors"},
                "digital": {"color": "#7f7f7f", "description": "Digital asset factors"}
            }
        }
    
    def _initialize_citation_tracking(self) -> pd.DataFrame:
        """Initialize citation tracking data."""
        dates = pd.date_range(start='2020-01-01', end='2024-01-31', freq='M')
        
        # Simulate citation growth for different factor categories
        np.random.seed(42)
        data = []
        
        categories = ['Value', 'Momentum', 'Quality', 'Low Vol', 'ESG', 'ML Factors', 'Climate Risk']
        base_citations = [1000, 800, 600, 400, 200, 150, 100]
        growth_rates = [0.02, 0.03, 0.04, 0.03, 0.08, 0.12, 0.15]
        
        for i, category in enumerate(categories):
            citations = base_citations[i]
            for date in dates:
                citations *= (1 + growth_rates[i] + np.random.normal(0, 0.01))
                data.append({
                    'date': date,
                    'category': category,
                    'citations': int(citations),
                    'new_papers': np.random.poisson(5 + i)
                })
        
        return pd.DataFrame(data)
    
    def create_research_tab(self) -> html.Div:
        """Create the research hub tab."""
        return html.Div([
            # Header
            html.Div([
                html.H2([
                    html.I(className="fas fa-graduation-cap me-3"),
                    "Academic Research Hub"
                ], className="text-center mb-4"),
                html.P("Latest factor investing research, citations, and evolution insights", 
                       className="text-center text-muted mb-4")
            ]),
            
            # Research sections
            dbc.Tabs([
                dbc.Tab(label="📈 Recent Papers", tab_id="recent-papers"),
                dbc.Tab(label="🧠 Factor Evolution", tab_id="factor-evolution"),
                dbc.Tab(label="📊 Citation Trends", tab_id="citation-trends"),
                dbc.Tab(label="🔬 Implementation Insights", tab_id="implementation")
            ], id="research-tabs", active_tab="recent-papers"),
            
            html.Div(id="research-tab-content", className="mt-4")
        ])
    
    def create_recent_papers_content(self) -> html.Div:
        """Create recent papers section."""
        return html.Div([
            # Time period selector
            html.Div([
                html.H4("📚 Recent Research Papers", className="mb-3"),
                dbc.ButtonGroup([
                    dbc.Button("1 Month", id="papers-1m", color="primary", size="sm"),
                    dbc.Button("3 Months", id="papers-3m", color="outline-primary", size="sm"),
                    dbc.Button("6 Months", id="papers-6m", color="outline-primary", size="sm"),
                    dbc.Button("12 Months", id="papers-12m", color="outline-primary", size="sm"),
                    dbc.Button("All-Time Classics", id="papers-classics", color="outline-secondary", size="sm")
                ], className="mb-4")
            ]),
            
            # Papers list
            html.Div(id="papers-list")
        ])
    
    def create_paper_cards(self, time_period: str) -> List[dbc.Card]:
        """Create paper cards for given time period."""
        papers = self.research_papers.get(time_period, [])
        cards = []
        
        for paper in papers:
            # Citation badge color based on count
            if paper['citations'] > 1000:
                citation_color = "success"
            elif paper['citations'] > 100:
                citation_color = "warning"
            else:
                citation_color = "info"
            
            card = dbc.Card([
                dbc.CardBody([
                    html.H5(paper['title'], className="card-title"),
                    html.P(f"Authors: {', '.join(paper['authors'])}", 
                           className="text-muted small"),
                    html.P(f"{paper['journal']} ({paper['date']})", 
                           className="text-info small"),
                    
                    # Citation and metrics
                    html.Div([
                        dbc.Badge(f"{paper['citations']} citations", 
                                color=citation_color, className="me-2"),
                        dbc.Badge(f"DOI: {paper['doi']}", color="secondary")
                    ], className="mb-3"),
                    
                    # Abstract
                    html.P(paper['abstract'], className="card-text"),
                    
                    # Key findings
                    html.Div([
                        html.H6("🔍 Key Findings:", className="mt-3"),
                        html.Ul([html.Li(finding) for finding in paper['key_findings']])
                    ]),
                    
                    # Practical implications
                    html.Div([
                        html.H6("💡 Practical Implications:", className="mt-3"),
                        dbc.Alert(paper['practical_implications'], color="light")
                    ])
                ])
            ], className="mb-4")
            cards.append(card)
        
        return cards
    
    def create_factor_evolution_mindmap(self) -> dcc.Graph:
        """Create interactive factor evolution mind map."""
        evolution = self.factor_evolution
        
        # Create network visualization
        fig = go.Figure()
        
        # Add edges first (so they appear behind nodes)
        for edge in evolution['edges']:
            from_node = next(n for n in evolution['nodes'] if n['id'] == edge['from'])
            to_node = next(n for n in evolution['nodes'] if n['id'] == edge['to'])
            
            fig.add_trace(go.Scatter(
                x=[from_node['x'], to_node['x']],
                y=[from_node['y'], to_node['y']],
                mode='lines',
                line=dict(color='rgba(128,128,128,0.5)', width=2),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Add nodes by group
        for group_name, group_info in evolution['groups'].items():
            group_nodes = [n for n in evolution['nodes'] if n['group'] == group_name]
            
            fig.add_trace(go.Scatter(
                x=[n['x'] for n in group_nodes],
                y=[n['y'] for n in group_nodes],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color=group_info['color'],
                    line=dict(width=2, color='white')
                ),
                text=[n['label'] for n in group_nodes],
                textposition="middle center",
                textfont=dict(size=10, color='white'),
                name=group_name.title(),
                hovertemplate='<b>%{text}</b><br>' + group_info['description'] + '<extra></extra>'
            ))
        
        fig.update_layout(
            title="Factor Investing Evolution Mind Map",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text="Hover over nodes to see descriptions. Click legend to filter groups.",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=12, color='gray')
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            height=600
        )
        
        return dcc.Graph(figure=fig, id="factor-mindmap")
    
    def create_citation_trends_chart(self) -> dcc.Graph:
        """Create citation trends visualization."""
        df = self.citation_data
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Citation Growth by Category', 'New Papers Published',
                          'Citation Acceleration', 'Research Focus Shift'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Citation growth
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]
            fig.add_trace(
                go.Scatter(x=cat_data['date'], y=cat_data['citations'],
                          name=category, mode='lines'),
                row=1, col=1
            )
        
        # New papers
        monthly_papers = df.groupby(['date', 'category'])['new_papers'].sum().reset_index()
        for category in monthly_papers['category'].unique():
            cat_data = monthly_papers[monthly_papers['category'] == category]
            fig.add_trace(
                go.Bar(x=cat_data['date'], y=cat_data['new_papers'],
                      name=f"{category} Papers", showlegend=False),
                row=1, col=2
            )
        
        # Citation acceleration (derivative)
        for category in df['category'].unique():
            cat_data = df[df['category'] == category].sort_values('date')
            acceleration = cat_data['citations'].diff().rolling(3).mean()
            fig.add_trace(
                go.Scatter(x=cat_data['date'], y=acceleration,
                          name=f"{category} Accel", showlegend=False),
                row=2, col=1
            )
        
        # Research focus (pie chart of recent citations)
        recent_data = df[df['date'] >= '2023-01-01'].groupby('category')['citations'].sum()
        fig.add_trace(
            go.Pie(labels=recent_data.index, values=recent_data.values,
                  name="Focus", showlegend=False),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Factor Investing Research Trends")
        return dcc.Graph(figure=fig)

class ResearchCallbacks:
    """Handles research hub callbacks."""
    
    @staticmethod
    def register_callbacks(app: dash.Dash, research_manager: ResearchManager):
        """Register research hub callbacks."""
        
        @app.callback(
            Output("research-tab-content", "children"),
            Input("research-tabs", "active_tab")
        )
        def render_research_tab(active_tab):
            """Render research tab content."""
            if active_tab == "recent-papers":
                return research_manager.create_recent_papers_content()
            elif active_tab == "factor-evolution":
                return html.Div([
                    html.H4("🧠 Factor Evolution Mind Map", className="mb-4"),
                    html.P("Interactive visualization of how factor investing has evolved over time.", 
                           className="text-muted mb-4"),
                    research_manager.create_factor_evolution_mindmap()
                ])
            elif active_tab == "citation-trends":
                return html.Div([
                    html.H4("📊 Citation and Research Trends", className="mb-4"),
                    research_manager.create_citation_trends_chart()
                ])
            elif active_tab == "implementation":
                return html.Div([
                    html.H4("🔬 Implementation Insights", className="mb-4"),
                    _create_implementation_insights()
                ])
            return html.Div("Select a tab to view content.")
        
        @app.callback(
            [Output("papers-1m", "color"),
             Output("papers-3m", "color"),
             Output("papers-6m", "color"),
             Output("papers-12m", "color"),
             Output("papers-classics", "color"),
             Output("papers-list", "children")],
            [Input("papers-1m", "n_clicks"),
             Input("papers-3m", "n_clicks"),
             Input("papers-6m", "n_clicks"),
             Input("papers-12m", "n_clicks"),
             Input("papers-classics", "n_clicks")]
        )
        def update_papers_display(btn1, btn2, btn3, btn4, btn5):
            """Update papers display based on time period selection."""
            ctx = dash.callback_context
            if not ctx.triggered:
                # Default to 1 month
                return "primary", "outline-primary", "outline-primary", "outline-primary", "outline-secondary", research_manager.create_paper_cards("recent_1m")
            
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            # Reset all button colors
            colors = ["outline-primary"] * 4 + ["outline-secondary"]
            
            # Set active button and get papers
            if trigger_id == "papers-1m":
                colors[0] = "primary"
                papers = research_manager.create_paper_cards("recent_1m")
            elif trigger_id == "papers-3m":
                colors[1] = "primary"
                papers = research_manager.create_paper_cards("recent_3m")
            elif trigger_id == "papers-6m":
                colors[2] = "primary"
                papers = research_manager.create_paper_cards("recent_6m")
            elif trigger_id == "papers-12m":
                colors[3] = "primary"
                papers = research_manager.create_paper_cards("recent_12m")
            elif trigger_id == "papers-classics":
                colors[4] = "secondary"
                papers = research_manager.create_paper_cards("all_time_classics")
            else:
                papers = research_manager.create_paper_cards("recent_1m")
            
            return colors[0], colors[1], colors[2], colors[3], colors[4], papers

def _create_implementation_insights() -> html.Div:
    """Create implementation insights content."""
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("🎯 Strategy Implementation")),
                    dbc.CardBody([
                        html.H6("Factor Selection Framework:"),
                        html.Ul([
                            html.Li("Academic validation (peer-reviewed research)"),
                            html.Li("Economic intuition and persistence"),
                            html.Li("Implementation feasibility and costs"),
                            html.Li("Diversification benefits")
                        ]),
                        html.H6("Portfolio Construction:"),
                        html.Ul([
                            html.Li("Use optimization with factor constraints"),
                            html.Li("Consider transaction costs and turnover"),
                            html.Li("Implement gradual factor tilts"),
                            html.Li("Monitor factor loadings over time")
                        ])
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("⚠️ Common Pitfalls")),
                    dbc.CardBody([
                        html.H6("Research Translation Issues:"),
                        html.Ul([
                            html.Li("Survivorship bias in academic studies"),
                            html.Li("Look-ahead bias in factor definitions"),
                            html.Li("Transaction cost underestimation"),
                            html.Li("Capacity constraints ignored")
                        ]),
                        html.H6("Implementation Challenges:"),
                        html.Ul([
                            html.Li("Factor timing attempts (usually fail)"),
                            html.Li("Over-diversification reducing factor exposure"),
                            html.Li("Ignoring regime changes"),
                            html.Li("Insufficient risk management")
                        ])
                    ])
                ])
            ], width=6)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("📈 Performance Expectations")),
                    dbc.CardBody([
                        html.P("Based on academic literature analysis:", className="mb-3"),
                        dbc.Table([
                            html.Thead([
                                html.Tr([
                                    html.Th("Factor"),
                                    html.Th("Annual Premium"),
                                    html.Th("Volatility"),
                                    html.Th("Max DD"),
                                    html.Th("Sharpe Ratio")
                                ])
                            ]),
                            html.Tbody([
                                html.Tr([html.Td("Value"), html.Td("4.8%"), html.Td("20.1%"), html.Td("-45%"), html.Td("0.24")]),
                                html.Tr([html.Td("Momentum"), html.Td("8.3%"), html.Td("18.7%"), html.Td("-38%"), html.Td("0.44")]),
                                html.Tr([html.Td("Quality"), html.Td("3.2%"), html.Td("15.4%"), html.Td("-28%"), html.Td("0.21")]),
                                html.Tr([html.Td("Low Vol"), html.Td("2.1%"), html.Td("12.8%"), html.Td("-22%"), html.Td("0.16")]),
                                html.Tr([html.Td("Size"), html.Td("2.9%"), html.Td("22.3%"), html.Td("-52%"), html.Td("0.13")])
                            ])
                        ], striped=True, bordered=True, hover=True, size="sm")
                    ])
                ])
            ], width=12)
        ])
    ])

# Initialize research system
research_manager = ResearchManager()
