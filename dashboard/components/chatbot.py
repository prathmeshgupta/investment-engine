"""
AI Chatbot for Strategy Deployment Assistance
Provides intelligent guidance for investment strategy implementation
"""

import dash
from dash import html, dcc, callback, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

class ChatbotManager:
    """Manages AI chatbot functionality for strategy deployment assistance."""
    
    def __init__(self):
        self.conversation_history = []
        self.strategy_templates = self._initialize_strategy_templates()
        self.deployment_steps = self._initialize_deployment_steps()
    
    def _initialize_strategy_templates(self) -> Dict[str, Dict]:
        """Initialize pre-built strategy templates."""
        return {
            "momentum": {
                "name": "Momentum Strategy",
                "description": "Cross-sectional momentum based on 12-1 month returns",
                "factors": ["momentum_12_1", "size", "value"],
                "rebalance_frequency": "monthly",
                "universe": "large_cap",
                "risk_budget": 0.15,
                "implementation_complexity": "medium"
            },
            "value": {
                "name": "Value Strategy",
                "description": "Multi-factor value strategy using P/E, P/B, and EV/EBITDA",
                "factors": ["book_to_market", "earnings_yield", "enterprise_multiple"],
                "rebalance_frequency": "quarterly",
                "universe": "all_cap",
                "risk_budget": 0.12,
                "implementation_complexity": "low"
            },
            "quality": {
                "name": "Quality Strategy",
                "description": "High-quality companies with strong fundamentals",
                "factors": ["profitability", "leverage", "earnings_quality"],
                "rebalance_frequency": "quarterly",
                "universe": "large_cap",
                "risk_budget": 0.10,
                "implementation_complexity": "medium"
            },
            "low_volatility": {
                "name": "Low Volatility Strategy",
                "description": "Minimum variance portfolio construction",
                "factors": ["volatility", "beta", "idiosyncratic_risk"],
                "rebalance_frequency": "monthly",
                "universe": "all_cap",
                "risk_budget": 0.08,
                "implementation_complexity": "high"
            }
        }
    
    def _initialize_deployment_steps(self) -> List[Dict]:
        """Initialize strategy deployment workflow steps."""
        return [
            {
                "step": 1,
                "title": "Strategy Selection",
                "description": "Choose or customize your investment strategy",
                "actions": ["Select template", "Define objectives", "Set constraints"]
            },
            {
                "step": 2,
                "title": "Universe Definition",
                "description": "Define your investment universe",
                "actions": ["Select asset classes", "Apply filters", "Set exclusions"]
            },
            {
                "step": 3,
                "title": "Factor Configuration",
                "description": "Configure factor exposures and weights",
                "actions": ["Select factors", "Set weights", "Define tilts"]
            },
            {
                "step": 4,
                "title": "Risk Management",
                "description": "Set risk controls and position limits",
                "actions": ["Set risk budget", "Define limits", "Configure stops"]
            },
            {
                "step": 5,
                "title": "Backtesting",
                "description": "Test strategy performance historically",
                "actions": ["Run backtest", "Analyze results", "Optimize parameters"]
            },
            {
                "step": 6,
                "title": "Implementation",
                "description": "Deploy strategy in live environment",
                "actions": ["Set up execution", "Monitor performance", "Rebalance schedule"]
            }
        ]
    
    def create_chatbot_interface(self) -> html.Div:
        """Create the chatbot interface component."""
        return html.Div([
            # Chatbot toggle button
            dbc.Button([
                html.I(className="fas fa-robot me-2"),
                "Strategy Assistant"
            ], id="chatbot-toggle", color="primary", size="sm", 
            className="position-fixed", 
            style={"bottom": "80px", "right": "20px", "z-index": "1050"}),
            
            # Chatbot panel
            dbc.Offcanvas([
                # Header
                html.Div([
                    html.H5([
                        html.I(className="fas fa-robot me-2"),
                        "Strategy Deployment Assistant"
                    ], className="mb-0"),
                    html.Small("AI-powered guidance for investment strategies", 
                             className="text-muted")
                ], className="mb-3"),
                
                # Quick actions
                html.Div([
                    html.H6("Quick Actions", className="mb-2"),
                    dbc.ButtonGroup([
                        dbc.Button("New Strategy", id="chatbot-new-strategy", 
                                 size="sm", outline=True),
                        dbc.Button("Optimize", id="chatbot-optimize", 
                                 size="sm", outline=True),
                        dbc.Button("Deploy", id="chatbot-deploy", 
                                 size="sm", outline=True)
                    ], className="mb-3")
                ]),
                
                # Chat history
                html.Div([
                    html.H6("Conversation", className="mb-2"),
                    html.Div(id="chatbot-messages", 
                           className="chat-history mb-3",
                           style={"height": "300px", "overflow-y": "auto",
                                  "border": "1px solid #dee2e6", "padding": "10px",
                                  "border-radius": "5px"})
                ]),
                
                # Input area
                html.Div([
                    dbc.InputGroup([
                        dbc.Input(id="chatbot-input", placeholder="Ask about strategy deployment...",
                                type="text"),
                        dbc.Button([html.I(className="fas fa-paper-plane")], 
                                 id="chatbot-send", color="primary")
                    ])
                ]),
                
                # Strategy templates
                html.Div([
                    html.Hr(),
                    html.H6("Strategy Templates", className="mb-2"),
                    html.Div(id="chatbot-templates")
                ])
                
            ], id="chatbot-panel", title="Strategy Assistant", is_open=False, 
            placement="end", style={"width": "400px"})
        ])
    
    def create_strategy_template_cards(self) -> List[dbc.Card]:
        """Create strategy template cards."""
        cards = []
        for key, template in self.strategy_templates.items():
            card = dbc.Card([
                dbc.CardBody([
                    html.H6(template["name"], className="card-title"),
                    html.P(template["description"], className="card-text small"),
                    html.Div([
                        dbc.Badge(f"Risk: {template['risk_budget']:.1%}", 
                                color="info", className="me-1"),
                        dbc.Badge(template["implementation_complexity"].title(), 
                                color="secondary")
                    ]),
                    dbc.Button("Select", id={"type": "template-select", "index": key},
                             size="sm", color="outline-primary", className="mt-2")
                ])
            ], className="mb-2")
            cards.append(card)
        return cards
    
    def process_user_message(self, message: str, context: Dict = None) -> str:
        """Process user message and generate AI response."""
        message_lower = message.lower()
        
        # Intent recognition
        if any(word in message_lower for word in ["new", "create", "build", "strategy"]):
            return self._handle_strategy_creation(message, context)
        elif any(word in message_lower for word in ["optimize", "improve", "enhance"]):
            return self._handle_optimization(message, context)
        elif any(word in message_lower for word in ["deploy", "implement", "execute"]):
            return self._handle_deployment(message, context)
        elif any(word in message_lower for word in ["risk", "volatility", "drawdown"]):
            return self._handle_risk_questions(message, context)
        elif any(word in message_lower for word in ["backtest", "performance", "returns"]):
            return self._handle_performance_questions(message, context)
        else:
            return self._handle_general_question(message, context)
    
    def _handle_strategy_creation(self, message: str, context: Dict) -> str:
        """Handle strategy creation requests."""
        return """I can help you create a new investment strategy! Here's what we'll need to define:

1. **Investment Objective**: What are you trying to achieve? (growth, income, risk reduction)
2. **Time Horizon**: Short-term (< 1 year), medium-term (1-5 years), or long-term (> 5 years)
3. **Risk Tolerance**: Conservative, moderate, or aggressive
4. **Factor Preferences**: Value, momentum, quality, size, low volatility

Would you like to start with one of our pre-built templates or create a custom strategy from scratch?"""
    
    def _handle_optimization(self, message: str, context: Dict) -> str:
        """Handle optimization requests."""
        return """I can help optimize your strategy! Here are the key areas we can improve:

**Portfolio Construction:**
- Factor weight optimization using mean-variance or Black-Litterman
- Risk parity or equal-weight alternatives
- Sector and geographic constraints

**Risk Management:**
- Dynamic position sizing based on volatility
- Drawdown controls and stop-loss rules
- Correlation-based diversification

**Execution:**
- Rebalancing frequency optimization
- Transaction cost minimization
- Tax-efficient implementation

What specific aspect would you like to optimize first?"""
    
    def _handle_deployment(self, message: str, context: Dict) -> str:
        """Handle deployment requests."""
        return """Ready to deploy your strategy? Here's the deployment checklist:

**Pre-Deployment:**
✓ Strategy backtesting completed
✓ Risk parameters validated
✓ Execution infrastructure ready
✓ Compliance approvals obtained

**Deployment Steps:**
1. Paper trading validation (recommended 30 days)
2. Gradual capital allocation (start with 10-25%)
3. Performance monitoring setup
4. Rebalancing schedule activation

**Post-Deployment:**
- Daily risk monitoring
- Weekly performance review
- Monthly strategy evaluation

Would you like me to guide you through any of these steps?"""
    
    def _handle_risk_questions(self, message: str, context: Dict) -> str:
        """Handle risk-related questions."""
        return """Risk management is crucial for successful investing. Here are key considerations:

**Risk Metrics to Monitor:**
- Value at Risk (VaR) at 95% and 99% confidence levels
- Maximum drawdown and recovery time
- Volatility clustering and regime changes
- Factor exposure drift

**Risk Controls:**
- Position limits (typically 2-5% per security)
- Sector concentration limits (max 20-30%)
- Geographic diversification requirements
- Leverage constraints

**Dynamic Risk Management:**
- Volatility targeting (adjust exposure based on realized vol)
- Correlation monitoring (reduce exposure when correlations spike)
- Tail risk hedging strategies

What specific risk aspect concerns you most?"""
    
    def _handle_performance_questions(self, message: str, context: Dict) -> str:
        """Handle performance-related questions."""
        return """Performance evaluation should be comprehensive and risk-adjusted:

**Key Performance Metrics:**
- Sharpe Ratio (risk-adjusted returns)
- Information Ratio (active return efficiency)
- Maximum Drawdown and recovery time
- Sortino Ratio (downside risk focus)

**Factor Attribution Analysis:**
- Fama-French 5-factor model attribution
- Custom factor exposure analysis
- Style drift monitoring
- Benchmark-relative performance

**Performance Persistence:**
- Rolling window analysis
- Regime-dependent performance
- Market cycle performance
- Out-of-sample validation

Would you like me to run a performance analysis on your current strategy?"""
    
    def _handle_general_question(self, message: str, context: Dict) -> str:
        """Handle general questions."""
        return """I'm here to help with all aspects of quantitative investment strategy development and deployment.

**I can assist with:**
- Strategy design and factor selection
- Portfolio optimization and risk management
- Backtesting and performance analysis
- Implementation and execution planning
- Academic research integration
- Regulatory and compliance considerations

**Popular topics:**
- "How do I build a momentum strategy?"
- "What's the optimal rebalancing frequency?"
- "How do I control portfolio risk?"
- "What factors work best in current markets?"

What specific question do you have about investment strategies?"""

class ChatbotCallbacks:
    """Handles chatbot-related callbacks."""
    
    @staticmethod
    def register_callbacks(app: dash.Dash, chatbot_manager: ChatbotManager):
        """Register all chatbot callbacks."""
        
        @app.callback(
            Output("chatbot-panel", "is_open"),
            Input("chatbot-toggle", "n_clicks"),
            State("chatbot-panel", "is_open")
        )
        def toggle_chatbot(n_clicks, is_open):
            """Toggle chatbot panel."""
            if n_clicks:
                return not is_open
            return is_open
        
        @app.callback(
            Output("chatbot-templates", "children"),
            Input("chatbot-panel", "is_open")
        )
        def load_strategy_templates(is_open):
            """Load strategy templates when panel opens."""
            if is_open:
                return chatbot_manager.create_strategy_template_cards()
            return []
        
        @app.callback(
            [Output("chatbot-messages", "children"),
             Output("chatbot-input", "value")],
            [Input("chatbot-send", "n_clicks"),
             Input("chatbot-input", "n_submit"),
             Input("chatbot-new-strategy", "n_clicks"),
             Input("chatbot-optimize", "n_clicks"),
             Input("chatbot-deploy", "n_clicks"),
             Input({"type": "template-select", "index": ALL}, "n_clicks")],
            [State("chatbot-input", "value"),
             State("chatbot-messages", "children")]
        )
        def handle_chat_interaction(send_clicks, input_submit, new_strategy, 
                                  optimize, deploy, template_clicks, 
                                  user_input, current_messages):
            """Handle chat interactions."""
            ctx = dash.callback_context
            if not ctx.triggered:
                return current_messages or [], ""
            
            trigger_id = ctx.triggered[0]["prop_id"]
            messages = current_messages or []
            
            # Handle user input
            if ("chatbot-send" in trigger_id or "chatbot-input" in trigger_id) and user_input:
                # Add user message
                messages.append(_create_message(user_input, "user"))
                
                # Generate AI response
                ai_response = chatbot_manager.process_user_message(user_input)
                messages.append(_create_message(ai_response, "assistant"))
                
                return messages, ""
            
            # Handle quick actions
            elif "chatbot-new-strategy" in trigger_id:
                messages.append(_create_message("I want to create a new strategy", "user"))
                ai_response = chatbot_manager._handle_strategy_creation("new strategy", {})
                messages.append(_create_message(ai_response, "assistant"))
                return messages, ""
            
            elif "chatbot-optimize" in trigger_id:
                messages.append(_create_message("Help me optimize my strategy", "user"))
                ai_response = chatbot_manager._handle_optimization("optimize", {})
                messages.append(_create_message(ai_response, "assistant"))
                return messages, ""
            
            elif "chatbot-deploy" in trigger_id:
                messages.append(_create_message("I'm ready to deploy my strategy", "user"))
                ai_response = chatbot_manager._handle_deployment("deploy", {})
                messages.append(_create_message(ai_response, "assistant"))
                return messages, ""
            
            # Handle template selection
            elif "template-select" in trigger_id and any(template_clicks):
                template_index = json.loads(trigger_id.split('.')[0])["index"]
                template = chatbot_manager.strategy_templates[template_index]
                
                user_msg = f"I'm interested in the {template['name']} template"
                messages.append(_create_message(user_msg, "user"))
                
                ai_response = f"""Great choice! The **{template['name']}** is an excellent strategy.

**Strategy Details:**
- **Description**: {template['description']}
- **Key Factors**: {', '.join(template['factors'])}
- **Rebalancing**: {template['rebalance_frequency'].title()}
- **Universe**: {template['universe'].replace('_', ' ').title()}
- **Risk Budget**: {template['risk_budget']:.1%}
- **Complexity**: {template['implementation_complexity'].title()}

Would you like me to:
1. Run a backtest of this strategy
2. Customize the parameters
3. Start the deployment process
4. Explain the academic research behind it"""

                messages.append(_create_message(ai_response, "assistant"))
                return messages, ""
            
            return messages, ""

def _create_message(content: str, sender: str) -> html.Div:
    """Create a chat message component."""
    is_user = sender == "user"
    
    return html.Div([
        html.Div([
            html.Strong("You: " if is_user else "Assistant: "),
            dcc.Markdown(content, className="d-inline")
        ], className=f"message {'user-message' if is_user else 'assistant-message'}",
        style={
            "background-color": "#e3f2fd" if is_user else "#f5f5f5",
            "padding": "10px",
            "margin": "5px 0",
            "border-radius": "10px",
            "margin-left": "20px" if is_user else "0px",
            "margin-right": "0px" if is_user else "20px"
        })
    ])

# Initialize chatbot system
chatbot_manager = ChatbotManager()
