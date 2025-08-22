"""
Usability enhancement components for the dashboard.
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import datetime
import pandas as pd


class KeyboardShortcuts:
    """Manages keyboard shortcuts for the dashboard."""
    
    SHORTCUTS = {
        'alt+p': {'action': 'navigate_portfolio', 'description': 'Go to Portfolio'},
        'alt+r': {'action': 'navigate_risk', 'description': 'Go to Risk Analysis'},
        'alt+s': {'action': 'navigate_strategy', 'description': 'Go to Strategy'},
        'alt+h': {'action': 'show_help', 'description': 'Show Help'},
        'alt+n': {'action': 'new_strategy', 'description': 'Create New Strategy'},
        'alt+b': {'action': 'run_backtest', 'description': 'Run Backtest'},
        'alt+e': {'action': 'export_report', 'description': 'Export Report'},
        'ctrl+z': {'action': 'undo', 'description': 'Undo Last Action'},
        'ctrl+y': {'action': 'redo', 'description': 'Redo Last Action'},
        'esc': {'action': 'close_modal', 'description': 'Close Modal/Dialog'},
    }
    
    @staticmethod
    def get_component():
        """Returns the keyboard shortcuts component."""
        return html.Div([
            # Hidden div to capture keyboard events
            html.Div(id='keyboard-listener', style={'display': 'none'}),
            
            # JavaScript for keyboard event handling
            html.Script("""
                document.addEventListener('keydown', function(e) {
                    var shortcut = '';
                    if (e.altKey) shortcut += 'alt+';
                    if (e.ctrlKey) shortcut += 'ctrl+';
                    if (e.shiftKey) shortcut += 'shift+';
                    shortcut += e.key.toLowerCase();
                    
                    // Send to Dash callback
                    var element = document.getElementById('keyboard-listener');
                    if (element) {
                        element.setAttribute('data-shortcut', shortcut);
                        element.click();
                    }
                });
            """)
        ])
    
    @staticmethod
    def get_help_modal():
        """Returns modal showing all keyboard shortcuts."""
        shortcuts_table = dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Shortcut", style={'width': '30%'}),
                    html.Th("Action")
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(html.Kbd(shortcut.upper())),
                    html.Td(info['description'])
                ]) for shortcut, info in KeyboardShortcuts.SHORTCUTS.items()
            ])
        ], striped=True, hover=True, size='sm')
        
        return dbc.Modal([
            dbc.ModalHeader([
                html.I(className="fas fa-keyboard me-2"),
                "Keyboard Shortcuts"
            ]),
            dbc.ModalBody(shortcuts_table),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-shortcuts", className="ms-auto")
            )
        ], id="shortcuts-modal", size="lg")


class QuickStartMode:
    """Simplified interface for beginners."""
    
    PRESET_STRATEGIES = {
        'conservative': {
            'name': 'Conservative Portfolio',
            'description': 'Low-risk balanced portfolio with bonds focus',
            'allocation': {'Stocks': 30, 'Bonds': 60, 'Cash': 10},
            'risk_level': 'Low',
            'expected_return': '4-6%'
        },
        'balanced': {
            'name': 'Balanced Portfolio',
            'description': 'Equal mix of growth and income',
            'allocation': {'Stocks': 50, 'Bonds': 40, 'Cash': 10},
            'risk_level': 'Medium',
            'expected_return': '6-8%'
        },
        'growth': {
            'name': 'Growth Portfolio',
            'description': 'Higher risk with growth focus',
            'allocation': {'Stocks': 70, 'Bonds': 25, 'Cash': 5},
            'risk_level': 'Medium-High',
            'expected_return': '8-12%'
        },
        'aggressive': {
            'name': 'Aggressive Portfolio',
            'description': 'Maximum growth potential',
            'allocation': {'Stocks': 90, 'Bonds': 5, 'Cash': 5},
            'risk_level': 'High',
            'expected_return': '10-15%'
        }
    }
    
    @staticmethod
    def get_component():
        """Returns the quick-start wizard component."""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H2([
                        html.I(className="fas fa-rocket me-2"),
                        "Quick Start Wizard"
                    ], className="mb-4"),
                    html.P(
                        "Get started quickly with preset strategies or create your own.",
                        className="lead text-muted"
                    )
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Choose Your Investment Style", className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            QuickStartMode._create_strategy_card(key, strategy)
                        ], md=6, className="mb-3")
                        for key, strategy in QuickStartMode.PRESET_STRATEGIES.items()
                    ])
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Hr(className="my-4"),
                    html.H4("Or Build Custom Strategy", className="mb-3"),
                    dbc.Card([
                        dbc.CardBody([
                            html.H5([
                                html.I(className="fas fa-tools me-2"),
                                "Custom Strategy Builder"
                            ]),
                            html.P("Create a personalized strategy with our guided builder"),
                            dbc.Button(
                                "Start Building",
                                id="start-custom-strategy",
                                color="primary",
                                size="lg",
                                className="mt-3"
                            )
                        ])
                    ])
                ])
            ])
        ], fluid=True, className="py-4")
    
    @staticmethod
    def _create_strategy_card(key, strategy):
        """Creates a card for a preset strategy."""
        risk_colors = {
            'Low': 'success',
            'Medium': 'warning',
            'Medium-High': 'warning',
            'High': 'danger'
        }
        
        return dbc.Card([
            dbc.CardHeader([
                html.H5(strategy['name'], className="mb-0"),
                dbc.Badge(
                    strategy['risk_level'],
                    color=risk_colors.get(strategy['risk_level'], 'secondary'),
                    className="ms-2"
                )
            ]),
            dbc.CardBody([
                html.P(strategy['description'], className="card-text"),
                html.Div([
                    html.Small("Asset Allocation:", className="text-muted"),
                    html.Ul([
                        html.Li(f"{asset}: {pct}%")
                        for asset, pct in strategy['allocation'].items()
                    ], className="mb-2")
                ]),
                html.Div([
                    html.Small("Expected Annual Return: ", className="text-muted"),
                    html.Strong(strategy['expected_return'])
                ])
            ]),
            dbc.CardFooter([
                dbc.Button(
                    "Select Strategy",
                    id=f"select-{key}",
                    color="primary",
                    className="w-100"
                )
            ])
        ])


class UserFeedback:
    """Enhanced user feedback and error handling."""
    
    @staticmethod
    def create_toast(message: str, title: str = "Notification", 
                    category: str = "info", duration: int = 4000):
        """Creates a toast notification."""
        icons = {
            'success': 'fas fa-check-circle',
            'error': 'fas fa-exclamation-circle',
            'warning': 'fas fa-exclamation-triangle',
            'info': 'fas fa-info-circle'
        }
        
        colors = {
            'success': 'success',
            'error': 'danger',
            'warning': 'warning',
            'info': 'info'
        }
        
        return dbc.Toast(
            [html.P(message, className="mb-0")],
            header=[
                html.I(className=f"{icons.get(category, 'fas fa-bell')} me-2"),
                title
            ],
            icon=colors.get(category, 'primary'),
            duration=duration,
            dismissable=True,
            style={'position': 'fixed', 'top': 20, 'right': 20, 'zIndex': 9999}
        )
    
    @staticmethod
    def create_progress_indicator(progress: float, message: str = "Processing..."):
        """Creates a progress indicator."""
        return dbc.Card([
            dbc.CardBody([
                html.H5(message, className="mb-3"),
                dbc.Progress(
                    value=progress,
                    label=f"{progress}%",
                    color="primary",
                    striped=True,
                    animated=True,
                    style={'height': '25px'}
                )
            ])
        ], className="shadow-sm")
    
    @staticmethod
    def create_error_boundary(error: Exception, context: str = ""):
        """Creates user-friendly error message."""
        return dbc.Alert([
            html.H4([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Something went wrong"
            ], className="alert-heading"),
            html.Hr(),
            html.P("We encountered an issue while processing your request."),
            html.Details([
                html.Summary("Technical Details", className="mb-2"),
                html.Pre(f"Context: {context}\nError: {str(error)}", 
                        className="bg-light p-2 rounded")
            ]),
            html.Div([
                dbc.Button("Retry", id="retry-action", color="primary", className="me-2"),
                dbc.Button("Report Issue", id="report-issue", outline=True, color="secondary")
            ], className="mt-3")
        ], color="danger", dismissable=True)


class DataValidation:
    """Input validation and helpers."""
    
    @staticmethod
    def create_validated_input(input_id: str, label: str, 
                              input_type: str = "text", 
                              validation_rules: Dict = None):
        """Creates an input with validation."""
        validation_rules = validation_rules or {}
        
        return dbc.FormGroup([
            dbc.Label(label, html_for=input_id),
            dbc.Input(
                id=input_id,
                type=input_type,
                placeholder=validation_rules.get('placeholder', ''),
                min=validation_rules.get('min'),
                max=validation_rules.get('max'),
                step=validation_rules.get('step'),
                required=validation_rules.get('required', False)
            ),
            dbc.FormFeedback(
                validation_rules.get('error_message', 'Invalid input'),
                type="invalid"
            ),
            dbc.FormText(
                validation_rules.get('help_text', ''),
                color="muted"
            ) if validation_rules.get('help_text') else None
        ])
    
    @staticmethod
    def create_smart_date_picker(picker_id: str, label: str):
        """Creates an intelligent date picker with presets."""
        return dbc.FormGroup([
            dbc.Label(label),
            dbc.InputGroup([
                dcc.DatePickerSingle(
                    id=picker_id,
                    date=datetime.now().date(),
                    display_format='YYYY-MM-DD',
                    style={'width': '100%'}
                ),
                dbc.InputGroupText([
                    dbc.ButtonGroup([
                        dbc.Button("1M", id=f"{picker_id}-1m", size="sm", outline=True),
                        dbc.Button("3M", id=f"{picker_id}-3m", size="sm", outline=True),
                        dbc.Button("6M", id=f"{picker_id}-6m", size="sm", outline=True),
                        dbc.Button("1Y", id=f"{picker_id}-1y", size="sm", outline=True),
                        dbc.Button("YTD", id=f"{picker_id}-ytd", size="sm", outline=True),
                    ])
                ])
            ])
        ])


class NavigationEnhancer:
    """Enhanced navigation with breadcrumbs and quick links."""
    
    @staticmethod
    def create_breadcrumbs(path: List[Dict[str, str]]):
        """Creates breadcrumb navigation."""
        items = []
        for i, item in enumerate(path):
            if i < len(path) - 1:
                items.append(dbc.BreadcrumbItem(
                    item['label'],
                    href=item.get('href', '#'),
                    active=False
                ))
            else:
                items.append(dbc.BreadcrumbItem(
                    item['label'],
                    active=True
                ))
        
        return dbc.Breadcrumb(items=items)
    
    @staticmethod
    def create_quick_actions():
        """Creates quick action buttons."""
        return dbc.ButtonGroup([
            dbc.Button([
                html.I(className="fas fa-plus me-1"),
                "New"
            ], id="quick-new", color="primary", size="sm"),
            dbc.Button([
                html.I(className="fas fa-save me-1"),
                "Save"
            ], id="quick-save", color="success", size="sm"),
            dbc.Button([
                html.I(className="fas fa-download me-1"),
                "Export"
            ], id="quick-export", color="info", size="sm"),
            dbc.Button([
                html.I(className="fas fa-cog me-1"),
                "Settings"
            ], id="quick-settings", color="secondary", size="sm"),
        ])


class UndoRedoManager:
    """Manages undo/redo functionality for user actions."""
    
    def __init__(self, max_history=50):
        self.history = []
        self.current_index = -1
        self.max_history = max_history
    
    def add_action(self, action: Dict[str, Any]):
        """Add a new action to history."""
        # Remove any forward history when new action is added
        self.history = self.history[:self.current_index + 1]
        
        # Add new action
        self.history.append(action)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        else:
            self.current_index += 1
    
    def undo(self) -> Optional[Dict[str, Any]]:
        """Undo the last action."""
        if self.can_undo():
            self.current_index -= 1
            return self.history[self.current_index]
        return None
    
    def redo(self) -> Optional[Dict[str, Any]]:
        """Redo the next action."""
        if self.can_redo():
            self.current_index += 1
            return self.history[self.current_index]
        return None
    
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return self.current_index > 0
    
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return self.current_index < len(self.history) - 1
    
    def get_current_state(self) -> Optional[Dict[str, Any]]:
        """Get the current state."""
        if 0 <= self.current_index < len(self.history):
            return self.history[self.current_index]
        return None


class SettingsManager:
    """Manages export/import functionality for user settings and configurations."""
    
    def __init__(self):
        self.settings_keys = [
            'portfolio_config',
            'risk_parameters',
            'strategy_settings',
            'ui_preferences',
            'keyboard_shortcuts',
            'alert_preferences',
            'data_sources',
            'backtesting_params'
        ]
    
    def export_settings(self, settings: Dict[str, Any]) -> str:
        """Export settings to JSON string."""
        export_data = {
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'settings': {}
        }
        
        for key in self.settings_keys:
            if key in settings:
                export_data['settings'][key] = settings[key]
        
        return json.dumps(export_data, indent=2)
    
    def import_settings(self, json_string: str) -> Dict[str, Any]:
        """Import settings from JSON string."""
        try:
            import_data = json.loads(json_string)
            
            # Validate structure
            if 'settings' not in import_data:
                raise ValueError("Invalid settings file format")
            
            # Extract settings
            imported_settings = import_data['settings']
            
            # Validate each setting key
            validated_settings = {}
            for key in self.settings_keys:
                if key in imported_settings:
                    validated_settings[key] = imported_settings[key]
            
            return {
                'success': True,
                'settings': validated_settings,
                'timestamp': import_data.get('timestamp'),
                'version': import_data.get('version')
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_export_component(self):
        """Create export/import UI component."""
        return html.Div([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-cog me-2"),
                    "Settings Management"
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Export Settings"),
                            html.P("Download your current configuration"),
                            dbc.Button(
                                [
                                    html.I(className="fas fa-download me-2"),
                                    "Export Settings"
                                ],
                                id="export-settings-btn",
                                color="primary",
                                className="mb-3"
                            ),
                            dcc.Download(id="download-settings")
                        ], md=6),
                        dbc.Col([
                            html.H5("Import Settings"),
                            html.P("Load a previously saved configuration"),
                            dcc.Upload(
                                id="upload-settings",
                                children=dbc.Button(
                                    [
                                        html.I(className="fas fa-upload me-2"),
                                        "Import Settings"
                                    ],
                                    color="success"
                                ),
                                multiple=False
                            ),
                            html.Div(id="import-status", className="mt-2")
                        ], md=6)
                    ])
                ])
            ])
        ])
