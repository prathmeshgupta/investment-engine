"""
Dashboard Components Module
Contains reusable UI components for the investment engine dashboard
"""

from .tutorial import TutorialManager, TutorialCallbacks
from .chatbot import ChatbotManager, ChatbotCallbacks
from .research import ResearchManager, ResearchCallbacks
from .visualizations import AdvancedVisualizationManager

__all__ = [
    'TutorialManager',
    'TutorialCallbacks', 
    'ChatbotManager',
    'ChatbotCallbacks',
    'ResearchManager',
    'ResearchCallbacks',
    'AdvancedVisualizationManager'
]
