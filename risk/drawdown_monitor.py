"""Drawdown monitoring and analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class DrawdownMonitor:
    """Monitor and analyze portfolio drawdowns."""
    
    def __init__(self):
        """Initialize drawdown monitor."""
        self.drawdown_history = []
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.peak_value = 0.0
        
    def calculate_drawdowns(self, 
                          returns: pd.Series,
                          cumulative: bool = True) -> Dict[str, float]:
        """Calculate drawdown metrics from returns.
        
        Args:
            returns: Return series
            cumulative: Whether returns are cumulative or periodic
            
        Returns:
            Dictionary of drawdown metrics
        """
        if cumulative:
            cumulative_returns = returns
        else:
            cumulative_returns = (1 + returns).cumprod()
        
        # Calculate running maximum (peak)
        running_max = cumulative_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Current drawdown
        current_dd = drawdown.iloc[-1] if len(drawdown) > 0 else 0
        
        # Maximum drawdown
        max_dd = drawdown.min()
        
        # Drawdown duration analysis
        dd_periods = self._analyze_drawdown_periods(drawdown)
        
        metrics = {
            'current_drawdown': current_dd,
            'max_drawdown': max_dd,
            'avg_drawdown': drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0,
            'drawdown_volatility': drawdown.std(),
            'max_drawdown_duration': dd_periods['max_duration'],
            'avg_drawdown_duration': dd_periods['avg_duration'],
            'current_drawdown_duration': dd_periods['current_duration'],
            'recovery_time': dd_periods['avg_recovery_time'],
            'drawdown_frequency': dd_periods['frequency']
        }
        
        # Update internal state
        self.current_drawdown = current_dd
        self.max_drawdown = max_dd
        if len(cumulative_returns) > 0:
            self.peak_value = running_max.iloc[-1]
        
        return metrics
    
    def _analyze_drawdown_periods(self, drawdown: pd.Series) -> Dict[str, float]:
        """Analyze drawdown periods and recovery times."""
        # Identify drawdown periods (when drawdown < 0)
        in_drawdown = drawdown < -0.001  # Small threshold to avoid noise
        
        # Find drawdown start and end points
        drawdown_changes = in_drawdown.astype(int).diff()
        drawdown_starts = drawdown_changes[drawdown_changes == 1].index
        drawdown_ends = drawdown_changes[drawdown_changes == -1].index
        
        # Handle edge cases
        if len(drawdown_starts) == 0:
            return {
                'max_duration': 0,
                'avg_duration': 0,
                'current_duration': 0,
                'avg_recovery_time': 0,
                'frequency': 0
            }
        
        # If we start in drawdown
        if in_drawdown.iloc[0]:
            drawdown_starts = pd.Index([drawdown.index[0]]).union(drawdown_starts)
        
        # If we end in drawdown
        if in_drawdown.iloc[-1]:
            drawdown_ends = drawdown_ends.union(pd.Index([drawdown.index[-1]]))
        
        # Calculate durations
        durations = []
        for i, start in enumerate(drawdown_starts):
            if i < len(drawdown_ends):
                end = drawdown_ends[i]
                duration = (end - start).days if hasattr(start, 'days') else i
                durations.append(duration)
        
        # Current drawdown duration
        current_duration = 0
        if in_drawdown.iloc[-1] and len(drawdown_starts) > 0:
            last_start = drawdown_starts[-1]
            current_duration = (drawdown.index[-1] - last_start).days if hasattr(last_start, 'days') else 0
        
        # Recovery time analysis
        recovery_times = []
        for i in range(len(drawdown_ends)):
            if i < len(drawdown_starts) - 1:
                recovery_start = drawdown_ends[i]
                next_drawdown = drawdown_starts[i + 1]
                recovery_time = (next_drawdown - recovery_start).days if hasattr(recovery_start, 'days') else 0
                recovery_times.append(recovery_time)
        
        return {
            'max_duration': max(durations) if durations else 0,
            'avg_duration': np.mean(durations) if durations else 0,
            'current_duration': current_duration,
            'avg_recovery_time': np.mean(recovery_times) if recovery_times else 0,
            'frequency': len(durations) / len(drawdown) * 252 if len(drawdown) > 0 else 0  # Annualized
        }
    
    def calculate_calmar_ratio(self, 
                             returns: pd.Series,
                             periods_per_year: int = 252) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown).
        
        Args:
            returns: Return series
            periods_per_year: Number of periods per year
            
        Returns:
            Calmar ratio
        """
        # Annualized return
        total_periods = len(returns)
        if total_periods == 0:
            return 0
        
        cumulative_return = (1 + returns).prod() - 1
        annualized_return = (1 + cumulative_return) ** (periods_per_year / total_periods) - 1
        
        # Maximum drawdown
        drawdown_metrics = self.calculate_drawdowns(returns, cumulative=False)
        max_dd = abs(drawdown_metrics['max_drawdown'])
        
        # Calmar ratio
        if max_dd > 0:
            calmar_ratio = annualized_return / max_dd
        else:
            calmar_ratio = 0
        
        return calmar_ratio
    
    def detect_drawdown_alerts(self, 
                             current_drawdown: float,
                             max_drawdown_threshold: float = 0.1,
                             duration_threshold: int = 30) -> List[Dict[str, str]]:
        """Detect drawdown-based alerts.
        
        Args:
            current_drawdown: Current drawdown level
            max_drawdown_threshold: Maximum acceptable drawdown
            duration_threshold: Maximum acceptable drawdown duration (days)
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        # Drawdown magnitude alert
        if abs(current_drawdown) > max_drawdown_threshold:
            alerts.append({
                'type': 'drawdown_magnitude',
                'severity': 'high' if abs(current_drawdown) > max_drawdown_threshold * 1.5 else 'medium',
                'message': f"Current drawdown {current_drawdown:.2%} exceeds threshold {max_drawdown_threshold:.2%}",
                'value': current_drawdown,
                'threshold': max_drawdown_threshold
            })
        
        # Drawdown duration alert (would need historical tracking)
        if hasattr(self, 'current_drawdown_duration'):
            if self.current_drawdown_duration > duration_threshold:
                alerts.append({
                    'type': 'drawdown_duration',
                    'severity': 'medium',
                    'message': f"Drawdown duration {self.current_drawdown_duration} days exceeds threshold {duration_threshold} days",
                    'value': self.current_drawdown_duration,
                    'threshold': duration_threshold
                })
        
        return alerts
    
    def calculate_underwater_curve(self, returns: pd.Series) -> pd.Series:
        """Calculate underwater curve (drawdown over time).
        
        Args:
            returns: Return series
            
        Returns:
            Underwater curve (drawdown series)
        """
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        underwater_curve = (cumulative_returns - running_max) / running_max
        
        return underwater_curve
    
    def get_worst_drawdown_periods(self, 
                                 returns: pd.Series, 
                                 n_periods: int = 5) -> List[Dict[str, any]]:
        """Get the worst drawdown periods.
        
        Args:
            returns: Return series
            n_periods: Number of worst periods to return
            
        Returns:
            List of worst drawdown period information
        """
        underwater_curve = self.calculate_underwater_curve(returns)
        cumulative_returns = (1 + returns).cumprod()
        
        # Find all drawdown periods
        in_drawdown = underwater_curve < -0.001
        drawdown_changes = in_drawdown.astype(int).diff()
        
        drawdown_starts = drawdown_changes[drawdown_changes == 1].index
        drawdown_ends = drawdown_changes[drawdown_changes == -1].index
        
        # Handle edge cases
        if in_drawdown.iloc[0]:
            drawdown_starts = pd.Index([underwater_curve.index[0]]).union(drawdown_starts)
        if in_drawdown.iloc[-1]:
            drawdown_ends = drawdown_ends.union(pd.Index([underwater_curve.index[-1]]))
        
        # Analyze each drawdown period
        drawdown_periods = []
        for i, start in enumerate(drawdown_starts):
            if i < len(drawdown_ends):
                end = drawdown_ends[i]
                period_dd = underwater_curve.loc[start:end]
                
                max_dd = period_dd.min()
                max_dd_date = period_dd.idxmin()
                duration = len(period_dd)
                
                # Peak and trough values
                peak_value = cumulative_returns.loc[start]
                trough_value = cumulative_returns.loc[max_dd_date]
                
                drawdown_periods.append({
                    'start_date': start,
                    'end_date': end,
                    'max_drawdown_date': max_dd_date,
                    'max_drawdown': max_dd,
                    'duration': duration,
                    'peak_value': peak_value,
                    'trough_value': trough_value,
                    'recovery_time': (end - max_dd_date).days if hasattr(end, 'days') else 0
                })
        
        # Sort by maximum drawdown and return worst periods
        drawdown_periods.sort(key=lambda x: x['max_drawdown'])
        
        return drawdown_periods[:n_periods]
