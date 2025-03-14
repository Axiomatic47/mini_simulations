"""
Weak Nuclear Physics Domain

This module contains functions that model suppression feedback and resistance
resurgence dynamics using principles from weak nuclear forces.
"""

from .resistance_resurgence import resistance_resurgence
from .suppression_feedback import suppression_feedback

__all__ = [
    'resistance_resurgence',
    'suppression_feedback',
]