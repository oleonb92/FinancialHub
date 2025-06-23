"""
Machine Learning module for FinancialHub.
This module contains all ML models and utilities for the AI system.
"""

from .base import BaseMLModel
from .classifiers.transaction import TransactionClassifier
from .predictors.expense import ExpensePredictor
from .analyzers.behavior import BehaviorAnalyzer
from .optimizers.budget_optimizer import BudgetOptimizer

__all__ = [
    'BaseMLModel',
    'TransactionClassifier',
    'ExpensePredictor',
    'BehaviorAnalyzer',
    'BudgetOptimizer',
] 