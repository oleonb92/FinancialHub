"""
Unit tests for the behavior analyzer.

Este módulo valida el correcto funcionamiento del BehaviorAnalyzer, incluyendo:
- Análisis de patrones de gasto
- Preparación de features
- Detección de anomalías
- Persistencia (guardar/cargar modelo)
- Manejo de errores y casos límite
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from transactions.models import Transaction, Category
from ai.ml.analyzers.behavior import BehaviorAnalyzer
import pandas as pd
from django.test import TestCase
from accounts.models import User
from decimal import Decimal
import uuid
from organizations.models import Organization
import os
from django.conf import settings

# Create a picklable mock model class
class MockModel:
    def __init__(self):
        self.value = 42
    
    def predict(self, x):
        return np.array([1, 2, 3])

@pytest.fixture(autouse=True)
def setup_ml_models_dir():
    """Ensure ML models directory exists for tests."""
    os.makedirs(settings.ML_MODELS_DIR, exist_ok=True)
    yield
    # Cleanup after tests
    for file in os.listdir(settings.ML_MODELS_DIR):
        os.remove(os.path.join(settings.ML_MODELS_DIR, file))

@pytest.fixture
def sample_transactions():
    """Create sample transactions for testing."""
    organization = Organization.objects.create(name="Test Organization")
    user = User.objects.create_user(username='testuser', email='test@example.com', password='testpass123')
    category = Category.objects.create(
        name="Groceries",
        organization=organization,
        created_by=user
    )
    transactions = []
    for i in range(5):
        transaction = Transaction.objects.create(
            amount=100.0 + i,
            description=f"Grocery purchase {i}",
            date=datetime.now() - timedelta(days=i),
            category=category,
            type='EXPENSE',
            organization=organization,
            created_by=user
        )
        transactions.append(transaction)
    return transactions

@pytest.fixture
def behavior_analyzer():
    """Create a BehaviorAnalyzer instance for testing."""
    return BehaviorAnalyzer()

def test_analyze_spending_patterns(sample_transactions, behavior_analyzer):
    """
    Verifica que el análisis de patrones de gasto retorna resultados válidos.
    """
    result = behavior_analyzer.analyze_spending_patterns(sample_transactions)
    assert isinstance(result, dict)
    assert 'category_patterns' in result
    assert 'overall_patterns' in result
    assert isinstance(result['category_patterns'], dict)
    assert isinstance(result['overall_patterns'], dict)
    assert 'total_transactions' in result['overall_patterns']
    assert 'total_spent' in result['overall_patterns']
    assert 'anomalies' in result['overall_patterns']

def test_analyze_spending_patterns_empty(behavior_analyzer):
    """
    Verifica que el análisis con transacciones vacías retorna un resultado vacío.
    """
    result = behavior_analyzer.analyze_spending_patterns([])
    assert isinstance(result, dict)
    assert 'category_patterns' in result
    assert 'overall_patterns' in result
    assert result['overall_patterns']['total_transactions'] == 0
    assert result['overall_patterns']['total_spent'] == 0.0
    assert result['overall_patterns']['anomalies'] == 0

class TestBehaviorAnalyzer(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        
        # Create test organization
        self.organization = Organization.objects.create(name="Test Organization")
        
        # Create test categories
        self.food_category = Category.objects.create(
            name='Food',
            organization=self.organization,
            created_by=self.user
        )
        self.transport_category = Category.objects.create(
            name='Transport',
            organization=self.organization,
            created_by=self.user
        )
        
        # Create test transactions
        self.transaction1 = Transaction.objects.create(
            amount=50.0,
            description='Lunch',
            date=datetime.now(),
            category=self.food_category,
            type='EXPENSE',
            organization=self.organization,
            created_by=self.user
        )
        self.transaction2 = Transaction.objects.create(
            amount=30.0,
            description='Bus fare',
            date=datetime.now(),
            category=self.transport_category,
            type='EXPENSE',
            organization=self.organization,
            created_by=self.user
        )

    def test_analyze_spending_patterns(self):
        """
        Verifica el análisis de patrones de gasto en la clase de test.
        """
        analyzer = BehaviorAnalyzer()
        result = analyzer.analyze_spending_patterns([self.transaction1, self.transaction2])
        assert isinstance(result, dict)
        assert 'category_patterns' in result
        assert 'overall_patterns' in result
        assert isinstance(result['category_patterns'], dict)
        assert isinstance(result['overall_patterns'], dict)
        assert 'total_transactions' in result['overall_patterns']
        assert 'total_spent' in result['overall_patterns']
        assert 'anomalies' in result['overall_patterns']

    def test_analyze_spending_patterns_no_transactions(self):
        """
        Verifica el análisis con lista vacía en la clase de test.
        """
        analyzer = BehaviorAnalyzer()
        result = analyzer.analyze_spending_patterns([])
        assert isinstance(result, dict)
        assert 'category_patterns' in result
        assert 'overall_patterns' in result
        assert result['overall_patterns']['total_transactions'] == 0
        assert result['overall_patterns']['total_spent'] == 0.0
        assert result['overall_patterns']['anomalies'] == 0

def test_prepare_features(sample_transactions, behavior_analyzer):
    """
    Verifica que la preparación de features retorna un DataFrame con la forma esperada.
    """
    features = behavior_analyzer._prepare_features(sample_transactions)
    assert isinstance(features, pd.DataFrame)
    assert len(features) == len(sample_transactions)
    assert all(col in features.columns for col in ['amount', 'day_of_week', 'hour', 'category_id', 'merchant_id'])
    assert all(isinstance(features['amount'].iloc[i], float) for i in range(len(features)))
    assert all(isinstance(features['day_of_week'].iloc[i], (int, np.integer)) for i in range(len(features)))
    assert all(isinstance(features['hour'].iloc[i], (int, np.integer)) for i in range(len(features)))
    assert all(isinstance(features['category_id'].iloc[i], (int, np.integer)) for i in range(len(features)))
    assert all(isinstance(features['merchant_id'].iloc[i], (int, np.integer)) for i in range(len(features)))

def test_detect_anomalies(sample_transactions, behavior_analyzer):
    """
    Verifica que la detección de anomalías retorna un array del tamaño esperado.
    """
    features = behavior_analyzer._prepare_features(sample_transactions)
    anomalies = behavior_analyzer._detect_anomalies(features)
    assert isinstance(anomalies, np.ndarray)
    assert len(anomalies) == len(sample_transactions)

def test_get_preferred_days(sample_transactions, behavior_analyzer):
    """
    Verifica que se obtienen los días preferidos de gasto.
    """
    features = behavior_analyzer._prepare_features(sample_transactions)
    preferred_days = behavior_analyzer._get_preferred_days(features)
    assert isinstance(preferred_days, dict)
    assert all(isinstance(day, str) for day in preferred_days.keys())
    assert all(isinstance(count, int) for count in preferred_days.values())

def test_get_preferred_hours(sample_transactions, behavior_analyzer):
    """
    Verifica que se obtienen las horas preferidas de gasto.
    """
    features = behavior_analyzer._prepare_features(sample_transactions)
    preferred_hours = behavior_analyzer._get_preferred_hours(features)
    assert isinstance(preferred_hours, dict)
    assert all(isinstance(hour, int) for hour in preferred_hours.keys())
    assert all(isinstance(count, int) for count in preferred_hours.values())

def test_analyze_spending_trends(sample_transactions, behavior_analyzer):
    """
    Verifica que el análisis de tendencias de gasto retorna un DataFrame.
    """
    trends = behavior_analyzer._analyze_spending_trend(sample_transactions)
    assert isinstance(trends, dict)
    assert 'trend_coefficient' in trends
    assert 'trend_direction' in trends
    assert 'daily_average' in trends

def test_analyze_category_distribution(sample_transactions, behavior_analyzer):
    """
    Verifica que el análisis de distribución por categoría retorna un DataFrame.
    """
    features = behavior_analyzer._prepare_features(sample_transactions)
    distribution = behavior_analyzer._analyze_category_distribution(features)
    assert isinstance(distribution, dict)
    assert all(isinstance(category_id, int) for category_id in distribution.keys())
    assert all(isinstance(amount, float) for amount in distribution.values())

def test_save_and_load_model(behavior_analyzer):
    """
    Verifica que el modelo guardado y cargado es igual al original.
    """
    # Create a mock model that can be pickled
    mock_model = MockModel()
    behavior_analyzer.model = mock_model
    
    # Test saving
    behavior_analyzer.save()
    
    # Test loading
    with patch('joblib.load', return_value=mock_model):
        behavior_analyzer.load()
        assert behavior_analyzer.model == mock_model

def test_handle_empty_transactions(behavior_analyzer):
    """
    Verifica que analizar una lista vacía de transacciones no lanza excepción y retorna resultado válido.
    """
    result = behavior_analyzer.analyze_spending_patterns([])
    assert isinstance(result, dict)
    assert result['overall_patterns']['total_transactions'] == 0
    assert result['overall_patterns']['total_spent'] == 0.0
    assert result['overall_patterns']['anomalies'] == 0

def test_handle_invalid_transaction(behavior_analyzer, sample_transactions):
    """
    Verifica que analizar una transacción inválida lanza una excepción.
    """
    invalid_transaction = Mock()
    invalid_transaction.amount = 'invalid'
    with pytest.raises(ValueError):
        behavior_analyzer.analyze_spending_patterns([invalid_transaction]) 