"""
Unit tests for the behavior analyzer.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from transactions.models import Transaction, Category
from ai.ml.analyzers.behavior import BehaviorAnalyzer

@pytest.fixture
def sample_transactions():
    """Create sample transactions for testing."""
    category = Category.objects.create(
        name="Groceries",
        organization=Mock(id=1),
        created_by=Mock(id=1)
    )
    
    transactions = []
    base_date = datetime.now().date()
    for i in range(30):  # Last 30 days
        transaction = Transaction.objects.create(
            type="EXPENSE",
            amount=100.0 + i,
            date=base_date - timedelta(days=i),
            description=f"Test transaction {i}",
            category=category,
            organization=Mock(id=1),
            created_by=Mock(id=1),
            merchant=f"Store {i}"
        )
        transactions.append(transaction)
    
    return transactions

@pytest.fixture
def analyzer():
    """Create an analyzer instance for testing."""
    return BehaviorAnalyzer()

def test_prepare_features(analyzer, sample_transactions):
    """Test feature preparation."""
    features = analyzer._prepare_features(sample_transactions)
    assert isinstance(features, np.ndarray)
    assert features.shape[0] == len(sample_transactions)
    assert features.shape[1] > 0

def test_detect_anomalies(analyzer, sample_transactions):
    """Test anomaly detection."""
    # Add an anomalous transaction
    anomalous = Transaction.objects.create(
        type="EXPENSE",
        amount=1000.0,  # Much larger than others
        date=datetime.now().date(),
        description="Anomalous transaction",
        category=sample_transactions[0].category,
        organization=Mock(id=1),
        created_by=Mock(id=1),
        merchant="Anomalous Store"
    )
    sample_transactions.append(anomalous)
    
    anomalies = analyzer._detect_anomalies(sample_transactions)
    assert isinstance(anomalies, list)
    assert len(anomalies) > 0
    assert anomalous in anomalies

def test_analyze_spending_patterns(analyzer, sample_transactions):
    """Test spending pattern analysis."""
    patterns = analyzer.analyze_spending_patterns(sample_transactions)
    
    assert isinstance(patterns, dict)
    assert 'category_patterns' in patterns
    assert 'overall_patterns' in patterns
    assert 'anomalies' in patterns

def test_get_preferred_days(analyzer, sample_transactions):
    """Test preferred days analysis."""
    days = analyzer._get_preferred_days(sample_transactions)
    assert isinstance(days, list)
    assert all(isinstance(d, int) for d in days)
    assert all(0 <= d <= 6 for d in days)

def test_get_preferred_hours(analyzer, sample_transactions):
    """Test preferred hours analysis."""
    hours = analyzer._get_preferred_hours(sample_transactions)
    assert isinstance(hours, list)
    assert all(isinstance(h, int) for h in hours)
    assert all(0 <= h <= 23 for h in hours)

def test_analyze_spending_trends(analyzer, sample_transactions):
    """Test spending trend analysis."""
    trends = analyzer._analyze_spending_trends(sample_transactions)
    assert isinstance(trends, dict)
    assert 'daily_average' in trends
    assert 'weekly_average' in trends
    assert 'monthly_average' in trends
    assert 'trend_direction' in trends

def test_analyze_category_distribution(analyzer, sample_transactions):
    """Test category distribution analysis."""
    distribution = analyzer._analyze_category_distribution(sample_transactions)
    assert isinstance(distribution, dict)
    assert all(isinstance(v, float) for v in distribution.values())
    assert abs(sum(distribution.values()) - 1.0) < 1e-6

def test_save_and_load_model(analyzer, sample_transactions, tmp_path):
    """Test model persistence."""
    analyzer.analyze_spending_patterns(sample_transactions)
    
    # Save model
    model_path = tmp_path / "model.joblib"
    analyzer.save_model(str(model_path))
    assert model_path.exists()
    
    # Load model
    new_analyzer = BehaviorAnalyzer()
    new_analyzer.load_model(str(model_path))
    
    # Compare analyses
    patterns1 = analyzer.analyze_spending_patterns(sample_transactions)
    patterns2 = new_analyzer.analyze_spending_patterns(sample_transactions)
    
    assert patterns1['overall_patterns'] == patterns2['overall_patterns']

def test_handle_empty_transactions(analyzer):
    """Test handling of empty transaction list."""
    with pytest.raises(ValueError):
        analyzer.analyze_spending_patterns([])

def test_handle_invalid_transaction(analyzer, sample_transactions):
    """Test handling of invalid transaction."""
    with pytest.raises(ValueError):
        analyzer.analyze_spending_patterns([None]) 