"""
Unit tests for the expense predictor.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from transactions.models import Transaction, Category
from ai.ml.predictors.expense import ExpensePredictor

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
def predictor():
    """Create a predictor instance for testing."""
    return ExpensePredictor()

def test_prepare_features(predictor, sample_transactions):
    """Test feature preparation."""
    features = predictor._prepare_features(sample_transactions)
    assert isinstance(features, np.ndarray)
    assert features.shape[0] == len(sample_transactions)
    assert features.shape[1] > 0

def test_prepare_sequence_features(predictor, sample_transactions):
    """Test sequence feature preparation."""
    features = predictor._prepare_sequence_features(sample_transactions)
    assert isinstance(features, np.ndarray)
    assert len(features.shape) == 3  # (samples, time_steps, features)
    assert features.shape[0] > 0
    assert features.shape[1] > 0
    assert features.shape[2] > 0

def test_train(predictor, sample_transactions):
    """Test model training."""
    predictor.train(sample_transactions)
    assert hasattr(predictor, 'model')
    assert hasattr(predictor, 'scaler')

def test_predict(predictor, sample_transactions):
    """Test prediction."""
    predictor.train(sample_transactions)
    
    future_date = datetime.now().date() + timedelta(days=7)
    prediction = predictor.predict(future_date, sample_transactions[0].category)
    
    assert isinstance(prediction, float)
    assert prediction < 0  # Should be negative for expenses

def test_predict_sequence(predictor, sample_transactions):
    """Test sequence prediction."""
    predictor.train(sample_transactions)
    
    future_dates = [datetime.now().date() + timedelta(days=i) for i in range(7)]
    predictions = predictor.predict_sequence(future_dates, sample_transactions[0].category)
    
    assert isinstance(predictions, list)
    assert len(predictions) == len(future_dates)
    assert all(isinstance(p, float) for p in predictions)
    assert all(p < 0 for p in predictions)  # All should be negative for expenses

def test_save_and_load_model(predictor, sample_transactions, tmp_path):
    """Test model persistence."""
    predictor.train(sample_transactions)
    
    # Save model
    model_path = tmp_path / "model.joblib"
    predictor.save_model(str(model_path))
    assert model_path.exists()
    
    # Load model
    new_predictor = ExpensePredictor()
    new_predictor.load_model(str(model_path))
    
    # Compare predictions
    future_date = datetime.now().date() + timedelta(days=7)
    pred1 = predictor.predict(future_date, sample_transactions[0].category)
    pred2 = new_predictor.predict(future_date, sample_transactions[0].category)
    
    assert abs(pred1 - pred2) < 1e-6

def test_handle_empty_transactions(predictor):
    """Test handling of empty transaction list."""
    with pytest.raises(ValueError):
        predictor.train([])

def test_handle_invalid_date(predictor, sample_transactions):
    """Test handling of invalid date."""
    predictor.train(sample_transactions)
    
    with pytest.raises(ValueError):
        predictor.predict(None, sample_transactions[0].category)

def test_handle_invalid_category(predictor, sample_transactions):
    """Test handling of invalid category."""
    predictor.train(sample_transactions)
    
    with pytest.raises(ValueError):
        predictor.predict(datetime.now().date(), None) 