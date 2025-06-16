"""
Unit tests for the transaction classifier.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch
from transactions.models import Transaction, Category
from ai.ml.classifiers.transaction import TransactionClassifier

@pytest.fixture
def sample_transactions():
    """Create sample transactions for testing."""
    category = Category.objects.create(
        name="Groceries",
        organization=Mock(id=1),
        created_by=Mock(id=1)
    )
    
    transactions = []
    for i in range(10):
        transaction = Transaction.objects.create(
            type="EXPENSE",
            amount=100.0 + i,
            date=datetime.now().date(),
            description=f"Test transaction {i}",
            category=category,
            organization=Mock(id=1),
            created_by=Mock(id=1),
            merchant=f"Store {i}"
        )
        transactions.append(transaction)
    
    return transactions

@pytest.fixture
def classifier():
    """Create a classifier instance for testing."""
    return TransactionClassifier()

def test_prepare_features(classifier, sample_transactions):
    """Test feature preparation."""
    features = classifier._prepare_features(sample_transactions)
    assert isinstance(features, np.ndarray)
    assert features.shape[0] == len(sample_transactions)
    assert features.shape[1] > 0

def test_train(classifier, sample_transactions):
    """Test model training."""
    classifier.train(sample_transactions)
    assert hasattr(classifier, 'model')
    assert hasattr(classifier, 'feature_names')

def test_predict(classifier, sample_transactions):
    """Test prediction."""
    classifier.train(sample_transactions)
    
    new_transaction = Transaction.objects.create(
        type="EXPENSE",
        amount=150.0,
        date=datetime.now().date(),
        description="New test transaction",
        category=sample_transactions[0].category,
        organization=Mock(id=1),
        created_by=Mock(id=1),
        merchant="New Store"
    )
    
    prediction, confidence = classifier.predict(new_transaction)
    assert isinstance(prediction, Category)
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1

def test_evaluate(classifier, sample_transactions):
    """Test model evaluation."""
    classifier.train(sample_transactions)
    metrics = classifier.evaluate(sample_transactions)
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 0 <= metrics['accuracy'] <= 1

def test_save_and_load_model(classifier, sample_transactions, tmp_path):
    """Test model persistence."""
    classifier.train(sample_transactions)
    
    # Save model
    model_path = tmp_path / "model.joblib"
    classifier.save_model(str(model_path))
    assert model_path.exists()
    
    # Load model
    new_classifier = TransactionClassifier()
    new_classifier.load_model(str(model_path))
    
    # Compare predictions
    test_transaction = sample_transactions[0]
    pred1, conf1 = classifier.predict(test_transaction)
    pred2, conf2 = new_classifier.predict(test_transaction)
    
    assert pred1.id == pred2.id
    assert abs(conf1 - conf2) < 1e-6

def test_handle_empty_transactions(classifier):
    """Test handling of empty transaction list."""
    with pytest.raises(ValueError):
        classifier.train([])

def test_handle_invalid_transaction(classifier, sample_transactions):
    """Test handling of invalid transaction."""
    classifier.train(sample_transactions)
    
    with pytest.raises(ValueError):
        classifier.predict(None) 