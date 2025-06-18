"""
Unit tests for the expense predictor.

Este módulo valida el correcto funcionamiento del ExpensePredictor, incluyendo:
- Preparación de features
- Entrenamiento y predicción
- Predicción de secuencias
- Persistencia (guardar/cargar modelo y scaler)
- Manejo de errores y casos límite
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from transactions.models import Transaction, Category
from ai.ml.predictors.expense import ExpensePredictor
from organizations.models import Organization
from accounts.models import User

@pytest.fixture
def sample_transactions(db):
    """Create sample transactions for testing."""
    organization = Organization.objects.create(name="Test Organization")
    user = User.objects.create_user(username='testuser', email='test@example.com', password='testpass123')
    category = Category.objects.create(
        name="Groceries",
        organization=organization,
        created_by=user
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
            organization=organization,
            created_by=user,
            merchant=f"Store {i}"
        )
        transactions.append(transaction)
    return transactions

@pytest.fixture
def predictor():
    """Create a predictor instance for testing."""
    return ExpensePredictor()

def test_prepare_features(predictor, sample_transactions):
    """
    Verifica que la preparación de features retorna un DataFrame con la forma esperada.
    """
    features = predictor._prepare_features(sample_transactions)
    assert isinstance(features, pd.DataFrame)
    assert features.shape[0] == len(sample_transactions)
    assert features.shape[1] > 0

def test_prepare_sequence_features(predictor, sample_transactions):
    """
    Verifica que la preparación de features secuenciales retorna un DataFrame.
    """
    features = predictor._prepare_sequence_features(sample_transactions)
    assert isinstance(features, pd.DataFrame)
    # No shape asserts, as implementation returns DataFrame

def test_train(predictor, sample_transactions):
    """
    Verifica que el modelo puede entrenarse correctamente y que los atributos principales existen.
    """
    predictor.train(sample_transactions)
    assert hasattr(predictor, 'model')
    assert hasattr(predictor, 'scaler')

def test_predict(predictor, sample_transactions):
    """
    Verifica que el modelo puede predecir el monto de una categoría en una fecha futura.
    """
    predictor.train(sample_transactions)
    future_date = datetime.now().date() + timedelta(days=7)
    prediction = predictor.predict(future_date, sample_transactions[0].category.id)
    assert isinstance(prediction, float)
    assert prediction >= 0  # El modelo asegura no-negativo

def test_predict_sequence(predictor, sample_transactions):
    """
    Verifica que el modelo puede predecir una secuencia de días y que los resultados son válidos.
    """
    predictor.train(sample_transactions)
    start_date = datetime.now().date() + timedelta(days=1)
    predictions = predictor.predict_sequence(start_date, days=7)
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == 7
    assert 'predicted_amount' in predictions.columns
    assert all(isinstance(p, float) for p in predictions['predicted_amount'])
    assert all(p >= 0 for p in predictions['predicted_amount'])

def test_save_and_load_model(predictor, sample_transactions, tmp_path):
    """
    Verifica que el modelo y el scaler guardados y cargados producen las mismas predicciones que el original.
    """
    predictor.train(sample_transactions)
    predictor.save()
    new_predictor = ExpensePredictor()
    new_predictor.load()
    future_date = datetime.now().date() + timedelta(days=7)
    pred1 = predictor.predict(future_date, sample_transactions[0].category.id)
    pred2 = new_predictor.predict(future_date, sample_transactions[0].category.id)
    assert abs(pred1 - pred2) < 1e-6

def test_handle_empty_transactions(predictor):
    """
    Verifica que entrenar con una lista vacía de transacciones lanza una excepción.
    """
    with pytest.raises(Exception):
        predictor.train([])

def test_handle_invalid_date(predictor, sample_transactions):
    """
    Verifica que predecir con una fecha inválida lanza una excepción.
    """
    predictor.train(sample_transactions)
    with pytest.raises(Exception):
        predictor.predict(None, sample_transactions[0].category.id)

def test_handle_invalid_category(predictor, sample_transactions):
    """
    Verifica que predecir con una categoría inválida lanza una excepción.
    """
    predictor.train(sample_transactions)
    with pytest.raises(Exception):
        predictor.predict(datetime.now().date(), None) 