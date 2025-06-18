"""
Unit tests for the transaction classifier.

Este módulo valida el correcto funcionamiento del TransactionClassifier, incluyendo:
- Preparación de features
- Entrenamiento y predicción
- Persistencia (guardar/cargar modelo)
- Manejo de errores y casos límite
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch
from transactions.models import Transaction, Category
from ai.ml.classifiers.transaction import TransactionClassifier
from organizations.models import Organization
from accounts.models import User

@pytest.fixture
def sample_transactions(db):
    """
    Crea transacciones de ejemplo para pruebas unitarias.
    Incluye una sola categoría y organización para simplificar la validación.
    """
    organization = Organization.objects.create(name="Test Organization")
    user = User.objects.create_user(username='testuser', email='test@example.com', password='testpass123')
    category = Category.objects.create(
        name="Groceries",
        organization=organization,
        created_by=user
    )
    transactions = []
    for i in range(10):
        transaction = Transaction.objects.create(
            type="EXPENSE",
            amount=100.0 + i,
            date=datetime.now().date(),
            description=f"Test transaction {i}",
            category=category,
            organization=organization,
            created_by=user,
            merchant=f"Store {i}"
        )
        transactions.append(transaction)
    return transactions

@pytest.fixture
def classifier():
    """
    Instancia un TransactionClassifier para pruebas.
    """
    return TransactionClassifier()

def test_prepare_features(classifier, sample_transactions):
    """
    Verifica que la preparación de features retorna un DataFrame con la forma esperada.
    """
    features = classifier._prepare_features(sample_transactions)
    assert isinstance(features, pd.DataFrame)
    assert features.shape[0] == len(sample_transactions)
    assert features.shape[1] > 0

def test_train(classifier, sample_transactions):
    """
    Verifica que el modelo puede entrenarse correctamente y que los atributos principales existen.
    """
    classifier.train(sample_transactions)
    assert hasattr(classifier, 'pipeline')
    assert hasattr(classifier, 'categories')

def test_predict(classifier, sample_transactions):
    """
    Verifica que el modelo puede predecir la categoría de una nueva transacción y que el tipo de salida es correcto.
    """
    classifier.train(sample_transactions)
    new_transaction = Transaction.objects.create(
        type="EXPENSE",
        amount=150.0,
        date=datetime.now().date(),
        description="New test transaction",
        category=sample_transactions[0].category,
        organization=sample_transactions[0].organization,
        created_by=sample_transactions[0].created_by,
        merchant="New Store"
    )
    prediction, confidence = classifier.predict(new_transaction)
    assert isinstance(prediction, (int, np.integer))
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1

def test_evaluate(classifier, sample_transactions):
    """
    Verifica que el método de evaluación retorna métricas válidas.
    """
    classifier.train(sample_transactions)
    metrics = classifier.evaluate(sample_transactions)
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 0 <= metrics['accuracy'] <= 1

def test_save_and_load_model(classifier, sample_transactions, tmp_path):
    """
    Verifica que el modelo guardado y cargado produce las mismas predicciones que el original.
    """
    classifier.train(sample_transactions)
    classifier.save()
    new_classifier = TransactionClassifier()
    new_classifier.load()
    test_transaction = sample_transactions[0]
    pred1, conf1 = classifier.predict(test_transaction)
    pred2, conf2 = new_classifier.predict(test_transaction)
    assert pred1 == pred2
    assert abs(conf1 - conf2) < 1e-6

def test_handle_empty_transactions(classifier):
    """
    Verifica que entrenar con una lista vacía de transacciones lanza una excepción.
    """
    with pytest.raises(Exception):
        classifier.train([])

def test_handle_invalid_transaction(classifier, sample_transactions):
    """
    Verifica que predecir con una transacción inválida lanza una excepción.
    """
    classifier.train(sample_transactions)
    with pytest.raises(Exception):
        classifier.predict(None) 