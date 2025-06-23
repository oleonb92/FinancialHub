import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch
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
    return ExpensePredictor() 