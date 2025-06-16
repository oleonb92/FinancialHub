"""
Transaction classifier for categorizing financial transactions.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from ..base import BaseMLModel
from django.db.models import Q
from transactions.models import Transaction, Category

class TransactionClassifier(BaseMLModel):
    """
    Classifier for categorizing financial transactions based on their description
    and other features.
    """
    
    def __init__(self):
        super().__init__('transaction_classifier')
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )),
            ('scaler', StandardScaler(with_mean=False)),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ))
        ])
        self.categories = None
    
    def _prepare_features(self, transactions):
        """
        Prepare features for training or prediction.
        
        Args:
            transactions: List of Transaction objects or single Transaction
            
        Returns:
            pd.DataFrame: Prepared features
        """
        if not isinstance(transactions, list):
            transactions = [transactions]
            
        features = pd.DataFrame({
            'description': [t.description for t in transactions],
            'amount': [float(t.amount) for t in transactions],
            'day_of_week': [t.date.weekday() for t in transactions],
            'day_of_month': [t.date.day for t in transactions],
            'month': [t.date.month for t in transactions]
        })
        
        return features
    
    def train(self, transactions):
        """
        Train the transaction classifier.
        
        Args:
            transactions: List of Transaction objects to train on
        """
        try:
            # Prepare features and labels
            X = self._prepare_features(transactions)
            y = [t.category.id for t in transactions]
            
            # Store category mapping
            self.categories = {t.category.id: t.category.name for t in transactions}
            
            # Train the pipeline
            self.pipeline.fit(X, y)
            
            # Save the trained model
            self.save()
            
            self.logger.info(f"Model trained on {len(transactions)} transactions")
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, transaction):
        """
        Predict the category for a transaction.
        
        Args:
            transaction: Transaction object to categorize
            
        Returns:
            tuple: (category_id, confidence_score)
        """
        try:
            # Prepare features
            X = self._prepare_features(transaction)
            
            # Get prediction probabilities
            probs = self.pipeline.predict_proba(X)
            
            # Get predicted category and confidence
            category_id = self.pipeline.classes_[np.argmax(probs)]
            confidence = np.max(probs)
            
            return category_id, confidence
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def evaluate(self, test_transactions):
        """
        Evaluate the model's performance.
        
        Args:
            test_transactions: List of Transaction objects to evaluate on
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        try:
            # Prepare test data
            X_test = self._prepare_features(test_transactions)
            y_test = [t.category.id for t in test_transactions]
            
            # Get predictions
            y_pred = self.pipeline.predict(X_test)
            
            # Calculate metrics
            accuracy = np.mean(y_pred == y_test)
            
            return {
                'accuracy': accuracy,
                'n_samples': len(test_transactions)
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise 