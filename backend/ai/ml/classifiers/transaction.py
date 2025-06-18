"""
Transaction classifier for categorizing financial transactions.

This classifier uses a scikit-learn Pipeline with a ColumnTransformer to process both text and numeric features:
- The 'description' field is vectorized using TfidfVectorizer.
- Numeric fields ('amount', 'day_of_week', 'day_of_month', 'month') are scaled with StandardScaler.

The entire pipeline is saved and loaded using joblib to preserve the fitted state.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Union
from ai.ml.base import BaseMLModel
from django.db.models import Q
from transactions.models import Transaction, Category

class TransactionClassifier(BaseMLModel):
    """
    Classifier for categorizing financial transactions based on their description
    and other features.

    Features used:
    - description (text, vectorized)
    - amount (numeric)
    - day_of_week (numeric)
    - day_of_month (numeric)
    - month (numeric)

    The model is trained using a RandomForestClassifier.
    The pipeline is saved and loaded as a whole to ensure the fitted state is preserved.
    """
    
    def __init__(self):
        super().__init__('transaction_classifier')
        self.feature_names = ['description', 'amount', 'day_of_week', 'day_of_month', 'month']
        self.text_features = ['description']
        self.numeric_features = ['amount', 'day_of_week', 'day_of_month', 'month']
        self.preprocessor = ColumnTransformer([
            ('text', TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            ), 'description'),
            ('num', StandardScaler(), ['amount', 'day_of_week', 'day_of_month', 'month'])
        ])
        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ))
        ])
        self.categories = None
        self.is_fitted = False
    
    def _prepare_features(self, transactions: Union[Transaction, List[Transaction]]) -> pd.DataFrame:
        """
        Prepare features for training or prediction.
        
        Args:
            transactions: List of Transaction objects or single Transaction
            
        Returns:
            pd.DataFrame: Prepared features
            
        Raises:
            ValueError: If transaction data is invalid
        """
        if not isinstance(transactions, list):
            transactions = [transactions]
            
        try:
            features = pd.DataFrame({
                'description': [t.description for t in transactions],
                'amount': [float(t.amount) for t in transactions],
                'day_of_week': [t.date.weekday() for t in transactions],
                'day_of_month': [t.date.day for t in transactions],
                'month': [t.date.month for t in transactions]
            })
            return features
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid transaction data: {str(e)}")
    
    def train(self, transactions: List[Transaction]) -> None:
        """
        Train the transaction classifier.
        
        Args:
            transactions: List of Transaction objects to train on
            
        Raises:
            ValueError: If no transactions provided or data is invalid
            RuntimeError: If training fails
        """
        try:
            if not transactions:
                raise ValueError("No transactions provided for training")
                
            # Prepare features and labels
            X = self._prepare_features(transactions)
            y = np.array([t.category.id for t in transactions])
            
            # Verify data consistency
            if len(X) != len(y):
                raise ValueError(f"Inconsistent data: {len(X)} features vs {len(y)} labels")
            
            # Store category mapping
            self.categories = {t.category.id: t.category.name for t in transactions}
            
            # Train the pipeline
            self.pipeline.fit(X, y)
            self.is_fitted = True
            
            # Save the trained model
            self.save()
            
            self.logger.info(f"Model trained on {len(transactions)} transactions")
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise RuntimeError(f"Failed to train model: {str(e)}")
    
    def predict(self, transaction: Transaction) -> Tuple[int, float]:
        """
        Predict the category for a transaction.
        
        Args:
            transaction: Transaction object to categorize
            
        Returns:
            tuple: (category_id, confidence_score)
            
        Raises:
            RuntimeError: If model is not trained
            ValueError: If transaction is invalid
        """
        try:
            if not self.is_fitted:
                raise RuntimeError("Model must be trained before making predictions")
                
            if transaction is None:
                raise ValueError("Transaction cannot be None")
                
            # Prepare features
            X = self._prepare_features(transaction)
            
            # Get prediction probabilities
            probs = self.pipeline.predict_proba(X)
            
            # Get predicted category and confidence
            category_id = self.pipeline.classes_[np.argmax(probs)]
            confidence = float(np.max(probs))
            
            return int(category_id), confidence
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            raise RuntimeError(f"Failed to make prediction: {str(e)}")
    
    def _calculate_metrics(self, predictions: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: Model predictions
            true_labels: True labels
            
        Returns:
            dict: Dictionary of metric names and values
        """
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, 
            predictions, 
            average='weighted'
        )
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
    
    def evaluate(self, test_transactions: List[Transaction]) -> Dict[str, float]:
        """
        Evaluate the model's performance.
        
        Args:
            test_transactions: List of Transaction objects to evaluate on
            
        Returns:
            dict: Dictionary containing evaluation metrics
            
        Raises:
            RuntimeError: If model is not trained
            ValueError: If test data is invalid
        """
        try:
            if not self.is_fitted:
                raise RuntimeError("Model must be trained before evaluation")
                
            if not test_transactions:
                raise ValueError("No test transactions provided")
                
            # Prepare test data
            X_test = self._prepare_features(test_transactions)
            y_test = np.array([t.category.id for t in test_transactions])
            
            # Get predictions
            y_pred = self.pipeline.predict(X_test)
            
            # Calculate metrics
            return self._calculate_metrics(y_pred, y_test)
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise RuntimeError(f"Failed to evaluate model: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            dict: Dictionary containing model information
        """
        info = super().get_model_info()
        info.update({
            'feature_names': self.feature_names,
            'categories': self.categories,
            'pipeline_steps': [step[0] for step in self.pipeline.steps]
        })
        return info

    def save(self):
        """
        Save the trained pipeline to disk using joblib.
        This ensures that the entire preprocessing and model pipeline, including fitted parameters, is preserved.
        """
        import joblib
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.pipeline, self.model_path)
            self.logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load(self):
        """
        Load the trained pipeline from disk using joblib.
        This restores the entire preprocessing and model pipeline, ready for predictions.
        """
        import joblib
        try:
            if self.model_path.exists():
                self.pipeline = joblib.load(self.model_path)
                self.logger.info(f"Model loaded from {self.model_path}")
                self.is_fitted = True
            else:
                self.logger.warning(f"No saved model found at {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def reset(self):
        """
        Reset the classifier to its initial state.
        """
        self.is_fitted = False
        self.categories = None
        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ))
        ])
        self.save()
        self.logger.info("Model reset") 