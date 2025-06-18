"""
Expense predictor for forecasting future expenses.

This predictor uses a GradientBoostingRegressor and a StandardScaler for feature scaling.
Both the model and the scaler are saved and loaded together using joblib to ensure correct predictions after loading.
"""
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from ai.ml.base import BaseMLModel
from django.db.models import Q
from transactions.models import Transaction
from datetime import timedelta
import joblib

class ExpensePredictor(BaseMLModel):
    """
    Predictor for forecasting future expenses based on historical transaction data.

    Features used:
    - day_of_week
    - day_of_month
    - month
    - category_id

    The model is trained using a GradientBoostingRegressor.
    Both the model and the scaler are saved and loaded together to preserve the fitted state.
    """
    
    def __init__(self):
        super().__init__('expense_predictor')
        self.scaler = StandardScaler()
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.feature_names = ['day_of_week', 'day_of_month', 'month', 'category_id']
        self.is_fitted = False
    
    def _prepare_features(self, transactions: List[Transaction]) -> pd.DataFrame:
        """
        Prepare features for training or prediction.
        
        Args:
            transactions: List of Transaction objects
            
        Returns:
            pd.DataFrame: Prepared features
            
        Raises:
            ValueError: If transaction data is invalid
        """
        try:
            features = pd.DataFrame({
                'day_of_week': [t.date.weekday() for t in transactions],
                'day_of_month': [t.date.day for t in transactions],
                'month': [t.date.month for t in transactions],
                'category_id': [t.category.id for t in transactions],
                'amount': [float(t.amount) for t in transactions]
            })
            return features
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid transaction data: {str(e)}")
    
    def _prepare_sequence_features(self, transactions: List[Transaction], 
                                 sequence_length: int = 30) -> pd.DataFrame:
        """
        Prepare sequence features for time series prediction.
        
        Args:
            transactions: List of Transaction objects
            sequence_length: Number of days to look back
            
        Returns:
            pd.DataFrame: Prepared sequence features
            
        Raises:
            ValueError: If transaction data is invalid or insufficient
        """
        try:
            # Group transactions by date
            daily_amounts = pd.DataFrame({
                'date': [t.date for t in transactions],
                'amount': [float(t.amount) for t in transactions]
            }).groupby('date')['amount'].sum().reset_index()
            
            if len(daily_amounts) < sequence_length + 1:
                raise ValueError(f"Insufficient data: need at least {sequence_length + 1} days")
            
            # Create sequence features
            sequences = []
            for i in range(len(daily_amounts) - sequence_length):
                sequence = daily_amounts.iloc[i:i+sequence_length]
                target = daily_amounts.iloc[i+sequence_length]['amount']
                sequences.append({
                    'sequence': sequence['amount'].values,
                    'target': target
                })
            
            return pd.DataFrame(sequences)
        except Exception as e:
            raise ValueError(f"Error preparing sequence features: {str(e)}")
    
    def train(self, transactions: List[Transaction]) -> None:
        """
        Train the expense predictor.
        
        Args:
            transactions: List of Transaction objects to train on
            
        Raises:
            ValueError: If no transactions provided or data is invalid
            RuntimeError: If training fails
        """
        try:
            if not transactions:
                raise ValueError("No transactions provided for training")
                
            # Prepare features
            X = self._prepare_features(transactions)
            y = X['amount']
            X = X.drop('amount', axis=1)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_fitted = True
            
            # Save the trained model
            self.save()
            
            self.logger.info(f"Model trained on {len(transactions)} transactions")
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise RuntimeError(f"Failed to train model: {str(e)}")
    
    def predict(self, date: pd.Timestamp, category_id: int) -> float:
        """
        Predict expenses for a given date and category.
        
        Args:
            date: Date to predict for
            category_id: Category ID to predict for
            
        Returns:
            float: Predicted expense amount
            
        Raises:
            RuntimeError: If model is not trained
            ValueError: If input data is invalid
        """
        try:
            if not self.is_fitted:
                raise RuntimeError("Model must be trained before making predictions")
                
            # Prepare features
            features = pd.DataFrame({
                'day_of_week': [date.weekday()],
                'day_of_month': [date.day],
                'month': [date.month],
                'category_id': [category_id]
            })
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = float(self.model.predict(features_scaled)[0])
            
            return max(0, prediction)  # Ensure non-negative prediction
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            raise RuntimeError(f"Failed to make prediction: {str(e)}")
    
    def predict_sequence(self, start_date: pd.Timestamp, days: int = 30) -> pd.DataFrame:
        """
        Predict expenses for a sequence of days.
        
        Args:
            start_date: Start date for prediction
            days: Number of days to predict
            
        Returns:
            pd.DataFrame: Predictions for each day
            
        Raises:
            RuntimeError: If model is not trained
            ValueError: If input data is invalid
        """
        try:
            if not self.is_fitted:
                raise RuntimeError("Model must be trained before making predictions")
                
            predictions = []
            current_date = start_date
            
            for _ in range(days):
                # Get all categories
                categories = Transaction.objects.values_list('category_id', flat=True).distinct()
                
                # Predict for each category
                daily_total = 0
                for category_id in categories:
                    prediction = self.predict(current_date, category_id)
                    daily_total += prediction
                
                predictions.append({
                    'date': current_date,
                    'predicted_amount': daily_total
                })
                
                current_date += timedelta(days=1)
            
            return pd.DataFrame(predictions)
            
        except Exception as e:
            self.logger.error(f"Error making sequence prediction: {str(e)}")
            raise RuntimeError(f"Failed to make sequence prediction: {str(e)}")

    def _calculate_metrics(self, predictions: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: Model predictions
            true_labels: True labels
            
        Returns:
            dict: Dictionary of metric names and values
        """
        return {
            'mse': float(mean_squared_error(true_labels, predictions)),
            'rmse': float(np.sqrt(mean_squared_error(true_labels, predictions))),
            'mae': float(mean_absolute_error(true_labels, predictions)),
            'r2': float(r2_score(true_labels, predictions))
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
            y_test = X_test['amount']
            X_test = X_test.drop('amount', axis=1)
            
            # Scale features
            X_test_scaled = self.scaler.transform(X_test)
            
            # Get predictions
            y_pred = self.model.predict(X_test_scaled)
            
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
            'model_params': self.model.get_params()
        })
        return info

    def save(self):
        """
        Save both the trained model and the scaler to disk using joblib.
        This ensures that predictions after loading use the same scaling as during training.
        """
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            model_data = {
                'model': self.model,
                'scaler': self.scaler
            }
            joblib.dump(model_data, self.model_path)
            self.logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load(self):
        """
        Load both the trained model and the scaler from disk using joblib.
        This restores the model and scaler for accurate predictions.
        """
        try:
            if self.model_path.exists():
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.logger.info(f"Model loaded from {self.model_path}")
                self.is_fitted = True
            else:
                self.logger.warning(f"No saved model found at {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def reset(self):
        """
        Reset the model to its initial state.
        """
        self.is_fitted = False
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.logger.info("Model reset") 