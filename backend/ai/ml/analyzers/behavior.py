"""
Behavior analyzer for identifying spending patterns and anomalies.
"""
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from ai.ml.base import BaseMLModel
from django.db.models import Q
from transactions.models import Transaction
from datetime import timedelta
import joblib
import logging

logger = logging.getLogger('ai.ml')

class BehaviorAnalyzer(BaseMLModel):
    """
    Analyzer for identifying spending patterns and anomalies in transaction data.
    """
    
    def __init__(self):
        super().__init__('behavior_analyzer')
        self.scaler = StandardScaler()
        self.clustering_model = DBSCAN(
            eps=0.5,
            min_samples=5,
            metric='euclidean'
        )
        self.feature_names = ['amount', 'day_of_week', 'hour', 'category_id', 'merchant_id']
        self.is_fitted = False
    
    def _prepare_features(self, transactions: Union[List[Transaction], List[Dict]]) -> pd.DataFrame:
        """
        Prepare features for behavior analysis.
        
        Args:
            transactions: List of Transaction objects or dictionaries
            
        Returns:
            pd.DataFrame: Prepared features
            
        Raises:
            ValueError: If transaction data is invalid
        """
        if not transactions:
            return pd.DataFrame(columns=self.feature_names)
        
        try:
            # Check if we're dealing with dictionaries or Transaction objects
            if transactions and isinstance(transactions[0], dict):
                # Handle dictionary format
                return pd.DataFrame({
                    'amount': [float(t.get('amount', 0)) for t in transactions],
                    'day_of_week': [t.get('date').weekday() if hasattr(t.get('date'), 'weekday') else 0 for t in transactions],
                    'hour': [t.get('date').hour if hasattr(t.get('date'), 'hour') else 0 for t in transactions],
                    'category_id': [t.get('category_id', 0) for t in transactions],
                    'merchant_id': [hash(t.get('merchant', '')) % 1000 for t in transactions]
                })
            else:
                # Handle Transaction objects
                return pd.DataFrame({
                    'amount': [float(t.amount) for t in transactions],
                    'day_of_week': [t.date.weekday() for t in transactions],
                    'hour': [t.date.hour for t in transactions],
                    'category_id': [t.category.id if t.category else 0 for t in transactions],
                    'merchant_id': [hash(getattr(t, 'merchant', None)) % 1000 if getattr(t, 'merchant', None) else 0 for t in transactions]
                })
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid transaction data: {str(e)}")
    
    def train(self, transactions: Union[List[Transaction], List[Dict]]) -> None:
        """
        Train the behavior analyzer.
        
        Args:
            transactions: List of Transaction objects or dictionaries to train on
            
        Raises:
            ValueError: If no transactions provided or data is invalid
            RuntimeError: If training fails
        """
        try:
            if not transactions:
                raise ValueError("No transactions provided for training")
                
            # Prepare features
            features = self._prepare_features(transactions)
            
            if features.empty:
                raise ValueError("No valid features extracted from transactions")
            
            # Filtrar filas con NaN en features esenciales
            essential_features = ['amount', 'day_of_week', 'hour', 'category_id', 'merchant_id']
            before = len(features)
            mask = features[essential_features].notnull().all(axis=1)
            features = features[mask]
            after = len(features)
            if after < before:
                self.logger.warning(f"Filtradas {before - after} filas con NaN en features esenciales para entrenamiento de comportamiento.")
            if len(features) == 0:
                raise ValueError("No valid data after filtering NaNs in essential features.")
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Fit clustering model
            self.clustering_model.fit(scaled_features)
            self.is_fitted = True
            self.is_trained = True  # Set the base class flag
            
            # Save the trained model
            self.save()
            
            self.logger.info(f"Model trained on {len(features)} transactions")
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise RuntimeError(f"Failed to train model: {str(e)}")
    
    def predict(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """
        Predict spending patterns and anomalies.
        
        Args:
            transactions: List of Transaction objects to analyze
            
        Returns:
            dict: Dictionary containing pattern analysis results
            
        Raises:
            RuntimeError: If model is not trained
            ValueError: If transaction data is invalid
        """
        try:
            if not self.is_fitted:
                raise RuntimeError("Model must be trained before making predictions")
                
            return self.analyze_spending_patterns(transactions)
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            raise RuntimeError(f"Failed to make prediction: {str(e)}")
    
    def _detect_anomalies(self, features: pd.DataFrame) -> np.ndarray:
        """
        Detect anomalies in transaction data.
        
        Args:
            features: DataFrame of transaction features
            
        Returns:
            np.array: Boolean array indicating anomalies
            
        Raises:
            ValueError: If features are invalid
        """
        if features.empty:
            return np.array([])
        
        try:
            if not self.is_fitted:
                # If not trained, fit the scaler and model
                scaled_features = self.scaler.fit_transform(features)
                self.clustering_model.fit(scaled_features)
                self.is_fitted = True
            else:
                scaled_features = self.scaler.transform(features)
                
            clusters = self.clustering_model.fit_predict(scaled_features)
            return clusters == -1
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {str(e)}")
            raise ValueError(f"Failed to detect anomalies: {str(e)}")
    
    def analyze_spending_patterns(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """
        Analyze spending patterns in transaction data.
        
        Args:
            transactions: List of Transaction objects
            
        Returns:
            dict: Dictionary containing pattern analysis results
            
        Raises:
            ValueError: If transaction data is invalid
            RuntimeError: If analysis fails
        """
        try:
            # Prepare features
            features = self._prepare_features(transactions)
            
            if features.empty:
                return {
                    'category_patterns': {},
                    'overall_patterns': {
                        'total_transactions': 0,
                        'total_spent': 0.0,
                        'avg_transaction': 0.0,
                        'anomalies': 0,
                        'spending_trend': {},
                        'category_distribution': {}
                    }
                }
            
            # Detect anomalies
            anomalies = self._detect_anomalies(features)
            
            # Analyze patterns
            category_patterns = {}
            for category_id in features['category_id'].unique():
                idx = features.index[features['category_id'] == category_id].tolist()
                category_transactions = [transactions[i] for i in idx]
                category_features = features.loc[idx]
                pattern = {
                    'total_spent': float(category_features['amount'].sum()),
                    'avg_amount': float(category_features['amount'].mean()),
                    'frequency': len(category_transactions),
                    'anomalies': int(anomalies[features['category_id'] == category_id].sum()),
                    'preferred_days': self._get_preferred_days(category_features),
                    'preferred_hours': self._get_preferred_hours(category_features)
                }
                category_patterns[category_id] = pattern
            
            # Analyze overall patterns
            overall_patterns = {
                'total_transactions': len(transactions),
                'total_spent': float(features['amount'].sum()),
                'avg_transaction': float(features['amount'].mean()),
                'anomalies': int(anomalies.sum()),
                'spending_trend': self._analyze_spending_trend(transactions),
                'category_distribution': self._analyze_category_distribution(features)
            }
            
            return {
                'category_patterns': category_patterns,
                'overall_patterns': overall_patterns
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing spending patterns: {str(e)}")
            raise RuntimeError(f"Failed to analyze spending patterns: {str(e)}")
    
    def _get_preferred_days(self, features: pd.DataFrame) -> Dict[str, int]:
        """
        Get preferred days of the week for transactions.
        
        Args:
            features: DataFrame of transaction features
            
        Returns:
            dict: Dictionary of day frequencies
        """
        day_counts = features['day_of_week'].value_counts()
        return {
            str(day): int(count)
            for day, count in day_counts.items()
        }
    
    def _get_preferred_hours(self, features: pd.DataFrame) -> Dict[int, int]:
        """
        Get preferred hours of the day for transactions.
        
        Args:
            features: DataFrame of transaction features
            
        Returns:
            dict: Dictionary of hour frequencies
        """
        hour_counts = features['hour'].value_counts()
        return {
            int(hour): int(count)
            for hour, count in hour_counts.items()
        }
    
    def _analyze_spending_trend(self, transactions: Union[List[Transaction], List[Dict]]) -> Dict[str, float]:
        """
        Analyze spending trend over time.
        
        Args:
            transactions: List of Transaction objects or dictionaries
            
        Returns:
            dict: Dictionary containing trend analysis
        """
        try:
            # Handle both Transaction objects and dictionaries
            if transactions and isinstance(transactions[0], dict):
                # Handle dictionary format
                daily_amounts = pd.DataFrame({
                    'date': [t.get('date') for t in transactions if t.get('date')],
                    'amount': [float(t.get('amount', 0)) for t in transactions if t.get('date')]
                })
            else:
                # Handle Transaction objects
                daily_amounts = pd.DataFrame({
                    'date': [t.date for t in transactions],
                    'amount': [float(t.amount) for t in transactions]
                })
            
            if daily_amounts.empty:
                return {
                    'trend_coefficient': 0.0,
                    'trend_direction': 'stable',
                    'daily_average': 0.0
                }
            
            # Group by date and sum amounts
            daily_amounts = daily_amounts.groupby('date')['amount'].sum().reset_index()
            
            # Calculate trend
            if len(daily_amounts) > 1:
                trend = np.polyfit(
                    range(len(daily_amounts)),
                    daily_amounts['amount'],
                    deg=1
                )[0]
            else:
                trend = 0
            
            return {
                'trend_coefficient': float(trend),
                'trend_direction': 'increasing' if trend > 0 else 'decreasing',
                'daily_average': float(daily_amounts['amount'].mean())
            }
                
        except Exception as e:
            self.logger.error(f"Error analyzing spending trend: {str(e)}")
            return {
                'trend_coefficient': 0.0,
                'trend_direction': 'stable',
                'daily_average': 0.0
            }
    
    def _analyze_category_distribution(self, features: pd.DataFrame) -> Dict[int, float]:
        """
        Analyze distribution of spending across categories.
        
        Args:
            features: DataFrame of transaction features
            
        Returns:
            dict: Dictionary containing category distribution
        """
        category_totals = features.groupby('category_id')['amount'].sum()
        total_spent = category_totals.sum()
        
        return {
            int(category_id): float(amount / total_spent)
            for category_id, amount in category_totals.items()
        }
    
    def _calculate_metrics(self, predictions: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: Model predictions (cluster assignments)
            true_labels: True labels (not used for clustering)
            
        Returns:
            dict: Dictionary of metric names and values
        """
        # For clustering, we can use silhouette score if there are multiple clusters
        if len(np.unique(predictions)) > 1:
            silhouette = silhouette_score(true_labels, predictions)
        else:
            silhouette = 0.0
            
        return {
            'silhouette_score': float(silhouette),
            'n_clusters': int(len(np.unique(predictions))),
            'n_noise': int(np.sum(predictions == -1))
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
            features = self._prepare_features(test_transactions)
            
            if features.empty:
                raise ValueError("No valid features extracted from test transactions")
            
            # Scale features
            scaled_features = self.scaler.transform(features)
            
            # Get predictions (cluster assignments)
            predictions = self.clustering_model.fit_predict(scaled_features)
            
            # Calculate metrics
            return self._calculate_metrics(predictions, scaled_features)
            
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
            'model_params': self.clustering_model.get_params()
        })
        return info

    def save(self):
        """Guarda el modelo entrenado."""
        try:
            if not self.is_trained:
                raise RuntimeError("Cannot save untrained model")
                
            # Guardar el modelo completo (scaler + clustering_model)
            model_data = {
                'scaler': self.scaler,
                'clustering_model': self.clustering_model,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted
            }
            
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model_data, self.model_path)
            self.logger.info(f"Model saved to {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise RuntimeError(f"Failed to save model: {str(e)}")
    
    def load(self):
        """Carga el modelo entrenado."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"No saved model found at {self.model_path}")
                
            model_data = joblib.load(self.model_path)
            self.scaler = model_data['scaler']
            self.clustering_model = model_data['clustering_model']
            self.feature_names = model_data['feature_names']
            self.is_fitted = model_data['is_fitted']
            self.is_trained = True
            
            self.logger.info(f"Model loaded from {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def reset(self):
        # ... existing code ...
        self.is_fitted = False
        # ... existing code ... 