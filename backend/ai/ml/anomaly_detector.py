"""
Detector de anomalías para transacciones financieras.

Este módulo implementa un detector de anomalías utilizando el algoritmo Isolation Forest
de scikit-learn para identificar transacciones inusuales en los datos financieros.
"""
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from transactions.models import Transaction
from datetime import datetime
import logging
from typing import Union, List, Dict

logger = logging.getLogger('ai.anomaly_detector')

class AnomalyDetector:
    """
    Detector de anomalías para transacciones financieras.
    
    Utiliza Isolation Forest para detectar transacciones inusuales basándose en:
    - Monto de la transacción
    - Día de la semana
    - Día del mes
    - Mes
    - Categoría
    - Hora
    """
    
    def __init__(self, contamination=0.1, random_state=42, n_estimators=100):
        """
        Inicializa el detector de anomalías.
        
        Args:
            contamination: Proporción esperada de anomalías en los datos
            random_state: Semilla aleatoria para reproducibilidad
            n_estimators: Número de árboles en el bosque
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators
        )
        self.is_fitted = False
        
    def prepare_features(self, transactions):
        """
        Prepara las características para el modelo.
        
        Args:
            transactions: Lista de objetos Transaction
            
        Returns:
            DataFrame: Características preparadas
        """
        if not transactions:
            return pd.DataFrame()
        if isinstance(transactions[0], dict):
            # Diccionarios
            dates = [t.get('date') for t in transactions]
            features = pd.DataFrame({
                'amount': [float(t.get('amount', 0)) for t in transactions],
                'day_of_week': [d.weekday() if d is not None else 0 for d in dates],
                'day_of_month': [d.day if d is not None else 1 for d in dates],
                'month': [d.month if d is not None else 1 for d in dates],
                'category_id': [t.get('category_id', 0) for t in transactions],
                'merchant_hash': [hash(t.get('merchant', '')) % 1000 for t in transactions],
            })
        else:
            # Objetos Transaction
            dates = [t.date for t in transactions]
            features = pd.DataFrame({
                'amount': [float(t.amount) for t in transactions],
                'day_of_week': [d.weekday() if d is not None else 0 for d in dates],
                'day_of_month': [d.day if d is not None else 1 for d in dates],
                'month': [d.month if d is not None else 1 for d in dates],
                'category_id': [t.category.id if t.category else 0 for t in transactions],
                'merchant_hash': [hash(t.merchant or '') % 1000 for t in transactions],
            })
        return features
    
    def train(self, transactions: Union[List[Transaction], List[Dict]]) -> None:
        """
        Entrena el modelo con las transacciones proporcionadas.
        
        Args:
            transactions: Lista de objetos Transaction
            
        Raises:
            ValueError: Si no hay transacciones para entrenar
        """
        try:
            if not transactions:
                raise ValueError("No transactions provided for training")
            features = self.prepare_features(transactions)
            if features.empty:
                raise ValueError("No valid features extracted from transactions")
            # Filtrar filas con NaN en features esenciales
            essential_features = ['amount', 'day_of_week', 'day_of_month', 'month', 'category_id', 'merchant_hash']
            before = len(features)
            mask = features[essential_features].notnull().all(axis=1)
            features = features[mask]
            after = len(features)
            if after < before:
                logger.warning(f"Filtradas {before - after} filas con NaN en features esenciales para entrenamiento de anomalías.")
            if len(features) == 0:
                raise ValueError("No valid data after filtering NaNs in essential features.")
            self.model.fit(features)
            self.is_fitted = True
            self.is_trained = True
            self.save()
            logger.info(f"Model trained on {len(features)} transactions")
        except Exception as e:
            logger.error(f"Error training anomaly detector: {str(e)}")
            raise RuntimeError(f"Input X contains NaN.")
    
    def detect_anomalies(self, transactions):
        """
        Detecta anomalías en las transacciones proporcionadas.
        
        Args:
            transactions: Lista de objetos Transaction
            
        Returns:
            list: Lista de anomalías detectadas
        """
        if not self.is_fitted:
            logger.warning("Model not fitted, attempting to train")
            self.train(transactions)
            
        try:
            features = self.prepare_features(transactions)
            scores = self.model.score_samples(features)
            predictions = self.model.predict(features)
            
            anomalies = []
            for i, (t, score, pred) in enumerate(zip(transactions, scores, predictions)):
                if pred == -1:  # Anomalía detectada
                    anomalies.append({
                        'transaction_id': t.id,
                        'amount': float(t.amount),
                        'date': t.date.isoformat(),
                        'category': t.category.name if t.category else 'Unknown',
                        'description': t.description,
                        'anomaly_score': float(score),
                        'reason': self._explain_anomaly(t, score)
                    })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return []
    
    def _explain_anomaly(self, transaction, score):
        """
        Genera una explicación para una anomalía detectada.
        
        Args:
            transaction: Transacción anómala
            score: Puntuación de anomalía
            
        Returns:
            str: Explicación de la anomalía
        """
        reasons = []
        
        # Verificar si el monto es inusual
        if abs(float(transaction.amount)) > 1000:
            reasons.append("unusual amount")
            
        # Verificar si la hora es inusual
        if transaction.date.hour < 6 or transaction.date.hour > 22:
            reasons.append("unusual time")
            
        # Verificar si el día de la semana es inusual
        if transaction.date.weekday() in [5, 6]:  # Fin de semana
            reasons.append("weekend transaction")
            
        return " and ".join(reasons) if reasons else "unusual pattern" 
    
    def save(self):
        """
        Guarda el modelo entrenado.
        """
        try:
            model_path = Path('backend/ml_models/anomaly_detector.joblib')
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise RuntimeError(f"Failed to save model: {str(e)}")
    
    def load(self):
        """
        Carga el modelo entrenado.
        """
        try:
            model_path = Path('backend/ml_models/anomaly_detector.joblib')
            if model_path.exists():
                self.model = joblib.load(model_path)
                self.is_fitted = True
                logger.info(f"Model loaded from {model_path}")
            else:
                logger.warning(f"No saved model found at {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def reset(self):
        """
        Resetea el modelo.
        """
        self.is_fitted = False
        self.model = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        logger.info("Model reset") 