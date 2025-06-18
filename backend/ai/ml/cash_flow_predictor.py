"""
Predictor de flujo de efectivo para transacciones financieras.

Este módulo implementa un predictor de flujo de efectivo utilizando Gradient Boosting
para predecir el flujo de efectivo futuro basado en patrones históricos.
"""
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger('ai.cash_flow_predictor')

class CashFlowPredictor:
    """
    Predictor de flujo de efectivo para transacciones financieras.
    
    Utiliza Gradient Boosting para predecir el flujo de efectivo futuro basado en:
    - Montos históricos
    - Patrones temporales
    - Categorías de transacciones
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
        """
        Inicializa el predictor de flujo de efectivo.
        
        Args:
            n_estimators: Número de árboles en el modelo
            learning_rate: Tasa de aprendizaje
            max_depth: Profundidad máxima de los árboles
            random_state: Semilla aleatoria para reproducibilidad
        """
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
        self.is_fitted = False
        self.prediction_days = 30
        
    def prepare_features(self, transactions, date=None):
        """
        Prepara las características para el modelo.
        
        Args:
            transactions: Lista de objetos Transaction
            date: Fecha de referencia para la predicción
            
        Returns:
            DataFrame: Características preparadas
        """
        if date is None:
            date = datetime.now()
            
        # Agrupar transacciones por día
        daily_data = pd.DataFrame({
            'date': [t.date for t in transactions],
            'amount': [float(t.amount) for t in transactions]
        })
        daily_data = daily_data.groupby('date')['amount'].sum().reset_index()
        
        # Crear características temporales
        features = []
        for i in range(len(daily_data)):
            if i >= 7:  # Necesitamos al menos 7 días de historia
                row = {
                    'amount_lag_1': daily_data.iloc[i-1]['amount'],
                    'amount_lag_2': daily_data.iloc[i-2]['amount'],
                    'amount_lag_3': daily_data.iloc[i-3]['amount'],
                    'amount_lag_7': daily_data.iloc[i-7]['amount'],
                    'day_of_week': daily_data.iloc[i]['date'].weekday(),
                    'day_of_month': daily_data.iloc[i]['date'].day,
                    'month': daily_data.iloc[i]['date'].month,
                    'is_weekend': int(daily_data.iloc[i]['date'].weekday() >= 5)
                }
                features.append(row)
                
        return pd.DataFrame(features)
    
    def train(self, transactions):
        """
        Entrena el modelo con las transacciones proporcionadas.
        
        Args:
            transactions: Lista de objetos Transaction
            
        Raises:
            ValueError: Si no hay suficientes transacciones para entrenar
        """
        if len(transactions) < 30:  # Necesitamos al menos 30 días de datos
            raise ValueError("Insufficient transaction data for training")
            
        try:
            features = self.prepare_features(transactions)
            if len(features) < 7:
                raise ValueError("Not enough daily data for training")
                
            # Preparar objetivo (monto del día siguiente)
            target = features['amount_lag_1'].shift(-1).dropna()
            features = features[:-1]  # Eliminar última fila que no tiene target
            
            self.model.fit(features, target)
            self.is_fitted = True
            logger.info("Cash flow predictor trained successfully")
            
        except Exception as e:
            logger.error(f"Error training cash flow predictor: {str(e)}")
            raise
    
    def predict(self, transactions, days=None):
        """
        Predice el flujo de efectivo para los próximos días.
        
        Args:
            transactions: Lista de objetos Transaction
            days: Número de días a predecir
            
        Returns:
            list: Lista de predicciones diarias
        """
        if not self.is_fitted:
            logger.warning("Model not fitted, attempting to train")
            self.train(transactions)
            
        if days is None:
            days = self.prediction_days
            
        try:
            # Preparar características iniciales
            features = self.prepare_features(transactions)
            if len(features) < 7:
                raise ValueError("Not enough data for prediction")
                
            predictions = []
            last_date = max(t.date for t in transactions)
            
            # Generar predicciones para cada día
            for i in range(days):
                pred_date = last_date + timedelta(days=i+1)
                pred_amount = self.model.predict(features.iloc[[-1]])[0]
                confidence = self._calculate_confidence(transactions, pred_amount)
                
                predictions.append({
                    'date': pred_date.isoformat(),
                    'predicted_amount': float(pred_amount),
                    'confidence': float(confidence)
                })
                
                # Actualizar características para la siguiente predicción
                new_row = features.iloc[-1].copy()
                new_row['amount_lag_1'] = pred_amount
                new_row['amount_lag_2'] = features.iloc[-1]['amount_lag_1']
                new_row['amount_lag_3'] = features.iloc[-1]['amount_lag_2']
                new_row['amount_lag_7'] = features.iloc[-7]['amount_lag_1']
                new_row['day_of_week'] = pred_date.weekday()
                new_row['day_of_month'] = pred_date.day
                new_row['month'] = pred_date.month
                new_row['is_weekend'] = int(pred_date.weekday() >= 5)
                
                features = pd.concat([features, pd.DataFrame([new_row])], ignore_index=True)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting cash flow: {str(e)}")
            return []
    
    def _calculate_confidence(self, transactions, predicted_amount):
        """
        Calcula el nivel de confianza para una predicción.
        
        Args:
            transactions: Lista de transacciones históricas
            predicted_amount: Monto predicho
            
        Returns:
            float: Nivel de confianza entre 0 y 1
        """
        try:
            # Calcular estadísticas de las transacciones recientes
            recent_amounts = [float(t.amount) for t in transactions[-30:]]
            mean_amount = np.mean(recent_amounts)
            std_amount = np.std(recent_amounts)
            
            # Calcular confianza basada en la desviación de la media
            z_score = abs(predicted_amount - mean_amount) / (std_amount + 1e-6)
            confidence = 1.0 / (1.0 + z_score)
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5  # Valor por defecto 