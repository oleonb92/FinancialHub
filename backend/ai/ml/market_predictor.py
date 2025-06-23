"""
Sistema de Predicción de Mercado para Fintech.

Este módulo implementa un sistema avanzado de predicción de mercado que incluye:
- Análisis de tendencias de mercado
- Predicción de precios de activos
- Análisis de sentimiento del mercado
- Machine Learning para predicciones
- Análisis de riesgo de mercado
- Alertas y recomendaciones
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from django.utils import timezone
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None
import requests
import json
from textblob import TextBlob
import re

logger = logging.getLogger('ai.market_predictor')

class MarketPredictor:
    """
    Sistema principal de predicción de mercado.
    
    Características:
    - Predicción de precios de activos
    - Análisis de tendencias
    - Análisis de sentimiento
    - Machine Learning adaptativo
    - Alertas de mercado
    - Recomendaciones de inversión
    """
    
    def __init__(self, model_path: str = 'backend/ml_models/market_predictor.joblib'):
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
        self.price_predictor = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        self.trend_predictor = GradientBoostingRegressor(
            n_estimators=100,
            random_state=42
        )
        self.is_trained = False
        self.feature_names = []
        
        # Configuración de predicción
        self.prediction_config = {
            'prediction_horizon': 30,  # Días a predecir
            'lookback_period': 90,     # Días de historial
            'update_frequency': 'daily',  # Frecuencia de actualización
            'confidence_threshold': 0.7,  # Umbral de confianza
            'risk_threshold': 0.3      # Umbral de riesgo
        }
        
        # Fuentes de datos
        self.data_sources = {
            'yahoo_finance': True,
            'alpha_vantage': False,  # Requiere API key
            'news_api': False,       # Requiere API key
            'social_media': False    # Requiere API key
        }
        
        # Cargar modelo si existe
        self.load()
        
    def fetch_market_data(self, symbol: str, period: str = '1y') -> pd.DataFrame:
        """
        Obtiene datos de mercado desde Yahoo Finance.
        
        Args:
            symbol: Símbolo del activo (ej: 'AAPL', 'MSFT')
            period: Período de datos ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            
        Returns:
            DataFrame: Datos de mercado
        """
        if not YFINANCE_AVAILABLE:
            logger.warning("yfinance not available. Market data fetching disabled.")
            return pd.DataFrame()
            
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                logger.warning(f"No data found for symbol: {symbol}")
                return pd.DataFrame()
            
            # Calcular indicadores técnicos
            data = self._add_technical_indicators(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega indicadores técnicos a los datos de mercado.
        
        Args:
            data: DataFrame con datos de mercado
            
        Returns:
            DataFrame: Datos con indicadores técnicos
        """
        try:
            # Medias móviles
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['EMA_12'] = data['Close'].ewm(span=12).mean()
            data['EMA_26'] = data['Close'].ewm(span=26).mean()
            
            # MACD
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            data['BB_Middle'] = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
            data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
            
            # Volatilidad
            data['Volatility'] = data['Close'].rolling(window=20).std()
            data['Volatility_Ratio'] = data['Volatility'] / data['Close']
            
            # Momentum
            data['Momentum'] = data['Close'] - data['Close'].shift(10)
            data['Rate_of_Change'] = (data['Close'] / data['Close'].shift(10) - 1) * 100
            
            # Volumen
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
            
            # Características de precio
            data['Price_Range'] = data['High'] - data['Low']
            data['Price_Range_Ratio'] = data['Price_Range'] / data['Close']
            data['Gap'] = data['Open'] - data['Close'].shift(1)
            data['Gap_Ratio'] = data['Gap'] / data['Close'].shift(1)
            
            return data
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            return data
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extrae características para predicción de mercado.
        
        Args:
            data: DataFrame con datos de mercado
            
        Returns:
            DataFrame: Características extraídas
        """
        try:
            features = pd.DataFrame()
            
            # Características de precio
            features['close'] = data['Close']
            features['open'] = data['Open']
            features['high'] = data['High']
            features['low'] = data['Low']
            features['volume'] = data['Volume']
            
            # Características de retornos
            features['returns'] = data['Close'].pct_change()
            features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
            features['returns_5d'] = data['Close'].pct_change(periods=5)
            features['returns_20d'] = data['Close'].pct_change(periods=20)
            
            # Indicadores técnicos
            technical_cols = ['SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'MACD', 
                            'MACD_Signal', 'MACD_Histogram', 'RSI', 'BB_Middle',
                            'BB_Upper', 'BB_Lower', 'BB_Width', 'Volatility',
                            'Volatility_Ratio', 'Momentum', 'Rate_of_Change',
                            'Volume_SMA', 'Volume_Ratio', 'Price_Range',
                            'Price_Range_Ratio', 'Gap', 'Gap_Ratio']
            
            for col in technical_cols:
                if col in data.columns:
                    features[col] = data[col]
            
            # Características de tendencia
            features['trend_5d'] = (data['Close'] > data['Close'].shift(5)).astype(int)
            features['trend_20d'] = (data['Close'] > data['Close'].shift(20)).astype(int)
            features['trend_50d'] = (data['Close'] > data['Close'].shift(50)).astype(int)
            
            # Características de volatilidad
            features['volatility_5d'] = data['Close'].rolling(window=5).std()
            features['volatility_20d'] = data['Close'].rolling(window=20).std()
            
            # Características de volumen
            features['volume_trend'] = (data['Volume'] > data['Volume'].shift(1)).astype(int)
            features['volume_surge'] = (data['Volume'] > data['Volume'].rolling(window=20).mean() * 2).astype(int)
            
            # Características temporales
            features['day_of_week'] = data.index.dayofweek
            features['month'] = data.index.month
            features['quarter'] = data.index.quarter
            features['is_month_end'] = data.index.is_month_end.astype(int)
            features['is_quarter_end'] = data.index.is_quarter_end.astype(int)
            
            # Eliminar filas con valores NaN
            features = features.dropna()
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return pd.DataFrame()
    
    def prepare_training_data(self, data: pd.DataFrame, target_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara datos para entrenamiento.
        
        Args:
            data: DataFrame con características
            target_horizon: Horizonte de predicción en días
            
        Returns:
            Tuple: Features y target
        """
        try:
            # Crear target (precio futuro)
            target = data['close'].shift(-target_horizon)
            
            # Eliminar filas donde no hay target
            valid_indices = ~target.isna()
            features = data[valid_indices].drop(['close'], axis=1)
            target = target[valid_indices]
            
            return features, target
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return pd.DataFrame(), pd.Series()
    
    def train(self, symbol: str, period: str = '2y'):
        """
        Entrena el modelo de predicción de mercado.
        
        Args:
            symbol: Símbolo del activo
            period: Período de datos para entrenamiento
        """
        try:
            # Obtener datos
            data = self.fetch_market_data(symbol, period)
            if data.empty:
                logger.error(f"No data available for training on {symbol}")
                return
            
            # Extraer características
            features = self.extract_features(data)
            if features.empty:
                logger.error("No features extracted")
                return
            
            # Preparar datos de entrenamiento
            X, y = self.prepare_training_data(features, self.prediction_config['prediction_horizon'])
            if X.empty or y.empty:
                logger.error("No training data prepared")
                return
            
            # Dividir datos (time series split)
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Entrenar modelo de precio
            self.price_predictor.fit(X, y)
            
            # Entrenar modelo de tendencia
            trend_target = (y > y.shift(1)).astype(int)
            self.trend_predictor.fit(X, trend_target)
            
            # Escalar características
            self.scaler.fit(X)
            
            # Guardar nombres de características
            self.feature_names = X.columns.tolist()
            
            # Evaluar modelo
            y_pred = self.price_predictor.predict(X)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            logger.info(f"Model trained for {symbol}")
            logger.info(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
            self.is_trained = True
            self.save()
            
        except Exception as e:
            logger.error(f"Error training market predictor: {str(e)}")
            raise
    
    def predict_price(self, symbol: str, days_ahead: int = 30) -> Dict[str, Any]:
        """
        Predice el precio de un activo.
        
        Args:
            symbol: Símbolo del activo
            days_ahead: Días a predecir
            
        Returns:
            dict: Predicciones y métricas
        """
        try:
            if not self.is_trained:
                return {
                    'error': 'Model not trained',
                    'status': 'not_ready'
                }
            
            # Obtener datos recientes
            data = self.fetch_market_data(symbol, '6mo')
            if data.empty:
                return {
                    'error': 'No data available',
                    'status': 'no_data'
                }
            
            # Extraer características
            features = self.extract_features(data)
            if features.empty:
                return {
                    'error': 'No features extracted',
                    'status': 'no_features'
                }
            
            # Usar los datos más recientes
            latest_features = features.iloc[-1:].drop(['close'], axis=1)
            
            # Escalar características
            latest_features_scaled = self.scaler.transform(latest_features)
            
            # Predicciones
            price_prediction = self.price_predictor.predict(latest_features_scaled)[0]
            trend_prediction = self.trend_predictor.predict(latest_features_scaled)[0]
            trend_probability = self.trend_predictor.predict_proba(latest_features_scaled)[0]
            
            # Calcular confianza
            confidence = max(trend_probability)
            
            # Calcular riesgo
            current_price = data['Close'].iloc[-1]
            price_change = abs(price_prediction - current_price) / current_price
            risk_score = min(price_change * 10, 1.0)  # Normalizar a 0-1
            
            return {
                'symbol': symbol,
                'current_price': float(current_price),
                'predicted_price': float(price_prediction),
                'price_change_percent': float((price_prediction - current_price) / current_price * 100),
                'trend_prediction': 'up' if trend_prediction == 1 else 'down',
                'trend_confidence': float(confidence),
                'prediction_confidence': float(confidence),
                'risk_score': float(risk_score),
                'prediction_date': timezone.now().isoformat(),
                'horizon_days': days_ahead,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error predicting price for {symbol}: {str(e)}")
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def analyze_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Analiza el sentimiento del mercado para un activo.
        
        Args:
            symbol: Símbolo del activo
            
        Returns:
            dict: Análisis de sentimiento
        """
        try:
            # Por ahora, un análisis básico basado en indicadores técnicos
            data = self.fetch_market_data(symbol, '3mo')
            if data.empty:
                return {'error': 'No data available'}
            
            # Calcular sentimiento basado en indicadores
            sentiment_score = 0
            signals = []
            
            # RSI
            current_rsi = data['RSI'].iloc[-1]
            if current_rsi > 70:
                sentiment_score -= 0.3
                signals.append('RSI indicates overbought')
            elif current_rsi < 30:
                sentiment_score += 0.3
                signals.append('RSI indicates oversold')
            
            # MACD
            current_macd = data['MACD'].iloc[-1]
            current_signal = data['MACD_Signal'].iloc[-1]
            if current_macd > current_signal:
                sentiment_score += 0.2
                signals.append('MACD bullish signal')
            else:
                sentiment_score -= 0.2
                signals.append('MACD bearish signal')
            
            # Bollinger Bands
            current_price = data['Close'].iloc[-1]
            bb_upper = data['BB_Upper'].iloc[-1]
            bb_lower = data['BB_Lower'].iloc[-1]
            
            if current_price > bb_upper:
                sentiment_score -= 0.2
                signals.append('Price above upper Bollinger Band')
            elif current_price < bb_lower:
                sentiment_score += 0.2
                signals.append('Price below lower Bollinger Band')
            
            # Moving Averages
            sma_20 = data['SMA_20'].iloc[-1]
            sma_50 = data['SMA_50'].iloc[-1]
            
            if current_price > sma_20 > sma_50:
                sentiment_score += 0.3
                signals.append('Strong uptrend')
            elif current_price < sma_20 < sma_50:
                sentiment_score -= 0.3
                signals.append('Strong downtrend')
            
            # Normalizar sentimiento a -1 a 1
            sentiment_score = max(-1, min(1, sentiment_score))
            
            # Determinar sentimiento
            if sentiment_score > 0.3:
                sentiment = 'bullish'
            elif sentiment_score < -0.3:
                sentiment = 'bearish'
            else:
                sentiment = 'neutral'
            
            return {
                'symbol': symbol,
                'sentiment_score': float(sentiment_score),
                'sentiment': sentiment,
                'signals': signals,
                'current_price': float(current_price),
                'rsi': float(current_rsi),
                'macd': float(current_macd),
                'analysis_date': timezone.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {str(e)}")
            return {'error': str(e)}
    
    def get_market_recommendations(self, symbol: str) -> Dict[str, Any]:
        """
        Genera recomendaciones de mercado.
        
        Args:
            symbol: Símbolo del activo
            
        Returns:
            dict: Recomendaciones
        """
        try:
            # Obtener predicción
            prediction = self.predict_price(symbol)
            if 'error' in prediction:
                return prediction
            
            # Obtener sentimiento
            sentiment = self.analyze_market_sentiment(symbol)
            if 'error' in sentiment:
                return sentiment
            
            # Generar recomendación
            recommendation = self._generate_recommendation(prediction, sentiment)
            
            return {
                'symbol': symbol,
                'prediction': prediction,
                'sentiment': sentiment,
                'recommendation': recommendation,
                'generated_at': timezone.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting market recommendations: {str(e)}")
            return {'error': str(e)}
    
    def _generate_recommendation(self, prediction: Dict[str, Any], sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera recomendación basada en predicción y sentimiento.
        
        Args:
            prediction: Resultados de predicción
            sentiment: Análisis de sentimiento
            
        Returns:
            dict: Recomendación
        """
        try:
            # Combinar señales
            price_signal = prediction.get('trend_prediction', 'neutral')
            sentiment_signal = sentiment.get('sentiment', 'neutral')
            confidence = prediction.get('prediction_confidence', 0.5)
            risk = prediction.get('risk_score', 0.5)
            
            # Determinar acción recomendada
            if price_signal == 'up' and sentiment_signal == 'bullish' and confidence > 0.7:
                action = 'buy'
                strength = 'strong'
            elif price_signal == 'down' and sentiment_signal == 'bearish' and confidence > 0.7:
                action = 'sell'
                strength = 'strong'
            elif price_signal == 'up' and confidence > 0.6:
                action = 'buy'
                strength = 'moderate'
            elif price_signal == 'down' and confidence > 0.6:
                action = 'sell'
                strength = 'moderate'
            else:
                action = 'hold'
                strength = 'weak'
            
            # Ajustar por riesgo
            if risk > 0.7:
                action = 'hold'
                strength = 'high_risk'
            
            # Generar mensaje
            messages = {
                'buy': 'Consider buying this asset based on positive technical indicators and market sentiment.',
                'sell': 'Consider selling this asset based on negative technical indicators and market sentiment.',
                'hold': 'Maintain current position. Market conditions are uncertain or neutral.'
            }
            
            return {
                'action': action,
                'strength': strength,
                'confidence': float(confidence),
                'risk_level': 'high' if risk > 0.7 else 'medium' if risk > 0.4 else 'low',
                'message': messages.get(action, 'No clear recommendation available.'),
                'price_target': prediction.get('predicted_price'),
                'time_horizon': prediction.get('horizon_days', 30)
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {str(e)}")
            return {
                'action': 'hold',
                'strength': 'unknown',
                'message': 'Unable to generate recommendation due to error.'
            }
    
    def save(self):
        """Guarda el modelo entrenado."""
        try:
            model_data = {
                'price_predictor': self.price_predictor,
                'trend_predictor': self.trend_predictor,
                'scaler': self.scaler,
                'is_trained': self.is_trained,
                'feature_names': self.feature_names,
                'prediction_config': self.prediction_config
            }
            
            joblib.dump(model_data, self.model_path)
            logger.info(f"Market predictor model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving market predictor model: {str(e)}")
            raise
    
    def load(self):
        """Carga el modelo entrenado."""
        try:
            model_data = joblib.load(self.model_path)
            
            self.price_predictor = model_data['price_predictor']
            self.trend_predictor = model_data['trend_predictor']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            self.feature_names = model_data['feature_names']
            self.prediction_config = model_data.get('prediction_config', self.prediction_config)
            
            logger.info("Market predictor model loaded successfully")
            
        except FileNotFoundError:
            logger.info("No pre-trained market predictor model found")
        except Exception as e:
            logger.error(f"Error loading market predictor model: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtiene información del modelo.
        
        Returns:
            dict: Información del modelo
        """
        return {
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'prediction_config': self.prediction_config,
            'model_path': self.model_path
        } 