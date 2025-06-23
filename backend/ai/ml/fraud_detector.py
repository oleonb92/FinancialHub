"""
Sistema de Detección de Fraude para Fintech.

Este módulo implementa un sistema avanzado de detección de fraude que incluye:
- Análisis de patrones de transacciones
- Detección de anomalías en tiempo real
- Análisis de riesgo de usuario
- Machine Learning para clasificación de fraude
- Reglas de negocio configurables
- Alertas y notificaciones
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from django.utils import timezone
from django.db.models import Q, Count, Avg, Max, Min
from transactions.models import Transaction
from accounts.models import User
import json

logger = logging.getLogger('ai.fraud_detector')

class FraudDetector:
    """
    Sistema principal de detección de fraude.
    
    Características:
    - Detección en tiempo real
    - Análisis de patrones históricos
    - Machine Learning adaptativo
    - Reglas de negocio configurables
    - Sistema de alertas
    """
    
    def __init__(self, model_path: str = 'backend/ml_models/fraud_detector.joblib'):
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.fraud_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        self.is_trained = False
        self.feature_names = []
        
        # Reglas de detección de fraude
        self.fraud_rules = {
            'high_amount_threshold': 10000.0,  # Transacciones de alto valor
            'velocity_threshold': 5,  # Transacciones por hora
            'location_mismatch': True,  # Detectar cambios de ubicación
            'time_anomaly': True,  # Transacciones en horarios inusuales
            'amount_pattern': True,  # Patrones de monto sospechosos
            'frequency_anomaly': True,  # Frecuencia anómala de transacciones
        }
        
        # Umbrales de riesgo
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
        
        # Cargar modelo si existe
        self.load()
        
    def extract_features(self, transaction: Transaction, user_transactions: List[Transaction] = None) -> Dict[str, float]:
        """
        Extrae características para detección de fraude.
        
        Args:
            transaction: Transacción a analizar
            user_transactions: Historial de transacciones del usuario
            
        Returns:
            dict: Características extraídas
        """
        try:
            features = {}
            
            # Características básicas de la transacción
            features['amount'] = float(transaction.amount)
            features['amount_log'] = np.log1p(float(transaction.amount))
            features['is_expense'] = 1 if transaction.type == 'EXPENSE' else 0
            features['is_income'] = 1 if transaction.type == 'INCOME' else 0
            
            # Características temporales
            features['hour_of_day'] = transaction.date.hour
            features['day_of_week'] = transaction.date.weekday()
            features['is_weekend'] = 1 if transaction.date.weekday() >= 5 else 0
            features['is_night'] = 1 if transaction.date.hour < 6 or transaction.date.hour > 22 else 0
            
            # Características de ubicación (si están disponibles)
            features['location_risk'] = self._calculate_location_risk(transaction)
            
            if user_transactions:
                # Características basadas en el historial del usuario
                features.update(self._extract_user_pattern_features(transaction, user_transactions))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return {}
    
    def _calculate_location_risk(self, transaction: Transaction) -> float:
        """
        Calcula el riesgo basado en la ubicación.
        
        Args:
            transaction: Transacción a analizar
            
        Returns:
            float: Score de riesgo de ubicación
        """
        try:
            # Por ahora, un score básico basado en el monto
            # En un sistema real, esto incluiría análisis de ubicación geográfica
            amount = float(transaction.amount)
            if amount > 5000:
                return 0.8
            elif amount > 1000:
                return 0.5
            else:
                return 0.2
        except:
            return 0.5
    
    def _extract_user_pattern_features(self, transaction: Transaction, user_transactions: List[Transaction]) -> Dict[str, float]:
        """
        Extrae características basadas en patrones del usuario.
        
        Args:
            transaction: Transacción actual
            user_transactions: Historial de transacciones
            
        Returns:
            dict: Características de patrones
        """
        try:
            features = {}
            
            if not user_transactions:
                return features
            
            # Calcular estadísticas del usuario
            amounts = [float(t.amount) for t in user_transactions]
            features['user_avg_amount'] = np.mean(amounts)
            features['user_std_amount'] = np.std(amounts)
            features['user_max_amount'] = np.max(amounts)
            
            # Análisis de frecuencia
            recent_transactions = [
                t for t in user_transactions 
                if t.date >= transaction.date - timedelta(hours=24)
            ]
            features['transactions_last_24h'] = len(recent_transactions)
            
            # Análisis de patrones de monto
            current_amount = float(transaction.amount)
            features['amount_vs_avg'] = current_amount / features['user_avg_amount'] if features['user_avg_amount'] > 0 else 1
            features['amount_vs_max'] = current_amount / features['user_max_amount'] if features['user_max_amount'] > 0 else 1
            
            # Análisis de categorías
            category_counts = {}
            for t in user_transactions[-50:]:  # Últimas 50 transacciones
                cat_name = t.category.name if t.category else 'unknown'
                category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
            
            current_category = transaction.category.name if transaction.category else 'unknown'
            features['category_frequency'] = category_counts.get(current_category, 0) / len(user_transactions[-50:])
            
            # Análisis de tiempo entre transacciones
            if len(user_transactions) > 1:
                time_diffs = []
                for i in range(1, len(user_transactions)):
                    diff = (user_transactions[i].date - user_transactions[i-1].date).total_seconds() / 3600
                    time_diffs.append(diff)
                
                features['avg_time_between_transactions'] = np.mean(time_diffs)
                features['time_since_last_transaction'] = (
                    transaction.date - user_transactions[-1].date
                ).total_seconds() / 3600
            else:
                features['avg_time_between_transactions'] = 24
                features['time_since_last_transaction'] = 24
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting user pattern features: {str(e)}")
            return {}
    
    def detect_fraud_rules(self, transaction: Transaction, user_transactions: List[Transaction] = None) -> Dict[str, Any]:
        """
        Aplica reglas de detección de fraude.
        
        Args:
            transaction: Transacción a analizar
            user_transactions: Historial de transacciones
            
        Returns:
            dict: Resultados de las reglas de fraude
        """
        try:
            results = {
                'fraud_score': 0.0,
                'triggered_rules': [],
                'risk_level': 'low',
                'alerts': []
            }
            
            amount = float(transaction.amount)
            
            # Regla 1: Transacciones de alto valor
            if amount > self.fraud_rules['high_amount_threshold']:
                results['triggered_rules'].append('high_amount')
                results['fraud_score'] += 0.3
                results['alerts'].append(f'Transacción de alto valor: ${amount:,.2f}')
            
            # Regla 2: Velocidad de transacciones
            if user_transactions:
                recent_transactions = [
                    t for t in user_transactions 
                    if t.date >= transaction.date - timedelta(hours=1)
                ]
                if len(recent_transactions) > self.fraud_rules['velocity_threshold']:
                    results['triggered_rules'].append('high_velocity')
                    results['fraud_score'] += 0.4
                    results['alerts'].append(f'Alta velocidad de transacciones: {len(recent_transactions)} en 1 hora')
            
            # Regla 3: Horarios inusuales
            if self.fraud_rules['time_anomaly']:
                hour = transaction.date.hour
                if hour < 6 or hour > 22:
                    results['triggered_rules'].append('unusual_time')
                    results['fraud_score'] += 0.2
                    results['alerts'].append(f'Transacción en horario inusual: {hour}:00')
            
            # Regla 4: Patrones de monto sospechosos
            if self.fraud_rules['amount_pattern'] and user_transactions:
                amounts = [float(t.amount) for t in user_transactions[-20:]]
                if amounts:
                    avg_amount = np.mean(amounts)
                    if amount > avg_amount * 5:  # 5x el promedio
                        results['triggered_rules'].append('amount_pattern')
                        results['fraud_score'] += 0.3
                        results['alerts'].append(f'Monto inusual: ${amount:,.2f} vs promedio ${avg_amount:,.2f}')
            
            # Regla 5: Frecuencia anómala
            if self.fraud_rules['frequency_anomaly'] and user_transactions:
                daily_transactions = [
                    t for t in user_transactions 
                    if t.date.date() == transaction.date.date()
                ]
                if len(daily_transactions) > 10:  # Más de 10 transacciones por día
                    results['triggered_rules'].append('high_frequency')
                    results['fraud_score'] += 0.2
                    results['alerts'].append(f'Alta frecuencia diaria: {len(daily_transactions)} transacciones')
            
            # Determinar nivel de riesgo
            if results['fraud_score'] >= self.risk_thresholds['high']:
                results['risk_level'] = 'high'
            elif results['fraud_score'] >= self.risk_thresholds['medium']:
                results['risk_level'] = 'medium'
            elif results['fraud_score'] >= self.risk_thresholds['low']:
                results['risk_level'] = 'low'
            else:
                results['risk_level'] = 'very_low'
            
            return results
            
        except Exception as e:
            logger.error(f"Error in fraud rules detection: {str(e)}")
            return {
                'fraud_score': 0.0,
                'triggered_rules': [],
                'risk_level': 'unknown',
                'alerts': [f'Error en detección: {str(e)}']
            }
    
    def detect_fraud_ml(self, transaction: Transaction, user_transactions: List[Transaction] = None) -> Dict[str, Any]:
        """
        Detecta fraude usando Machine Learning.
        
        Args:
            transaction: Transacción a analizar
            user_transactions: Historial de transacciones
            
        Returns:
            dict: Resultados de ML
        """
        try:
            if not self.is_trained:
                return {
                    'fraud_probability': 0.5,
                    'anomaly_score': 0.5,
                    'confidence': 0.0,
                    'model_status': 'not_trained'
                }
            
            # Extraer características
            features = self.extract_features(transaction, user_transactions)
            if not features:
                return {
                    'fraud_probability': 0.5,
                    'anomaly_score': 0.5,
                    'confidence': 0.0,
                    'model_status': 'no_features'
                }
            
            # Convertir a array
            feature_values = list(features.values())
            feature_array = np.array(feature_values).reshape(1, -1)
            
            # Escalar características
            feature_array_scaled = self.scaler.transform(feature_array)
            
            # Predicciones
            fraud_prob = self.fraud_classifier.predict_proba(feature_array_scaled)[0][1]
            anomaly_score = self.isolation_forest.decision_function(feature_array_scaled)[0]
            
            # Normalizar anomaly score
            anomaly_score = 1 / (1 + np.exp(-anomaly_score))
            
            # Calcular confianza
            confidence = abs(fraud_prob - 0.5) * 2  # 0 = no confianza, 1 = alta confianza
            
            return {
                'fraud_probability': float(fraud_prob),
                'anomaly_score': float(anomaly_score),
                'confidence': float(confidence),
                'model_status': 'trained',
                'features_used': len(features)
            }
            
        except Exception as e:
            logger.error(f"Error in ML fraud detection: {str(e)}")
            return {
                'fraud_probability': 0.5,
                'anomaly_score': 0.5,
                'confidence': 0.0,
                'model_status': 'error',
                'error': str(e)
            }
    
    def detect_fraud(self, transaction: Transaction, user_id: int = None) -> Dict[str, Any]:
        """
        Detecta fraude usando múltiples métodos.
        
        Args:
            transaction: Transacción a analizar
            user_id: ID del usuario (opcional)
            
        Returns:
            dict: Resultados completos de detección de fraude
        """
        try:
            # Obtener historial del usuario
            user_transactions = []
            if user_id:
                user_transactions = list(Transaction.objects.filter(
                    user_id=user_id,
                    date__lt=transaction.date
                ).order_by('-date')[:100])
            
            # Detección por reglas
            rules_result = self.detect_fraud_rules(transaction, user_transactions)
            
            # Detección por ML
            ml_result = self.detect_fraud_ml(transaction, user_transactions)
            
            # Combinar resultados
            combined_score = (rules_result['fraud_score'] * 0.4 + 
                            ml_result['fraud_probability'] * 0.6)
            
            # Determinar nivel de riesgo final
            if combined_score >= self.risk_thresholds['high']:
                final_risk = 'high'
            elif combined_score >= self.risk_thresholds['medium']:
                final_risk = 'medium'
            elif combined_score >= self.risk_thresholds['low']:
                final_risk = 'low'
            else:
                final_risk = 'very_low'
            
            return {
                'transaction_id': transaction.id,
                'user_id': user_id,
                'amount': float(transaction.amount),
                'date': transaction.date.isoformat(),
                'combined_fraud_score': float(combined_score),
                'risk_level': final_risk,
                'rules_analysis': rules_result,
                'ml_analysis': ml_result,
                'recommendation': self._get_recommendation(final_risk, combined_score),
                'timestamp': timezone.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in fraud detection: {str(e)}")
            return {
                'error': str(e),
                'risk_level': 'unknown',
                'recommendation': 'review_manually'
            }
    
    def _get_recommendation(self, risk_level: str, score: float) -> str:
        """
        Genera recomendaciones basadas en el nivel de riesgo.
        
        Args:
            risk_level: Nivel de riesgo
            score: Score de fraude
            
        Returns:
            str: Recomendación
        """
        if risk_level == 'high':
            return 'block_transaction'
        elif risk_level == 'medium':
            return 'require_verification'
        elif risk_level == 'low':
            return 'monitor_closely'
        else:
            return 'allow_transaction'
    
    def train(self, transactions: List[Transaction], fraud_labels: List[int] = None):
        """
        Entrena el modelo de detección de fraude.
        
        Args:
            transactions: Lista de transacciones
            fraud_labels: Etiquetas de fraude (1 = fraude, 0 = legítimo)
        """
        try:
            if not transactions:
                logger.warning("No transactions provided for training")
                return
            
            # Extraer características
            features_list = []
            for transaction in transactions:
                features = self.extract_features(transaction)
                if features:
                    features_list.append(features)
            
            if not features_list:
                logger.warning("No valid features extracted")
                return
            
            # Convertir a DataFrame
            df = pd.DataFrame(features_list)
            self.feature_names = df.columns.tolist()
            
            # Escalar características
            X_scaled = self.scaler.fit_transform(df)
            
            # Entrenar Isolation Forest para detección de anomalías
            self.isolation_forest.fit(X_scaled)
            
            # Si hay etiquetas de fraude, entrenar clasificador
            if fraud_labels and len(fraud_labels) == len(transactions):
                # Filtrar transacciones con características válidas
                valid_indices = [i for i, t in enumerate(transactions) if self.extract_features(t)]
                y = [fraud_labels[i] for i in valid_indices]
                
                if len(y) == len(X_scaled):
                    # Dividir datos
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    # Entrenar clasificador
                    self.fraud_classifier.fit(X_train, y_train)
                    
                    # Evaluar
                    y_pred = self.fraud_classifier.predict(X_test)
                    accuracy = self.fraud_classifier.score(X_test, y_test)
                    
                    logger.info(f"Fraud classifier trained with accuracy: {accuracy:.3f}")
                    
                    # Guardar reporte de clasificación
                    report = classification_report(y_test, y_pred, output_dict=True)
                    logger.info(f"Classification report: {report}")
            
            self.is_trained = True
            self.save()
            
            logger.info("Fraud detection model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training fraud detection model: {str(e)}")
            raise
    
    def save(self):
        """Guarda el modelo entrenado."""
        try:
            model_data = {
                'scaler': self.scaler,
                'isolation_forest': self.isolation_forest,
                'fraud_classifier': self.fraud_classifier,
                'is_trained': self.is_trained,
                'feature_names': self.feature_names,
                'fraud_rules': self.fraud_rules,
                'risk_thresholds': self.risk_thresholds
            }
            
            joblib.dump(model_data, self.model_path)
            logger.info(f"Fraud detection model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving fraud detection model: {str(e)}")
            raise
    
    def load(self):
        """Carga el modelo entrenado."""
        try:
            model_data = joblib.load(self.model_path)
            
            self.scaler = model_data['scaler']
            self.isolation_forest = model_data['isolation_forest']
            self.fraud_classifier = model_data['fraud_classifier']
            self.is_trained = model_data['is_trained']
            self.feature_names = model_data['feature_names']
            self.fraud_rules = model_data.get('fraud_rules', self.fraud_rules)
            self.risk_thresholds = model_data.get('risk_thresholds', self.risk_thresholds)
            
            logger.info("Fraud detection model loaded successfully")
            
        except FileNotFoundError:
            logger.info("No pre-trained fraud detection model found")
        except Exception as e:
            logger.error(f"Error loading fraud detection model: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtiene información del modelo.
        
        Returns:
            dict: Información del modelo
        """
        return {
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'fraud_rules': self.fraud_rules,
            'risk_thresholds': self.risk_thresholds,
            'model_path': self.model_path
        }
    
    def update_fraud_rules(self, new_rules: Dict[str, Any]):
        """
        Actualiza las reglas de detección de fraude.
        
        Args:
            new_rules: Nuevas reglas
        """
        try:
            self.fraud_rules.update(new_rules)
            logger.info("Fraud rules updated")
        except Exception as e:
            logger.error(f"Error updating fraud rules: {str(e)}")
    
    def get_fraud_statistics(self, days: int = 30) -> Dict[str, Any]:
        """
        Obtiene estadísticas de detección de fraude.
        
        Args:
            days: Número de días para analizar
            
        Returns:
            dict: Estadísticas de fraude
        """
        try:
            start_date = timezone.now() - timedelta(days=days)
            
            transactions = Transaction.objects.filter(
                date__gte=start_date
            ).order_by('-date')
            
            total_transactions = transactions.count()
            high_risk_count = 0
            medium_risk_count = 0
            low_risk_count = 0
            
            for transaction in transactions[:1000]:  # Limitar para rendimiento
                result = self.detect_fraud(transaction, transaction.user_id)
                risk_level = result.get('risk_level', 'unknown')
                
                if risk_level == 'high':
                    high_risk_count += 1
                elif risk_level == 'medium':
                    medium_risk_count += 1
                elif risk_level == 'low':
                    low_risk_count += 1
            
            return {
                'period_days': days,
                'total_transactions_analyzed': total_transactions,
                'high_risk_transactions': high_risk_count,
                'medium_risk_transactions': medium_risk_count,
                'low_risk_transactions': low_risk_count,
                'fraud_rate_estimate': (high_risk_count + medium_risk_count * 0.5) / total_transactions if total_transactions > 0 else 0,
                'generated_at': timezone.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting fraud statistics: {str(e)}")
            return {'error': str(e)} 