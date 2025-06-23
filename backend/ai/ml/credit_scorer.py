"""
Sistema de Scoring de Crédito para Fintech.

Este módulo implementa un sistema avanzado de scoring de crédito que incluye:
- Evaluación de riesgo crediticio
- Predicción de elegibilidad de préstamos
- Análisis de capacidad de pago
- Scoring de comportamiento
- Machine Learning para clasificación de riesgo
- Análisis de tendencias crediticias
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from django.utils import timezone
from transactions.models import Transaction
from accounts.models import User
import json

logger = logging.getLogger('ai.credit_scorer')

class CreditScorer:
    """
    Sistema principal de scoring de crédito.
    
    Características:
    - Evaluación de riesgo crediticio
    - Predicción de default
    - Análisis de capacidad de pago
    - Scoring de comportamiento
    - Machine Learning adaptativo
    - Alertas de riesgo crediticio
    """
    
    def __init__(self, model_path: str = 'backend/ml_models/credit_scorer.joblib'):
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.risk_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        self.behavior_scorer = GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        )
        self.is_trained = False
        self.feature_names = []
        
        # Configuración de scoring
        self.scoring_config = {
            'score_range': (300, 850),  # Rango de score crediticio
            'risk_thresholds': {
                'excellent': 750,
                'good': 700,
                'fair': 650,
                'poor': 600,
                'very_poor': 300
            },
            'default_threshold': 0.7,  # Probabilidad de default
            'min_income': 1000,  # Ingreso mínimo mensual
            'max_debt_ratio': 0.43,  # Ratio máximo de deuda
        }
        
        # Factores de peso para scoring
        self.scoring_weights = {
            'payment_history': 0.35,
            'credit_utilization': 0.30,
            'credit_history_length': 0.15,
            'credit_mix': 0.10,
            'new_credit': 0.10
        }
        
        # Cargar modelo si existe
        self.load()
        
    def extract_credit_features(self, user_id: int, user_transactions: List[Transaction] = None) -> Dict[str, float]:
        """
        Extrae características para scoring de crédito.
        
        Args:
            user_id: ID del usuario
            user_transactions: Lista de transacciones del usuario
            
        Returns:
            dict: Características extraídas
        """
        try:
            features = {}
            
            if not user_transactions:
                user_transactions = list(Transaction.objects.filter(
                    user_id=user_id
                ).order_by('-date'))
            
            if not user_transactions:
                return self._get_default_features()
            
            # Características básicas de ingresos
            features.update(self._extract_income_features(user_transactions))
            
            # Características de gastos
            features.update(self._extract_expense_features(user_transactions))
            
            # Características de comportamiento de pago
            features.update(self._extract_payment_features(user_transactions))
            
            # Características de estabilidad financiera
            features.update(self._extract_stability_features(user_transactions))
            
            # Características de utilización de crédito
            features.update(self._extract_credit_utilization_features(user_transactions))
            
            # Características de historial crediticio
            features.update(self._extract_credit_history_features(user_transactions))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting credit features: {str(e)}")
            return self._get_default_features()
    
    def _get_default_features(self) -> Dict[str, float]:
        """Retorna características por defecto para usuarios sin historial."""
        return {
            'avg_monthly_income': 0.0,
            'income_stability': 0.0,
            'avg_monthly_expenses': 0.0,
            'expense_volatility': 0.0,
            'payment_consistency': 0.0,
            'late_payments_ratio': 1.0,
            'savings_rate': 0.0,
            'debt_to_income_ratio': 1.0,
            'credit_utilization': 0.0,
            'credit_history_length': 0.0,
            'credit_mix_score': 0.0,
            'new_credit_activity': 0.0
        }
    
    def _extract_income_features(self, transactions: List[Transaction]) -> Dict[str, float]:
        """Extrae características relacionadas con ingresos."""
        try:
            income_transactions = [t for t in transactions if t.type == 'INCOME']
            
            if not income_transactions:
                return {
                    'avg_monthly_income': 0.0,
                    'income_stability': 0.0
                }
            
            # Calcular ingresos mensuales
            monthly_incomes = {}
            for transaction in income_transactions:
                month_key = f"{transaction.date.year}-{transaction.date.month:02d}"
                if month_key not in monthly_incomes:
                    monthly_incomes[month_key] = 0
                monthly_incomes[month_key] += float(transaction.amount)
            
            incomes = list(monthly_incomes.values())
            
            return {
                'avg_monthly_income': np.mean(incomes),
                'income_stability': 1.0 - (np.std(incomes) / np.mean(incomes)) if np.mean(incomes) > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error extracting income features: {str(e)}")
            return {'avg_monthly_income': 0.0, 'income_stability': 0.0}
    
    def _extract_expense_features(self, transactions: List[Transaction]) -> Dict[str, float]:
        """Extrae características relacionadas con gastos."""
        try:
            expense_transactions = [t for t in transactions if t.type == 'EXPENSE']
            
            if not expense_transactions:
                return {
                    'avg_monthly_expenses': 0.0,
                    'expense_volatility': 0.0
                }
            
            # Calcular gastos mensuales
            monthly_expenses = {}
            for transaction in expense_transactions:
                month_key = f"{transaction.date.year}-{transaction.date.month:02d}"
                if month_key not in monthly_expenses:
                    monthly_expenses[month_key] = 0
                monthly_expenses[month_key] += float(transaction.amount)
            
            expenses = list(monthly_expenses.values())
            
            return {
                'avg_monthly_expenses': np.mean(expenses),
                'expense_volatility': np.std(expenses) / np.mean(expenses) if np.mean(expenses) > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error extracting expense features: {str(e)}")
            return {'avg_monthly_expenses': 0.0, 'expense_volatility': 0.0}
    
    def _extract_payment_features(self, transactions: List[Transaction]) -> Dict[str, float]:
        """Extrae características relacionadas con comportamiento de pago."""
        try:
            # Simular análisis de pagos (en un sistema real, esto vendría de datos de crédito)
            total_transactions = len(transactions)
            recent_transactions = [t for t in transactions if t.date >= timezone.now() - timedelta(days=90)]
            
            # Calcular consistencia de pagos (simulado)
            payment_consistency = len(recent_transactions) / max(total_transactions, 1)
            
            # Simular ratio de pagos tardíos
            late_payments_ratio = 0.05  # 5% por defecto, en sistema real se calcularía
            
            return {
                'payment_consistency': payment_consistency,
                'late_payments_ratio': late_payments_ratio
            }
            
        except Exception as e:
            logger.error(f"Error extracting payment features: {str(e)}")
            return {'payment_consistency': 0.0, 'late_payments_ratio': 1.0}
    
    def _extract_stability_features(self, transactions: List[Transaction]) -> Dict[str, float]:
        """Extrae características de estabilidad financiera."""
        try:
            income_transactions = [t for t in transactions if t.type == 'INCOME']
            expense_transactions = [t for t in transactions if t.type == 'EXPENSE']
            
            total_income = sum(float(t.amount) for t in income_transactions)
            total_expenses = sum(float(t.amount) for t in expense_transactions)
            
            # Ratio de ahorro
            savings_rate = (total_income - total_expenses) / total_income if total_income > 0 else 0.0
            
            # Ratio de deuda a ingresos (simulado)
            debt_to_income_ratio = min(total_expenses / total_income if total_income > 0 else 1.0, 1.0)
            
            return {
                'savings_rate': max(0.0, savings_rate),
                'debt_to_income_ratio': debt_to_income_ratio
            }
            
        except Exception as e:
            logger.error(f"Error extracting stability features: {str(e)}")
            return {'savings_rate': 0.0, 'debt_to_income_ratio': 1.0}
    
    def _extract_credit_utilization_features(self, transactions: List[Transaction]) -> Dict[str, float]:
        """Extrae características de utilización de crédito."""
        try:
            # Simular utilización de crédito basada en patrones de gastos
            recent_expenses = [t for t in transactions 
                             if t.type == 'EXPENSE' and t.date >= timezone.now() - timedelta(days=30)]
            
            total_recent_expenses = sum(float(t.amount) for t in recent_expenses)
            
            # Simular límite de crédito (basado en ingresos históricos)
            income_transactions = [t for t in transactions if t.type == 'INCOME']
            avg_monthly_income = np.mean([float(t.amount) for t in income_transactions]) if income_transactions else 1000
            
            # Límite de crédito simulado (3x ingresos mensuales)
            credit_limit = avg_monthly_income * 3
            credit_utilization = min(total_recent_expenses / credit_limit if credit_limit > 0 else 1.0, 1.0)
            
            return {
                'credit_utilization': credit_utilization
            }
            
        except Exception as e:
            logger.error(f"Error extracting credit utilization features: {str(e)}")
            return {'credit_utilization': 0.0}
    
    def _extract_credit_history_features(self, transactions: List[Transaction]) -> Dict[str, float]:
        """Extrae características de historial crediticio."""
        try:
            if not transactions:
                return {
                    'credit_history_length': 0.0,
                    'credit_mix_score': 0.0,
                    'new_credit_activity': 0.0
                }
            
            # Longitud del historial crediticio (en meses)
            first_transaction = min(transactions, key=lambda x: x.date)
            last_transaction = max(transactions, key=lambda x: x.date)
            history_length = (last_transaction.date - first_transaction.date).days / 30
            
            # Score de mezcla de crédito (simulado basado en categorías)
            categories = set(t.category.name if t.category else 'unknown' for t in transactions)
            credit_mix_score = min(len(categories) / 10, 1.0)  # Normalizado a 10 categorías
            
            # Actividad de nuevo crédito (transacciones recientes)
            recent_transactions = [t for t in transactions 
                                 if t.date >= timezone.now() - timedelta(days=30)]
            new_credit_activity = len(recent_transactions) / max(len(transactions), 1)
            
            return {
                'credit_history_length': min(history_length / 120, 1.0),  # Normalizado a 10 años
                'credit_mix_score': credit_mix_score,
                'new_credit_activity': new_credit_activity
            }
            
        except Exception as e:
            logger.error(f"Error extracting credit history features: {str(e)}")
            return {
                'credit_history_length': 0.0,
                'credit_mix_score': 0.0,
                'new_credit_activity': 0.0
            }
    
    def calculate_credit_score(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Calcula el score crediticio usando características extraídas.
        
        Args:
            features: Características del usuario
            
        Returns:
            dict: Score crediticio y análisis
        """
        try:
            # Calcular componentes del score
            payment_score = (1 - features.get('late_payments_ratio', 1.0)) * 100
            utilization_score = (1 - features.get('credit_utilization', 1.0)) * 100
            history_score = features.get('credit_history_length', 0.0) * 100
            mix_score = features.get('credit_mix_score', 0.0) * 100
            new_credit_score = (1 - features.get('new_credit_activity', 1.0)) * 100
            
            # Calcular score ponderado
            weighted_score = (
                payment_score * self.scoring_weights['payment_history'] +
                utilization_score * self.scoring_weights['credit_utilization'] +
                history_score * self.scoring_weights['credit_history_length'] +
                mix_score * self.scoring_weights['credit_mix'] +
                new_credit_score * self.scoring_weights['new_credit']
            )
            
            # Normalizar al rango de score crediticio
            min_score, max_score = self.scoring_config['score_range']
            credit_score = int(min_score + (weighted_score / 100) * (max_score - min_score))
            
            # Determinar categoría de riesgo
            risk_category = self._determine_risk_category(credit_score)
            
            # Calcular probabilidad de default
            default_probability = self._calculate_default_probability(features)
            
            return {
                'credit_score': credit_score,
                'risk_category': risk_category,
                'default_probability': float(default_probability),
                'score_components': {
                    'payment_history': float(payment_score),
                    'credit_utilization': float(utilization_score),
                    'credit_history_length': float(history_score),
                    'credit_mix': float(mix_score),
                    'new_credit': float(new_credit_score)
                },
                'recommendation': self._get_credit_recommendation(credit_score, default_probability),
                'calculated_at': timezone.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating credit score: {str(e)}")
            return {
                'credit_score': 300,
                'risk_category': 'very_poor',
                'default_probability': 1.0,
                'error': str(e)
            }
    
    def _determine_risk_category(self, credit_score: int) -> str:
        """Determina la categoría de riesgo basada en el score."""
        thresholds = self.scoring_config['risk_thresholds']
        
        if credit_score >= thresholds['excellent']:
            return 'excellent'
        elif credit_score >= thresholds['good']:
            return 'good'
        elif credit_score >= thresholds['fair']:
            return 'fair'
        elif credit_score >= thresholds['poor']:
            return 'poor'
        else:
            return 'very_poor'
    
    def _calculate_default_probability(self, features: Dict[str, float]) -> float:
        """
        Calcula la probabilidad de default usando ML si está entrenado.
        
        Args:
            features: Características del usuario
            
        Returns:
            float: Probabilidad de default
        """
        try:
            if not self.is_trained:
                # Cálculo heurístico si el modelo no está entrenado
                risk_factors = [
                    features.get('late_payments_ratio', 1.0),
                    features.get('debt_to_income_ratio', 1.0),
                    features.get('credit_utilization', 1.0),
                    1 - features.get('payment_consistency', 0.0),
                    1 - features.get('income_stability', 0.0)
                ]
                return np.mean(risk_factors)
            
            # Usar modelo entrenado
            feature_values = list(features.values())
            feature_array = np.array(feature_values).reshape(1, -1)
            feature_array_scaled = self.scaler.transform(feature_array)
            
            default_prob = self.risk_classifier.predict_proba(feature_array_scaled)[0][1]
            return float(default_prob)
            
        except Exception as e:
            logger.error(f"Error calculating default probability: {str(e)}")
            return 0.5
    
    def _get_credit_recommendation(self, credit_score: int, default_probability: float) -> Dict[str, Any]:
        """
        Genera recomendaciones basadas en el score crediticio.
        
        Args:
            credit_score: Score crediticio
            default_probability: Probabilidad de default
            
        Returns:
            dict: Recomendaciones
        """
        try:
            if default_probability > self.scoring_config['default_threshold']:
                recommendation = 'reject'
                message = 'Alto riesgo de default. Solicitud rechazada.'
            elif credit_score >= self.scoring_config['risk_thresholds']['excellent']:
                recommendation = 'approve_premium'
                message = 'Score excelente. Aprobado con mejores términos.'
            elif credit_score >= self.scoring_config['risk_thresholds']['good']:
                recommendation = 'approve_standard'
                message = 'Score bueno. Aprobado con términos estándar.'
            elif credit_score >= self.scoring_config['risk_thresholds']['fair']:
                recommendation = 'approve_restricted'
                message = 'Score regular. Aprobado con restricciones.'
            elif credit_score >= self.scoring_config['risk_thresholds']['poor']:
                recommendation = 'approve_high_risk'
                message = 'Score bajo. Aprobado con términos de alto riesgo.'
            else:
                recommendation = 'reject'
                message = 'Score muy bajo. Solicitud rechazada.'
            
            return {
                'action': recommendation,
                'message': message,
                'confidence': 1.0 - default_probability
            }
            
        except Exception as e:
            logger.error(f"Error generating credit recommendation: {str(e)}")
            return {
                'action': 'review_manually',
                'message': 'Revisión manual requerida.',
                'confidence': 0.0
            }
    
    def assess_loan_eligibility(self, user_id: int, loan_amount: float, 
                              loan_term: int, loan_type: str = 'personal') -> Dict[str, Any]:
        """
        Evalúa la elegibilidad para un préstamo.
        
        Args:
            user_id: ID del usuario
            loan_amount: Monto del préstamo
            loan_term: Plazo del préstamo en meses
            loan_type: Tipo de préstamo
            
        Returns:
            dict: Evaluación de elegibilidad
        """
        try:
            # Extraer características
            features = self.extract_credit_features(user_id)
            
            # Calcular score crediticio
            credit_analysis = self.calculate_credit_score(features)
            
            # Calcular capacidad de pago
            monthly_income = features.get('avg_monthly_income', 0)
            monthly_expenses = features.get('avg_monthly_expenses', 0)
            
            # Calcular pago mensual estimado (simplificado)
            annual_rate = 0.15  # 15% anual (ajustar según score)
            if credit_analysis['risk_category'] == 'excellent':
                annual_rate = 0.08
            elif credit_analysis['risk_category'] == 'good':
                annual_rate = 0.12
            elif credit_analysis['risk_category'] == 'fair':
                annual_rate = 0.18
            else:
                annual_rate = 0.25
            
            monthly_rate = annual_rate / 12
            monthly_payment = (loan_amount * monthly_rate * (1 + monthly_rate)**loan_term) / ((1 + monthly_rate)**loan_term - 1)
            
            # Calcular capacidad de pago
            available_income = monthly_income - monthly_expenses
            debt_service_ratio = monthly_payment / available_income if available_income > 0 else 1.0
            
            # Determinar elegibilidad
            is_eligible = (
                credit_analysis['default_probability'] < self.scoring_config['default_threshold'] and
                debt_service_ratio < self.scoring_config['max_debt_ratio'] and
                available_income > 0
            )
            
            return {
                'user_id': user_id,
                'loan_amount': float(loan_amount),
                'loan_term': loan_term,
                'loan_type': loan_type,
                'is_eligible': is_eligible,
                'credit_score': credit_analysis['credit_score'],
                'risk_category': credit_analysis['risk_category'],
                'default_probability': credit_analysis['default_probability'],
                'monthly_payment': float(monthly_payment),
                'debt_service_ratio': float(debt_service_ratio),
                'available_income': float(available_income),
                'recommended_max_amount': float(available_income * self.scoring_config['max_debt_ratio'] * loan_term / 12),
                'annual_rate': float(annual_rate),
                'recommendation': credit_analysis['recommendation'],
                'assessed_at': timezone.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error assessing loan eligibility: {str(e)}")
            return {
                'error': str(e),
                'is_eligible': False
            }
    
    def train(self, training_data: List[Dict[str, Any]], default_labels: List[int] = None):
        """
        Entrena el modelo de scoring de crédito.
        
        Args:
            training_data: Lista de diccionarios con características
            default_labels: Etiquetas de default (1 = default, 0 = no default)
        """
        try:
            if not training_data:
                logger.warning("No training data provided")
                return
            
            # Convertir a DataFrame
            df = pd.DataFrame(training_data)
            self.feature_names = df.columns.tolist()
            
            # Escalar características
            X_scaled = self.scaler.fit_transform(df)
            
            if default_labels and len(default_labels) == len(training_data):
                # Dividir datos
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, default_labels, test_size=0.2, random_state=42, stratify=default_labels
                )
                
                # Entrenar clasificador de riesgo
                self.risk_classifier.fit(X_train, y_train)
                
                # Evaluar modelo
                y_pred = self.risk_classifier.predict(X_test)
                accuracy = self.risk_classifier.score(X_test, y_test)
                auc_score = roc_auc_score(y_test, self.risk_classifier.predict_proba(X_test)[:, 1])
                
                logger.info(f"Credit risk model trained - Accuracy: {accuracy:.3f}, AUC: {auc_score:.3f}")
                
                # Guardar reporte
                report = classification_report(y_test, y_pred, output_dict=True)
                logger.info(f"Classification report: {report}")
            
            self.is_trained = True
            self.save()
            
            logger.info("Credit scoring model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training credit scoring model: {str(e)}")
            raise
    
    def save(self):
        """Guarda el modelo entrenado."""
        try:
            model_data = {
                'scaler': self.scaler,
                'risk_classifier': self.risk_classifier,
                'behavior_scorer': self.behavior_scorer,
                'is_trained': self.is_trained,
                'feature_names': self.feature_names,
                'scoring_config': self.scoring_config,
                'scoring_weights': self.scoring_weights
            }
            
            joblib.dump(model_data, self.model_path)
            logger.info(f"Credit scoring model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving credit scoring model: {str(e)}")
            raise
    
    def load(self):
        """Carga el modelo entrenado."""
        try:
            model_data = joblib.load(self.model_path)
            
            self.scaler = model_data['scaler']
            self.risk_classifier = model_data['risk_classifier']
            self.behavior_scorer = model_data['behavior_scorer']
            self.is_trained = model_data['is_trained']
            self.feature_names = model_data['feature_names']
            self.scoring_config = model_data.get('scoring_config', self.scoring_config)
            self.scoring_weights = model_data.get('scoring_weights', self.scoring_weights)
            
            logger.info("Credit scoring model loaded successfully")
            
        except FileNotFoundError:
            logger.info("No pre-trained credit scoring model found")
        except Exception as e:
            logger.error(f"Error loading credit scoring model: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtiene información del modelo.
        
        Returns:
            dict: Información del modelo
        """
        return {
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'scoring_config': self.scoring_config,
            'scoring_weights': self.scoring_weights,
            'model_path': self.model_path
        } 