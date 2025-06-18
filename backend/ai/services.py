"""
Servicio unificado de IA que integra todos los modelos de machine learning.

Este servicio proporciona una interfaz única para acceder a todas las capacidades
de IA del sistema, incluyendo clasificación de transacciones, predicción de gastos,
análisis de comportamiento, recomendaciones personalizadas, detección de anomalías
y predicción de flujo de efectivo.
"""
from django.conf import settings
from django.utils import timezone
from datetime import timedelta
from django.db.models import Q, Count
from django.core.cache import cache
from .models import AIInteraction, AIInsight, AIPrediction
from ai.ml.classifiers.transaction import TransactionClassifier
from ai.ml.predictors.expense import ExpensePredictor
from ai.ml.analyzers.behavior import BehaviorAnalyzer
from ai.ml.recommendation_engine import RecommendationEngine
from ai.ml.anomaly_detector import AnomalyDetector
from ai.ml.cash_flow_predictor import CashFlowPredictor
from ai.ml.risk_analyzer import RiskAnalyzer
from ai.ml.utils.metrics import ModelMetrics
from transactions.models import Transaction, Category
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from django.db import transaction

logger = logging.getLogger('ai.services')

class AIService:
    """
    Servicio unificado que integra todos los modelos de IA del sistema.
    
    Este servicio maneja:
    - Clasificación automática de transacciones
    - Predicción de gastos futuros
    - Análisis de patrones de comportamiento
    - Generación de recomendaciones personalizadas
    - Detección de anomalías
    - Predicción de flujo de efectivo
    - Análisis de riesgo personalizado
    """
    
    def __init__(self):
        """Inicializa todos los modelos de IA."""
        self.transaction_classifier = TransactionClassifier()
        self.expense_predictor = ExpensePredictor()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.recommendation_engine = RecommendationEngine()
        self.anomaly_detector = AnomalyDetector()
        self.cash_flow_predictor = CashFlowPredictor()
        self.risk_analyzer = RiskAnalyzer()
        
        # Inicializar métricas
        self.metrics = {
            'transaction_classifier': ModelMetrics('transaction_classifier'),
            'expense_predictor': ModelMetrics('expense_predictor'),
            'behavior_analyzer': ModelMetrics('behavior_analyzer'),
            'risk_analyzer': ModelMetrics('risk_analyzer')
        }
        
        # Load trained models if available
        self._load_models()
        
    def _load_models(self):
        """Carga los modelos entrenados de manera asíncrona."""
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(self.transaction_classifier.load),
                    executor.submit(self.expense_predictor.load),
                    executor.submit(self.behavior_analyzer.load)
                ]
                for future in futures:
                    future.result()
        except Exception as e:
            logger.warning(f"Could not load trained models: {str(e)}")
    
    @lru_cache(maxsize=100)
    def _get_user_transactions(self, user_id, days=90):
        """Obtiene las transacciones del usuario con caché."""
        try:
            cache_key = f'user_transactions_{user_id}_{days}'
            cached_data = cache.get(cache_key)
            
            if cached_data is None:
                transactions = list(Transaction.objects.filter(
                    created_by_id=user_id,
                    date__gte=timezone.now() - timedelta(days=days)
                ).select_related('category'))
                try:
                    cache.set(cache_key, transactions, timeout=3600)  # Cache for 1 hour
                except Exception as e:
                    logger.warning(f"Error caching transactions: {str(e)}")
                return transactions
                
            return cached_data
        except Exception as e:
            logger.error(f"Error getting user transactions: {str(e)}")
            # Si hay error con el caché, obtener datos directamente de la base de datos
            return list(Transaction.objects.filter(
                created_by_id=user_id,
                date__gte=timezone.now() - timedelta(days=days)
            ).select_related('category'))
    
    def process_query(self, user, query, context=None, interaction_type='general'):
        """
        Process a user query and generate an AI response.
        
        Args:
            user: User object
            query: User query string
            context: Additional context data
            interaction_type: Type of interaction
            
        Returns:
            dict: Response data
        """
        try:
            # Create interaction record
            interaction = AIInteraction.objects.create(
                user=user,
                type=interaction_type,
                query=query,
                context=context or {}
            )
            
            # Process based on interaction type
            if interaction_type == 'transaction':
                response = self._process_transaction_query(query, context)
            elif interaction_type == 'budget':
                response = self._process_budget_query(query, context)
            elif interaction_type == 'prediction':
                response = self._process_prediction_query(query, context)
            else:
                response = self._process_general_query(query, context)
            
            # Update interaction with response
            interaction.response = response
            interaction.confidence_score = self._calculate_confidence_score(response)
            interaction.save()
            
            # Generate insights if applicable
            if interaction_type in ['transaction', 'budget', 'goal']:
                self._generate_insights(user, interaction)
            
            return {
                'response': response,
                'confidence_score': interaction.confidence_score,
                'interaction_id': interaction.id
            }
            
        except Exception as e:
            logger.error(f"Error processing AI query: {str(e)}")
            return {
                'error': 'Unable to process your request at this time',
                'details': str(e)
            }
    
    def analyze_user_risk(self, user, transactions=None):
        """
        Analiza el riesgo financiero del usuario.
        
        Args:
            user: Usuario a analizar
            transactions: Lista de transacciones (opcional)
            
        Returns:
            dict: Análisis de riesgo con datos serializables
        """
        try:
            # Obtener transacciones si no se proporcionan
            if transactions is None:
                transactions = self._get_user_transactions(user.id)
            
            if not transactions:
                return {
                    'risk_score': 0.0,  # Score mínimo cuando no hay datos
                    'risk_level': 'low',
                    'metrics': {
                        'expense_trend': 0,
                        'volatility': 0,
                        'anomaly_count': 0,
                        'savings_rate': 0,
                        'debt_ratio': 0
                    },
                    'anomalies': [],
                    'recommendations': [{
                        'type': 'data_insufficiency',
                        'priority': 'medium',
                        'message': 'No hay suficientes datos para un análisis detallado. Se recomienda comenzar a registrar transacciones.',
                        'confidence': 1.0
                    }]
                }
            
            # Realizar análisis de riesgo
            risk_analysis = self.risk_analyzer.analyze_user_risk(user, transactions)
            
            # Asegurar que los datos sean serializables
            serializable_analysis = {
                'risk_score': float(risk_analysis['risk_score']),
                'risk_level': str(risk_analysis['risk_level']),
                'metrics': {
                    k: float(v) if isinstance(v, (int, float)) else v 
                    for k, v in risk_analysis['metrics'].items()
                },
                'anomalies': [
                    {
                        'transaction_id': anomaly['transaction'].id,
                        'amount': float(anomaly['amount']),
                        'date': anomaly['date'].isoformat(),
                        'category': str(anomaly['category']),
                        'description': str(anomaly['description']),
                        'anomaly_score': float(anomaly['anomaly_score']),
                        'reason': str(anomaly['reason'])
                    }
                    for anomaly in risk_analysis['anomalies']
                ],
                'recommendations': [
                    {
                        'type': rec['type'],
                        'priority': rec['priority'],
                        'message': rec['message'],
                        'confidence': float(rec.get('confidence', 0.8))
                    }
                    for rec in risk_analysis['recommendations']
                ]
            }
            
            # Evaluar métricas del modelo solo si hay suficientes datos
            if len(transactions) > 1:
                try:
                    self.metrics['risk_analyzer'].evaluate_regression(
                        y_true=[float(t.amount) for t in transactions],
                        y_pred=[float(serializable_analysis['metrics']['expense_trend']) for _ in transactions]
                    )
                except Exception as e:
                    logger.warning(f"Error evaluating risk metrics: {str(e)}")
            
            return serializable_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing user risk: {str(e)}", exc_info=True)
            raise
    
    def analyze_transaction(self, transaction):
        """
        Analiza una transacción y genera recomendaciones.
        
        Args:
            transaction: Transacción a analizar
            
        Returns:
            dict: Resultados del análisis
        """
        try:
            # Asegurarse de que el modelo esté entrenado
            if not self.transaction_classifier.is_fitted:
                self.train_models()
            
            # Clasificar transacción
            category_id, confidence = self.transaction_classifier.predict(transaction)
            
            # Detectar anomalías
            anomalies = self.anomaly_detector.detect_anomalies([transaction])
            anomaly_score = anomalies[0]['score'] if anomalies else 0
            
            # Actualizar transacción
            transaction.ai_analyzed = True
            transaction.ai_confidence = confidence
            transaction.ai_category_suggestion_id = category_id
            transaction.save()
            
            # Evaluar métricas del modelo
            try:
                self.metrics['transaction_classifier'].evaluate_classification(
                    y_true=[transaction.category.id],
                    y_pred=[category_id],
                    y_prob=[confidence]
                )
            except Exception as metrics_exc:
                logger.warning(f"Error evaluating transaction classification metrics: {metrics_exc}")
            
            # Generar recomendaciones personalizadas
            recommendations = self.get_personalized_recommendations(transaction.created_by)
            
            return {
                'category_suggestion': category_id,
                'confidence_score': confidence,
                'anomalies': anomalies,
                'anomaly_score': anomaly_score,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error analyzing transaction: {str(e)}")
            raise
    
    def predict_expenses(self, user, category_id, start_date, end_date):
        """
        Predice gastos futuros para una categoría en un rango de fechas.
        
        Args:
            user: Usuario para el que se predicen los gastos
            category_id: ID de la categoría
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            list: Lista de predicciones diarias
        """
        try:
            # Obtener transacciones históricas
            transactions = Transaction.objects.filter(
                organization=user.organization,
                category_id=category_id,
                type='EXPENSE',
                date__gte=timezone.now().date() - timedelta(days=90)
            )
            
            # Entrenar modelo si es necesario
            if not self.expense_predictor.is_trained():
                self.expense_predictor.train(transactions)
                
            # Generar predicciones
            predictions = []
            current_date = start_date
            while current_date <= end_date:
                amount = self.expense_predictor.predict(current_date, category_id)
                predictions.append({
                    'date': current_date,
                    'amount': amount
                })
                current_date += timedelta(days=1)
            
            # Evaluar métricas del modelo
            if transactions.exists():
                self.metrics['expense_predictor'].evaluate_regression(
                    y_true=[t.amount for t in transactions],
                    y_pred=[p['amount'] for p in predictions[:len(transactions)]]
                )
                
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting expenses: {str(e)}")
            raise
    
    def predict_cash_flow(self, user, days=30):
        """
        Predice el flujo de efectivo para los próximos días.
        
        Args:
            user: Usuario para el que se predice el flujo de efectivo
            days: Número de días a predecir
            
        Returns:
            list: Lista de predicciones diarias
        """
        try:
            # Obtener transacciones históricas
            transactions = self._get_user_transactions(user.id)
            
            # Asegurarse de que el modelo esté entrenado
            if not self.cash_flow_predictor.is_fitted:
                self.cash_flow_predictor.train(transactions)
                
            # Generar predicciones
            predictions = self.cash_flow_predictor.predict(transactions, days=days)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting cash flow: {str(e)}")
            raise
    
    def analyze_behavior(self, user):
        """
        Analiza el comportamiento financiero del usuario.
        
        Args:
            user: Usuario a analizar
            
        Returns:
            dict: Análisis de comportamiento
        """
        try:
            # Obtener transacciones recientes
            transactions = self._get_user_transactions(user.id)
            
            # Asegurarse de que el modelo esté entrenado
            if not self.behavior_analyzer.is_fitted:
                self.behavior_analyzer.train(transactions)
            
            # Analizar patrones
            patterns = self.behavior_analyzer.analyze_spending_patterns(transactions)
            
            # Detectar anomalías
            anomalies = self.anomaly_detector.detect_anomalies(transactions)
            
            # Actualizar perfil de usuario
            self.recommendation_engine.build_user_profile(user, transactions)
            
            return {
                'patterns': patterns,
                'anomalies': anomalies
            }
            
        except Exception as e:
            logger.error(f"Error analyzing behavior: {str(e)}")
            raise
    
    def get_recommendations(self, user):
        """
        Genera recomendaciones personalizadas para el usuario.
        
        Args:
            user: Usuario para el que se generan las recomendaciones
            
        Returns:
            list: Lista de recomendaciones personalizadas
        """
        try:
            return self.recommendation_engine.generate_recommendations(user.id)
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise
    
    def get_personalized_recommendations(self, user):
        """
        Genera recomendaciones personalizadas para un usuario.
        
        Args:
            user: Usuario para el que generar recomendaciones
            
        Returns:
            list: Lista de recomendaciones personalizadas
        """
        try:
            # Obtener transacciones recientes
            recent_transactions = Transaction.objects.filter(
                created_by=user,
                date__gte=timezone.now() - timedelta(days=90)
            )
            
            # Actualizar perfil del usuario
            self.recommendation_engine.update_profile(user, recent_transactions)
            
            # Generar recomendaciones
            return self.recommendation_engine.generate_recommendations(user.id)
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []
    
    @transaction.atomic
    def train_models(self):
        """Entrena todos los modelos de IA."""
        try:
            # Obtener datos de entrenamiento
            transactions = Transaction.objects.filter(
                Q(ai_analyzed=True) & 
                Q(created_at__gte=timezone.now() - timedelta(days=90))
            )
            
            # Entrenar modelos
            self.transaction_classifier.train(transactions)
            self.expense_predictor.train(transactions)
            
            # Actualizar análisis de comportamiento
            patterns = self.behavior_analyzer.analyze_spending_patterns(transactions)
            
            # Evaluar métricas después del entrenamiento
            self._evaluate_models(transactions)
            
            return {
                'status': 'success',
                'models_trained': ['classifier', 'predictor', 'behavior_analyzer']
            }
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise
            
    def _evaluate_models(self, transactions):
        """Evalúa el rendimiento de los modelos después del entrenamiento."""
        try:
            # Evaluar clasificador de transacciones
            if transactions.exists():
                y_true = [t.category.id for t in transactions]
                y_pred = [self.transaction_classifier.predict(t)[0] for t in transactions]
                y_prob = [self.transaction_classifier.predict(t)[1] for t in transactions]
                
                self.metrics['transaction_classifier'].evaluate_classification(
                    y_true=y_true,
                    y_pred=y_pred,
                    y_prob=y_prob
                )
            
            # Evaluar predictor de gastos
            if transactions.filter(type='EXPENSE').exists():
                expense_transactions = transactions.filter(type='EXPENSE')
                y_true = [t.amount for t in expense_transactions]
                y_pred = [
                    self.expense_predictor.predict(t.date, t.category.id)
                    for t in expense_transactions
                ]
                
                self.metrics['expense_predictor'].evaluate_regression(
                    y_true=y_true,
                    y_pred=y_pred
                )
            
            # Evaluar analizador de comportamiento
            if transactions.exists():
                patterns = self.behavior_analyzer.analyze_spending_patterns(transactions)
                self.metrics['behavior_analyzer'].evaluate_clustering(
                    X=[[t.amount, t.date.weekday()] for t in transactions],
                    labels=[p['cluster_id'] for p in patterns]
                )
                
        except Exception as e:
            logger.error(f"Error evaluating models: {str(e)}")
            raise
            
    def get_model_metrics(self, model_name, days=30):
        """
        Obtiene las métricas de rendimiento de un modelo.
        
        Args:
            model_name: Nombre del modelo
            days: Número de días de historial
            
        Returns:
            dict: Métricas del modelo
        """
        if model_name not in self.metrics:
            raise ValueError(f"Modelo no encontrado: {model_name}")
            
        return {
            'latest': self.metrics[model_name].get_latest_metrics(),
            'history': self.metrics[model_name].get_metrics_history(days),
            'trends': {
                metric: self.metrics[model_name].get_metrics_trend(metric, days)
                for metric in self.metrics[model_name].get_latest_metrics().keys()
            }
        }
        
    def export_model_metrics(self, model_name, format='json'):
        """
        Exporta las métricas de un modelo.
        
        Args:
            model_name: Nombre del modelo
            format: Formato de exportación ('json' o 'csv')
            
        Returns:
            str: Métricas exportadas
        """
        if model_name not in self.metrics:
            raise ValueError(f"Modelo no encontrado: {model_name}")
            
        return self.metrics[model_name].export_metrics(format)
    
    def _process_transaction_query(self, query, context):
        """
        Process a transaction-related query.
        """
        # Implement transaction query processing
        return "Transaction analysis completed"
    
    def _process_budget_query(self, query, context):
        """
        Process a budget-related query.
        """
        # Implement budget query processing
        return "Budget analysis completed"
    
    def _process_prediction_query(self, query, context):
        """
        Process a prediction-related query.
        """
        # Implement prediction query processing
        return "Prediction analysis completed"
    
    def _process_general_query(self, query, context):
        """
        Process a general query.
        """
        # Implement general query processing
        return "General analysis completed"
    
    def _calculate_confidence_score(self, response):
        """
        Calculate confidence score for a response.
        """
        # Implement confidence score calculation
        return 0.8
    
    def _generate_insights(self, user, interaction):
        """
        Generate insights based on an interaction.
        """
        try:
            # Create insight record
            AIInsight.objects.create(
                user=user,
                type=interaction.type,
                title=f"Insight from {interaction.type} analysis",
                description=interaction.response,
                data={
                    'interaction_id': interaction.id,
                    'query': interaction.query,
                    'response': interaction.response
                }
            )
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}") 