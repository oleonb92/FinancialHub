"""
Servicio de IA Avanzado que integra todos los componentes modernos.

Este servicio reemplaza gradualmente la dependencia de APIs externas con:
- Procesamiento de lenguaje natural propio
- Aprendizaje federado
- AutoML y optimización automática
- Transformers especializados
- A/B testing y experimentación
- Análisis avanzado de patrones
"""
from django.conf import settings
from django.utils import timezone
from datetime import timedelta
from django.db.models import Q, Count
from django.core.cache import cache
from .models import AIInteraction, AIInsight, AIPrediction
from .ml.classifiers.transaction import TransactionClassifier
from .ml.predictors.expense import ExpensePredictor
from .ml.analyzers.behavior import BehaviorAnalyzer
from .ml.recommendation_engine import RecommendationEngine
from .ml.anomaly_detector import AnomalyDetector
from .ml.cash_flow_predictor import CashFlowPredictor
from .ml.risk_analyzer import RiskAnalyzer
from .ml.nlp.text_processor import TextProcessor
from .ml.federated.federated_learning import FederatedLearningManager, FederatedModelMixin
from .ml.automl.auto_ml_optimizer import AutoMLOptimizer, FeatureOptimizer
from .ml.transformers.financial_transformer import FinancialTextProcessor
from .ml.experimentation.ab_testing import ABTestingManager
from .ml.utils.metrics import ModelMetrics
from transactions.models import Transaction, Category
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from django.db import transaction
import numpy as np
import pandas as pd

logger = logging.getLogger('ai.services_advanced')

class AdvancedAIService:
    """
    Servicio de IA avanzado que minimiza dependencias externas.
    
    Características principales:
    - NLP propio para análisis de texto
    - Aprendizaje federado para privacidad
    - AutoML para optimización automática
    - Transformers especializados
    - A/B testing para experimentación
    - Análisis avanzado de patrones
    """
    
    def __init__(self):
        """Inicializa todos los componentes avanzados."""
        # Modelos ML existentes (mejorados con federación)
        self.transaction_classifier = self._create_federated_model(TransactionClassifier())
        self.expense_predictor = self._create_federated_model(ExpensePredictor())
        self.behavior_analyzer = self._create_federated_model(BehaviorAnalyzer())
        self.recommendation_engine = RecommendationEngine()
        self.anomaly_detector = AnomalyDetector()
        self.cash_flow_predictor = CashFlowPredictor()
        self.risk_analyzer = RiskAnalyzer()
        
        # Nuevos componentes avanzados
        self.text_processor = TextProcessor()
        self.federated_manager = FederatedLearningManager()
        self.auto_ml_optimizer = AutoMLOptimizer()
        self.feature_optimizer = FeatureOptimizer()
        self.financial_transformer = FinancialTextProcessor()
        self.ab_testing_manager = ABTestingManager()
        
        # Métricas y monitoreo
        self.metrics = {
            'transaction_classifier': ModelMetrics('transaction_classifier'),
            'expense_predictor': ModelMetrics('expense_predictor'),
            'behavior_analyzer': ModelMetrics('behavior_analyzer'),
            'financial_transformer': ModelMetrics('financial_transformer')
        }
        
        # Cargar modelos entrenados
        self._load_models()
        
        # Configurar experimentos activos
        self._setup_experiments()
    
    def _create_federated_model(self, model):
        """Crea un modelo con capacidades federadas."""
        if not hasattr(model, 'get_model_params'):
            # Agregar mixin de federación
            model.__class__ = type(
                f'Federated{model.__class__.__name__}',
                (FederatedModelMixin, model.__class__),
                {}
            )
        return model
    
    def _load_models(self):
        """Carga modelos entrenados de manera asíncrona."""
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(self.transaction_classifier.load),
                    executor.submit(self.expense_predictor.load),
                    executor.submit(self.behavior_analyzer.load),
                    executor.submit(self._load_financial_transformer)
                ]
                for future in futures:
                    future.result()
        except Exception as e:
            logger.warning(f"Could not load trained models: {str(e)}")
    
    def _load_financial_transformer(self):
        """Carga el transformer financiero."""
        try:
            model_path = f"{settings.ML_MODELS_DIR}/financial_transformer.pth"
            tokenizer_path = f"{settings.ML_MODELS_DIR}/financial_tokenizer.pkl"
            self.financial_transformer.load_model(model_path, tokenizer_path)
        except Exception as e:
            logger.warning(f"Could not load financial transformer: {str(e)}")
    
    def _setup_experiments(self):
        """Configura experimentos activos."""
        try:
            active_experiments = self.ab_testing_manager.get_active_experiments()
            for experiment in active_experiments:
                logger.info(f"Experimento activo: {experiment.name}")
        except Exception as e:
            logger.warning(f"Error setting up experiments: {str(e)}")
    
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
                    cache.set(cache_key, transactions, timeout=3600)
                except Exception as e:
                    logger.warning(f"Error caching transactions: {str(e)}")
                return transactions
                
            return cached_data
        except Exception as e:
            logger.error(f"Error getting user transactions: {str(e)}")
            return list(Transaction.objects.filter(
                created_by_id=user_id,
                date__gte=timezone.now() - timedelta(days=days)
            ).select_related('category'))
    
    def process_natural_language_query(self, user, query: str, context: Dict = None) -> Dict[str, Any]:
        """
        Procesa consultas en lenguaje natural usando NLP propio.
        
        Args:
            user: Usuario que hace la consulta
            query: Consulta en lenguaje natural
            context: Contexto adicional
            
        Returns:
            dict: Respuesta procesada
        """
        try:
            # Procesar texto con NLP propio
            nlp_analysis = self.text_processor.process_text(query, context)
            
            # Determinar experimento activo para el usuario
            experiment_variant = self._get_user_experiment_variant(user)
            
            # Procesar según la intención detectada
            intent = nlp_analysis.get('intent', {}).get('intent', 'general_question')
            
            if intent == 'query_balance':
                response = self._handle_balance_query(user, nlp_analysis)
            elif intent == 'add_transaction':
                response = self._handle_add_transaction(user, nlp_analysis)
            elif intent == 'analyze_spending':
                response = self._handle_spending_analysis(user, nlp_analysis)
            elif intent == 'set_budget':
                response = self._handle_budget_setting(user, nlp_analysis)
            elif intent == 'get_recommendations':
                response = self._handle_recommendations(user, nlp_analysis)
            elif intent == 'predict_expenses':
                response = self._handle_expense_prediction(user, nlp_analysis)
            else:
                response = self._handle_general_query(user, nlp_analysis)
            
            # Usar transformer financiero si está disponible
            if self.financial_transformer.model:
                transformer_response = self.financial_transformer.predict(query)
                response['transformer_analysis'] = transformer_response
            
            # Registrar evento de experimento
            self._track_experiment_event(user, experiment_variant, 'nlp_query', {
                'query': query,
                'intent': intent,
                'confidence': nlp_analysis.get('confidence', 0.0)
            })
            
            return {
                'response': response,
                'nlp_analysis': nlp_analysis,
                'experiment_variant': experiment_variant,
                'confidence': nlp_analysis.get('confidence', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error processing NLP query: {str(e)}")
            return {
                'error': 'No pude procesar tu consulta en este momento',
                'details': str(e)
            }
    
    def _get_user_experiment_variant(self, user) -> str:
        """Obtiene la variante de experimento para el usuario."""
        try:
            # Buscar experimentos activos
            active_experiments = self.ab_testing_manager.get_active_experiments()
            
            for experiment in active_experiments:
                variant = self.ab_testing_manager.assign_user_to_variant(
                    str(experiment.id), user.id
                )
                return variant
            
            return 'control'
        except Exception as e:
            logger.warning(f"Error getting experiment variant: {str(e)}")
            return 'control'
    
    def _track_experiment_event(self, user, variant: str, event_type: str, event_data: Dict):
        """Registra evento de experimento."""
        try:
            active_experiments = self.ab_testing_manager.get_active_experiments()
            for experiment in active_experiments:
                self.ab_testing_manager.track_event(
                    str(experiment.id), user.id, event_type, event_data
                )
        except Exception as e:
            logger.warning(f"Error tracking experiment event: {str(e)}")
    
    def _handle_balance_query(self, user, nlp_analysis: Dict) -> Dict[str, Any]:
        """Maneja consultas de saldo."""
        try:
            transactions = self._get_user_transactions(user.id, days=30)
            
            # Calcular saldo actual
            total_income = sum(t.amount for t in transactions if t.amount > 0)
            total_expenses = sum(abs(t.amount) for t in transactions if t.amount < 0)
            current_balance = total_income - total_expenses
            
            # Extraer entidades del análisis NLP
            entities = nlp_analysis.get('entities', {})
            amounts = entities.get('amounts', [])
            
            return {
                'type': 'balance_query',
                'current_balance': current_balance,
                'total_income': total_income,
                'total_expenses': total_expenses,
                'period': '30 days',
                'detected_amounts': amounts
            }
        except Exception as e:
            logger.error(f"Error handling balance query: {str(e)}")
            return {'error': 'No pude obtener tu saldo en este momento'}
    
    def _handle_add_transaction(self, user, nlp_analysis: Dict) -> Dict[str, Any]:
        """Maneja solicitudes de agregar transacciones."""
        try:
            entities = nlp_analysis.get('entities', {})
            amounts = entities.get('amounts', [])
            categories = entities.get('categories', [])
            
            # Usar clasificador para sugerir categoría
            if amounts and not categories:
                # Simular transacción para clasificación
                mock_transaction = type('MockTransaction', (), {
                    'description': nlp_analysis.get('original_text', ''),
                    'amount': float(amounts[0].replace('$', '').replace(',', '')),
                    'date': timezone.now(),
                    'category': type('MockCategory', (), {'id': 1})()
                })
                
                category_id, confidence = self.transaction_classifier.predict(mock_transaction)
                categories = [f"Category {category_id}"]
            
            return {
                'type': 'add_transaction',
                'suggested_amount': amounts[0] if amounts else None,
                'suggested_category': categories[0] if categories else None,
                'confidence': confidence if 'confidence' in locals() else 0.0
            }
        except Exception as e:
            logger.error(f"Error handling add transaction: {str(e)}")
            return {'error': 'No pude procesar la transacción'}
    
    def _handle_spending_analysis(self, user, nlp_analysis: Dict) -> Dict[str, Any]:
        """Maneja análisis de gastos."""
        try:
            transactions = self._get_user_transactions(user.id, days=90)
            
            # Usar analizador de comportamiento
            behavior_analysis = self.behavior_analyzer.analyze_spending_patterns(transactions)
            
            return {
                'type': 'spending_analysis',
                'analysis': behavior_analysis,
                'total_transactions': len(transactions),
                'period': '90 days'
            }
        except Exception as e:
            logger.error(f"Error handling spending analysis: {str(e)}")
            return {'error': 'No pude analizar tus gastos'}
    
    def _handle_budget_setting(self, user, nlp_analysis: Dict) -> Dict[str, Any]:
        """Maneja configuración de presupuesto."""
        try:
            entities = nlp_analysis.get('entities', {})
            amounts = entities.get('amounts', [])
            
            return {
                'type': 'budget_setting',
                'suggested_budget': amounts[0] if amounts else None,
                'message': 'Te ayudo a configurar tu presupuesto'
            }
        except Exception as e:
            logger.error(f"Error handling budget setting: {str(e)}")
            return {'error': 'No pude configurar el presupuesto'}
    
    def _handle_recommendations(self, user, nlp_analysis: Dict) -> Dict[str, Any]:
        """Maneja solicitudes de recomendaciones."""
        try:
            recommendations = self.recommendation_engine.generate_recommendations(user.id)
            
            return {
                'type': 'recommendations',
                'recommendations': recommendations,
                'count': len(recommendations)
            }
        except Exception as e:
            logger.error(f"Error handling recommendations: {str(e)}")
            return {'error': 'No pude generar recomendaciones'}
    
    def _handle_expense_prediction(self, user, nlp_analysis: Dict) -> Dict[str, Any]:
        """Maneja predicciones de gastos."""
        try:
            # Usar predictor de gastos
            future_date = timezone.now() + timedelta(days=30)
            prediction = self.expense_predictor.predict(future_date, category_id=1)
            
            return {
                'type': 'expense_prediction',
                'predicted_amount': prediction,
                'prediction_date': future_date.isoformat(),
                'category': 'General'
            }
        except Exception as e:
            logger.error(f"Error handling expense prediction: {str(e)}")
            return {'error': 'No pude predecir gastos futuros'}
    
    def _handle_general_query(self, user, nlp_analysis: Dict) -> Dict[str, Any]:
        """Maneja consultas generales."""
        return {
            'type': 'general_query',
            'message': 'Entiendo tu consulta. ¿En qué más puedo ayudarte?',
            'suggestions': [
                'Consulta tu saldo',
                'Analiza tus gastos',
                'Obtén recomendaciones',
                'Predice gastos futuros'
            ]
        }
    
    def optimize_models_automatically(self) -> Dict[str, Any]:
        """
        Optimiza automáticamente todos los modelos usando AutoML.
        
        Returns:
            dict: Resultados de la optimización
        """
        try:
            # Obtener datos de entrenamiento
            transactions = Transaction.objects.filter(
                created_at__gte=timezone.now() - timedelta(days=90)
            ).select_related('category')
            
            if not transactions:
                return {'error': 'No hay suficientes datos para optimización'}
            
            # Preparar datos para AutoML
            df = pd.DataFrame([{
                'amount': float(t.amount),
                'day_of_week': t.date.weekday(),
                'day_of_month': t.date.day,
                'month': t.date.month,
                'category_id': t.category.id,
                'description': t.description
            } for t in transactions])
            
            # Optimizar features
            X_optimized = self.feature_optimizer.optimize_features(
                df.drop('category_id', axis=1), 
                df['category_id'],
                method='mutual_info',
                n_features=10
            )
            
            # Optimizar modelo de clasificación
            classification_optimizer = AutoMLOptimizer(task_type='classification')
            classification_results = classification_optimizer.optimize(
                X_optimized, df['category_id'], cv_folds=5
            )
            
            # Optimizar modelo de regresión (para predicción de gastos)
            regression_optimizer = AutoMLOptimizer(task_type='regression')
            regression_results = regression_optimizer.optimize(
                X_optimized, df['amount'], cv_folds=5
            )
            
            return {
                'status': 'success',
                'classification_optimization': classification_results,
                'regression_optimization': regression_results,
                'feature_optimization': {
                    'selected_features': self.feature_optimizer.get_selected_features(),
                    'feature_importance': self.feature_optimizer.get_feature_importance()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in automatic model optimization: {str(e)}")
            return {'error': str(e)}
    
    def train_federated_models(self, organization_id: str) -> Dict[str, Any]:
        """
        Entrena modelos usando aprendizaje federado.
        
        Args:
            organization_id: ID de la organización
            
        Returns:
            dict: Resultados del entrenamiento federado
        """
        try:
            # Obtener datos de la organización
            transactions = Transaction.objects.filter(
                created_by__organization_id=organization_id,
                created_at__gte=timezone.now() - timedelta(days=90)
            ).select_related('category')
            
            if not transactions:
                return {'error': 'No hay suficientes datos para entrenamiento federado'}
            
            # Registrar modelos para federación
            self.federated_manager.register_local_model(
                'transaction_classifier', self.transaction_classifier, organization_id
            )
            self.federated_manager.register_local_model(
                'expense_predictor', self.expense_predictor, organization_id
            )
            
            # Preparar datos de entrenamiento
            training_data = {
                organization_id: {
                    'transactions': list(transactions)
                }
            }
            
            # Entrenar modelos federados
            results = {}
            for model_name in ['transaction_classifier', 'expense_predictor']:
                result = self.federated_manager.train_federated_model(model_name, training_data)
                results[model_name] = result
            
            return {
                'status': 'success',
                'federated_training_results': results,
                'organization_id': organization_id
            }
            
        except Exception as e:
            logger.error(f"Error in federated training: {str(e)}")
            return {'error': str(e)}
    
    def create_and_run_experiment(self, name: str, description: str,
                                control_model: str, variant_models: Dict[str, str]) -> Dict[str, Any]:
        """
        Crea y ejecuta un experimento A/B.
        
        Args:
            name: Nombre del experimento
            description: Descripción del experimento
            control_model: Modelo de control
            variant_models: Modelos de variantes
            
        Returns:
            dict: Resultados del experimento
        """
        try:
            # Crear experimento
            experiment = self.ab_testing_manager.create_experiment(
                name=name,
                description=description,
                control_model=control_model,
                variant_models=variant_models,
                traffic_split={'control': 0.5, 'variant_a': 0.5},
                primary_metric='accuracy'
            )
            
            # Iniciar experimento
            success = self.ab_testing_manager.start_experiment(str(experiment.id))
            
            if success:
                return {
                    'status': 'success',
                    'experiment_id': str(experiment.id),
                    'experiment_name': experiment.name,
                    'message': 'Experimento creado y iniciado exitosamente'
                }
            else:
                return {'error': 'No se pudo iniciar el experimento'}
                
        except Exception as e:
            logger.error(f"Error creating experiment: {str(e)}")
            return {'error': str(e)}
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """
        Obtiene resultados de un experimento.
        
        Args:
            experiment_id: ID del experimento
            
        Returns:
            dict: Resultados del experimento
        """
        try:
            return self.ab_testing_manager.get_experiment_results(experiment_id)
        except Exception as e:
            logger.error(f"Error getting experiment results: {str(e)}")
            return {'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado completo del sistema de IA.
        
        Returns:
            dict: Estado del sistema
        """
        try:
            # Estado de modelos
            model_status = {}
            for model_name, model in [
                ('transaction_classifier', self.transaction_classifier),
                ('expense_predictor', self.expense_predictor),
                ('behavior_analyzer', self.behavior_analyzer),
                ('financial_transformer', self.financial_transformer)
            ]:
                model_status[model_name] = {
                    'is_trained': getattr(model, 'is_fitted', False) or getattr(model, 'is_trained', False),
                    'model_type': type(model).__name__,
                    'last_updated': getattr(model, 'last_update', None)
                }
            
            # Estado de federación
            federation_status = self.federated_manager.get_federation_status()
            
            # Experimentos activos
            active_experiments = self.ab_testing_manager.get_active_experiments()
            
            return {
                'model_status': model_status,
                'federation_status': federation_status,
                'active_experiments': len(active_experiments),
                'experiment_names': [exp.name for exp in active_experiments],
                'system_health': 'healthy',
                'last_updated': timezone.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {'error': str(e)} 