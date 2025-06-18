"""
Vistas para el servicio de IA.

Este módulo proporciona endpoints REST para acceder a las funcionalidades
de IA del sistema, incluyendo análisis de transacciones, predicciones
y recomendaciones.
"""
from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.utils import timezone
from datetime import timedelta
from .services import AIService
from transactions.models import Transaction
from .serializers import (
    TransactionAnalysisSerializer,
    ExpensePredictionSerializer,
    BehaviorAnalysisSerializer,
    RecommendationSerializer,
    CashFlowPredictionSerializer,
    AnomalyDetectionSerializer,
    RiskAnalysisSerializer,
    ModelMetricsSerializer
)
import traceback
import logging

logger = logging.getLogger("ai")

class AIViewSet(viewsets.ViewSet):
    """
    ViewSet para acceder a las funcionalidades de IA.
    
    Endpoints:
    - POST /api/ai/analyze-transaction/: Analiza una transacción
    - POST /api/ai/predict-expenses/: Predice gastos futuros
    - GET /api/ai/analyze-behavior/: Analiza comportamiento financiero
    - GET /api/ai/recommendations/: Obtiene recomendaciones personalizadas
    - POST /api/ai/train/: Entrena los modelos con datos recientes
    - GET /api/ai/predict-cash-flow/: Predice el flujo de efectivo para los próximos días
    - GET /api/ai/detect-anomalies/: Detecta anomalías en las transacciones recientes
    - GET /api/ai/analyze-risk/: Analiza el riesgo financiero del usuario
    - GET /api/ai/get-model-metrics/: Obtiene las métricas de rendimiento de los modelos
    - GET /api/ai/export-metrics/: Exporta las métricas de rendimiento de un modelo
    - POST /api/ai/train-models/: Entrena todos los modelos de IA
    """
    permission_classes = [IsAuthenticated]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ai_service = AIService()
    
    @action(detail=False, methods=['post'])
    def analyze_transaction(self, request):
        """
        Analiza una transacción y sugiere una categoría.
        
        Args:
            request.data:
                - transaction_id: ID de la transacción a analizar
                
        Returns:
            dict: Resultados del análisis
        """
        try:
            # Validar que se proporcionó transaction_id
            if 'transaction_id' not in request.data:
                return Response(
                    {'error': 'transaction_id is required'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Validar que transaction_id es un número válido
            try:
                transaction_id = int(request.data['transaction_id'])
            except (ValueError, TypeError):
                return Response(
                    {'error': 'transaction_id must be a valid number'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Obtener la transacción
            try:
                transaction = Transaction.objects.get(
                    id=transaction_id,
                    organization=request.user.organization
                )
            except Transaction.DoesNotExist:
                return Response(
                    {'error': 'Transaction not found'},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Analizar la transacción
            try:
                analysis = self.ai_service.analyze_transaction(transaction)
                serializer = TransactionAnalysisSerializer(analysis)
                return Response(serializer.data)
            except RuntimeError as e:
                if "Model must be trained" in str(e):
                    # Si el modelo no está entrenado, intentar entrenarlo
                    try:
                        self.ai_service.train_models()
                        analysis = self.ai_service.analyze_transaction(transaction)
                        serializer = TransactionAnalysisSerializer(analysis)
                        return Response(serializer.data)
                    except Exception as train_error:
                        logger.error(f"Error training models: {str(train_error)}", exc_info=True)
                        return Response(
                            {
                                'error': 'Models need to be trained before analysis',
                                'details': str(train_error)
                            },
                            status=status.HTTP_503_SERVICE_UNAVAILABLE
                        )
                else:
                    raise
            
        except Exception as e:
            logger.error(f"Error en analyze_transaction: {str(e)}", exc_info=True)
            return Response(
                {
                    'error': 'An error occurred while analyzing the transaction',
                    'details': str(e)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['post'])
    def predict_expenses(self, request):
        """
        Predice gastos futuros para una categoría.
        
        Args:
            request.data:
                - category_id: ID de la categoría
                - start_date: Fecha de inicio (YYYY-MM-DD)
                - end_date: Fecha de fin (YYYY-MM-DD)
                
        Returns:
            list: Predicciones diarias
        """
        try:
            serializer = ExpensePredictionSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(
                    serializer.errors,
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            predictions = self.ai_service.predict_expenses(
                user=request.user,
                category_id=serializer.validated_data['category_id'],
                start_date=serializer.validated_data['start_date'],
                end_date=serializer.validated_data['end_date']
            )
            
            return Response(predictions)
            
        except Exception as e:
            logger.error(f"Error en predict_expenses: {str(e)}", exc_info=True)
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['get'])
    def analyze_behavior(self, request):
        """
        Analiza el comportamiento financiero del usuario.
        
        Returns:
            dict: Análisis de comportamiento
        """
        try:
            analysis = self.ai_service.analyze_behavior(request.user)
            serializer = BehaviorAnalysisSerializer(analysis)
            
            return Response(serializer.data)
            
        except Exception as e:
            logger.error(f"Error en analyze_behavior: {str(e)}", exc_info=True)
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['get'])
    def recommendations(self, request):
        """
        Obtiene recomendaciones personalizadas.
        
        Returns:
            dict: {'status': 'success', 'recommendations': [...]}
        """
        try:
            recommendations = self.ai_service.get_recommendations(request.user)
            serializer = RecommendationSerializer(recommendations, many=True)
            return Response({
                'status': 'success',
                'recommendations': serializer.data
            })
        except Exception as e:
            logger.error(f"Error en recommendations: {str(e)}", exc_info=True)
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['post'])
    def train(self, request):
        """
        Entrena los modelos con datos recientes.
        
        Returns:
            dict: Estado del entrenamiento
        """
        try:
            result = self.ai_service.train_models()
            return Response(result)
            
        except Exception as e:
            logger.error(f"Error en train: {str(e)}", exc_info=True)
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['get'])
    def predict_cash_flow(self, request):
        """
        Predice el flujo de efectivo para los próximos días.
        
        Query Parameters:
            days (int): Número de días a predecir (default: 30, max: 90)
            
        Returns:
            list: Predicciones de flujo de efectivo
        """
        try:
            # Validar días
            days = request.query_params.get('days', 30)
            try:
                days = int(days)
                if days < 1 or days > 90:
                    return Response(
                        {'error': 'El número de días debe estar entre 1 y 90'},
                        status=status.HTTP_400_BAD_REQUEST
                    )
            except ValueError:
                return Response(
                    {'error': 'El parámetro days debe ser un número entero'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Obtener predicciones
            predictions = self.ai_service.predict_cash_flow(request.user, days)
            
            # Limitar a 15 días si es necesario
            if len(predictions) > 15:
                predictions = predictions[:15]
            
            return Response(predictions)
            
        except Exception as e:
            logger.error(f"Error predicting cash flow: {str(e)}", exc_info=True)
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['get'])
    def detect_anomalies(self, request):
        """
        Detecta anomalías en las transacciones recientes del usuario.
        
        Query Parameters:
            days (int): Número de días hacia atrás para buscar transacciones (default: 30, max: 90)
            
        Returns:
            list: Lista de anomalías detectadas con sus detalles
        """
        try:
            # Validar el parámetro days
            try:
                days = int(request.query_params.get('days', 30))
                if days < 1 or days > 90:
                    return Response(
                        {'error': 'days parameter must be between 1 and 90'},
                        status=status.HTTP_400_BAD_REQUEST
                    )
            except ValueError:
                return Response(
                    {'error': 'days parameter must be a valid integer'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Obtener transacciones recientes
            transactions = Transaction.objects.filter(
                created_by=request.user,
                date__gte=timezone.now() - timedelta(days=days)
            ).order_by('-date')
            
            if not transactions.exists():
                return Response(
                    {'message': 'No transactions found in the specified period'},
                    status=status.HTTP_200_OK
                )
            
            # Detectar anomalías
            anomalies = self.ai_service.anomaly_detector.detect_anomalies(transactions)
            
            if not anomalies:
                return Response(
                    {'message': 'No anomalies detected in the specified period'},
                    status=status.HTTP_200_OK
                )
            
            # Serializar las anomalías
            serializer = AnomalyDetectionSerializer(anomalies, many=True)
            return Response(serializer.data)
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}", exc_info=True)
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['get'])
    def analyze_risk(self, request):
        """
        Analiza el riesgo financiero del usuario.
        
        Returns:
            dict: Análisis de riesgo con métricas y recomendaciones
        """
        try:
            # Obtener transacciones recientes
            transactions = Transaction.objects.filter(
                created_by=request.user,
                date__gte=timezone.now() - timedelta(days=90)
            ).order_by('-date')
            
            # Analizar riesgo
            risk_analysis = self.ai_service.analyze_user_risk(request.user, transactions)
            
            # Serializar respuesta
            serializer = RiskAnalysisSerializer(risk_analysis)
            return Response(serializer.data)
            
        except Exception as e:
            logger.error(f"Error analyzing risk: {str(e)}", exc_info=True)
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['get'])
    def get_model_metrics(self, request):
        """
        Obtiene las métricas de un modelo específico.
        
        Query Parameters:
            model_name (str): Nombre del modelo
            days (int): Número de días a considerar (default: 30)
            
        Returns:
            dict: Métricas del modelo
        """
        try:
            model_name = request.query_params.get('model_name')
            if not model_name:
                return Response(
                    {'error': 'Se requiere el parámetro model_name'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            days = request.query_params.get('days', 30)
            try:
                days = int(days)
                if days < 1:
                    return Response(
                        {'error': 'El número de días debe ser positivo'},
                        status=status.HTTP_400_BAD_REQUEST
                    )
            except ValueError:
                return Response(
                    {'error': 'El parámetro days debe ser un número entero'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            metrics = self.ai_service.get_model_metrics(model_name, days)
            return Response(metrics)
            
        except Exception as e:
            logger.error(f"Error getting model metrics: {str(e)}", exc_info=True)
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['get'])
    def export_metrics(self, request):
        """
        Exporta las métricas de rendimiento de un modelo.
        """
        try:
            model_name = request.query_params.get('model')
            format = request.query_params.get('format', 'json')
            
            if not model_name:
                return Response(
                    {'error': 'Model name is required'},
                    status=status.HTTP_400_BAD_REQUEST
                )
                
            metrics = self.ai_service.export_model_metrics(model_name, format)
            
            if format == 'csv':
                response = Response(metrics, content_type='text/csv')
                response['Content-Disposition'] = f'attachment; filename="{model_name}_metrics.csv"'
            else:
                response = Response(metrics, content_type='application/json')
                
            return response
            
        except ValueError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            logger.error(f"Error exporting metrics: {str(e)}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['post'])
    def train_models(self, request):
        """
        Entrena todos los modelos de IA.
        """
        try:
            result = self.ai_service.train_models()
            return Response(result)
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            ) 