"""
Serializers para el servicio de IA.

Este módulo proporciona los serializers necesarios para validar y transformar
los datos de entrada y salida de los endpoints de IA.
"""
from rest_framework import serializers
from .models import AIInteraction, AIInsight, AIPrediction
from datetime import datetime
from ai.ml import BaseMLModel
from transactions.models import Transaction, Category
from django.utils import timezone
from datetime import timedelta
from organizations.models import Organization

class AIInteractionSerializer(serializers.ModelSerializer):
    """Serializer para interacciones de IA"""
    type_display = serializers.CharField(source='get_type_display', read_only=True)

    class Meta:
        model = AIInteraction
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at')

class AIInsightSerializer(serializers.ModelSerializer):
    """Serializer para insights de IA"""
    type_display = serializers.CharField(source='get_type_display', read_only=True)

    class Meta:
        model = AIInsight
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at')

class AIPredictionSerializer(serializers.ModelSerializer):
    """Serializer para predicciones de IA"""
    type_display = serializers.CharField(source='get_type_display', read_only=True)

    class Meta:
        model = AIPrediction
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at')

class AIQuerySerializer(serializers.Serializer):
    query = serializers.CharField()
    context = serializers.JSONField(required=False)
    type = serializers.ChoiceField(choices=AIInteraction.INTERACTION_TYPES)

class AIFeedbackSerializer(serializers.Serializer):
    feedback = serializers.BooleanField()
    feedback_comment = serializers.CharField(required=False, allow_blank=True)

class TransactionAnalysisSerializer(serializers.Serializer):
    """Serializer para análisis de transacciones"""
    category_suggestion = serializers.CharField(required=False)
    confidence = serializers.FloatField(required=False)
    risk_score = serializers.FloatField(required=False)
    anomaly_detected = serializers.BooleanField(required=False)
    insights = serializers.ListField(required=False)

class ExpensePredictionSerializer(serializers.Serializer):
    """Serializer para predicciones de gastos"""
    predicted_amount = serializers.FloatField()
    confidence_interval = serializers.ListField()
    factors = serializers.ListField(required=False)
    trend = serializers.CharField(required=False)

class BehaviorAnalysisSerializer(serializers.Serializer):
    """Serializer para análisis de comportamiento"""
    spending_patterns = serializers.DictField()
    risk_factors = serializers.ListField()
    recommendations = serializers.ListField()
    score = serializers.FloatField()

class AnomalyDetectionSerializer(serializers.Serializer):
    """Serializer para detección de anomalías"""
    anomalies = serializers.ListField()
    total_anomalies = serializers.IntegerField()
    risk_level = serializers.CharField()

class BudgetOptimizationSerializer(serializers.Serializer):
    """Serializer para optimización de presupuesto"""
    optimized_budget = serializers.DictField()
    savings_potential = serializers.FloatField()
    recommendations = serializers.ListField()

class CashFlowPredictionSerializer(serializers.Serializer):
    """Serializer para predicciones de flujo de efectivo"""
    monthly_predictions = serializers.ListField()
    total_predicted = serializers.FloatField()
    confidence = serializers.FloatField()

class RiskAnalysisSerializer(serializers.Serializer):
    """Serializer para análisis de riesgo"""
    overall_risk_score = serializers.FloatField()
    risk_factors = serializers.ListField()
    mitigation_strategies = serializers.ListField()

class RecommendationSerializer(serializers.Serializer):
    """Serializer para recomendaciones"""
    recommendations = serializers.ListField()
    priority = serializers.CharField()
    impact_score = serializers.FloatField()

class ModelStatusSerializer(serializers.Serializer):
    """Serializer para estado de modelos"""
    model_name = serializers.CharField()
    status = serializers.CharField()
    accuracy = serializers.FloatField(required=False)
    last_trained = serializers.DateTimeField(required=False)

class PerformanceMetricsSerializer(serializers.Serializer):
    """Serializer para métricas de rendimiento"""
    response_time = serializers.FloatField()
    accuracy = serializers.FloatField()
    throughput = serializers.IntegerField()
    error_rate = serializers.FloatField()

class AIConfigSerializer(serializers.Serializer):
    """Serializer para configuración de IA"""
    model_parameters = serializers.DictField()
    training_settings = serializers.DictField()
    inference_settings = serializers.DictField()

class ExperimentSerializer(serializers.Serializer):
    """Serializer para experimentos"""
    experiment_id = serializers.CharField()
    status = serializers.CharField()
    results = serializers.DictField(required=False)

class FederatedLearningSerializer(serializers.Serializer):
    """Serializer para aprendizaje federado"""
    participants = serializers.IntegerField()
    rounds = serializers.IntegerField()
    status = serializers.CharField()

class NLPAnalysisSerializer(serializers.Serializer):
    """Serializer para análisis NLP"""
    entities = serializers.ListField()
    sentiment = serializers.CharField()
    topics = serializers.ListField()

class AutoMLResultSerializer(serializers.Serializer):
    """Serializer para resultados de AutoML"""
    best_model = serializers.CharField()
    performance_metrics = serializers.DictField()
    optimization_time = serializers.FloatField()

class ABTestingResultSerializer(serializers.Serializer):
    """Serializer para resultados de A/B Testing"""
    test_id = serializers.CharField()
    variant_a_performance = serializers.DictField()
    variant_b_performance = serializers.DictField()
    winner = serializers.CharField(required=False)

# Serializers para requests
class TransactionAnalysisRequestSerializer(serializers.Serializer):
    """Serializer para requests de análisis de transacciones"""
    description = serializers.CharField()
    amount = serializers.DecimalField(max_digits=10, decimal_places=2)
    type = serializers.ChoiceField(choices=[('expense', 'Expense'), ('income', 'Income')])

class ExpensePredictionRequestSerializer(serializers.Serializer):
    """Serializer para requests de predicción de gastos"""
    days_ahead = serializers.IntegerField(min_value=1, max_value=365, default=30)

class BehaviorAnalysisRequestSerializer(serializers.Serializer):
    """Serializer para requests de análisis de comportamiento"""
    user_id = serializers.IntegerField(required=False)
    time_period = serializers.CharField(required=False)

class BudgetOptimizationRequestSerializer(serializers.Serializer):
    """Serializer para requests de optimización de presupuesto"""
    current_budget = serializers.DictField()
    goals = serializers.ListField(required=False)

class CashFlowPredictionRequestSerializer(serializers.Serializer):
    """Serializer para requests de predicción de flujo de efectivo"""
    months_ahead = serializers.IntegerField(min_value=1, max_value=12, default=3)

class ModelTrainingRequestSerializer(serializers.Serializer):
    """Serializer para requests de entrenamiento de modelos"""
    model_type = serializers.CharField(required=False)
    force_retrain = serializers.BooleanField(default=False)

class AIConfigRequestSerializer(serializers.Serializer):
    """Serializer para requests de configuración de IA"""
    model_parameters = serializers.DictField(required=False)
    training_settings = serializers.DictField(required=False)
    inference_settings = serializers.DictField(required=False)

class ExperimentRequestSerializer(serializers.Serializer):
    """Serializer para requests de experimentos"""
    experiment_type = serializers.CharField()
    parameters = serializers.DictField()

class FederatedLearningRequestSerializer(serializers.Serializer):
    """Serializer para requests de aprendizaje federado"""
    participants = serializers.ListField()
    rounds = serializers.IntegerField(min_value=1, max_value=100)

class NLPRequestSerializer(serializers.Serializer):
    """Serializer para requests de NLP"""
    text = serializers.CharField()
    analysis_type = serializers.ChoiceField(choices=['sentiment', 'entities', 'topics', 'all'])

class AutoMLRequestSerializer(serializers.Serializer):
    """Serializer para requests de AutoML"""
    dataset_id = serializers.CharField()
    target_column = serializers.CharField()
    optimization_metric = serializers.CharField(required=False)

class ABTestingRequestSerializer(serializers.Serializer):
    """Serializer para requests de A/B Testing"""
    variant_a = serializers.DictField()
    variant_b = serializers.DictField()
    test_duration_days = serializers.IntegerField(min_value=1, max_value=30) 