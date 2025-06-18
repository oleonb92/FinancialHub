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

class AIInteractionSerializer(serializers.ModelSerializer):
    type_display = serializers.CharField(source='get_type_display', read_only=True)

    class Meta:
        model = AIInteraction
        fields = [
            'id', 'type', 'type_display', 'query', 'response',
            'context', 'created_at', 'confidence_score',
            'feedback', 'feedback_comment'
        ]
        read_only_fields = ['created_at', 'confidence_score']

class AIInsightSerializer(serializers.ModelSerializer):
    type_display = serializers.CharField(source='get_type_display', read_only=True)

    class Meta:
        model = AIInsight
        fields = [
            'id', 'type', 'type_display', 'title', 'description',
            'data', 'created_at', 'is_read', 'action_taken',
            'action_description', 'impact_score'
        ]
        read_only_fields = ['created_at', 'impact_score']

class AIPredictionSerializer(serializers.ModelSerializer):
    type_display = serializers.CharField(source='get_type_display', read_only=True)

    class Meta:
        model = AIPrediction
        fields = [
            'id', 'type', 'type_display', 'prediction',
            'confidence_score', 'created_at', 'prediction_date',
            'actual_result', 'accuracy_score'
        ]
        read_only_fields = ['created_at', 'confidence_score', 'accuracy_score']

class AIQuerySerializer(serializers.Serializer):
    query = serializers.CharField()
    context = serializers.JSONField(required=False)
    type = serializers.ChoiceField(choices=AIInteraction.INTERACTION_TYPES)

class AIFeedbackSerializer(serializers.Serializer):
    feedback = serializers.BooleanField()
    feedback_comment = serializers.CharField(required=False, allow_blank=True)

class TransactionAnalysisSerializer(serializers.Serializer):
    transaction = serializers.PrimaryKeyRelatedField(queryset=Transaction.objects.all())

class ExpensePredictionSerializer(serializers.Serializer):
    category_id = serializers.IntegerField()
    start_date = serializers.DateField()
    end_date = serializers.DateField()
    
    def validate(self, data):
        if data['start_date'] > data['end_date']:
            raise serializers.ValidationError("End date must be after start date")
        if data['end_date'] > timezone.now().date() + timedelta(days=365):
            raise serializers.ValidationError("Cannot predict more than 1 year ahead")
        return data

class BehaviorAnalysisSerializer(serializers.Serializer):
    """
    Serializer para el análisis de comportamiento.
    
    Fields:
        spending_patterns: Patrones de gasto
        category_distribution: Distribución de categorías
        saving_behavior: Comportamiento de ahorro
        risk_score: Puntuación de riesgo
    """
    spending_patterns = serializers.DictField()
    category_distribution = serializers.DictField()
    saving_behavior = serializers.DictField()
    risk_score = serializers.FloatField(min_value=0.0, max_value=1.0)

class CashFlowPredictionSerializer(serializers.Serializer):
    days = serializers.IntegerField(min_value=1, max_value=365, required=False, default=30)
    
class AnomalyDetectionSerializer(serializers.Serializer):
    transaction_id = serializers.IntegerField()
    amount = serializers.FloatField()
    date = serializers.DateTimeField()
    category = serializers.CharField()
    description = serializers.CharField()
    anomaly_score = serializers.FloatField()
    reason = serializers.CharField()
    
    class Meta:
        fields = ['transaction_id', 'amount', 'date', 'category', 'description', 'anomaly_score', 'reason']

class RiskAnalysisSerializer(serializers.Serializer):
    risk_score = serializers.FloatField()
    risk_level = serializers.CharField()
    metrics = serializers.DictField()
    anomalies = serializers.ListField()
    recommendations = serializers.ListField(child=serializers.DictField())
    
    class Meta:
        fields = ['risk_score', 'risk_level', 'metrics', 'anomalies', 'recommendations']

class ModelMetricsSerializer(serializers.Serializer):
    latest = serializers.DictField()
    history = serializers.ListField()
    trends = serializers.DictField()
    
    class Meta:
        fields = ['latest', 'history', 'trends']

class RecommendationSerializer(serializers.Serializer):
    type = serializers.CharField()
    priority = serializers.CharField()
    message = serializers.CharField()
    confidence = serializers.FloatField() 
    
    class Meta:
        fields = ['type', 'priority', 'message', 'confidence'] 