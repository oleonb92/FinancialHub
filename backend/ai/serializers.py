from rest_framework import serializers
from .models import AIInteraction, AIInsight, AIPrediction

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