from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.utils import timezone
from .models import AIInteraction, AIInsight, AIPrediction
from .serializers import (
    AIInteractionSerializer, AIInsightSerializer, AIPredictionSerializer,
    AIQuerySerializer, AIFeedbackSerializer
)
from .services import AIService

class AIInteractionViewSet(viewsets.ModelViewSet):
    serializer_class = AIInteractionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return AIInteraction.objects.filter(user=self.request.user)

    @action(detail=False, methods=['post'])
    def query(self, request):
        serializer = AIQuerySerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        ai_service = AIService()
        response = ai_service.process_query(
            user=request.user,
            query=serializer.validated_data['query'],
            context=serializer.validated_data.get('context'),
            interaction_type=serializer.validated_data['type']
        )

        return Response(response, status=status.HTTP_200_OK)

    @action(detail=True, methods=['post'])
    def provide_feedback(self, request, pk=None):
        interaction = self.get_object()
        serializer = AIFeedbackSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        interaction.feedback = serializer.validated_data['feedback']
        interaction.feedback_comment = serializer.validated_data.get('feedback_comment', '')
        interaction.save()

        return Response({'status': 'feedback recorded'})

class AIInsightViewSet(viewsets.ModelViewSet):
    serializer_class = AIInsightSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return AIInsight.objects.filter(user=self.request.user)

    @action(detail=True, methods=['post'])
    def mark_as_read(self, request, pk=None):
        insight = self.get_object()
        insight.is_read = True
        insight.save()
        return Response({'status': 'insight marked as read'})

    @action(detail=True, methods=['post'])
    def record_action(self, request, pk=None):
        insight = self.get_object()
        insight.action_taken = True
        insight.action_description = request.data.get('action_description', '')
        insight.save()
        return Response({'status': 'action recorded'})

class AIPredictionViewSet(viewsets.ModelViewSet):
    serializer_class = AIPredictionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return AIPrediction.objects.filter(user=self.request.user)

    @action(detail=True, methods=['post'])
    def record_actual_result(self, request, pk=None):
        prediction = self.get_object()
        prediction.actual_result = request.data.get('actual_result')
        prediction.accuracy_score = request.data.get('accuracy_score')
        prediction.save()
        return Response({'status': 'actual result recorded'}) 