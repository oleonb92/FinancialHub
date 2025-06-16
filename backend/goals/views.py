from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.utils import timezone
from .models import FinancialGoal, GoalMilestone, GoalContribution
from .serializers import (
    FinancialGoalSerializer, GoalMilestoneSerializer,
    GoalContributionSerializer, GoalProgressUpdateSerializer,
    GoalContributionCreateSerializer
)
from notifications.services import NotificationService

class FinancialGoalViewSet(viewsets.ModelViewSet):
    serializer_class = FinancialGoalSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return FinancialGoal.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        goal = serializer.save(user=self.request.user)
        self._create_initial_notification(goal)

    @action(detail=True, methods=['post'])
    def update_progress(self, request, pk=None):
        goal = self.get_object()
        serializer = GoalProgressUpdateSerializer(data=request.data)
        
        if serializer.is_valid():
            goal.current_amount = serializer.validated_data['current_amount']
            if 'status' in serializer.validated_data:
                goal.status = serializer.validated_data['status']
            goal.save()
            
            # Notificar si el goal está cerca de completarse
            if goal.progress_percentage >= 90:
                self._notify_goal_progress(goal)
            
            return Response(FinancialGoalSerializer(goal).data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['post'])
    def add_contribution(self, request, pk=None):
        goal = self.get_object()
        serializer = GoalContributionCreateSerializer(data=request.data)
        
        if serializer.is_valid():
            contribution = serializer.save(goal=goal)
            goal.current_amount += contribution.amount
            goal.save()
            
            # Notificar sobre la nueva contribución
            self._notify_contribution(goal, contribution)
            
            return Response(GoalContributionSerializer(contribution).data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['post'])
    def add_milestone(self, request, pk=None):
        goal = self.get_object()
        serializer = GoalMilestoneSerializer(data=request.data)
        
        if serializer.is_valid():
            milestone = serializer.save(goal=goal)
            return Response(GoalMilestoneSerializer(milestone).data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def _create_initial_notification(self, goal):
        notification_service = NotificationService()
        notification_service.create_notification(
            user=goal.user,
            type='goal',
            title='Nuevo Objetivo Financiero Creado',
            message=f'Has creado un nuevo objetivo: {goal.name}',
            priority='medium'
        )

    def _notify_goal_progress(self, goal):
        notification_service = NotificationService()
        notification_service.create_notification(
            user=goal.user,
            type='goal',
            title='¡Objetivo Cerca de Completarse!',
            message=f'Tu objetivo {goal.name} está al {goal.progress_percentage}% de completarse',
            priority='high'
        )

    def _notify_contribution(self, goal, contribution):
        notification_service = NotificationService()
        notification_service.create_notification(
            user=goal.user,
            type='goal',
            title='Nueva Contribución al Objetivo',
            message=f'Has contribuido ${contribution.amount} a tu objetivo {goal.name}',
            priority='medium'
        )

class GoalMilestoneViewSet(viewsets.ModelViewSet):
    serializer_class = GoalMilestoneSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return GoalMilestone.objects.filter(goal__user=self.request.user)

    @action(detail=True, methods=['post'])
    def mark_completed(self, request, pk=None):
        milestone = self.get_object()
        milestone.is_completed = True
        milestone.completed_at = timezone.now()
        milestone.save()
        return Response(GoalMilestoneSerializer(milestone).data)

class GoalContributionViewSet(viewsets.ModelViewSet):
    serializer_class = GoalContributionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return GoalContribution.objects.filter(goal__user=self.request.user) 