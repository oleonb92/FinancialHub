from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import FinancialGoalViewSet, GoalMilestoneViewSet, GoalContributionViewSet

router = DefaultRouter()
router.register(r'goals', FinancialGoalViewSet, basename='goal')
router.register(r'milestones', GoalMilestoneViewSet, basename='goal-milestone')
router.register(r'contributions', GoalContributionViewSet, basename='goal-contribution')

urlpatterns = [
    path('', include(router.urls)),
] 