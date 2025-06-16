from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import AIInteractionViewSet, AIInsightViewSet, AIPredictionViewSet

router = DefaultRouter()
router.register(r'interactions', AIInteractionViewSet, basename='ai-interaction')
router.register(r'insights', AIInsightViewSet, basename='ai-insight')
router.register(r'predictions', AIPredictionViewSet, basename='ai-prediction')

urlpatterns = [
    path('', include(router.urls)),
] 