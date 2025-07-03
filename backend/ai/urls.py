"""
URLs para los endpoints de IA.
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'interactions', views.AIInteractionViewSet)
router.register(r'insights', views.AIInsightViewSet)
router.register(r'predictions', views.AIPredictionViewSet)

urlpatterns = [
    # Rutas principales de IA
    path('', include(router.urls)),
    
    # Endpoints de análisis y predicción
    path('analyze/transaction/', views.AnalyzeTransactionView.as_view(), name='ai-analyze-transaction'),
    path('predict/expenses/', views.PredictExpensesView.as_view(), name='ai-predict-expenses'),
    path('analyze/behavior/', views.AnalyzeBehaviorView.as_view(), name='ai-analyze-behavior'),
    path('detect/anomalies/', views.DetectAnomaliesView.as_view(), name='ai-detect-anomalies'),
    path('optimize/budget/', views.OptimizeBudgetView.as_view(), name='ai-optimize-budget'),
    path('predict/cashflow/', views.PredictCashFlowView.as_view(), name='ai-predict-cash-flow'),
    path('analyze/risk/', views.AnalyzeRiskView.as_view(), name='ai-analyze-risk'),
    path('recommendations/', views.GetRecommendationsView.as_view(), name='ai-recommendations'),
    
    # Endpoints de entrenamiento y gestión de modelos
    path('train/', views.TrainModelsView.as_view(), name='ai-train-models'),
    path('models/status/', views.ModelsStatusView.as_view(), name='ai-models-status'),
    path('models/evaluate/', views.EvaluateModelsView.as_view(), name='ai-evaluate-models'),
    path('models/update/', views.UpdateModelsView.as_view(), name='ai-update-models'),
    
    # Endpoints de monitoreo y métricas
    path('monitor/performance/', views.MonitorPerformanceView.as_view(), name='ai-monitor-performance'),
    path('metrics/', views.AIMetricsView.as_view(), name='ai-metrics'),
    path('metrics/export/', views.ExportMetricsView.as_view(), name='ai-export-metrics'),
    path('metrics/model/', views.GetModelMetricsView.as_view(), name='ai-get-model-metrics'),
    path('health/', views.AIHealthView.as_view(), name='ai-health'),
    
    # Endpoints de configuración avanzada
    path('config/', views.AIConfigView.as_view(), name='ai-config'),
    path('experiments/', views.AIExperimentsView.as_view(), name='ai-experiments'),
    path('federated/', views.FederatedLearningView.as_view(), name='ai-federated-learning'),
    
    # Endpoints de procesamiento de texto y NLP
    path('nlp/analyze/', views.NLPAnalyzeView.as_view(), name='ai-nlp-analyze'),
    path('nlp/sentiment/', views.NLPSentimentView.as_view(), name='ai-nlp-sentiment'),
    path('nlp/extract/', views.NLPExtractView.as_view(), name='ai-nlp-extract'),
    
    # Endpoints de AutoML
    path('automl/optimize/', views.AutoMLOptimizeView.as_view(), name='ai-automl-optimize'),
    path('automl/status/', views.AutoMLStatusView.as_view(), name='ai-automl-status'),
    
    # Endpoints de A/B Testing
    path('ab-testing/', views.ABTestingView.as_view(), name='ai-ab-testing'),
    path('ab-testing/results/', views.ABTestingResultsView.as_view(), name='ai-ab-testing-results'),

    # Quality Gate endpoints
    path('quality/status/', views.quality_gate_status, name='quality_gate_status'),
    path('quality/retrain/', views.retrain_low_performance_models, name='retrain_models'),
    path('quality/model/<str:model_name>/', views.model_quality_details, name='model_quality_details'),
    path('quality/check/', views.force_quality_check, name='force_quality_check'),
    path('quality/alerts/', views.quality_alerts, name='quality_alerts'),
    path('quality/threshold/', views.update_quality_threshold, name='update_quality_threshold'),

    # Chatbot endpoints
    path('chat/', views.chat_with_ai, name='chat_with_ai'),
    path('chat/stats/', views.get_chat_stats, name='get_chat_stats'),
    path('chat/clear/', views.clear_chat_context, name='clear_chat_context'),
] 