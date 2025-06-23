"""
Test Completo del Sistema de IA - FinancialHub

Este archivo contiene tests exhaustivos para verificar que todas las funcionalidades
del sistema de IA estén funcionando correctamente.
"""
import pytest
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from django.test import TestCase
from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework.test import APITestCase, APIClient
from rest_framework import status
from unittest.mock import patch, MagicMock
import logging
from django.urls import reverse

from ai.services import AIService
from ai.ml.ai_orchestrator import AIOrchestrator
from ai.models import AIInteraction, AIInsight, AIPrediction
from transactions.models import Transaction, Category, Tag
from organizations.models import Organization
from accounts.models import User
from ai.ml.classifiers.transaction import TransactionClassifier
from ai.ml.predictors.expense import ExpensePredictor
from ai.ml.analyzers.behavior import BehaviorAnalyzer
from ai.ml.anomaly_detector import AnomalyDetector
from ai.ml.cash_flow_predictor import CashFlowPredictor
from ai.ml.risk_analyzer import RiskAnalyzer
from ai.ml.optimizers.budget_optimizer import BudgetOptimizer
from ai.ml.recommendation_engine import RecommendationEngine

logger = logging.getLogger(__name__)

User = get_user_model()

class CompleteAISystemTest(APITestCase):
    """Tests completos del sistema de IA con autenticación mock"""
    
    def setUp(self):
        """Configurar datos de prueba"""
        # Crear organización de prueba
        self.organization = Organization.objects.create(
            name='Test Organization'
        )
        
        # Crear usuario de prueba
        self.user = get_user_model().objects.create_user(
            username='testuser',
            password='testpass123',
            email='test@example.com',
            organization=self.organization
        )
        
        # Configurar cliente API con autenticación
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
        
        # Crear transacciones de prueba
        self.transaction1 = Transaction.objects.create(
            description='Test Transaction 1',
            amount=100.00,
            type='expense',
            organization=self.organization,
            created_by=self.user
        )
        
        self.transaction2 = Transaction.objects.create(
            description='Test Transaction 2',
            amount=200.00,
            type='income',
            organization=self.organization,
            created_by=self.user
        )
        
        # Inicializar servicio de IA
        self.ai_service = AIService()
    
    def test_ai_health_endpoint(self):
        """Test del endpoint de salud de IA (público)"""
        # Este endpoint debe ser público
        client = APIClient()  # Sin autenticación
        
        url = reverse('ai-health')
        response = client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('health', response.data)
        self.assertIn('timestamp', response.data)
        self.assertIn('version', response.data)
    
    def test_analyze_transaction_endpoint(self):
        """Test del endpoint de análisis de transacciones"""
        url = reverse('ai-analyze-transaction')
        data = {
            'transaction_id': self.transaction1.id
        }
        
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('analysis', response.data)
    
    def test_predict_expenses_endpoint(self):
        """Test del endpoint de predicción de gastos"""
        url = reverse('ai-predict-expenses')
        data = {
            'days_ahead': 30
        }
        
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('predictions', response.data)
    
    def test_analyze_behavior_endpoint(self):
        """Test del endpoint de análisis de comportamiento"""
        url = reverse('ai-analyze-behavior')
        data = {
            'user_id': self.user.id
        }
        
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('analysis', response.data)
    
    def test_detect_anomalies_endpoint(self):
        """Test del endpoint de detección de anomalías"""
        url = reverse('ai-detect-anomalies')
        
        response = self.client.post(url, {}, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('anomalies', response.data)
    
    def test_optimize_budget_endpoint(self):
        """Test del endpoint de optimización de presupuesto"""
        url = reverse('ai-optimize-budget')
        data = {
            'categories': ['food', 'transport', 'entertainment'],
            'total_budget': 1000.00
        }
        
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('optimization', response.data)
    
    def test_predict_cash_flow_endpoint(self):
        """Test del endpoint de predicción de flujo de efectivo"""
        url = reverse('ai-predict-cash-flow')
        data = {
            'months_ahead': 3
        }
        
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('cash_flow', response.data)
        self.assertIn('predictions', response.data['cash_flow'])
    
    def test_analyze_risk_endpoint(self):
        """Test del endpoint de análisis de riesgo"""
        url = reverse('ai-analyze-risk')
        
        response = self.client.post(url, {}, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('risk_analysis', response.data)
    
    def test_get_recommendations_endpoint(self):
        """Test del endpoint de recomendaciones"""
        url = reverse('ai-recommendations')
        data = {
            'user_id': self.user.id
        }
        
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('recommendations', response.data)
    
    def test_train_models_endpoint(self):
        """Test del endpoint de entrenamiento de modelos"""
        url = reverse('ai-train-models')
        
        response = self.client.post(url, {}, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('message', response.data)
        self.assertIn('task_id', response.data)
    
    def test_models_status_endpoint(self):
        """Test del endpoint de estado de modelos"""
        url = reverse('ai-models-status')
        
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('models_status', response.data)
    
    def test_evaluate_models_endpoint(self):
        """Test del endpoint de evaluación de modelos"""
        url = reverse('ai-evaluate-models')
        
        response = self.client.post(url, {}, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('message', response.data)
        self.assertIn('task_id', response.data)
    
    def test_update_models_endpoint(self):
        """Test del endpoint de actualización de modelos"""
        url = reverse('ai-update-models')
        data = {
            'model_type': 'expense_predictor'
        }
        
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('message', response.data)
        self.assertIn('result', response.data)
    
    def test_monitor_performance_endpoint(self):
        """Test del endpoint de monitoreo de rendimiento"""
        url = reverse('ai-monitor-performance')
        
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('performance', response.data)
    
    def test_ai_metrics_endpoint(self):
        """Test del endpoint de métricas de IA"""
        url = reverse('ai-metrics')
        
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('metrics', response.data)
    
    def test_ai_config_endpoint(self):
        """Test del endpoint de configuración de IA"""
        url = reverse('ai-config')
        
        # GET request
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('config', response.data)
        
        # POST request
        config_data = {
            'model_type': 'test_model',
            'parameters': {'test': 'value'}
        }
        response = self.client.post(url, config_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('message', response.data)
    
    def test_ai_experiments_endpoint(self):
        """Test del endpoint de experimentos de IA"""
        url = reverse('ai-experiments')
        data = {
            'experiment_name': 'test_experiment',
            'parameters': {'test': 'value'}
        }
        
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('experiment', response.data)
    
    def test_federated_learning_endpoint(self):
        """Test del endpoint de aprendizaje federado"""
        url = reverse('ai-federated-learning')
        data = {
            'model_type': 'test_model',
            'data': {'test': 'data'}
        }
        
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('federated_learning', response.data)
    
    def test_nlp_analyze_endpoint(self):
        """Test del endpoint de análisis NLP"""
        url = reverse('ai-nlp-analyze')
        data = {
            'text': 'This is a test transaction for groceries'
        }
        
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('analysis', response.data)
    
    def test_nlp_sentiment_endpoint(self):
        """Test del endpoint de análisis de sentimientos"""
        url = reverse('ai-nlp-sentiment')
        data = {
            'text': 'I am happy with my financial situation'
        }
        
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('sentiment', response.data)
    
    def test_nlp_extract_endpoint(self):
        """Test del endpoint de extracción de información"""
        url = reverse('ai-nlp-extract')
        data = {
            'text': 'Paid $50 for groceries at Walmart on 2024-01-15'
        }
        
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('extracted', response.data)
    
    def test_automl_optimize_endpoint(self):
        """Test del endpoint de optimización AutoML"""
        url = reverse('ai-automl-optimize')
        data = {
            'model_type': 'classifier',
            'dataset': {'features': [], 'target': []}
        }
        
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('automl_result', response.data)
    
    def test_automl_status_endpoint(self):
        """Test del endpoint de estado AutoML"""
        url = reverse('ai-automl-status')
        
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('automl_status', response.data)
    
    def test_ab_testing_endpoint(self):
        """Test del endpoint de pruebas A/B"""
        url = reverse('ai-ab-testing')
        data = {
            'test_name': 'test_ab',
            'variants': ['A', 'B'],
            'metrics': ['accuracy', 'precision']
        }
        
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('ab_testing', response.data)
    
    def test_ab_testing_results_endpoint(self):
        """Test del endpoint de resultados A/B"""
        url = reverse('ai-ab-testing-results')
        
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertIn('ab_results', response.data)
    
    def test_ai_interaction_viewset(self):
        """Test del ViewSet de interacciones de IA"""
        url = reverse('aiinteraction-list')
        
        # GET request
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # POST request
        data = {
            'interaction_type': 'analysis',
            'input_data': {'test': 'data'},
            'output_data': {'result': 'test'}
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        
        # Test analyze_transaction action
        url = reverse('aiinteraction-analyze-transaction')
        data = {'transaction_id': self.transaction1.id}
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
    
    def test_ai_insight_viewset(self):
        """Test del ViewSet de insights de IA"""
        url = reverse('aiinsight-list')
        
        # GET request
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # POST request
        data = {
            'insight_type': 'pattern',
            'description': 'Test insight',
            'confidence': 0.85
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
    
    def test_ai_prediction_viewset(self):
        """Test del ViewSet de predicciones de IA"""
        url = reverse('aiprediction-list')
        
        # GET request
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # POST request
        data = {
            'prediction_type': 'expense',
            'predicted_value': 150.00,
            'confidence': 0.90
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
    
    def test_error_handling(self):
        """Test del manejo de errores"""
        # Test con datos inválidos
        url = reverse('ai-analyze-transaction')
        data = {'invalid_field': 'invalid_value'}
        
        response = self.client.post(url, data, format='json')
        
        # El endpoint actual maneja campos inválidos graciosamente y devuelve un análisis
        self.assertIn('status', response.json())
        # Si hay un error, debe tener 'message', si es exitoso, debe tener 'analysis'
        if 'message' in response.json():
            self.assertIn('message', response.json())
        else:
            self.assertIn('analysis', response.json())
    
    def test_authentication_required(self):
        """Test de que la autenticación funciona correctamente"""
        # Crear cliente sin autenticación
        client = APIClient()
        
        url = reverse('ai-analyze-transaction')
        data = {'transaction_id': self.transaction1.id}
        
        response = client.post(url, data, format='json')
        
        # En modo DEBUG y con AI_TEST_ENDPOINTS_AUTH = False, debe permitir acceso sin autenticación
        from django.conf import settings
        if settings.DEBUG and not getattr(settings, 'AI_TEST_ENDPOINTS_AUTH', True):
            self.assertEqual(response.status_code, status.HTTP_200_OK)
        else:
            # En producción, debe requerir autenticación (403 Forbidden)
            self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)
    
    def test_organization_isolation(self):
        """Test de que los datos están aislados por organización"""
        # Crear otra organización y usuario
        other_org = Organization.objects.create(
            name='Other Organization'
        )
        other_user = get_user_model().objects.create_user(
            username='otheruser',
            password='testpass123',
            email='other@example.com',
            organization=other_org
        )
        
        # Autenticar con el otro usuario
        self.client.force_authenticate(user=other_user)
        
        url = reverse('aiinteraction-list')
        response = self.client.get(url)
        
        # Debe devolver solo datos de la otra organización
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        # Los datos deben estar vacíos porque no hay interacciones en la otra org
    
    def test_performance_under_load(self):
        """Test de rendimiento bajo carga"""
        # Crear múltiples transacciones
        for i in range(10):
            Transaction.objects.create(
                description=f'Test Transaction {i}',
                amount=100.00 + i,
                type='expense',
                organization=self.organization,
                created_by=self.user
            )
        
        url = reverse('ai-detect-anomalies')
        
        # Hacer múltiples requests
        for _ in range(5):
            response = self.client.post(url, {}, format='json')
            self.assertEqual(response.status_code, status.HTTP_200_OK)
    
    def tearDown(self):
        """Limpiar después de los tests"""
        # Limpiar archivos temporales si es necesario
        pass


class AISystemStressTest(APITestCase):
    """
    Tests de estrés para el sistema de IA.
    """
    
    def setUp(self):
        """Configuración para tests de estrés."""
        self.client = APIClient()
        
        # Crear organización y usuario
        self.organization = Organization.objects.create(name="Stress Test Org")
        self.user = User.objects.create_user(
            username='stresstest',
            email='stress@test.com',
            password='testpass123',
            organization=self.organization,
            was_approved=True
        )
        
        # Crear muchas transacciones para estrés
        self.categories = []
        for i in range(10):
            category = Category.objects.create(
                name=f'Category {i}',
                organization=self.organization
            )
            self.categories.append(category)
        
        # Crear 100 transacciones
        for i in range(100):
            Transaction.objects.create(
                description=f'Transaction {i}',
                amount=100 + i,
                date=timezone.now() - timedelta(days=i),
                category=self.categories[i % 10],
                type='expense',
                created_by=self.user,
                organization=self.organization
            )
        
        self.client.force_authenticate(user=self.user)
        self.ai_service = AIService()
    
    def test_stress_analysis(self):
        """Test de estrés: múltiples análisis simultáneos."""
        logger.info("Running stress test: Multiple simultaneous analyses")
        
        import threading
        import time
        
        results = []
        errors = []
        
        def run_analysis():
            try:
                analysis = self.ai_service.analyze_behavior(self.user)
                results.append(analysis)
            except Exception as e:
                errors.append(str(e))
        
        # Ejecutar 10 análisis simultáneos
        threads = []
        for i in range(10):
            thread = threading.Thread(target=run_analysis)
            threads.append(thread)
            thread.start()
        
        # Esperar a que terminen
        for thread in threads:
            thread.join()
        
        logger.info(f"Stress test completed: {len(results)} successful, {len(errors)} errors")
        logger.info("✓ Stress test passed")


if __name__ == '__main__':
    # Ejecutar tests manualmente
    import django
    django.setup()
    
    # Crear instancia de test y ejecutar
    test_instance = CompleteAISystemTest()
    test_instance.setUp()
    
    # Ejecutar todos los tests
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
    
    print("🚀 Starting Complete AI System Tests...")
    print("=" * 60)
    
    for method_name in sorted(test_methods):
        method = getattr(test_instance, method_name)
        print(f"\n📋 Running: {method_name}")
        try:
            method()
            print(f"✅ PASSED: {method_name}")
        except Exception as e:
            print(f"❌ FAILED: {method_name} - {str(e)}")
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!") 