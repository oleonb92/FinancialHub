from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status
from transactions.models import Transaction, Category
from organizations.models import Organization, OrganizationMembership
from accounts.models import User
from django.utils import timezone
from datetime import timedelta
from ai.services import AIService
import json

class AIViewSetTests(TestCase):
    def setUp(self):
        # Crear organización y usuario
        self.organization = Organization.objects.create(name="Test Organization")
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123',
            organization=self.organization
        )
        # Crear membresía explícita
        OrganizationMembership.objects.create(user=self.user, organization=self.organization)
        # Crear varias categorías
        self.category1 = Category.objects.create(
            name="Test Category 1",
            organization=self.organization,
            created_by=self.user
        )
        self.category2 = Category.objects.create(
            name="Test Category 2",
            organization=self.organization,
            created_by=self.user
        )
        # Crear al menos 35 transacciones en días distintos
        self.transactions = []
        base_date = timezone.now() - timedelta(days=34)
        for i in range(35):
            t = Transaction.objects.create(
                amount=100.0 + (i % 5) * 10,
                description=f"Test Transaction {i+1}",
                type="EXPENSE",
                date=base_date + timedelta(days=i),
                category=self.category1 if i % 2 == 0 else self.category2,
                organization=self.organization,
                created_by=self.user
            )
            self.transactions.append(t)
        self.transaction = self.transactions[0]
        self.client = APIClient()
        response = self.client.post('/api/token/', {
            'username': 'testuser',
            'password': 'testpass123'
        }, format='json')
        self.assertEqual(response.status_code, 200)
        self.token = response.data['access']
        self.auth_headers = {
            'HTTP_AUTHORIZATION': f'Bearer {self.token}',
            'HTTP_X_ORGANIZATION_ID': str(self.organization.id)
        }
        # Inicializar y entrenar modelos
        self.ai_service = AIService()
        try:
            self.ai_service.train_models()
        except Exception as e:
            print(f"Error training models: {str(e)}")
            pass

    def test_analyze_transaction_success(self):
        url = reverse('ai-analyze-transaction')
        data = {'transaction_id': self.transaction.id}
        response = self.client.post(url, data, format='json', **self.auth_headers)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('category_suggestion', response.data)
        self.assertIn('confidence_score', response.data)

    def test_analyze_transaction_not_found(self):
        url = reverse('ai-analyze-transaction')
        data = {'transaction_id': 99999}
        response = self.client.post(url, data, format='json', **self.auth_headers)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn('error', response.data)

    def test_analyze_transaction_unauthorized(self):
        url = reverse('ai-analyze-transaction')
        data = {'transaction_id': self.transaction.id}
        response = self.client.post(url, data, format='json')  # Sin headers
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_analyze_transaction_wrong_organization(self):
        # Crear otra organización y usuario
        other_org = Organization.objects.create(name="Other Organization")
        other_user = User.objects.create_user(
            username='otheruser',
            email='other@example.com',
            password='testpass123',
            organization=other_org
        )
        # Crear membresía explícita para el otro usuario
        OrganizationMembership.objects.create(user=other_user, organization=other_org)
        # Obtener token para el otro usuario
        response = self.client.post('/api/token/', {
            'username': 'otheruser',
            'password': 'testpass123'
        }, format='json')
        self.assertEqual(response.status_code, 200)
        other_token = response.data['access']
        other_headers = {
            'HTTP_AUTHORIZATION': f'Bearer {other_token}',
            'HTTP_X_ORGANIZATION_ID': str(other_org.id)
        }
        url = reverse('ai-analyze-transaction')
        data = {'transaction_id': self.transaction.id}
        response = self.client.post(url, data, format='json', **other_headers)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn('error', response.data)

    def test_analyze_transaction_invalid_data(self):
        url = reverse('ai-analyze-transaction')
        data = {}  # Falta transaction_id
        response = self.client.post(url, data, format='json', **self.auth_headers)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.data)

    def test_predict_cash_flow_success(self):
        """Test predicting cash flow successfully"""
        url = reverse('ai-predict-cash-flow')
        response = self.client.get(url, **self.auth_headers)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIsInstance(response.data, list)
        
        if response.data:
            prediction = response.data[0]
            self.assertIn('date', prediction)
            self.assertIn('predicted_amount', prediction)
            self.assertIn('confidence', prediction)
            
    def test_predict_cash_flow_with_days(self):
        """Test predicting cash flow with specific number of days"""
        url = reverse('ai-predict-cash-flow')
        response = self.client.get(f"{url}?days=15", **self.auth_headers)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIsInstance(response.data, list)
        self.assertLessEqual(len(response.data), 15)
        
    def test_predict_cash_flow_invalid_days(self):
        """Test predicting cash flow with invalid days parameter"""
        url = reverse('ai-predict-cash-flow')
        response = self.client.get(f"{url}?days=invalid", **self.auth_headers)
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.data)
        
    def test_predict_cash_flow_unauthorized(self):
        """Test predicting cash flow without authentication"""
        url = reverse('ai-predict-cash-flow')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_detect_anomalies_success(self):
        """Test detecting anomalies successfully"""
        url = reverse('ai-detect-anomalies')
        response = self.client.get(url, **self.auth_headers)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIsInstance(response.data, list)
        
        if response.data:
            anomaly = response.data[0]
            self.assertIn('transaction_id', anomaly)
            self.assertIn('amount', anomaly)
            self.assertIn('date', anomaly)
            self.assertIn('category', anomaly)
            self.assertIn('description', anomaly)
            self.assertIn('anomaly_score', anomaly)
            self.assertIn('reason', anomaly)
            
    def test_detect_anomalies_with_days(self):
        """Test detecting anomalies with specific number of days"""
        url = reverse('ai-detect-anomalies')
        response = self.client.get(f"{url}?days=15", **self.auth_headers)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIsInstance(response.data, list)
        
    def test_detect_anomalies_invalid_days(self):
        """Test detecting anomalies with invalid days parameter"""
        url = reverse('ai-detect-anomalies')
        response = self.client.get(f"{url}?days=invalid", **self.auth_headers)
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.data)
        
    def test_detect_anomalies_unauthorized(self):
        """Test detecting anomalies without authentication"""
        url = reverse('ai-detect-anomalies')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_detect_anomalies_with_anomalous_transaction(self):
        """Test detecting anomalies with an explicitly anomalous transaction"""
        # Crear una transacción con un monto muy alto (anómalo)
        anomalous_transaction = Transaction.objects.create(
            amount=10000.0,  # Monto muy alto
            description="Anomalous Transaction",
            type="EXPENSE",
            date=timezone.now(),
            category=self.category1,
            organization=self.organization,
            created_by=self.user
        )
        
        url = reverse('ai-detect-anomalies')
        response = self.client.get(url, **self.auth_headers)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 1)
        anomaly = data[0]
        # IsolationForest: los scores negativos indican anomalía
        self.assertLess(anomaly['anomaly_score'], 0)  # Score negativo para anomalías

    # Nuevas pruebas para el análisis de riesgo
    def test_analyze_risk_success(self):
        """Test analyzing user risk successfully"""
        url = reverse('ai-analyze-risk')
        response = self.client.get(url, **self.auth_headers)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('risk_score', response.data)
        self.assertIn('risk_level', response.data)
        self.assertIn('metrics', response.data)
        self.assertIn('anomalies', response.data)
        self.assertIn('recommendations', response.data)
        
        # Verificar tipos de datos
        self.assertIsInstance(response.data['risk_score'], float)
        self.assertIsInstance(response.data['risk_level'], str)
        self.assertIsInstance(response.data['metrics'], dict)
        self.assertIsInstance(response.data['anomalies'], list)
        self.assertIsInstance(response.data['recommendations'], list)
        
        # Verificar rangos de valores
        self.assertGreaterEqual(response.data['risk_score'], 0)
        self.assertLessEqual(response.data['risk_score'], 1)
        self.assertIn(response.data['risk_level'], ['low', 'medium', 'high'])

    def test_analyze_risk_unauthorized(self):
        """Test analyzing risk without authentication"""
        url = reverse('ai-analyze-risk')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_analyze_risk_no_transactions(self):
        """Test analyzing risk for user with no transactions"""
        # Crear nuevo usuario sin transacciones
        new_user = User.objects.create_user(
            username='newuser',
            email='new@example.com',
            password='testpass123',
            organization=self.organization
        )
        OrganizationMembership.objects.create(user=new_user, organization=self.organization)
        
        # Obtener token para el nuevo usuario
        response = self.client.post('/api/token/', {
            'username': 'newuser',
            'password': 'testpass123'
        }, format='json')
        self.assertEqual(response.status_code, 200)
        new_token = response.data['access']
        new_headers = {
            'HTTP_AUTHORIZATION': f'Bearer {new_token}',
            'HTTP_X_ORGANIZATION_ID': str(self.organization.id)
        }
        
        url = reverse('ai-analyze-risk')
        response = self.client.get(url, **new_headers)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['risk_score'], 0)
        self.assertEqual(response.data['risk_level'], 'low')
        self.assertEqual(len(response.data['anomalies']), 0)

    # Nuevas pruebas para métricas de rendimiento
    def test_get_model_metrics_success(self):
        """Test getting model metrics successfully"""
        url = reverse('ai-get-model-metrics')
        response = self.client.get(f"{url}?model=transaction_classifier", **self.auth_headers)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('latest', response.data)
        self.assertIn('history', response.data)
        self.assertIn('trends', response.data)
        
        # Verificar estructura de métricas
        self.assertIsInstance(response.data['latest'], dict)
        self.assertIsInstance(response.data['history'], list)
        self.assertIsInstance(response.data['trends'], dict)

    def test_get_model_metrics_invalid_model(self):
        """Test getting metrics for invalid model"""
        url = reverse('ai-get-model-metrics')
        response = self.client.get(f"{url}?model=invalid_model", **self.auth_headers)
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.data)

    def test_get_model_metrics_unauthorized(self):
        """Test getting metrics without authentication"""
        url = reverse('ai-get-model-metrics')
        response = self.client.get(f"{url}?model=transaction_classifier")
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_export_metrics_success(self):
        """Test exporting metrics successfully"""
        url = reverse('ai-export-metrics')
        response = self.client.get(f"{url}?model=transaction_classifier&format=json", **self.auth_headers)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response['Content-Type'], 'application/json')
        
        # Probar exportación CSV
        response = self.client.get(f"{url}?model=transaction_classifier&format=csv", **self.auth_headers)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response['Content-Type'], 'text/csv')
        self.assertIn('attachment', response['Content-Disposition'])

    def test_export_metrics_invalid_format(self):
        """Test exporting metrics with invalid format"""
        url = reverse('ai-export-metrics')
        response = self.client.get(f"{url}?model=transaction_classifier&format=invalid", **self.auth_headers)
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.data)

    def test_export_metrics_unauthorized(self):
        """Test exporting metrics without authentication"""
        url = reverse('ai-export-metrics')
        response = self.client.get(f"{url}?model=transaction_classifier")
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_train_models_success(self):
        """Test training models successfully"""
        url = reverse('ai-train-models')
        response = self.client.post(url, **self.auth_headers)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertEqual(response.data['status'], 'success')
        self.assertIn('models_trained', response.data)

    def test_train_models_unauthorized(self):
        """Test training models without authentication"""
        url = reverse('ai-train-models')
        response = self.client.post(url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

class AIViewsTest(TestCase):
    def setUp(self):
        # Crear usuario y organización
        self.organization = Organization.objects.create(name="Test Org")
        self.user = User.objects.create_user(
            username="testuser",
            email="test@example.com",
            password="testpass123",
            organization=self.organization
        )
        
        # Crear membresía explícita
        OrganizationMembership.objects.create(user=self.user, organization=self.organization)
        
        # Crear categorías
        self.category1 = Category.objects.create(
            name="Groceries",
            organization=self.organization,
            created_by=self.user
        )
        self.category2 = Category.objects.create(
            name="Entertainment",
            organization=self.organization,
            created_by=self.user
        )
        
        # Crear transacciones para análisis
        self.transactions = []
        for i in range(10):
            transaction = Transaction.objects.create(
                created_by=self.user,
                amount=-100.00 if i % 2 == 0 else 500.00,  # Alternar entre gastos e ingresos
                description=f"Test transaction {i}",
                date=timezone.now(),
                category=self.category1 if i % 2 == 0 else self.category2,
                organization=self.organization
            )
            self.transactions.append(transaction)
            
        # Configurar cliente API
        self.client = APIClient()
        
    def authenticate_with_token(self, user):
        response = self.client.post('/api/token/', {
            'username': user.username,
            'password': 'testpass123'
        })
        token = response.data['access']
        self.client.credentials(
            HTTP_AUTHORIZATION=f'Bearer {token}',
            HTTP_X_ORGANIZATION_ID=str(self.organization.id)
        )

    def test_get_recommendations_success(self):
        """Test getting personalized recommendations successfully"""
        self.authenticate_with_token(self.user)
        url = reverse('ai-recommendations')
        response = self.client.get(url)
        
        # Verificar respuesta
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], 'success')
        self.assertIn('recommendations', response.data)
        
        # Verificar estructura de recomendaciones
        recommendations = response.data['recommendations']
        self.assertIsInstance(recommendations, list)
        
        if recommendations:
            recommendation = recommendations[0]
            self.assertIn('type', recommendation)
            self.assertIn('action', recommendation)
            self.assertIn('confidence', recommendation)
            self.assertIn('message', recommendation)
            
    def test_get_recommendations_unauthorized(self):
        """Test getting recommendations without authentication"""
        self.client.credentials(HTTP_X_ORGANIZATION_ID=str(self.organization.id))
        url = reverse('ai-recommendations')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
        
    def test_get_recommendations_no_transactions(self):
        """Test getting recommendations for user with no transactions"""
        new_user = User.objects.create_user(
            username="newuser",
            email="new@example.com",
            password="testpass123",
            organization=self.organization
        )
        # Crear membresía explícita para el nuevo usuario
        OrganizationMembership.objects.create(user=new_user, organization=self.organization)
        self.authenticate_with_token(new_user)
        url = reverse('ai-recommendations')
        response = self.client.get(url)
        
        # Verificar respuesta
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], 'success')
        self.assertEqual(response.data['recommendations'], []) 