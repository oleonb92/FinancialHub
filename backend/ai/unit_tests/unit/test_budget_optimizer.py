"""
Pruebas unitarias para el BudgetOptimizer.
"""
import pytest
from django.test import TestCase
from django.utils import timezone
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from ai.ml.optimizers.budget_optimizer import BudgetOptimizer
from transactions.models import Transaction, Category, Budget
from organizations.models import Organization
from django.contrib.auth import get_user_model

User = get_user_model()

class BudgetOptimizerTestCase(TestCase):
    """Pruebas para el BudgetOptimizer."""
    
    def setUp(self):
        """Configuración inicial para las pruebas."""
        # Crear organización de prueba
        self.organization = Organization.objects.create(
            name="Test Organization"
        )
        
        # Crear usuario de prueba
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        
        # Crear categorías de prueba
        self.category1 = Category.objects.create(
            name="Alimentación",
            organization=self.organization
        )
        self.category2 = Category.objects.create(
            name="Transporte",
            organization=self.organization
        )
        self.category3 = Category.objects.create(
            name="Entretenimiento",
            organization=self.organization
        )
        
        # Crear transacciones de prueba
        self.create_test_transactions()
        
        # Inicializar optimizador
        self.optimizer = BudgetOptimizer()
    
    def create_test_transactions(self):
        """Crea transacciones de prueba para el entrenamiento."""
        base_date = timezone.now() - timedelta(days=30)
        
        # Transacciones de alimentación
        for i in range(10):
            Transaction.objects.create(
                amount=50.0 + (i * 5),
                type='EXPENSE',
                description=f'Compra de alimentos {i}',
                category=self.category1,
                organization=self.organization,
                created_by=self.user,
                date=base_date + timedelta(days=i),
                ai_analyzed=True
            )
        
        # Transacciones de transporte
        for i in range(8):
            Transaction.objects.create(
                amount=30.0 + (i * 3),
                type='EXPENSE',
                description=f'Transporte {i}',
                category=self.category2,
                organization=self.organization,
                created_by=self.user,
                date=base_date + timedelta(days=i*2),
                ai_analyzed=True
            )
        
        # Transacciones de entretenimiento
        for i in range(5):
            Transaction.objects.create(
                amount=100.0 + (i * 10),
                type='EXPENSE',
                description=f'Entretenimiento {i}',
                category=self.category3,
                organization=self.organization,
                created_by=self.user,
                date=base_date + timedelta(days=i*3),
                ai_analyzed=True
            )
    
    def test_initialization(self):
        """Prueba la inicialización del optimizador."""
        self.assertIsNotNone(self.optimizer)
        self.assertFalse(self.optimizer.is_fitted)
        self.assertFalse(self.optimizer.is_trained)
        self.assertEqual(self.optimizer.model_name, 'budget_optimizer')
    
    def test_prepare_training_data(self):
        """Prueba la preparación de datos de entrenamiento."""
        transactions = Transaction.objects.filter(
            organization=self.organization
        )[:5]
        
        training_data = self.optimizer._prepare_training_data(transactions)
        
        self.assertIsNotNone(training_data)
        self.assertGreater(len(training_data), 0)
        self.assertIn('amount', training_data.columns)
        self.assertIn('category_id', training_data.columns)
        self.assertIn('month', training_data.columns)
    
    def test_calculate_seasonal_factor(self):
        """Prueba el cálculo de factores estacionales."""
        # Enero (post-navidad)
        factor_jan = self.optimizer._calculate_seasonal_factor(1)
        self.assertEqual(factor_jan, 1.1)
        
        # Julio (vacaciones)
        factor_jul = self.optimizer._calculate_seasonal_factor(7)
        self.assertEqual(factor_jul, 1.3)
        
        # Diciembre (navidad)
        factor_dec = self.optimizer._calculate_seasonal_factor(12)
        self.assertEqual(factor_dec, 1.4)
    
    def test_calculate_historical_average(self):
        """Prueba el cálculo de promedios históricos."""
        category_id = self.category1.id
        date = timezone.now()
        
        avg = self.optimizer._calculate_historical_average(category_id, date)
        
        # Debería ser mayor que 0 ya que hay transacciones
        self.assertGreater(avg, 0)
    
    def test_train_model(self):
        """Prueba el entrenamiento del modelo."""
        transactions = Transaction.objects.filter(
            organization=self.organization
        )
        
        # Convertir a formato de diccionario
        transaction_data = []
        for t in transactions:
            transaction_data.append({
                'amount': float(t.amount),
                'date': t.date,
                'category_id': t.category.id if t.category else 0
            })
        
        # Entrenar modelo
        self.optimizer.train(transaction_data)
        
        # Verificar que el modelo está entrenado
        self.assertTrue(self.optimizer.is_fitted)
        self.assertTrue(self.optimizer.is_trained)
    
    def test_predict_without_training(self):
        """Prueba que la predicción falle sin entrenamiento."""
        test_data = [{
            'amount': 50.0,
            'date': timezone.now(),
            'category_id': self.category1.id
        }]
        
        with self.assertRaises(RuntimeError):
            self.optimizer.predict(test_data)
    
    def test_predict_with_training(self):
        """Prueba la predicción después del entrenamiento."""
        # Entrenar modelo primero
        transactions = Transaction.objects.filter(
            organization=self.organization
        )
        transaction_data = []
        for t in transactions:
            transaction_data.append({
                'amount': float(t.amount),
                'date': t.date,
                'category_id': t.category.id if t.category else 0
            })
        
        self.optimizer.train(transaction_data)
        
        # Realizar predicción con datos en el mismo formato que el entrenamiento
        test_data = []
        for t in transactions[:1]:  # Solo una transacción para la prueba
            test_data.append({
                'amount': float(t.amount),
                'date': t.date,
                'category_id': t.category.id if t.category else 0
            })
        
        result = self.optimizer.predict(test_data)
        
        # Verificar estructura de respuesta
        self.assertIn('predicted_expense', result)
        self.assertIn('optimization_suggestions', result)
        self.assertIn('efficiency_score', result)
        self.assertIn('confidence', result)
    
    def test_optimize_budget_allocation(self):
        """Prueba la optimización de asignación de presupuesto."""
        # Entrenar modelo primero
        transactions = Transaction.objects.filter(
            organization=self.organization
        )
        transaction_data = []
        for t in transactions:
            transaction_data.append({
                'amount': float(t.amount),
                'date': t.date,
                'category_id': t.category.id if t.category else 0
            })
        
        self.optimizer.train(transaction_data)
        
        # Realizar optimización
        total_budget = 10000.0
        result = self.optimizer.optimize_budget_allocation(
            self.organization.id, total_budget
        )
        
        # Verificar estructura de respuesta
        self.assertIn('suggested_allocation', result)
        self.assertIn('category_analysis', result)
        self.assertIn('recommendations', result)
        
        # Verificar que la suma de asignaciones sea igual al presupuesto total
        if 'suggested_allocation' in result:
            total_allocated = sum(result['suggested_allocation'].values())
            self.assertAlmostEqual(total_allocated, total_budget, places=2)
    
    def test_analyze_budget_efficiency(self):
        """Prueba el análisis de eficiencia presupuestaria."""
        # Crear presupuestos de prueba
        current_period = timezone.now().strftime('%Y-%m')
        
        Budget.objects.create(
            category=self.category1,
            organization=self.organization,
            amount=1000.0,
            period=current_period
        )
        
        Budget.objects.create(
            category=self.category2,
            organization=self.organization,
            amount=500.0,
            period=current_period
        )
        
        # Realizar análisis
        result = self.optimizer.analyze_budget_efficiency(
            self.organization.id
        )
        
        # Verificar estructura de respuesta
        self.assertIn('overall_efficiency', result)
        self.assertIn('category_efficiencies', result)
        self.assertIn('recommendations', result)
    
    def test_get_optimization_recommendation(self):
        """Prueba la generación de recomendaciones de optimización."""
        # Eficiencia muy baja
        recommendation = self.optimizer._get_optimization_recommendation(100, 50, 0.2)
        self.assertIn("reducir gastos", recommendation.lower())
        
        # Eficiencia moderada
        recommendation = self.optimizer._get_optimization_recommendation(100, 50, 0.5)
        self.assertIn("evalúa", recommendation.lower())
        
        # Gasto excede predicción significativamente
        recommendation = self.optimizer._get_optimization_recommendation(150, 100, 0.7)
        self.assertIn("excede", recommendation.lower())
    
    def test_save_and_load_model(self):
        """Prueba el guardado y carga del modelo."""
        # Entrenar modelo
        transactions = Transaction.objects.filter(
            organization=self.organization
        )
        transaction_data = []
        for t in transactions:
            transaction_data.append({
                'amount': float(t.amount),
                'date': t.date,
                'category_id': t.category.id if t.category else 0
            })
        
        self.optimizer.train(transaction_data)
        
        # Guardar modelo
        self.optimizer.save()
        
        # Crear nuevo optimizador y cargar modelo
        new_optimizer = BudgetOptimizer()
        new_optimizer.load()
        
        # Verificar que está entrenado
        self.assertTrue(new_optimizer.is_fitted)
        self.assertTrue(new_optimizer.is_trained)
    
    def test_reset_model(self):
        """Prueba el reseteo del modelo."""
        # Entrenar modelo
        transactions = Transaction.objects.filter(
            organization=self.organization
        )
        transaction_data = []
        for t in transactions:
            transaction_data.append({
                'amount': float(t.amount),
                'date': t.date,
                'category_id': t.category.id if t.category else 0
            })
        
        self.optimizer.train(transaction_data)
        self.assertTrue(self.optimizer.is_trained)
        
        # Resetear modelo
        self.optimizer.reset()
        
        # Verificar que está reseteado
        self.assertFalse(self.optimizer.is_fitted)
        self.assertFalse(self.optimizer.is_trained)
    
    def test_get_model_info(self):
        """Prueba la obtención de información del modelo."""
        info = self.optimizer.get_model_info()
        
        # Verificar estructura de información
        self.assertIn('model_name', info)
        self.assertIn('is_trained', info)
        self.assertIn('feature_names', info)
        self.assertIn('expense_predictor_type', info)
        self.assertIn('efficiency_analyzer_type', info) 