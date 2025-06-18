"""
Tests de integración para el sistema de entrenamiento de modelos.
"""
from django.test import TestCase
from django.utils import timezone
from datetime import timedelta
from ..ml.utils.versioning.model_versioning import ModelVersioning
from ..ml.utils.monitoring.resource_monitor import ResourceMonitor
from ..ml.utils.cache.model_cache import ModelCache
from ..tasks.training import train_models, evaluate_models, cleanup_old_versions
from transactions.models import Transaction, Category
from accounts.models import User, Organization
import numpy as np
import os
import shutil

class ModelTrainingIntegrationTests(TestCase):
    def setUp(self):
        """Configuración inicial para los tests"""
        # Crear organización de prueba
        self.organization = Organization.objects.create(
            name="Test Organization"
        )
        
        # Crear usuario de prueba
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123',
            organization=self.organization
        )
        
        # Crear categorías de prueba
        self.categories = [
            Category.objects.create(
                name=f"Category {i}",
                organization=self.organization
            )
            for i in range(3)
        ]
        
        # Crear transacciones de prueba
        self.transactions = []
        for i in range(100):
            transaction = Transaction.objects.create(
                amount=100.0 + i,
                description=f"Test transaction {i}",
                date=timezone.now() - timedelta(days=i),
                category=self.categories[i % 3],
                created_by=self.user,
                organization=self.organization
            )
            self.transactions.append(transaction)
            
        # Inicializar servicios
        self.versioning = ModelVersioning()
        self.monitor = ResourceMonitor()
        self.cache = ModelCache()
        
        # Crear directorio temporal para modelos
        self.models_dir = 'test_models'
        os.makedirs(self.models_dir, exist_ok=True)
        
    def tearDown(self):
        """Limpieza después de los tests"""
        # Eliminar directorio de modelos de prueba
        if os.path.exists(self.models_dir):
            shutil.rmtree(self.models_dir)
            
    def test_train_models_task(self):
        """Test de la tarea de entrenamiento de modelos"""
        # Ejecutar tarea
        result = train_models()
        
        # Verificar resultado
        self.assertEqual(result['status'], 'success')
        self.assertIn('results', result)
        self.assertIn('timestamp', result)
        
        # Verificar que se crearon versiones
        for model_name in result['results']:
            versions = self.versioning.get_model_versions(model_name)
            self.assertGreater(len(versions), 0)
            
    def test_evaluate_models_task(self):
        """Test de la tarea de evaluación de modelos"""
        # Primero entrenar modelos
        train_models()
        
        # Ejecutar evaluación
        result = evaluate_models()
        
        # Verificar resultado
        self.assertEqual(result['status'], 'success')
        self.assertIn('results', result)
        self.assertIn('timestamp', result)
        
        # Verificar métricas
        for model_name, model_result in result['results'].items():
            self.assertEqual(model_result['status'], 'success')
            self.assertIn('metrics', model_result)
            
    def test_cleanup_old_versions_task(self):
        """Test de la tarea de limpieza de versiones"""
        # Crear múltiples versiones
        for i in range(10):
            train_models()
            
        # Ejecutar limpieza
        result = cleanup_old_versions()
        
        # Verificar resultado
        self.assertEqual(result['status'], 'success')
        
        # Verificar que solo quedan 5 versiones
        for model_name in result['results']:
            versions = self.versioning.get_model_versions(model_name)
            self.assertLessEqual(len(versions), 5)
            
    def test_resource_monitoring_during_training(self):
        """Test del monitoreo de recursos durante el entrenamiento"""
        # Iniciar monitoreo
        self.monitor.start_monitoring(interval=1)
        
        # Ejecutar entrenamiento
        train_models()
        
        # Detener monitoreo
        self.monitor.stop_monitoring()
        
        # Verificar métricas
        metrics = self.monitor.get_metrics_history()
        self.assertGreater(len(metrics), 0)
        
        # Verificar que se registraron métricas de CPU y memoria
        latest_metrics = self.monitor.get_latest_metrics()
        self.assertIn('cpu', latest_metrics)
        self.assertIn('memory', latest_metrics)
        
    def test_model_caching(self):
        """Test del sistema de caché de modelos"""
        # Entrenar modelo
        train_models()
        
        # Verificar que el modelo está en caché
        for model_name in ['transaction_classifier', 'expense_predictor']:
            self.assertTrue(self.cache.is_cached(model_name))
            
        # Invalidar caché
        self.cache.clear_all_cache()
        
        # Verificar que se limpió el caché
        cache_info = self.cache.get_cache_info()
        self.assertEqual(cache_info['total_entries'], 0)
        
    def test_model_versioning(self):
        """Test del sistema de versionado de modelos"""
        # Entrenar modelo
        train_models()
        
        # Obtener versiones
        versions = self.versioning.get_model_versions('transaction_classifier')
        self.assertGreater(len(versions), 0)
        
        # Obtener última versión
        latest_version = max(versions.keys())
        model = self.versioning.load_model_version('transaction_classifier', latest_version)
        self.assertIsNotNone(model)
        
        # Realizar rollback
        old_version = min(versions.keys())
        new_version = self.versioning.rollback_model('transaction_classifier', old_version)
        
        # Verificar que se creó nueva versión
        self.assertNotEqual(new_version, old_version)
        self.assertIn(new_version, self.versioning.get_model_versions('transaction_classifier'))
        
    def test_training_with_insufficient_data(self):
        """Test de entrenamiento con datos insuficientes"""
        # Eliminar todas las transacciones
        Transaction.objects.all().delete()
        
        # Intentar entrenar
        result = train_models()
        
        # Verificar que se saltó el entrenamiento
        self.assertEqual(result['status'], 'skipped')
        self.assertEqual(result['reason'], 'insufficient_data')
        
    def test_training_with_high_system_load(self):
        """Test de entrenamiento con alta carga del sistema"""
        # Simular alta carga
        self.monitor.alert_thresholds = {
            'cpu_percent': 0.0,  # Cualquier uso de CPU activará la alerta
            'memory_percent': 0.0
        }
        
        # Intentar entrenar
        result = train_models()
        
        # Verificar que se pospuso el entrenamiento
        self.assertEqual(result['status'], 'postponed')
        self.assertEqual(result['reason'], 'high_system_load')
        self.assertIn('metrics', result) 