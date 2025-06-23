#!/usr/bin/env python3
"""
Script de configuración para el Sistema de IA Avanzado.

Este script configura todos los componentes del sistema de IA avanzado:
- Instala dependencias adicionales
- Configura modelos pre-entrenados
- Inicializa componentes de federación
- Configura experimentos base
- Valida la instalación
"""
import os
import sys
import subprocess
import json
import logging
from pathlib import Path
from django.conf import settings
from django.core.management import execute_from_command_line
import django

# Configurar Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'financialhub.settings')
django.setup()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedAISetup:
    """Configurador del sistema de IA avanzado."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.backend_dir = self.base_dir / 'backend'
        self.ai_dir = self.backend_dir / 'ai'
        self.ml_models_dir = self.ai_dir / 'ml_models'
        
    def run_setup(self):
        """Ejecuta la configuración completa."""
        logger.info("🚀 Iniciando configuración del Sistema de IA Avanzado")
        
        try:
            # 1. Verificar dependencias
            self.check_dependencies()
            
            # 2. Crear directorios necesarios
            self.create_directories()
            
            # 3. Descargar modelos pre-entrenados
            self.download_pretrained_models()
            
            # 4. Configurar modelos base
            self.setup_base_models()
            
            # 5. Inicializar sistema federado
            self.initialize_federated_system()
            
            # 6. Configurar experimentos base
            self.setup_base_experiments()
            
            # 7. Validar instalación
            self.validate_installation()
            
            logger.info("✅ Configuración del Sistema de IA Avanzado completada exitosamente")
            
        except Exception as e:
            logger.error(f"❌ Error en la configuración: {str(e)}")
            sys.exit(1)
    
    def check_dependencies(self):
        """Verifica que todas las dependencias estén instaladas."""
        logger.info("📦 Verificando dependencias...")
        
        required_packages = [
            'torch', 'transformers', 'optuna', 'scikit-learn',
            'pandas', 'numpy', 'matplotlib', 'seaborn'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"  ✅ {package}")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"  ❌ {package} - No encontrado")
        
        if missing_packages:
            logger.error(f"Faltan las siguientes dependencias: {', '.join(missing_packages)}")
            logger.info("Ejecuta: pip install -r requirements.txt")
            raise Exception("Dependencias faltantes")
    
    def create_directories(self):
        """Crea los directorios necesarios."""
        logger.info("📁 Creando directorios...")
        
        directories = [
            self.ml_models_dir,
            self.ai_dir / 'ml' / 'federated',
            self.ai_dir / 'ml' / 'automl',
            self.ai_dir / 'ml' / 'transformers',
            self.ai_dir / 'ml' / 'experimentation',
            self.ai_dir / 'ml' / 'nlp',
            self.backend_dir / 'logs',
            self.backend_dir / 'staticfiles' / 'ai_models'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"  ✅ {directory}")
    
    def download_pretrained_models(self):
        """Descarga modelos pre-entrenados."""
        logger.info("🤖 Descargando modelos pre-entrenados...")
        
        # Modelos de transformers para español
        try:
            from transformers import AutoTokenizer, AutoModel
            
            # Tokenizer para español
            tokenizer = AutoTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
            tokenizer.save_pretrained(self.ml_models_dir / 'spanish_tokenizer')
            logger.info("  ✅ Tokenizer español descargado")
            
            # Modelo base para español
            model = AutoModel.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
            model.save_pretrained(self.ml_models_dir / 'spanish_model')
            logger.info("  ✅ Modelo español descargado")
            
        except Exception as e:
            logger.warning(f"  ⚠️ No se pudieron descargar modelos pre-entrenados: {str(e)}")
    
    def setup_base_models(self):
        """Configura modelos base del sistema."""
        logger.info("🔧 Configurando modelos base...")
        
        try:
            # Importar servicios
            from ai.ml.classifiers.transaction import TransactionClassifier
            from ai.ml.predictors.expense import ExpensePredictor
            from ai.ml.analyzers.behavior import BehaviorAnalyzer
            
            # Crear modelos base
            models = {
                'transaction_classifier': TransactionClassifier(),
                'expense_predictor': ExpensePredictor(),
                'behavior_analyzer': BehaviorAnalyzer()
            }
            
            # Guardar modelos base
            for name, model in models.items():
                model_path = self.ml_models_dir / f"{name}.joblib"
                model.save(str(model_path))
                logger.info(f"  ✅ {name} configurado")
                
        except Exception as e:
            logger.warning(f"  ⚠️ Error configurando modelos base: {str(e)}")
    
    def initialize_federated_system(self):
        """Inicializa el sistema de aprendizaje federado."""
        logger.info("🌐 Inicializando sistema federado...")
        
        try:
            from ai.ml.federated.federated_learning import FederatedLearningManager
            
            # Crear gestor federado
            federated_manager = FederatedLearningManager()
            
            # Configurar experimento federado base
            experiment_config = {
                'name': 'federated_learning_base',
                'description': 'Experimento base de aprendizaje federado',
                'min_participants': 2,
                'max_rounds': 5
            }
            
            # Guardar configuración
            config_path = self.ai_dir / 'ml' / 'federated' / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(experiment_config, f, indent=2)
            
            logger.info("  ✅ Sistema federado inicializado")
            
        except Exception as e:
            logger.warning(f"  ⚠️ Error inicializando sistema federado: {str(e)}")
    
    def setup_base_experiments(self):
        """Configura experimentos base."""
        logger.info("🧪 Configurando experimentos base...")
        
        try:
            from ai.ml.experimentation.ab_testing import ABTestingManager
            
            ab_manager = ABTestingManager()
            
            # Experimento base para clasificación de transacciones
            experiment = ab_manager.create_experiment(
                name='transaction_classification_optimization',
                description='Optimización de clasificación de transacciones',
                control_model='random_forest_baseline',
                variant_models={
                    'variant_a': 'gradient_boosting_optimized',
                    'variant_b': 'neural_network_advanced'
                },
                traffic_split={'control': 0.4, 'variant_a': 0.3, 'variant_b': 0.3},
                primary_metric='accuracy'
            )
            
            logger.info("  ✅ Experimento base configurado")
            
        except Exception as e:
            logger.warning(f"  ⚠️ Error configurando experimentos: {str(e)}")
    
    def validate_installation(self):
        """Valida que la instalación sea correcta."""
        logger.info("🔍 Validando instalación...")
        
        # Verificar archivos críticos
        critical_files = [
            self.ml_models_dir / 'transaction_classifier.joblib',
            self.ai_dir / 'ml' / 'federated' / 'config.json',
            self.ai_dir / 'ml' / 'nlp' / 'text_processor.py',
            self.ai_dir / 'ml' / 'automl' / 'auto_ml_optimizer.py',
            self.ai_dir / 'ml' / 'transformers' / 'financial_transformer.py',
            self.ai_dir / 'ml' / 'experimentation' / 'ab_testing.py'
        ]
        
        for file_path in critical_files:
            if file_path.exists():
                logger.info(f"  ✅ {file_path.name}")
            else:
                logger.warning(f"  ⚠️ {file_path.name} - No encontrado")
        
        # Verificar imports
        try:
            from ai.ml.nlp.text_processor import TextProcessor
            from ai.ml.automl.auto_ml_optimizer import AutoMLOptimizer
            from ai.ml.transformers.financial_transformer import FinancialTextProcessor
            from ai.ml.experimentation.ab_testing import ABTestingManager
            
            logger.info("  ✅ Todos los módulos importan correctamente")
            
        except Exception as e:
            logger.error(f"  ❌ Error en imports: {str(e)}")
            raise
    
    def create_sample_data(self):
        """Crea datos de muestra para testing."""
        logger.info("📊 Creando datos de muestra...")
        
        try:
            # Crear categorías de muestra
            from transactions.models import Category
            
            sample_categories = [
                {'name': 'Alimentación', 'description': 'Gastos en comida'},
                {'name': 'Transporte', 'description': 'Gastos de transporte'},
                {'name': 'Entretenimiento', 'description': 'Gastos de ocio'},
                {'name': 'Salud', 'description': 'Gastos médicos'},
                {'name': 'Educación', 'description': 'Gastos educativos'}
            ]
            
            for cat_data in sample_categories:
                Category.objects.get_or_create(
                    name=cat_data['name'],
                    defaults={'description': cat_data['description']}
                )
            
            logger.info("  ✅ Datos de muestra creados")
            
        except Exception as e:
            logger.warning(f"  ⚠️ Error creando datos de muestra: {str(e)}")
    
    def run_migrations(self):
        """Ejecuta las migraciones de base de datos."""
        logger.info("🗄️ Ejecutando migraciones...")
        
        try:
            # Ejecutar makemigrations
            execute_from_command_line(['manage.py', 'makemigrations', 'ai'])
            
            # Ejecutar migrate
            execute_from_command_line(['manage.py', 'migrate'])
            
            logger.info("  ✅ Migraciones completadas")
            
        except Exception as e:
            logger.error(f"  ❌ Error en migraciones: {str(e)}")
            raise

def main():
    """Función principal."""
    setup = AdvancedAISetup()
    
    print("=" * 60)
    print("🤖 CONFIGURADOR DEL SISTEMA DE IA AVANZADO")
    print("=" * 60)
    
    # Ejecutar configuración
    setup.run_setup()
    
    # Crear datos de muestra
    setup.create_sample_data()
    
    # Ejecutar migraciones
    setup.run_migrations()
    
    print("\n" + "=" * 60)
    print("🎉 ¡CONFIGURACIÓN COMPLETADA!")
    print("=" * 60)
    print("\nPróximos pasos:")
    print("1. Ejecuta: python manage.py runserver")
    print("2. Accede a: http://localhost:8000/admin")
    print("3. Crea un superusuario si es necesario")
    print("4. Prueba las funcionalidades de IA en la aplicación")
    print("\nPara más información, consulta la documentación.")

if __name__ == "__main__":
    main() 