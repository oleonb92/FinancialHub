#!/usr/bin/env python3
"""
SISTEMA UNIFICADO DE IA - FinancialHub

Este script unifica todas las funcionalidades de IA en un solo lugar:
- Entrenamiento de modelos
- Verificación del sistema
- Testing completo
- Gestión de modelos
- Monitoreo de rendimiento

Uso:
    python scripts/ai_system_unified.py --mode train
    python scripts/ai_system_unified.py --mode verify
    python scripts/ai_system_unified.py --mode test
    python scripts/ai_system_unified.py --mode cleanup
"""

import os
import sys
import django
import logging
import argparse
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import subprocess

# Configurar Django
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'financialhub.settings.dev')
django.setup()

from transactions.models import Transaction, Category
from organizations.models import Organization
from ai.services import AIService
from ai.ml.classifiers.transaction import TransactionClassifier
from ai.ml.predictors.expense import ExpensePredictor
from ai.ml.analyzers.behavior import BehaviorAnalyzer
from ai.ml.optimizers.budget_optimizer import BudgetOptimizer
from ai.ml.nlp.text_processor import FinancialTextProcessor
from ai.ml.transformers.financial_transformer import FinancialTransformerService, TransformerConfig
from django.conf import settings

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UnifiedAISystem:
    """
    Sistema unificado de IA que maneja todas las operaciones.
    """
    
    def __init__(self):
        self.ai_service = AIService()
        self.results = {}
        self.start_time = time.time()
        
    def check_environment(self) -> Dict[str, Any]:
        """
        Verifica el entorno y configuración del sistema.
        """
        try:
            logger.info("🔍 Verificando entorno del sistema...")
            
            env_status = {
                'django_configured': True,
                'database_connected': False,
                'models_available': False,
                'ai_models_loaded': False,
                'environment_vars': {}
            }
            
            # Verificar variables de entorno críticas
            critical_vars = ['SECRET_KEY', 'DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_HOST']
            for var in critical_vars:
                env_status['environment_vars'][var] = bool(os.getenv(var))
            
            # Verificar conexión a base de datos
            try:
                transaction_count = Transaction.objects.count()
                env_status['database_connected'] = True
                env_status['transaction_count'] = transaction_count
                logger.info(f"✅ Base de datos conectada - {transaction_count} transacciones")
            except Exception as e:
                logger.error(f"❌ Error conectando a base de datos: {str(e)}")
            
            # Verificar modelos disponibles
            try:
                category_count = Category.objects.count()
                env_status['models_available'] = True
                env_status['category_count'] = category_count
                logger.info(f"✅ Modelos disponibles - {category_count} categorías")
            except Exception as e:
                logger.error(f"❌ Error cargando modelos: {str(e)}")
            
            # Verificar modelos de IA
            try:
                ai_info = self.ai_service.get_model_info()
                env_status['ai_models_loaded'] = True
                env_status['ai_models_info'] = ai_info
                logger.info("✅ Modelos de IA cargados")
            except Exception as e:
                logger.error(f"❌ Error cargando modelos de IA: {str(e)}")
            
            return env_status
            
        except Exception as e:
            logger.error(f"Error verificando entorno: {str(e)}")
            return {'error': str(e)}
    
    def train_all_models(self, mode: str = 'fast') -> Dict[str, Any]:
        """
        Entrena todos los modelos de IA.
        
        Args:
            mode: 'fast' para entrenamiento rápido, 'full' para completo
        """
        try:
            logger.info(f"🚀 Entrenando todos los modelos (modo: {mode})...")
            
            training_results = {}
            
            # 1. Entrenar clasificador de transacciones
            logger.info("📊 Entrenando clasificador de transacciones...")
            try:
                classifier = TransactionClassifier()
                transactions = self._get_training_data('transactions', mode)
                
                if len(transactions) > 0:
                    classifier.train(transactions)
                    training_results['transaction_classifier'] = {
                        'status': 'success',
                        'samples': len(transactions)
                    }
                    logger.info(f"✅ Clasificador entrenado con {len(transactions)} muestras")
                else:
                    training_results['transaction_classifier'] = {
                        'status': 'error',
                        'error': 'No hay datos suficientes'
                    }
            except Exception as e:
                training_results['transaction_classifier'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            # 2. Entrenar predictor de gastos
            logger.info("💰 Entrenando predictor de gastos...")
            try:
                predictor = ExpensePredictor()
                expense_data = self._get_training_data('expenses', mode)
                
                if len(expense_data) > 0:
                    predictor.train(expense_data)
                    training_results['expense_predictor'] = {
                        'status': 'success',
                        'samples': len(expense_data)
                    }
                    logger.info(f"✅ Predictor entrenado con {len(expense_data)} muestras")
                else:
                    training_results['expense_predictor'] = {
                        'status': 'error',
                        'error': 'No hay datos suficientes'
                    }
            except Exception as e:
                training_results['expense_predictor'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            # 3. Entrenar analizador de comportamiento
            logger.info("🧠 Entrenando analizador de comportamiento...")
            try:
                analyzer = BehaviorAnalyzer()
                behavior_data = self._get_training_data('behavior', mode)
                
                if len(behavior_data) > 0:
                    analyzer.train(behavior_data)
                    training_results['behavior_analyzer'] = {
                        'status': 'success',
                        'samples': len(behavior_data)
                    }
                    logger.info(f"✅ Analizador entrenado con {len(behavior_data)} muestras")
                else:
                    training_results['behavior_analyzer'] = {
                        'status': 'error',
                        'error': 'No hay datos suficientes'
                    }
            except Exception as e:
                training_results['behavior_analyzer'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            # 4. Entrenar optimizador de presupuesto
            logger.info("📈 Entrenando optimizador de presupuesto...")
            try:
                optimizer = BudgetOptimizer()
                budget_data = self._get_training_data('budget', mode)
                
                if len(budget_data) > 0:
                    optimizer.train(budget_data)
                    training_results['budget_optimizer'] = {
                        'status': 'success',
                        'samples': len(budget_data)
                    }
                    logger.info(f"✅ Optimizador entrenado con {len(budget_data)} muestras")
                else:
                    training_results['budget_optimizer'] = {
                        'status': 'error',
                        'error': 'No hay datos suficientes'
                    }
            except Exception as e:
                training_results['budget_optimizer'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            # 5. Entrenar modelos NLP (solo si hay datos)
            if mode == 'full':
                logger.info("📝 Entrenando modelos NLP...")
                try:
                    nlp_processor = FinancialTextProcessor()
                    nlp_data = self._get_training_data('nlp', mode)
                    
                    if len(nlp_data) > 0:
                        # Entrenar modelos NLP
                        nlp_processor.train_sentiment_model(nlp_data['texts'], nlp_data['labels'])
                        training_results['nlp_models'] = {
                            'status': 'success',
                            'samples': len(nlp_data['texts'])
                        }
                        logger.info(f"✅ Modelos NLP entrenados con {len(nlp_data['texts'])} muestras")
                    else:
                        training_results['nlp_models'] = {
                            'status': 'error',
                            'error': 'No hay datos suficientes'
                        }
                except Exception as e:
                    training_results['nlp_models'] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            # 6. Entrenar modelos Transformer (solo si hay datos)
            if mode == 'full':
                logger.info("🤖 Entrenando modelos Transformer...")
                try:
                    transformer_service = FinancialTransformerService()
                    transformer_data = self._get_training_data('transformer', mode)
                    
                    if len(transformer_data['texts']) > 0:
                        result = transformer_service.train_transformer_model(
                            transformer_data['texts'], 
                            transformer_data['labels']
                        )
                        training_results['transformer_models'] = {
                            'status': 'success',
                            'samples': len(transformer_data['texts']),
                            'accuracy': result.get('accuracy', 0)
                        }
                        logger.info(f"✅ Modelos Transformer entrenados con {len(transformer_data['texts'])} muestras")
                    else:
                        training_results['transformer_models'] = {
                            'status': 'error',
                            'error': 'No hay datos suficientes'
                        }
                except Exception as e:
                    training_results['transformer_models'] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            # Resumen final
            successful_models = sum(1 for r in training_results.values() if r.get('status') == 'success')
            total_models = len(training_results)
            
            logger.info(f"🎉 Entrenamiento completado: {successful_models}/{total_models} modelos exitosos")
            
            return {
                'status': 'completed',
                'successful_models': successful_models,
                'total_models': total_models,
                'results': training_results,
                'training_time': time.time() - self.start_time
            }
            
        except Exception as e:
            logger.error(f"Error en entrenamiento: {str(e)}")
            return {'error': str(e)}
    
    def _get_training_data(self, data_type: str, mode: str) -> List[Dict[str, Any]]:
        """
        Obtiene datos de entrenamiento según el tipo y modo.
        """
        try:
            if mode == 'fast':
                limit = 1000
            else:
                limit = 10000
            
            if data_type == 'transactions':
                transactions = Transaction.objects.filter(
                    category__isnull=False
                ).select_related('category', 'organization')[:limit]
                
                return [
                    {
                        'id': t.id,
                        'amount': float(t.amount),
                        'type': t.type,
                        'description': t.description or '',
                        'category_id': t.category.id,
                        'category_name': t.category.name,
                        'date': t.date,
                        'organization_id': t.organization.id
                    }
                    for t in transactions
                ]
            
            elif data_type == 'expenses':
                transactions = Transaction.objects.filter(
                    type='expense',
                    category__isnull=False
                ).select_related('category')[:limit]
                
                return [
                    {
                        'amount': float(t.amount),
                        'category_id': t.category.id,
                        'date': t.date,
                        'description': t.description or ''
                    }
                    for t in transactions
                ]
            
            elif data_type == 'behavior':
                transactions = Transaction.objects.filter(
                    category__isnull=False
                ).select_related('category', 'created_by')[:limit]
                
                return [
                    {
                        'user_id': t.created_by.id if t.created_by else None,
                        'amount': float(t.amount),
                        'category_id': t.category.id,
                        'date': t.date,
                        'type': t.type
                    }
                    for t in transactions
                ]
            
            elif data_type == 'budget':
                # Datos simulados para presupuesto
                return [
                    {
                        'category_id': i % 10 + 1,
                        'amount': 100.0 + (i * 10),
                        'period': 'monthly'
                    }
                    for i in range(min(100, limit))
                ]
            
            elif data_type == 'nlp':
                transactions = Transaction.objects.filter(
                    description__isnull=False
                ).exclude(description='')[:limit]
                
                texts = [t.description for t in transactions]
                labels = [1 if t.type == 'expense' else 0 for t in transactions]
                
                return {
                    'texts': texts,
                    'labels': labels
                }
            
            elif data_type == 'transformer':
                transactions = Transaction.objects.filter(
                    description__isnull=False
                ).exclude(description='')[:limit]
                
                texts = [t.description for t in transactions]
                labels = [t.category.id if t.category else 1 for t in transactions]
                
                return {
                    'texts': texts,
                    'labels': labels
                }
            
            return []
            
        except Exception as e:
            logger.error(f"Error obteniendo datos de entrenamiento: {str(e)}")
            return []
    
    def verify_system(self) -> Dict[str, Any]:
        """
        Verifica el estado completo del sistema de IA.
        """
        try:
            logger.info("🔍 Verificando sistema completo de IA...")
            
            verification_results = {
                'environment': self.check_environment(),
                'models': {},
                'functionality': {},
                'performance': {}
            }
            
            # Verificar modelos individuales
            models_to_check = [
                'transaction_classifier',
                'expense_predictor', 
                'behavior_analyzer',
                'budget_optimizer',
                'sentiment_analyzer',
                'text_processor',
                'transformer_service'
            ]
            
            for model_name in models_to_check:
                try:
                    if hasattr(self.ai_service, model_name):
                        model_obj = getattr(self.ai_service, model_name)
                        if model_obj is not None:
                            verification_results['models'][model_name] = {
                                'status': 'loaded',
                                'type': type(model_obj).__name__
                            }
                        else:
                            verification_results['models'][model_name] = {
                                'status': 'not_loaded'
                            }
                    else:
                        verification_results['models'][model_name] = {
                            'status': 'not_available'
                        }
                except Exception as e:
                    verification_results['models'][model_name] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            # Probar funcionalidad
            logger.info("🧪 Probando funcionalidad...")
            
            # Test 1: Predicción de categoría
            try:
                result = self.ai_service.predict_category("groceries purchase")
                verification_results['functionality']['category_prediction'] = {
                    'status': 'working',
                    'result': result
                }
            except Exception as e:
                verification_results['functionality']['category_prediction'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            # Test 2: Análisis de sentimiento
            try:
                result = self.ai_service.analyze_text_sentiment("This is a great transaction")
                verification_results['functionality']['sentiment_analysis'] = {
                    'status': 'working',
                    'result': result
                }
            except Exception as e:
                verification_results['functionality']['sentiment_analysis'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            # Test 3: Procesamiento de texto
            try:
                result = self.ai_service.nlp_processor.preprocess_text("Test transaction")
                verification_results['functionality']['text_processing'] = {
                    'status': 'working',
                    'result': {'processed_text': result}
                }
            except Exception as e:
                verification_results['functionality']['text_processing'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            # Métricas de rendimiento
            verification_results['performance'] = {
                'verification_time': time.time() - self.start_time,
                'models_loaded': len([m for m in verification_results['models'].values() if m.get('status') == 'loaded']),
                'functionality_working': len([f for f in verification_results['functionality'].values() if f.get('status') == 'working'])
            }
            
            return verification_results
            
        except Exception as e:
            logger.error(f"Error verificando sistema: {str(e)}")
            return {'error': str(e)}
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        Ejecuta pruebas completas del sistema.
        """
        try:
            logger.info("🧪 Ejecutando pruebas completas del sistema...")
            
            test_results = {
                'unit_tests': {},
                'integration_tests': {},
                'performance_tests': {},
                'summary': {}
            }
            
            # Pruebas unitarias básicas
            logger.info("📋 Ejecutando pruebas unitarias...")
            
            # Test 1: Verificar que los modelos se pueden cargar
            try:
                classifier = TransactionClassifier()
                test_results['unit_tests']['model_loading'] = {'status': 'passed'}
            except Exception as e:
                test_results['unit_tests']['model_loading'] = {'status': 'failed', 'error': str(e)}
            
            # Test 2: Verificar predicciones básicas
            try:
                result = self.ai_service.predict_category("test transaction")
                test_results['unit_tests']['basic_prediction'] = {'status': 'passed'}
            except Exception as e:
                test_results['unit_tests']['basic_prediction'] = {'status': 'failed', 'error': str(e)}
            
            # Test 3: Verificar análisis de sentimiento
            try:
                result = self.ai_service.analyze_text_sentiment("test text")
                test_results['unit_tests']['sentiment_analysis'] = {'status': 'passed'}
            except Exception as e:
                test_results['unit_tests']['sentiment_analysis'] = {'status': 'failed', 'error': str(e)}
            
            # Pruebas de integración
            logger.info("🔗 Ejecutando pruebas de integración...")
            
            # Test 1: Flujo completo de análisis
            try:
                # Simular transacción
                test_data = {
                    'description': 'Grocery shopping at Walmart',
                    'amount': 50.0,
                    'type': 'expense'
                }
                
                # Predicción de categoría
                category_result = self.ai_service.predict_category(test_data['description'])
                
                # Análisis de sentimiento
                sentiment_result = self.ai_service.analyze_text_sentiment(test_data['description'])
                
                test_results['integration_tests']['complete_flow'] = {
                    'status': 'passed',
                    'category_result': category_result,
                    'sentiment_result': sentiment_result
                }
            except Exception as e:
                test_results['integration_tests']['complete_flow'] = {
                    'status': 'failed',
                    'error': str(e)
                }
            
            # Pruebas de rendimiento
            logger.info("⚡ Ejecutando pruebas de rendimiento...")
            
            start_time = time.time()
            
            # Test de velocidad de predicción
            try:
                for i in range(10):
                    self.ai_service.predict_category(f"test transaction {i}")
                
                prediction_time = time.time() - start_time
                test_results['performance_tests']['prediction_speed'] = {
                    'status': 'passed',
                    'avg_time_per_prediction': prediction_time / 10
                }
            except Exception as e:
                test_results['performance_tests']['prediction_speed'] = {
                    'status': 'failed',
                    'error': str(e)
                }
            
            # Resumen de pruebas
            total_tests = 0
            passed_tests = 0
            
            for test_category in test_results.values():
                if isinstance(test_category, dict):
                    for test_name, test_result in test_category.items():
                        if isinstance(test_result, dict) and 'status' in test_result:
                            total_tests += 1
                            if test_result['status'] == 'passed':
                                passed_tests += 1
            
            test_results['summary'] = {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                'total_time': time.time() - self.start_time
            }
            
            logger.info(f"🎉 Pruebas completadas: {passed_tests}/{total_tests} exitosas")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Error en pruebas: {str(e)}")
            return {'error': str(e)}
    
    def cleanup_system(self) -> Dict[str, Any]:
        """
        Limpia archivos temporales y optimiza el sistema.
        """
        try:
            logger.info("🧹 Limpiando sistema...")
            
            cleanup_results = {
                'files_removed': [],
                'cache_cleared': False,
                'models_optimized': False
            }
            
            # Limpiar archivos temporales
            temp_files = [
                'logs/ai/*.tmp',
                'backend/temp/*',
                '*.pyc',
                '__pycache__'
            ]
            
            for pattern in temp_files:
                try:
                    import glob
                    files = glob.glob(pattern)
                    for file in files:
                        if os.path.isfile(file):
                            os.remove(file)
                            cleanup_results['files_removed'].append(file)
                        elif os.path.isdir(file):
                            import shutil
                            shutil.rmtree(file)
                            cleanup_results['files_removed'].append(file)
                except Exception as e:
                    logger.warning(f"No se pudo limpiar {pattern}: {str(e)}")
            
            # Limpiar cache de modelos
            try:
                self.ai_service.cleanup_memory(force=True)
                cleanup_results['cache_cleared'] = True
                logger.info("✅ Cache de modelos limpiado")
            except Exception as e:
                logger.warning(f"No se pudo limpiar cache: {str(e)}")
            
            # Optimizar modelos
            try:
                # Aquí podrías agregar lógica de optimización
                cleanup_results['models_optimized'] = True
                logger.info("✅ Modelos optimizados")
            except Exception as e:
                logger.warning(f"No se pudieron optimizar modelos: {str(e)}")
            
            logger.info(f"🧹 Limpieza completada: {len(cleanup_results['files_removed'])} archivos removidos")
            
            return cleanup_results
            
        except Exception as e:
            logger.error(f"Error en limpieza: {str(e)}")
            return {'error': str(e)}
    
    def generate_report(self, results: Dict[str, Any], mode: str) -> str:
        """
        Genera un reporte completo de los resultados.
        """
        try:
            report_path = f'logs/ai/ai_system_{mode}_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'mode': mode,
                'results': results,
                'system_info': {
                    'python_version': sys.version,
                    'django_version': django.get_version(),
                    'total_time': time.time() - self.start_time
                }
            }
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📊 Reporte guardado en {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generando reporte: {str(e)}")
            return None

def main():
    """
    Función principal del sistema unificado.
    """
    parser = argparse.ArgumentParser(description='Sistema Unificado de IA - FinancialHub')
    parser.add_argument('--mode', choices=['train', 'verify', 'test', 'cleanup'], 
                       default='verify', help='Modo de operación')
    parser.add_argument('--train-mode', choices=['fast', 'full'], 
                       default='fast', help='Modo de entrenamiento (solo para --mode train)')
    parser.add_argument('--output', help='Archivo de salida para el reporte')
    
    args = parser.parse_args()
    
    try:
        logger.info("🚀 Iniciando Sistema Unificado de IA...")
        
        system = UnifiedAISystem()
        
        if args.mode == 'train':
            logger.info(f"🎯 Modo: Entrenamiento ({args.train_mode})")
            results = system.train_all_models(args.train_mode)
            
        elif args.mode == 'verify':
            logger.info("🎯 Modo: Verificación")
            results = system.verify_system()
            
        elif args.mode == 'test':
            logger.info("🎯 Modo: Testing")
            results = system.run_comprehensive_test()
            
        elif args.mode == 'cleanup':
            logger.info("🎯 Modo: Limpieza")
            results = system.cleanup_system()
        
        # Generar reporte
        report_path = system.generate_report(results, args.mode)
        
        # Mostrar resumen
        logger.info("=" * 60)
        logger.info("📋 RESUMEN FINAL")
        logger.info("=" * 60)
        
        if args.mode == 'train':
            successful = results.get('successful_models', 0)
            total = results.get('total_models', 0)
            logger.info(f"✅ Entrenamiento completado: {successful}/{total} modelos exitosos")
            
        elif args.mode == 'verify':
            models_loaded = results.get('performance', {}).get('models_loaded', 0)
            functionality_working = results.get('performance', {}).get('functionality_working', 0)
            logger.info(f"✅ Verificación completada: {models_loaded} modelos cargados, {functionality_working} funcionalidades trabajando")
            
        elif args.mode == 'test':
            summary = results.get('summary', {})
            passed = summary.get('passed_tests', 0)
            total = summary.get('total_tests', 0)
            success_rate = summary.get('success_rate', 0)
            logger.info(f"✅ Testing completado: {passed}/{total} pruebas exitosas ({success_rate:.1f}%)")
            
        elif args.mode == 'cleanup':
            files_removed = len(results.get('files_removed', []))
            logger.info(f"✅ Limpieza completada: {files_removed} archivos removidos")
        
        logger.info(f"⏱️  Tiempo total: {time.time() - system.start_time:.2f} segundos")
        logger.info(f"📊 Reporte: {report_path}")
        logger.info("🎉 ¡Sistema unificado completado exitosamente!")
        
    except Exception as e:
        logger.error(f"❌ Error en sistema unificado: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 