#!/usr/bin/env python3
"""
Script de verificación y diagnóstico del sistema de IA.
Este script verifica el estado de todos los modelos, archivos y configuraciones.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Agregar el directorio backend al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Configurar Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'financialhub.settings.dev')
import django
django.setup()

from ai.services import AIService
from transactions.models import Transaction
from django.db import connection
import joblib

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AISystemVerifier:
    """Verificador completo del sistema de IA."""
    
    def __init__(self):
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'unknown',
            'checks': {},
            'recommendations': []
        }
    
    def verify_model_files(self):
        """Verifica la existencia de archivos de modelos."""
        logger.info("Verificando archivos de modelos...")
        
        models_dir = backend_dir / 'ml_models'
        required_models = [
            'transaction_classifier.joblib',
            'expense_predictor.joblib', 
            'behavior_analyzer.joblib',
            'budget_optimizer.joblib'
        ]
        
        optional_models = [
            'credit_scorer.joblib',
            'market_predictor.joblib',
            'fraud_detector.joblib',
            'sentiment_analyzer.joblib',
            'nlp_sentiment_model.joblib',
            'nlp_topic_model.joblib',
            'nlp_classifier_model.joblib',
            'transformer_model.pth',
            'tokenizer.joblib'
        ]
        
        check_result = {
            'status': 'pass',
            'required_models': {},
            'optional_models': {},
            'missing_required': [],
            'missing_optional': []
        }
        
        # Verificar modelos requeridos
        for model in required_models:
            model_path = models_dir / model
            if model_path.exists():
                size = model_path.stat().st_size
                check_result['required_models'][model] = {
                    'exists': True,
                    'size_bytes': size,
                    'size_mb': round(size / (1024 * 1024), 2)
                }
            else:
                check_result['required_models'][model] = {'exists': False}
                check_result['missing_required'].append(model)
                check_result['status'] = 'fail'
        
        # Verificar modelos opcionales
        for model in optional_models:
            model_path = models_dir / model
            if model_path.exists():
                size = model_path.stat().st_size
                check_result['optional_models'][model] = {
                    'exists': True,
                    'size_bytes': size,
                    'size_mb': round(size / (1024 * 1024), 2)
                }
            else:
                check_result['optional_models'][model] = {'exists': False}
                check_result['missing_optional'].append(model)
        
        self.report['checks']['model_files'] = check_result
        logger.info(f"Verificación de archivos completada. Estado: {check_result['status']}")
        
        return check_result
    
    def verify_model_loading(self):
        """Verifica que los modelos se puedan cargar correctamente."""
        logger.info("Verificando carga de modelos...")
        
        check_result = {
            'status': 'pass',
            'models': {},
            'errors': []
        }
        
        try:
            ai_service = AIService()
            
            # Verificar modelos principales
            models_to_check = [
                ('transaction_classifier', ai_service.transaction_classifier),
                ('expense_predictor', ai_service.expense_predictor),
                ('behavior_analyzer', ai_service.behavior_analyzer),
                ('budget_optimizer', ai_service.budget_optimizer)
            ]
            
            for name, model in models_to_check:
                try:
                    if model is not None:
                        # Intentar hacer una predicción simple
                        if hasattr(model, 'predict'):
                            check_result['models'][name] = {
                                'loaded': True,
                                'predictable': True,
                                'type': type(model).__name__
                            }
                        else:
                            check_result['models'][name] = {
                                'loaded': True,
                                'predictable': False,
                                'type': type(model).__name__
                            }
                    else:
                        check_result['models'][name] = {'loaded': False}
                        check_result['errors'].append(f"Model {name} is None")
                        check_result['status'] = 'fail'
                except Exception as e:
                    check_result['models'][name] = {'loaded': False, 'error': str(e)}
                    check_result['errors'].append(f"Error loading {name}: {str(e)}")
                    check_result['status'] = 'fail'
            
            # Verificar modelos opcionales
            optional_models = [
                ('nlp_processor', ai_service.nlp_processor),
                ('transformer_service', ai_service.transformer_service),
                ('automl_optimizer', ai_service.automl_optimizer)
            ]
            
            for name, model in optional_models:
                if model is not None:
                    check_result['models'][name] = {
                        'loaded': True,
                        'type': type(model).__name__
                    }
                else:
                    check_result['models'][name] = {'loaded': False}
            
        except Exception as e:
            check_result['status'] = 'fail'
            check_result['errors'].append(f"Error initializing AI service: {str(e)}")
        
        self.report['checks']['model_loading'] = check_result
        logger.info(f"Verificación de carga completada. Estado: {check_result['status']}")
        
        return check_result
    
    def verify_training_data(self):
        """Verifica la disponibilidad de datos de entrenamiento."""
        logger.info("Verificando datos de entrenamiento...")
        
        check_result = {
            'status': 'pass',
            'transaction_count': 0,
            'categorized_transactions': 0,
            'categories': {},
            'data_quality': {}
        }
        
        try:
            # Contar transacciones totales
            total_transactions = Transaction.objects.count()
            check_result['transaction_count'] = total_transactions
            
            # Contar transacciones categorizadas
            categorized_transactions = Transaction.objects.filter(category__isnull=False).count()
            check_result['categorized_transactions'] = categorized_transactions
            
            # Verificar distribución de categorías
            from django.db.models import Count
            category_distribution = Transaction.objects.filter(
                category__isnull=False
            ).values('category__name').annotate(
                count=Count('id')
            ).order_by('-count')
            
            check_result['categories'] = {
                item['category__name']: item['count'] 
                for item in category_distribution[:10]  # Top 10 categorías
            }
            
            # Evaluar calidad de datos
            if total_transactions == 0:
                check_result['status'] = 'fail'
                check_result['data_quality']['error'] = 'No transactions found'
            elif categorized_transactions == 0:
                check_result['status'] = 'fail'
                check_result['data_quality']['error'] = 'No categorized transactions found'
            elif categorized_transactions < 10:
                check_result['status'] = 'warning'
                check_result['data_quality']['warning'] = 'Very few categorized transactions'
            else:
                categorization_rate = categorized_transactions / total_transactions
                check_result['data_quality']['categorization_rate'] = round(categorization_rate, 3)
                
                if categorization_rate < 0.5:
                    check_result['status'] = 'warning'
                    check_result['data_quality']['warning'] = 'Low categorization rate'
        
        except Exception as e:
            check_result['status'] = 'fail'
            check_result['data_quality']['error'] = str(e)
        
        self.report['checks']['training_data'] = check_result
        logger.info(f"Verificación de datos completada. Estado: {check_result['status']}")
        
        return check_result
    
    def verify_model_training(self):
        """Verifica que los modelos se puedan entrenar."""
        logger.info("Verificando capacidad de entrenamiento...")
        
        check_result = {
            'status': 'pass',
            'models': {},
            'errors': []
        }
        
        try:
            # Obtener datos de entrenamiento
            transactions = Transaction.objects.filter(
                category__isnull=False
            ).select_related('category')[:100]  # Limitar a 100 para prueba
            
            if len(transactions) < 10:
                check_result['status'] = 'fail'
                check_result['errors'].append('Insufficient training data')
                self.report['checks']['model_training'] = check_result
                return check_result
            
            # Convertir a formato de diccionario
            transaction_data = []
            for t in transactions:
                transaction_data.append({
                    'id': t.id,
                    'amount': float(t.amount),
                    'type': t.type,
                    'description': t.description or '',
                    'category_id': t.category.id,
                    'category_name': t.category.name,
                    'date': t.date,
                    'merchant': t.merchant or '',
                    'payment_method': getattr(t, 'payment_method', '') or '',
                    'location': getattr(t, 'location', '') or '',
                    'notes': t.notes or ''
                })
            
            # Probar entrenamiento de cada modelo
            ai_service = AIService()
            
            # Transaction Classifier
            try:
                if ai_service.transaction_classifier:
                    ai_service.transaction_classifier.train(transaction_data)
                    check_result['models']['transaction_classifier'] = {'trainable': True}
                else:
                    check_result['models']['transaction_classifier'] = {'trainable': False}
                    check_result['errors'].append('Transaction classifier not available')
            except Exception as e:
                check_result['models']['transaction_classifier'] = {'trainable': False, 'error': str(e)}
                check_result['errors'].append(f'Error training transaction classifier: {str(e)}')
            
            # Expense Predictor
            try:
                if ai_service.expense_predictor:
                    ai_service.expense_predictor.train(transaction_data)
                    check_result['models']['expense_predictor'] = {'trainable': True}
                else:
                    check_result['models']['expense_predictor'] = {'trainable': False}
                    check_result['errors'].append('Expense predictor not available')
            except Exception as e:
                check_result['models']['expense_predictor'] = {'trainable': False, 'error': str(e)}
                check_result['errors'].append(f'Error training expense predictor: {str(e)}')
            
            # Behavior Analyzer
            try:
                if ai_service.behavior_analyzer:
                    patterns = ai_service.behavior_analyzer.analyze_spending_patterns(transaction_data)
                    check_result['models']['behavior_analyzer'] = {'trainable': True}
                else:
                    check_result['models']['behavior_analyzer'] = {'trainable': False}
                    check_result['errors'].append('Behavior analyzer not available')
            except Exception as e:
                check_result['models']['behavior_analyzer'] = {'trainable': False, 'error': str(e)}
                check_result['errors'].append(f'Error training behavior analyzer: {str(e)}')
            
            if check_result['errors']:
                check_result['status'] = 'fail'
        
        except Exception as e:
            check_result['status'] = 'fail'
            check_result['errors'].append(f'Error in training verification: {str(e)}')
        
        self.report['checks']['model_training'] = check_result
        logger.info(f"Verificación de entrenamiento completada. Estado: {check_result['status']}")
        
        return check_result
    
    def verify_system_health(self):
        """Verifica la salud general del sistema."""
        logger.info("Verificando salud del sistema...")
        
        check_result = {
            'status': 'pass',
            'database': {},
            'memory': {},
            'disk': {},
            'errors': []
        }
        
        try:
            # Verificar conexión a base de datos
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                check_result['database']['connected'] = True
        except Exception as e:
            check_result['database']['connected'] = False
            check_result['errors'].append(f'Database connection failed: {str(e)}')
            check_result['status'] = 'fail'
        
        # Verificar espacio en disco
        try:
            import shutil
            total, used, free = shutil.disk_usage(backend_dir)
            check_result['disk'] = {
                'total_gb': round(total / (1024**3), 2),
                'used_gb': round(used / (1024**3), 2),
                'free_gb': round(free / (1024**3), 2),
                'usage_percent': round((used / total) * 100, 2)
            }
            
            if check_result['disk']['usage_percent'] > 90:
                check_result['status'] = 'warning'
                check_result['errors'].append('Disk usage is high')
        except Exception as e:
            check_result['disk']['error'] = str(e)
        
        # Verificar memoria
        try:
            import psutil
            memory = psutil.virtual_memory()
            check_result['memory'] = {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2),
                'usage_percent': memory.percent
            }
            
            if check_result['memory']['usage_percent'] > 80:
                check_result['status'] = 'warning'
                check_result['errors'].append('Memory usage is high')
        except Exception as e:
            check_result['memory']['error'] = str(e)
        
        self.report['checks']['system_health'] = check_result
        logger.info(f"Verificación de salud completada. Estado: {check_result['status']}")
        
        return check_result
    
    def generate_recommendations(self):
        """Genera recomendaciones basadas en los resultados de verificación."""
        logger.info("Generando recomendaciones...")
        
        recommendations = []
        
        # Verificar archivos de modelos
        model_files = self.report['checks'].get('model_files', {})
        if model_files.get('status') == 'fail':
            missing = model_files.get('missing_required', [])
            if missing:
                recommendations.append(f"Missing required model files: {', '.join(missing)}")
                recommendations.append("Run model training to generate missing models")
        
        # Verificar datos de entrenamiento
        training_data = self.report['checks'].get('training_data', {})
        if training_data.get('status') == 'fail':
            recommendations.append("No training data available. Add categorized transactions to the database")
        elif training_data.get('status') == 'warning':
            recommendations.append("Low amount of training data. Add more categorized transactions")
        
        # Verificar entrenamiento
        model_training = self.report['checks'].get('model_training', {})
        if model_training.get('status') == 'fail':
            recommendations.append("Model training failed. Check logs for specific errors")
        
        # Verificar salud del sistema
        system_health = self.report['checks'].get('system_health', {})
        if system_health.get('status') == 'warning':
            recommendations.append("System resources are running low. Consider cleanup or upgrade")
        
        # Recomendaciones generales
        if not recommendations:
            recommendations.append("System appears to be healthy. Monitor performance regularly")
        
        self.report['recommendations'] = recommendations
        return recommendations
    
    def determine_overall_status(self):
        """Determina el estado general del sistema."""
        statuses = []
        
        for check_name, check_result in self.report['checks'].items():
            status = check_result.get('status', 'unknown')
            statuses.append(status)
        
        if 'fail' in statuses:
            self.report['overall_status'] = 'critical'
        elif 'warning' in statuses:
            self.report['overall_status'] = 'warning'
        else:
            self.report['overall_status'] = 'healthy'
    
    def run_full_verification(self):
        """Ejecuta todas las verificaciones."""
        logger.info("Iniciando verificación completa del sistema de IA...")
        
        # Ejecutar todas las verificaciones
        self.verify_model_files()
        self.verify_model_loading()
        self.verify_training_data()
        self.verify_model_training()
        self.verify_system_health()
        
        # Generar recomendaciones
        self.generate_recommendations()
        
        # Determinar estado general
        self.determine_overall_status()
        
        logger.info(f"Verificación completada. Estado general: {self.report['overall_status']}")
        
        return self.report
    
    def save_report(self, filename=None):
        """Guarda el reporte en un archivo JSON."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'ai_system_verify_report_{timestamp}.json'
        
        report_path = backend_dir / 'logs' / 'ai' / filename
        
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(self.report, f, indent=2, default=str)
            
            logger.info(f"Reporte guardado en: {report_path}")
            return str(report_path)
        except Exception as e:
            logger.error(f"Error guardando reporte: {str(e)}")
            return None
    
    def print_summary(self):
        """Imprime un resumen del reporte."""
        print("\n" + "="*60)
        print("VERIFICACIÓN DEL SISTEMA DE IA")
        print("="*60)
        print(f"Timestamp: {self.report['timestamp']}")
        print(f"Estado General: {self.report['overall_status'].upper()}")
        print()
        
        for check_name, check_result in self.report['checks'].items():
            status = check_result.get('status', 'unknown')
            status_symbol = {
                'pass': '✅',
                'warning': '⚠️',
                'fail': '❌',
                'unknown': '❓'
            }.get(status, '❓')
            
            print(f"{status_symbol} {check_name.replace('_', ' ').title()}: {status}")
        
        print("\n" + "-"*60)
        print("RECOMENDACIONES:")
        print("-"*60)
        
        if self.report['recommendations']:
            for i, rec in enumerate(self.report['recommendations'], 1):
                print(f"{i}. {rec}")
        else:
            print("No hay recomendaciones específicas.")
        
        print("\n" + "="*60)

def main():
    """Función principal."""
    verifier = AISystemVerifier()
    
    try:
        # Ejecutar verificación completa
        report = verifier.run_full_verification()
        
        # Imprimir resumen
        verifier.print_summary()
        
        # Guardar reporte
        report_path = verifier.save_report()
        if report_path:
            print(f"\nReporte detallado guardado en: {report_path}")
        
        # Retornar código de salida basado en el estado
        if report['overall_status'] == 'critical':
            sys.exit(1)
        elif report['overall_status'] == 'warning':
            sys.exit(2)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Error en la verificación: {str(e)}")
        print(f"❌ Error en la verificación: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 