#!/usr/bin/env python3
"""
Script para verificar el estado completo de todos los modelos de IA en FinancialHub.

Este script verifica:
1. Modelos principales (transaction_classifier, expense_predictor, etc.)
2. Modelos de NLP (sentiment_analyzer, topic_model, etc.)
3. Modelos de transformers
4. Modelos especializados (fraud_detector, credit_scorer, etc.)
"""

import os
import sys
import django
import logging
import json
from datetime import datetime
from typing import Dict, List, Any

# Configurar Django
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'financialhub.settings.dev')
django.setup()

from ai.services import AIService
from ai.ml.sentiment_analyzer import SentimentAnalyzer
from ai.ml.nlp.text_processor import FinancialTextProcessor
from ai.ml.transformers.financial_transformer import FinancialTransformerService
from django.conf import settings

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIModelVerifier:
    """
    Verificador completo de modelos de IA.
    """
    
    def __init__(self):
        self.ai_service = AIService()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.text_processor = FinancialTextProcessor()
        self.transformer_service = FinancialTransformerService()
        self.verification_results = {}
        
    def check_model_files(self) -> Dict[str, Any]:
        """
        Verifica que los archivos de modelos existan.
        """
        try:
            logger.info("🔍 Verificando archivos de modelos...")
            
            models_dir = 'ml_models'
            expected_models = {
                'transaction_classifier': 'transaction_classifier.joblib',
                'expense_predictor': 'expense_predictor.joblib',
                'behavior_analyzer': 'behavior_analyzer.joblib',
                'budget_optimizer': 'budget_optimizer.joblib',
                'sentiment_analyzer': 'sentiment_analyzer.joblib',
                'credit_scorer': 'credit_scorer.joblib',
                'fraud_detector': 'fraud_detector.joblib',
                'market_predictor': 'market_predictor.joblib',
                'nlp_sentiment_model': 'nlp_sentiment_model.joblib',
                'nlp_topic_model': 'nlp_topic_model.joblib',
                'nlp_classifier_model': 'nlp_classifier_model.joblib'
            }
            
            file_status = {}
            for model_name, filename in expected_models.items():
                filepath = os.path.join(models_dir, filename)
                if os.path.exists(filepath):
                    file_size = os.path.getsize(filepath)
                    file_status[model_name] = {
                        'status': 'exists',
                        'filepath': filepath,
                        'size_bytes': file_size,
                        'size_mb': round(file_size / (1024 * 1024), 2)
                    }
                else:
                    file_status[model_name] = {
                        'status': 'missing',
                        'filepath': filepath
                    }
            
            # Verificar modelos transformer
            transformer_models = {
                'transformer_model': 'transformer_model.pth',
                'tokenizer': 'tokenizer.joblib',
                'transformer_config': 'transformer_config.json'
            }
            
            for model_name, filename in transformer_models.items():
                filepath = os.path.join(models_dir, filename)
                if os.path.exists(filepath):
                    file_size = os.path.getsize(filepath)
                    file_status[f'transformer_{model_name}'] = {
                        'status': 'exists',
                        'filepath': filepath,
                        'size_bytes': file_size,
                        'size_mb': round(file_size / (1024 * 1024), 2)
                    }
                else:
                    file_status[f'transformer_{model_name}'] = {
                        'status': 'missing',
                        'filepath': filepath
                    }
            
            return file_status
            
        except Exception as e:
            logger.error(f"Error verificando archivos: {str(e)}")
            return {'error': str(e)}
    
    def test_model_functionality(self) -> Dict[str, Any]:
        """
        Prueba la funcionalidad de los modelos.
        """
        try:
            logger.info("🧪 Probando funcionalidad de modelos...")
            
            test_results = {}
            
            # Test sentiment analyzer
            try:
                sentiment_result = self.sentiment_analyzer.analyze_text_sentiment(
                    "This is a great transaction", method='ml'
                )
                test_results['sentiment_analyzer'] = {
                    'status': 'working',
                    'test_result': sentiment_result
                }
            except Exception as e:
                test_results['sentiment_analyzer'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            # Test AI service
            try:
                ai_result = self.ai_service.predict_category("groceries purchase")
                test_results['ai_service'] = {
                    'status': 'working',
                    'test_result': ai_result
                }
            except Exception as e:
                test_results['ai_service'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            # Test text processor
            try:
                text_result = self.text_processor.preprocess_text("Test transaction")
                test_results['text_processor'] = {
                    'status': 'working',
                    'test_result': {'processed_text': text_result}
                }
            except Exception as e:
                test_results['text_processor'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            return test_results
            
        except Exception as e:
            logger.error(f"Error probando funcionalidad: {str(e)}")
            return {'error': str(e)}
    
    def get_model_metrics(self) -> Dict[str, Any]:
        """
        Obtiene métricas de los modelos.
        """
        try:
            logger.info("📊 Obteniendo métricas de modelos...")
            
            metrics = {}
            
            # Métricas del AI service
            try:
                ai_info = self.ai_service.get_model_info()
                metrics['ai_service'] = ai_info
            except Exception as e:
                metrics['ai_service'] = {'error': str(e)}
            
            # Métricas del sentiment analyzer
            try:
                sentiment_info = self.sentiment_analyzer.get_model_info()
                metrics['sentiment_analyzer'] = sentiment_info
            except Exception as e:
                metrics['sentiment_analyzer'] = {'error': str(e)}
            
            # Métricas del transformer service
            try:
                transformer_info = self.transformer_service.get_model_info()
                metrics['transformer_service'] = transformer_info
            except Exception as e:
                metrics['transformer_service'] = {'error': str(e)}
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error obteniendo métricas: {str(e)}")
            return {'error': str(e)}
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Genera un reporte completo del estado de todos los modelos.
        """
        try:
            logger.info("📋 Generando reporte completo...")
            
            # Verificar archivos
            file_status = self.check_model_files()
            
            # Probar funcionalidad
            functionality_tests = self.test_model_functionality()
            
            # Obtener métricas
            model_metrics = self.get_model_metrics()
            
            # Contar modelos
            total_models = len(file_status)
            existing_models = len([m for m in file_status.values() if m.get('status') == 'exists'])
            missing_models = total_models - existing_models
            
            working_models = len([m for m in functionality_tests.values() if m.get('status') == 'working'])
            error_models = len([m for m in functionality_tests.values() if m.get('status') == 'error'])
            
            # Generar reporte
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_models': total_models,
                    'existing_models': existing_models,
                    'missing_models': missing_models,
                    'working_models': working_models,
                    'error_models': error_models,
                    'success_rate': round((working_models / total_models) * 100, 2) if total_models > 0 else 0
                },
                'file_status': file_status,
                'functionality_tests': functionality_tests,
                'model_metrics': model_metrics,
                'recommendations': self._generate_recommendations(file_status, functionality_tests)
            }
            
            # Guardar reporte
            report_path = 'logs/ai/ai_models_comprehensive_report.json'
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📊 Reporte completo guardado en {report_path}")
            return report
            
        except Exception as e:
            logger.error(f"Error generando reporte: {str(e)}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, file_status: Dict, functionality_tests: Dict) -> List[str]:
        """
        Genera recomendaciones basadas en el estado de los modelos.
        """
        recommendations = []
        
        # Verificar modelos faltantes
        missing_models = [name for name, status in file_status.items() if status.get('status') == 'missing']
        if missing_models:
            recommendations.append(f"Entrenar modelos faltantes: {', '.join(missing_models)}")
        
        # Verificar modelos con errores
        error_models = [name for name, status in functionality_tests.items() if status.get('status') == 'error']
        if error_models:
            recommendations.append(f"Arreglar modelos con errores: {', '.join(error_models)}")
        
        # Verificar modelos transformer
        transformer_models = [name for name in file_status.keys() if 'transformer' in name]
        missing_transformers = [name for name in transformer_models if file_status[name].get('status') == 'missing']
        if missing_transformers:
            recommendations.append("Entrenar modelos transformer faltantes")
        
        # Verificar modelos de NLP
        nlp_models = ['sentiment_analyzer', 'nlp_sentiment_model', 'nlp_topic_model', 'nlp_classifier_model']
        missing_nlp = [name for name in nlp_models if name in file_status and file_status[name].get('status') == 'missing']
        if missing_nlp:
            recommendations.append("Completar entrenamiento de modelos NLP")
        
        if not recommendations:
            recommendations.append("✅ Todos los modelos están funcionando correctamente")
        
        return recommendations
    
    def print_summary(self, report: Dict[str, Any]):
        """
        Imprime un resumen del reporte.
        """
        try:
            summary = report.get('summary', {})
            
            print("\n" + "="*60)
            print("🤖 REPORTE COMPLETO DE MODELOS DE IA")
            print("="*60)
            
            print(f"📊 Total de modelos: {summary.get('total_models', 0)}")
            print(f"✅ Modelos existentes: {summary.get('existing_models', 0)}")
            print(f"❌ Modelos faltantes: {summary.get('missing_models', 0)}")
            print(f"🟢 Modelos funcionando: {summary.get('working_models', 0)}")
            print(f"🔴 Modelos con errores: {summary.get('error_models', 0)}")
            print(f"📈 Tasa de éxito: {summary.get('success_rate', 0)}%")
            
            print("\n" + "-"*60)
            print("📋 ESTADO DE ARCHIVOS DE MODELOS")
            print("-"*60)
            
            file_status = report.get('file_status', {})
            for model_name, status in file_status.items():
                if status.get('status') == 'exists':
                    size_mb = status.get('size_mb', 0)
                    print(f"✅ {model_name}: {size_mb} MB")
                else:
                    print(f"❌ {model_name}: FALTANTE")
            
            print("\n" + "-"*60)
            print("🧪 PRUEBAS DE FUNCIONALIDAD")
            print("-"*60)
            
            functionality_tests = report.get('functionality_tests', {})
            for model_name, status in functionality_tests.items():
                if status.get('status') == 'working':
                    print(f"✅ {model_name}: FUNCIONANDO")
                else:
                    print(f"❌ {model_name}: ERROR - {status.get('error', 'Unknown')}")
            
            print("\n" + "-"*60)
            print("💡 RECOMENDACIONES")
            print("-"*60)
            
            recommendations = report.get('recommendations', [])
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
            
            print("\n" + "="*60)
            
        except Exception as e:
            logger.error(f"Error imprimiendo resumen: {str(e)}")

def main():
    """
    Función principal.
    """
    try:
        logger.info("🎯 Iniciando verificación completa de modelos de IA...")
        
        verifier = AIModelVerifier()
        report = verifier.generate_comprehensive_report()
        
        if 'error' in report:
            logger.error(f"❌ Error en verificación: {report['error']}")
            sys.exit(1)
        else:
            logger.info("🎉 Verificación completada exitosamente!")
            verifier.print_summary(report)
        
    except Exception as e:
        logger.error(f"Error en main: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 