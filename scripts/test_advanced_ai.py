#!/usr/bin/env python3
"""
Script de prueba para verificar el funcionamiento de todos los sistemas avanzados de AI.

Este script prueba:
- AutoML y optimización automática
- Federated Learning
- A/B Testing
- NLP avanzado
- Transformers personalizados
- Monitoreo de recursos
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Agregar el directorio backend al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_automl():
    """Prueba el sistema de AutoML"""
    logger.info("🧪 Probando sistema de AutoML...")
    
    try:
        from ai.ml.automl.auto_ml_optimizer import AutoMLOptimizer
        
        # Crear datos de prueba
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randn(1000),
            'feature4': np.random.randn(1000)
        })
        y = pd.Series(X['feature1'] * 2 + X['feature2'] * 1.5 + np.random.randn(1000) * 0.1)
        
        # Inicializar AutoML
        automl = AutoMLOptimizer(task_type='regression')
        
        # Optimizar modelo
        results = automl.optimize(X, y, cv=3)
        
        logger.info(f"✅ AutoML completado. Mejor score: {results['best_score']:.4f}")
        logger.info(f"   Mejor modelo: {results['best_model_name']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en AutoML: {str(e)}")
        return False

def test_federated_learning():
    """Prueba el sistema de Federated Learning"""
    logger.info("🧪 Probando sistema de Federated Learning...")
    
    try:
        from ai.ml.federated.federated_learning import FederatedLearning, AggregationMethod
        
        # Inicializar federated learning
        fl = FederatedLearning(
            task_type='classification',
            aggregation_method=AggregationMethod.FEDAVG,
            min_clients=2
        )
        
        # Agregar clientes
        fl.add_client('client1', 1000)
        fl.add_client('client2', 1500)
        
        # Crear datos de prueba
        np.random.seed(42)
        X1 = np.random.randn(1000, 4)
        y1 = (X1[:, 0] + X1[:, 1] > 0).astype(int)
        
        X2 = np.random.randn(1500, 4)
        y2 = (X2[:, 0] + X2[:, 1] > 0).astype(int)
        
        # Entrenar clientes
        results1 = fl.train_client('client1', X1, y1)
        results2 = fl.train_client('client2', X2, y2)
        
        logger.info(f"✅ Federated Learning completado")
        logger.info(f"   Cliente 1 - Accuracy: {results1['accuracy']:.4f}")
        logger.info(f"   Cliente 2 - Accuracy: {results2['accuracy']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en Federated Learning: {str(e)}")
        return False

def test_ab_testing():
    """Prueba el sistema de A/B Testing"""
    logger.info("🧪 Probando sistema de A/B Testing...")
    
    try:
        from ai.ml.experimentation.ab_testing import ABTesting, ExperimentConfig, MetricType
        from datetime import datetime, timedelta
        
        # Inicializar A/B Testing
        ab = ABTesting()
        
        # Crear experimento
        config = ExperimentConfig(
            experiment_id='test_exp_001',
            name='Test Experiment',
            description='Experimento de prueba',
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=7),
            traffic_split={'A': 0.5, 'B': 0.5},
            primary_metric='conversion_rate',
            secondary_metrics=['revenue', 'engagement'],
            sample_size=1000
        )
        
        experiment_id = ab.create_experiment(config)
        
        # Asignar usuarios
        for i in range(100):
            user_id = f"user_{i}"
            variant = ab.assign_user_to_variant(experiment_id, user_id)
            
            # Registrar métricas
            if variant == 'A':
                ab.record_metric(experiment_id, user_id, 'conversion_rate', 0.15, MetricType.BINARY)
                ab.record_metric(experiment_id, user_id, 'revenue', 25.50, MetricType.CONTINUOUS)
            else:
                ab.record_metric(experiment_id, user_id, 'conversion_rate', 0.18, MetricType.BINARY)
                ab.record_metric(experiment_id, user_id, 'revenue', 28.75, MetricType.CONTINUOUS)
        
        # Analizar resultados
        analysis = ab.analyze_experiment(experiment_id, 'conversion_rate')
        
        logger.info(f"✅ A/B Testing completado")
        logger.info(f"   Experiment ID: {experiment_id}")
        logger.info(f"   Significancia: {analysis['overall_significance']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en A/B Testing: {str(e)}")
        return False

def test_nlp():
    """Prueba el sistema de NLP"""
    logger.info("🧪 Probando sistema de NLP...")
    
    try:
        from ai.ml.nlp.text_processor import FinancialTextProcessor
        
        # Inicializar NLP
        nlp = FinancialTextProcessor()
        
        # Textos de prueba
        texts = [
            "Our quarterly revenue increased by 15% to $2.5 million, exceeding expectations.",
            "The company reported a loss of $500,000 due to market volatility.",
            "Investors are bullish on our new product line with strong growth potential."
        ]
        
        results = []
        for text in texts:
            # Análisis de sentimiento
            sentiment = nlp.analyze_sentiment(text, method='financial')
            
            # Extracción de entidades
            entities = nlp.extract_financial_entities(text)
            
            # Generar resumen
            summary = nlp.generate_summary(text, max_sentences=2)
            
            results.append({
                'text': text,
                'sentiment': sentiment,
                'entities': entities,
                'summary': summary
            })
        
        logger.info(f"✅ NLP completado")
        logger.info(f"   Textos procesados: {len(texts)}")
        logger.info(f"   Sentimiento promedio: {np.mean([r['sentiment']['compound'] for r in results]):.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en NLP: {str(e)}")
        return False

def test_transformers():
    """Prueba el sistema de Transformers"""
    logger.info("🧪 Probando sistema de Transformers...")
    
    try:
        from ai.ml.transformers.financial_transformer import FinancialTransformerService, TransformerConfig
        
        # Configurar transformer
        config = TransformerConfig(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            max_seq_length=64,
            epochs=2  # Solo 2 épocas para prueba rápida
        )
        
        # Inicializar servicio
        transformer_service = FinancialTransformerService(config)
        
        # Datos de prueba
        texts = [
            "Revenue increased by 20% this quarter",
            "Company reported losses due to market conditions",
            "Strong performance in all business segments",
            "Financial results exceeded expectations"
        ]
        
        # Etiquetas de sentimiento (0=negativo, 1=positivo)
        labels = [1.0, 0.0, 1.0, 1.0]
        
        # Entrenar modelo
        transformer_service.train_model(texts, labels)
        
        # Realizar predicciones
        predictions = transformer_service.predict(texts)
        
        # Analizar sentimiento
        sentiment_results = transformer_service.analyze_sentiment(texts)
        
        logger.info(f"✅ Transformers completado")
        logger.info(f"   Modelo entrenado exitosamente")
        logger.info(f"   Predicciones realizadas: {len(predictions)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en Transformers: {str(e)}")
        return False

def test_monitoring():
    """Prueba el sistema de monitoreo"""
    logger.info("🧪 Probando sistema de monitoreo...")
    
    try:
        from ai.ml.utils.monitoring.resource_monitor import ResourceMonitor
        
        # Inicializar monitor
        monitor = ResourceMonitor()
        
        # Recolectar métricas
        metrics = monitor.collect_metrics()
        
        # Verificar métricas
        assert 'cpu' in metrics
        assert 'memory' in metrics
        assert 'disk' in metrics
        
        logger.info(f"✅ Monitoreo completado")
        logger.info(f"   CPU: {metrics['cpu']['percent']:.1f}%")
        logger.info(f"   Memoria: {metrics['memory']['percent']:.1f}%")
        logger.info(f"   Disco: {metrics['disk']['percent']:.1f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en monitoreo: {str(e)}")
        return False

def test_ai_service():
    """Prueba el servicio principal de AI"""
    logger.info("🧪 Probando servicio principal de AI...")
    
    try:
        from ai.services import AIService
        
        # Inicializar servicio
        ai_service = AIService()
        
        # Obtener capacidades
        capabilities = ai_service.get_advanced_ai_capabilities()
        
        # Verificar que todos los sistemas estén disponibles
        systems = ['automl', 'federated_learning', 'ab_testing', 'nlp', 'transformers', 'monitoring']
        
        for system in systems:
            if system in capabilities and capabilities[system]['available']:
                logger.info(f"   ✅ {system}: Disponible")
            else:
                logger.warning(f"   ⚠️ {system}: No disponible")
        
        logger.info(f"✅ Servicio de AI inicializado correctamente")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en servicio de AI: {str(e)}")
        return False

def main():
    """Función principal"""
    logger.info("🚀 Iniciando pruebas de sistemas avanzados de AI...")
    
    # Lista de pruebas
    tests = [
        ("AutoML", test_automl),
        ("Federated Learning", test_federated_learning),
        ("A/B Testing", test_ab_testing),
        ("NLP", test_nlp),
        ("Transformers", test_transformers),
        ("Monitoreo", test_monitoring),
        ("Servicio de AI", test_ai_service)
    ]
    
    # Ejecutar pruebas
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Ejecutando prueba: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            logger.error(f"Error inesperado en {test_name}: {str(e)}")
            results[test_name] = False
    
    # Resumen de resultados
    logger.info(f"\n{'='*60}")
    logger.info("📊 RESUMEN DE PRUEBAS")
    logger.info(f"{'='*60}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASÓ" if success else "❌ FALLÓ"
        logger.info(f"{test_name:.<30} {status}")
    
    logger.info(f"\nResultado final: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        logger.info("🎉 ¡Todas las pruebas pasaron! El sistema de AI está funcionando correctamente.")
        return 0
    else:
        logger.warning(f"⚠️ {total - passed} pruebas fallaron. Revisa los errores arriba.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 