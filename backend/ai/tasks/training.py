"""
Tareas de entrenamiento de modelos ML.

Este módulo proporciona tareas Celery para:
- Entrenamiento periódico de modelos
- Evaluación de rendimiento
- Actualización de versiones
"""
from celery import shared_task
from django.utils import timezone
from datetime import timedelta
from ..ml.utils.metrics import ModelMetrics
from ..ml.utils.versioning.model_versioning import ModelVersioning
from ..ml.utils.monitoring.resource_monitor import ResourceMonitor
from ..services import AIService
from transactions.models import Transaction
import logging
from typing import Dict, Any

logger = logging.getLogger('ai.tasks')

@shared_task
def train_models():
    """
    Tarea periódica para entrenar modelos.
    
    Returns:
        Dict: Resultados del entrenamiento
    """
    try:
        # Inicializar servicios
        ai_service = AIService()
        metrics = ModelMetrics()
        versioning = ModelVersioning()
        monitor = ResourceMonitor()
        
        # Verificar recursos antes de entrenar
        system_metrics = monitor.collect_metrics()
        if system_metrics['cpu']['percent'] > 80 or system_metrics['memory']['percent'] > 80:
            logger.warning("Recursos del sistema altos, posponiendo entrenamiento")
            return {
                'status': 'postponed',
                'reason': 'high_system_load',
                'metrics': system_metrics
            }
            
        # Obtener datos de entrenamiento
        transactions = Transaction.objects.filter(
            created_at__gte=timezone.now() - timedelta(days=90)
        ).order_by('-date')
        
        if not transactions:
            logger.warning("No hay suficientes datos para entrenar")
            return {
                'status': 'skipped',
                'reason': 'insufficient_data'
            }
            
        # Entrenar modelos
        results = {}
        for model_name, model in ai_service.get_models().items():
            try:
                # Entrenar modelo
                model.train(transactions)
                
                # Evaluar rendimiento
                model_metrics = metrics.evaluate_model(model_name, model)
                
                # Guardar nueva versión
                version = versioning.save_model_version(
                    model_name,
                    model,
                    model_metrics
                )
                
                results[model_name] = {
                    'status': 'success',
                    'version': version,
                    'metrics': model_metrics
                }
                
            except Exception as e:
                logger.error(f"Error entrenando modelo {model_name}: {str(e)}")
                results[model_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                
        return {
            'status': 'success',
            'timestamp': timezone.now().isoformat(),
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error en tarea de entrenamiento: {str(e)}")
        raise
        
@shared_task
def evaluate_models():
    """
    Tarea periódica para evaluar modelos.
    
    Returns:
        Dict: Resultados de la evaluación
    """
    try:
        ai_service = AIService()
        metrics = ModelMetrics()
        versioning = ModelVersioning()
        
        # Obtener datos de prueba
        transactions = Transaction.objects.filter(
            created_at__gte=timezone.now() - timedelta(days=30)
        ).order_by('-date')
        
        if not transactions:
            return {
                'status': 'skipped',
                'reason': 'insufficient_data'
            }
            
        # Evaluar cada modelo
        results = {}
        for model_name, model in ai_service.get_models().items():
            try:
                # Cargar última versión
                model = versioning.load_model_version(model_name)
                
                # Evaluar rendimiento
                model_metrics = metrics.evaluate_model(model_name, model)
                
                results[model_name] = {
                    'status': 'success',
                    'metrics': model_metrics
                }
                
            except Exception as e:
                logger.error(f"Error evaluando modelo {model_name}: {str(e)}")
                results[model_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                
        return {
            'status': 'success',
            'timestamp': timezone.now().isoformat(),
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error en tarea de evaluación: {str(e)}")
        raise
        
@shared_task
def cleanup_old_versions():
    """
    Tarea periódica para limpiar versiones antiguas de modelos.
    
    Returns:
        Dict: Resultados de la limpieza
    """
    try:
        versioning = ModelVersioning()
        
        # Obtener todos los modelos
        models = versioning.get_all_models()
        
        results = {}
        for model_name in models:
            try:
                # Obtener versiones
                versions = versioning.get_model_versions(model_name)
                
                # Mantener solo las últimas 5 versiones
                if len(versions) > 5:
                    # Ordenar versiones por fecha
                    sorted_versions = sorted(
                        versions.keys(),
                        key=lambda x: versions[x]['timestamp'],
                        reverse=True
                    )
                    
                    # Eliminar versiones antiguas
                    for version in sorted_versions[5:]:
                        versioning.delete_model_version(model_name, version)
                        
                    results[model_name] = {
                        'status': 'success',
                        'versions_removed': len(sorted_versions) - 5
                    }
                else:
                    results[model_name] = {
                        'status': 'skipped',
                        'reason': 'insufficient_versions'
                    }
                    
            except Exception as e:
                logger.error(f"Error limpiando versiones de {model_name}: {str(e)}")
                results[model_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                
        return {
            'status': 'success',
            'timestamp': timezone.now().isoformat(),
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error en tarea de limpieza: {str(e)}")
        raise 