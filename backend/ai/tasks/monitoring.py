"""
Tareas de monitoreo de recursos del sistema.
"""
import logging
from celery import shared_task
from ai.ml.utils.monitoring.resource_monitor import ResourceMonitor
from ai.ml.utils.cache.model_cache import ModelCache
from django.conf import settings
from django.core.cache import cache as django_cache

logger = logging.getLogger(__name__)

@shared_task(name='ai.tasks.monitoring.monitor_resources')
def monitor_resources():
    """
    Tarea periódica para monitorear recursos del sistema.
    Monitorea CPU, memoria y uso de disco.
    """
    try:
        monitor = ResourceMonitor()
        metrics = monitor.collect_metrics()
        
        # Registrar métricas
        logger.info("Métricas de recursos del sistema:", extra={
            'cpu_percent': metrics['cpu']['percent'],
            'memory_percent': metrics['memory']['percent'],
            'disk_percent': metrics['disk']['percent']
        })
        
        # Verificar si los recursos están por encima del umbral
        if metrics['cpu']['percent'] > getattr(settings, 'AI_CPU_THRESHOLD', 80):
            logger.warning(f"Uso de CPU por encima del umbral: {metrics['cpu']['percent']}%")
            
        if metrics['memory']['percent'] > getattr(settings, 'AI_MEMORY_THRESHOLD', 80):
            logger.warning(f"Uso de memoria por encima del umbral: {metrics['memory']['percent']}%")
            
        if metrics['disk']['percent'] > getattr(settings, 'AI_DISK_THRESHOLD', 90):
            logger.warning(f"Uso de disco por encima del umbral: {metrics['disk']['percent']}%")
            
        # Limpiar caché si es necesario
        if metrics['memory']['percent'] > getattr(settings, 'AI_MEMORY_THRESHOLD', 80):
            try:
                cache = ModelCache()
                cache.clear_all_cache()
                logger.info("Caché de modelos limpiada debido a alto uso de memoria")
            except Exception as cache_error:
                logger.error(f"Error limpiando caché: {cache_error}")
            
        return {
            'status': 'success',
            'metrics': metrics
        }
        
    except Exception as e:
        logger.error(f"Error al monitorear recursos: {str(e)}")
        return {
            'status': 'error',
            'error': str(e)
        } 