"""
Tareas de monitoreo de recursos del sistema.
"""
import logging
from celery import shared_task
from ai.ml.utils.monitoring import ResourceMonitor
from ai.ml.utils.cache.model_cache import ModelCache
from django.conf import settings

logger = logging.getLogger(__name__)

@shared_task(name='ai.tasks.monitoring.monitor_resources')
def monitor_resources():
    """
    Tarea periódica para monitorear recursos del sistema.
    Monitorea CPU, memoria y uso de disco.
    """
    try:
        monitor = ResourceMonitor()
        metrics = monitor.get_current_metrics()
        
        # Registrar métricas
        logger.info("Métricas de recursos del sistema:", extra={
            'cpu_percent': metrics['cpu_percent'],
            'memory_percent': metrics['memory_percent'],
            'disk_percent': metrics['disk_percent']
        })
        
        # Verificar si los recursos están por encima del umbral
        if metrics['cpu_percent'] > settings.AI_CPU_THRESHOLD:
            logger.warning(f"Uso de CPU por encima del umbral: {metrics['cpu_percent']}%")
            
        if metrics['memory_percent'] > settings.AI_MEMORY_THRESHOLD:
            logger.warning(f"Uso de memoria por encima del umbral: {metrics['memory_percent']}%")
            
        if metrics['disk_percent'] > settings.AI_DISK_THRESHOLD:
            logger.warning(f"Uso de disco por encima del umbral: {metrics['disk_percent']}%")
            
        # Limpiar caché si es necesario
        if metrics['memory_percent'] > settings.AI_MEMORY_THRESHOLD:
            cache = ModelCache()
            cache.clear_all()
            logger.info("Caché de modelos limpiada debido a alto uso de memoria")
            
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