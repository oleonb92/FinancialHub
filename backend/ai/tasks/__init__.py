"""
Tareas de Celery para el módulo AI.
"""
from .monitoring import monitor_resources
from .training import train_models, evaluate_models, cleanup_old_versions

__all__ = [
    'monitor_resources',
    'train_models',
    'evaluate_models',
    'cleanup_old_versions'
] 