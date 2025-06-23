"""
Configuración de Celery para tareas asíncronas.
"""
import os
from celery import Celery

# Configurar el módulo de configuración de Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'financialhub.settings')

# Crear aplicación Celery
app = Celery('financialhub')

# Configurar Celery usando la configuración de Django
app.config_from_object('django.conf:settings', namespace='CELERY')

# Cargar configuración desde config/celery.py
app.config_from_object('config.celery')

# Cargar tareas automáticamente
app.autodiscover_tasks() 

# Configurar tareas periódicas
app.conf.beat_schedule = {
    'train-models-daily': {
        'task': 'ai.tasks.training.train_models',
        'schedule': 86400.0,  # 24 horas
        'options': {
            'queue': 'training',
            'expires': 3600  # Expira después de 1 hora si no se ejecuta
        }
    },
    'evaluate-models-daily': {
        'task': 'ai.tasks.training.evaluate_models',
        'schedule': 86400.0,  # 24 horas
        'options': {
            'queue': 'evaluation',
            'expires': 3600
        }
    },
    'cleanup-versions-weekly': {
        'task': 'ai.tasks.training.cleanup_old_versions',
        'schedule': 604800.0,  # 7 días
        'options': {
            'queue': 'maintenance',
            'expires': 3600
        }
    },
    'monitor-resources-hourly': {
        'task': 'ai.tasks.monitoring.monitor_resources',
        'schedule': 3600.0,  # 1 hora
        'options': {
            'queue': 'monitoring',
            'expires': 300  # Expira después de 5 minutos si no se ejecuta
        }
    }
}

# Configuración de colas
app.conf.task_queues = {
    'training': {
        'exchange': 'training',
        'routing_key': 'training',
    },
    'evaluation': {
        'exchange': 'evaluation',
        'routing_key': 'evaluation',
    },
    'maintenance': {
        'exchange': 'maintenance',
        'routing_key': 'maintenance',
    },
    'monitoring': {
        'exchange': 'monitoring',
        'routing_key': 'monitoring',
    }
}

# Configuración de rutas
app.conf.task_routes = {
    'ai.tasks.training.train_models': {'queue': 'training'},
    'ai.tasks.training.evaluate_models': {'queue': 'evaluation'},
    'ai.tasks.training.cleanup_old_versions': {'queue': 'maintenance'},
    'ai.tasks.monitoring.monitor_resources': {'queue': 'monitoring'},
}

@app.task(bind=True)
def debug_task(self):
    """Tarea de prueba para verificar que Celery está funcionando"""
    print(f'Request: {self.request!r}') 