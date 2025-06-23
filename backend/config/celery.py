"""
Configuración de Celery para el proyecto.
"""
import os
import sys

# Agregar el directorio raíz al PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configuración de Celery
broker_url = 'redis://localhost:6379/0'
result_backend = 'django-db'
accept_content = ['json']
task_serializer = 'json'
result_serializer = 'json'
timezone = 'UTC'
task_track_started = True
task_time_limit = 30 * 60  # 30 minutos
task_soft_time_limit = 25 * 60  # 25 minutos
worker_prefetch_multiplier = 1
worker_max_tasks_per_child = 1000 