"""
Inicialización de la aplicación AI.
"""
from financialhub.celery import app as celery_app

__all__ = ('celery_app',) 