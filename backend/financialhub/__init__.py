# Este archivo inicializa el paquete financialhub.
# Celery eliminado para entorno local.

"""
Inicialización de la aplicación FinancialHub.
"""
from .celery import app as celery_app

__all__ = ('celery_app',)
