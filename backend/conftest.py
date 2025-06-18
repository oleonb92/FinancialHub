"""
Configuración de pytest para el proyecto.
"""

import pytest
from django.conf import settings
from django.core.management import call_command
from django.db import connections
from django.test.utils import setup_databases, teardown_databases

@pytest.fixture(autouse=True)
def enable_db_access_for_all_tests(db):
    pass

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Configura el entorno de prueba."""
    settings.DEBUG = False
    settings.TESTING = True

@pytest.fixture(scope='session')
def django_db_setup(django_db_setup, django_db_blocker):
    """
    Configura la base de datos de test y asegura que se limpie después de las pruebas.
    """
    with django_db_blocker.unblock():
        # Asegurarse de que las migraciones se apliquen en el orden correcto
        call_command('migrate', 'accounts', '--noinput')
        call_command('migrate', 'organizations', '--noinput')
        call_command('migrate', 'transactions', '--noinput')
        call_command('migrate', 'chartofaccounts', '--noinput')
        call_command('migrate', '--noinput')  # Aplicar el resto de migraciones
        
    yield
    
    # Limpiar después de las pruebas
    with django_db_blocker.unblock():
        for connection in connections.all():
            connection.close()

@pytest.fixture(autouse=True)
def _django_db_marker(request):
    """
    Asegura que las pruebas usen la base de datos de test.
    """
    if 'django_db' in request.keywords:
        return
    pytest.mark.django_db(request.node) 