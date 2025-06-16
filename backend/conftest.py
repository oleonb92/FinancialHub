"""
Configuración de pytest para el proyecto.
"""

import pytest
from django.conf import settings

@pytest.fixture(autouse=True)
def enable_db_access_for_all_tests(db):
    pass

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Configura el entorno de prueba."""
    settings.DEBUG = False
    settings.TESTING = True 