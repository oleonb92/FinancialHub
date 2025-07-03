"""
Configuración por defecto para FinancialHub
Carga la configuración base automáticamente
"""

import os

# Determinar qué configuración usar basado en el entorno
if os.getenv('DJANGO_ENV') == 'production':
    from .prod import *
elif os.getenv('DJANGO_ENV') == 'test':
    from .test import *
else:
    # Por defecto usar configuración de desarrollo
    from .dev import * 