#!/bin/bash

# Obtener el directorio del script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Configurar el entorno
export PYTHONPATH=$PROJECT_ROOT
export DJANGO_SETTINGS_MODULE=financialhub.settings

# Iniciar el beat scheduler de Celery
cd "$PROJECT_ROOT"
celery -A financialhub beat -l info 