#!/bin/bash

# Obtener el directorio del script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Agregar el directorio al PYTHONPATH
export PYTHONPATH=$DIR:$PYTHONPATH

# Iniciar Celery worker
celery -A financialhub worker -l info 