#!/bin/bash

# Colores para output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

export PYTHONPATH=.

echo "🚀 Iniciando proceso de pruebas..."

# Verificar que estamos en el directorio correcto
if [ ! -f "manage.py" ]; then
    echo -e "${RED}❌ Error: Debes ejecutar este script desde el directorio backend${NC}"
    exit 1
fi

# Activar entorno virtual si existe
if [ -d ".venv" ]; then
    echo "📦 Activando entorno virtual..."
    source .venv/bin/activate
fi

# Instalar dependencias si es necesario
echo "📦 Verificando dependencias..."
pip install -r requirements.txt

# Configurar base de datos de test
echo "🔧 Configurando base de datos de test..."
python tools/test_db_setup.py

# Ejecutar tests
echo "🧪 Ejecutando tests..."
python -m pytest -v --reuse-db --create-db "$@"

# Verificar resultado
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Tests completados exitosamente${NC}"
else
    echo -e "${RED}❌ Tests fallaron${NC}"
    exit 1
fi

# Limpiar
echo "🧹 Limpiando..."
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete

echo "✨ Proceso completado" 