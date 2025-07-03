#!/bin/bash

# Script de configuración de producción para FinancialHub en macOS
# Adaptado para funcionar sin permisos de administrador

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Variables de configuración para macOS
PROJECT_NAME="financialhub"
PROJECT_DIR="$HOME/Desktop/FinancialHub_backup"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"
NGINX_CONF="/opt/homebrew/etc/nginx/servers/financialhub.conf"

log "🚀 Iniciando configuración de producción para FinancialHub en macOS"

# 1. Verificar dependencias del sistema
log "📋 Verificando dependencias del sistema..."

# Verificar Python
if ! command -v python3 &> /dev/null; then
    error "Python 3 no está instalado"
fi

# Verificar pip
if ! command -v pip3 &> /dev/null; then
    error "pip3 no está instalado"
fi

# Verificar Node.js
if ! command -v node &> /dev/null; then
    error "Node.js no está instalado"
fi

# Verificar npm
if ! command -v npm &> /dev/null; then
    error "npm no está instalado"
fi

# Verificar PostgreSQL
if ! command -v psql &> /dev/null; then
    error "PostgreSQL no está instalado"
fi

# Verificar Redis
if ! command -v redis-cli &> /dev/null; then
    error "Redis no está instalado"
fi

# Verificar Nginx
if ! command -v nginx &> /dev/null; then
    error "Nginx no está instalado"
fi

# Verificar Homebrew
if ! command -v brew &> /dev/null; then
    error "Homebrew no está instalado"
fi

log "✅ Todas las dependencias están instaladas"

# 2. Crear directorios del proyecto
log "📁 Creando estructura de directorios..."

mkdir -p $PROJECT_DIR/logs
mkdir -p $PROJECT_DIR/media
mkdir -p $PROJECT_DIR/staticfiles
mkdir -p $PROJECT_DIR/backups
mkdir -p $PROJECT_DIR/certs

log "✅ Directorios creados"

# 3. Configurar entorno virtual de Python
log "🐍 Configurando entorno virtual de Python..."

cd $PROJECT_DIR

# Verificar si ya existe el entorno virtual
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

# Instalar dependencias de Python
log "📦 Instalando dependencias de Python..."
pip install --upgrade pip
pip install -r $PROJECT_DIR/requirements.txt

# Instalar dependencias adicionales para producción
pip install gunicorn uwsgi whitenoise django-redis psycopg2-binary

log "✅ Entorno virtual configurado"

# 4. Configurar base de datos PostgreSQL
log "🗄️ Configurando base de datos PostgreSQL..."

# Crear base de datos y usuario
psql postgres << EOF
CREATE DATABASE financialhub_prod;
CREATE USER financialhub_user WITH PASSWORD 'financialhub_password_2024';
GRANT ALL PRIVILEGES ON DATABASE financialhub_prod TO financialhub_user;
ALTER USER financialhub_user CREATEDB;
\q
EOF

log "✅ Base de datos configurada"

# 5. Configurar Redis
log "🔴 Configurando Redis..."

# Verificar que Redis esté ejecutándose
if ! brew services list | grep -q "redis.*started"; then
    brew services start redis
fi

log "✅ Redis configurado"

# 6. Configurar variables de entorno
log "🔧 Configurando variables de entorno..."

cat > $PROJECT_DIR/.env.prod << EOF
# Configuración de Django
DJANGO_SETTINGS_MODULE=financialhub.settings.production
SECRET_KEY=$(python3 -c 'from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())')
DEBUG=False
ALLOWED_HOSTS=localhost,127.0.0.1

# Configuración de base de datos
DB_NAME=financialhub_prod
DB_USER=financialhub_user
DB_PASSWORD=financialhub_password_2024
DB_HOST=localhost
DB_PORT=5432

# Configuración de Redis
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0

# Configuración de email
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password
DEFAULT_FROM_EMAIL=noreply@financialhub.com

# Configuración de CORS
CORS_ALLOWED_ORIGINS=http://localhost:3000,https://localhost:3000

# Configuración de AI
AI_MODEL_CACHE_TIMEOUT=7200
AI_MAX_CONCURRENT_REQUESTS=10
AI_REQUEST_TIMEOUT=30

# Configuración de backup
BACKUP_ENABLED=True
BACKUP_RETENTION_DAYS=30

# Configuración de monitoreo
RESOURCE_MONITORING_ENABLED=True
RESOURCE_ALERT_THRESHOLD=80.0
EOF

log "✅ Variables de entorno configuradas"

# 7. Configurar Nginx
log "🌐 Configurando Nginx..."

# Crear configuración de Nginx para macOS
sudo tee $NGINX_CONF > /dev/null << EOF
# Configuración de Nginx para FinancialHub en macOS

# Configuración del upstream para Django
upstream django_backend {
    server 127.0.0.1:8000;
    keepalive 32;
}

# Configuración del servidor HTTP
server {
    listen 8080;
    server_name localhost;
    
    # Configuración de archivos estáticos
    location /static/ {
        alias $PROJECT_DIR/staticfiles/;
        expires 1y;
        add_header Cache-Control "public, immutable";
        access_log off;
    }
    
    # Configuración de archivos media
    location /media/ {
        alias $PROJECT_DIR/media/;
        expires 1y;
        add_header Cache-Control "public";
        access_log off;
    }
    
    # Configuración de la API
    location /api/ {
        proxy_pass http://django_backend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Configuración de timeout
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    # Configuración principal de Django
    location / {
        proxy_pass http://django_backend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Configuración de timeout
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    # Configuración de logging
    access_log $PROJECT_DIR/logs/nginx_access.log;
    error_log $PROJECT_DIR/logs/nginx_error.log;
}
EOF

# Verificar configuración de Nginx
sudo nginx -t

# Reiniciar Nginx
brew services restart nginx

log "✅ Nginx configurado"

# 8. Configurar Django
log "🐍 Configurando Django..."

cd $BACKEND_DIR

# Activar entorno virtual
source $PROJECT_DIR/.venv/bin/activate

# Cargar variables de entorno de producción
export $(grep -v '^#' $PROJECT_DIR/.env.prod | xargs)

# Ejecutar migraciones
python manage.py migrate --settings=financialhub.settings.production

# Recolectar archivos estáticos
python manage.py collectstatic --noinput --settings=financialhub.settings.production

# Crear superusuario si no existe
if ! python manage.py shell --settings=financialhub.settings.production -c "from django.contrib.auth import get_user_model; User = get_user_model(); User.objects.filter(is_superuser=True).exists()" 2>/dev/null | grep -q "True"; then
    log "👤 Creando superusuario..."
    python manage.py createsuperuser --settings=financialhub.settings.production
fi

log "✅ Django configurado"

# 9. Configurar frontend
log "⚛️ Configurando frontend..."

cd $FRONTEND_DIR

# Instalar dependencias
npm install

# Construir para producción
npm run build

log "✅ Frontend configurado"

# 10. Crear scripts de gestión
log "📝 Creando scripts de gestión..."

# Script para iniciar Django
cat > $PROJECT_DIR/start_django.sh << 'EOF'
#!/bin/bash
cd /Users/osmanileon/Desktop/FinancialHub_backup/backend
source ../.venv/bin/activate
export $(grep -v '^#' ../.env.prod | xargs)
python manage.py runserver 0.0.0.0:8000 --settings=financialhub.settings.production
EOF

# Script para iniciar Celery
cat > $PROJECT_DIR/start_celery.sh << 'EOF'
#!/bin/bash
cd /Users/osmanileon/Desktop/FinancialHub_backup/backend
source ../.venv/bin/activate
export $(grep -v '^#' ../.env.prod | xargs)
celery -A financialhub worker --loglevel=info
EOF

# Script para iniciar Celery Beat
cat > $PROJECT_DIR/start_celerybeat.sh << 'EOF'
#!/bin/bash
cd /Users/osmanileon/Desktop/FinancialHub_backup/backend
source ../.venv/bin/activate
export $(grep -v '^#' ../.env.prod | xargs)
celery -A financialhub beat --loglevel=info
EOF

# Script de monitoreo
cat > $PROJECT_DIR/monitor.sh << 'EOF'
#!/bin/bash

# Script de monitoreo para FinancialHub

LOG_FILE="/Users/osmanileon/Desktop/FinancialHub_backup/logs/monitor.log"

# Función para logging
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> $LOG_FILE
}

# Verificar servicios
check_service() {
    local service_name=$1
    if ! brew services list | grep -q "$service_name.*started"; then
        log "ERROR: Servicio $service_name no está ejecutándose"
        return 1
    fi
    return 0
}

# Verificar uso de CPU
check_cpu() {
    local cpu_usage=$(top -l 1 | grep "CPU usage" | awk '{print $3}' | cut -d'%' -f1)
    if (( $(echo "$cpu_usage > 80" | bc -l) )); then
        log "WARNING: Uso de CPU alto: ${cpu_usage}%"
    fi
}

# Verificar uso de memoria
check_memory() {
    local mem_usage=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
    local total_mem=$(sysctl hw.memsize | awk '{print $2}')
    local free_mem=$((mem_usage * 4096))
    local used_mem=$((total_mem - free_mem))
    local mem_percent=$((used_mem * 100 / total_mem))
    
    if [ $mem_percent -gt 80 ]; then
        log "WARNING: Uso de memoria alto: ${mem_percent}%"
    fi
}

# Verificar uso de disco
check_disk() {
    local disk_usage=$(df / | tail -1 | awk '{print $5}' | cut -d'%' -f1)
    if [ $disk_usage -gt 80 ]; then
        log "WARNING: Uso de disco alto: ${disk_usage}%"
    fi
}

# Verificar conectividad de base de datos
check_database() {
    if ! pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
        log "ERROR: Base de datos no está disponible"
        return 1
    fi
    return 0
}

# Verificar conectividad de Redis
check_redis() {
    if ! redis-cli ping > /dev/null 2>&1; then
        log "ERROR: Redis no está disponible"
        return 1
    fi
    return 0
}

# Ejecutar verificaciones
log "Iniciando verificación de monitoreo"

check_service postgresql
check_service redis
check_service nginx

check_cpu
check_memory
check_disk
check_database
check_redis

log "Verificación de monitoreo completada"
EOF

# Script de backup
cat > $PROJECT_DIR/backup.sh << 'EOF'
#!/bin/bash

# Script de backup para FinancialHub

BACKUP_DIR="/Users/osmanileon/Desktop/FinancialHub_backup/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="financialhub_backup_$DATE.sql"
MEDIA_BACKUP="financialhub_media_$DATE.tar.gz"

# Crear directorio de backup si no existe
mkdir -p $BACKUP_DIR

# Backup de base de datos
pg_dump -h localhost -U financialhub_user -d financialhub_prod > $BACKUP_DIR/$BACKUP_FILE

# Backup de archivos media
tar -czf $BACKUP_DIR/$MEDIA_BACKUP -C /Users/osmanileon/Desktop/FinancialHub_backup media/

# Comprimir backup de base de datos
gzip $BACKUP_DIR/$BACKUP_FILE

# Eliminar backups antiguos (más de 30 días)
find $BACKUP_DIR -name "financialhub_backup_*.sql.gz" -mtime +30 -delete
find $BACKUP_DIR -name "financialhub_media_*.tar.gz" -mtime +30 -delete

echo "Backup completado: $BACKUP_FILE.gz, $MEDIA_BACKUP"
EOF

# Dar permisos de ejecución
chmod +x $PROJECT_DIR/start_django.sh
chmod +x $PROJECT_DIR/start_celery.sh
chmod +x $PROJECT_DIR/start_celerybeat.sh
chmod +x $PROJECT_DIR/monitor.sh
chmod +x $PROJECT_DIR/backup.sh

log "✅ Scripts de gestión creados"

# 11. Configurar cron para monitoreo y backup
log "⏰ Configurando tareas programadas..."

# Configurar cron para monitoreo
(crontab -l 2>/dev/null; echo "*/5 * * * * $PROJECT_DIR/monitor.sh") | crontab -

# Configurar cron para backup diario
(crontab -l 2>/dev/null; echo "0 2 * * * $PROJECT_DIR/backup.sh") | crontab -

log "✅ Tareas programadas configuradas"

# 12. Verificar que todo funciona
log "🔍 Verificando que todo funciona..."

# Verificar que Django funciona
cd $BACKEND_DIR
source $PROJECT_DIR/.venv/bin/activate
export $(grep -v '^#' $PROJECT_DIR/.env.prod | xargs)

if python manage.py check --settings=financialhub.settings.production; then
    log "✅ Django funciona correctamente"
else
    error "❌ Django no funciona correctamente"
fi

# Verificar que Nginx funciona
if brew services list | grep -q "nginx.*started"; then
    log "✅ Nginx está ejecutándose"
else
    error "❌ Nginx no está ejecutándose"
fi

# Verificar que PostgreSQL funciona
if brew services list | grep -q "postgresql.*started"; then
    log "✅ PostgreSQL está ejecutándose"
else
    error "❌ PostgreSQL no está ejecutándose"
fi

# Verificar que Redis funciona
if brew services list | grep -q "redis.*started"; then
    log "✅ Redis está ejecutándose"
else
    error "❌ Redis no está ejecutándose"
fi

# 13. Mostrar información final
log "🎉 ¡Configuración de producción completada!"

echo -e "${BLUE}"
echo "=========================================="
echo "    FINANCIALHUB - PRODUCCIÓN LISTA"
echo "=========================================="
echo ""
echo "📁 Directorio del proyecto: $PROJECT_DIR"
echo "🌐 URL principal: http://localhost:8080"
echo "🔧 Admin Django: http://localhost:8080/admin"
echo "📊 Monitoreo: $PROJECT_DIR/monitor.sh"
echo "💾 Backup: $PROJECT_DIR/backup.sh"
echo ""
echo "📋 Comandos útiles:"
echo "  - Iniciar Django: $PROJECT_DIR/start_django.sh"
echo "  - Iniciar Celery: $PROJECT_DIR/start_celery.sh"
echo "  - Iniciar Celery Beat: $PROJECT_DIR/start_celerybeat.sh"
echo "  - Monitoreo manual: $PROJECT_DIR/monitor.sh"
echo "  - Backup manual: $PROJECT_DIR/backup.sh"
echo ""
echo "🔒 IMPORTANTE:"
echo "  - Cambiar contraseñas por defecto"
echo "  - Configurar email real"
echo "  - Revisar configuración de seguridad"
echo "=========================================="
echo -e "${NC}"

log "✅ ¡FinancialHub está listo para producción en macOS!" 