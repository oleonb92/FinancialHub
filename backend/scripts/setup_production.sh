#!/bin/bash

# Script de configuración de producción para FinancialHub
# Sin Docker - Configuración nativa para máximo rendimiento

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

# Verificar si se ejecuta como root
if [[ $EUID -eq 0 ]]; then
   error "Este script no debe ejecutarse como root"
fi

# Variables de configuración
PROJECT_NAME="financialhub"
PROJECT_DIR="/var/www/$PROJECT_NAME"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"
NGINX_CONF="/etc/nginx/sites-available/$PROJECT_NAME"
NGINX_ENABLED="/etc/nginx/sites-enabled/$PROJECT_NAME"
SYSTEMD_DIR="/etc/systemd/system"

log "🚀 Iniciando configuración de producción para FinancialHub"

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

log "✅ Todas las dependencias están instaladas"

# 2. Crear directorios del proyecto
log "📁 Creando estructura de directorios..."

sudo mkdir -p $PROJECT_DIR
sudo mkdir -p $PROJECT_DIR/logs
sudo mkdir -p $PROJECT_DIR/media
sudo mkdir -p $PROJECT_DIR/staticfiles
sudo mkdir -p $PROJECT_DIR/backups
sudo mkdir -p $PROJECT_DIR/certs

# Cambiar permisos
sudo chown -R $USER:$USER $PROJECT_DIR
sudo chmod -R 755 $PROJECT_DIR

log "✅ Directorios creados"

# 3. Configurar entorno virtual de Python
log "🐍 Configurando entorno virtual de Python..."

cd $PROJECT_DIR
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias de Python
log "📦 Instalando dependencias de Python..."
pip install --upgrade pip
pip install -r $BACKEND_DIR/requirements.txt

# Instalar dependencias adicionales para producción
pip install gunicorn uwsgi whitenoise django-redis psycopg2-binary

log "✅ Entorno virtual configurado"

# 4. Configurar base de datos PostgreSQL
log "🗄️ Configurando base de datos PostgreSQL..."

# Crear base de datos y usuario
sudo -u postgres psql << EOF
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
if ! systemctl is-active --quiet redis; then
    sudo systemctl start redis
    sudo systemctl enable redis
fi

log "✅ Redis configurado"

# 6. Configurar variables de entorno
log "🔧 Configurando variables de entorno..."

cat > $PROJECT_DIR/.env << EOF
# Configuración de Django
DJANGO_SETTINGS_MODULE=financialhub.settings.production
SECRET_KEY=$(python3 -c 'from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())')
DEBUG=False
ALLOWED_HOSTS=financialhub.com,www.financialhub.com,localhost,127.0.0.1

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
CORS_ALLOWED_ORIGINS=https://financialhub.com,https://www.financialhub.com,https://app.financialhub.com

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

# Crear configuración de Nginx
sudo tee $NGINX_CONF > /dev/null << EOF
# Configuración de Nginx para FinancialHub
# Optimizada para producción con SSL, compresión y seguridad

# Configuración del upstream para Django
upstream django_backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
    keepalive 32;
}

# Configuración del upstream para WebSocket
upstream websocket_backend {
    server 127.0.0.1:8000;
    keepalive 32;
}

# Configuración del servidor HTTP (redirección a HTTPS)
server {
    listen 80;
    server_name financialhub.com www.financialhub.com;
    
    # Redirección a HTTPS
    return 301 https://\$server_name\$request_uri;
}

# Configuración del servidor HTTPS principal
server {
    listen 443 ssl http2;
    server_name financialhub.com www.financialhub.com;
    
    # Configuración SSL (usar certificados reales en producción)
    ssl_certificate /etc/ssl/certs/financialhub.crt;
    ssl_certificate_key /etc/ssl/private/financialhub.key;
    
    # Configuración SSL optimizada
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_session_tickets off;
    
    # Configuración de seguridad adicional
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Referrer-Policy "strict-origin-when-cross-origin";
    
    # Configuración de compresión
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;
    
    # Configuración de rate limiting
    limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone \$binary_remote_addr zone=login:10m rate=5r/m;
    
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
    
    # Configuración de WebSocket para chat en tiempo real
    location /ws/ {
        proxy_pass http://websocket_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
        proxy_read_timeout 86400;
    }
    
    # Configuración de la API con rate limiting
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://django_backend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header X-Forwarded-Host \$server_name;
        
        # Configuración de timeout
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # Configuración de buffer
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        proxy_busy_buffers_size 8k;
    }
    
    # Configuración de autenticación con rate limiting
    location /api/auth/ {
        limit_req zone=login burst=5 nodelay;
        
        proxy_pass http://django_backend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    # Configuración principal de Django
    location / {
        proxy_pass http://django_backend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header X-Forwarded-Host \$server_name;
        
        # Configuración de timeout
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # Configuración de buffer
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        proxy_busy_buffers_size 8k;
    }
    
    # Configuración de logging
    access_log /var/log/nginx/financialhub_access.log;
    error_log /var/log/nginx/financialhub_error.log;
}
EOF

# Habilitar el sitio
sudo ln -sf $NGINX_CONF $NGINX_ENABLED

# Verificar configuración de Nginx
sudo nginx -t

# Reiniciar Nginx
sudo systemctl restart nginx
sudo systemctl enable nginx

log "✅ Nginx configurado"

# 8. Configurar servicios systemd
log "⚙️ Configurando servicios systemd..."

# Servicio de Django
sudo tee $SYSTEMD_DIR/financialhub-django.service > /dev/null << EOF
[Unit]
Description=FinancialHub Django Application
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=notify
User=$USER
Group=$USER
WorkingDirectory=$BACKEND_DIR
Environment=PATH=$PROJECT_DIR/venv/bin
Environment=DJANGO_SETTINGS_MODULE=financialhub.settings.production
ExecStart=$PROJECT_DIR/venv/bin/gunicorn \\
    --workers 4 \\
    --bind 127.0.0.1:8000 \\
    --access-logfile $PROJECT_DIR/logs/gunicorn_access.log \\
    --error-logfile $PROJECT_DIR/logs/gunicorn_error.log \\
    --log-level info \\
    --timeout 120 \\
    --keep-alive 5 \\
    --max-requests 1000 \\
    --max-requests-jitter 100 \\
    financialhub.asgi:application \\
    -k uvicorn.workers.UvicornWorker
ExecReload=/bin/kill -s HUP \$MAINPID
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Servicio de Celery Worker
sudo tee $SYSTEMD_DIR/financialhub-celery.service > /dev/null << EOF
[Unit]
Description=FinancialHub Celery Worker
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$BACKEND_DIR
Environment=PATH=$PROJECT_DIR/venv/bin
Environment=DJANGO_SETTINGS_MODULE=financialhub.settings.production
ExecStart=$PROJECT_DIR/venv/bin/celery -A financialhub worker \\
    --loglevel=info \\
    --concurrency=4 \\
    --max-tasks-per-child=1000 \\
    --logfile=$PROJECT_DIR/logs/celery.log
ExecReload=/bin/kill -s HUP \$MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Servicio de Celery Beat
sudo tee $SYSTEMD_DIR/financialhub-celerybeat.service > /dev/null << EOF
[Unit]
Description=FinancialHub Celery Beat
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$BACKEND_DIR
Environment=PATH=$PROJECT_DIR/venv/bin
Environment=DJANGO_SETTINGS_MODULE=financialhub.settings.production
ExecStart=$PROJECT_DIR/venv/bin/celery -A financialhub beat \\
    --loglevel=info \\
    --logfile=$PROJECT_DIR/logs/celerybeat.log \\
    --scheduler=django_celery_beat.schedulers:DatabaseScheduler
ExecReload=/bin/kill -s HUP \$MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Recargar systemd
sudo systemctl daemon-reload

log "✅ Servicios systemd configurados"

# 9. Configurar Django
log "🐍 Configurando Django..."

cd $BACKEND_DIR

# Activar entorno virtual
source $PROJECT_DIR/venv/bin/activate

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

# 10. Configurar frontend
log "⚛️ Configurando frontend..."

cd $FRONTEND_DIR

# Instalar dependencias
npm install

# Construir para producción
npm run build

log "✅ Frontend configurado"

# 11. Configurar firewall
log "🔥 Configurando firewall..."

# Verificar si ufw está disponible
if command -v ufw &> /dev/null; then
    sudo ufw allow 22/tcp
    sudo ufw allow 80/tcp
    sudo ufw allow 443/tcp
    sudo ufw --force enable
    log "✅ Firewall configurado"
else
    warn "ufw no está disponible, configurar firewall manualmente"
fi

# 12. Configurar SSL (certbot)
log "🔒 Configurando SSL..."

# Verificar si certbot está disponible
if command -v certbot &> /dev/null; then
    log "📝 Configurando certificados SSL con Let's Encrypt..."
    sudo certbot --nginx -d financialhub.com -d www.financialhub.com --non-interactive --agree-tos --email admin@financialhub.com
    log "✅ SSL configurado"
else
    warn "certbot no está disponible, configurar SSL manualmente"
fi

# 13. Configurar monitoreo
log "📊 Configurando monitoreo..."

# Crear script de monitoreo
cat > $PROJECT_DIR/monitor.sh << 'EOF'
#!/bin/bash

# Script de monitoreo para FinancialHub

LOG_FILE="/var/www/financialhub/logs/monitor.log"
ALERT_EMAIL="admin@financialhub.com"

# Función para logging
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> $LOG_FILE
}

# Verificar servicios
check_service() {
    local service_name=$1
    if ! systemctl is-active --quiet $service_name; then
        log "ERROR: Servicio $service_name no está ejecutándose"
        echo "Servicio $service_name no está ejecutándose" | mail -s "Alerta FinancialHub" $ALERT_EMAIL
        return 1
    fi
    return 0
}

# Verificar uso de CPU
check_cpu() {
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    if (( $(echo "$cpu_usage > 80" | bc -l) )); then
        log "WARNING: Uso de CPU alto: ${cpu_usage}%"
    fi
}

# Verificar uso de memoria
check_memory() {
    local mem_usage=$(free | grep Mem | awk '{printf("%.2f", $3/$2 * 100.0)}')
    if (( $(echo "$mem_usage > 80" | bc -l) )); then
        log "WARNING: Uso de memoria alto: ${mem_usage}%"
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
        echo "Base de datos no está disponible" | mail -s "Alerta FinancialHub" $ALERT_EMAIL
        return 1
    fi
    return 0
}

# Verificar conectividad de Redis
check_redis() {
    if ! redis-cli ping > /dev/null 2>&1; then
        log "ERROR: Redis no está disponible"
        echo "Redis no está disponible" | mail -s "Alerta FinancialHub" $ALERT_EMAIL
        return 1
    fi
    return 0
}

# Ejecutar verificaciones
log "Iniciando verificación de monitoreo"

check_service financialhub-django
check_service financialhub-celery
check_service financialhub-celerybeat
check_service nginx
check_service postgresql
check_service redis

check_cpu
check_memory
check_disk
check_database
check_redis

log "Verificación de monitoreo completada"
EOF

chmod +x $PROJECT_DIR/monitor.sh

# Configurar cron para monitoreo
(crontab -l 2>/dev/null; echo "*/5 * * * * $PROJECT_DIR/monitor.sh") | crontab -

log "✅ Monitoreo configurado"

# 14. Configurar backup automático
log "💾 Configurando backup automático..."

# Crear script de backup
cat > $PROJECT_DIR/backup.sh << 'EOF'
#!/bin/bash

# Script de backup para FinancialHub

BACKUP_DIR="/var/www/financialhub/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="financialhub_backup_$DATE.sql"
MEDIA_BACKUP="financialhub_media_$DATE.tar.gz"

# Crear directorio de backup si no existe
mkdir -p $BACKUP_DIR

# Backup de base de datos
pg_dump -h localhost -U financialhub_user -d financialhub_prod > $BACKUP_DIR/$BACKUP_FILE

# Backup de archivos media
tar -czf $BACKUP_DIR/$MEDIA_BACKUP -C /var/www/financialhub media/

# Comprimir backup de base de datos
gzip $BACKUP_DIR/$BACKUP_FILE

# Eliminar backups antiguos (más de 30 días)
find $BACKUP_DIR -name "financialhub_backup_*.sql.gz" -mtime +30 -delete
find $BACKUP_DIR -name "financialhub_media_*.tar.gz" -mtime +30 -delete

echo "Backup completado: $BACKUP_FILE.gz, $MEDIA_BACKUP"
EOF

chmod +x $PROJECT_DIR/backup.sh

# Configurar cron para backup diario
(crontab -l 2>/dev/null; echo "0 2 * * * $PROJECT_DIR/backup.sh") | crontab -

log "✅ Backup automático configurado"

# 15. Iniciar servicios
log "🚀 Iniciando servicios..."

sudo systemctl enable financialhub-django
sudo systemctl enable financialhub-celery
sudo systemctl enable financialhub-celerybeat

sudo systemctl start financialhub-django
sudo systemctl start financialhub-celery
sudo systemctl start financialhub-celerybeat

log "✅ Servicios iniciados"

# 16. Verificar estado de servicios
log "🔍 Verificando estado de servicios..."

sleep 5

if systemctl is-active --quiet financialhub-django; then
    log "✅ Django está ejecutándose"
else
    error "❌ Django no está ejecutándose"
fi

if systemctl is-active --quiet financialhub-celery; then
    log "✅ Celery Worker está ejecutándose"
else
    error "❌ Celery Worker no está ejecutándose"
fi

if systemctl is-active --quiet financialhub-celerybeat; then
    log "✅ Celery Beat está ejecutándose"
else
    error "❌ Celery Beat no está ejecutándose"
fi

if systemctl is-active --quiet nginx; then
    log "✅ Nginx está ejecutándose"
else
    error "❌ Nginx no está ejecutándose"
fi

# 17. Mostrar información final
log "🎉 ¡Configuración de producción completada!"

echo -e "${BLUE}"
echo "=========================================="
echo "    FINANCIALHUB - PRODUCCIÓN LISTA"
echo "=========================================="
echo ""
echo "📁 Directorio del proyecto: $PROJECT_DIR"
echo "🌐 URL principal: https://financialhub.com"
echo "🔧 Admin Django: https://financialhub.com/admin"
echo "📊 Monitoreo: $PROJECT_DIR/monitor.sh"
echo "💾 Backup: $PROJECT_DIR/backup.sh"
echo ""
echo "📋 Comandos útiles:"
echo "  - Ver logs: sudo journalctl -u financialhub-django -f"
echo "  - Reiniciar servicios: sudo systemctl restart financialhub-*"
echo "  - Verificar estado: sudo systemctl status financialhub-*"
echo "  - Monitoreo manual: $PROJECT_DIR/monitor.sh"
echo "  - Backup manual: $PROJECT_DIR/backup.sh"
echo ""
echo "🔒 IMPORTANTE:"
echo "  - Cambiar contraseñas por defecto"
echo "  - Configurar certificados SSL reales"
echo "  - Configurar email real"
echo "  - Revisar configuración de firewall"
echo "=========================================="
echo -e "${NC}"

log "✅ ¡FinancialHub está listo para producción!" 