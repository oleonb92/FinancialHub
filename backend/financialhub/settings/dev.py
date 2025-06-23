"""
Configuración de desarrollo para FinancialHub
Extiende la configuración base con configuraciones específicas para desarrollo
"""

from .base import *

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'

# Configuración de hosts permitidos desde variables de entorno
ALLOWED_HOSTS = os.getenv('DJANGO_ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

# Database - PostgreSQL para desarrollo
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('DB_NAME', 'financialhub'),
        'USER': os.getenv('DB_USER', 'oleonb'),
        'PASSWORD': os.getenv('DB_PASSWORD', 'Natali@rca1992'),
        'HOST': os.getenv('DB_HOST', 'localhost'),
        'PORT': os.getenv('DB_PORT', '5432'),
    }
}

# Development-specific settings
if DEBUG:
    import mimetypes
    mimetypes.add_type("application/javascript", ".js", True)
    LOGGING['root']['level'] = 'DEBUG'
    LOGGING['loggers']['django']['level'] = 'DEBUG'

# Redis settings desde variables de entorno
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_URL = os.getenv('REDIS_URL', f'redis://{REDIS_HOST}:{REDIS_PORT}/0')

# Stripe settings desde variables de entorno
STRIPE_SECRET_KEY = os.getenv('STRIPE_SECRET_KEY')
STRIPE_PUBLISHABLE_KEY = os.getenv('STRIPE_PUBLISHABLE_KEY')
STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_WEBHOOK_SECRET')

# AI settings desde variables de entorno
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
AI_MODEL = os.getenv('AI_MODEL', 'gpt-4')
AI_TEMPERATURE = float(os.getenv('AI_TEMPERATURE', '0.7'))
AI_MAX_TOKENS = int(os.getenv('AI_MAX_TOKENS', '2000'))

# AI Resource Thresholds desde variables de entorno
AI_CPU_THRESHOLD = float(os.getenv('AI_CPU_THRESHOLD', '80.0'))
AI_MEMORY_THRESHOLD = float(os.getenv('AI_MEMORY_THRESHOLD', '80.0'))
AI_DISK_THRESHOLD = float(os.getenv('AI_DISK_THRESHOLD', '90.0'))
AI_MIN_MEMORY_GB = float(os.getenv('AI_MIN_MEMORY_GB', '1.0'))
AI_MIN_DISK_GB = float(os.getenv('AI_MIN_DISK_GB', '5.0'))

# AI Model Cache Settings desde variables de entorno
AI_MODEL_CACHE_TIMEOUT = int(os.getenv('AI_MODEL_CACHE_TIMEOUT', '3600'))
AI_MODEL_CACHE_PREFIX = os.getenv('AI_MODEL_CACHE_PREFIX', 'ai_model_')

# AI Model Versioning Settings desde variables de entorno
AI_MAX_MODEL_VERSIONS = int(os.getenv('AI_MAX_MODEL_VERSIONS', '5'))

# AI Feature Flags desde variables de entorno
AI_ENABLED = os.getenv('AI_ENABLED', 'true').lower() == 'true'
AI_CACHE_ENABLED = os.getenv('AI_CACHE_ENABLED', 'true').lower() == 'true'
AI_MONITORING_ENABLED = os.getenv('AI_MONITORING_ENABLED', 'true').lower() == 'true'

# Configuración de seguridad relajada para desarrollo
SECURE_SSL_REDIRECT = False
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False

# Configuración de CORS desde variables de entorno
CORS_ALLOWED_ORIGINS = os.getenv('CORS_ALLOWED_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000').split(',')
CORS_ALLOW_ALL_ORIGINS = True  # Para desarrollo

# Configuración de CSRF desde variables de entorno
CSRF_TRUSTED_ORIGINS = os.getenv('DJANGO_CSRF_TRUSTED_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000').split(',')

# Configuración de email desde variables de entorno
EMAIL_BACKEND = os.getenv('EMAIL_BACKEND', 'django.core.mail.backends.console.EmailBackend')
EMAIL_HOST = os.getenv('EMAIL_HOST', 'localhost')
EMAIL_PORT = int(os.getenv('EMAIL_PORT', '1025'))
EMAIL_USE_TLS = os.getenv('EMAIL_USE_TLS', 'False').lower() == 'true'
EMAIL_HOST_USER = os.getenv('EMAIL_HOST_USER', '')
EMAIL_HOST_PASSWORD = os.getenv('EMAIL_HOST_PASSWORD', '')
DEFAULT_FROM_EMAIL = os.getenv('DEFAULT_FROM_EMAIL', 'osmanileon92@gmail.com')

# Configuración de Celery desde variables de entorno
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'django-db')

# Configuración de Channels desde variables de entorno
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            "hosts": [(REDIS_HOST, REDIS_PORT)],
        },
    },
}

# Configuración de archivos estáticos desde variables de entorno
STATIC_URL = os.getenv('STATIC_URL', '/static/')
STATIC_ROOT = os.getenv('STATIC_ROOT', 'staticfiles')
MEDIA_URL = os.getenv('MEDIA_URL', '/media/')
MEDIA_ROOT = os.getenv('MEDIA_ROOT', 'media')

# Configuración de MLflow desde variables de entorno
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'financial_hub_ai')

# Configuración de monitoreo desde variables de entorno
PROMETHEUS_ENABLED = os.getenv('PROMETHEUS_ENABLED', 'true').lower() == 'true'
GRAFANA_ENABLED = os.getenv('GRAFANA_ENABLED', 'true').lower() == 'true'

# Configuración de privacidad desde variables de entorno
FEDERATED_LEARNING_ENABLED = os.getenv('FEDERATED_LEARNING_ENABLED', 'true').lower() == 'true'
PRIVACY_BUDGET = float(os.getenv('PRIVACY_BUDGET', '1.0'))

# Configuración de experimentación desde variables de entorno
AB_TESTING_ENABLED = os.getenv('AB_TESTING_ENABLED', 'true').lower() == 'true'
EXPERIMENT_TRACKING_ENABLED = os.getenv('EXPERIMENT_TRACKING_ENABLED', 'true').lower() == 'true'

# Configuración de Redis Cache desde variables de entorno
REDIS_CACHE_TIMEOUT = int(os.getenv('REDIS_CACHE_TIMEOUT', '3600'))

# Configuración de sitio desde variables de entorno
SITE_URL = os.getenv('SITE_URL', 'http://localhost:3000')

# Configuración de superusuario desde variables de entorno
DJANGO_SUPERUSER_USERNAME = os.getenv('DJANGO_SUPERUSER_USERNAME', 'oleonb')
DJANGO_SUPERUSER_EMAIL = os.getenv('DJANGO_SUPERUSER_EMAIL', 'osmanileon92@gmail.com')
DJANGO_SUPERUSER_PASSWORD = os.getenv('DJANGO_SUPERUSER_PASSWORD', 'Natali@rca1992') 