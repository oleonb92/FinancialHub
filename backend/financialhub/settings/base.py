"""
Configuración base para FinancialHub
Contiene todas las configuraciones comunes compartidas entre entornos
"""

import os
from dotenv import load_dotenv
from datetime import timedelta
from pathlib import Path
from core.logging import setup_logging
import sys

# Cargar variables de entorno desde el root del proyecto
# Buscar .env en múltiples ubicaciones posibles
env_paths = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), '.env'),  # Desde backend/
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'),  # Desde raíz del proyecto
    os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'),  # Desde financialhub/
]

for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        break

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Initialize logging SOLO si no estamos en modo test
if 'test_settings' not in os.environ.get('DJANGO_SETTINGS_MODULE', '') and not any('pytest' in arg for arg in sys.argv):
    # Pass BASE_DIR directly to avoid circular import
    from core.logging import setup_logging
    setup_logging(base_dir=str(BASE_DIR))

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get('SECRET_KEY', 'django-insecure-default-key-for-development')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'

# Hosts configuration
ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1,0.0.0.0').split(',')

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'rest_framework.authtoken',
    'corsheaders',
    'accounts',
    'organizations',
    'transactions',
    'chartofaccounts',
    'goals',
    'chat',
    'notifications',
    'ai',
    'audit',
    'payments',
    'incentives',
    'api',
    'core',
    'django_celery_results',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.locale.LocaleMiddleware',  # Internationalization
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'accounts.middleware.OrganizationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'financialhub.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'financialhub.wsgi.application'
ASGI_APPLICATION = 'financialhub.asgi.application'

# Database configuration - desde variables de entorno
DATABASES = {
    'default': {
        'ENGINE': os.getenv('DB_ENGINE', 'django.db.backends.sqlite3'),
        'NAME': os.getenv('DB_NAME', os.path.join(BASE_DIR, 'db.sqlite3')),
        'USER': os.getenv('DB_USER', ''),
        'PASSWORD': os.getenv('DB_PASSWORD', ''),
        'HOST': os.getenv('DB_HOST', ''),
        'PORT': os.getenv('DB_PORT', ''),
        'OPTIONS': {
            'charset': 'utf8mb4',
        } if os.getenv('DB_ENGINE') == 'django.db.backends.mysql' else {},
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_L10N = True
USE_TZ = True

LANGUAGES = [
    ('en', 'English'),
    ('es', 'Spanish'),
]

LOCALE_PATHS = [
    os.path.join(BASE_DIR, 'locale'),
]

# Static files (CSS, JavaScript, Images) - desde variables de entorno
STATIC_URL = os.getenv('STATIC_URL', '/static/')
STATIC_ROOT = os.getenv('STATIC_ROOT', os.path.join(BASE_DIR, 'staticfiles'))
MEDIA_URL = os.getenv('MEDIA_URL', '/media/')
MEDIA_ROOT = os.getenv('MEDIA_ROOT', os.path.join(BASE_DIR, 'media'))

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# CORS settings - desde variables de entorno
CORS_ALLOWED_ORIGINS = os.getenv('CORS_ALLOWED_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000').split(',')

CORS_ALLOW_CREDENTIALS = True

CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
    'x-organization-id',
    'X-Organization-ID',
]

CORS_ALLOW_METHODS = [
    'DELETE',
    'GET',
    'OPTIONS',
    'PATCH',
    'POST',
    'PUT',
]

# REST Framework settings
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ),
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',
    ),
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 100,
    'DEFAULT_FILTER_BACKENDS': (
        'django_filters.rest_framework.DjangoFilterBackend',
        'rest_framework.filters.SearchFilter',
        'rest_framework.filters.OrderingFilter',
    ),
}

# JWT settings
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=60),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=1),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
}

# Channels settings - desde variables de entorno
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))

CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            "hosts": [(REDIS_HOST, REDIS_PORT)],
        },
    },
}

# Celery settings - desde variables de entorno
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'django-db')
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = TIME_ZONE
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = 30 * 60  # 30 minutos
CELERY_TASK_SOFT_TIME_LIMIT = 25 * 60  # 25 minutos
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
CELERY_WORKER_MAX_TASKS_PER_CHILD = 1000

# Email settings - desde variables de entorno
EMAIL_BACKEND = os.getenv('EMAIL_BACKEND', 'django.core.mail.backends.smtp.EmailBackend')
EMAIL_HOST = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.getenv('EMAIL_PORT', '587'))
EMAIL_USE_TLS = os.getenv('EMAIL_USE_TLS', 'True').lower() == 'true'
EMAIL_HOST_USER = os.getenv('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = os.getenv('EMAIL_HOST_PASSWORD')
DEFAULT_FROM_EMAIL = os.getenv('DEFAULT_FROM_EMAIL', 'noreply@financialhub.com')

# AI settings - desde variables de entorno
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
AI_MODEL = os.getenv('AI_MODEL', 'gpt-4')
AI_TEMPERATURE = float(os.getenv('AI_TEMPERATURE', '0.7'))
AI_MAX_TOKENS = int(os.getenv('AI_MAX_TOKENS', '2000'))

# AI Authentication Settings for Tests
AI_TEST_ENDPOINTS_AUTH = os.getenv('AI_TEST_ENDPOINTS_AUTH', 'False') == 'True'

# AI Resource Thresholds - desde variables de entorno
AI_CPU_THRESHOLD = float(os.getenv('AI_CPU_THRESHOLD', '80.0'))
AI_MEMORY_THRESHOLD = float(os.getenv('AI_MEMORY_THRESHOLD', '80.0'))
AI_DISK_THRESHOLD = float(os.getenv('AI_DISK_THRESHOLD', '90.0'))
AI_MIN_MEMORY_GB = float(os.getenv('AI_MIN_MEMORY_GB', '1.0'))
AI_MIN_DISK_GB = float(os.getenv('AI_MIN_DISK_GB', '5.0'))

# AI Model Cache Settings - desde variables de entorno
AI_MODEL_CACHE_TIMEOUT = int(os.getenv('AI_MODEL_CACHE_TIMEOUT', '3600'))
AI_MODEL_CACHE_PREFIX = os.getenv('AI_MODEL_CACHE_PREFIX', 'ai_model_')

# AI Model Versioning Settings - desde variables de entorno
AI_MODEL_VERSION_DIR = os.path.join(BASE_DIR, 'ai', 'models', 'versions')
AI_MAX_MODEL_VERSIONS = int(os.getenv('AI_MAX_MODEL_VERSIONS', '5'))

# Asegurar que el directorio de versiones existe
os.makedirs(AI_MODEL_VERSION_DIR, exist_ok=True)

# Asegurar que el directorio de logs existe
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)

# Stripe settings - desde variables de entorno
STRIPE_SECRET_KEY = os.getenv('STRIPE_SECRET_KEY')
STRIPE_PUBLISHABLE_KEY = os.getenv('STRIPE_PUBLISHABLE_KEY')
STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_WEBHOOK_SECRET')
STRIPE_PRICE_ID = os.getenv('STRIPE_PRICE_ID')

# Feature flags
ENABLE_AI_INSIGHTS = True
ENABLE_AUDIT_LOG = True
ENABLE_GOALS = True
ENABLE_CHAT = True
ENABLE_NOTIFICATIONS = True

# Swagger/OpenAPI settings
SWAGGER_SETTINGS = {
    'SECURITY_DEFINITIONS': {
        'Bearer': {
            'type': 'apiKey',
            'name': 'Authorization',
            'in': 'header'
        }
    },
    'USE_SESSION_AUTH': False,
    'JSON_EDITOR': True,
    'OPERATIONS_SORTER': 'alpha',
    'TAGS_SORTER': 'alpha',
    'DOC_EXPANSION': 'none',
    'DEFAULT_MODEL_RENDERING': 'example',
}

# Cache settings - desde variables de entorno
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/1')
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': REDIS_URL,
        'KEY_PREFIX': 'financialhub',
        'TIMEOUT': int(os.getenv('REDIS_CACHE_TIMEOUT', '3600')),
    }
}

# Logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, 'logs/financialhub.log'),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 10,
            'formatter': 'verbose',
        },
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, 'logs/error.log'),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 10,
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True,
        },
        'django.request': {
            'handlers': ['error_file'],
            'level': 'ERROR',
            'propagate': False,
        },
        'django.db.backends': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': False,
        },
        'stripe': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'payments': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'access_control': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
}

# Modelo de usuario personalizado
AUTH_USER_MODEL = 'accounts.User'

# ML Models directory for AI tests and services - desde variables de entorno
ML_MODELS_DIR = os.getenv('AI_MODELS_PATH', os.path.join(BASE_DIR, 'ml_models')) 