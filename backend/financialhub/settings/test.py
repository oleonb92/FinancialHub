"""
Configuración específica para tests
Permite endpoints de IA sin autenticación para facilitar testing
"""

from .base import *

# Configuración de test - DEBE estar al inicio
TESTING = True

SECRET_KEY = 'django-insecure-test-key'

# Configuración para tests
DEBUG = True
AI_TEST_ENDPOINTS_AUTH = False

# Usar base de datos en memoria para tests
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': ':memory:',
    }
}

# Configuración de caché para tests
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'unique-snowflake',
    }
}

# Configuración de Celery para tests
CELERY_TASK_ALWAYS_EAGER = True
CELERY_TASK_EAGER_PROPAGATES = True

# Configuración de logging para tests
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
        'test_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, 'logs', 'test', 'test.log'),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'DEBUG',
    },
    'loggers': {
        'test': {
            'handlers': ['console', 'test_file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'accounts.middleware': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
}

# Configuración de REST Framework para tests
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.TokenAuthentication',
    ),
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.AllowAny',  # Permitir acceso sin autenticación en tests
    ),
    'TEST_REQUEST_DEFAULT_FORMAT': 'json',
}

# Configuración de AI para tests
AI_MODEL = 'gpt-3.5-turbo'  # Modelo más rápido para tests
AI_TEMPERATURE = 0.1  # Menos aleatoriedad para tests consistentes
AI_MAX_TOKENS = 1000  # Menos tokens para tests más rápidos

# Configuración de modelos de ML para tests
ML_MODELS_DIR = os.path.join(BASE_DIR, 'ml_models', 'test')

# Configuración de archivos estáticos para tests
STATICFILES_STORAGE = 'django.contrib.staticfiles.storage.StaticFilesStorage'

# Configuración de media para tests
MEDIA_ROOT = os.path.join(BASE_DIR, 'media', 'test')

# Configuración de CORS para tests
CORS_ALLOW_ALL_ORIGINS = True

# Configuración de seguridad para tests
SECURE_SSL_REDIRECT = False
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False

# Configuración de email para tests
EMAIL_BACKEND = 'django.core.mail.backends.locmem.EmailBackend'

# Configuración de Stripe para tests
STRIPE_SECRET_KEY = 'sk_test_dummy'
STRIPE_PUBLISHABLE_KEY = 'pk_test_dummy'
STRIPE_WEBHOOK_SECRET = 'whsec_dummy'

# Configuración de Redis para tests
REDIS_URL = 'redis://localhost:6379/2'  # Base de datos separada para tests

# Configuración de Channels para tests
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels.layers.InMemoryChannelLayer',
    },
}

# Configuración de Celery para tests
CELERY_BROKER_URL = 'memory://'
CELERY_RESULT_BACKEND = 'rpc://'

# Configuración de Swagger para tests
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

# Configuración de middleware para tests
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'accounts.middleware.OrganizationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# Configuración de apps para tests
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

# Configuración de templates para tests
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

# Configuración de internacionalización para tests
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

# Configuración de archivos estáticos para tests
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles', 'test')
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media', 'test')

# Configuración de modelo de usuario para tests
AUTH_USER_MODEL = 'accounts.User'

# Configuración de URLs para tests
ROOT_URLCONF = 'financialhub.urls'

# Configuración de directorios para tests
os.makedirs(ML_MODELS_DIR, exist_ok=True)
os.makedirs(MEDIA_ROOT, exist_ok=True)
os.makedirs(STATIC_ROOT, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'logs', 'test'), exist_ok=True)

# Configuración de feature flags para tests
ENABLE_AI_INSIGHTS = True
ENABLE_AUDIT_LOG = True
ENABLE_GOALS = True
ENABLE_CHAT = True
ENABLE_NOTIFICATIONS = True

# Configuración de JWT para tests
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=60),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=1),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
    'ALGORITHM': 'HS256',
    'SIGNING_KEY': SECRET_KEY,
    'VERIFYING_KEY': None,
    'AUTH_HEADER_TYPES': ('Bearer',),
    'USER_ID_FIELD': 'id',
    'USER_ID_CLAIM': 'user_id',
    'AUTH_TOKEN_CLASSES': ('rest_framework_simplejwt.tokens.AccessToken',),
    'TOKEN_TYPE_CLAIM': 'token_type',
    'JTI_CLAIM': 'jti',
}

# Configuración de paginación para tests
REST_FRAMEWORK['DEFAULT_PAGINATION_CLASS'] = 'rest_framework.pagination.PageNumberPagination'
REST_FRAMEWORK['PAGE_SIZE'] = 10  # Menos elementos por página para tests

# Configuración de filtros para tests
REST_FRAMEWORK['DEFAULT_FILTER_BACKENDS'] = (
    'django_filters.rest_framework.DjangoFilterBackend',
    'rest_framework.filters.SearchFilter',
    'rest_framework.filters.OrderingFilter',
)

# Configuración de validación de contraseñas para tests
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
        'OPTIONS': {
            'min_length': 6,  # Contraseñas más cortas para tests
        }
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Configuración de hosts permitidos para tests
ALLOWED_HOSTS = ['*']

# Configuración de CSRF para tests
CSRF_TRUSTED_ORIGINS = ['http://localhost:8000', 'http://127.0.0.1:8000']

# Configuración de sesión para tests
SESSION_ENGINE = 'django.contrib.sessions.backends.db'
SESSION_COOKIE_AGE = 1209600  # 2 semanas

# Configuración de archivos temporales para tests
TEMP_DIR = os.path.join(BASE_DIR, 'temp', 'test')
os.makedirs(TEMP_DIR, exist_ok=True)

# Configuración de modelos de IA para tests
AI_MODEL_VERSION_DIR = os.path.join(BASE_DIR, 'ai', 'models', 'versions', 'test')
os.makedirs(AI_MODEL_VERSION_DIR, exist_ok=True)

# Configuración de caché de modelos para tests
AI_MODEL_CACHE_TIMEOUT = 300  # 5 minutos para tests
AI_MODEL_CACHE_PREFIX = 'ai_model_test_'

# Configuración de versiones de modelos para tests
AI_MAX_MODEL_VERSIONS = 3  # Menos versiones para tests

# Configuración de umbrales de recursos para tests
AI_CPU_THRESHOLD = 90.0  # Umbral más alto para tests
AI_MEMORY_THRESHOLD = 90.0
AI_DISK_THRESHOLD = 95.0
AI_MIN_MEMORY_GB = 0.5  # Menos memoria requerida para tests
AI_MIN_DISK_GB = 1.0  # Menos disco requerido para tests 