from .settings import *

# Usar la base de datos de prueba específica
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'financialhub',
        'USER': 'oleonb',
        'PASSWORD': 'Natali@rca1992',
        'HOST': 'localhost',
        'PORT': '5432',
        'TEST': {
            'NAME': 'financialhub',  # Fuerza a usar la base real
        },
    }
}

# Deshabilitar los hashers de contraseña para pruebas más rápidas
PASSWORD_HASHERS = [
    'django.contrib.auth.hashers.MD5PasswordHasher',
]

# Deshabilitar el logging durante las pruebas
LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'handlers': {
        'null': {
            'class': 'logging.NullHandler',
        },
    },
    'root': {
        'handlers': ['null'],
        'level': 'CRITICAL',
    },
} 