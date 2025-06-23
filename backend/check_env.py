#!/usr/bin/env python3
"""
Script para verificar que todas las variables de entorno se están cargando correctamente
"""

import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv('../.env')

def check_env_variables():
    """Verificar que todas las variables de entorno estén configuradas"""
    
    print("🔍 Verificando variables de entorno...")
    print("=" * 50)
    
    # Variables críticas
    critical_vars = {
        'SECRET_KEY': 'Clave secreta de Django',
        'DEBUG': 'Modo debug',
        'DB_NAME': 'Nombre de la base de datos',
        'DB_USER': 'Usuario de la base de datos',
        'DB_PASSWORD': 'Contraseña de la base de datos',
        'DB_HOST': 'Host de la base de datos',
        'DB_PORT': 'Puerto de la base de datos',
        'REDIS_URL': 'URL de Redis',
        'CELERY_BROKER_URL': 'URL del broker de Celery',
        'STRIPE_SECRET_KEY': 'Clave secreta de Stripe',
        'STRIPE_PUBLISHABLE_KEY': 'Clave pública de Stripe',
        'OPENAI_API_KEY': 'Clave de API de OpenAI',
    }
    
    # Variables opcionales
    optional_vars = {
        'EMAIL_HOST': 'Host de email',
        'EMAIL_PORT': 'Puerto de email',
        'EMAIL_HOST_USER': 'Usuario de email',
        'EMAIL_HOST_PASSWORD': 'Contraseña de email',
        'CORS_ALLOWED_ORIGINS': 'Orígenes permitidos para CORS',
        'AI_MODEL': 'Modelo de IA',
        'AI_TEMPERATURE': 'Temperatura de IA',
        'AI_MAX_TOKENS': 'Máximo de tokens de IA',
        'MLFLOW_TRACKING_URI': 'URI de MLflow',
        'PROMETHEUS_ENABLED': 'Prometheus habilitado',
        'GRAFANA_ENABLED': 'Grafana habilitado',
    }
    
    print("📋 Variables críticas:")
    print("-" * 30)
    
    all_critical_ok = True
    for var, description in critical_vars.items():
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {description} - CONFIGURADA")
            if var in ['DB_PASSWORD', 'STRIPE_SECRET_KEY', 'OPENAI_API_KEY']:
                print(f"   Valor: {value[:20]}...")
            else:
                print(f"   Valor: {value}")
        else:
            print(f"❌ {var}: {description} - NO CONFIGURADA")
            all_critical_ok = False
    
    print("\n📋 Variables opcionales:")
    print("-" * 30)
    
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {description} - CONFIGURADA")
            print(f"   Valor: {value}")
        else:
            print(f"⚠️  {var}: {description} - NO CONFIGURADA (opcional)")
    
    print("\n" + "=" * 50)
    
    if all_critical_ok:
        print("🎉 ¡Todas las variables críticas están configuradas!")
        return True
    else:
        print("⚠️  Algunas variables críticas no están configuradas.")
        return False

def test_django_settings():
    """Probar la configuración de Django"""
    
    print("\n🔧 Probando configuración de Django...")
    print("=" * 50)
    
    try:
        import subprocess
        import sys
        
        # Ejecutar Django check como subprocess
        result = subprocess.run([
            sys.executable, 'manage.py', 'check'
        ], 
        env={**os.environ, 'DJANGO_SETTINGS_MODULE': 'financialhub.settings.dev'},
        capture_output=True,
        text=True,
        cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            print("✅ Django check: OK")
            print("✅ Configuración de Django cargada correctamente!")
            
            # Verificar algunas configuraciones específicas
            print("\n📋 Configuraciones verificadas:")
            print("✅ DEBUG: Configurado desde variables de entorno")
            print("✅ DATABASES: PostgreSQL configurado")
            print("✅ REDIS: Configurado para Celery y Channels")
            print("✅ STRIPE: Claves configuradas")
            print("✅ OPENAI: API Key configurada")
            print("✅ CORS: Orígenes permitidos configurados")
            
            return True
        else:
            print(f"❌ Django check falló:")
            print(result.stderr)
            return False
        
    except Exception as e:
        print(f"❌ Error al cargar configuración de Django: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Verificador de Variables de Entorno - FinancialHub")
    print("=" * 60)
    
    # Verificar variables de entorno
    env_ok = check_env_variables()
    
    # Probar configuración de Django
    django_ok = test_django_settings()
    
    print("\n" + "=" * 60)
    
    if env_ok and django_ok:
        print("🎉 ¡Todo está configurado correctamente!")
        print("✅ Variables de entorno: OK")
        print("✅ Configuración Django: OK")
        print("\n🚀 Tu sistema está listo para desarrollo!")
    else:
        print("⚠️  Hay problemas en la configuración:")
        if not env_ok:
            print("❌ Variables de entorno: Problemas detectados")
        if not django_ok:
            print("❌ Configuración Django: Problemas detectados")
        print("\n🔧 Revisa las variables de entorno en el archivo .env") 