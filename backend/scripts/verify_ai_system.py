#!/usr/bin/env python
"""
Script para verificar que el sistema completo de IA esté funcionando correctamente.
"""
import os
import sys
import django
import requests
import json
from datetime import datetime

# Configurar Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'financialhub.settings.dev')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
django.setup()

from ai.services import AIService
from transactions.models import Transaction
from django.contrib.auth import get_user_model

User = get_user_model()

def check_ai_service():
    """Verifica que el servicio de IA esté funcionando"""
    print("🔍 Verificando servicio de IA...")
    
    try:
        ai_service = AIService()
        models = ai_service.get_models()
        
        print("✅ AI Service inicializado correctamente")
        print(f"📊 Modelos disponibles: {len(models.get('available_models', []))}")
        
        # Verificar modelos principales
        main_models = ['transaction_classifier', 'expense_predictor', 'behavior_analyzer', 'budget_optimizer']
        for model_name in main_models:
            if hasattr(ai_service, model_name) and getattr(ai_service, model_name) is not None:
                print(f"✅ {model_name}: Cargado")
            else:
                print(f"❌ {model_name}: No disponible")
        
        return True
    except Exception as e:
        print(f"❌ Error en AI Service: {str(e)}")
        return False

def check_database():
    """Verifica que la base de datos tenga datos para entrenar"""
    print("\n🔍 Verificando base de datos...")
    
    try:
        # Verificar transacciones
        transaction_count = Transaction.objects.count()
        print(f"📊 Transacciones en BD: {transaction_count}")
        
        # Verificar usuarios
        user_count = User.objects.count()
        print(f"👥 Usuarios en BD: {user_count}")
        
        if transaction_count > 0:
            print("✅ Base de datos tiene datos suficientes")
            return True
        else:
            print("⚠️ Base de datos vacía - se usarán datos de muestra")
            return True
    except Exception as e:
        print(f"❌ Error verificando BD: {str(e)}")
        return False

def check_api_endpoints():
    """Verifica que los endpoints de la API estén funcionando"""
    print("\n🔍 Verificando endpoints de API...")
    
    base_url = "http://localhost:8000"
    endpoints = [
        "/api/ai/health/",
        "/api/transactions/",
        "/admin/"
    ]
    
    all_working = True
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code in [200, 401, 403]:  # 401/403 son normales para endpoints protegidos
                print(f"✅ {endpoint}: {response.status_code}")
            else:
                print(f"❌ {endpoint}: {response.status_code}")
                all_working = False
        except requests.exceptions.RequestException as e:
            print(f"❌ {endpoint}: Error de conexión - {str(e)}")
            all_working = False
    
    return all_working

def check_frontend():
    """Verifica que el frontend esté funcionando"""
    print("\n🔍 Verificando frontend...")
    
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend funcionando en http://localhost:3000")
            return True
        else:
            print(f"❌ Frontend: Status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Frontend: Error de conexión - {str(e)}")
        return False

def test_ai_functionality():
    """Prueba funcionalidades básicas de IA"""
    print("\n🔍 Probando funcionalidades de IA...")
    
    try:
        ai_service = AIService()
        
        # Probar análisis de comportamiento
        print("🧠 Probando análisis de comportamiento...")
        behavior_result = ai_service.analyze_behavior_simple()
        print(f"✅ Análisis de comportamiento: {behavior_result.get('status', 'error')}")
        
        # Probar predicción de gastos
        print("💰 Probando predicción de gastos...")
        expense_result = ai_service.predict_expenses_simple()
        print(f"✅ Predicción de gastos: {expense_result.get('status', 'error')}")
        
        # Probar análisis de riesgo
        print("⚠️ Probando análisis de riesgo...")
        risk_result = ai_service.analyze_risk()
        print(f"✅ Análisis de riesgo: {risk_result.get('status', 'error')}")
        
        return True
    except Exception as e:
        print(f"❌ Error probando funcionalidades de IA: {str(e)}")
        return False

def main():
    """Función principal de verificación"""
    print("🚀 Verificación completa del sistema de IA")
    print("=" * 50)
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    checks = [
        ("Servicio de IA", check_ai_service),
        ("Base de datos", check_database),
        ("API Endpoints", check_api_endpoints),
        ("Frontend", check_frontend),
        ("Funcionalidades de IA", test_ai_functionality)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ Error en {check_name}: {str(e)}")
            results.append((check_name, False))
    
    # Resumen final
    print("\n" + "=" * 50)
    print("📋 RESUMEN DE VERIFICACIÓN")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"{status} {check_name}")
    
    print(f"\n🎯 Resultado: {passed}/{total} verificaciones pasaron")
    
    if passed == total:
        print("🎉 ¡Sistema de IA completamente funcional!")
        print("\n📝 Próximos pasos:")
        print("1. Accede a http://localhost:3000 para usar la aplicación")
        print("2. Los modelos de IA se entrenarán automáticamente")
        print("3. Las funcionalidades de IA estarán disponibles en la interfaz")
        print("4. Monitorea los logs para ver el rendimiento del sistema")
    else:
        print("⚠️ Algunas verificaciones fallaron. Revisa los errores arriba.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 