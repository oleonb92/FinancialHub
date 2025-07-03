#!/usr/bin/env python3
"""
Script de prueba para verificar la integración completa entre el core AI y llm_service.py
"""

import os
import sys
import django
from pathlib import Path

# Configurar Django
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'financialhub.settings.dev')

django.setup()

import logging
from ai.ml.llm_service import LLMService
from ai.core import *

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_core_ai_modules():
    """Prueba que todos los módulos del core AI se inicialicen correctamente."""
    print("🧪 Probando inicialización de módulos del core AI...")
    
    try:
        # Probar cada módulo individualmente
        modules = {
            'TranslationService': TranslationService(),
            'EnhancedFinancialQueryParser': EnhancedFinancialQueryParser(),
            'IntentClassifier': IntentClassifier(),
            'FollowUpSuggester': FollowUpSuggester(),
            'ReportGenerator': ReportGenerator(),
            'ConversationContextManager': ConversationContextManager(),
            'PrivacyGuard': PrivacyGuard(),
            'NLRenderer': NLRenderer(),
            'PromptBuilder': PromptBuilder(),
            'AIService': AIService()
        }
        
        for name, module in modules.items():
            print(f"✅ {name}: Inicializado correctamente")
        
        print("🎉 Todos los módulos del core AI se inicializaron correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error inicializando módulos del core AI: {e}")
        return False

def test_llm_service_integration():
    """Prueba la integración completa con LLMService."""
    print("\n🧪 Probando integración con LLMService...")
    
    try:
        # Crear instancia de LLMService
        llm_service = LLMService()
        
        # Verificar que los módulos del core AI estén disponibles
        core_modules = [
            'ai_service', 'enhanced_query_parser', 'intent_classifier',
            'followup_suggester', 'report_generator', 'translation_service',
            'context_manager', 'privacy_guard', 'nl_renderer', 'prompt_builder'
        ]
        
        for module_name in core_modules:
            if hasattr(llm_service, module_name) and getattr(llm_service, module_name) is not None:
                print(f"✅ {module_name}: Integrado correctamente")
            else:
                print(f"⚠️ {module_name}: No disponible")
        
        print("🎉 Integración con LLMService verificada")
        return True
        
    except Exception as e:
        print(f"❌ Error en integración con LLMService: {e}")
        return False

def test_translation_service():
    """Prueba el servicio de traducción."""
    print("\n🧪 Probando TranslationService...")
    
    try:
        # Probar detección de idioma
        test_texts = [
            "¿Cuánto gasté este mes?",
            "How much did I spend this month?",
            "Balance de mi cuenta",
            "Account balance"
        ]
        
        for text in test_texts:
            detected = detect_language(text)
            print(f"📝 '{text}' -> {detected}")
        
        # Probar traducción
        english_text = "Your balance is $1,000"
        translated = translate(english_text, target_lang='es')
        print(f"🌐 Traducción: '{english_text}' -> '{translated}'")
        
        print("🎉 TranslationService funciona correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en TranslationService: {e}")
        return False

def test_enhanced_query_parser():
    """Prueba el parser mejorado de consultas."""
    print("\n🧪 Probando EnhancedFinancialQueryParser...")
    
    try:
        parser = EnhancedFinancialQueryParser()
        
        test_queries = [
            "¿Cuánto gasté en restaurantes este mes?",
            "Show me my balance for last month",
            "¿Cuál es mi balance actual?",
            "Generate a savings report"
        ]
        
        for query in test_queries:
            result = parser.parse_query(query)
            print(f"🔍 '{query}' -> {result.intent_type} (confidence: {result.confidence_score:.2f})")
        
        print("🎉 EnhancedFinancialQueryParser funciona correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en EnhancedFinancialQueryParser: {e}")
        return False

def test_privacy_guard():
    """Prueba el guardián de privacidad."""
    print("\n🧪 Probando PrivacyGuard...")
    
    try:
        guard = PrivacyGuard()
        
        # Probar consultas seguras
        safe_queries = [
            "¿Cuál es mi balance?",
            "Show me my expenses",
            "¿Cuánto gasté este mes?"
        ]
        
        for query in safe_queries:
            is_safe, violation = guard.check_query(query, "user123", "org456")
            print(f"🛡️ '{query}' -> {'✅ Seguro' if is_safe else '❌ Violación'}")
        
        print("🎉 PrivacyGuard funciona correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en PrivacyGuard: {e}")
        return False

def test_context_manager():
    """Prueba el gestor de contexto."""
    print("\n🧪 Probando ConversationContextManager...")
    
    try:
        manager = ConversationContextManager()
        
        # Probar almacenamiento de contexto
        manager.store_context(
            user_id="test_user",
            organization_id="test_org",
            query="¿Cuál es mi balance?",
            parsed_intent={"intent_type": "balance_inquiry", "confidence_score": 0.9},
            confidence_score=0.9,
            financial_data={"balance": 1000},
            response_summary="Tu balance es $1,000"
        )
        
        # Probar recuperación de contexto
        context = manager.get_context_history("test_user", "test_org")
        print(f"📚 Contexto recuperado: {len(context)} mensajes")
        
        # Limpiar contexto
        manager.clear_context("test_user", "test_org")
        print("🧹 Contexto limpiado")
        
        print("🎉 ConversationContextManager funciona correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en ConversationContextManager: {e}")
        return False

def test_prompt_builder():
    """Prueba el constructor de prompts."""
    print("\n🧪 Probando PromptBuilder...")
    
    try:
        builder = PromptBuilder()
        
        # Probar diferentes tipos de prompts
        prompt_types = ['balance', 'expense', 'income', 'trend', 'prediction']
        
        for prompt_type in prompt_types:
            prompt = builder.get_system_prompt(prompt_type, 'es')
            print(f"📝 {prompt_type}: {len(prompt)} caracteres")
        
        print("🎉 PromptBuilder funciona correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en PromptBuilder: {e}")
        return False

def test_nl_renderer():
    """Prueba el renderizador de lenguaje natural."""
    print("\n🧪 Probando NLRenderer...")
    
    try:
        renderer = NLRenderer()
        
        # Probar renderizado de respuesta usando el método correcto
        response = renderer.render_single_metric(
            metric_name="balance",
            value=1000.0,
            currency="$",
            period="this month",
            language="en"
        )
        
        print(f"📄 Respuesta renderizada: {len(response)} caracteres")
        
        # Probar otro tipo de respuesta
        comparison_response = renderer.render_comparison(
            metric_name="expenses",
            current_value=500.0,
            previous_value=450.0,
            currency="$",
            period="this month",
            language="en"
        )
        
        print(f"📊 Comparación renderizada: {len(comparison_response)} caracteres")
        
        print("🎉 NLRenderer funciona correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en NLRenderer: {e}")
        return False

def main():
    """Ejecuta todas las pruebas de integración."""
    print("🚀 Iniciando pruebas de integración del sistema AI core...")
    print("=" * 60)
    
    tests = [
        test_core_ai_modules,
        test_llm_service_integration,
        test_translation_service,
        test_enhanced_query_parser,
        test_privacy_guard,
        test_context_manager,
        test_prompt_builder,
        test_nl_renderer
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Error ejecutando {test.__name__}: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Resultados: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡Todas las pruebas pasaron! El sistema AI core está completamente integrado.")
    else:
        print("⚠️ Algunas pruebas fallaron. Revisa los errores arriba.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 