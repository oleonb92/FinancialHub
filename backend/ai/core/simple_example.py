#!/usr/bin/env python3
"""
Ejemplo simple del AI Core System
Demuestra el uso de los módulos principales sin dependencias de Django
"""

import sys
import os
from pathlib import Path

# Agregar el directorio padre al path para importar los módulos
sys.path.insert(0, str(Path(__file__).parent))

# Importar los módulos del AI Core
from privacy_guard import PrivacyGuard
from context_manager import ConversationContextManager
from nl_renderer import NLRenderer
from prompt_builder import PromptBuilder

def main():
    print("🤖 AI Core System - Ejemplo de Uso")
    print("=" * 50)
    
    # 1. Privacy Guard
    print("\n🔒 1. Privacy Guard")
    print("-" * 30)
    
    privacy_guard = PrivacyGuard()
    
    # Ejemplos de consultas
    test_queries = [
        "¿Cuál es mi saldo actual?",
        "¿Cuánto gastó Juan el mes pasado?",
        "¿Qué acciones debo comprar?",
        "¿Cuál es la información de la cuenta 12345?",
        "¿Cuánto gasté en comida este mes?"
    ]
    
    for query in test_queries:
        is_safe, violation = privacy_guard.check_query(
            query, 
            user_id="1", 
            organization_id="1",
            user_permissions=["user"]
        )
        status = "❌ BLOQUEADO" if not is_safe else "✅ PERMITIDO"
        print(f"{status}: {query}")
        if not is_safe and violation:
            print(f"   Razón: {violation.reason}")
    
    # 2. Context Manager
    print("\n💬 2. Context Manager")
    print("-" * 30)
    
    context_manager = ConversationContextManager()
    user_id = 123
    
    # Simular una conversación
    context_data = {
        "last_query": "¿Cuánto gasté en comida este mes?",
        "response_summary": "Gastaste $450 en comida este mes",
        "financial_data": {
            "total_spent": 450,
            "category": "food",
            "period": "this_month"
        }
    }
    
    # Guardar contexto
    context_manager.store_context(
        user_id,
        organization_id="1",
        query=context_data["last_query"],
        parsed_intent={"type": "spending_query"},
        confidence_score=0.95,
        financial_data=context_data["financial_data"],
        response_summary=context_data["response_summary"]
    )
    print(f"✅ Contexto guardado para usuario {user_id}")
    
    # Resolver follow-up
    follow_up_queries = [
        "¿Y el mes pasado?",
        "¿Cuánto gasté en total?",
        "¿Cuál es mi saldo actual?"
    ]
    
    for query in follow_up_queries:
        resolved = context_manager.resolve_follow_up(user_id, "1", query)
        print(f"Consulta: {query}")
        print(f"Resuelta: {resolved}")
    
    # 3. NL Renderer
    print("\n📝 3. NL Renderer")
    print("-" * 30)
    
    renderer = NLRenderer()
    
    # Ejemplo de métrica única
    single_metric_data = {
        "metric_name": "Gastos Mensuales",
        "value": 1250.50,
        "currency": "USD",
        "change": 15.5,
        "period": "este mes"
    }
    
    response = renderer.render_response("single_metric", single_metric_data, "es")
    print("📊 Métrica única:")
    print(response)
    
    # Ejemplo de comparación
    comparison_data = {
        "metric_name": "Gastos",
        "current_value": 1250.50,
        "previous_value": 1100.00,
        "currency": "USD",
        "period": "este mes vs mes pasado"
    }
    
    response = renderer.render_response("comparison", comparison_data, "es")
    print("\n📈 Comparación:")
    print(response)
    
    # 4. Prompt Builder
    print("\n🔧 4. Prompt Builder")
    print("-" * 30)
    
    prompt_builder = PromptBuilder()
    
    # Ejemplo de prompt para análisis financiero
    system_prompt = "Eres un asistente financiero experto."
    user_history = [
        {"role": "user", "content": "¿Cuánto gasté este mes?"},
        {"role": "assistant", "content": "Gastaste $1250.50 este mes."}
    ]
    orchestration_results = {
        "query_type": "spending_analysis",
        "confidence": 0.95,
        "data_points": ["monthly_spending", "category_breakdown"]
    }
    
    prompt = prompt_builder.build_prompt(
        prompt_type="financial_analysis",
        system_prompt=system_prompt,
        user_history=user_history,
        orchestration_results=orchestration_results,
        max_tokens=1000
    )
    
    print("📋 Prompt generado:")
    print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
    
    print("\n✅ Ejemplo completado exitosamente!")
    print("\n🎯 Próximos pasos:")
    print("1. Integrar con el chatbot existente")
    print("2. Configurar Redis para caché")
    print("3. Ejecutar tests unitarios")
    print("4. Personalizar templates según necesidades")

if __name__ == "__main__":
    main() 