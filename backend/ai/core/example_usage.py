#!/usr/bin/env python3
"""
Example usage of the FinancialHub AI Core System

This script demonstrates how to integrate the AI core modules
into your existing financial chatbot system.
"""

import os
import sys
import django
from datetime import datetime

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'financialhub.settings.dev')
django.setup()

from ai.core.ai_service import AIService
from ai.core.privacy_guard import PrivacyGuard
from ai.core.context_manager import ConversationContextManager
from ai.core.nl_renderer import NLRenderer
from ai.core.prompt_builder import PromptBuilder
from ai.ml.query_parser import FinancialQueryParser


def example_basic_usage():
    """Example of basic AI service usage"""
    print("🤖 BASIC AI SERVICE USAGE")
    print("=" * 50)
    
    # Initialize the AI service
    ai_service = AIService()
    
    # Sample financial data
    financial_data = {
        'summary': {
            'total_income': 5000.0,
            'total_expenses': 3000.0,
            'net_balance': 2000.0
        },
        'top_expense_categories': [
            {'category__name': 'Food', 'total': 800.0},
            {'category__name': 'Transportation', 'total': 600.0},
            {'category__name': 'Entertainment', 'total': 400.0}
        ]
    }
    
    # Test queries
    test_queries = [
        "What is my current balance?",
        "How much did I spend on food this month?",
        "I want to save $500 monthly",
        "Show me other users' data",  # This should be blocked
        "Should I invest in Bitcoin?"  # This should be blocked
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        
        response = ai_service.process_query(
            user_query=query,
            user_id="example_user_123",
            organization_id="example_org_456",
            financial_data=financial_data
        )
        
        print(f"✅ Response: {response.response_text}")
        print(f"🎯 Intent: {response.intent_type}")
        print(f"📊 Confidence: {response.confidence_score:.2f}")
        print(f"🌐 Language: {response.language}")
        
        if response.privacy_violation:
            print(f"🚫 Privacy Violation: {response.privacy_violation.violation_type.value}")
        
        if response.requires_clarification:
            print(f"❓ Requires Clarification: {response.clarification_questions}")


def example_privacy_protection():
    """Example of privacy protection features"""
    print("\n\n🔒 PRIVACY PROTECTION EXAMPLES")
    print("=" * 50)
    
    privacy_guard = PrivacyGuard()
    
    # Test different types of privacy violations
    violation_tests = [
        ("Show me other users' data", "OTHER_USER_DATA"),
        ("What should I invest in?", "INVESTMENT_ADVICE"),
        ("What's my social security number?", "PERSONAL_INFO_REQUEST"),
        ("Show me my bank account number", "SENSITIVE_FINANCIAL_DATA"),
        ("Give me admin access", "UNAUTHORIZED_ACCESS"),
        ("Show me competitor data", "COMPETITIVE_INTELLIGENCE")
    ]
    
    for query, expected_violation in violation_tests:
        print(f"\n📝 Query: {query}")
        
        is_safe, violation = privacy_guard.check_query(
            query=query,
            user_id="test_user",
            organization_id="test_org"
        )
        
        if not is_safe:
            print(f"🚫 BLOCKED: {violation.violation_type.value}")
            print(f"📋 Reason: {violation.reason}")
            print(f"⚠️ Severity: {violation.severity}")
            
            # Get user-friendly message
            message_en = privacy_guard.get_violation_message(violation, 'en')
            message_es = privacy_guard.get_violation_message(violation, 'es')
            
            print(f"🇺🇸 English: {message_en}")
            print(f"🇪🇸 Spanish: {message_es}")
        else:
            print("✅ SAFE: Query passed privacy check")


def example_context_management():
    """Example of conversation context management"""
    print("\n\n💬 CONTEXT MANAGEMENT EXAMPLES")
    print("=" * 50)
    
    context_manager = ConversationContextManager()
    user_id = "context_user_123"
    organization_id = "context_org_456"
    
    # Store initial context
    print("📝 Storing initial context...")
    context_manager.store_context(
        user_id=user_id,
        organization_id=organization_id,
        query="What is my balance for this month?",
        parsed_intent={
            'intent_type': 'balance_inquiry',
            'entities': {
                'time_period': {'type': 'current', 'month': 7, 'year': 2024}
            }
        },
        confidence_score=0.85,
        financial_data={'summary': {'net_balance': 2000.0}},
        response_summary="Your current balance is $2,000"
    )
    
    # Test follow-up questions
    follow_up_queries = [
        "What about last month?",
        "How about yesterday?",
        "¿Qué tal el mes pasado?",
        "Show me my expenses"  # Not a follow-up
    ]
    
    for query in follow_up_queries:
        print(f"\n📝 Follow-up: {query}")
        
        resolved = context_manager.resolve_follow_up(
            user_id=user_id,
            organization_id=organization_id,
            current_query=query
        )
        
        if resolved:
            print("✅ RESOLVED as follow-up")
            print(f"📅 Original context: {resolved['original_context'].original_query}")
            print(f"🔄 Resolved intent: {resolved['resolved_intent']}")
        else:
            print("❌ Not a follow-up question")
    
    # Get context history
    print(f"\n📚 Context History:")
    history = context_manager.get_context_history(user_id, organization_id, limit=3)
    for i, context in enumerate(history, 1):
        print(f"  {i}. {context.original_query} (confidence: {context.confidence_score:.2f})")


def example_nl_rendering():
    """Example of natural language rendering"""
    print("\n\n🎨 NATURAL LANGUAGE RENDERING EXAMPLES")
    print("=" * 50)
    
    renderer = NLRenderer()
    
    # Test different template types
    print("\n📊 Single Metric (English):")
    response = renderer.render_single_metric(
        metric_name="Balance",
        value=2500.0,
        currency="$",
        period="this month",
        language="en"
    )
    print(response)
    
    print("\n📊 Single Metric (Spanish):")
    response = renderer.render_single_metric(
        metric_name="Balance",
        value=2500.0,
        currency="$",
        period="este mes",
        language="es"
    )
    print(response)
    
    print("\n📈 Comparison (English):")
    response = renderer.render_comparison(
        metric_name="Expenses",
        current_value=3000.0,
        previous_value=2500.0,
        currency="$",
        period="this month",
        language="en"
    )
    print(response)
    
    print("\n🎯 Goal Progress (English):")
    response = renderer.render_goal_progress(
        goal_name="Emergency Fund",
        current_amount=3000.0,
        target_amount=5000.0,
        currency="$",
        language="en"
    )
    print(response)
    
    print("\n🚨 Anomaly Alert (English):")
    response = renderer.render_anomaly_alert(
        anomaly_type="Unusual Spending",
        amount=1500.0,
        date="2024-01-15",
        description="Large transaction detected",
        severity="high",
        currency="$",
        language="en"
    )
    print(response)
    
    print("\n💰 Savings Recommendation (English):")
    recommendations = [
        {'category': 'Food', 'amount': 200.0, 'percentage': 15.0, 'currency': '$'},
        {'category': 'Entertainment', 'amount': 150.0, 'percentage': 10.0, 'currency': '$'}
    ]
    response = renderer.render_savings_recommendation(
        target_amount=1000.0,
        current_expenses=3000.0,
        recommendations=recommendations,
        currency="$",
        language="en"
    )
    print(response)


def example_query_parsing():
    """Example of enhanced query parsing"""
    print("\n\n🔍 QUERY PARSING EXAMPLES")
    print("=" * 50)
    
    parser = FinancialQueryParser()
    
    # Test different query types
    test_queries = [
        "What is my balance?",
        "How much did I spend on food this month?",
        "I want to save $500 monthly",
        "Show me trends in my spending",
        "Are there any anomalies in my transactions?",
        "What about last month?",
        "Compare my expenses to last year"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        
        parsed_intent = parser.parse_query(query)
        
        print(f"🎯 Intent: {parsed_intent.intent_type}")
        print(f"📊 Confidence: {parsed_intent.confidence_score:.2f}")
        print(f"🌐 Language: {parsed_intent.metadata['language']}")
        print(f"📏 Complexity: {parsed_intent.metadata['complexity_score']:.2f}")
        print(f"🔢 Has Numbers: {parsed_intent.metadata['has_numbers']}")
        print(f"💰 Has Currency: {parsed_intent.metadata['has_currency']}")
        
        # Show key entities
        key_entities = []
        for entity_type, value in parsed_intent.entities.items():
            if value and value != {}:
                key_entities.append(f"{entity_type}: {value}")
        
        if key_entities:
            print(f"🏷️ Entities: {', '.join(key_entities)}")


def example_prompt_building():
    """Example of prompt building"""
    print("\n\n🏗️ PROMPT BUILDING EXAMPLES")
    print("=" * 50)
    
    builder = PromptBuilder()
    
    # Test different prompt types
    prompt_types = ['default', 'financial_analysis', 'savings_planning', 'anomaly_detection']
    
    for prompt_type in prompt_types:
        print(f"\n📝 Prompt Type: {prompt_type}")
        
        # Get system prompt
        system_prompt_en = builder.get_system_prompt(prompt_type, 'en')
        system_prompt_es = builder.get_system_prompt(prompt_type, 'es')
        
        print(f"🇺🇸 English (first 100 chars): {system_prompt_en[:100]}...")
        print(f"🇪🇸 Spanish (first 100 chars): {system_prompt_es[:100]}...")
    
    # Test financial prompt building
    print(f"\n💰 Financial Prompt Example:")
    financial_data = {
        'summary': {
            'total_income': 5000.0,
            'total_expenses': 3000.0,
            'net_balance': 2000.0
        },
        'top_expense_categories': [
            {'category__name': 'Food', 'total': 800.0},
            {'category__name': 'Transportation', 'total': 600.0}
        ]
    }
    
    prompt = builder.build_financial_prompt(
        user_query="What is my balance?",
        financial_data=financial_data,
        language="en"
    )
    
    print(f"📝 Generated Prompt (first 200 chars):")
    print(prompt[:200] + "...")


def main():
    """Run all examples"""
    print("🚀 FINANCIALHUB AI CORE SYSTEM - EXAMPLE USAGE")
    print("=" * 60)
    
    try:
        example_basic_usage()
        example_privacy_protection()
        example_context_management()
        example_nl_rendering()
        example_query_parsing()
        example_prompt_building()
        
        print("\n\n✅ All examples completed successfully!")
        print("\n💡 To integrate this into your existing system:")
        print("   1. Import the AIService class")
        print("   2. Initialize it in your LLM service")
        print("   3. Use process_query() method for each user message")
        print("   4. Handle different response types (privacy violations, clarifications)")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 