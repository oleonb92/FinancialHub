# FinancialHub AI Core System

A comprehensive, modular AI backend system for financial analysis and chatbot functionality. This system provides privacy protection, context management, natural language rendering, and intelligent query processing.

## 🏗️ Architecture Overview

The AI Core System consists of five main modules:

1. **PrivacyGuard** - Protects against privacy violations and inappropriate requests
2. **ConversationContextManager** - Manages user session context and resolves follow-up questions
3. **NLRenderer** - Renders structured responses using Jinja2 templates with i18n support
4. **PromptBuilder** - Assembles system prompts, user history, and orchestration results
5. **FinancialQueryParser** - Parses financial queries with confidence scoring and fallback logic

## 📁 File Structure

```
backend/ai/core/
├── __init__.py                 # Core module exports
├── privacy_guard.py           # Privacy protection module
├── context_manager.py         # Conversation context management
├── nl_renderer.py            # Natural language response rendering
├── prompt_builder.py         # Prompt assembly and management
├── ai_service.py             # Main integration service
├── tests/                    # Unit tests
│   ├── __init__.py
│   ├── test_privacy_guard.py
│   ├── test_context_manager.py
│   └── test_nl_renderer.py
└── README.md                 # This documentation
```

## 🚀 Quick Start

### Basic Usage

```python
from ai.core.ai_service import AIService

# Initialize the AI service
ai_service = AIService()

# Process a user query
response = ai_service.process_query(
    user_query="What is my current balance?",
    user_id="user_123",
    organization_id="org_456",
    financial_data={
        'summary': {
            'total_income': 5000.0,
            'total_expenses': 3000.0,
            'net_balance': 2000.0
        }
    }
)

print(response.response_text)
print(f"Confidence: {response.confidence_score}")
print(f"Intent: {response.intent_type}")
```

### Privacy Protection

```python
# Check if a query violates privacy rules
is_safe, violation = ai_service.check_privacy(
    query="Show me other users' data",
    user_id="user_123",
    organization_id="org_456"
)

if not is_safe:
    print(f"Privacy violation: {violation.reason}")
```

### Context Management

```python
# Get conversation history
history = ai_service.get_context_history("user_123", "org_456", limit=5)

# Clear context
ai_service.clear_context("user_123", "org_456")
```

## 🔧 Core Modules

### 1. PrivacyGuard

Protects against privacy violations and inappropriate requests.

**Features:**
- Keyword-based violation detection
- Regex pattern matching
- Context-aware permission checking
- Multi-language violation messages
- Configurable severity levels

**Violation Types:**
- `OTHER_USER_DATA` - Attempts to access other users' data
- `INVESTMENT_ADVICE` - Requests for investment advice
- `PERSONAL_INFO_REQUEST` - Requests for personal information
- `SENSITIVE_FINANCIAL_DATA` - Requests for sensitive financial data
- `UNAUTHORIZED_ACCESS` - Unauthorized access attempts
- `COMPETITIVE_INTELLIGENCE` - Competitive intelligence requests

**Usage:**
```python
from ai.core.privacy_guard import PrivacyGuard

guard = PrivacyGuard()
is_safe, violation = guard.check_query(
    query="What is my balance?",
    user_id="user_123",
    organization_id="org_456"
)
```

### 2. ConversationContextManager

Manages user session context and resolves follow-up questions.

**Features:**
- Session-based context storage using Django cache
- Follow-up question detection and resolution
- Time reference resolution ("last month", "yesterday")
- Context serialization and deserialization
- Automatic context expiration

**Usage:**
```python
from ai.core.context_manager import ConversationContextManager

context_manager = ConversationContextManager()

# Store context
context_manager.store_context(
    user_id="user_123",
    organization_id="org_456",
    query="What is my balance?",
    parsed_intent={'intent_type': 'balance_inquiry'},
    confidence_score=0.85
)

# Resolve follow-up
resolved = context_manager.resolve_follow_up(
    user_id="user_123",
    organization_id="org_456",
    current_query="What about last month?"
)
```

### 3. NLRenderer

Renders structured responses using Jinja2 templates with internationalization support.

**Features:**
- Jinja2 template engine
- Bilingual support (English/Spanish)
- Pre-built templates for common response types
- Automatic template creation
- Fallback error handling

**Template Types:**
- `single_metric` - Single financial metric display
- `comparison` - Period-over-period comparisons
- `goal_progress` - Goal tracking and progress
- `anomaly_alert` - Anomaly detection alerts
- `savings_recommendation` - Savings plan recommendations
- `privacy_refusal` - Privacy violation responses
- `clarification_request` - Clarification questions

**Usage:**
```python
from ai.core.nl_renderer import NLRenderer

renderer = NLRenderer()

# Render a single metric
response = renderer.render_single_metric(
    metric_name="Balance",
    value=2500.0,
    currency="$",
    period="this month",
    language="en"
)

# Render a comparison
response = renderer.render_comparison(
    metric_name="Expenses",
    current_value=3000.0,
    previous_value=2500.0,
    currency="$",
    language="en"
)
```

### 4. PromptBuilder

Assembles system prompts, user history, and orchestration results into concise prompts for LLM.

**Features:**
- Multiple prompt types for different scenarios
- Bilingual system prompts
- Financial context formatting
- Token limit management
- Intelligent prompt truncation

**Prompt Types:**
- `default` - General financial assistance
- `financial_analysis` - Spending and income analysis
- `savings_planning` - Savings plan creation
- `anomaly_detection` - Anomaly detection and explanation
- `goal_tracking` - Goal progress monitoring
- `clarification` - Clarification requests

**Usage:**
```python
from ai.core.prompt_builder import PromptBuilder, PromptComponents

builder = PromptBuilder()

components = PromptComponents(
    system_prompt=builder.get_system_prompt('financial_analysis', 'en'),
    user_query="What is my balance?",
    financial_context="💰 FINANCIAL SUMMARY:\n  • Total income: $5,000.00",
    language="en"
)

prompt = builder.build_prompt(components)
```

### 5. FinancialQueryParser (Enhanced)

Parses financial queries with confidence scoring and fallback logic.

**Features:**
- Softmax-based confidence scoring
- Intent classification with weights
- Entity extraction (time, categories, amounts)
- Language detection
- Complexity scoring
- Fallback logic for low confidence

**Intent Types:**
- `balance_inquiry` - Balance and account queries
- `spending_analysis` - Spending pattern analysis
- `savings_planning` - Savings plan requests
- `trend_analysis` - Trend and pattern analysis
- `anomaly_detection` - Anomaly detection queries
- `goal_tracking` - Goal progress queries
- `comparison` - Comparative analysis
- `prediction` - Future predictions

**Usage:**
```python
from ai.core.ml.query_parser import FinancialQueryParser

parser = FinancialQueryParser()
parsed_intent = parser.parse_query("What is my balance for this month?")

print(f"Intent: {parsed_intent.intent_type}")
print(f"Confidence: {parsed_intent.confidence_score}")
print(f"Entities: {parsed_intent.entities}")
```

## 🚀 New Intelligent Modules & Enhancements (2024)

### 1. IntentClassifier
- **Descripción:** Clasificador de intenciones basado en embeddings de SentenceTransformers y regresión logística.
- **Características:**
  - Entrenamiento supervisado, predicción, guardado y carga de modelos.
  - Soporte multilingüe (inglés y español).
  - Reemplaza el matching por palabras clave.
- **Uso:**
```python
from ai.core.intent_classifier import IntentClassifier
classifier = IntentClassifier(language='en')
classifier.train_intent_classifier(dataset)
intent, confidence = classifier.predict_intent("Show me my balance")
```

### 2. FollowUpSuggester
- **Descripción:** Generador inteligente de preguntas de seguimiento basadas en la intención detectada y contexto.
- **Características:**
  - Sugerencias contextuales (2-3) tras cada consulta.
  - Plantillas bilingües y memoria de sugerencias recientes.
- **Uso:**
```python
from ai.core.followup_suggester import FollowUpSuggester
suggester = FollowUpSuggester(language='es')
suggestions = suggester.generate_followup_suggestions('balance_inquiry')
```

### 3. ReportGenerator
- **Descripción:** Generador de reportes financieros mensuales con visualizaciones (Plotly/Matplotlib) y salida HTML/PDF usando Jinja2.
- **Características:**
  - Plantillas automáticas, soporte de temas, generación de datos de ejemplo.
  - Exportación a HTML y PDF.
- **Uso:**
```python
from ai.core.report_generator import ReportGenerator, ReportData
report_gen = ReportGenerator()
data = report_gen.create_sample_data()
html = report_gen.generate_monthly_report(data)
```

### 4. EnhancedFinancialQueryParser
- **Descripción:** Parser avanzado con spaCy y dateparser para extracción de entidades y fechas relativas.
- **Características:**
  - Extracción robusta de entidades, fechas, montos, categorías y cuentas.
  - Detección de idioma y scoring de confianza.
- **Uso:**
```python
from ai.core.enhanced_query_parser import EnhancedFinancialQueryParser
parser = EnhancedFinancialQueryParser(language='en')
parsed = parser.parse_query("How much did I spend last month?")
```

### 5. AIQueryLog (Django Model)
- **Descripción:** Modelo para registrar todas las consultas, intenciones, scores, plantillas y timestamps en PostgreSQL.
- **Características:**
  - Métodos para analítica, entrenamiento y dashboarding.
  - Registro de metadatos, entidades, usuario, organización y errores.

### 6. TranslationService (i18n)
- **Descripción:** Servicio centralizado de traducción usando deep-translator para traducciones en tiempo de ejecución.
- **Características:**
  - Traducciones automáticas con Google Translate API, detección de idioma, formateo de moneda, fechas y números.
  - Cache de traducciones, fallback a inglés, soporte para inglés y español.
- **Uso:**
```python
from ai.core.translation_service import TranslationService, translate
service = TranslationService()
text = service.translate('Hello world', target_lang='es')
# Or use convenience function
text = translate('Hello world', target_lang='es')
```

### 7. Unit Tests
- **Descripción:** Tests exhaustivos para todos los módulos nuevos en `backend/ai/unit_tests/test_enhanced_modules.py`.
- **Cobertura:**
  - Inicialización, funcionalidad, integración, soporte multilingüe y edge cases.

## 🧩 Integración y Ejemplo de Flujo

```python
from ai.core.intent_classifier import IntentClassifier
from ai.core.followup_suggester import FollowUpSuggester
from ai.core.report_generator import ReportGenerator
from ai.core.enhanced_query_parser import EnhancedFinancialQueryParser
from ai.core.translation_service import TranslationService, translate

# Clasificación de intención
classifier = IntentClassifier(language='en')
intent, confidence = classifier.predict_intent("Show me my balance")

# Sugerencias de seguimiento
suggester = FollowUpSuggester(language='en')
suggestions = suggester.generate_followup_suggestions(intent)

# Parsing avanzado
parser = EnhancedFinancialQueryParser(language='en')
parsed = parser.parse_query("How much did I spend last month?")

# Generación de reporte
report_gen = ReportGenerator()
data = report_gen.create_sample_data()
html = report_gen.generate_monthly_report(data)

# Traducción
service = TranslationService()
text = service.translate('Hello world', target_lang='es')
# Or use convenience function
text = translate('Hello world', target_lang='es')
```

## 🧪 Testing

Run the unit tests:

```bash
# Run all tests
python manage.py test ai.core.tests

# Run specific test modules
python manage.py test ai.core.tests.test_privacy_guard
python manage.py test ai.core.tests.test_context_manager
python manage.py test ai.core.tests.test_nl_renderer
```

### Test Coverage

- **PrivacyGuard**: Tests for all violation types, regex patterns, and error handling
- **ContextManager**: Tests for context storage, retrieval, follow-up resolution, and error handling
- **NLRenderer**: Tests for template rendering, bilingual support, and error handling

## 🔄 Integration with Existing System

### Integration with LLM Service

The AI Core System is designed to integrate with your existing LLM service:

```python
# In your existing LLM service
from ai.core.ai_service import AIService

class EnhancedLLMService:
    def __init__(self):
        self.ai_service = AIService()
        # ... existing initialization
    
    def process_chat_message(self, message, user_id, organization_id, financial_data):
        # Use the AI service for processing
        ai_response = self.ai_service.process_query(
            user_query=message,
            user_id=user_id,
            organization_id=organization_id,
            financial_data=financial_data
        )
        
        # Handle different response types
        if ai_response.privacy_violation:
            return ai_response.response_text
        
        if ai_response.requires_clarification:
            return ai_response.response_text
        
        # Use existing LLM for final response generation
        final_response = self._generate_llm_response(ai_response.response_text)
        return final_response
```

### Template Customization

Customize templates by creating new Jinja2 files in the templates directory:

```jinja2
<!-- custom_template.jinja2 -->
{% if language == 'es' %}
🎯 **{{ title }}**
{{ content }}
{% else %}
🎯 **{{ title }}**
{{ content }}
{% endif %}
```

## 🔧 Configuration

### Django Settings

Add to your Django settings:

```python
# AI Core Configuration
AI_CORE_CONFIG = {
    'CACHE_TIMEOUT': 3600,  # Context cache timeout in seconds
    'TEMPLATE_DIR': 'ai/templates',  # Template directory
    'DEFAULT_LANGUAGE': 'en',  # Default language
    'CONFIDENCE_THRESHOLD': 0.3,  # Minimum confidence for processing
}
```

### Cache Configuration

Ensure Redis or Memcached is configured for context storage:

```python
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
    }
}
```

## 🚀 Deployment

### Requirements

- Python 3.10+
- Django 4.0+
- Jinja2
- Redis (for caching)
- deep-translator>=1.11.4 (for runtime translations)
- langdetect>=1.0.9 (for language detection)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install spaCy models (for enhanced query parsing)
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm

# Run migrations (if needed)
python manage.py migrate

# Run tests
pytest backend/ai/unit_tests/
```

## 📊 Performance Considerations

- **Caching**: Context is cached using Django's cache framework
- **Template Caching**: Jinja2 templates are compiled and cached
- **Token Limits**: Prompts are automatically truncated to fit token limits
- **Error Handling**: Graceful fallbacks for all error conditions

## 🔒 Security Features

- **Privacy Protection**: Comprehensive privacy violation detection
- **Permission Checking**: Context-aware permission validation
- **Input Validation**: Robust input validation and sanitization
- **Error Handling**: Secure error handling without information leakage

## 🌐 Internationalization

The system supports both English and Spanish using the TranslationService:

- **Runtime Translation**: Uses deep-translator with Google Translate API
- **Automatic Language Detection**: Based on query content using langdetect
- **Translation Caching**: Avoids repeated API calls for the same translations
- **Fallback to English**: All unsupported languages default to English
- **Formatting Support**: Currency, numbers, and dates formatted per locale

## 🔄 Extensibility

The system is designed to be easily extensible:

- **New Intent Types**: Add new intent patterns to the query parser
- **Custom Templates**: Create new Jinja2 templates for specific use cases
- **Privacy Rules**: Add new privacy violation types and rules
- **Prompt Types**: Extend the prompt builder with new prompt types

## 📝 Contributing

1. Follow the existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for any changes
4. Ensure all tests pass before submitting

## 📄 License

This module is part of the FinancialHub project and follows the same licensing terms. 