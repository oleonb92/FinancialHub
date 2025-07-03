"""
Natural Language Renderer for FinancialHub AI
Uses Jinja2 templates to render structured responses from AI orchestration results
"""

import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template
from django.conf import settings

logger = logging.getLogger(__name__)


class NLRenderer:
    """
    Renders structured responses using Jinja2 templates
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize the NL renderer
        
        Args:
            template_dir: Directory containing templates (defaults to ai/templates)
        """
        if template_dir is None:
            template_dir = os.path.join(settings.BASE_DIR, 'ai', 'templates')
        
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Create default templates if they don't exist
        self._create_default_templates()
    
    def _create_default_templates(self):
        """Create default templates if they don't exist"""
        templates = {
            'single_metric': self._get_single_metric_template(),
            'comparison': self._get_comparison_template(),
            'goal_progress': self._get_goal_progress_template(),
            'anomaly_alert': self._get_anomaly_alert_template(),
            'savings_recommendation': self._get_savings_recommendation_template(),
            'privacy_refusal': self._get_privacy_refusal_template(),
            'clarification_request': self._get_clarification_request_template()
        }
        
        for template_name, content in templates.items():
            template_path = self.template_dir / f"{template_name}.jinja2"
            if not template_path.exists():
                template_path.write_text(content)
                logger.info(f"Created default template: {template_name}")
    
    def render_response(self, 
                       template_name: str, 
                       data: Dict[str, Any], 
                       language: str = 'en') -> str:
        """
        Render a response using a template
        
        Args:
            template_name: Name of the template to use
            data: Data to pass to the template
            language: Language code ('en' or 'es')
            
        Returns:
            Rendered response string
        """
        try:
            # Add language to data
            data['language'] = language
            
            # Get template
            template = self.env.get_template(f"{template_name}.jinja2")
            
            # Render template
            response = template.render(**data)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error rendering template {template_name}: {e}")
            return self._get_fallback_response(template_name, data, language)
    
    def render_single_metric(self, 
                           metric_name: str, 
                           value: float, 
                           currency: str = '$',
                           period: str = 'this month',
                           language: str = 'en') -> str:
        """Render a single metric response"""
        data = {
            'metric_name': metric_name,
            'value': value,
            'currency': currency,
            'period': period,
            'formatted_value': f"{currency}{value:,.2f}"
        }
        return self.render_response('single_metric', data, language)
    
    def render_comparison(self, 
                         metric_name: str,
                         current_value: float,
                         previous_value: float,
                         currency: str = '$',
                         period: str = 'this month',
                         language: str = 'en') -> str:
        """Render a comparison response"""
        change = current_value - previous_value
        change_percentage = (change / previous_value * 100) if previous_value != 0 else 0
        
        data = {
            'metric_name': metric_name,
            'current_value': current_value,
            'previous_value': previous_value,
            'change': change,
            'change_percentage': change_percentage,
            'currency': currency,
            'period': period,
            'formatted_current': f"{currency}{current_value:,.2f}",
            'formatted_previous': f"{currency}{previous_value:,.2f}",
            'formatted_change': f"{currency}{change:+,.2f}",
            'formatted_percentage': f"{change_percentage:+.1f}%",
            'is_positive': change > 0
        }
        return self.render_response('comparison', data, language)
    
    def render_goal_progress(self, 
                           goal_name: str,
                           current_amount: float,
                           target_amount: float,
                           currency: str = '$',
                           language: str = 'en') -> str:
        """Render a goal progress response"""
        progress_percentage = (current_amount / target_amount * 100) if target_amount != 0 else 0
        remaining = target_amount - current_amount
        
        data = {
            'goal_name': goal_name,
            'current_amount': current_amount,
            'target_amount': target_amount,
            'remaining': remaining,
            'progress_percentage': progress_percentage,
            'currency': currency,
            'formatted_current': f"{currency}{current_amount:,.2f}",
            'formatted_target': f"{currency}{target_amount:,.2f}",
            'formatted_remaining': f"{currency}{remaining:,.2f}",
            'formatted_percentage': f"{progress_percentage:.1f}%",
            'is_complete': current_amount >= target_amount
        }
        return self.render_response('goal_progress', data, language)
    
    def render_anomaly_alert(self, 
                           anomaly_type: str,
                           amount: float,
                           date: str,
                           description: str,
                           severity: str = 'medium',
                           currency: str = '$',
                           language: str = 'en') -> str:
        """Render an anomaly alert response"""
        data = {
            'anomaly_type': anomaly_type,
            'amount': amount,
            'date': date,
            'description': description,
            'severity': severity,
            'currency': currency,
            'formatted_amount': f"{currency}{amount:,.2f}",
            'severity_emoji': self._get_severity_emoji(severity)
        }
        return self.render_response('anomaly_alert', data, language)
    
    def render_savings_recommendation(self, 
                                    target_amount: float,
                                    current_expenses: float,
                                    recommendations: List[Dict[str, Any]],
                                    currency: str = '$',
                                    language: str = 'en') -> str:
        """Render a savings recommendation response"""
        data = {
            'target_amount': target_amount,
            'current_expenses': current_expenses,
            'recommendations': recommendations,
            'currency': currency,
            'formatted_target': f"{currency}{target_amount:,.2f}",
            'formatted_expenses': f"{currency}{current_expenses:,.2f}"
        }
        return self.render_response('savings_recommendation', data, language)
    
    def render_privacy_refusal(self, 
                             violation_type: str,
                             reason: str,
                             suggested_action: Optional[str] = None,
                             language: str = 'en') -> str:
        """Render a privacy refusal response"""
        data = {
            'violation_type': violation_type,
            'reason': reason,
            'suggested_action': suggested_action
        }
        return self.render_response('privacy_refusal', data, language)
    
    def render_clarification_request(self, 
                                   original_query: str,
                                   clarification_questions: List[str],
                                   language: str = 'en') -> str:
        """Render a clarification request response"""
        data = {
            'original_query': original_query,
            'clarification_questions': clarification_questions
        }
        return self.render_response('clarification_request', data, language)
    
    def _get_severity_emoji(self, severity: str) -> str:
        """Get emoji for severity level"""
        emojis = {
            'low': '🟡',
            'medium': '🟠', 
            'high': '🔴',
            'critical': '🚨'
        }
        return emojis.get(severity, '⚠️')
    
    def _get_fallback_response(self, 
                              template_name: str, 
                              data: Dict[str, Any], 
                              language: str) -> str:
        """Get a fallback response when template rendering fails"""
        if language == 'es':
            return f"Error al procesar la respuesta para {template_name}. Por favor intenta de nuevo."
        else:
            return f"Error processing response for {template_name}. Please try again."
    
    # Template content methods
    def _get_single_metric_template(self) -> str:
        return """{% if language == 'es' %}
📊 **{{ metric_name }}**
Tu {{ metric_name.lower() }} para {{ period }} es {{ formatted_value }}.
{% else %}
📊 **{{ metric_name }}**
Your {{ metric_name.lower() }} for {{ period }} is {{ formatted_value }}.
{% endif %}"""

    def _get_comparison_template(self) -> str:
        return """{% if language == 'es' %}
📈 **Comparación de {{ metric_name }}**
{{ period | title }}: {{ formatted_current }}
{{ 'mes anterior' if period == 'este mes' else 'período anterior' | title }}: {{ formatted_previous }}
Cambio: {{ formatted_change }} ({{ formatted_percentage }})
{% if is_positive %}📈 Tendencia positiva{% else %}📉 Tendencia negativa{% endif %}
{% else %}
📈 **{{ metric_name }} Comparison**
{{ period | title }}: {{ formatted_current }}
{{ 'previous month' if period == 'this month' else 'previous period' | title }}: {{ formatted_previous }}
Change: {{ formatted_change }} ({{ formatted_percentage }})
{% if is_positive %}📈 Positive trend{% else %}📉 Negative trend{% endif %}
{% endif %}"""

    def _get_goal_progress_template(self) -> str:
        return """{% if language == 'es' %}
🎯 **Progreso hacia {{ goal_name }}**
Monto actual: {{ formatted_current }}
Meta: {{ formatted_target }}
Progreso: {{ formatted_percentage }}
{% if is_complete %}✅ ¡Meta alcanzada!{% else %}Faltan: {{ formatted_remaining }}{% endif %}
{% else %}
🎯 **Progress towards {{ goal_name }}**
Current amount: {{ formatted_current }}
Target: {{ formatted_target }}
Progress: {{ formatted_percentage }}
{% if is_complete %}✅ Goal achieved!{% else %}Remaining: {{ formatted_remaining }}{% endif %}
{% endif %}"""

    def _get_anomaly_alert_template(self) -> str:
        return """{% if language == 'es' %}
{{ severity_emoji }} **Alerta de Anomalía**
Tipo: {{ anomaly_type }}
Monto: {{ formatted_amount }}
Fecha: {{ date }}
Descripción: {{ description }}
{% else %}
{{ severity_emoji }} **Anomaly Alert**
Type: {{ anomaly_type }}
Amount: {{ formatted_amount }}
Date: {{ date }}
Description: {{ description }}
{% endif %}"""

    def _get_savings_recommendation_template(self) -> str:
        return """{% if language == 'es' %}
💰 **Plan de Ahorro Personalizado**
Objetivo: {{ formatted_target }}/mes
Gastos actuales: {{ formatted_expenses }}/mes

**Recomendaciones:**
{% for rec in recommendations %}
- {{ rec.category }}: reducir {{ rec.currency }}{{ rec.amount:,.0f }} ({{ rec.percentage }}%)
{% endfor %}
{% else %}
💰 **Personalized Savings Plan**
Target: {{ formatted_target }}/month
Current expenses: {{ formatted_expenses }}/month

**Recommendations:**
{% for rec in recommendations %}
- {{ rec.category }}: reduce {{ rec.currency }}{{ rec.amount:,.0f }} ({{ rec.percentage }}%)
{% endfor %}
{% endif %}"""

    def _get_privacy_refusal_template(self) -> str:
        return """{% if language == 'es' %}
🚫 **Solicitud Bloqueada**
Razón: {{ reason }}
{% if suggested_action %}
Sugerencia: {{ suggested_action }}
{% endif %}
{% else %}
🚫 **Request Blocked**
Reason: {{ reason }}
{% if suggested_action %}
Suggestion: {{ suggested_action }}
{% endif %}
{% endif %}"""

    def _get_clarification_request_template(self) -> str:
        return """{% if language == 'es' %}
❓ **Necesito más información**
Tu pregunta: "{{ original_query }}"

Para darte una respuesta más precisa, necesito que aclares:
{% for question in clarification_questions %}
{{ loop.index }}. {{ question }}
{% endfor %}
{% else %}
❓ **I need more information**
Your question: "{{ original_query }}"

To give you a more accurate answer, I need you to clarify:
{% for question in clarification_questions %}
{{ loop.index }}. {{ question }}
{% endfor %}
{% endif %}""" 