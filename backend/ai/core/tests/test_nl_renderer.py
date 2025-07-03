"""
Unit tests for NLRenderer module
"""

import unittest
import tempfile
import os
from unittest.mock import Mock, patch
from django.test import TestCase
from django.conf import settings

from ..nl_renderer import NLRenderer


class NLRendererTestCase(TestCase):
    """Test cases for NLRenderer"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary directory for templates
        self.temp_dir = tempfile.mkdtemp()
        self.renderer = NLRenderer(template_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test NLRenderer initialization"""
        self.assertIsNotNone(self.renderer.env)
        self.assertEqual(self.renderer.template_dir, self.temp_dir)
    
    def test_create_default_templates(self):
        """Test that default templates are created"""
        template_files = [
            'single_metric.jinja2',
            'comparison.jinja2',
            'goal_progress.jinja2',
            'anomaly_alert.jinja2',
            'savings_recommendation.jinja2',
            'privacy_refusal.jinja2',
            'clarification_request.jinja2'
        ]
        
        for template_file in template_files:
            template_path = os.path.join(self.temp_dir, template_file)
            self.assertTrue(os.path.exists(template_path), f"Template {template_file} should exist")
    
    def test_render_single_metric_english(self):
        """Test rendering single metric in English"""
        response = self.renderer.render_single_metric(
            metric_name="Balance",
            value=2500.0,
            currency="$",
            period="this month",
            language="en"
        )
        
        self.assertIn("Balance", response)
        self.assertIn("$2,500.00", response)
        self.assertIn("this month", response)
        self.assertIn("📊", response)
    
    def test_render_single_metric_spanish(self):
        """Test rendering single metric in Spanish"""
        response = self.renderer.render_single_metric(
            metric_name="Balance",
            value=2500.0,
            currency="$",
            period="este mes",
            language="es"
        )
        
        self.assertIn("Balance", response)
        self.assertIn("$2,500.00", response)
        self.assertIn("este mes", response)
        self.assertIn("📊", response)
    
    def test_render_comparison_english(self):
        """Test rendering comparison in English"""
        response = self.renderer.render_comparison(
            metric_name="Expenses",
            current_value=3000.0,
            previous_value=2500.0,
            currency="$",
            period="this month",
            language="en"
        )
        
        self.assertIn("Expenses", response)
        self.assertIn("$3,000.00", response)
        self.assertIn("$2,500.00", response)
        self.assertIn("+$500.00", response)
        self.assertIn("+20.0%", response)
        self.assertIn("📈", response)
    
    def test_render_comparison_spanish(self):
        """Test rendering comparison in Spanish"""
        response = self.renderer.render_comparison(
            metric_name="Gastos",
            current_value=3000.0,
            previous_value=2500.0,
            currency="$",
            period="este mes",
            language="es"
        )
        
        self.assertIn("Gastos", response)
        self.assertIn("$3,000.00", response)
        self.assertIn("$2,500.00", response)
        self.assertIn("+$500.00", response)
        self.assertIn("+20.0%", response)
        self.assertIn("📈", response)
    
    def test_render_goal_progress_english(self):
        """Test rendering goal progress in English"""
        response = self.renderer.render_goal_progress(
            goal_name="Emergency Fund",
            current_amount=3000.0,
            target_amount=5000.0,
            currency="$",
            language="en"
        )
        
        self.assertIn("Emergency Fund", response)
        self.assertIn("$3,000.00", response)
        self.assertIn("$5,000.00", response)
        self.assertIn("60.0%", response)
        self.assertIn("$2,000.00", response)  # Remaining
        self.assertIn("🎯", response)
    
    def test_render_goal_progress_spanish(self):
        """Test rendering goal progress in Spanish"""
        response = self.renderer.render_goal_progress(
            goal_name="Fondo de Emergencia",
            current_amount=3000.0,
            target_amount=5000.0,
            currency="$",
            language="es"
        )
        
        self.assertIn("Fondo de Emergencia", response)
        self.assertIn("$3,000.00", response)
        self.assertIn("$5,000.00", response)
        self.assertIn("60.0%", response)
        self.assertIn("$2,000.00", response)  # Remaining
        self.assertIn("🎯", response)
    
    def test_render_goal_progress_complete(self):
        """Test rendering completed goal progress"""
        response = self.renderer.render_goal_progress(
            goal_name="Vacation Fund",
            current_amount=5000.0,
            target_amount=5000.0,
            currency="$",
            language="en"
        )
        
        self.assertIn("✅ Goal achieved!", response)
    
    def test_render_anomaly_alert(self):
        """Test rendering anomaly alert"""
        response = self.renderer.render_anomaly_alert(
            anomaly_type="Unusual Spending",
            amount=1500.0,
            date="2024-01-15",
            description="Large transaction detected",
            severity="high",
            currency="$",
            language="en"
        )
        
        self.assertIn("Unusual Spending", response)
        self.assertIn("$1,500.00", response)
        self.assertIn("2024-01-15", response)
        self.assertIn("Large transaction detected", response)
        self.assertIn("🔴", response)  # High severity emoji
    
    def test_render_savings_recommendation(self):
        """Test rendering savings recommendation"""
        recommendations = [
            {'category': 'Food', 'amount': 200.0, 'percentage': 15.0, 'currency': '$'},
            {'category': 'Entertainment', 'amount': 150.0, 'percentage': 10.0, 'currency': '$'}
        ]
        
        response = self.renderer.render_savings_recommendation(
            target_amount=1000.0,
            current_expenses=3000.0,
            recommendations=recommendations,
            currency="$",
            language="en"
        )
        
        self.assertIn("$1,000.00", response)
        self.assertIn("$3,000.00", response)
        self.assertIn("Food", response)
        self.assertIn("Entertainment", response)
        self.assertIn("$200", response)
        self.assertIn("$150", response)
        self.assertIn("💰", response)
    
    def test_render_privacy_refusal(self):
        """Test rendering privacy refusal"""
        response = self.renderer.render_privacy_refusal(
            violation_type="Investment Advice",
            reason="Cannot provide investment advice",
            suggested_action="Please consult a financial advisor",
            language="en"
        )
        
        self.assertIn("Investment Advice", response)
        self.assertIn("Cannot provide investment advice", response)
        self.assertIn("Please consult a financial advisor", response)
        self.assertIn("🚫", response)
    
    def test_render_clarification_request(self):
        """Test rendering clarification request"""
        clarification_questions = [
            "Which month are you referring to?",
            "Do you mean total expenses or a specific category?"
        ]
        
        response = self.renderer.render_clarification_request(
            original_query="Show me my expenses",
            clarification_questions=clarification_questions,
            language="en"
        )
        
        self.assertIn("Show me my expenses", response)
        self.assertIn("Which month are you referring to?", response)
        self.assertIn("Do you mean total expenses or a specific category?", response)
        self.assertIn("❓", response)
    
    def test_get_severity_emoji(self):
        """Test severity emoji mapping"""
        self.assertEqual(self.renderer._get_severity_emoji('low'), '🟡')
        self.assertEqual(self.renderer._get_severity_emoji('medium'), '🟠')
        self.assertEqual(self.renderer._get_severity_emoji('high'), '🔴')
        self.assertEqual(self.renderer._get_severity_emoji('critical'), '🚨')
        self.assertEqual(self.renderer._get_severity_emoji('unknown'), '⚠️')
    
    def test_render_response_with_custom_data(self):
        """Test rendering with custom data"""
        custom_data = {
            'user_name': 'John',
            'account_type': 'Savings',
            'balance': 5000.0
        }
        
        # Create a custom template
        custom_template = """
        Hello {{ user_name }}!
        Your {{ account_type }} account balance is ${{ "%.2f"|format(balance) }}.
        """
        
        template_path = os.path.join(self.temp_dir, 'custom.jinja2')
        with open(template_path, 'w') as f:
            f.write(custom_template)
        
        response = self.renderer.render_response('custom', custom_data, 'en')
        
        self.assertIn("Hello John!", response)
        self.assertIn("Your Savings account balance is $5000.00.", response)
    
    def test_render_response_fallback(self):
        """Test fallback response when template rendering fails"""
        # Try to render a non-existent template
        response = self.renderer.render_response('nonexistent', {}, 'en')
        
        self.assertIn("Error processing response", response)
        self.assertIn("nonexistent", response)
    
    def test_render_response_fallback_spanish(self):
        """Test fallback response in Spanish"""
        response = self.renderer.render_response('nonexistent', {}, 'es')
        
        self.assertIn("Error al procesar", response)
        self.assertIn("nonexistent", response)
    
    def test_comparison_negative_change(self):
        """Test comparison with negative change"""
        response = self.renderer.render_comparison(
            metric_name="Income",
            current_value=2000.0,
            previous_value=2500.0,
            currency="$",
            period="this month",
            language="en"
        )
        
        self.assertIn("-$500.00", response)
        self.assertIn("-20.0%", response)
        self.assertIn("📉 Negative trend", response)
    
    def test_comparison_zero_change(self):
        """Test comparison with zero change"""
        response = self.renderer.render_comparison(
            metric_name="Balance",
            current_value=1000.0,
            previous_value=1000.0,
            currency="$",
            period="this month",
            language="en"
        )
        
        self.assertIn("$0.00", response)
        self.assertIn("0.0%", response)
    
    def test_goal_progress_zero_target(self):
        """Test goal progress with zero target"""
        response = self.renderer.render_goal_progress(
            goal_name="Test Goal",
            current_amount=100.0,
            target_amount=0.0,
            currency="$",
            language="en"
        )
        
        # Should handle division by zero gracefully
        self.assertIn("Test Goal", response)
        self.assertIn("$100.00", response)
    
    def test_template_rendering_with_special_characters(self):
        """Test template rendering with special characters"""
        data = {
            'message': 'This has "quotes" and <tags>',
            'amount': 1000.0
        }
        
        # Create template with special characters
        template_content = """
        Message: {{ message }}
        Amount: ${{ "%.2f"|format(amount) }}
        """
        
        template_path = os.path.join(self.temp_dir, 'special.jinja2')
        with open(template_path, 'w') as f:
            f.write(template_content)
        
        response = self.renderer.render_response('special', data, 'en')
        
        self.assertIn('This has "quotes" and <tags>', response)
        self.assertIn('$1000.00', response)
    
    def test_error_handling_malformed_template(self):
        """Test error handling with malformed template"""
        # Create a malformed template
        malformed_template = """
        {% if condition %}
        {{ undefined_variable }}
        {% endif %}
        """
        
        template_path = os.path.join(self.temp_dir, 'malformed.jinja2')
        with open(template_path, 'w') as f:
            f.write(malformed_template)
        
        # Should not raise exception, should return fallback
        response = self.renderer.render_response('malformed', {}, 'en')
        self.assertIn("Error processing response", response)


if __name__ == '__main__':
    unittest.main() 