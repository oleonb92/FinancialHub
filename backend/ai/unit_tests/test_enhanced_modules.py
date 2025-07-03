"""
Unit Tests for Enhanced AI Modules

This module contains comprehensive unit tests for all the enhanced AI modules:
- IntentClassifier
- FollowUpSuggester
- ReportGenerator
- EnhancedFinancialQueryParser
- I18nManager
"""

import unittest
import tempfile
import os
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import shutil

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.core.intent_classifier import IntentClassifier, IntentPrediction
from ai.core.followup_suggester import FollowUpSuggester, FollowUpSuggestion
from ai.core.report_generator import ReportGenerator, ReportData, ReportConfig
from ai.core.enhanced_query_parser import EnhancedFinancialQueryParser, EnhancedParsedIntent
from ai.core.translation_service import TranslationService


class TestIntentClassifier(unittest.TestCase):
    """Test cases for the IntentClassifier module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, 'test_intent_classifier.joblib')
        self.classifier = IntentClassifier(model_path=self.model_path, language='en')
        
        # Sample training data
        self.training_data = [
            ("What's my balance?", "balance_inquiry"),
            ("How much did I spend?", "spending_analysis"),
            ("I want to save money", "savings_planning"),
            ("Show me trends", "trend_analysis"),
            ("Is this unusual?", "anomaly_detection"),
            ("How are my goals?", "goal_tracking"),
            ("Compare this month", "comparison"),
            ("What will happen next?", "prediction")
        ]
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test IntentClassifier initialization."""
        self.assertIsNotNone(self.classifier)
        self.assertEqual(self.classifier.language, 'en')
        self.assertFalse(self.classifier.is_trained)
        self.assertIsNotNone(self.classifier.intent_categories)
    
    def test_basic_embedding(self):
        """Test basic embedding generation."""
        text = "What's my balance?"
        embedding = self.classifier._basic_embedding(text)
        
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)
        self.assertTrue(all(isinstance(x, (int, float)) for x in embedding))
    
    def test_train_intent_classifier(self):
        """Test intent classifier training."""
        results = self.classifier.train_intent_classifier(self.training_data)
        
        self.assertTrue(self.classifier.is_trained)
        self.assertIsNotNone(results)
        self.assertIn('accuracy', results)
        self.assertIn('classification_report', results)
        self.assertGreater(results['accuracy'], 0.0)
    
    def test_predict_intent(self):
        """Test intent prediction."""
        # Train first
        self.classifier.train_intent_classifier(self.training_data)
        
        # Test prediction
        intent, confidence = self.classifier.predict_intent("What's my current balance?")
        
        self.assertIsInstance(intent, str)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_predict_intent_detailed(self):
        """Test detailed intent prediction."""
        # Train first
        self.classifier.train_intent_classifier(self.training_data)
        
        # Test detailed prediction
        prediction = self.classifier.predict_intent_detailed("What's my current balance?")
        
        self.assertIsInstance(prediction, IntentPrediction)
        self.assertIsInstance(prediction.intent, str)
        self.assertIsInstance(prediction.confidence, float)
        self.assertIsInstance(prediction.all_scores, dict)
    
    def test_save_and_load_model(self):
        """Test model saving and loading."""
        # Train and save
        self.classifier.train_intent_classifier(self.training_data)
        self.classifier.save_model()
        
        # Create new classifier and load
        new_classifier = IntentClassifier(model_path=self.model_path, language='en')
        new_classifier.load_model()
        
        self.assertTrue(new_classifier.is_trained)
        self.assertEqual(new_classifier.language, 'en')
    
    def test_get_training_data_template(self):
        """Test getting training data template."""
        template_data = self.classifier.get_training_data_template()
        
        self.assertIsInstance(template_data, list)
        self.assertGreater(len(template_data), 0)
        self.assertTrue(all(isinstance(item, tuple) and len(item) == 2 for item in template_data))
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        # Train first
        self.classifier.train_intent_classifier(self.training_data)
        
        # Test evaluation
        test_data = [
            ("Show my balance", "balance_inquiry"),
            ("How much spent?", "spending_analysis")
        ]
        
        results = self.classifier.evaluate_model(test_data)
        
        self.assertIsInstance(results, dict)
        self.assertIn('accuracy', results)
        self.assertIn('classification_report', results)
        self.assertIn('confidence_stats', results)


class TestFollowUpSuggester(unittest.TestCase):
    """Test cases for the FollowUpSuggester module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.suggester = FollowUpSuggester(language='en')
    
    def test_initialization(self):
        """Test FollowUpSuggester initialization."""
        self.assertIsNotNone(self.suggester)
        self.assertEqual(self.suggester.language, 'en')
        self.assertIsNotNone(self.suggester.suggestion_templates)
    
    def test_generate_followup_suggestions(self):
        """Test follow-up suggestion generation."""
        suggestions = self.suggester.generate_followup_suggestions(
            last_intent='balance_inquiry',
            context={'has_goals': True}
        )
        
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
        self.assertLessEqual(len(suggestions), 3)
        
        for suggestion in suggestions:
            self.assertIsInstance(suggestion, FollowUpSuggestion)
            self.assertIsInstance(suggestion.question, str)
            self.assertIsInstance(suggestion.intent, str)
            self.assertIsInstance(suggestion.confidence, float)
            self.assertIsInstance(suggestion.category, str)
            self.assertIsInstance(suggestion.reasoning, str)
    
    def test_generate_suggestions_with_context(self):
        """Test suggestion generation with context."""
        context = {
            'has_goals': True,
            'has_anomalies': False,
            'is_comparative': True,
            'needs_action': False
        }
        
        suggestions = self.suggester.generate_followup_suggestions(
            last_intent='spending_analysis',
            context=context
        )
        
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
    
    def test_general_suggestions_fallback(self):
        """Test fallback to general suggestions."""
        suggestions = self.suggester.generate_followup_suggestions(
            last_intent='unknown_intent'
        )
        
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
    
    def test_context_memory(self):
        """Test context memory functionality."""
        # Generate suggestions to populate memory
        self.suggester.generate_followup_suggestions('balance_inquiry')
        
        context = self.suggester.get_conversation_context()
        
        self.assertIsInstance(context, dict)
        self.assertIn('recent_intents', context)
        self.assertIn('used_suggestions', context)
        self.assertIn('conversation_length', context)
    
    def test_clear_context_memory(self):
        """Test clearing context memory."""
        # Generate suggestions to populate memory
        self.suggester.generate_followup_suggestions('balance_inquiry')
        
        # Clear memory
        self.suggester.clear_context_memory()
        
        context = self.suggester.get_conversation_context()
        self.assertEqual(context['conversation_length'], 0)
    
    def test_add_custom_suggestion(self):
        """Test adding custom suggestions."""
        self.suggester.add_custom_suggestion(
            intent='balance_inquiry',
            question='Custom question?',
            target_intent='spending_analysis',
            category='custom',
            confidence=0.8
        )
        
        # Verify suggestion was added
        suggestions = self.suggester.generate_followup_suggestions('balance_inquiry')
        custom_questions = [s.question for s in suggestions]
        self.assertIn('Custom question?', custom_questions)


class TestReportGenerator(unittest.TestCase):
    """Test cases for the ReportGenerator module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.templates_dir = os.path.join(self.temp_dir, 'templates')
        self.output_dir = os.path.join(self.temp_dir, 'output')
        
        self.generator = ReportGenerator(
            templates_dir=self.templates_dir,
            output_dir=self.output_dir
        )
        
        # Sample report data
        self.sample_data = self.generator.create_sample_data()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test ReportGenerator initialization."""
        self.assertIsNotNone(self.generator)
        self.assertEqual(self.generator.templates_dir, self.templates_dir)
        self.assertEqual(self.generator.output_dir, self.output_dir)
        self.assertIsNotNone(self.generator.color_schemes)
    
    def test_create_sample_data(self):
        """Test sample data creation."""
        data = self.generator.create_sample_data()
        
        self.assertIsInstance(data, ReportData)
        self.assertIsInstance(data.monthly_data, list)
        self.assertIsInstance(data.summary, dict)
        self.assertIsInstance(data.categories, list)
        self.assertIsInstance(data.insights, list)
        self.assertIsInstance(data.goals, list)
        self.assertIsInstance(data.anomalies, list)
        self.assertIsInstance(data.trends, dict)
    
    def test_generate_monthly_report_html(self):
        """Test monthly report generation in HTML format."""
        config = ReportConfig(
            format='html',
            include_charts=True,
            include_insights=True,
            include_goals=True,
            include_anomalies=True
        )
        
        report_content = self.generator.generate_monthly_report(
            self.sample_data,
            config
        )
        
        self.assertIsInstance(report_content, str)
        self.assertIn('<!DOCTYPE html>', report_content)
        self.assertIn('Monthly Financial Report', report_content)
    
    def test_generate_quick_summary(self):
        """Test quick summary generation."""
        summary_content = self.generator.generate_quick_summary(self.sample_data)
        
        self.assertIsInstance(summary_content, str)
        self.assertIn('Financial Summary Report', summary_content)
    
    def test_save_report(self):
        """Test report saving."""
        config = ReportConfig(format='html')
        
        report_content = self.generator.generate_monthly_report(
            self.sample_data,
            config
        )
        
        saved_path = self.generator.save_report(
            report_content,
            'test_report',
            config
        )
        
        self.assertIsInstance(saved_path, str)
        self.assertTrue(os.path.exists(saved_path))
    
    def test_chart_generation(self):
        """Test chart generation."""
        config = ReportConfig(include_charts=True)
        
        charts = self.generator._generate_charts(self.sample_data, config)
        
        self.assertIsInstance(charts, dict)
        self.assertIn('monthly_chart_data', charts)
        self.assertIn('categories_chart_data', charts)
        self.assertIn('income_expenses_chart_data', charts)
    
    def test_calculate_savings_rate(self):
        """Test savings rate calculation."""
        summary = {
            'total_income': 10000.0,
            'total_expenses': 7000.0
        }
        
        savings_rate = self.generator._calculate_savings_rate(summary)
        
        self.assertEqual(savings_rate, 30.0)  # (10000-7000)/10000 * 100
    
    def test_calculate_savings_rate_zero_income(self):
        """Test savings rate calculation with zero income."""
        summary = {
            'total_income': 0.0,
            'total_expenses': 1000.0
        }
        
        savings_rate = self.generator._calculate_savings_rate(summary)
        
        self.assertEqual(savings_rate, 0.0)


class TestEnhancedFinancialQueryParser(unittest.TestCase):
    """Test cases for the EnhancedFinancialQueryParser module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = EnhancedFinancialQueryParser(language='en')
    
    def test_initialization(self):
        """Test EnhancedFinancialQueryParser initialization."""
        self.assertIsNotNone(self.parser)
        self.assertEqual(self.parser.language, 'en')
        self.assertIsNotNone(self.parser.entity_patterns)
        self.assertIsNotNone(self.parser.amount_patterns)
        self.assertIsNotNone(self.parser.category_mapping)
    
    def test_parse_query_basic(self):
        """Test basic query parsing."""
        query = "What's my balance?"
        result = self.parser.parse_query(query)
        
        self.assertIsInstance(result, EnhancedParsedIntent)
        self.assertIsInstance(result.intent_type, str)
        self.assertIsInstance(result.confidence_score, float)
        self.assertIsInstance(result.entities, dict)
        self.assertIsInstance(result.metadata, dict)
        self.assertIsInstance(result.extracted_dates, list)
        self.assertIsInstance(result.extracted_amounts, list)
        self.assertIsInstance(result.extracted_categories, list)
        self.assertIsInstance(result.extracted_accounts, list)
        self.assertIsInstance(result.language, str)
        self.assertIsInstance(result.processing_time, float)
    
    def test_extract_amounts_enhanced(self):
        """Test enhanced amount extraction."""
        query = "I spent $150.50 on groceries and 200 dollars on gas"
        amounts = self.parser._extract_amounts_enhanced(query, 'en')
        
        self.assertIsInstance(amounts, list)
        self.assertGreater(len(amounts), 0)
        
        for amount in amounts:
            self.assertIsInstance(amount, dict)
            self.assertIn('text', amount)
            self.assertIn('value', amount)
            self.assertIn('currency', amount)
            self.assertIn('confidence', amount)
    
    def test_extract_categories_enhanced(self):
        """Test enhanced category extraction."""
        query = "I spent money on food and transportation"
        categories = self.parser._extract_categories_enhanced(query, 'en')
        
        self.assertIsInstance(categories, list)
        # Should extract 'food' and 'transportation' categories
    
    def test_extract_accounts_enhanced(self):
        """Test enhanced account extraction."""
        query = "Show me my checking account balance"
        accounts = self.parser._extract_accounts_enhanced(query, 'en')
        
        self.assertIsInstance(accounts, list)
    
    def test_detect_language(self):
        """Test language detection."""
        english_text = "What's my balance?"
        spanish_text = "¿Cuál es mi saldo?"
        
        english_detected = self.parser._detect_language(english_text)
        spanish_detected = self.parser._detect_language(spanish_text)
        
        self.assertEqual(english_detected, 'en')
        self.assertEqual(spanish_detected, 'es')
    
    def test_calculate_complexity_score(self):
        """Test complexity score calculation."""
        simple_query = "What's my balance?"
        complex_query = "How does my spending this month compare to last month and what trends can you identify?"
        
        simple_score = self.parser._calculate_complexity_score(simple_query)
        complex_score = self.parser._calculate_complexity_score(complex_query)
        
        self.assertIsInstance(simple_score, float)
        self.assertIsInstance(complex_score, float)
        self.assertLess(simple_score, complex_score)
    
    def test_has_entity(self):
        """Test entity detection."""
        query_with_time = "Show me my balance from last month"
        query_with_amount = "I spent $100 on groceries"
        
        has_time = self.parser._has_entity(query_with_time, 'time_period')
        has_amount = self.parser._has_entity(query_with_amount, 'amount_range')
        
        self.assertTrue(has_time)
        self.assertTrue(has_amount)


class TestTranslationService(unittest.TestCase):
    """Test cases for the TranslationService module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.translation_service = TranslationService(default_language='en')
    
    def test_initialization(self):
        """Test TranslationService initialization."""
        self.assertIsNotNone(self.translation_service)
        self.assertEqual(self.translation_service.default_language, 'en')
        self.assertIsInstance(self.translation_service.supported_languages, list)
        self.assertIn('en', self.translation_service.supported_languages)
        self.assertIn('es', self.translation_service.supported_languages)
    
    def test_translate(self):
        """Test translation functionality."""
        # Test basic translation (should return original if deep-translator not available)
        result = self.translation_service.translate('Hello world', 'es')
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
    
    def test_detect_language(self):
        """Test language detection."""
        # Test English detection
        result = self.translation_service.detect_language("Hello world")
        self.assertEqual(result, 'en')
        
        # Test Spanish detection
        result = self.translation_service.detect_language("Hola mundo")
        self.assertEqual(result, 'es')
    
    def test_format_currency(self):
        """Test currency formatting."""
        # Test English formatting
        result = self.translation_service.format_currency(1234.56, 'en', 'USD')
        self.assertIsInstance(result, str)
        self.assertIn('$', result)
        
        # Test Spanish formatting
        result = self.translation_service.format_currency(1234.56, 'es', 'USD')
        self.assertIsInstance(result, str)
        self.assertIn('$', result)
    
    def test_format_number(self):
        """Test number formatting."""
        result = self.translation_service.format_number(1234.56, 'en')
        self.assertIsInstance(result, str)
    
    def test_format_date(self):
        """Test date formatting."""
        test_date = datetime(2024, 1, 15)
        
        # Test English formatting
        result = self.translation_service.format_date(test_date, 'long', 'en')
        self.assertIsInstance(result, str)
        self.assertIn('2024', result)
        
        # Test Spanish formatting
        result = self.translation_service.format_date(test_date, 'long', 'es')
        self.assertIsInstance(result, str)
        self.assertIn('2024', result)
    
    def test_get_supported_languages(self):
        """Test getting supported languages."""
        languages = self.translation_service.get_supported_languages()
        self.assertIsInstance(languages, list)
        self.assertIn('en', languages)
        self.assertIn('es', languages)
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Add something to cache
        self.translation_service.translate('test', 'es')
        initial_cache_size = len(self.translation_service.translation_cache)
        
        # Clear cache
        self.translation_service.clear_cache()
        self.assertEqual(len(self.translation_service.translation_cache), 0)
    
    def test_get_cache_stats(self):
        """Test cache statistics."""
        stats = self.translation_service.get_cache_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('cache_size', stats)
        self.assertIn('supported_languages', stats)
        self.assertIn('default_language', stats)


class TestIntegration(unittest.TestCase):
    """Integration tests for the enhanced AI modules."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.translation_service = TranslationService(default_language='en')
        self.parser = EnhancedFinancialQueryParser(language='en')
        self.suggester = FollowUpSuggester(language='en')
        self.classifier = IntentClassifier(language='en')
        self.generator = ReportGenerator()
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # 1. Parse a query
        query = "What's my balance and how does it compare to last month?"
        parsed = self.parser.parse_query(query)
        
        self.assertIsInstance(parsed, EnhancedParsedIntent)
        self.assertGreater(parsed.confidence_score, 0.0)
        
        # 2. Generate follow-up suggestions
        suggestions = self.suggester.generate_followup_suggestions(
            parsed.intent_type,
            context={'is_comparative': True}
        )
        
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
        
        # 3. Generate a report
        report_data = self.generator.create_sample_data()
        report_content = self.generator.generate_monthly_report(report_data)
        
        self.assertIsInstance(report_content, str)
        self.assertIn('Monthly Financial Report', report_content)
        
        # 4. Test internationalization
        translated_intent = self.translation_service.translate(f'intents.{parsed.intent_type}', 'en')
        self.assertIsInstance(translated_intent, str)
    
    def test_multilingual_workflow(self):
        """Test multilingual workflow."""
        # Test Spanish query
        spanish_query = "¿Cuál es mi saldo?"
        parsed_es = self.parser.parse_query(spanish_query, language='es')
        
        self.assertEqual(parsed_es.language, 'es')
        
        # Test Spanish suggestions
        suggestions_es = self.suggester.generate_followup_suggestions(
            parsed_es.intent_type,
            context={}
        )
        
        self.assertIsInstance(suggestions_es, list)
        
        # Test Spanish translations
        translated_es = self.translation_service.translate('messages.welcome', 'es')
        self.assertIsInstance(translated_es, str)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2) 