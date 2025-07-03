"""
Unit tests for ConversationContextManager module
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from django.test import TestCase
from django.core.cache import cache

from ..context_manager import ConversationContextManager, QueryContext


class ConversationContextManagerTestCase(TestCase):
    """Test cases for ConversationContextManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.context_manager = ConversationContextManager(cache_timeout=300)
        self.user_id = "test_user_123"
        self.organization_id = "test_org_456"
        self.test_query = "What is my balance?"
        self.test_intent = {
            'intent_type': 'balance_inquiry',
            'confidence_score': 0.85,
            'entities': {'time_period': 'current'}
        }
    
    def tearDown(self):
        """Clean up after tests"""
        cache.clear()
    
    def test_init(self):
        """Test ConversationContextManager initialization"""
        self.assertEqual(self.context_manager.cache_timeout, 300)
        self.assertEqual(self.context_manager.cache_prefix, "ai_context")
    
    def test_get_cache_key(self):
        """Test cache key generation"""
        expected_key = f"ai_context:{self.organization_id}:{self.user_id}"
        actual_key = self.context_manager._get_cache_key(self.user_id, self.organization_id)
        self.assertEqual(actual_key, expected_key)
    
    def test_store_and_get_context(self):
        """Test storing and retrieving context"""
        # Store context
        self.context_manager.store_context(
            user_id=self.user_id,
            organization_id=self.organization_id,
            query=self.test_query,
            parsed_intent=self.test_intent,
            confidence_score=0.85
        )
        
        # Retrieve context
        context = self.context_manager.get_context(self.user_id, self.organization_id)
        
        # Verify context
        self.assertIsNotNone(context)
        self.assertEqual(context.user_id, self.user_id)
        self.assertEqual(context.organization_id, self.organization_id)
        self.assertEqual(context.original_query, self.test_query)
        self.assertEqual(context.parsed_intent, self.test_intent)
        self.assertEqual(context.confidence_score, 0.85)
        self.assertIsInstance(context.timestamp, datetime)
    
    def test_store_context_with_financial_data(self):
        """Test storing context with financial data"""
        financial_data = {
            'summary': {
                'total_income': 5000.0,
                'total_expenses': 3000.0,
                'net_balance': 2000.0
            }
        }
        
        self.context_manager.store_context(
            user_id=self.user_id,
            organization_id=self.organization_id,
            query=self.test_query,
            parsed_intent=self.test_intent,
            confidence_score=0.85,
            financial_data=financial_data
        )
        
        context = self.context_manager.get_context(self.user_id, self.organization_id)
        self.assertEqual(context.financial_data, financial_data)
    
    def test_store_context_with_response_summary(self):
        """Test storing context with response summary"""
        response_summary = "Your current balance is $2,000"
        
        self.context_manager.store_context(
            user_id=self.user_id,
            organization_id=self.organization_id,
            query=self.test_query,
            parsed_intent=self.test_intent,
            confidence_score=0.85,
            response_summary=response_summary
        )
        
        context = self.context_manager.get_context(self.user_id, self.organization_id)
        self.assertEqual(context.response_summary, response_summary)
    
    def test_get_context_not_found(self):
        """Test getting context that doesn't exist"""
        context = self.context_manager.get_context("nonexistent_user", self.organization_id)
        self.assertIsNone(context)
    
    def test_resolve_follow_up_questions(self):
        """Test resolving follow-up questions"""
        # Store initial context
        self.context_manager.store_context(
            user_id=self.user_id,
            organization_id=self.organization_id,
            query="What is my balance?",
            parsed_intent={'intent_type': 'balance_inquiry', 'entities': {}},
            confidence_score=0.85
        )
        
        # Test follow-up questions
        follow_up_queries = [
            "What about last month?",
            "How about yesterday?",
            "¿Qué tal el mes pasado?",
            "¿Y ayer?"
        ]
        
        for query in follow_up_queries:
            resolved = self.context_manager.resolve_follow_up(
                self.user_id, self.organization_id, query
            )
            self.assertIsNotNone(resolved, f"Should resolve follow-up: {query}")
            self.assertTrue(resolved['is_follow_up'])
            self.assertIn('resolved_intent', resolved)
    
    def test_resolve_follow_up_not_follow_up(self):
        """Test that non-follow-up questions return None"""
        # Store initial context
        self.context_manager.store_context(
            user_id=self.user_id,
            organization_id=self.organization_id,
            query="What is my balance?",
            parsed_intent={'intent_type': 'balance_inquiry', 'entities': {}},
            confidence_score=0.85
        )
        
        # Test non-follow-up questions
        non_follow_up_queries = [
            "What is my balance?",
            "Show me my expenses",
            "How much did I spend?",
            "¿Cuál es mi balance?",
            "Muéstrame mis gastos"
        ]
        
        for query in non_follow_up_queries:
            resolved = self.context_manager.resolve_follow_up(
                self.user_id, self.organization_id, query
            )
            self.assertIsNone(resolved, f"Should not resolve as follow-up: {query}")
    
    def test_resolve_follow_up_no_context(self):
        """Test resolving follow-up when no context exists"""
        resolved = self.context_manager.resolve_follow_up(
            self.user_id, self.organization_id, "What about last month?"
        )
        self.assertIsNone(resolved)
    
    def test_resolve_time_references(self):
        """Test resolving time references in follow-up questions"""
        # Create a context with date range
        context = QueryContext(
            user_id=self.user_id,
            organization_id=self.organization_id,
            original_query="What is my balance?",
            parsed_intent={
                'intent_type': 'balance_inquiry',
                'date_range': {
                    'start_date': '2024-01-01',
                    'end_date': '2024-01-31'
                }
            },
            confidence_score=0.85,
            timestamp=datetime.now()
        )
        
        # Test time reference resolution
        resolved = self.context_manager._resolve_time_references(
            context, "What about last month?"
        )
        
        self.assertIn('resolved_intent', resolved)
        self.assertTrue(resolved['is_follow_up'])
        
        # Check that date range was updated
        resolved_intent = resolved['resolved_intent']
        self.assertIn('date_range', resolved_intent)
    
    def test_clear_context(self):
        """Test clearing context"""
        # Store context
        self.context_manager.store_context(
            user_id=self.user_id,
            organization_id=self.organization_id,
            query=self.test_query,
            parsed_intent=self.test_intent,
            confidence_score=0.85
        )
        
        # Verify context exists
        context = self.context_manager.get_context(self.user_id, self.organization_id)
        self.assertIsNotNone(context)
        
        # Clear context
        self.context_manager.clear_context(self.user_id, self.organization_id)
        
        # Verify context is gone
        context = self.context_manager.get_context(self.user_id, self.organization_id)
        self.assertIsNone(context)
    
    def test_get_context_history(self):
        """Test getting context history"""
        # Store a context
        self.context_manager.store_context(
            user_id=self.user_id,
            organization_id=self.organization_id,
            query=self.test_query,
            parsed_intent=self.test_intent,
            confidence_score=0.85
        )
        
        # Get history
        history = self.context_manager.get_context_history(
            self.user_id, self.organization_id, limit=5
        )
        
        # Should return current context
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].original_query, self.test_query)
    
    def test_query_context_to_dict(self):
        """Test QueryContext serialization to dictionary"""
        context = QueryContext(
            user_id=self.user_id,
            organization_id=self.organization_id,
            original_query=self.test_query,
            parsed_intent=self.test_intent,
            confidence_score=0.85,
            timestamp=datetime.now()
        )
        
        context_dict = context.to_dict()
        
        self.assertIn('user_id', context_dict)
        self.assertIn('organization_id', context_dict)
        self.assertIn('original_query', context_dict)
        self.assertIn('parsed_intent', context_dict)
        self.assertIn('confidence_score', context_dict)
        self.assertIn('timestamp', context_dict)
        self.assertIsInstance(context_dict['timestamp'], str)
    
    def test_query_context_from_dict(self):
        """Test QueryContext deserialization from dictionary"""
        original_context = QueryContext(
            user_id=self.user_id,
            organization_id=self.organization_id,
            original_query=self.test_query,
            parsed_intent=self.test_intent,
            confidence_score=0.85,
            timestamp=datetime.now()
        )
        
        context_dict = original_context.to_dict()
        restored_context = QueryContext.from_dict(context_dict)
        
        self.assertEqual(restored_context.user_id, original_context.user_id)
        self.assertEqual(restored_context.organization_id, original_context.organization_id)
        self.assertEqual(restored_context.original_query, original_context.original_query)
        self.assertEqual(restored_context.parsed_intent, original_context.parsed_intent)
        self.assertEqual(restored_context.confidence_score, original_context.confidence_score)
        self.assertIsInstance(restored_context.timestamp, datetime)
    
    def test_error_handling_store_context(self):
        """Test error handling in store_context"""
        with patch('django.core.cache.cache.set') as mock_set:
            mock_set.side_effect = Exception("Cache error")
            
            # Should not raise exception
            self.context_manager.store_context(
                user_id=self.user_id,
                organization_id=self.organization_id,
                query=self.test_query,
                parsed_intent=self.test_intent,
                confidence_score=0.85
            )
    
    def test_error_handling_get_context(self):
        """Test error handling in get_context"""
        with patch('django.core.cache.cache.get') as mock_get:
            mock_get.side_effect = Exception("Cache error")
            
            # Should return None on error
            context = self.context_manager.get_context(self.user_id, self.organization_id)
            self.assertIsNone(context)
    
    def test_error_handling_clear_context(self):
        """Test error handling in clear_context"""
        with patch('django.core.cache.cache.delete') as mock_delete:
            mock_delete.side_effect = Exception("Cache error")
            
            # Should not raise exception
            self.context_manager.clear_context(self.user_id, self.organization_id)


if __name__ == '__main__':
    unittest.main() 