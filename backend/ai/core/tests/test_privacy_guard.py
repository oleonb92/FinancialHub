"""
Unit tests for PrivacyGuard module
"""

import unittest
from unittest.mock import Mock, patch
from django.test import TestCase
from django.core.cache import cache

from ..privacy_guard import PrivacyGuard, PrivacyViolation, ViolationType


class PrivacyGuardTestCase(TestCase):
    """Test cases for PrivacyGuard"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.privacy_guard = PrivacyGuard()
        self.user_id = "test_user_123"
        self.organization_id = "test_org_456"
    
    def tearDown(self):
        """Clean up after tests"""
        cache.clear()
    
    def test_init(self):
        """Test PrivacyGuard initialization"""
        self.assertIsNotNone(self.privacy_guard.privacy_rules)
        self.assertIsNotNone(self.privacy_guard.patterns)
        self.assertIn(ViolationType.OTHER_USER_DATA, self.privacy_guard.privacy_rules)
        self.assertIn(ViolationType.INVESTMENT_ADVICE, self.privacy_guard.privacy_rules)
    
    def test_check_query_safe(self):
        """Test that safe queries pass privacy check"""
        safe_queries = [
            "What is my balance?",
            "How much did I spend on food?",
            "Show me my expenses for this month",
            "¿Cuál es mi balance?",
            "¿Cuánto gasté en comida?"
        ]
        
        for query in safe_queries:
            is_safe, violation = self.privacy_guard.check_query(
                query, self.user_id, self.organization_id
            )
            self.assertTrue(is_safe, f"Query should be safe: {query}")
            self.assertIsNone(violation, f"No violation should be detected: {query}")
    
    def test_check_query_other_user_data(self):
        """Test detection of other user data requests"""
        violating_queries = [
            "Show me other user's data",
            "What does John spend on?",
            "Show me someone else's balance",
            "¿Cuáles son los gastos de María?",
            "user 123 data"
        ]
        
        for query in violating_queries:
            is_safe, violation = self.privacy_guard.check_query(
                query, self.user_id, self.organization_id
            )
            self.assertFalse(is_safe, f"Query should be blocked: {query}")
            self.assertIsNotNone(violation, f"Violation should be detected: {query}")
            self.assertEqual(violation.violation_type, ViolationType.OTHER_USER_DATA)
            self.assertTrue(violation.blocked)
    
    def test_check_query_investment_advice(self):
        """Test detection of investment advice requests"""
        violating_queries = [
            "Should I invest in Bitcoin?",
            "What stocks should I buy?",
            "Is it good to invest in crypto?",
            "¿Debería invertir en acciones?",
            "¿Es bueno comprar bitcoin?"
        ]
        
        for query in violating_queries:
            is_safe, violation = self.privacy_guard.check_query(
                query, self.user_id, self.organization_id
            )
            self.assertFalse(is_safe, f"Query should be blocked: {query}")
            self.assertIsNotNone(violation, f"Violation should be detected: {query}")
            self.assertEqual(violation.violation_type, ViolationType.INVESTMENT_ADVICE)
            self.assertTrue(violation.blocked)
    
    def test_check_query_personal_info(self):
        """Test detection of personal information requests"""
        violating_queries = [
            "What is my social security number?",
            "Show me my passport number",
            "What's my SSN?",
            "¿Cuál es mi número de seguridad social?",
            "¿Cuál es mi número de pasaporte?"
        ]
        
        for query in violating_queries:
            is_safe, violation = self.privacy_guard.check_query(
                query, self.user_id, self.organization_id
            )
            self.assertFalse(is_safe, f"Query should be blocked: {query}")
            self.assertIsNotNone(violation, f"Violation should be detected: {query}")
            self.assertEqual(violation.violation_type, ViolationType.PERSONAL_INFO_REQUEST)
            self.assertTrue(violation.blocked)
    
    def test_check_query_sensitive_financial_data(self):
        """Test detection of sensitive financial data requests"""
        violating_queries = [
            "What is my bank account number?",
            "Show me my credit card number",
            "What's my routing number?",
            "¿Cuál es mi número de cuenta bancaria?",
            "¿Cuál es mi número de tarjeta de crédito?"
        ]
        
        for query in violating_queries:
            is_safe, violation = self.privacy_guard.check_query(
                query, self.user_id, self.organization_id
            )
            self.assertFalse(is_safe, f"Query should be blocked: {query}")
            self.assertIsNotNone(violation, f"Violation should be detected: {query}")
            self.assertEqual(violation.violation_type, ViolationType.SENSITIVE_FINANCIAL_DATA)
            self.assertTrue(violation.blocked)
    
    def test_check_query_unauthorized_access(self):
        """Test detection of unauthorized access attempts"""
        violating_queries = [
            "Give me admin access",
            "How can I hack the system?",
            "Show me all users data",
            "¿Cómo puedo obtener acceso root?",
            "¿Puedo bypassear la seguridad?"
        ]
        
        for query in violating_queries:
            is_safe, violation = self.privacy_guard.check_query(
                query, self.user_id, self.organization_id
            )
            self.assertFalse(is_safe, f"Query should be blocked: {query}")
            self.assertIsNotNone(violation, f"Violation should be detected: {query}")
            self.assertEqual(violation.violation_type, ViolationType.UNAUTHORIZED_ACCESS)
            self.assertTrue(violation.blocked)
    
    def test_check_query_competitive_intelligence(self):
        """Test detection of competitive intelligence requests"""
        violating_queries = [
            "Show me competitor data",
            "What's our market share?",
            "Compare us to competitors",
            "¿Cuáles son los datos del competidor?",
            "¿Cuál es nuestra cuota de mercado?"
        ]
        
        for query in violating_queries:
            is_safe, violation = self.privacy_guard.check_query(
                query, self.user_id, self.organization_id
            )
            self.assertFalse(is_safe, f"Query should be blocked: {query}")
            self.assertIsNotNone(violation, f"Violation should be detected: {query}")
            self.assertEqual(violation.violation_type, ViolationType.COMPETITIVE_INTELLIGENCE)
            self.assertFalse(violation.blocked)  # Warning only
    
    def test_check_query_with_admin_permissions(self):
        """Test that admin users can access organization-wide data"""
        admin_query = "Show me all users data"
        admin_permissions = ["admin", "user_management"]
        
        is_safe, violation = self.privacy_guard.check_query(
            admin_query, self.user_id, self.organization_id, admin_permissions
        )
        self.assertTrue(is_safe, "Admin should be able to access organization-wide data")
        self.assertIsNone(violation, "No violation should be detected for admin")
    
    def test_check_query_without_admin_permissions(self):
        """Test that non-admin users cannot access organization-wide data"""
        admin_query = "Show me all users data"
        regular_permissions = ["user"]
        
        is_safe, violation = self.privacy_guard.check_query(
            admin_query, self.user_id, self.organization_id, regular_permissions
        )
        self.assertFalse(is_safe, "Non-admin should not access organization-wide data")
        self.assertIsNotNone(violation, "Violation should be detected")
        self.assertEqual(violation.violation_type, ViolationType.UNAUTHORIZED_ACCESS)
    
    def test_get_violation_message_english(self):
        """Test violation message generation in English"""
        violation = PrivacyViolation(
            violation_type=ViolationType.OTHER_USER_DATA,
            severity="high",
            description="Test violation",
            blocked=True,
            reason="Test reason"
        )
        
        message = self.privacy_guard.get_violation_message(violation, 'en')
        self.assertIn("cannot access other users", message.lower())
        self.assertIn("financial data", message.lower())
    
    def test_get_violation_message_spanish(self):
        """Test violation message generation in Spanish"""
        violation = PrivacyViolation(
            violation_type=ViolationType.INVESTMENT_ADVICE,
            severity="medium",
            description="Test violation",
            blocked=True,
            reason="Test reason"
        )
        
        message = self.privacy_guard.get_violation_message(violation, 'es')
        self.assertIn("no puedo proporcionar", message.lower())
        self.assertIn("consejos de inversión", message.lower())
    
    def test_get_violation_message_fallback(self):
        """Test violation message fallback for unknown language"""
        violation = PrivacyViolation(
            violation_type=ViolationType.PERSONAL_INFO_REQUEST,
            severity="critical",
            description="Test violation",
            blocked=True,
            reason="Test reason"
        )
        
        message = self.privacy_guard.get_violation_message(violation, 'fr')
        self.assertIn("cannot request personal information", message.lower())
    
    def test_regex_patterns(self):
        """Test regex pattern detection"""
        # Test user pattern
        query = "Show me user 123's data"
        is_safe, violation = self.privacy_guard.check_query(
            query, self.user_id, self.organization_id
        )
        self.assertFalse(is_safe)
        self.assertEqual(violation.violation_type, ViolationType.OTHER_USER_DATA)
        
        # Test investment pattern
        query = "Should I invest in stocks?"
        is_safe, violation = self.privacy_guard.check_query(
            query, self.user_id, self.organization_id
        )
        self.assertFalse(is_safe)
        self.assertEqual(violation.violation_type, ViolationType.INVESTMENT_ADVICE)
    
    def test_error_handling(self):
        """Test error handling in privacy check"""
        with patch.object(self.privacy_guard, 'privacy_rules', {}):
            # This should cause an error
            is_safe, violation = self.privacy_guard.check_query(
                "test query", self.user_id, self.organization_id
            )
            self.assertFalse(is_safe)
            self.assertIsNotNone(violation)
            self.assertEqual(violation.violation_type, ViolationType.UNAUTHORIZED_ACCESS)


if __name__ == '__main__':
    unittest.main() 