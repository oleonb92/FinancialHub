"""
Privacy Guard for FinancialHub AI
Checks if queries violate privacy rules and blocks inappropriate requests
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """Types of privacy violations"""
    OTHER_USER_DATA = "other_user_data"
    INVESTMENT_ADVICE = "investment_advice"
    PERSONAL_INFO_REQUEST = "personal_info_request"
    SENSITIVE_FINANCIAL_DATA = "sensitive_financial_data"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    COMPETITIVE_INTELLIGENCE = "competitive_intelligence"


@dataclass
class PrivacyViolation:
    """Represents a privacy violation"""
    violation_type: ViolationType
    severity: str  # "low", "medium", "high", "critical"
    description: str
    blocked: bool
    reason: str
    suggested_action: Optional[str] = None


class PrivacyGuard:
    """
    Guards against privacy violations in financial queries
    """
    
    def __init__(self):
        """Initialize the privacy guard with rules and patterns"""
        self._setup_rules()
        self._setup_patterns()
    
    def _setup_rules(self):
        """Setup privacy rules and keywords"""
        self.privacy_rules = {
            ViolationType.OTHER_USER_DATA: {
                'keywords': [
                    'other user', 'another user', 'someone else', 'colleague',
                    'otro usuario', 'otra persona', 'colega', 'compañero',
                    'user data', 'user information', 'user account',
                    'datos de usuario', 'información de usuario', 'cuenta de usuario'
                ],
                'severity': 'high',
                'blocked': True
            },
            ViolationType.INVESTMENT_ADVICE: {
                'keywords': [
                    'invest in', 'buy stock', 'sell stock', 'stock market',
                    'invertir en', 'comprar acciones', 'vender acciones', 'bolsa',
                    'cryptocurrency', 'bitcoin', 'ethereum', 'crypto',
                    'criptomoneda', 'bitcoin', 'ethereum',
                    'investment advice', 'financial advice', 'trading advice',
                    'consejo de inversión', 'consejo financiero', 'consejo de trading'
                ],
                'severity': 'medium',
                'blocked': True
            },
            ViolationType.PERSONAL_INFO_REQUEST: {
                'keywords': [
                    'personal information', 'private data', 'confidential',
                    'información personal', 'datos privados', 'confidencial',
                    'social security', 'ssn', 'passport', 'driver license',
                    'seguridad social', 'pasaporte', 'licencia de conducir'
                ],
                'severity': 'critical',
                'blocked': True
            },
            ViolationType.SENSITIVE_FINANCIAL_DATA: {
                'keywords': [
                    'bank account number', 'credit card number', 'account details',
                    'número de cuenta bancaria', 'número de tarjeta de crédito',
                    'detalles de cuenta', 'routing number', 'swift code',
                    'número de routing', 'código swift'
                ],
                'severity': 'critical',
                'blocked': True
            },
            ViolationType.UNAUTHORIZED_ACCESS: {
                'keywords': [
                    'admin access', 'root access', 'system access',
                    'acceso de administrador', 'acceso root', 'acceso al sistema',
                    'bypass security', 'hack', 'exploit', 'vulnerability',
                    'bypassear seguridad', 'hackear', 'explotar', 'vulnerabilidad'
                ],
                'severity': 'high',
                'blocked': True
            },
            ViolationType.COMPETITIVE_INTELLIGENCE: {
                'keywords': [
                    'competitor data', 'competitor information', 'market share',
                    'datos del competidor', 'información del competidor',
                    'cuota de mercado', 'business intelligence', 'inteligencia empresarial'
                ],
                'severity': 'medium',
                'blocked': False  # Warning only
            }
        }
    
    def _setup_patterns(self):
        """Setup regex patterns for more complex detection"""
        self.patterns = {
            ViolationType.OTHER_USER_DATA: [
                r'user\s+\d+',
                r'usuario\s+\d+',
                r'account\s+\d+',
                r'cuenta\s+\d+',
                r'[a-zA-Z]+\'s\s+(data|information|account)',
                r'[a-zA-Z]+\'s\s+(datos|información|cuenta)'
            ],
            ViolationType.INVESTMENT_ADVICE: [
                r'should\s+I\s+(invest|buy|sell)',
                r'debería\s+(invertir|comprar|vender)',
                r'what\s+should\s+I\s+(invest|buy|sell)',
                r'qué\s+debería\s+(invertir|comprar|vender)',
                r'is\s+it\s+good\s+to\s+(invest|buy|sell)',
                r'es\s+bueno\s+(invertir|comprar|vender)'
            ],
            ViolationType.PERSONAL_INFO_REQUEST: [
                r'(ssn|social\s+security)\s+number',
                r'número\s+(ssn|seguridad\s+social)',
                r'passport\s+number',
                r'número\s+de\s+pasaporte',
                r'credit\s+card\s+number',
                r'número\s+de\s+tarjeta\s+de\s+crédito'
            ]
        }
    
    def check_query(self, 
                   query: str, 
                   user_id: str, 
                   organization_id: str,
                   user_permissions: Optional[List[str]] = None) -> Tuple[bool, Optional[PrivacyViolation]]:
        """
        Check if a query violates privacy rules
        
        Args:
            query: User query to check
            user_id: User identifier
            organization_id: Organization identifier
            user_permissions: List of user permissions
            
        Returns:
            Tuple of (is_safe, violation_object)
        """
        try:
            query_lower = query.lower()
            
            # Check for violations
            for violation_type, rule in self.privacy_rules.items():
                # Check keywords
                if any(keyword in query_lower for keyword in rule['keywords']):
                    violation = self._create_violation(violation_type, rule, query)
                    return False, violation
                
                # Check patterns
                if violation_type in self.patterns:
                    for pattern in self.patterns[violation_type]:
                        if re.search(pattern, query_lower, re.IGNORECASE):
                            violation = self._create_violation(violation_type, rule, query)
                            return False, violation
            
            # Additional context-based checks
            context_violation = self._check_context_violations(
                query, user_id, organization_id, user_permissions
            )
            if context_violation:
                return False, context_violation
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error checking privacy: {e}")
            # Default to blocking on error
            violation = PrivacyViolation(
                violation_type=ViolationType.UNAUTHORIZED_ACCESS,
                severity="high",
                description="Error occurred during privacy check",
                blocked=True,
                reason="System error during privacy validation"
            )
            return False, violation
    
    def _create_violation(self, 
                         violation_type: ViolationType, 
                         rule: Dict[str, Any], 
                         query: str) -> PrivacyViolation:
        """Create a privacy violation object"""
        
        descriptions = {
            ViolationType.OTHER_USER_DATA: "Query attempts to access other user data",
            ViolationType.INVESTMENT_ADVICE: "Query requests investment advice",
            ViolationType.PERSONAL_INFO_REQUEST: "Query requests personal information",
            ViolationType.SENSITIVE_FINANCIAL_DATA: "Query requests sensitive financial data",
            ViolationType.UNAUTHORIZED_ACCESS: "Query attempts unauthorized access",
            ViolationType.COMPETITIVE_INTELLIGENCE: "Query requests competitive intelligence"
        }
        
        reasons = {
            ViolationType.OTHER_USER_DATA: "Cannot access other users' financial data",
            ViolationType.INVESTMENT_ADVICE: "Cannot provide investment advice",
            ViolationType.PERSONAL_INFO_REQUEST: "Cannot request personal information",
            ViolationType.SENSITIVE_FINANCIAL_DATA: "Cannot request sensitive financial data",
            ViolationType.UNAUTHORIZED_ACCESS: "Unauthorized access attempt",
            ViolationType.COMPETITIVE_INTELLIGENCE: "Competitive intelligence requests not allowed"
        }
        
        suggestions = {
            ViolationType.OTHER_USER_DATA: "Please ask about your own financial data",
            ViolationType.INVESTMENT_ADVICE: "Please consult a financial advisor for investment advice",
            ViolationType.PERSONAL_INFO_REQUEST: "Please contact support for personal information",
            ViolationType.SENSITIVE_FINANCIAL_DATA: "Please contact your bank for account details",
            ViolationType.UNAUTHORIZED_ACCESS: "Please contact support for access issues",
            ViolationType.COMPETITIVE_INTELLIGENCE: "Please focus on your own financial data"
        }
        
        return PrivacyViolation(
            violation_type=violation_type,
            severity=rule['severity'],
            description=descriptions.get(violation_type, "Privacy violation detected"),
            blocked=rule['blocked'],
            reason=reasons.get(violation_type, "Query violates privacy rules"),
            suggested_action=suggestions.get(violation_type)
        )
    
    def _check_context_violations(self, 
                                 query: str, 
                                 user_id: str, 
                                 organization_id: str,
                                 user_permissions: Optional[List[str]] = None) -> Optional[PrivacyViolation]:
        """Check for context-based violations"""
        
        # Check for admin-only operations
        admin_keywords = ['all users', 'all accounts', 'system wide', 'organization wide']
        if any(keyword in query.lower() for keyword in admin_keywords):
            if not user_permissions or 'admin' not in user_permissions:
                return PrivacyViolation(
                    violation_type=ViolationType.UNAUTHORIZED_ACCESS,
                    severity="high",
                    description="Query requests organization-wide data",
                    blocked=True,
                    reason="Admin permissions required for organization-wide queries",
                    suggested_action="Contact your administrator for access"
                )
        
        return None
    
    def get_violation_message(self, 
                            violation: PrivacyViolation, 
                            language: str = 'en') -> str:
        """
        Get a user-friendly message for a privacy violation
        
        Args:
            violation: PrivacyViolation object
            language: Language code ('en' or 'es')
            
        Returns:
            User-friendly message
        """
        
        messages = {
            'en': {
                ViolationType.OTHER_USER_DATA: "I cannot access other users' financial data. Please ask about your own information.",
                ViolationType.INVESTMENT_ADVICE: "I cannot provide investment advice. Please consult a qualified financial advisor.",
                ViolationType.PERSONAL_INFO_REQUEST: "I cannot request personal information. Please contact support for assistance.",
                ViolationType.SENSITIVE_FINANCIAL_DATA: "I cannot request sensitive financial data. Please contact your bank directly.",
                ViolationType.UNAUTHORIZED_ACCESS: "This request requires special permissions. Please contact your administrator.",
                ViolationType.COMPETITIVE_INTELLIGENCE: "I cannot provide competitive intelligence. Please focus on your own financial data."
            },
            'es': {
                ViolationType.OTHER_USER_DATA: "No puedo acceder a los datos financieros de otros usuarios. Por favor pregunta sobre tu propia información.",
                ViolationType.INVESTMENT_ADVICE: "No puedo proporcionar consejos de inversión. Por favor consulta a un asesor financiero calificado.",
                ViolationType.PERSONAL_INFO_REQUEST: "No puedo solicitar información personal. Por favor contacta al soporte para ayuda.",
                ViolationType.SENSITIVE_FINANCIAL_DATA: "No puedo solicitar datos financieros sensibles. Por favor contacta a tu banco directamente.",
                ViolationType.UNAUTHORIZED_ACCESS: "Esta solicitud requiere permisos especiales. Por favor contacta a tu administrador.",
                ViolationType.COMPETITIVE_INTELLIGENCE: "No puedo proporcionar inteligencia competitiva. Por favor enfócate en tus propios datos financieros."
            }
        }
        
        return messages.get(language, messages['en']).get(
            violation.violation_type, 
            "This request cannot be processed due to privacy restrictions."
        ) 