"""
Main AI Service for FinancialHub
Integrates all core modules: PrivacyGuard, ContextManager, NLRenderer, and PromptBuilder
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .privacy_guard import PrivacyGuard, PrivacyViolation
from .context_manager import ConversationContextManager
from .nl_renderer import NLRenderer
from .prompt_builder import PromptBuilder, PromptComponents
from ..ml.query_parser import FinancialQueryParser, ParsedIntent

logger = logging.getLogger(__name__)


@dataclass
class AIResponse:
    """Represents a complete AI response"""
    response_text: str
    confidence_score: float
    intent_type: str
    language: str
    privacy_violation: Optional[PrivacyViolation] = None
    requires_clarification: bool = False
    clarification_questions: Optional[list] = None
    metadata: Optional[Dict[str, Any]] = None


class AIService:
    """
    Main AI service that orchestrates all core modules
    """
    
    def __init__(self):
        """Initialize the AI service with all core modules"""
        self.privacy_guard = PrivacyGuard()
        self.context_manager = ConversationContextManager()
        self.nl_renderer = NLRenderer()
        self.prompt_builder = PromptBuilder()
        self.query_parser = FinancialQueryParser()
    
    def process_query(self, 
                     user_query: str,
                     user_id: str,
                     organization_id: str,
                     financial_data: Optional[Dict[str, Any]] = None,
                     user_permissions: Optional[list] = None,
                     language: Optional[str] = None) -> AIResponse:
        """
        Process a user query through the complete AI pipeline
        
        Args:
            user_query: User's question
            user_id: User identifier
            organization_id: Organization identifier
            financial_data: Financial data from database
            user_permissions: User permissions list
            language: Preferred language ('en' or 'es')
            
        Returns:
            AIResponse object with complete response
        """
        try:
            # Step 1: Privacy Check
            is_safe, privacy_violation = self.privacy_guard.check_query(
                user_query, user_id, organization_id, user_permissions
            )
            
            if not is_safe:
                return self._handle_privacy_violation(privacy_violation, language)
            
            # Step 2: Context Resolution
            resolved_context = self.context_manager.resolve_follow_up(
                user_id, organization_id, user_query
            )
            
            # Step 3: Query Parsing
            parsed_intent = self.query_parser.parse_query(user_query)
            
            # Step 4: Language Detection
            detected_language = language or parsed_intent.metadata.get('language', 'en')
            
            # Step 5: Confidence Check and Fallback
            if parsed_intent.confidence_score < 0.3:
                return self._handle_low_confidence(parsed_intent, user_query, detected_language)
            
            # Step 6: Build Prompt
            prompt = self._build_prompt(
                user_query, parsed_intent, financial_data, resolved_context, detected_language
            )
            
            # Step 7: Generate Response (stub for now)
            response_text = self._generate_response(prompt, parsed_intent, financial_data)
            
            # Step 8: Store Context
            self.context_manager.store_context(
                user_id=user_id,
                organization_id=organization_id,
                query=user_query,
                parsed_intent=parsed_intent.__dict__,
                confidence_score=parsed_intent.confidence_score,
                financial_data=financial_data,
                response_summary=response_text[:200]  # Store summary
            )
            
            # Step 9: Return Response
            return AIResponse(
                response_text=response_text,
                confidence_score=parsed_intent.confidence_score,
                intent_type=parsed_intent.intent_type,
                language=detected_language,
                metadata=parsed_intent.metadata
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return self._handle_error(user_query, language or 'en')
    
    def _handle_privacy_violation(self, 
                                 violation: PrivacyViolation, 
                                 language: str) -> AIResponse:
        """Handle privacy violations"""
        response_text = self.privacy_guard.get_violation_message(violation, language)
        
        return AIResponse(
            response_text=response_text,
            confidence_score=1.0,
            intent_type='privacy_violation',
            language=language,
            privacy_violation=violation
        )
    
    def _handle_low_confidence(self, 
                              parsed_intent: ParsedIntent, 
                              user_query: str, 
                              language: str) -> AIResponse:
        """Handle low confidence queries with clarification request"""
        clarification_questions = self._generate_clarification_questions(
            parsed_intent, user_query, language
        )
        
        response_text = self.nl_renderer.render_clarification_request(
            original_query=user_query,
            clarification_questions=clarification_questions,
            language=language
        )
        
        return AIResponse(
            response_text=response_text,
            confidence_score=parsed_intent.confidence_score,
            intent_type='clarification_request',
            language=language,
            requires_clarification=True,
            clarification_questions=clarification_questions,
            metadata=parsed_intent.metadata
        )
    
    def _handle_error(self, user_query: str, language: str) -> AIResponse:
        """Handle processing errors"""
        if language == 'es':
            response_text = "Lo siento, hubo un error procesando tu consulta. Por favor intenta de nuevo."
        else:
            response_text = "Sorry, there was an error processing your query. Please try again."
        
        return AIResponse(
            response_text=response_text,
            confidence_score=0.0,
            intent_type='error',
            language=language
        )
    
    def _build_prompt(self, 
                     user_query: str,
                     parsed_intent: ParsedIntent,
                     financial_data: Optional[Dict[str, Any]],
                     resolved_context: Optional[Dict[str, Any]],
                     language: str) -> str:
        """Build the complete prompt for the LLM"""
        
        # Determine prompt type based on intent
        prompt_type = self._map_intent_to_prompt_type(parsed_intent.intent_type)
        
        # Get system prompt
        system_prompt = self.prompt_builder.get_system_prompt(prompt_type, language)
        
        # Format financial context
        financial_context = None
        if financial_data:
            financial_context = self.prompt_builder._format_financial_context(
                financial_data, language
            )
        
        # Build components
        components = PromptComponents(
            system_prompt=system_prompt,
            user_query=user_query,
            financial_context=financial_context,
            language=language
        )
        
        # Add resolved context if available
        if resolved_context:
            components.orchestration_result = str(resolved_context.get('resolved_intent', {}))
        
        return self.prompt_builder.build_prompt(components)
    
    def _map_intent_to_prompt_type(self, intent_type: str) -> str:
        """Map intent type to prompt type"""
        mapping = {
            'balance_inquiry': 'financial_analysis',
            'spending_analysis': 'financial_analysis',
            'savings_planning': 'savings_planning',
            'trend_analysis': 'financial_analysis',
            'anomaly_detection': 'anomaly_detection',
            'goal_tracking': 'goal_tracking',
            'comparison': 'financial_analysis',
            'prediction': 'financial_analysis',
            'clarification_request': 'clarification',
            'general_inquiry': 'default'
        }
        return mapping.get(intent_type, 'default')
    
    def _generate_clarification_questions(self, 
                                        parsed_intent: ParsedIntent, 
                                        user_query: str, 
                                        language: str) -> list:
        """Generate clarification questions based on intent"""
        questions = []
        
        if language == 'es':
            if 'time_period' not in parsed_intent.entities or not parsed_intent.entities['time_period']:
                questions.append("¿A qué período te refieres? (este mes, mes pasado, etc.)")
            
            if 'financial_entity' not in parsed_intent.entities or not parsed_intent.entities['financial_entity']:
                questions.append("¿Qué información específica necesitas? (gastos, ingresos, balance, etc.)")
            
            if not questions:
                questions.append("¿Podrías ser más específico sobre lo que necesitas saber?")
        else:
            if 'time_period' not in parsed_intent.entities or not parsed_intent.entities['time_period']:
                questions.append("What time period are you referring to? (this month, last month, etc.)")
            
            if 'financial_entity' not in parsed_intent.entities or not parsed_intent.entities['financial_entity']:
                questions.append("What specific information do you need? (expenses, income, balance, etc.)")
            
            if not questions:
                questions.append("Could you be more specific about what you need to know?")
        
        return questions
    
    def _generate_response(self, 
                          prompt: str, 
                          parsed_intent: ParsedIntent,
                          financial_data: Optional[Dict[str, Any]]) -> str:
        """
        Generate response using the LLM (stub implementation)
        In production, this would call the actual LLM service
        """
        # This is a stub - in production, you would call your LLM service here
        # For now, return a template-based response
        
        if parsed_intent.intent_type == 'balance_inquiry':
            if financial_data and 'summary' in financial_data:
                balance = financial_data['summary'].get('net_balance', 0)
                return f"Your current balance is ${balance:,.2f}."
            else:
                return "I don't have enough financial data to show your balance."
        
        elif parsed_intent.intent_type == 'savings_planning':
            return "I can help you create a savings plan. Please provide your target amount."
        
        elif parsed_intent.intent_type == 'anomaly_detection':
            return "I'll analyze your transactions for any unusual patterns."
        
        else:
            return "I understand your query. Let me analyze your financial data and provide insights."
    
    def get_context_history(self, 
                           user_id: str, 
                           organization_id: str, 
                           limit: int = 5) -> list:
        """Get conversation history for a user"""
        return self.context_manager.get_context_history(
            user_id, organization_id, limit
        )
    
    def clear_context(self, user_id: str, organization_id: str) -> None:
        """Clear conversation context for a user"""
        self.context_manager.clear_context(user_id, organization_id)
    
    def render_template_response(self, 
                               template_name: str, 
                               data: Dict[str, Any], 
                               language: str = 'en') -> str:
        """Render a response using a specific template"""
        return self.nl_renderer.render_response(template_name, data, language)
    
    def check_privacy(self, 
                     query: str, 
                     user_id: str, 
                     organization_id: str,
                     user_permissions: Optional[list] = None) -> Tuple[bool, Optional[PrivacyViolation]]:
        """Check if a query violates privacy rules"""
        return self.privacy_guard.check_query(
            query, user_id, organization_id, user_permissions
        ) 