"""
Prompt Builder for FinancialHub AI
Assembles system prompts, user history, and orchestration results into concise prompts for LLM
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PromptComponents:
    """Components for building a prompt"""
    system_prompt: str
    user_history: Optional[str] = None
    financial_context: Optional[str] = None
    orchestration_result: Optional[str] = None
    user_query: Optional[str] = None
    language: str = 'en'


class PromptBuilder:
    """
    Builds prompts for LLM interactions
    """
    
    def __init__(self):
        """Initialize the prompt builder"""
        self._setup_system_prompts()
    
    def _setup_system_prompts(self):
        """Setup system prompts for different scenarios"""
        self.system_prompts = {
            'en': {
                'default': self._get_default_system_prompt_en(),
                'financial_analysis': self._get_financial_analysis_prompt_en(),
                'savings_planning': self._get_savings_planning_prompt_en(),
                'anomaly_detection': self._get_anomaly_detection_prompt_en(),
                'goal_tracking': self._get_goal_tracking_prompt_en(),
                'clarification': self._get_clarification_prompt_en()
            },
            'es': {
                'default': self._get_default_system_prompt_es(),
                'financial_analysis': self._get_financial_analysis_prompt_es(),
                'savings_planning': self._get_savings_planning_prompt_es(),
                'anomaly_detection': self._get_anomaly_detection_prompt_es(),
                'goal_tracking': self._get_goal_tracking_prompt_es(),
                'clarification': self._get_clarification_prompt_es()
            }
        }
    
    def build_prompt(self, 
                    components: PromptComponents,
                    max_tokens: int = 4000) -> str:
        """
        Build a complete prompt from components
        
        Args:
            components: PromptComponents object
            max_tokens: Maximum tokens for the prompt
            
        Returns:
            Complete prompt string
        """
        try:
            prompt_parts = []
            
            # Add system prompt
            if components.system_prompt:
                prompt_parts.append(components.system_prompt)
            
            # Add financial context if available
            if components.financial_context:
                prompt_parts.append(f"\n📊 FINANCIAL CONTEXT:\n{components.financial_context}")
            
            # Add orchestration result if available
            if components.orchestration_result:
                prompt_parts.append(f"\n🤖 AI ANALYSIS:\n{components.orchestration_result}")
            
            # Add user history if available
            if components.user_history:
                prompt_parts.append(f"\n💬 CONVERSATION HISTORY:\n{components.user_history}")
            
            # Add current user query
            if components.user_query:
                prompt_parts.append(f"\n👤 USER QUERY: {components.user_query}")
            
            # Combine all parts
            full_prompt = "\n".join(prompt_parts)
            
            # Truncate if too long
            if len(full_prompt) > max_tokens * 4:  # Rough estimate: 4 chars per token
                full_prompt = self._truncate_prompt(full_prompt, max_tokens)
            
            return full_prompt.strip()
            
        except Exception as e:
            logger.error(f"Error building prompt: {e}")
            return self._get_fallback_prompt(components.language)
    
    def get_system_prompt(self, 
                         prompt_type: str = 'default', 
                         language: str = 'en') -> str:
        """
        Get a system prompt by type and language
        
        Args:
            prompt_type: Type of system prompt
            language: Language code ('en' or 'es')
            
        Returns:
            System prompt string
        """
        return self.system_prompts.get(language, self.system_prompts['en']).get(
            prompt_type, 
            self.system_prompts[language]['default']
        )
    
    def build_financial_prompt(self, 
                             user_query: str,
                             financial_data: Dict[str, Any],
                             language: str = 'en',
                             include_history: bool = True,
                             user_history: Optional[str] = None) -> str:
        """
        Build a prompt specifically for financial queries
        
        Args:
            user_query: User's question
            financial_data: Financial data from database
            language: Language code
            include_history: Whether to include conversation history
            user_history: Previous conversation history
            
        Returns:
            Complete financial prompt
        """
        # Determine prompt type based on query
        prompt_type = self._determine_prompt_type(user_query)
        
        # Get system prompt
        system_prompt = self.get_system_prompt(prompt_type, language)
        
        # Format financial context
        financial_context = self._format_financial_context(financial_data, language)
        
        # Build components
        components = PromptComponents(
            system_prompt=system_prompt,
            user_query=user_query,
            financial_context=financial_context,
            language=language
        )
        
        # Add history if requested
        if include_history and user_history:
            components.user_history = user_history
        
        return self.build_prompt(components)
    
    def _determine_prompt_type(self, query: str) -> str:
        """Determine the type of prompt based on the query"""
        query_lower = query.lower()
        
        # Check for savings-related queries
        savings_keywords = ['save', 'savings', 'ahorrar', 'ahorro', 'budget', 'presupuesto']
        if any(keyword in query_lower for keyword in savings_keywords):
            return 'savings_planning'
        
        # Check for anomaly-related queries
        anomaly_keywords = ['anomaly', 'unusual', 'strange', 'anomalía', 'inusual', 'extraño']
        if any(keyword in query_lower for keyword in anomaly_keywords):
            return 'anomaly_detection'
        
        # Check for goal-related queries
        goal_keywords = ['goal', 'target', 'progress', 'meta', 'objetivo', 'progreso']
        if any(keyword in query_lower for keyword in goal_keywords):
            return 'goal_tracking'
        
        # Check for clarification requests
        clarification_keywords = ['clarify', 'explain', 'what do you mean', 'aclarar', 'explicar']
        if any(keyword in query_lower for keyword in clarification_keywords):
            return 'clarification'
        
        # Default to financial analysis
        return 'financial_analysis'
    
    def _format_financial_context(self, 
                                 financial_data: Dict[str, Any], 
                                 language: str) -> str:
        """Format financial data into context string"""
        try:
            context_parts = []
            
            # Basic financial summary
            if 'summary' in financial_data:
                summary = financial_data['summary']
                if language == 'es':
                    context_parts.append(f"💰 RESUMEN FINANCIERO:")
                    context_parts.append(f"  • Ingresos totales: ${summary.get('total_income', 0):,.2f}")
                    context_parts.append(f"  • Gastos totales: ${summary.get('total_expenses', 0):,.2f}")
                    context_parts.append(f"  • Balance neto: ${summary.get('net_balance', 0):,.2f}")
                else:
                    context_parts.append(f"💰 FINANCIAL SUMMARY:")
                    context_parts.append(f"  • Total income: ${summary.get('total_income', 0):,.2f}")
                    context_parts.append(f"  • Total expenses: ${summary.get('total_expenses', 0):,.2f}")
                    context_parts.append(f"  • Net balance: ${summary.get('net_balance', 0):,.2f}")
            
            # Top expense categories
            if 'top_expense_categories' in financial_data:
                categories = financial_data['top_expense_categories'][:3]
                if categories:
                    if language == 'es':
                        context_parts.append(f"\n💸 TOP GASTOS:")
                    else:
                        context_parts.append(f"\n💸 TOP EXPENSES:")
                    
                    for i, cat in enumerate(categories, 1):
                        context_parts.append(f"  {i}. {cat.get('category__name', 'Unknown')}: ${cat.get('total', 0):,.2f}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error formatting financial context: {e}")
            return "Financial data available"
    
    def _truncate_prompt(self, prompt: str, max_tokens: int) -> str:
        """Truncate prompt to fit within token limit"""
        # Simple truncation - in production, use proper tokenization
        max_chars = max_tokens * 4
        if len(prompt) <= max_chars:
            return prompt
        
        # Keep system prompt and user query, truncate middle parts
        lines = prompt.split('\n')
        system_lines = []
        other_lines = []
        
        in_system = True
        for line in lines:
            if line.startswith('👤 USER QUERY:'):
                in_system = False
                break
            if in_system:
                system_lines.append(line)
            else:
                other_lines.append(line)
        
        # Calculate available space for other parts
        system_text = '\n'.join(system_lines)
        available_chars = max_chars - len(system_text) - 100  # Buffer for user query
        
        # Truncate other parts
        other_text = '\n'.join(other_lines)
        if len(other_text) > available_chars:
            other_text = other_text[:available_chars] + "..."
        
        return system_text + '\n' + other_text
    
    def _get_fallback_prompt(self, language: str) -> str:
        """Get a fallback prompt when building fails"""
        if language == 'es':
            return "Eres un asistente financiero inteligente. Responde de manera clara y útil."
        else:
            return "You are an intelligent financial assistant. Respond clearly and helpfully."
    
    # System prompt templates
    def _get_default_system_prompt_en(self) -> str:
        return """You are an intelligent financial assistant for FinancialHub. Your role is to:

1. Analyze financial data and provide clear, actionable insights
2. Answer questions about spending, income, and financial trends
3. Help users understand their financial situation
4. Provide personalized recommendations based on their data
5. Respond in a friendly, professional tone

IMPORTANT RULES:
- Always use real data from the provided financial context
- Never invent or estimate numbers unless explicitly stated
- Be concise but informative
- Use emojis to make responses more engaging
- If you don't have enough data, ask for clarification
- Focus on actionable insights and recommendations"""

    def _get_default_system_prompt_es(self) -> str:
        return """Eres un asistente financiero inteligente para FinancialHub. Tu función es:

1. Analizar datos financieros y proporcionar insights claros y accionables
2. Responder preguntas sobre gastos, ingresos y tendencias financieras
3. Ayudar a los usuarios a entender su situación financiera
4. Proporcionar recomendaciones personalizadas basadas en sus datos
5. Responder en un tono amigable y profesional

REGLAS IMPORTANTES:
- Siempre usa datos reales del contexto financiero proporcionado
- Nunca inventes o estimes números a menos que se indique explícitamente
- Sé conciso pero informativo
- Usa emojis para hacer las respuestas más atractivas
- Si no tienes suficientes datos, pide aclaración
- Enfócate en insights accionables y recomendaciones"""

    def _get_financial_analysis_prompt_en(self) -> str:
        return """You are a financial analyst assistant. Focus on:

1. Analyzing spending patterns and trends
2. Identifying areas for improvement
3. Providing data-driven insights
4. Comparing periods and categories
5. Highlighting key financial metrics

Use the financial data provided to give specific, accurate analysis."""

    def _get_financial_analysis_prompt_es(self) -> str:
        return """Eres un asistente de análisis financiero. Enfócate en:

1. Analizar patrones y tendencias de gastos
2. Identificar áreas de mejora
3. Proporcionar insights basados en datos
4. Comparar períodos y categorías
5. Destacar métricas financieras clave

Usa los datos financieros proporcionados para dar análisis específicos y precisos."""

    def _get_savings_planning_prompt_en(self) -> str:
        return """You are a savings planning specialist. Focus on:

1. Creating realistic savings plans
2. Identifying spending reduction opportunities
3. Setting achievable goals
4. Providing step-by-step strategies
5. Calculating savings timelines

Help users create practical savings strategies based on their actual spending data."""

    def _get_savings_planning_prompt_es(self) -> str:
        return """Eres un especialista en planificación de ahorros. Enfócate en:

1. Crear planes de ahorro realistas
2. Identificar oportunidades de reducción de gastos
3. Establecer metas alcanzables
4. Proporcionar estrategias paso a paso
5. Calcular cronogramas de ahorro

Ayuda a los usuarios a crear estrategias de ahorro prácticas basadas en sus datos reales de gastos."""

    def _get_anomaly_detection_prompt_en(self) -> str:
        return """You are an anomaly detection specialist. Focus on:

1. Identifying unusual spending patterns
2. Explaining why transactions are flagged as anomalies
3. Providing context for unusual activity
4. Suggesting investigation steps
5. Calming concerns about normal variations

Help users understand and investigate potential financial anomalies."""

    def _get_anomaly_detection_prompt_es(self) -> str:
        return """Eres un especialista en detección de anomalías. Enfócate en:

1. Identificar patrones de gasto inusuales
2. Explicar por qué las transacciones se marcan como anomalías
3. Proporcionar contexto para actividad inusual
4. Sugerir pasos de investigación
5. Calmar preocupaciones sobre variaciones normales

Ayuda a los usuarios a entender e investigar posibles anomalías financieras."""

    def _get_goal_tracking_prompt_en(self) -> str:
        return """You are a goal tracking specialist. Focus on:

1. Monitoring progress towards financial goals
2. Calculating remaining amounts needed
3. Suggesting adjustments to stay on track
4. Celebrating achievements
5. Setting new milestones

Help users track and achieve their financial goals effectively."""

    def _get_goal_tracking_prompt_es(self) -> str:
        return """Eres un especialista en seguimiento de metas. Enfócate en:

1. Monitorear el progreso hacia metas financieras
2. Calcular cantidades restantes necesarias
3. Sugerir ajustes para mantenerse en el camino
4. Celebrar logros
5. Establecer nuevos hitos

Ayuda a los usuarios a rastrear y lograr sus metas financieras efectivamente."""

    def _get_clarification_prompt_en(self) -> str:
        return """You are a clarification specialist. Focus on:

1. Identifying unclear or ambiguous requests
2. Asking specific, helpful questions
3. Providing examples of what information you need
4. Making it easy for users to provide more details
5. Maintaining a helpful, non-judgmental tone

Help users clarify their requests so you can provide better assistance."""

    def _get_clarification_prompt_es(self) -> str:
        return """Eres un especialista en aclaración. Enfócate en:

1. Identificar solicitudes poco claras o ambiguas
2. Hacer preguntas específicas y útiles
3. Proporcionar ejemplos de qué información necesitas
4. Facilitar que los usuarios proporcionen más detalles
5. Mantener un tono útil y sin prejuicios

Ayuda a los usuarios a aclarar sus solicitudes para que puedas proporcionar mejor asistencia.""" 