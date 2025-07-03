"""
Follow-up Suggester Module

This module generates intelligent follow-up questions based on the user's last intent
and conversation context. It provides contextual suggestions to guide users toward
deeper financial insights and analysis.
"""

import logging
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class FollowUpSuggestion:
    """Represents a follow-up suggestion with metadata"""
    question: str
    intent: str
    confidence: float
    category: str
    reasoning: str


class FollowUpSuggester:
    """
    Intelligent follow-up question generator for financial conversations.
    
    This module analyzes the user's last intent and generates contextual
    follow-up questions to guide them toward deeper insights.
    """
    
    def __init__(self, language: str = 'en'):
        """
        Initialize the follow-up suggester.
        
        Args:
            language: Language for suggestions ('en' or 'es')
        """
        self.language = language
        self.suggestion_templates = self._load_suggestion_templates()
        self.context_memory = {}  # Store recent conversation context
    
    def _load_suggestion_templates(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Load suggestion templates for different intents and languages."""
        return {
            'en': {
                'balance_inquiry': [
                    {
                        'question': "Would you like to see your balance trend over the last 3 months?",
                        'intent': 'trend_analysis',
                        'confidence': 0.9,
                        'category': 'trend',
                        'reasoning': 'Balance inquiry often leads to trend analysis'
                    },
                    {
                        'question': "How does this compare to your balance last month?",
                        'intent': 'comparison',
                        'confidence': 0.8,
                        'category': 'comparison',
                        'reasoning': 'Natural progression from current balance to comparison'
                    },
                    {
                        'question': "Would you like to see a breakdown of your recent transactions?",
                        'intent': 'spending_analysis',
                        'confidence': 0.7,
                        'category': 'detail',
                        'reasoning': 'Balance often leads to transaction analysis'
                    }
                ],
                'spending_analysis': [
                    {
                        'question': "Which category has the highest spending this month?",
                        'intent': 'expense_categorization',
                        'confidence': 0.9,
                        'category': 'categorization',
                        'reasoning': 'Spending analysis naturally leads to category breakdown'
                    },
                    {
                        'question': "How does this month's spending compare to last month?",
                        'intent': 'comparison',
                        'confidence': 0.8,
                        'category': 'comparison',
                        'reasoning': 'Monthly spending comparison is common follow-up'
                    },
                    {
                        'question': "Would you like to set up spending alerts for unusual transactions?",
                        'intent': 'anomaly_detection',
                        'confidence': 0.7,
                        'category': 'monitoring',
                        'reasoning': 'Spending analysis can lead to monitoring setup'
                    }
                ],
                'savings_planning': [
                    {
                        'question': "What's your current savings rate compared to your goal?",
                        'intent': 'goal_tracking',
                        'confidence': 0.9,
                        'category': 'goal',
                        'reasoning': 'Savings planning naturally leads to goal tracking'
                    },
                    {
                        'question': "Would you like to see ways to increase your savings?",
                        'intent': 'budget_analysis',
                        'confidence': 0.8,
                        'category': 'optimization',
                        'reasoning': 'Savings planning often involves budget optimization'
                    },
                    {
                        'question': "How much could you save by reducing your top expense category?",
                        'intent': 'expense_categorization',
                        'confidence': 0.7,
                        'category': 'actionable',
                        'reasoning': 'Specific actionable advice based on spending'
                    }
                ],
                'trend_analysis': [
                    {
                        'question': "What factors are driving these trends?",
                        'intent': 'anomaly_detection',
                        'confidence': 0.8,
                        'category': 'analysis',
                        'reasoning': 'Trend analysis often leads to factor identification'
                    },
                    {
                        'question': "Would you like to see predictions for next month?",
                        'intent': 'prediction',
                        'confidence': 0.7,
                        'category': 'prediction',
                        'reasoning': 'Trends naturally lead to predictions'
                    },
                    {
                        'question': "How do these trends compare to industry averages?",
                        'intent': 'comparison',
                        'confidence': 0.6,
                        'category': 'benchmark',
                        'reasoning': 'Trends can be compared to benchmarks'
                    }
                ],
                'anomaly_detection': [
                    {
                        'question': "Would you like to investigate this anomaly further?",
                        'intent': 'spending_analysis',
                        'confidence': 0.9,
                        'category': 'investigation',
                        'reasoning': 'Anomalies often require deeper investigation'
                    },
                    {
                        'question': "Should we set up alerts for similar patterns?",
                        'intent': 'goal_tracking',
                        'confidence': 0.8,
                        'category': 'monitoring',
                        'reasoning': 'Anomaly detection leads to monitoring setup'
                    },
                    {
                        'question': "How does this compare to your normal spending patterns?",
                        'intent': 'comparison',
                        'confidence': 0.7,
                        'category': 'comparison',
                        'reasoning': 'Anomalies are defined by comparison to normal'
                    }
                ],
                'goal_tracking': [
                    {
                        'question': "What adjustments can help you reach your goal faster?",
                        'intent': 'budget_analysis',
                        'confidence': 0.9,
                        'category': 'optimization',
                        'reasoning': 'Goal tracking often leads to optimization suggestions'
                    },
                    {
                        'question': "Would you like to set up automatic goal contributions?",
                        'intent': 'savings_planning',
                        'confidence': 0.8,
                        'category': 'automation',
                        'reasoning': 'Goals often benefit from automation'
                    },
                    {
                        'question': "How does your progress compare to similar users?",
                        'intent': 'comparison',
                        'confidence': 0.6,
                        'category': 'benchmark',
                        'reasoning': 'Goal progress can be benchmarked'
                    }
                ],
                'comparison': [
                    {
                        'question': "What factors contributed to these differences?",
                        'intent': 'trend_analysis',
                        'confidence': 0.8,
                        'category': 'analysis',
                        'reasoning': 'Comparisons often lead to factor analysis'
                    },
                    {
                        'question': "Would you like to see predictions for the next period?",
                        'intent': 'prediction',
                        'confidence': 0.7,
                        'category': 'prediction',
                        'reasoning': 'Comparisons can inform predictions'
                    },
                    {
                        'question': "How can you improve based on this comparison?",
                        'intent': 'budget_analysis',
                        'confidence': 0.6,
                        'category': 'actionable',
                        'reasoning': 'Comparisons should lead to actionable insights'
                    }
                ],
                'prediction': [
                    {
                        'question': "What actions can you take to improve this prediction?",
                        'intent': 'budget_analysis',
                        'confidence': 0.9,
                        'category': 'actionable',
                        'reasoning': 'Predictions should lead to actionable steps'
                    },
                    {
                        'question': "Would you like to set up monitoring for these predictions?",
                        'intent': 'goal_tracking',
                        'confidence': 0.8,
                        'category': 'monitoring',
                        'reasoning': 'Predictions benefit from monitoring'
                    },
                    {
                        'question': "How confident are you in these predictions?",
                        'intent': 'clarification_request',
                        'confidence': 0.7,
                        'category': 'confidence',
                        'reasoning': 'Prediction confidence is important'
                    }
                ]
            },
            'es': {
                'consulta_saldo': [
                    {
                        'question': "¿Te gustaría ver la tendencia de tu saldo en los últimos 3 meses?",
                        'intent': 'analisis_tendencias',
                        'confidence': 0.9,
                        'category': 'tendencia',
                        'reasoning': 'La consulta de saldo suele llevar al análisis de tendencias'
                    },
                    {
                        'question': "¿Cómo se compara con tu saldo del mes pasado?",
                        'intent': 'comparacion',
                        'confidence': 0.8,
                        'category': 'comparacion',
                        'reasoning': 'Progresión natural del saldo actual a la comparación'
                    },
                    {
                        'question': "¿Te gustaría ver un desglose de tus transacciones recientes?",
                        'intent': 'analisis_gastos',
                        'confidence': 0.7,
                        'category': 'detalle',
                        'reasoning': 'El saldo suele llevar al análisis de transacciones'
                    }
                ],
                'analisis_gastos': [
                    {
                        'question': "¿Qué categoría tiene el gasto más alto este mes?",
                        'intent': 'categorizacion_gastos',
                        'confidence': 0.9,
                        'category': 'categorizacion',
                        'reasoning': 'El análisis de gastos naturalmente lleva al desglose por categorías'
                    },
                    {
                        'question': "¿Cómo se comparan los gastos de este mes con el anterior?",
                        'intent': 'comparacion',
                        'confidence': 0.8,
                        'category': 'comparacion',
                        'reasoning': 'La comparación mensual de gastos es un seguimiento común'
                    },
                    {
                        'question': "¿Te gustaría configurar alertas para transacciones inusuales?",
                        'intent': 'deteccion_anomalias',
                        'confidence': 0.7,
                        'category': 'monitoreo',
                        'reasoning': 'El análisis de gastos puede llevar a la configuración de monitoreo'
                    }
                ],
                'planificacion_ahorro': [
                    {
                        'question': "¿Cuál es tu tasa de ahorro actual comparada con tu meta?",
                        'intent': 'seguimiento_metas',
                        'confidence': 0.9,
                        'category': 'meta',
                        'reasoning': 'La planificación de ahorro naturalmente lleva al seguimiento de metas'
                    },
                    {
                        'question': "¿Te gustaría ver formas de aumentar tus ahorros?",
                        'intent': 'analisis_presupuesto',
                        'confidence': 0.8,
                        'category': 'optimizacion',
                        'reasoning': 'La planificación de ahorro suele involucrar optimización de presupuesto'
                    },
                    {
                        'question': "¿Cuánto podrías ahorrar reduciendo tu categoría de gasto principal?",
                        'intent': 'categorizacion_gastos',
                        'confidence': 0.7,
                        'category': 'accionable',
                        'reasoning': 'Consejo accionable específico basado en gastos'
                    }
                ],
                'analisis_tendencias': [
                    {
                        'question': "¿Qué factores están impulsando estas tendencias?",
                        'intent': 'deteccion_anomalias',
                        'confidence': 0.8,
                        'category': 'analisis',
                        'reasoning': 'El análisis de tendencias suele llevar a la identificación de factores'
                    },
                    {
                        'question': "¿Te gustaría ver predicciones para el próximo mes?",
                        'intent': 'prediccion',
                        'confidence': 0.7,
                        'category': 'prediccion',
                        'reasoning': 'Las tendencias naturalmente llevan a predicciones'
                    },
                    {
                        'question': "¿Cómo se comparan estas tendencias con los promedios de la industria?",
                        'intent': 'comparacion',
                        'confidence': 0.6,
                        'category': 'benchmark',
                        'reasoning': 'Las tendencias se pueden comparar con benchmarks'
                    }
                ],
                'deteccion_anomalias': [
                    {
                        'question': "¿Te gustaría investigar esta anomalía más a fondo?",
                        'intent': 'analisis_gastos',
                        'confidence': 0.9,
                        'category': 'investigacion',
                        'reasoning': 'Las anomalías suelen requerir investigación más profunda'
                    },
                    {
                        'question': "¿Deberíamos configurar alertas para patrones similares?",
                        'intent': 'seguimiento_metas',
                        'confidence': 0.8,
                        'category': 'monitoreo',
                        'reasoning': 'La detección de anomalías lleva a la configuración de monitoreo'
                    },
                    {
                        'question': "¿Cómo se compara con tus patrones de gasto normales?",
                        'intent': 'comparacion',
                        'confidence': 0.7,
                        'category': 'comparacion',
                        'reasoning': 'Las anomalías se definen por comparación con lo normal'
                    }
                ],
                'seguimiento_metas': [
                    {
                        'question': "¿Qué ajustes pueden ayudarte a alcanzar tu meta más rápido?",
                        'intent': 'analisis_presupuesto',
                        'confidence': 0.9,
                        'category': 'optimizacion',
                        'reasoning': 'El seguimiento de metas suele llevar a sugerencias de optimización'
                    },
                    {
                        'question': "¿Te gustaría configurar contribuciones automáticas a tu meta?",
                        'intent': 'planificacion_ahorro',
                        'confidence': 0.8,
                        'category': 'automatizacion',
                        'reasoning': 'Las metas suelen beneficiarse de la automatización'
                    },
                    {
                        'question': "¿Cómo se compara tu progreso con usuarios similares?",
                        'intent': 'comparacion',
                        'confidence': 0.6,
                        'category': 'benchmark',
                        'reasoning': 'El progreso de metas se puede comparar'
                    }
                ],
                'comparacion': [
                    {
                        'question': "¿Qué factores contribuyeron a estas diferencias?",
                        'intent': 'analisis_tendencias',
                        'confidence': 0.8,
                        'category': 'analisis',
                        'reasoning': 'Las comparaciones suelen llevar al análisis de factores'
                    },
                    {
                        'question': "¿Te gustaría ver predicciones para el próximo período?",
                        'intent': 'prediccion',
                        'confidence': 0.7,
                        'category': 'prediccion',
                        'reasoning': 'Las comparaciones pueden informar predicciones'
                    },
                    {
                        'question': "¿Cómo puedes mejorar basándote en esta comparación?",
                        'intent': 'analisis_presupuesto',
                        'confidence': 0.6,
                        'category': 'accionable',
                        'reasoning': 'Las comparaciones deben llevar a insights accionables'
                    }
                ],
                'prediccion': [
                    {
                        'question': "¿Qué acciones puedes tomar para mejorar esta predicción?",
                        'intent': 'analisis_presupuesto',
                        'confidence': 0.9,
                        'category': 'accionable',
                        'reasoning': 'Las predicciones deben llevar a pasos accionables'
                    },
                    {
                        'question': "¿Te gustaría configurar monitoreo para estas predicciones?",
                        'intent': 'seguimiento_metas',
                        'confidence': 0.8,
                        'category': 'monitoreo',
                        'reasoning': 'Las predicciones se benefician del monitoreo'
                    },
                    {
                        'question': "¿Qué tan confiado estás en estas predicciones?",
                        'intent': 'solicitud_aclaracion',
                        'confidence': 0.7,
                        'category': 'confianza',
                        'reasoning': 'La confianza en las predicciones es importante'
                    }
                ]
            }
        }
    
    def generate_followup_suggestions(self, 
                                    last_intent: str, 
                                    context: Dict[str, Any] = None,
                                    num_suggestions: int = 3) -> List[FollowUpSuggestion]:
        """
        Generate follow-up suggestions based on the last intent.
        
        Args:
            last_intent: The last detected intent
            context: Additional context information
            num_suggestions: Number of suggestions to generate
            
        Returns:
            List of FollowUpSuggestion objects
        """
        if not context:
            context = {}
        
        # Get templates for the current language and intent
        templates = self.suggestion_templates.get(self.language, {})
        intent_templates = templates.get(last_intent, [])
        
        if not intent_templates:
            # Fallback to general suggestions
            intent_templates = self._get_general_suggestions()
        
        # Filter and rank suggestions based on context
        ranked_suggestions = self._rank_suggestions(intent_templates, context)
        
        # Select top suggestions
        selected_suggestions = ranked_suggestions[:num_suggestions]
        
        # Convert to FollowUpSuggestion objects
        suggestions = []
        for suggestion_data in selected_suggestions:
            suggestion = FollowUpSuggestion(
                question=suggestion_data['question'],
                intent=suggestion_data['intent'],
                confidence=suggestion_data['confidence'],
                category=suggestion_data['category'],
                reasoning=suggestion_data['reasoning']
            )
            suggestions.append(suggestion)
        
        # Store context for future use
        self._update_context_memory(last_intent, context)
        
        return suggestions
    
    def _rank_suggestions(self, 
                         templates: List[Dict[str, Any]], 
                         context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rank suggestions based on context and relevance.
        
        Args:
            templates: List of suggestion templates
            context: Current conversation context
            
        Returns:
            Ranked list of suggestion templates
        """
        ranked = []
        
        for template in templates:
            score = template['confidence']
            
            # Boost score based on context relevance
            if context.get('has_goals') and template['category'] == 'goal':
                score += 0.1
            
            if context.get('has_anomalies') and template['category'] == 'investigation':
                score += 0.1
            
            if context.get('is_comparative') and template['category'] == 'comparison':
                score += 0.1
            
            if context.get('needs_action') and template['category'] == 'actionable':
                score += 0.1
            
            # Reduce score for recently used suggestions
            if self._was_recently_used(template['question']):
                score -= 0.2
            
            ranked.append((score, template))
        
        # Sort by score (descending) and return templates
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [template for score, template in ranked]
    
    def _get_general_suggestions(self) -> List[Dict[str, Any]]:
        """Get general suggestions when no specific intent templates are available."""
        general_templates = {
            'en': [
                {
                    'question': "Would you like to see a summary of your financial health?",
                    'intent': 'general_inquiry',
                    'confidence': 0.6,
                    'category': 'summary',
                    'reasoning': 'General inquiry often leads to summary'
                },
                {
                    'question': "Is there a specific financial area you'd like to explore?",
                    'intent': 'clarification_request',
                    'confidence': 0.5,
                    'category': 'clarification',
                    'reasoning': 'General inquiry may need clarification'
                },
                {
                    'question': "Would you like to set up financial goals?",
                    'intent': 'goal_tracking',
                    'confidence': 0.4,
                    'category': 'setup',
                    'reasoning': 'General inquiry can lead to goal setup'
                }
            ],
            'es': [
                {
                    'question': "¿Te gustaría ver un resumen de tu salud financiera?",
                    'intent': 'consulta_general',
                    'confidence': 0.6,
                    'category': 'resumen',
                    'reasoning': 'La consulta general suele llevar a un resumen'
                },
                {
                    'question': "¿Hay algún área financiera específica que te gustaría explorar?",
                    'intent': 'solicitud_aclaracion',
                    'confidence': 0.5,
                    'category': 'aclaracion',
                    'reasoning': 'La consulta general puede necesitar aclaración'
                },
                {
                    'question': "¿Te gustaría configurar metas financieras?",
                    'intent': 'seguimiento_metas',
                    'confidence': 0.4,
                    'category': 'configuracion',
                    'reasoning': 'La consulta general puede llevar a la configuración de metas'
                }
            ]
        }
        
        return general_templates.get(self.language, general_templates['en'])
    
    def _update_context_memory(self, intent: str, context: Dict[str, Any]):
        """Update the context memory with recent conversation information."""
        timestamp = datetime.now()
        
        # Store recent intent
        if 'recent_intents' not in self.context_memory:
            self.context_memory['recent_intents'] = []
        
        self.context_memory['recent_intents'].append({
            'intent': intent,
            'timestamp': timestamp,
            'context': context
        })
        
        # Keep only last 10 intents
        self.context_memory['recent_intents'] = self.context_memory['recent_intents'][-10:]
        
        # Store used suggestions
        if 'used_suggestions' not in self.context_memory:
            self.context_memory['used_suggestions'] = []
        
        # Add current suggestions to used list (will be populated by caller)
        self.context_memory['used_suggestions'].append({
            'timestamp': timestamp,
            'intent': intent
        })
        
        # Keep only last 20 used suggestions
        self.context_memory['used_suggestions'] = self.context_memory['used_suggestions'][-20:]
    
    def _was_recently_used(self, question: str) -> bool:
        """Check if a suggestion was recently used."""
        if 'used_suggestions' not in self.context_memory:
            return False
        
        # Check if this question was used in the last 5 minutes
        cutoff_time = datetime.now() - timedelta(minutes=5)
        
        for used in self.context_memory['used_suggestions']:
            if used['timestamp'] > cutoff_time:
                # This is a simplified check - in practice you'd store the actual questions
                return True
        
        return False
    
    def get_conversation_context(self) -> Dict[str, Any]:
        """Get the current conversation context for analysis."""
        return {
            'recent_intents': self.context_memory.get('recent_intents', []),
            'used_suggestions': self.context_memory.get('used_suggestions', []),
            'conversation_length': len(self.context_memory.get('recent_intents', [])),
            'last_intent': self.context_memory.get('recent_intents', [{}])[-1].get('intent') if self.context_memory.get('recent_intents') else None
        }
    
    def clear_context_memory(self):
        """Clear the conversation context memory."""
        self.context_memory = {}
        logger.info("Conversation context memory cleared")
    
    def add_custom_suggestion(self, 
                            intent: str, 
                            question: str, 
                            target_intent: str,
                            category: str = 'custom',
                            confidence: float = 0.5):
        """
        Add a custom suggestion template.
        
        Args:
            intent: The intent this suggestion follows
            question: The follow-up question
            target_intent: The intent this suggestion leads to
            category: Category of the suggestion
            confidence: Confidence score for the suggestion
        """
        if self.language not in self.suggestion_templates:
            self.suggestion_templates[self.language] = {}
        
        if intent not in self.suggestion_templates[self.language]:
            self.suggestion_templates[self.language][intent] = []
        
        custom_suggestion = {
            'question': question,
            'intent': target_intent,
            'confidence': confidence,
            'category': category,
            'reasoning': f'Custom suggestion for {intent}'
        }
        
        self.suggestion_templates[self.language][intent].append(custom_suggestion)
        logger.info(f"Added custom suggestion for intent '{intent}': {question}") 