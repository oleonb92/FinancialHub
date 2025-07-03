"""
Enhanced Financial Query Parser with Confidence Scoring

This module provides:
- Semantic analysis of financial questions
- Entity extraction (time, categories, amounts, etc.)
- Query type classification with confidence scoring
- User intent detection with fallback logic
- Softmax-based confidence calculation
"""
import re
import math
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParsedIntent:
    """Represents a parsed intent with confidence scoring"""
    intent_type: str
    confidence_score: float
    entities: Dict[str, Any]
    metadata: Dict[str, Any]


class FinancialQueryParser:
    """Enhanced parser for analyzing financial questions with confidence scoring."""
    
    def __init__(self):
        self.month_mapping = {
            'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
            'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12,
            'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
            'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        self.time_keywords = {
            'hoy': 'today', 'ayer': 'yesterday', 'mañana': 'tomorrow',
            'esta semana': 'this_week', 'semana pasada': 'last_week',
            'este mes': 'this_month', 'mes pasado': 'last_month',
            'este año': 'this_year', 'año pasado': 'last_year',
            'este trimestre': 'this_quarter', 'trimestre pasado': 'last_quarter',
            'último mes': 'last_month', 'próximo mes': 'next_month'
        }
        
        self.comparison_keywords = {
            'más alto': 'max', 'más bajo': 'min', 'máximo': 'max', 'mínimo': 'min',
            'mejor': 'max', 'peor': 'min', 'mayor': 'max', 'menor': 'min',
            'promedio': 'avg', 'promedio': 'average', 'total': 'sum',
            'top': 'top', 'peor': 'worst', 'mejor': 'best'
        }
        
        self.financial_keywords = {
            'ingresos': 'income', 'gastos': 'expenses', 'balance': 'balance',
            'ahorro': 'savings', 'deuda': 'debt', 'inversión': 'investment',
            'presupuesto': 'budget', 'meta': 'goal', 'categoría': 'category',
            'cuenta': 'account', 'transacción': 'transaction', 'saldo': 'balance'
        }
        
        # Intent patterns with weights for confidence scoring
        self.intent_patterns = {
            'balance_inquiry': {
                'keywords': ['balance', 'saldo', 'cuánto tengo', 'how much do i have'],
                'weight': 0.8
            },
            'spending_analysis': {
                'keywords': ['gasté', 'spent', 'gastos', 'expenses', 'cuánto gasté'],
                'weight': 0.9
            },
            'savings_planning': {
                'keywords': ['ahorrar', 'save', 'savings', 'plan de ahorro'],
                'weight': 0.95
            },
            'trend_analysis': {
                'keywords': ['tendencia', 'trend', 'cambio', 'change', 'evolución'],
                'weight': 0.7
            },
            'anomaly_detection': {
                'keywords': ['anomalía', 'anomaly', 'inusual', 'unusual', 'extraño'],
                'weight': 0.8
            },
            'goal_tracking': {
                'keywords': ['meta', 'goal', 'objetivo', 'target', 'progreso'],
                'weight': 0.85
            },
            'comparison': {
                'keywords': ['comparar', 'compare', 'vs', 'versus', 'respecto'],
                'weight': 0.75
            },
            'prediction': {
                'keywords': ['predicción', 'prediction', 'futuro', 'future', 'próximo'],
                'weight': 0.6
            }
        }
    
    def parse_query(self, message: str) -> ParsedIntent:
        """
        Parse the query and extract relevant entities with confidence scoring.
        
        Args:
            message: User query string
            
        Returns:
            ParsedIntent object with confidence score and extracted entities
        """
        message_lower = message.lower()
        
        # Extract entities
        entities = {
            'time_period': self._extract_time_period(message_lower),
            'comparison_type': self._extract_comparison_type(message_lower),
            'financial_entity': self._extract_financial_entity(message_lower),
            'categories': self._extract_categories(message_lower),
            'accounts': self._extract_accounts(message_lower),
            'users': self._extract_users(message_lower),
            'amount_range': self._extract_amount_range(message_lower),
            'is_historical': self._is_historical_query(message_lower),
            'is_comparative': self._is_comparative_query(message_lower),
            'is_analytical': self._is_analytical_query(message_lower),
            'is_trend_analysis': self._is_trend_analysis_query(message_lower),
            'is_anomaly_detection': self._is_anomaly_detection_query(message_lower)
        }
        
        # Determine intent and confidence
        intent_type, confidence_score = self._determine_intent_with_confidence(message_lower)
        
        # Add metadata
        metadata = {
            'query_length': len(message),
            'has_numbers': bool(re.search(r'\d', message)),
            'has_currency': bool(re.search(r'\$', message)),
            'language': self._detect_language(message),
            'complexity_score': self._calculate_complexity_score(message)
        }
        
        return ParsedIntent(
            intent_type=intent_type,
            confidence_score=confidence_score,
            entities=entities,
            metadata=metadata
        )
    
    def _determine_intent_with_confidence(self, message: str) -> Tuple[str, float]:
        """
        Determine the primary intent and calculate confidence score using softmax.
        
        Args:
            message: Lowercase message
            
        Returns:
            Tuple of (intent_type, confidence_score)
        """
        intent_scores = {}
        
        # Calculate raw scores for each intent
        for intent, pattern in self.intent_patterns.items():
            score = 0.0
            for keyword in pattern['keywords']:
                if keyword in message:
                    score += pattern['weight']
            
            # Bonus for multiple keyword matches
            if score > 0:
                score *= 1.2
            
            intent_scores[intent] = score
        
        # Apply softmax to get confidence scores
        confidence_scores = self._softmax(intent_scores)
        
        # Find the highest scoring intent
        if confidence_scores:
            best_intent = max(confidence_scores, key=confidence_scores.get)
            best_score = confidence_scores[best_intent]
            
            # Apply fallback logic for low confidence
            if best_score < 0.3:
                return 'clarification_request', 0.5
            
            return best_intent, best_score
        
        # Default fallback
        return 'general_inquiry', 0.4
    
    def _softmax(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Apply softmax function to convert scores to probabilities.
        
        Args:
            scores: Dictionary of intent scores
            
        Returns:
            Dictionary of confidence scores (probabilities)
        """
        if not scores:
            return {}
        
        # Find maximum score to prevent overflow
        max_score = max(scores.values()) if scores.values() else 0
        
        # Apply softmax
        exp_scores = {intent: math.exp(score - max_score) for intent, score in scores.items()}
        sum_exp_scores = sum(exp_scores.values())
        
        if sum_exp_scores == 0:
            return {intent: 1.0 / len(scores) for intent in scores.keys()}
        
        return {intent: exp_score / sum_exp_scores for intent, exp_score in exp_scores.items()}
    
    def _detect_language(self, message: str) -> str:
        """Detect the language of the message."""
        spanish_words = ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las', 'una', 'como', 'más', 'pero', 'sus', 'me', 'hasta', 'hay', 'donde', 'han', 'quien', 'están', 'estado', 'desde', 'todo', 'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante', 'ellos', 'e', 'esto', 'mí', 'antes', 'algunos', 'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos', 'cual', 'poco', 'ella', 'estar', 'estas', 'algunas', 'algo', 'nosotros']
        english_words = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us']
        
        spanish_count = sum(1 for word in message.split() if word.lower() in spanish_words)
        english_count = sum(1 for word in message.split() if word.lower() in english_words)
        
        if spanish_count > english_count:
            return 'es'
        elif english_count > spanish_count:
            return 'en'
        else:
            return 'unknown'
    
    def _calculate_complexity_score(self, message: str) -> float:
        """Calculate a complexity score for the query."""
        score = 0.0
        
        # Length factor
        score += min(len(message.split()) / 20.0, 1.0) * 0.3
        
        # Number of entities
        entities = ['time_period', 'comparison_type', 'financial_entity', 'amount_range']
        entity_count = sum(1 for entity in entities if self._has_entity(message, entity))
        score += min(entity_count / 4.0, 1.0) * 0.4
        
        # Question complexity
        if '?' in message:
            score += 0.2
        
        # Multiple questions
        question_count = message.count('?')
        score += min(question_count / 3.0, 1.0) * 0.1
        
        return min(score, 1.0)
    
    def _has_entity(self, message: str, entity_type: str) -> bool:
        """Check if message has a specific entity type."""
        if entity_type == 'time_period':
            return any(keyword in message for keyword in self.time_keywords.keys())
        elif entity_type == 'comparison_type':
            return any(keyword in message for keyword in self.comparison_keywords.keys())
        elif entity_type == 'financial_entity':
            return any(keyword in message for keyword in self.financial_keywords.keys())
        elif entity_type == 'amount_range':
            return bool(re.search(r'\$?\d+(?:,\d{3})*(?:\.\d{2})?', message))
        return False
    
    def _extract_time_period(self, message: str) -> Dict[str, Any]:
        """Extrae información de tiempo de la pregunta."""
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # Buscar meses específicos
        for month_name, month_num in self.month_mapping.items():
            if month_name in message:
                year_match = re.search(r'20\d{2}', message)
                year = int(year_match.group()) if year_match else current_year
                if month_num > current_month and not year_match:
                    year = current_year - 1
                return {'type': 'specific_month', 'month': month_num, 'year': year}
        
        # Buscar palabras clave de tiempo
        for keyword, period in self.time_keywords.items():
            if keyword in message:
                return {'type': 'relative', 'period': period}
        
        # Buscar rangos de fechas
        if 'últimos' in message or 'pasados' in message:
            number_match = re.search(r'(\d+)', message)
            if number_match:
                number = int(number_match.group())
                if 'días' in message:
                    return {'type': 'days_back', 'days': number}
                elif 'semanas' in message:
                    return {'type': 'weeks_back', 'weeks': number}
                elif 'meses' in message:
                    return {'type': 'months_back', 'months': number}
        
        return {'type': 'current', 'month': current_month, 'year': current_year}
    
    def _extract_comparison_type(self, message: str) -> Optional[str]:
        """Extrae el tipo de comparación solicitada."""
        for keyword, comp_type in self.comparison_keywords.items():
            if keyword in message:
                return comp_type
        return None
    
    def _extract_financial_entity(self, message: str) -> Optional[str]:
        """Extrae la entidad financiera principal."""
        for keyword, entity in self.financial_keywords.items():
            if keyword in message:
                return entity
        return None
    
    def _extract_categories(self, message: str) -> List[str]:
        """Extrae categorías mencionadas."""
        # Implementar lógica para detectar categorías específicas
        return []
    
    def _extract_accounts(self, message: str) -> List[str]:
        """Extrae cuentas mencionadas."""
        # Implementar lógica para detectar cuentas específicas
        return []
    
    def _extract_users(self, message: str) -> List[str]:
        """Extrae usuarios mencionados."""
        # Implementar lógica para detectar usuarios específicos
        return []
    
    def _extract_amount_range(self, message: str) -> Optional[Dict[str, float]]:
        """Extrae rangos de montos mencionados."""
        amount_pattern = r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)'
        amounts = re.findall(amount_pattern, message)
        if amounts:
            amounts = [float(amt.replace(',', '')) for amt in amounts]
            return {'min': min(amounts), 'max': max(amounts)}
        return None
    
    def _is_historical_query(self, message: str) -> bool:
        """Determina si es una pregunta histórica."""
        historical_keywords = ['histórico', 'historia', 'pasado', 'anterior', 'último', 'tendencia']
        return any(keyword in message for keyword in historical_keywords)
    
    def _is_comparative_query(self, message: str) -> bool:
        """Determina si es una pregunta comparativa."""
        comparative_keywords = ['comparar', 'comparación', 'vs', 'versus', 'respecto', 'en comparación']
        return any(keyword in message for keyword in comparative_keywords)
    
    def _is_analytical_query(self, message: str) -> bool:
        """Determina si es una pregunta analítica."""
        analytical_keywords = ['análisis', 'analizar', 'por qué', 'causa', 'razón', 'tendencia', 'patrón']
        return any(keyword in message for keyword in analytical_keywords)
    
    def _is_trend_analysis_query(self, message: str) -> bool:
        """Determina si es una pregunta de análisis de tendencias."""
        trend_keywords = ['tendencia', 'evolución', 'crecimiento', 'decrecimiento', 'cambio']
        return any(keyword in message for keyword in trend_keywords)
    
    def _is_anomaly_detection_query(self, message: str) -> bool:
        """Determina si es una pregunta de detección de anomalías."""
        anomaly_keywords = ['inusual', 'extraño', 'anómalo', 'diferente', 'sospechoso']
        return any(keyword in message for keyword in anomaly_keywords) 