"""
Enhanced Financial Query Parser

This module provides an enhanced version of the FinancialQueryParser with:
- spaCy integration for better entity extraction
- dateparser for parsing relative dates
- Improved entity recognition and confidence scoring
- Better multilingual support
"""

import re
import logging
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import unicodedata

# Try to import spaCy
try:
    import spacy
    SPACY_AVAILABLE = True
    # Load English model
    try:
        nlp_en = spacy.load("en_core_web_sm")
    except OSError:
        nlp_en = None
        logging.warning("spaCy English model not found. Run: python -m spacy download en_core_web_sm")
    
    # Load Spanish model
    try:
        nlp_es = spacy.load("es_core_news_sm")
    except OSError:
        nlp_es = None
        logging.warning("spaCy Spanish model not found. Run: python -m spacy download es_core_news_sm")
        
except ImportError:
    SPACY_AVAILABLE = False
    nlp_en = None
    nlp_es = None
    logging.warning("spaCy not available, using basic entity extraction")

# Try to import dateparser
try:
    import dateparser
    DATEPARSER_AVAILABLE = True
except ImportError:
    DATEPARSER_AVAILABLE = False
    logging.warning("dateparser not available, using basic date parsing")

logger = logging.getLogger(__name__)


@dataclass
class EnhancedParsedIntent:
    """Enhanced parsed intent with detailed entity information"""
    intent_type: str
    confidence_score: float
    entities: Dict[str, Any]
    metadata: Dict[str, Any]
    extracted_dates: List[Dict[str, Any]]
    extracted_amounts: List[Dict[str, Any]]
    extracted_categories: List[str]
    extracted_accounts: List[str]
    language: str
    processing_time: float
    multiple_questions: bool = False
    question_parts: List[str] = None


class EnhancedFinancialQueryParser:
    """
    Enhanced financial query parser with spaCy and dateparser integration.
    
    This parser provides superior entity extraction and date parsing capabilities
    compared to the basic keyword-based approach.
    """
    
    def __init__(self, language: str = 'en'):
        """
        Initialize the enhanced query parser.
        
        Args:
            language: Default language for parsing ('en' or 'es')
        """
        self.language = language
        self.nlp = self._get_spacy_model(language)
        
        # Enhanced entity patterns
        self.entity_patterns = self._load_entity_patterns()
        
        # Date parsing configuration
        self.date_settings = {
            'PREFER_DAY_OF_MONTH': 'first',
            'PREFER_DATES_FROM': 'past',
            'RELATIVE_BASE': datetime.now()
        }
        
        # Amount patterns with currency support
        self.amount_patterns = {
            'en': [
                r'\$[\d,]+(?:\.\d{2})?',
                r'[\d,]+(?:\.\d{2})?\s*(?:dollars?|USD)',
                r'[\d,]+(?:\.\d{2})?\s*(?:euros?|EUR)',
                r'[\d,]+(?:\.\d{2})?\s*(?:pounds?|GBP)'
            ],
            'es': [
                r'\$[\d,]+(?:\.\d{2})?',
                r'[\d,]+(?:\.\d{2})?\s*(?:dólares?|USD)',
                r'[\d,]+(?:\.\d{2})?\s*(?:euros?|EUR)',
                r'[\d,]+(?:\.\d{2})?\s*(?:pesos?|MXN)'
            ]
        }
        
        # Category mapping
        self.category_mapping = {
            'en': {
                'food': ['food', 'restaurant', 'dining', 'groceries', 'meal', 'lunch', 'dinner', 'breakfast'],
                'transportation': ['transport', 'gas', 'fuel', 'uber', 'lyft', 'taxi', 'bus', 'train', 'metro'],
                'entertainment': ['entertainment', 'movie', 'theater', 'concert', 'game', 'sport', 'hobby'],
                'shopping': ['shopping', 'clothes', 'electronics', 'amazon', 'walmart', 'target'],
                'utilities': ['utility', 'electricity', 'water', 'gas', 'internet', 'phone', 'cable'],
                'healthcare': ['health', 'medical', 'doctor', 'pharmacy', 'insurance', 'dental'],
                'education': ['education', 'school', 'university', 'course', 'book', 'tuition'],
                'housing': ['rent', 'mortgage', 'home', 'house', 'apartment', 'maintenance']
            },
            'es': {
                'comida': ['comida', 'restaurante', 'cena', 'almuerzo', 'desayuno', 'supermercado', 'grocery'],
                'transporte': ['transporte', 'gasolina', 'uber', 'taxi', 'autobús', 'metro', 'tren'],
                'entretenimiento': ['entretenimiento', 'película', 'teatro', 'concierto', 'juego', 'deporte'],
                'compras': ['compras', 'ropa', 'electrónicos', 'amazon', 'walmart', 'target'],
                'servicios': ['servicios', 'electricidad', 'agua', 'gas', 'internet', 'teléfono', 'cable'],
                'salud': ['salud', 'médico', 'farmacia', 'seguro', 'dental'],
                'educación': ['educación', 'escuela', 'universidad', 'curso', 'libro', 'matrícula'],
                'vivienda': ['renta', 'hipoteca', 'casa', 'apartamento', 'mantenimiento']
            }
        }
    
    def _get_spacy_model(self, language: str):
        """Get the appropriate spaCy model for the language."""
        if not SPACY_AVAILABLE:
            return None
        
        if language == 'en' and nlp_en:
            return nlp_en
        elif language == 'es' and nlp_es:
            return nlp_es
        else:
            # Fallback to English if Spanish not available
            return nlp_en
    
    def _load_entity_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Load enhanced entity patterns for different languages."""
        return {
            'en': {
                'time_keywords': {
                    'today': ['today', 'now', 'current', 'this day'],
                    'yesterday': ['yesterday', 'last day'],
                    'tomorrow': ['tomorrow', 'next day'],
                    'this_week': ['this week', 'current week', 'this week'],
                    'last_week': ['last week', 'previous week', 'past week'],
                    'this_month': ['this month', 'current month'],
                    'last_month': ['last month', 'previous month', 'past month'],
                    'this_year': ['this year', 'current year'],
                    'last_year': ['last year', 'previous year', 'past year'],
                    'this_quarter': ['this quarter', 'current quarter'],
                    'last_quarter': ['last quarter', 'previous quarter']
                },
                'comparison_keywords': {
                    'more': ['more', 'higher', 'greater', 'above', 'over'],
                    'less': ['less', 'lower', 'below', 'under', 'fewer'],
                    'same': ['same', 'equal', 'similar', 'comparable'],
                    'different': ['different', 'unlike', 'varying', 'diverse']
                },
                'financial_keywords': {
                    'income': ['income', 'salary', 'wage', 'earnings', 'revenue'],
                    'expense': ['expense', 'cost', 'spending', 'payment', 'bill'],
                    'balance': ['balance', 'account', 'funds', 'money', 'cash'],
                    'savings': ['savings', 'save', 'saved', 'reserve', 'emergency fund'],
                    'debt': ['debt', 'loan', 'credit', 'owe', 'borrow'],
                    'investment': ['investment', 'invest', 'stock', 'bond', 'portfolio']
                }
            },
            'es': {
                'time_keywords': {
                    'hoy': ['hoy', 'actual', 'actualmente'],
                    'ayer': ['ayer', 'pasado'],
                    'mañana': ['mañana', 'próximo'],
                    'esta_semana': ['esta semana', 'semana actual'],
                    'semana_pasada': ['semana pasada', 'semana anterior'],
                    'este_mes': ['este mes', 'mes actual'],
                    'mes_pasado': ['mes pasado', 'mes anterior'],
                    'este_año': ['este año', 'año actual'],
                    'año_pasado': ['año pasado', 'año anterior'],
                    'este_trimestre': ['este trimestre', 'trimestre actual'],
                    'trimestre_pasado': ['trimestre pasado', 'trimestre anterior']
                },
                'comparison_keywords': {
                    'más': ['más', 'mayor', 'superior', 'alto'],
                    'menos': ['menos', 'menor', 'inferior', 'bajo'],
                    'igual': ['igual', 'mismo', 'similar', 'comparable'],
                    'diferente': ['diferente', 'distinto', 'vario', 'diverso']
                },
                'financial_keywords': {
                    'ingreso': ['ingreso', 'salario', 'sueldo', 'ganancia', 'renta'],
                    'gasto': ['gasto', 'costo', 'pago', 'factura', 'cuenta'],
                    'saldo': ['saldo', 'cuenta', 'fondos', 'dinero', 'efectivo'],
                    'ahorro': ['ahorro', 'ahorrar', 'reserva', 'fondo de emergencia'],
                    'deuda': ['deuda', 'préstamo', 'crédito', 'deber', 'pedir prestado'],
                    'inversión': ['inversión', 'invertir', 'acción', 'bono', 'portafolio']
                }
            }
        }
    
    def parse_query(self, message: str, language: str = None) -> EnhancedParsedIntent:
        """
        Parse the query with enhanced entity extraction.
        
        Args:
            message: User query string
            language: Language for parsing (overrides default)
            
        Returns:
            EnhancedParsedIntent object with detailed information
        """
        import time
        start_time = time.time()
        
        # Determine language
        if language is None:
            language = self._detect_language(message)
        
        # Update spaCy model if language changed
        if language != self.language:
            self.language = language
            self.nlp = self._get_spacy_model(language)
        
        # Detect multiple questions
        multiple_questions, question_parts = self._detect_multiple_questions(message, language)
        
        # Extract entities using spaCy if available
        if self.nlp:
            doc = self.nlp(message)
            entities = self._extract_entities_spacy(doc, language)
        else:
            entities = self._extract_entities_basic(message, language)
        
        # Extract dates using dateparser
        extracted_dates = self._extract_dates_enhanced(message, language)
        
        # Extract amounts
        extracted_amounts = self._extract_amounts_enhanced(message, language)
        
        # Extract categories
        extracted_categories = self._extract_categories_enhanced(message, language)
        
        # Extract accounts
        extracted_accounts = self._extract_accounts_enhanced(message, language)
        
        # Determine intent and confidence
        intent_type, confidence_score = self._determine_intent_enhanced(message, entities, language)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare metadata
        metadata = {
            'query_length': len(message),
            'has_numbers': bool(re.search(r'\d', message)),
            'has_currency': bool(re.search(r'\$', message)),
            'language': language,
            'complexity_score': self._calculate_complexity_score(message),
            'entity_count': len(entities),
            'date_count': len(extracted_dates),
            'amount_count': len(extracted_amounts),
            'spacy_used': self.nlp is not None,
            'dateparser_used': DATEPARSER_AVAILABLE,
            'multiple_questions': multiple_questions,
            'question_count': len(question_parts) if multiple_questions else 1
        }
        
        return EnhancedParsedIntent(
            intent_type=intent_type,
            confidence_score=confidence_score,
            entities=entities,
            metadata=metadata,
            extracted_dates=extracted_dates,
            extracted_amounts=extracted_amounts,
            extracted_categories=extracted_categories,
            extracted_accounts=extracted_accounts,
            language=language,
            processing_time=processing_time,
            multiple_questions=multiple_questions,
            question_parts=question_parts
        )
    
    def _extract_entities_spacy(self, doc, language: str) -> Dict[str, Any]:
        """Extract entities using spaCy."""
        entities = {
            'time_period': None,
            'comparison_type': None,
            'financial_entity': None,
            'categories': [],
            'accounts': [],
            'users': [],
            'amount_range': None,
            'is_historical': False,
            'is_comparative': False,
            'is_analytical': False,
            'is_trend_analysis': False,
            'is_anomaly_detection': False,
            'named_entities': []
        }
        
        # Extract named entities
        for ent in doc.ents:
            entity_info = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'description': spacy.explain(ent.label_)
            }
            entities['named_entities'].append(entity_info)
            
            # Categorize entities
            if ent.label_ in ['DATE', 'TIME']:
                entities['time_period'] = ent.text
            elif ent.label_ in ['MONEY', 'CARDINAL']:
                entities['amount_range'] = ent.text
            elif ent.label_ == 'PERSON':
                entities['users'].append(ent.text)
            elif ent.label_ == 'ORG':
                entities['accounts'].append(ent.text)
        
        # Extract additional entities using patterns
        patterns = self.entity_patterns.get(language, self.entity_patterns['en'])
        
        # Time keywords
        for period, keywords in patterns['time_keywords'].items():
            if any(keyword in doc.text.lower() for keyword in keywords):
                entities['time_period'] = period
                break
        
        # Financial keywords
        for entity, keywords in patterns['financial_keywords'].items():
            if any(keyword in doc.text.lower() for keyword in keywords):
                entities['financial_entity'] = entity
                break
        
        # Comparison keywords
        for comp_type, keywords in patterns['comparison_keywords'].items():
            if any(keyword in doc.text.lower() for keyword in keywords):
                entities['comparison_type'] = comp_type
                entities['is_comparative'] = True
                break
        
        return entities
    
    def _extract_entities_basic(self, message: str, language: str) -> Dict[str, Any]:
        """Extract entities using basic pattern matching (fallback)."""
        # This is a simplified version of the basic extraction
        # In practice, you'd want to implement more sophisticated pattern matching
        entities = {
            'time_period': None,
            'comparison_type': None,
            'financial_entity': None,
            'categories': [],
            'accounts': [],
            'users': [],
            'amount_range': None,
            'is_historical': False,
            'is_comparative': False,
            'is_analytical': False,
            'is_trend_analysis': False,
            'is_anomaly_detection': False,
            'named_entities': []
        }
        
        # Basic pattern matching
        patterns = self.entity_patterns.get(language, self.entity_patterns['en'])
        
        # Time keywords
        for period, keywords in patterns['time_keywords'].items():
            if any(keyword in message.lower() for keyword in keywords):
                entities['time_period'] = period
                break
        
        # Financial keywords
        for entity, keywords in patterns['financial_keywords'].items():
            if any(keyword in message.lower() for keyword in keywords):
                entities['financial_entity'] = entity
                break
        
        return entities
    
    def _extract_dates_enhanced(self, message: str, language: str) -> List[Dict[str, Any]]:
        """Extract dates using dateparser."""
        if not DATEPARSER_AVAILABLE:
            return []
        
        dates = []
        
        # Configure dateparser settings
        settings = self.date_settings.copy()
        if language == 'es':
            settings['LANGUAGE'] = 'es'
        
        # Try to extract dates from the message
        try:
            # Look for date patterns in the text
            date_patterns = [
                r'\b(today|yesterday|tomorrow|last week|this week|next week)\b',
                r'\b(last month|this month|next month)\b',
                r'\b(last year|this year|next year)\b',
                r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b',
                r'\b(\d{4}-\d{2}-\d{2})\b'
            ]
            
            for pattern in date_patterns:
                matches = re.finditer(pattern, message, re.IGNORECASE)
                for match in matches:
                    date_text = match.group()
                    try:
                        parsed_date = dateparser.parse(date_text, settings=settings)
                        if parsed_date:
                            dates.append({
                                'text': date_text,
                                'parsed_date': parsed_date,
                                'start': match.start(),
                                'end': match.end(),
                                'confidence': 0.9
                            })
                    except Exception as e:
                        logger.debug(f"Error parsing date '{date_text}': {e}")
        
        except Exception as e:
            logger.warning(f"Error in date extraction: {e}")
        
        return dates
    
    def _extract_amounts_enhanced(self, message: str, language: str) -> List[Dict[str, Any]]:
        """Extract amounts with enhanced pattern matching."""
        amounts = []
        patterns = self.amount_patterns.get(language, self.amount_patterns['en'])
        
        for pattern in patterns:
            matches = re.finditer(pattern, message, re.IGNORECASE)
            for match in matches:
                amount_text = match.group()
                try:
                    # Clean and parse amount
                    clean_amount = re.sub(r'[^\d.,]', '', amount_text)
                    if ',' in clean_amount and '.' in clean_amount:
                        # Handle both comma and decimal
                        clean_amount = clean_amount.replace(',', '')
                    elif ',' in clean_amount:
                        # Assume comma is thousands separator
                        clean_amount = clean_amount.replace(',', '')
                    
                    amount_value = float(clean_amount)
                    
                    amounts.append({
                        'text': amount_text,
                        'value': amount_value,
                        'start': match.start(),
                        'end': match.end(),
                        'currency': self._extract_currency(amount_text),
                        'confidence': 0.8
                    })
                except ValueError:
                    logger.debug(f"Could not parse amount: {amount_text}")
        
        return amounts
    
    def _extract_currency(self, amount_text: str) -> str:
        """Extract currency from amount text."""
        currency_patterns = {
            'USD': r'\$|dollars?|USD',
            'EUR': r'euros?|EUR',
            'GBP': r'pounds?|GBP',
            'MXN': r'pesos?|MXN'
        }
        
        for currency, pattern in currency_patterns.items():
            if re.search(pattern, amount_text, re.IGNORECASE):
                return currency
        
        return 'USD'  # Default
    
    def _extract_categories_enhanced(self, message: str, language: str) -> List[str]:
        """Extract categories using enhanced pattern matching."""
        categories = []
        category_map = self.category_mapping.get(language, self.category_mapping['en'])
        
        message_lower = message.lower()
        
        for category, keywords in category_map.items():
            if any(keyword in message_lower for keyword in keywords):
                categories.append(category)
        
        return categories
    
    def _extract_accounts_enhanced(self, message: str, language: str) -> List[str]:
        """Extract account information."""
        accounts = []
        
        # Common account patterns
        account_patterns = [
            r'\b(checking|savings|credit|debit)\s+(account|card)\b',
            r'\b(account|card)\s+ending\s+in\s+\d{4}\b',
            r'\b(Chase|Bank of America|Wells Fargo|Citibank)\b'
        ]
        
        for pattern in account_patterns:
            matches = re.finditer(pattern, message, re.IGNORECASE)
            for match in matches:
                accounts.append(match.group())
        
        return accounts
    
    def _determine_intent_enhanced(self, 
                                 message: str, 
                                 entities: Dict[str, Any], 
                                 language: str) -> Tuple[str, float]:
        """Determine intent with enhanced logic."""
        # Enhanced intent patterns with weights
        intent_patterns = {
            'en': {
                'balance_inquiry': {
                    'keywords': ['balance', 'how much', 'account', 'money', 'funds', 'total'],
                    'weight': 0.8,
                    'entity_boost': ['financial_entity']
                },
                'spending_analysis': {
                    'keywords': ['spent', 'spending', 'expense', 'cost', 'paid', 'gasté'],
                    'weight': 0.9,
                    'entity_boost': ['financial_entity']
                },
                'savings_planning': {
                    'keywords': ['save', 'savings', 'budget', 'plan', 'goal', 'ahorrar'],
                    'weight': 0.95,
                    'entity_boost': ['financial_entity']
                },
                'trend_analysis': {
                    'keywords': ['trend', 'change', 'over time', 'pattern', 'history', 'evolution', 'tendencia'],
                    'weight': 0.7,
                    'entity_boost': ['time_period']
                },
                'anomaly_detection': {
                    'keywords': ['unusual', 'strange', 'anomaly', 'different', 'unexpected', 'extraño', 'anomalía'],
                    'weight': 0.8,
                    'entity_boost': []
                },
                'goal_tracking': {
                    'keywords': ['goal', 'target', 'progress', 'achieve', 'reach', 'meta', 'objetivo'],
                    'weight': 0.85,
                    'entity_boost': []
                },
                'comparison': {
                    'keywords': ['compare', 'vs', 'versus', 'difference', 'relative', 'comparar', 'diferencia'],
                    'weight': 0.75,
                    'entity_boost': ['comparison_type']
                },
                'temporal_comparison': {
                    'keywords': ['this month vs', 'last month', 'previous month', 'compared to', 'vs last', 'este mes vs'],
                    'weight': 0.85,
                    'entity_boost': ['time_period', 'comparison_type']
                },
                'category_comparison': {
                    'keywords': ['category', 'categories', 'food vs', 'transport vs', 'categoría', 'comida vs'],
                    'weight': 0.8,
                    'entity_boost': ['categories']
                },
                'prediction': {
                    'keywords': ['predict', 'forecast', 'future', 'next', 'will', 'predecir', 'futuro'],
                    'weight': 0.6,
                    'entity_boost': ['time_period']
                },
                'optimization': {
                    'keywords': ['optimize', 'optimization', 'best', 'improve', 'optimizar', 'mejorar'],
                    'weight': 0.7,
                    'entity_boost': []
                }
            },
            'es': {
                'consulta_saldo': {
                    'keywords': ['saldo', 'cuánto', 'cuenta', 'dinero', 'fondos', 'total'],
                    'weight': 0.8,
                    'entity_boost': ['financial_entity']
                },
                'analisis_gastos': {
                    'keywords': ['gasté', 'gastos', 'gasto', 'pagué', 'costo', 'spent'],
                    'weight': 0.9,
                    'entity_boost': ['financial_entity']
                },
                'planificacion_ahorro': {
                    'keywords': ['ahorrar', 'ahorro', 'presupuesto', 'plan', 'meta', 'save'],
                    'weight': 0.95,
                    'entity_boost': ['financial_entity']
                },
                'analisis_tendencias': {
                    'keywords': ['tendencia', 'cambio', 'historia', 'patrón', 'evolución', 'trend'],
                    'weight': 0.7,
                    'entity_boost': ['time_period']
                },
                'deteccion_anomalias': {
                    'keywords': ['inusual', 'extraño', 'anomalía', 'diferente', 'inesperado', 'unusual'],
                    'weight': 0.8,
                    'entity_boost': []
                },
                'seguimiento_metas': {
                    'keywords': ['meta', 'objetivo', 'progreso', 'lograr', 'alcanzar', 'goal'],
                    'weight': 0.85,
                    'entity_boost': []
                },
                'comparacion': {
                    'keywords': ['comparar', 'vs', 'versus', 'diferencia', 'respecto', 'compare'],
                    'weight': 0.75,
                    'entity_boost': ['comparison_type']
                },
                'comparacion_temporal': {
                    'keywords': ['este mes vs', 'mes pasado', 'mes anterior', 'comparado con', 'vs mes pasado'],
                    'weight': 0.85,
                    'entity_boost': ['time_period', 'comparison_type']
                },
                'comparacion_categorias': {
                    'keywords': ['categoría', 'categorías', 'comida vs', 'transporte vs', 'category'],
                    'weight': 0.8,
                    'entity_boost': ['categories']
                },
                'prediccion': {
                    'keywords': ['predecir', 'pronóstico', 'futuro', 'próximo', 'será', 'predict'],
                    'weight': 0.6,
                    'entity_boost': ['time_period']
                },
                'optimizacion': {
                    'keywords': ['optimizar', 'optimización', 'mejor', 'mejorar', 'optimize'],
                    'weight': 0.7,
                    'entity_boost': []
                }
            }
        }
        
        patterns = intent_patterns.get(language, intent_patterns['en'])
        intent_scores = {}
        
        message_lower = message.lower()
        
        for intent, pattern in patterns.items():
            score = 0.0
            
            # Keyword matching
            for keyword in pattern['keywords']:
                if keyword in message_lower:
                    score += pattern['weight']
            
            # Entity boost
            for entity_type in pattern['entity_boost']:
                if entities.get(entity_type):
                    score += 0.2
            
            # Special patterns for temporal comparison
            if intent in ['temporal_comparison', 'comparacion_temporal']:
                # Check for specific temporal comparison patterns
                temporal_patterns = {
                    'en': ['this month vs', 'last month', 'previous month', 'compared to', 'vs last'],
                    'es': ['este mes vs', 'mes pasado', 'mes anterior', 'comparado con', 'vs mes pasado']
                }
                temp_patterns = temporal_patterns.get(language, temporal_patterns['en'])
                if any(pattern in message_lower for pattern in temp_patterns):
                    score += 0.3
            
            # Special patterns for category comparison
            if intent in ['category_comparison', 'comparacion_categorias']:
                # Check for category-specific patterns
                category_patterns = {
                    'en': ['food vs', 'transport vs', 'entertainment vs', 'shopping vs'],
                    'es': ['comida vs', 'transporte vs', 'entretenimiento vs', 'compras vs']
                }
                cat_patterns = category_patterns.get(language, category_patterns['en'])
                if any(pattern in message_lower for pattern in cat_patterns):
                    score += 0.3
            
            # Multiple keyword bonus
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
                fallback_intent = 'clarification_request' if language == 'en' else 'solicitud_aclaracion'
                return fallback_intent, 0.5
            
            return best_intent, best_score
        
        # Default fallback
        default_intent = 'general_inquiry' if language == 'en' else 'consulta_general'
        return default_intent, 0.4
    
    def _softmax(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Apply softmax function to convert scores to probabilities."""
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
        from ai.core.translation_service import detect_language
        return detect_language(message)
    
    def _calculate_complexity_score(self, message: str) -> float:
        """Calculate a complexity score for the query."""
        score = 0.0
        
        # Length factor
        score += min(len(message.split()) / 20.0, 1.0) * 0.3
        
        # Number of entities (estimated)
        entity_indicators = ['time_period', 'comparison_type', 'financial_entity', 'amount_range']
        entity_count = sum(1 for entity in entity_indicators if self._has_entity(message, entity))
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
        if self.nlp:
            doc = self.nlp(message)
            return any(ent.label_ == entity_type for ent in doc.ents)
        return False
    
    def _detect_multiple_questions(self, message: str, language: str) -> Tuple[bool, List[str]]:
        """Detect if the query contains multiple questions and split them."""
        multiple_indicators = {
            'en': [
                'and', 'also', 'additionally', 'furthermore', 'moreover', 'besides',
                'as well as', 'in addition', 'on the other hand', 'however', 'but',
                'while', 'whereas', 'similarly', 'likewise', 'in the same way'
            ],
            'es': [
                'y', 'también', 'además', 'asimismo', 'igualmente', 'así mismo',
                'por otro lado', 'por otra parte', 'en segundo lugar', 'finalmente',
                'últimamente', 'pero', 'sin embargo', 'aunque', 'mientras que',
                'por el contrario', 'no obstante', 'a pesar de'
            ]
        }
        
        indicators = multiple_indicators.get(language, multiple_indicators['en'])
        
        # Count question marks
        question_marks = message.count('?')
        if question_marks > 1:
            return True, self._split_multiple_questions(message, language)
        
        # Look for multiple question indicators
        for indicator in indicators:
            if indicator in message.lower():
                # For common words like 'and', 'but', check for more specific patterns
                if indicator in ['and', 'y', 'but', 'pero', 'aunque']:
                    specific_patterns = {
                        'en': ['and how much', 'and what', 'and which', 'but how much', 'but what'],
                        'es': ['y cuánto', 'y cuál', 'y qué', 'pero cuánto', 'pero cuál', 'aunque cuánto']
                    }
                    patterns = specific_patterns.get(language, specific_patterns['en'])
                    if any(pattern in message.lower() for pattern in patterns):
                        return True, self._split_multiple_questions(message, language)
                else:
                    return True, self._split_multiple_questions(message, language)
        
        return False, [message]
    
    def _split_multiple_questions(self, message: str, language: str) -> List[str]:
        """Split a multiple question query into individual questions."""
        import re
        
        # Split by question marks first
        parts = re.split(r'\?+', message)
        questions = []
        
        for part in parts:
            part = part.strip()
            if part and len(part) > 10:  # Filter very short parts
                # Add question mark back
                questions.append(part + '?')
        
        # If not split well, try splitting by connectors
        if len(questions) <= 1:
            connectors = {
                'en': [' and ', ' also ', ' additionally ', ' furthermore ', ' but ', ' however '],
                'es': [' y ', ' también ', ' además ', ' por otro lado ', ' pero ', ' sin embargo ']
            }
            conn_list = connectors.get(language, connectors['en'])
            
            for connector in conn_list:
                if connector in message.lower():
                    parts = re.split(connector, message, flags=re.IGNORECASE)
                    questions = [part.strip() for part in parts if part.strip()]
                    # Add question marks if missing
                    questions = [q + '?' if not q.endswith('?') else q for q in questions]
                    break
        
        return questions if questions else [message]