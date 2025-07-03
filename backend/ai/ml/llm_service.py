"""
Servicio de LLM local (Llama 3) para chatbot financiero ROBUSTO con AI personalizada integrada.

Este módulo proporciona:
- Parser inteligente de consultas financieras
- Contexto financiero completo y dinámico con AI personalizada
- Análisis semántico de preguntas
- Respuestas contextuales basadas en datos financieros del usuario
- Soporte para cualquier tipo de pregunta financiera
- Integración completa con el sistema de AI personalizado
"""
import logging
import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from django.conf import settings
from django.core.cache import cache
from django.utils import timezone
from django.db.models import Sum, Count, Avg, Max, Min, Q
import requests
import subprocess
import threading
import time
import unicodedata
from calendar import monthrange
from deep_translator import GoogleTranslator
from langdetect import detect

# Importar servicios de AI personalizada
from .analyzers.behavior import BehaviorAnalyzer
from .predictors.expense import ExpensePredictor
from .optimizers.budget_optimizer import BudgetOptimizer
from .anomaly_detector import AnomalyDetector
from .ai_orchestrator import AIOrchestrator

# Importar módulos del core AI
from ..core.ai_service import AIService, AIResponse
from ..core.enhanced_query_parser import EnhancedFinancialQueryParser, EnhancedParsedIntent
from ..core.intent_classifier import IntentClassifier, IntentPrediction
from ..core.followup_suggester import FollowUpSuggester, FollowUpSuggestion
from ..core.report_generator import ReportGenerator, ReportData, ReportConfig
from ..core.translation_service import TranslationService, translate, detect_language
from ..core.context_manager import ConversationContextManager
from ..core.privacy_guard import PrivacyGuard
from ..core.nl_renderer import NLRenderer
from ..core.prompt_builder import PromptBuilder

logger = logging.getLogger('ai.llm')

class FinancialQueryParser:
    """Parser inteligente para analizar preguntas financieras."""
    
    def __init__(self):
        # Palabras clave para diferentes tipos de consultas
        self.time_keywords = {
            'mes': ['mes', 'mensual', 'mensualmente'],
            'semana': ['semana', 'semanal', 'semanalmente'],
            'año': ['año', 'anual', 'anualmente', 'anual'],
            'trimestre': ['trimestre', 'trimestral', 'trimestralmente'],
            'hoy': ['hoy', 'actual', 'actualmente'],
            'ayer': ['ayer', 'pasado'],
            'último': ['último', 'última', 'reciente'],
            'próximo': ['próximo', 'próxima', 'siguiente', 'futuro']
        }
        
        self.comparison_keywords = {
            'comparar': ['comparar', 'comparación', 'vs', 'versus', 'respecto', 'en comparación'],
            'diferencia': ['diferencia', 'diferente', 'cambio', 'variación'],
            'más': ['más', 'mayor', 'superior', 'alto'],
            'menos': ['menos', 'menor', 'inferior', 'bajo']
        }
        
        self.financial_keywords = {
            'gasto': ['gasto', 'gastos', 'gastar', 'gastado'],
            'ingreso': ['ingreso', 'ingresos', 'ganancia', 'ganancias'],
            'balance': ['balance', 'saldo', 'balance'],
            'presupuesto': ['presupuesto', 'presupuestar'],
            'ahorro': ['ahorro', 'ahorrar', 'ahorrado'],
            'deuda': ['deuda', 'deudas', 'deber'],
            'inversión': ['inversión', 'invertir', 'invertido']
        }
    
    def parse_query(self, message: str) -> Dict[str, Any]:
        """Analiza la pregunta y extrae entidades relevantes."""
        message_lower = message.lower()
        
        # Detectar si es una pregunta múltiple
        multiple_questions = self._detect_multiple_questions(message)
        
        parsed = {
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
            'is_anomaly_detection': self._is_anomaly_detection_query(message_lower),
            'is_prediction': self._is_prediction_query(message_lower),
            'is_optimization': self._is_optimization_query(message_lower),
            'is_net_balance': self._is_net_balance_query(message_lower),
            'multiple_questions': multiple_questions,
            'question_parts': self._split_multiple_questions(message) if multiple_questions else [message]
        }
        
        return parsed
    
    def _extract_time_period(self, message: str) -> Optional[str]:
        """Extrae el período de tiempo mencionado."""
        for period, keywords in self.time_keywords.items():
            if any(keyword in message for keyword in keywords):
                return period
        return None
    
    def _extract_comparison_type(self, message: str) -> Optional[str]:
        """Extrae el tipo de comparación."""
        for comp_type, keywords in self.comparison_keywords.items():
            if any(keyword in message for keyword in keywords):
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
        anomaly_keywords = ['inusual', 'extraño', 'anómalo', 'diferente', 'sospechoso', 'anomalía']
        return any(keyword in message for keyword in anomaly_keywords)
    
    def _is_prediction_query(self, message: str) -> bool:
        """Determina si es una pregunta de predicción."""
        prediction_keywords = ['predecir', 'predicción', 'futuro', 'próximo', 'siguiente', 'esperar', 'pronóstico']
        return any(keyword in message for keyword in prediction_keywords)
    
    def _is_optimization_query(self, message: str) -> bool:
        """Determina si es una pregunta de optimización."""
        optimization_keywords = ['optimizar', 'optimización', 'mejorar', 'recomendación', 'sugerencia', 'consejo']
        return any(keyword in message for keyword in optimization_keywords)
    
    def _is_net_balance_query(self, message: str) -> bool:
        """Determina si es una pregunta sobre balance neto (sin arrastre)."""
        net_balance_keywords = [
            'solo', 'únicamente', 'neto', 'generado', 'producido', 'sin contar', 
            'sin arrastre', 'exclusivamente', 'solamente', 'puro', 'directo',
            'de ese mes', 'del mes', 'en ese mes', 'en el mes', 'solo en',
            'únicamente en', 'exclusivamente en', 'neto de', 'generado en'
        ]
        return any(keyword in message for keyword in net_balance_keywords)
    
    def _detect_multiple_questions(self, message: str) -> bool:
        """Detecta si la consulta contiene múltiples preguntas."""
        multiple_indicators = [
            'y', 'también', 'además', 'asimismo', 'igualmente', 'así mismo', 
            'por otro lado', 'por otra parte', 'en segundo lugar', 'finalmente', 
            'últimamente', '¿', '?', ';', '.', 'pero', 'sin embargo', 'aunque'
        ]
        
        # Contar signos de interrogación
        question_marks = message.count('?')
        if question_marks > 1:
            return True
        
        # Buscar indicadores de múltiples preguntas
        for indicator in multiple_indicators:
            if indicator in message.lower():
                # Verificar que no sea solo una palabra común
                if indicator in ['y', 'pero', 'aunque']:
                    # Buscar patrones más específicos
                    if any(pattern in message.lower() for pattern in [
                        'y cuánto', 'y cuál', 'y qué', 'pero cuánto', 'pero cuál',
                        'aunque cuánto', 'aunque cuál'
                    ]):
                        return True
                else:
                    return True
        
        return False
    
    def _split_multiple_questions(self, message: str) -> List[str]:
        """Divide una consulta múltiple en preguntas individuales."""
        # Dividir por signos de interrogación
        parts = re.split(r'\?+', message)
        questions = []
        
        for part in parts:
            part = part.strip()
            if part and len(part) > 10:  # Filtrar partes muy cortas
                # Agregar signo de interrogación de vuelta
                questions.append(part + '?')
        
        # Si no se dividió bien, intentar dividir por conectores
        if len(questions) <= 1:
            connectors = [' y ', ' también ', ' además ', ' por otro lado ', ' pero ']
            for connector in connectors:
                if connector in message.lower():
                    parts = re.split(connector, message, flags=re.IGNORECASE)
                    questions = [part.strip() for part in parts if part.strip()]
                    break
        
        return questions if questions else [message]

class LLMService:
    """
    Servicio para manejar Llama 3 local y conversaciones financieras ROBUSTAS con AI personalizada.
    """
    
    def __init__(self):
        """Inicializa el servicio de LLM con AI personalizada integrada."""
        self.model_name = "llama3:8b"
        self.api_url = getattr(settings, 'LLM_API_URL', 'http://localhost:11434')
        self.context_window = 4096
        self.max_tokens = 512
        self.temperature = 0.7
        self.timeout = 60  # Timeout en segundos para llamadas al LLM (aumentado para AI personalizada)
        self.fine_tuned_model = None
        self.query_parser = FinancialQueryParser()
        
        # Configuración de contexto
        self.max_context_length = 10  # Máximo 10 mensajes en contexto
        self.context_ttl = 3600  # 1 hora
        
        # Inicializar servicios de AI personalizada
        self._initialize_ai_services()
        
        # Inicializar módulos del core AI
        self._initialize_core_ai_modules()
        
        # Verificar si Ollama está disponible
        self._check_ollama_availability()
    
    def _initialize_ai_services(self):
        """Inicializa todos los servicios de AI personalizada."""
        try:
            logger.info("Inicializando servicios de AI personalizada...")
            
            # Servicios principales de AI
            self.behavior_analyzer = BehaviorAnalyzer()
            self.expense_predictor = ExpensePredictor()
            self.budget_optimizer = BudgetOptimizer()
            self.anomaly_detector = AnomalyDetector()
            self.ai_orchestrator = AIOrchestrator()
            
            # Cargar modelos entrenados si existen
            self._load_ai_models()
            
            logger.info("Servicios de AI personalizada inicializados correctamente")
            
        except Exception as e:
            logger.error(f"Error inicializando servicios de AI: {str(e)}")
            # Continuar sin AI personalizada si hay errores
            self.behavior_analyzer = None
            self.expense_predictor = None
            self.budget_optimizer = None
            self.anomaly_detector = None
            self.ai_orchestrator = None
    
    def _load_ai_models(self):
        """Carga los modelos de AI entrenados."""
        try:
            # Intentar cargar modelos si existen
            if self.behavior_analyzer:
                self.behavior_analyzer.load()
            if self.expense_predictor:
                self.expense_predictor.load()
            if self.budget_optimizer:
                self.budget_optimizer.load()
            if self.anomaly_detector:
                self.anomaly_detector.load()
                
            logger.info("Modelos de AI cargados correctamente")
            
        except Exception as e:
            logger.warning(f"No se pudieron cargar algunos modelos de AI: {str(e)}")
            # Los modelos se entrenarán automáticamente cuando sea necesario
    
    def _initialize_core_ai_modules(self):
        """Inicializa todos los módulos del core AI."""
        try:
            logger.info("Inicializando módulos del core AI...")
            
            # Servicios principales del core
            self.ai_service = AIService()
            self.enhanced_query_parser = EnhancedFinancialQueryParser()
            self.intent_classifier = IntentClassifier()
            self.followup_suggester = FollowUpSuggester()
            self.report_generator = ReportGenerator()
            self.translation_service = TranslationService()
            self.context_manager = ConversationContextManager()
            self.privacy_guard = PrivacyGuard()
            self.nl_renderer = NLRenderer()
            self.prompt_builder = PromptBuilder()
            
            logger.info("Módulos del core AI inicializados correctamente")
            
        except Exception as e:
            logger.error(f"Error inicializando módulos del core AI: {str(e)}")
            # Continuar sin módulos del core si hay errores
            self.ai_service = None
            self.enhanced_query_parser = None
            self.intent_classifier = None
            self.followup_suggester = None
            self.report_generator = None
            self.translation_service = None
            self.context_manager = None
            self.privacy_guard = None
            self.nl_renderer = None
            self.prompt_builder = None
    
    def _train_ai_models_if_needed(self, organization_id: int):
        """Entrena automáticamente los modelos de AI si no están entrenados."""
        try:
            from transactions.models import Transaction
            transactions = list(Transaction.objects.filter(organization_id=organization_id))
            
            if not transactions or len(transactions) < 10:
                logger.warning("No hay suficientes transacciones para entrenar modelos de AI")
                return
            
            logger.info(f"Entrenando modelos de AI con {len(transactions)} transacciones...")
            
            # Entrenar Behavior Analyzer si no está entrenado
            if self.behavior_analyzer and not self.behavior_analyzer.is_trained:
                try:
                    logger.info("Entrenando Behavior Analyzer...")
                    self.behavior_analyzer.train(transactions)
                    logger.info("Behavior Analyzer entrenado exitosamente")
                except Exception as e:
                    logger.error(f"Error entrenando Behavior Analyzer: {str(e)}")
            
            # Entrenar Expense Predictor si no está entrenado
            if self.expense_predictor and not self.expense_predictor.is_trained:
                try:
                    logger.info("Entrenando Expense Predictor...")
                    self.expense_predictor.train(transactions)
                    logger.info("Expense Predictor entrenado exitosamente")
                except Exception as e:
                    logger.error(f"Error entrenando Expense Predictor: {str(e)}")
            
            # Entrenar Budget Optimizer si no está entrenado
            if self.budget_optimizer and not self.budget_optimizer.is_trained:
                try:
                    logger.info("Entrenando Budget Optimizer...")
                    self.budget_optimizer.train(transactions)
                    logger.info("Budget Optimizer entrenado exitosamente")
                except Exception as e:
                    logger.error(f"Error entrenando Budget Optimizer: {str(e)}")
            
            logger.info("Entrenamiento de modelos de AI completado")
            
        except Exception as e:
            logger.error(f"Error en entrenamiento automático de modelos: {str(e)}")

    def _check_ollama_availability(self):
        """Verifica si Ollama está disponible y el modelo está descargado."""
        try:
            # Verificar si Ollama está corriendo
            response = requests.get(f"{self.api_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                if self.model_name in model_names:
                    logger.info(f"Modelo {self.model_name} disponible en Ollama")
                    return True
                else:
                    logger.warning(f"Modelo {self.model_name} no encontrado. Descargando...")
                    self._download_model()
                    return True
            else:
                logger.error("Ollama no está disponible")
                return False
                
        except Exception as e:
            logger.error(f"Error verificando Ollama: {str(e)}")
            return False
    
    def _download_model(self):
        """Descarga el modelo Llama 3 8B si no está disponible."""
        try:
            logger.info(f"Descargando modelo {self.model_name}...")
            
            # Comando para descargar el modelo
            cmd = f"ollama pull {self.model_name}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Modelo {self.model_name} descargado exitosamente")
                return True
            else:
                logger.error(f"Error descargando modelo: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error en descarga de modelo: {str(e)}")
            return False
    
    def _get_conversation_context(self, user_id: int, organization_id: int) -> List[Dict[str, str]]:
        """Obtiene el contexto de conversación del usuario desde Redis."""
        try:
            cache_key = f"chat_context:{user_id}:{organization_id}"
            context = cache.get(cache_key, [])
            return context
        except Exception as e:
            logger.error(f"Error obteniendo contexto: {str(e)}")
            return []
    
    def _save_conversation_context(self, user_id: int, organization_id: int, 
                                 context: List[Dict[str, str]]):
        """Guarda el contexto de conversación en Redis."""
        try:
            cache_key = f"chat_context:{user_id}:{organization_id}"
            # Mantener solo los últimos N mensajes
            if len(context) > self.max_context_length:
                context = context[-self.max_context_length:]
            cache.set(cache_key, context, self.context_ttl)
        except Exception as e:
            logger.error(f"Error guardando contexto: {str(e)}")
    
    def _get_comprehensive_financial_data(self, organization_id: int, end_date: datetime = None, message: str = "", query_analysis: Dict = None) -> Dict[str, Any]:
        """Obtiene datos financieros completos y detallados, incluyendo balances mensuales y promedios."""
        from transactions.models import Transaction
        from django.db.models import Sum, Count, Avg, Max, Min
        from collections import OrderedDict
        from datetime import datetime
        try:
            # Filtro de fecha: por defecto hasta la fecha actual, pero respeta el parámetro end_date
            if end_date is None:
                end_date = timezone.now()
            
            print(f"🔍 DEBUG: Filtrando transacciones hasta: {end_date}")
            logger.info(f"Filtrando transacciones hasta: {end_date}")
            
            # Determinar si se solicita balance neto (sin arrastre)
            if query_analysis:
                if isinstance(query_analysis, dict):
                    is_net_balance = query_analysis.get('is_net_balance', False)
                else:
                    is_net_balance = getattr(query_analysis, 'is_net_balance', False)
            else:
                is_net_balance = False
            
            # Filtrar transacciones por organización y hasta la fecha especificada
            base_filter = Transaction.objects.filter(
                organization_id=organization_id,
                date__lte=end_date
            )
            
            # Debug: verificar qué fechas están siendo incluidas
            all_dates = base_filter.values_list('date', flat=True).order_by('date')
            if all_dates:
                print(f"🔍 DEBUG: Rango de fechas en transacciones: {all_dates.first()} a {all_dates.last()}")
                print(f"🔍 DEBUG: Total de transacciones filtradas: {base_filter.count()}")
                logger.info(f"Rango de fechas en transacciones: {all_dates.first()} a {all_dates.last()}")
                logger.info(f"Total de transacciones filtradas: {base_filter.count()}")
            else:
                print(f"🔍 DEBUG: NO HAY TRANSACCIONES ENCONTRADAS para org {organization_id}")
                logger.error(f"NO HAY TRANSACCIONES ENCONTRADAS para org {organization_id}")
            
            # Datos generales (filtrados por fecha)
            total_transactions = base_filter.count()
            total_income = base_filter.filter(type='INCOME').aggregate(total=Sum('amount'))['total'] or 0
            total_expenses = base_filter.filter(type='EXPENSE').aggregate(total=Sum('amount'))['total'] or 0
            
            # Datos por mes (balances mensuales) - filtrados por fecha
            transactions = base_filter.order_by('date')
            months_data = OrderedDict()
            
            # Debug: verificar transacciones de junio 2024
            june_2024_transactions = [t for t in transactions if t.date.year == 2024 and t.date.month == 6]
            if june_2024_transactions:
                print(f"🔍 DEBUG: Encontradas {len(june_2024_transactions)} transacciones de junio 2024:")
                for t in june_2024_transactions[:3]:  # Mostrar solo las primeras 3
                    print(f"  - {t.date}: {t.description} (${t.amount})")
            
            current_date = timezone.now()
            current_year = current_date.year
            current_month = current_date.month
            current_day = current_date.day
            
            for t in transactions:
                year = t.date.year
                month = t.date.month
                key = f"{year}-{month:02d}"
                if key not in months_data:
                    months_data[key] = {'year': year, 'month': month, 'income': 0, 'expenses': 0}
                
                # Para el mes actual, solo incluir transacciones hasta el día actual
                if year == current_year and month == current_month:
                    if t.date.day <= current_day:
                        if t.type == 'INCOME':
                            months_data[key]['income'] += float(t.amount)
                        elif t.type == 'EXPENSE':
                            months_data[key]['expenses'] += float(t.amount)
                else:
                    # Para otros meses, incluir todas las transacciones
                    if t.type == 'INCOME':
                        months_data[key]['income'] += float(t.amount)
                    elif t.type == 'EXPENSE':
                        months_data[key]['expenses'] += float(t.amount)
            
            # Calcular balance según el tipo solicitado
            if is_net_balance:
                # Balance neto: solo ingresos menos gastos de cada mes (sin arrastre)
                for data in months_data.values():
                    data['balance'] = data['income'] - data['expenses']
                    data['balance_type'] = 'neto'
            else:
                # Balance acumulado: incluye arrastre de meses anteriores
                accumulated_balance = 0
                for data in months_data.values():
                    monthly_balance = data['income'] - data['expenses']
                    accumulated_balance += monthly_balance
                    data['balance'] = accumulated_balance
                    data['balance_type'] = 'acumulado'
                    data['monthly_net'] = monthly_balance  # Guardar también el neto del mes
            
            # Filtrar solo los meses del año actual hasta el mes actual si la pregunta es de balance mensual
            if 'balance' in message.lower() or 'mes' in message.lower() or 'mensual' in message.lower() or 'cada mes' in message.lower():
                now = timezone.now()
                current_year = now.year
                current_month = now.month
                months_data = OrderedDict((k, v) for k, v in months_data.items() if v['year'] == current_year and v['month'] <= current_month)
            
            # Promedios mensuales
            months_count = len(months_data)
            avg_monthly_income = float(total_income) / months_count if months_count else 0
            avg_monthly_expenses = float(total_expenses) / months_count if months_count else 0
            avg_monthly_balance = avg_monthly_income - avg_monthly_expenses
            
            # Top categorías de gastos (filtradas por fecha)
            top_expense_categories = base_filter.filter(
                type='EXPENSE'
            ).values('category__name').annotate(
                total=Sum('amount'),
                count=Count('id')
            ).order_by('-total')[:10]
            
            # Top categorías de ingresos (filtradas por fecha)
            top_income_categories = base_filter.filter(
                type='INCOME'
            ).values('category__name').annotate(
                total=Sum('amount'),
                count=Count('id')
            ).order_by('-total')[:10]
            
            # Estadísticas de transacciones (filtradas por fecha)
            transaction_stats = base_filter.aggregate(
                avg_amount=Avg('amount'),
                max_amount=Max('amount'),
                min_amount=Min('amount')
            )
            
            # Transacciones recientes (últimas 10, filtradas por fecha)
            recent_transactions = base_filter.select_related('category').order_by('-date')[:10]
            
            # Obtener gastos por categoría para el mes actual
            current_month_start = timezone.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            current_month_expenses = base_filter.filter(
                type='EXPENSE',
                date__gte=current_month_start,
                date__lte=end_date
            ).values('category__name').annotate(
                total=Sum('amount'),
                count=Count('id')
            ).order_by('-total')
            
            result = {
                'summary': {
                    'total_transactions': total_transactions,
                    'total_income': float(total_income),
                    'total_expenses': float(total_expenses),
                    'net_balance': float(total_income - total_expenses),
                    'avg_transaction': float(transaction_stats['avg_amount'] or 0),
                    'max_transaction': float(transaction_stats['max_amount'] or 0),
                    'min_transaction': float(transaction_stats['min_amount'] or 0),
                    'avg_monthly_income': avg_monthly_income,
                    'avg_monthly_expenses': avg_monthly_expenses,
                    'avg_monthly_balance': avg_monthly_balance
                },
                'monthly_balances': list(months_data.values()),
                'top_expense_categories': list(top_expense_categories),
                'top_income_categories': list(top_income_categories),
                'current_month_expenses': list(current_month_expenses),
                'recent_transactions': [
                    {
                        'date': t.date.strftime('%Y-%m-%d'),
                        'description': t.description,
                        'amount': float(t.amount),
                        'type': t.type,
                        'category': t.category.name if t.category else 'Sin categoría'
                    }
                    for t in recent_transactions
                ]
            }
            
            print(f"🔍 DEBUG: Datos financieros obtenidos exitosamente:")
            print(f"  - Total transacciones: {total_transactions}")
            print(f"  - Balance neto: ${result['summary']['net_balance']:,.2f}")
            print(f"  - Ingresos: ${result['summary']['total_income']:,.2f}")
            print(f"  - Gastos: ${result['summary']['total_expenses']:,.2f}")
            print(f"  - Meses con datos: {len(months_data)}")
            
            return result
        except Exception as e:
            logger.error(f"Error obteniendo datos financieros: {str(e)}")
            return {}

    def _get_monthly_comparison_data(self, organization_id: int) -> dict:
        """Obtiene datos comparativos de todos los meses disponibles."""
        from transactions.models import Transaction
        from django.db.models import Sum
        from datetime import datetime
        
        # Obtener todos los meses con datos de ingresos
        income_data = Transaction.objects.filter(
            organization_id=organization_id,
            type='INCOME'
        ).extra(
            select={'year': 'EXTRACT(year FROM date)', 'month': 'EXTRACT(month FROM date)'}
        ).values('year', 'month').annotate(
            total=Sum('amount')
        ).order_by('year', 'month')
        
        # Obtener todos los meses con datos de gastos
        expense_data = Transaction.objects.filter(
            organization_id=organization_id,
            type='EXPENSE'
        ).extra(
            select={'year': 'EXTRACT(year FROM date)', 'month': 'EXTRACT(month FROM date)'}
        ).values('year', 'month').annotate(
            total=Sum('amount')
        ).order_by('year', 'month')
        
        # Procesar datos
        months_info = {}
        
        # Procesar ingresos
        for data in income_data:
            year = int(data['year'])
            month = int(data['month'])
            key = f"{year}-{month:02d}"
            
            if key not in months_info:
                months_info[key] = {'year': year, 'month': month, 'income': 0, 'expenses': 0, 'balance': 0}
            
            months_info[key]['income'] = float(data['total'] or 0)
        
        # Procesar gastos
        for data in expense_data:
            year = int(data['year'])
            month = int(data['month'])
            key = f"{year}-{month:02d}"
            
            if key not in months_info:
                months_info[key] = {'year': year, 'month': month, 'income': 0, 'expenses': 0, 'balance': 0}
            
            months_info[key]['expenses'] = float(data['total'] or 0)
        
        # Calcular balance
        for month_data in months_info.values():
            month_data['balance'] = month_data['income'] - month_data['expenses']
        
        return months_info

    def _find_transactions_by_merchant(self, organization_id: int, merchant_query: str) -> list:
        """Busca transacciones que coincidan con un comercio/marca en description o merchant (fuzzy, insensible a mayúsculas/acentos)."""
        from transactions.models import Transaction
        from django.db.models import Q

        def normalize(text):
            if not text:
                return ""
            return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8').lower()

        merchant_query_norm = normalize(merchant_query)
        # Buscar en ambos campos usando contains insensible a mayúsculas/acentos
        transactions = Transaction.objects.filter(
            organization_id=organization_id
        ).filter(
            Q(description__icontains=merchant_query) | Q(merchant__icontains=merchant_query)
        )
        # Filtro extra: fuzzy manual por normalización
        results = []
        for t in transactions:
            desc = normalize(t.description)
            merch = normalize(t.merchant)
            if merchant_query_norm in desc or merchant_query_norm in merch:
                results.append({
                    'date': t.date.strftime('%Y-%m-%d'),
                    'month': t.date.strftime('%Y-%m'),
                    'description': t.description,
                    'merchant': t.merchant,
                    'amount': float(t.amount),
                    'type': t.type,
                    'category': t.category.name if t.category else 'Sin categoría'
                })
        return results

    def _prepare_comprehensive_financial_context(self, user_id: int, organization_id: int, 
                                               message: str = "", query_analysis: Dict = None) -> str:
        """Prepara un contexto financiero completo y dinámico, incluyendo balances mensuales y promedios."""
        try:
            # Determinar la fecha de fin basada en el análisis de la consulta
            end_date = self._determine_end_date_from_query(message, query_analysis)
            
            financial_data = self._get_comprehensive_financial_data(organization_id, end_date, message, query_analysis)
            if not financial_data:
                return "No hay datos financieros disponibles."
            context_parts = []
            summary = financial_data['summary']
            context_parts.append("💰 RESUMEN:")
            context_parts.append(f"Balance actual: ${summary['net_balance']:,.2f} | Ingresos totales: ${summary['total_income']:,.2f} | Gastos totales: ${summary['total_expenses']:,.2f}")
            context_parts.append(f"Transacciones: {summary['total_transactions']} | Promedio por transacción: ${summary['avg_transaction']:,.2f}")
            context_parts.append(f"Promedio mensual: Ingresos=${summary['avg_monthly_income']:,.2f}, Gastos=${summary['avg_monthly_expenses']:,.2f}, Balance=${summary['avg_monthly_balance']:,.2f}")
            
            # Tabla de balances mensuales
            if financial_data['monthly_balances']:
                context_parts.append("\n📅 BALANCE POR MES:")
                for data in financial_data['monthly_balances']:
                    context_parts.append(f"{data['year']}-{data['month']:02d}: Ingresos=${data['income']:,.2f}, Gastos=${data['expenses']:,.2f}, Balance=${data['balance']:,.2f}")
            
            # Top categorías - CONCISO
            if financial_data['top_expense_categories']:
                context_parts.append("\n💸 TOP GASTOS:")
                for i, cat in enumerate(financial_data['top_expense_categories'][:3], 1):
                    context_parts.append(f"{i}. {cat['category__name']}: ${cat['total']:,.2f}")
            
            # Gastos del mes actual por categoría
            if financial_data.get('current_month_expenses'):
                context_parts.append("\n📅 GASTOS ESTE MES:")
                for i, cat in enumerate(financial_data['current_month_expenses'][:5], 1):
                    context_parts.append(f"{i}. {cat['category__name']}: ${cat['total']:,.2f}")
            
            if financial_data['top_income_categories']:
                context_parts.append("\n💵 TOP INGRESOS:")
                for i, cat in enumerate(financial_data['top_income_categories'][:3], 1):
                    context_parts.append(f"{i}. {cat['category__name']}: ${cat['total']:,.2f}")
            # Transacciones recientes - CONCISO
            if financial_data['recent_transactions']:
                context_parts.append("\n🕒 ÚLTIMAS:")
                for t in financial_data['recent_transactions'][:3]:
                    context_parts.append(f"{t['date']}: {t['description']} (${t['amount']:,.2f})")
            # ANÁLISIS DE AI PERSONALIZADA - CONCISO
            ai_insights = self._get_ai_insights(organization_id, message, query_analysis)
            if ai_insights:
                context_parts.append("\n🧠 AI INSIGHTS:")
                context_parts.extend(ai_insights)
            return "\n".join(context_parts)
        except Exception as e:
            logger.error(f"Error preparando contexto financiero: {str(e)}")
            return "Error obteniendo datos financieros."
    
    def _determine_end_date_from_query(self, message: str, query_analysis: Dict = None) -> datetime:
        """Determina la fecha de fin basada en el análisis de la consulta."""
        from datetime import datetime, timedelta
        import re
        
        message_lower = message.lower()
        current_date = timezone.now()
        
        # Si la consulta menciona "hasta ahora", "actual", "hoy", etc., usar fecha actual
        if any(keyword in message_lower for keyword in ['hasta ahora', 'actual', 'hoy', 'actualmente', 'ahora']):
            # Para balances mensuales con "hasta ahora", usar la fecha actual (no el fin del mes)
            if any(keyword in message_lower for keyword in ['balance', 'mes', 'mensual', 'cada mes']):
                # Usar la fecha actual para que solo incluya hasta hoy
                return current_date
            else:
                return current_date
        
        # Si la consulta menciona un mes específico
        month_patterns = {
            'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
            'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
        }
        
        for month_name, month_num in month_patterns.items():
            if month_name in message_lower:
                # Buscar año en la consulta
                year_match = re.search(r'20\d{2}', message)
                if year_match:
                    year = int(year_match.group())
                else:
                    # Si no se especifica año, usar el año actual
                    year = current_date.year
                
                # Crear fecha del último día del mes especificado
                if month_num == 12:
                    end_date = datetime(year, month_num, 31)
                else:
                    end_date = datetime(year, month_num + 1, 1) - timedelta(days=1)
                
                return timezone.make_aware(end_date)
        
        # Si la consulta menciona "último mes", "mes pasado", etc.
        if any(keyword in message_lower for keyword in ['último mes', 'mes pasado', 'anterior']):
            # Calcular el último día del mes anterior
            first_day_current_month = current_date.replace(day=1)
            last_day_previous_month = first_day_current_month - timedelta(days=1)
            return last_day_previous_month
        
        # Si la consulta menciona "este mes", "mes actual"
        if any(keyword in message_lower for keyword in ['este mes', 'mes actual', 'corriente']):
            # Calcular el último día del mes actual
            last_day = monthrange(current_date.year, current_date.month)[1]
            end_date = current_date.replace(day=last_day, hour=23, minute=59, second=59)
            return end_date
        
        # Por defecto, usar fecha actual (hasta ahora)
        return current_date

    def _get_ai_insights(self, organization_id: int, message: str, query_analysis: Dict = None) -> List[str]:
        """Obtiene insights inteligentes de la AI personalizada."""
        insights = []
        
        try:
            # Obtener transacciones de la organización
            from transactions.models import Transaction
            transactions = list(Transaction.objects.filter(organization_id=organization_id))
            
            if not transactions:
                return insights
            
            # Entrenar modelos automáticamente si no están entrenados
            self._train_ai_models_if_needed(organization_id)
            
            # Determinar el tipo de análisis basado en el query_analysis
            intent_type = None
            if query_analysis:
                if hasattr(query_analysis, 'intent_type'):
                    intent_type = query_analysis.intent_type
                elif isinstance(query_analysis, dict):
                    intent_type = query_analysis.get('intent_type')
            
            # 1. ANÁLISIS DE COMPORTAMIENTO (para consultas generales o de comportamiento)
            if (not intent_type or 
                'behavior' in intent_type.lower() or 
                'comportamiento' in intent_type.lower() or
                'trend' in intent_type.lower() or
                'tendencia' in intent_type.lower()):
                
                if self.behavior_analyzer and self.behavior_analyzer.is_trained:
                    try:
                        behavior_analysis = self.behavior_analyzer.analyze_spending_patterns(transactions)
                        if behavior_analysis:
                            insights.append("📊 COMPORTAMIENTO:")
                            
                            # Patrones de gasto
                            if behavior_analysis.get('overall_patterns'):
                                patterns = behavior_analysis['overall_patterns']
                                insights.append(f"Días pico: {patterns.get('preferred_days', 'N/A')} | Horas: {patterns.get('preferred_hours', 'N/A')}")
                            
                            # Tendencias
                            if behavior_analysis.get('spending_trend'):
                                trend = behavior_analysis['spending_trend']
                                insights.append(f"Tendencia: {trend.get('trend_direction', 'N/A')} | Variabilidad: {trend.get('monthly_variability', 'N/A'):.1f}")
                            
                            # Anomalías detectadas
                            if behavior_analysis.get('anomalies_detected', 0) > 0:
                                insights.append(f"⚠️ {behavior_analysis['anomalies_detected']} anomalías detectadas")
                            
                    except Exception as e:
                        logger.warning(f"Error en análisis de comportamiento: {str(e)}")
            
            # 2. PREDICCIONES DE GASTOS (para consultas de predicción)
            if (intent_type and 
                ('prediction' in intent_type.lower() or 
                 'prediccion' in intent_type.lower() or
                 'future' in intent_type.lower() or
                 'futuro' in intent_type.lower())):
                
                if self.expense_predictor and self.expense_predictor.is_trained:
                    try:
                        # Predecir gastos para el próximo mes
                        next_month = timezone.now() + timedelta(days=30)
                        predicted_expense = self.expense_predictor.predict(next_month, 1)  # category_id=1 como ejemplo
                        
                        insights.append("🔮 PREDICCIÓN:")
                        insights.append(f"Próximo mes: ${predicted_expense:,.2f}")
                        
                        # Comparar con promedio histórico
                        avg_monthly_expense = float(sum(t.amount for t in transactions if t.type == 'EXPENSE')) / 12
                        if predicted_expense > avg_monthly_expense * 1.2:
                            insights.append(f"⚠️ 20% mayor al promedio (${avg_monthly_expense:,.2f})")
                        elif predicted_expense < avg_monthly_expense * 0.8:
                            insights.append(f"✅ 20% menor al promedio (${avg_monthly_expense:,.2f})")
                        
                    except Exception as e:
                        logger.warning(f"Error en predicción de gastos: {str(e)}")
            
            # 3. OPTIMIZACIÓN DE PRESUPUESTO (para consultas de optimización)
            if (intent_type and 
                ('optimization' in intent_type.lower() or 
                 'optimizacion' in intent_type.lower() or
                 'budget' in intent_type.lower() or
                 'presupuesto' in intent_type.lower())):
                
                if self.budget_optimizer and self.budget_optimizer.is_trained:
                    try:
                        # Obtener presupuesto actual (ejemplo)
                        current_budget = 10000  # Valor de ejemplo
                        optimization = self.budget_optimizer.optimize_budget_allocation(organization_id, current_budget)
                        
                        if optimization and not optimization.get('error'):
                            insights.append("💰 OPTIMIZACIÓN DE PRESUPUESTO:")
                            
                            suggestions = optimization.get('recommendations', [])
                            for suggestion in suggestions[:3]:  # Top 3 recomendaciones
                                insights.append(f"- {suggestion.get('message', 'Recomendación disponible')}")
                            
                            # Score de eficiencia
                            if optimization.get('category_analysis'):
                                avg_efficiency = sum(cat.get('efficiency', 0) for cat in optimization['category_analysis'].values()) / len(optimization['category_analysis'])
                                insights.append(f"- Eficiencia promedio del presupuesto: {avg_efficiency:.1%}")
                        
                    except Exception as e:
                        logger.warning(f"Error en optimización de presupuesto: {str(e)}")
            
            # 4. DETECCIÓN DE ANOMALÍAS (para consultas de anomalías)
            if (intent_type and 
                ('anomaly' in intent_type.lower() or 
                 'anomalia' in intent_type.lower() or
                 'unusual' in intent_type.lower() or
                 'extraño' in intent_type.lower())):
                
                if self.anomaly_detector:
                    try:
                        anomalies = self.anomaly_detector.detect_anomalies(transactions)
                        
                        if anomalies and len(anomalies) > 0:
                            insights.append("🚨 DETECCIÓN DE ANOMALÍAS:")
                            
                            for anomaly in anomalies[:3]:  # Top 3 anomalías
                                insights.append(f"- ⚠️ {anomaly.get('description', 'Anomalía detectada')}")
                                insights.append(f"  Monto: ${anomaly.get('amount', 0):,.2f} - Fecha: {anomaly.get('date', 'N/A')}")
                                insights.append(f"  Score de anomalía: {anomaly.get('anomaly_score', 0):.2f}")
                        
                    except Exception as e:
                        logger.warning(f"Error en detección de anomalías: {str(e)}")
            
            # 5. ANÁLISIS COMPARATIVO (para consultas de comparación)
            if (intent_type and 
                ('comparison' in intent_type.lower() or 
                 'comparacion' in intent_type.lower() or
                 'vs' in intent_type.lower() or
                 'versus' in intent_type.lower())):
                
                insights.append("📈 ANÁLISIS COMPARATIVO:")
                
                # Comparación mensual
                current_month = timezone.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                last_month = (current_month - timedelta(days=1)).replace(day=1)
                
                current_month_transactions = [t for t in transactions if t.date >= current_month]
                last_month_transactions = [t for t in transactions if t.date >= last_month and t.date < current_month]
                
                if current_month_transactions and last_month_transactions:
                    current_total = float(sum(t.amount for t in current_month_transactions if t.type == 'EXPENSE'))
                    last_total = float(sum(t.amount for t in last_month_transactions if t.type == 'EXPENSE'))
                    
                    if current_total > last_total:
                        change_percent = ((current_total - last_total) / last_total) * 100
                        insights.append(f"- 📈 Gastos aumentaron {change_percent:.1f}% vs mes anterior")
                    else:
                        change_percent = ((last_total - current_total) / last_total) * 100
                        insights.append(f"- 📉 Gastos disminuyeron {change_percent:.1f}% vs mes anterior")
            
            # 6. ANÁLISIS GENERAL DE TENDENCIAS (para consultas generales)
            if not intent_type or 'general' in intent_type.lower():
                if transactions:
                    insights.append("📈 ANÁLISIS DE TENDENCIAS:")
                    
                    # Calcular tendencias básicas
                    recent_transactions = [t for t in transactions if t.date >= timezone.now() - timedelta(days=30)]
                    older_transactions = [t for t in transactions if t.date < timezone.now() - timedelta(days=30) and t.date >= timezone.now() - timedelta(days=60)]
                    
                    if recent_transactions and older_transactions:
                        recent_total = float(sum(t.amount for t in recent_transactions if t.type == 'EXPENSE'))
                        older_total = float(sum(t.amount for t in older_transactions if t.type == 'EXPENSE'))
                        
                        if recent_total > older_total * 1.1:
                            insights.append(f"- 📈 Los gastos han aumentado {(recent_total/older_total - 1)*100:.1f}% en el último mes")
                        elif recent_total < older_total * 0.9:
                            insights.append(f"- 📉 Los gastos han disminuido {(1 - recent_total/older_total)*100:.1f}% en el último mes")
                        else:
                            insights.append("- ➡️ Los gastos se mantienen estables en el último mes")
                    
                    # Categoría con mayor crecimiento
                    if recent_transactions:
                        category_totals = {}
                        for t in recent_transactions:
                            if t.category:
                                cat_name = t.category.name
                                category_totals[cat_name] = category_totals.get(cat_name, 0) + float(abs(t.amount))
                        
                        if category_totals:
                            top_category = max(category_totals.items(), key=lambda x: x[1])
                            insights.append(f"- 🎯 Categoría con mayor gasto reciente: {top_category[0]} (${top_category[1]:,.2f})")
            
            # 7. RECOMENDACIONES PERSONALIZADAS PARA AHORRO
            if query_analysis and any(word in message.lower() for word in ['ahorrar', 'save', 'optimizar', 'optimize']):
                insights.append("\n💰 ANÁLISIS DE AHORRO:")
                
                # Calcular categorías con mayor potencial de ahorro
                if transactions:
                    category_expenses = {}
                    for t in transactions:
                        if t.type == 'EXPENSE' and t.category:
                            cat_name = t.category.name
                            category_expenses[cat_name] = category_expenses.get(cat_name, 0) + float(abs(t.amount))
                    
                    if category_expenses:
                        # Top 3 categorías con mayor gasto
                        top_categories = sorted(category_expenses.items(), key=lambda x: x[1], reverse=True)[:3]
                        
                        insights.append("🎯 Categorías con mayor potencial de ahorro:")
                        for i, (cat_name, amount) in enumerate(top_categories, 1):
                            potential_savings = amount * 0.1  # 10% de reducción
                            insights.append(f"{i}. {cat_name}: ${amount:,.2f} (ahorro potencial: ${potential_savings:,.2f})")
                        
                        # Recomendación específica
                        top_category, top_amount = top_categories[0]
                        insights.append(f"💡 Recomendación: Reduce 10% en {top_category} para ahorrar ${top_amount * 0.1:,.2f}/mes")
                        insights.append("❓ ¿Cuánto te gustaría ahorrar mensualmente? Te ayudo a crear un plan personalizado")
            
            # 8. RECOMENDACIONES GENERALES
            elif insights:
                insights.append("\n💡 RECOMENDACIONES PERSONALIZADAS:")
                
                # Basadas en patrones detectados
                if any("anomalías" in insight.lower() for insight in insights):
                    insights.append("- 🔍 Revisa las transacciones marcadas como anómalas para detectar posibles errores")
                
                if any("aumentado" in insight.lower() for insight in insights):
                    insights.append("- 📊 Considera revisar tus categorías de mayor gasto para identificar oportunidades de ahorro")
                
                if any("predicción" in insight.lower() for insight in insights):
                    insights.append("- 🎯 Planifica tu presupuesto basándote en las predicciones para evitar sorpresas")
                
                insights.append("- 📱 Revisa regularmente tus patrones de gasto para mantener el control financiero")
            
        except Exception as e:
            logger.error(f"Error obteniendo insights de AI: {str(e)}")
            insights.append("⚠️ No se pudieron obtener insights de AI en este momento")
        
        return insights

    def _create_savings_plan(self, organization_id: int, target_amount: float, message: str = "") -> Dict[str, Any]:
        """Crea un plan de ahorro personalizado basado en el objetivo del usuario."""
        try:
            print(f"🔍 SAVINGS_PLAN: Iniciando creación de plan para ${target_amount}")
            
            # Obtener datos financieros actuales
            financial_data = self._get_comprehensive_financial_data(organization_id)
            print(f"🔍 SAVINGS_PLAN: Datos financieros obtenidos: {len(financial_data)} campos")
            
            # Extraer información relevante
            monthly_expenses = float(financial_data.get('summary', {}).get('avg_monthly_expenses', 0))
            top_categories = financial_data.get('top_expense_categories', [])
            
            print(f"🔍 SAVINGS_PLAN: Gastos mensuales: ${monthly_expenses}")
            print(f"🔍 SAVINGS_PLAN: Categorías top: {len(top_categories)} categorías")
            print(f"🔍 SAVINGS_PLAN: Datos completos: {financial_data.keys()}")
            print(f"🔍 SAVINGS_PLAN: Summary keys: {financial_data.get('summary', {}).keys()}")
            
            if not top_categories:
                print(f"🔍 SAVINGS_PLAN: No hay categorías disponibles")
                return {
                    'success': False,
                    'message': 'No hay suficientes datos para crear un plan de ahorro'
                }
            
            # Calcular reducciones necesarias
            plan = {
                'target_amount': target_amount,
                'current_monthly_expenses': monthly_expenses,
                'reductions_needed': [],
                'strategies': [],
                'timeline': '3-6 meses',
                'success': True
            }
            
            # Distribuir la reducción entre las categorías más altas
            total_reduction_needed = target_amount
            remaining_reduction = total_reduction_needed
            
            # Calcular el total de gastos en las top categorías
            total_top_expenses = sum(float(cat.get('total', 0)) for cat in top_categories[:3])
            
            for i, category in enumerate(top_categories[:3]):  # Top 3 categorías
                category_name = category.get('category__name', 'Unknown')
                category_amount = float(category.get('total', 0))  # Convertir a float
                
                if category_amount <= 0:
                    continue
                
                # Calcular reducción proporcional basada en el peso de la categoría
                category_weight = category_amount / total_top_expenses if total_top_expenses > 0 else 0
                reduction_amount = min(
                    category_amount * 0.25,  # Máximo 25% de reducción por categoría
                    remaining_reduction * category_weight * 1.5  # Distribuir proporcionalmente
                )
                
                if reduction_amount > 0 and remaining_reduction > 0:
                    plan['reductions_needed'].append({
                        'category': category_name,
                        'current_amount': category_amount,
                        'reduction_amount': round(reduction_amount, 2),
                        'reduction_percentage': round((reduction_amount / category_amount) * 100, 1)
                    })
                    
                    # Crear estrategia específica
                    strategy = self._create_category_strategy(category_name, reduction_amount, category_amount)
                    plan['strategies'].append(strategy)
                    
                    remaining_reduction -= reduction_amount
                
                if remaining_reduction <= 0:
                    break
            
            # Si aún queda reducción por hacer, distribuir entre las categorías restantes
            if remaining_reduction > 0 and len(plan['reductions_needed']) > 0:
                # Distribuir el resto proporcionalmente
                total_allocated = sum(r['reduction_amount'] for r in plan['reductions_needed'])
                for reduction in plan['reductions_needed']:
                    if total_allocated > 0:
                        additional_reduction = (reduction['reduction_amount'] / total_allocated) * remaining_reduction
                        reduction['reduction_amount'] = round(reduction['reduction_amount'] + additional_reduction, 2)
                        reduction['reduction_percentage'] = round((reduction['reduction_amount'] / reduction['current_amount']) * 100, 1)
            
            print(f"🔍 SAVINGS_PLAN: Plan creado exitosamente con {len(plan['reductions_needed'])} reducciones")
            return plan
            
        except Exception as e:
            print(f"⚠️ ERROR creando plan de ahorro: {e}")
            import traceback
            print(f"🔍 SAVINGS_PLAN: Traceback completo:")
            traceback.print_exc()
            return {
                'success': False,
                'message': f'Error creando plan de ahorro: {str(e)}'
            }

    def _create_category_strategy(self, category: str, reduction_amount: float, current_amount: float) -> str:
        """Crea una estrategia específica para reducir gastos en una categoría."""
        strategies = {
            'Food & Dining': f'Reduce comidas fuera en 20% (ahorro: ${reduction_amount:.0f}/mes)',
            'Transportation': f'Usa transporte público 2 días/semana (ahorro: ${reduction_amount:.0f}/mes)',
            'Entertainment': f'Limita salidas de entretenimiento (ahorro: ${reduction_amount:.0f}/mes)',
            'Shopping': f'Implementa regla 24h antes de compras (ahorro: ${reduction_amount:.0f}/mes)',
            'Utilities': f'Optimiza uso de servicios básicos (ahorro: ${reduction_amount:.0f}/mes)',
            'Other Personal Expenses': f'Revisa gastos personales recurrentes (ahorro: ${reduction_amount:.0f}/mes)'
        }
        
        return strategies.get(category, f'Reduce gastos en {category} (ahorro: ${reduction_amount:.0f}/mes)')

    def _extract_savings_target(self, message: str) -> Optional[float]:
        """Extrae el objetivo de ahorro del mensaje del usuario con tolerancia a errores de ortografía."""
        import re
        from fuzzywuzzy import fuzz
        
        # Normalizar el mensaje para búsqueda fuzzy
        message_lower = message.lower().strip()
        
        # Palabras clave relacionadas con ahorro (con variantes comunes)
        savings_keywords = [
            'ahorrar', 'ahorra', 'ahorro', 'ahorros',  # Variantes de ahorrar
            'quiero', 'quisiera', 'quisiera', 'quisiera',  # Variantes de querer
            'meta', 'metas', 'objetivo', 'objetivos',  # Variantes de meta/objetivo
            'mensual', 'mensualmente', 'mes', 'por mes', 'al mes'  # Variantes de tiempo
        ]
        
        # Verificar si el mensaje contiene palabras relacionadas con ahorro
        has_savings_intent = False
        
        # Palabras que indican claramente intención de ahorro
        strong_savings_indicators = ['ahorrar', 'ahorra', 'ahorro', 'meta', 'objetivo']
        
        # Palabras que pueden ser ambiguas (necesitan contexto)
        weak_savings_indicators = ['quiero', 'quisiera', 'mensual', 'mensualmente']
        
        # Verificar indicadores fuertes primero
        for keyword in strong_savings_indicators:
            if fuzz.partial_ratio(message_lower, keyword) >= 80:
                has_savings_intent = True
                break
        
        # Si no hay indicadores fuertes, verificar indicadores débiles con contexto
        if not has_savings_intent:
            weak_count = 0
            for keyword in weak_savings_indicators:
                if fuzz.partial_ratio(message_lower, keyword) >= 80:
                    weak_count += 1
            
            # Solo considerar intención de ahorro si hay múltiples indicadores débiles
            if weak_count >= 2:
                has_savings_intent = True
        
        # Verificar que no sea una pregunta de consulta (cuánto, qué, cómo, etc.)
        query_words = ['cuanto', 'cuánto', 'que', 'qué', 'como', 'cómo', 'donde', 'dónde', 'cuando', 'cuándo']
        is_query = any(word in message_lower for word in query_words)
        
        # Si es una pregunta y no tiene indicadores fuertes de ahorro, no es intención de ahorro
        if is_query and not any(keyword in message_lower for keyword in strong_savings_indicators):
            has_savings_intent = False
        
        if not has_savings_intent:
            return None
        
        # Patrones mejorados para detectar cantidades (más robustos)
        patterns = [
            # Patrones específicos para ahorro con mejor formato de números
            r'ahorr[ao]r?\s*[:]?\s*\$?(\d{1,4}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'quier[ao]\s+ahorr[ao]r?\s*[:]?\s*\$?(\d{1,4}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'quisier[ao]\s+ahorr[ao]r?\s*[:]?\s*\$?(\d{1,4}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'meta\s*(?:de)?\s*ahorr[ao]r?\s*[:]?\s*\$?(\d{1,4}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'objetivo\s*(?:es)?\s*ahorr[ao]r?\s*[:]?\s*\$?(\d{1,4}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            
            # Patrones para emergencias y otros montos específicos
            r'emergencia\s*(?:de)?\s*\$?(\d{1,4}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'necesito\s*\$?(\d{1,4}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'dinero\s*(?:para)?\s*\$?(\d{1,4}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'suficiente\s*(?:para)?\s*\$?(\d{1,4}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            
            # Patrones con números seguidos de palabras de tiempo
            r'(\d{1,4}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s+(?:dólares?|dolares?|pesos?|euros?)',
            r'\$(\d{1,4}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'(\d{1,4}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s+mensual(?:mente)?',
            r'(\d{1,4}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s+por\s+mes',
            r'(\d{1,4}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s+al\s+mes',
            
            # Patrones más flexibles para capturar números cerca de palabras de ahorro
            r'(\d{1,4}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*(?:para\s+)?ahorr[ao]r?',
            r'ahorr[ao]r?\s*(\d{1,4}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            
            # Patrones para números sin formato específico (fallback)
            r'(\d{1,4}(?:[.,]\d{3})*(?:[.,]\d{2})?)(?:\s|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message.lower())
            if match:
                num = match.group(1)
                # Normalizar: 1.000,50 o 1,000.50 → 1000.50
                if ',' in num and '.' in num:
                    if num.find(',') > num.find('.'):
                        # 1.000,50 (europeo)
                        num = num.replace('.', '').replace(',', '.')
                    else:
                        # 1,000.50 (americano)
                        num = num.replace(',', '')
                elif ',' in num:
                    # Si solo hay coma, puede ser decimal o miles
                    if num.count(',') == 1 and len(num.split(',')[1]) == 2:
                        # 1000,50 → 1000.50
                        num = num.replace(',', '.')
                    else:
                        # 1,000 → 1000
                        num = num.replace(',', '')
                elif '.' in num:
                    # Si solo hay punto, puede ser decimal
                    if num.count('.') == 1 and len(num.split('.')[1]) == 2:
                        pass  # 1000.50 ya está bien
                    else:
                        # 1.000 → 1000
                        num = num.replace('.', '')
                try:
                    return float(num)
                except ValueError:
                    continue
        
        print(f"🔍 EXTRACT_SAVINGS_TARGET: '{message}'")
        print(f"🔍 PATRONES PROBADOS: {len(patterns)}")
        
        return None

    def _force_correct_language(self, response: str, user_message: str) -> str:
        """Fuerza el idioma correcto en la respuesta basado en el mensaje del usuario."""
        print(f"🔍 FORCE_CORRECT_LANGUAGE: Llamada con mensaje '{user_message[:50]}...'")
        detected_language = self._detect_language(user_message)
        
        if detected_language == 'spanish':
            try:
                # Detectar el idioma de la respuesta usando langdetect
                response_language = detect(response)
                
                # Si la respuesta está en inglés, traducir al español
                if response_language == 'en':
                    print(f"🔍 TRADUCCIÓN: Detectado inglés en respuesta, traduciendo al español...")
                    if self.translation_service:
                        response = self.translation_service.translate(response, target_lang='es')
                    else:
                        from ai.core.translation_service import translate
                        response = translate(response, target_lang='es')
                    print(f"✅ Traducción completada")
                else:
                    print(f"🔍 IDIOMA: Respuesta ya está en español ({response_language})")
                    
            except Exception as e:
                print(f"Error detectando/traduciendo idioma: {e}")
                # Fallback: traducir si hay muchas palabras en inglés
                english_words = ['the', 'and', 'for', 'with', 'from', 'by', 'on', 'in', 'to', 'of', 'is', 'are', 'was', 'were', 'be', 'have', 'has', 'had']
                english_count = sum(1 for word in english_words if word in response.lower())
                if english_count > 3:
                    try:
                        from ai.core.translation_service import translate
                        response = translate(response, target_lang='es')
                    except Exception as e2:
                        print(f"Error en traducción fallback: {e2}")
            
            # Lógica inteligente para formato conciso
            print(f"🔍 ANTES DE FORMATO CONCISO: {len(response)} chars")
            
            # Solo aplicar formato conciso si:
            # 1. Es muy larga (>1500 chars) Y
            # 2. No contiene información específica útil (números, fechas, categorías)
            # 3. No es un plan de ahorro personalizado
            should_force_concise = False
            
            if len(response) > 1500:
                # Verificar si la respuesta contiene información específica útil
                has_specific_info = (
                    any(char.isdigit() for char in response) or  # Contiene números
                    '$' in response or  # Contiene símbolos de moneda
                    '%' in response or  # Contiene porcentajes
                    any(word in response.lower() for word in ['balance', 'gasto', 'ingreso', 'ahorro', 'predicción', 'plan']) or  # Contiene términos financieros específicos
                    any(word in response.lower() for word in ['coffee', 'restaurante', 'utilities', 'internet', 'phone'])  # Contiene categorías específicas
                )
                
                # Verificar si es un plan de ahorro personalizado
                is_savings_plan = any(word in response.lower() for word in ['plan de ahorro', 'objetivo:', 'reducir'])
                
                # Solo forzar formato si NO tiene información específica útil Y NO es un plan de ahorro
                should_force_concise = not has_specific_info and not is_savings_plan
                
                if should_force_concise:
                    response = self._force_concise_format(response)
                    print(f"🔍 FORMATO CONCISO APLICADO: Respuesta larga sin info específica")
                else:
                    print(f"🔍 RESPUESTA MANTENIDA: Contiene información específica útil o es plan de ahorro")
            else:
                print(f"🔍 RESPUESTA MANTENIDA: Longitud aceptable ({len(response)} chars)")
            
        return response

    def _force_concise_format(self, response: str) -> str:
        """Fuerza el formato conciso en la respuesta."""
        try:
            # Si la respuesta es muy larga (más de 200 caracteres), forzar formato conciso
            if len(response) > 200:
                print(f"🔍 FORMATO CONCISO: Respuesta muy larga ({len(response)} chars), forzando formato...")
                
                # Detectar tipo de pregunta basado en palabras clave
                response_lower = response.lower()
                
                if any(word in response_lower for word in ['ahorrar', 'save', 'gastar', 'spend']):
                    # Detectar si hay un objetivo específico en el mensaje original
                    if any(word in response_lower for word in ['$', 'dólar', 'dolar', 'peso', 'euro', '500', '1000', '2000', '3000', '4000', '5000']):
                        return """🎯 **Plan de Ahorro Personalizado**

• Analiza tus gastos actuales
• Calcula reducciones necesarias
• Define estrategias específicas

💡 **Te ayudo a crear un plan detallado para alcanzar tu meta**"""
                    else:
                        return """💰 **Optimización de Gastos**

• Analiza tus categorías más altas
• Identifica oportunidades de ahorro
• Revisa tendencias de gasto

💡 **¿Cuánto te gustaría ahorrar mensualmente? Te ayudo a crear un plan personalizado**"""
                
                elif any(word in response_lower for word in ['balance', 'dinero', 'money', 'cuanto']):
                    return """💰 **Balance Actual**

• Revisa tus datos financieros
• Analiza tendencias mensuales
• Identifica patrones de gasto

💡 **¿Necesitas ayuda con algo específico?**"""
                
                elif any(word in response_lower for word in ['prediccion', 'predict', 'futuro', 'next']):
                    return """🔮 **Análisis Predictivo**

• Revisa tendencias históricas
• Analiza patrones de comportamiento
• Identifica factores de riesgo

💡 **¿Qué período quieres analizar?**"""
                
                else:
                    return """💰 **Resumen Financiero**

• Revisa tus datos actuales
• Analiza patrones de gasto
• Identifica oportunidades

💡 **¿En qué te puedo ayudar específicamente?**"""
            
            return response
            
        except Exception as e:
            print(f"⚠️ ERROR EN FORMATO CONCISO: {e}")
            return response

    def _detect_language(self, message: str) -> str:
        """Detecta el idioma de la pregunta del usuario usando TranslationService del core AI."""
        try:
            if self.translation_service:
                detected_lang = self.translation_service.detect_language(message)
                print(f"🔍 LANGDETECT: '{message}' -> {detected_lang}")
                
                # Mapear códigos de langdetect a nuestros idiomas
                if detected_lang == 'es':  # Español
                    return 'spanish'
                elif detected_lang == 'en':  # Inglés
                    return 'english'
                else:
                    # Fallback: usar palabras clave para otros idiomas
                    return self._fallback_language_detection(message)
            else:
                # Fallback al método original
                from ai.core.translation_service import detect_language
                detected_lang = detect_language(message)
                print(f"🔍 LANGDETECT: '{message}' -> {detected_lang}")
                
                # Mapear códigos de langdetect a nuestros idiomas
                if detected_lang == 'es':  # Español
                    return 'spanish'
                elif detected_lang == 'en':  # Inglés
                    return 'english'
                else:
                    # Fallback: usar palabras clave para otros idiomas
                    return self._fallback_language_detection(message)
                
        except Exception as e:
            print(f"⚠️ ERROR LANGDETECT: {e}, usando fallback")
            return self._fallback_language_detection(message)
    
    def _fallback_language_detection(self, message: str) -> str:
        """Detección de idioma por palabras clave como fallback."""
        # Palabras únicas por idioma
        spanish_words = ['cuanto', 'mes', 'hasta', 'ahora', 'cada', 'quedo', 'dinero', 'gasto', 'ingreso', 'me', 'de', 'como', 'puedo', 'ahorrar', 'mas', 'que', 'cual', 'es', 'mi', 'balance', 'actual', 'tengo', 'necesito', 'ayuda', 'informacion', 'datos', 'cuenta', 'banco', 'tarjeta', 'credito', 'debito', 'prestamo', 'inversion', 'ahorro', 'gasto', 'ingreso', 'salario', 'sueldo', 'freelance', 'trabajo', 'negocio', 'empresa', 'factura', 'recibo', 'pago', 'cobro', 'transferencia', 'deposito', 'retiro', 'cajero', 'atm', 'bancario', 'financiero', 'economico', 'presupuesto', 'planificacion', 'objetivo', 'meta', 'objetivo', 'proyeccion', 'tendencia', 'analisis', 'reporte', 'resumen', 'estado', 'situacion', 'posicion', 'liquidez', 'solvencia', 'rentabilidad', 'productividad', 'eficiencia', 'optimizacion', 'mejora', 'reduccion', 'aumento', 'incremento', 'disminucion', 'variacion', 'cambio', 'evolucion', 'crecimiento', 'desarrollo', 'progreso', 'avance', 'retroceso', 'caida', 'subida', 'bajada', 'alza', 'descenso', 'ascenso', 'descenso']
        
        english_words = ['how', 'much', 'month', 'until', 'now', 'each', 'left', 'money', 'spent', 'income', 'do', 'have', 'can', 'save', 'more', 'what', 'is', 'my', 'balance', 'current', 'need', 'help', 'information', 'data', 'account', 'bank', 'card', 'credit', 'debit', 'loan', 'investment', 'saving', 'expense', 'salary', 'wage', 'freelance', 'work', 'business', 'company', 'bill', 'receipt', 'payment', 'charge', 'transfer', 'deposit', 'withdrawal', 'atm', 'banking', 'financial', 'economic', 'budget', 'planning', 'goal', 'target', 'projection', 'trend', 'analysis', 'report', 'summary', 'status', 'situation', 'position', 'liquidity', 'solvency', 'profitability', 'productivity', 'efficiency', 'optimization', 'improvement', 'reduction', 'increase', 'growth', 'decrease', 'variation', 'change', 'evolution', 'development', 'progress', 'advance', 'decline', 'rise', 'fall', 'drop', 'climb', 'descent', 'ascent']
        
        message_lower = message.lower()
        
        spanish_count = sum(1 for word in spanish_words if word in message_lower)
        english_count = sum(1 for word in english_words if word in message_lower)
        
        # Palabras decisivas (si están presentes, determinan el idioma)
        spanish_decisive = ['cuanto', 'quedo', 'me', 'como', 'puedo', 'ahorrar', 'mas', 'cual', 'es', 'mi', 'tengo', 'necesito', 'ayuda', 'informacion', 'datos', 'cuenta', 'banco', 'tarjeta', 'credito', 'debito', 'prestamo', 'inversion', 'ahorro', 'gasto', 'ingreso', 'salario', 'sueldo', 'freelance', 'trabajo', 'negocio', 'empresa', 'factura', 'recibo', 'pago', 'cobro', 'transferencia', 'deposito', 'retiro', 'cajero', 'atm', 'bancario', 'financiero', 'economico', 'presupuesto', 'planificacion', 'objetivo', 'meta', 'objetivo', 'proyeccion', 'tendencia', 'analisis', 'reporte', 'resumen', 'estado', 'situacion', 'posicion', 'liquidez', 'solvencia', 'rentabilidad', 'productividad', 'eficiencia', 'optimizacion', 'mejora', 'reduccion', 'aumento', 'incremento', 'disminucion', 'variacion', 'cambio', 'evolucion', 'crecimiento', 'desarrollo', 'progreso', 'avance', 'retroceso', 'caida', 'subida', 'bajada', 'alza', 'descenso', 'ascenso', 'descenso']
        
        english_decisive = ['how', 'much', 'do', 'have', 'can', 'save', 'more', 'what', 'is', 'my', 'need', 'help', 'information', 'data', 'account', 'bank', 'card', 'credit', 'debit', 'loan', 'investment', 'saving', 'expense', 'salary', 'wage', 'freelance', 'work', 'business', 'company', 'bill', 'receipt', 'payment', 'charge', 'transfer', 'deposit', 'withdrawal', 'atm', 'banking', 'financial', 'economic', 'budget', 'planning', 'goal', 'target', 'projection', 'trend', 'analysis', 'report', 'summary', 'status', 'situation', 'position', 'liquidity', 'solvency', 'profitability', 'productivity', 'efficiency', 'optimization', 'improvement', 'reduction', 'increase', 'growth', 'decrease', 'variation', 'change', 'evolution', 'development', 'progress', 'advance', 'decline', 'rise', 'fall', 'drop', 'climb', 'descent', 'ascent']
        
        for word in spanish_decisive:
            if word in message_lower:
                detected_language = 'spanish'
                break
        else:
            for word in english_decisive:
                if word in message_lower:
                    detected_language = 'english'
                    break
            else:
                detected_language = 'spanish' if spanish_count > english_count else 'english'
        
        # Debug log
        print(f"🔍 FALLBACK IDIOMA: '{message}'")
        print(f"   Palabras españolas encontradas: {spanish_count}")
        print(f"   Palabras inglesas encontradas: {english_count}")
        print(f"   Idioma detectado: {detected_language}")
        
        return detected_language

    def _map_query_analysis_to_prompt_type(self, query_analysis) -> str:
        """Mapea el análisis de consulta al tipo de prompt apropiado."""
        if hasattr(query_analysis, 'intent_type'):
            intent_type = query_analysis.intent_type
        else:
            intent_type = query_analysis.get('intent_type', 'general')
        
        # Mapear tipos de intención a tipos de prompt
        intent_to_prompt_mapping = {
            'balance_inquiry': 'balance',
            'expense_analysis': 'expense',
            'income_analysis': 'income',
            'trend_analysis': 'trend',
            'prediction': 'prediction',
            'optimization': 'optimization',
            'anomaly_detection': 'anomaly',
            'comparison': 'comparison',
            'savings_plan': 'savings',
            'budget_analysis': 'budget',
            'report_generation': 'report',
            'clarification_request': 'clarification'
        }
        
        return intent_to_prompt_mapping.get(intent_type, 'general')

    def _create_enhanced_system_prompt(self, financial_context: str, query_analysis: Dict = None, user_message: str = "") -> str:
        """Crea un prompt del sistema mejorado y específico según el tipo de consulta."""
        
        # Detectar idioma de la pregunta
        language = self._detect_language(user_message)
        
        # Determinar el tipo de consulta
        intent_type = None
        if query_analysis:
            if hasattr(query_analysis, 'intent_type'):
                intent_type = query_analysis.intent_type
            elif isinstance(query_analysis, dict):
                intent_type = query_analysis.get('intent_type')
        
        print(f"🔍 DEBUG PROMPT: Idioma={language}, Intent={intent_type}")
        
        # Crear prompt base según el idioma
        if language == 'spanish':
            base_prompt = f"""Eres un asistente financiero inteligente y útil. Responde en español de manera clara y específica usando los datos financieros proporcionados.

DATOS FINANCIEROS:
{financial_context}

**INSTRUCCIONES PRINCIPALES:**
1. **USA LOS DATOS REALES** proporcionados para dar respuestas específicas
2. **RESPONDE LA PREGUNTA DIRECTAMENTE** sin evasivas
3. **INCLUYE NÚMEROS CONCRETOS** cuando estén disponibles
4. **SÉ ÚTIL Y ESPECÍFICO** en lugar de genérico

**TIPOS DE PREGUNTAS Y RESPUESTAS:**

**Para preguntas sobre gastos específicos:**
- Busca en "top_expense_categories" la categoría mencionada
- Proporciona el monto exacto y porcentaje del total
- Incluye comparación con meses anteriores si está disponible

**Para preguntas sobre balance:**
- Usa "Monthly Average Income" y "Monthly Average Expenses" del summary
- Calcula: Balance = Ingresos - Gastos
- Menciona si es positivo (ahorro) o negativo (déficit)

**Para predicciones:**
- Usa los datos históricos disponibles
- Proporciona estimaciones basadas en tendencias
- Menciona el nivel de confianza

**Para anomalías:**
- Analiza los datos de transacciones recientes
- Identifica patrones inusuales
- Proporciona explicaciones específicas

**Para tendencias:**
- Usa los datos de "monthly_balances"
- Identifica patrones claros
- Proporciona porcentajes de cambio

**Para comparaciones:**
- Compara períodos específicos
- Proporciona diferencias en montos y porcentajes
- Identifica tendencias en la comparación

**FORMATO DE RESPUESTA:**
- Responde directamente la pregunta
- Incluye números específicos cuando sea posible
- Usa emojis para claridad (💰, 📊, 🔍, etc.)
- Mantén un tono amigable pero profesional
- Si no tienes datos suficientes, dilo claramente

**IMPORTANTE:**
- NO uses números de ejemplo, usa los datos reales
- NO seas genérico, sé específico
- NO evadas la pregunta, responde directamente
- SIEMPRE usa los datos proporcionados"""
        else:
            # Prompt en inglés
            base_prompt = f"""You are a direct and precise financial assistant. Respond in a CONCRETE, HUMAN way without unnecessary verbosity.

**FUNDAMENTAL RULES:**
1. **LANGUAGE**: ALWAYS RESPOND IN ENGLISH
2. **FORMAT**: For monthly balances, ALWAYS include month and year: "2025-01: $X,XXX.XX"
3. **DIRECT**: Get to the point immediately, without introductory phrases.

FINANCIAL DATA:
{financial_context}

CRITICAL INSTRUCTIONS:
1. **BE DIRECT**: Get to the point immediately. Don't use unnecessary introductory phrases.
2. **CONCRETE RESPONSES**: Give specific numbers, exact dates, precise percentages.
3. **HUMAN LANGUAGE**: Speak like a financial expert friend, not a corporate bot.
4. **NO BEATING AROUND THE BUSH**: If you don't have data, say it clearly. If you have data, use it.
5. **CLEAR FORMAT**: Use $ and numbers with commas. Use emojis strategically for clarity.
6. **PRIORITIZE**: If there are multiple points, order by importance.

QUESTION TYPES:
- Balances, income, expenses → Exact numbers
- Comparisons → Differences in $ and %
- Predictions → Estimates based on real data
- Anomalies → What, when, how much, why
- Optimization → 3-5 specific actions
- Patterns → Clear trends and data

AI CAPABILITIES:
- Behavior analysis
- ML predictions
- Budget optimization
- Anomaly detection
- Personalized recommendations

RESPONSE FORMAT:
- Maximum 3-4 paragraphs
- Specific numbers always
- Concrete actions
- No filler phrases
- **Use AI insights if available**

SPECIFIC INSTRUCTIONS FOR BALANCES:
- When asked about "monthly balance" or "how much I have left each month", use data from "BALANCE BY MONTH" table
- DON'T use terms like "residual balance" - use "balance" or "monthly balance"
- For monthly averages, use "Monthly Average" data from summary
- For current balance, use "Current Balance" from summary

SPECIFIC RESPONSES:
- If they ask "balance each month" or "monthly balance" → LIST all months with their individual balance
- If they ask "average" → Only monthly average
- If they ask "current balance" → Only total current balance
- If they ask "how much I have left" → LIST all months with their individual balance

**MANDATORY FORMAT FOR MONTHLY BALANCES:**
- ALWAYS include month and year: "2025-01: $X,XXX.XX"
- NOT just numbers: "$X,XXX.XX"
- Use bullets (•) or numbers (1., 2., etc.)

Remember: Your goal is to be helpful, precise and educational in financial matters, making the most of the available personalized AI capabilities.

**FINAL INSTRUCTION: ALWAYS RESPOND IN ENGLISH.**"""
        
        # Agregar instrucciones específicas según el tipo de intent
        if intent_type:
            specific_instructions = ""
            
            if 'comparison' in intent_type.lower() or 'comparacion' in intent_type.lower():
                specific_instructions = """

**INSTRUCCIONES ESPECÍFICAS PARA COMPARACIONES:**
- Compara períodos específicos (mes actual vs mes anterior, etc.)
- Proporciona diferencias exactas en montos y porcentajes
- Identifica tendencias en la comparación
- Usa datos de "monthly_balances" para comparaciones temporales
- Para comparaciones de categorías, usa "top_expense_categories"
- Siempre incluye el contexto temporal de la comparación"""
            
            elif 'anomaly' in intent_type.lower() or 'anomalia' in intent_type.lower():
                specific_instructions = """

**INSTRUCCIONES ESPECÍFICAS PARA DETECCIÓN DE ANOMALÍAS:**
- Analiza transacciones recientes para patrones inusuales
- Identifica montos que se desvían significativamente del promedio
- Proporciona explicaciones específicas para las anomalías detectadas
- Usa datos de "recent_transactions" para análisis detallado
- Incluye recomendaciones para prevenir futuras anomalías"""
            
            elif 'prediction' in intent_type.lower() or 'prediccion' in intent_type.lower():
                specific_instructions = """

**INSTRUCCIONES ESPECÍFICAS PARA PREDICCIONES:**
- Usa datos históricos de "monthly_balances" para tendencias
- Proporciona estimaciones basadas en patrones reales
- Menciona el nivel de confianza de la predicción
- Incluye factores que podrían afectar la predicción
- Usa datos de "Monthly Average" como línea base"""
            
            elif 'optimization' in intent_type.lower() or 'optimizacion' in intent_type.lower():
                specific_instructions = """

**INSTRUCCIONES ESPECÍFICAS PARA OPTIMIZACIÓN:**
- Analiza "top_expense_categories" para identificar oportunidades
- Proporciona 3-5 acciones específicas y concretas
- Incluye estimaciones de ahorro potencial
- Usa datos de "Monthly Average Expenses" como referencia
- Prioriza las categorías con mayor potencial de optimización"""
            
            elif 'trend' in intent_type.lower() or 'tendencia' in intent_type.lower():
                specific_instructions = """

**INSTRUCCIONES ESPECÍFICAS PARA ANÁLISIS DE TENDENCIAS:**
- Usa datos de "monthly_balances" para identificar patrones
- Proporciona porcentajes de cambio entre períodos
- Identifica tendencias claras (creciente, decreciente, estable)
- Incluye factores que podrían explicar las tendencias
- Usa datos históricos para proyecciones futuras"""
            
            if specific_instructions:
                base_prompt += specific_instructions
        
        return base_prompt

    def chat(self, user_id: int, organization_id: int, message: str) -> Dict[str, Any]:
        """Procesa un mensaje del usuario y genera una respuesta usando el sistema AI core completo."""
        start_time = time.time()
        
        try:
            # Usar el enhanced query parser del core AI si está disponible
            if self.enhanced_query_parser:
                query_analysis = self.enhanced_query_parser.parse_query(message)
                logger.info(f"Enhanced query analysis: {query_analysis.intent_type} (confidence: {query_analysis.confidence_score})")
            else:
                # Fallback al parser básico
                query_analysis = self.query_parser.parse_query(message)
            
            # Verificar privacidad usando PrivacyGuard del core AI
            if self.privacy_guard:
                is_safe, privacy_violation = self.privacy_guard.check_query(
                    message, str(user_id), str(organization_id)
                )
                if not is_safe:
                    logger.warning(f"Privacy violation detected: {privacy_violation}")
                    return {
                        'success': False,
                        'response': self.privacy_guard.get_violation_message(privacy_violation, 'es'),
                        'model_used': self.model_name,
                        'tokens_used': 0,
                        'processing_time': time.time() - start_time,
                        'timestamp': timezone.now().isoformat(),
                        'privacy_violation': True
                    }
            
            # Detectar si el usuario especifica un objetivo de ahorro
            print(f"🔍 CHAT: Analizando mensaje: '{message}'")
            savings_target = self._extract_savings_target(message)
            print(f"🔍 CHAT: Objetivo de ahorro detectado: {savings_target}")
            
            # Obtener contexto de conversación usando ContextManager del core AI
            if self.context_manager:
                conversation_context = self.context_manager.get_context_history(str(user_id), str(organization_id))
            else:
                conversation_context = self._get_conversation_context(user_id, organization_id)
            
            # Manejar preguntas múltiples si se detectan
            multiple_questions = getattr(query_analysis, 'multiple_questions', False)
            if multiple_questions:
                return self._handle_multiple_questions(user_id, organization_id, message, query_analysis, conversation_context, start_time)
            
            # Preparar contexto financiero completo
            financial_context = self._prepare_comprehensive_financial_context(
                user_id, organization_id, message, query_analysis
            )
            
            # Si hay un objetivo de ahorro específico, crear un plan personalizado
            plan_summary = None
            if savings_target and savings_target > 0:
                print(f"🔍 CHAT: Creando plan de ahorro para ${savings_target}")
                savings_plan = self._create_savings_plan(organization_id, savings_target, message)
                if savings_plan.get('success'):
                    print(f"🔍 CHAT: Plan de ahorro creado exitosamente")
                    # Agregar el plan al contexto financiero
                    plan_context = f"\n\n🎯 PLAN DE AHORRO PERSONALIZADO (Objetivo: ${savings_target:,.2f}/mes):\n"
                    plan_context += f"• Gastos mensuales actuales: ${savings_plan['current_monthly_expenses']:,.2f}\n"
                    for reduction in savings_plan['reductions_needed']:
                        plan_context += f"• {reduction['category']}: Reducir ${reduction['reduction_amount']:,.2f} ({reduction['reduction_percentage']}%)\n"
                    for strategy in savings_plan['strategies']:
                        plan_context += f"• Estrategia: {strategy}\n"
                    plan_context += f"• Timeline: {savings_plan['timeline']}\n"
                    financial_context += plan_context

                    # Crear resumen conciso para el usuario
                    plan_summary = "\n" + "🎯 **Plan de Ahorro Personalizado**\n"
                    plan_summary += f"Objetivo: ${savings_target:,.2f}/mes\n"
                    plan_summary += f"Gasto mensual actual: ${savings_plan['current_monthly_expenses']:,.2f}\n"
                    
                    # Calcular total de reducciones
                    total_reductions = sum(r['reduction_amount'] for r in savings_plan['reductions_needed'])
                    plan_summary += f"Total reducciones: ${total_reductions:,.2f}/mes\n\n"
                    
                    for reduction in savings_plan['reductions_needed']:
                        plan_summary += f"- {reduction['category']}: reducir ${reduction['reduction_amount']:,.0f} ({reduction['reduction_percentage']}%)\n"
                    
                    plan_summary += f"\n⏳ Plazo estimado: {savings_plan['timeline']}\n"
                    
                    # Agregar recomendación adicional si el total no alcanza el objetivo
                    if total_reductions < savings_target:
                        remaining = savings_target - total_reductions
                        plan_summary += f"💡 Considera aumentar ingresos en ${remaining:,.2f}/mes para alcanzar tu objetivo completo\n"
                else:
                    print(f"🔍 CHAT: Error creando plan de ahorro: {savings_plan.get('error', 'Error desconocido')}")
            else:
                print(f"🔍 CHAT: No se detectó objetivo de ahorro válido")
            
            # Crear prompt del sistema usando PromptBuilder del core AI
            if self.prompt_builder:
                # Determinar el tipo de prompt basado en el análisis de la consulta
                prompt_type = self._map_query_analysis_to_prompt_type(query_analysis)
                system_prompt = self.prompt_builder.get_system_prompt(prompt_type, 'es')
                
                # Agregar contexto financiero al prompt
                if financial_context:
                    system_prompt += f"\n\nDATOS FINANCIEROS:\n{financial_context}"
            else:
                # Fallback al método original
                system_prompt = self._create_enhanced_system_prompt(financial_context, query_analysis, message)
            
            # Construir mensajes para el LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
            
            # Agregar contexto de conversación si existe
            if conversation_context:
                # Insertar mensajes de conversación antes del mensaje actual
                for msg in conversation_context[-4:]:  # Últimos 4 mensajes
                    messages.insert(-1, msg)
            
            # Llamar al LLM
            response = self._call_llm(messages)
            
            if response.get('success'):
                # Usar NLRenderer del core AI para formatear la respuesta
                if self.nl_renderer:
                    try:
                        # Determinar el tipo de template basado en el intent
                        intent_type = query_analysis.intent_type if hasattr(query_analysis, 'intent_type') else 'general'
                        template_name = 'single_metric'  # Default template
                        
                        if 'balance' in intent_type.lower():
                            template_name = 'single_metric'
                        elif 'comparison' in intent_type.lower():
                            template_name = 'comparison'
                        elif 'goal' in intent_type.lower():
                            template_name = 'goal_progress'
                        elif 'anomaly' in intent_type.lower():
                            template_name = 'anomaly_alert'
                        
                        # Preparar datos para el template basado en el tipo
                        if template_name == 'single_metric':
                            # Extraer información del balance de la respuesta del LLM
                            # Buscar un número en la respuesta que represente el balance
                            import re
                            balance_match = re.search(r'\$?([\d,]+\.?\d*)', response['response'])
                            balance_value = float(balance_match.group(1).replace(',', '')) if balance_match else 0.0
                            
                            final_response = self.nl_renderer.render_single_metric(
                                metric_name='Balance',
                                value=balance_value,
                                currency='$',
                                period='actual',
                                language='es'
                            )
                        else:
                            # Para otros templates, usar el método render_response
                            template_data = {
                                'response': response['response'],
                                'intent_type': intent_type,
                                'financial_data': financial_data if 'financial_data' in locals() else {}
                            }
                            
                            final_response = self.nl_renderer.render_response(
                                template_name=template_name,
                                data=template_data,
                                language='es'
                            )
                    except Exception as e:
                        logger.warning(f"Error usando NLRenderer, fallback a método original: {str(e)}")
                        final_response = self._force_correct_language(response['response'], message)
                else:
                    # Fallback al método original
                    final_response = self._force_correct_language(response['response'], message)
                
                # Si hay un plan de ahorro, anteponer el resumen al mensaje final
                if plan_summary:
                    final_response = plan_summary + "\n" + final_response
                
                # Actualizar contexto de conversación usando ContextManager del core AI
                if self.context_manager:
                    self.context_manager.store_context(
                        user_id=str(user_id),
                        organization_id=str(organization_id),
                        query=message,
                        parsed_intent=query_analysis.__dict__ if hasattr(query_analysis, '__dict__') else query_analysis,
                        confidence_score=query_analysis.confidence_score if hasattr(query_analysis, 'confidence_score') else 0.5,
                        financial_data=financial_data if 'financial_data' in locals() else None,
                        response_summary=final_response[:200]
                    )
                else:
                    # Fallback al método original
                    conversation_context.extend([
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": final_response}
                    ])
                    self._save_conversation_context(user_id, organization_id, conversation_context)
                
                processing_time = time.time() - start_time
                
                # Generar sugerencias de seguimiento usando FollowUpSuggester del core AI
                followup_suggestions = []
                if self.followup_suggester:
                    try:
                        intent_type = query_analysis.intent_type if hasattr(query_analysis, 'intent_type') else 'general'
                        followup_suggestions = self.followup_suggester.generate_followup_suggestions(
                            last_intent=intent_type,
                            context={'user_query': message, 'financial_data': financial_data if 'financial_data' in locals() else None},
                            num_suggestions=3
                        )
                        # Convertir a formato simple para la respuesta
                        followup_suggestions = [s.question if hasattr(s, 'question') else str(s) for s in followup_suggestions[:3]]  # Top 3 sugerencias
                    except Exception as e:
                        logger.warning(f"Error generando sugerencias de seguimiento: {str(e)}")
                
                # Serializar todos los campos posibles
                def safe_serialize(obj):
                    if isinstance(obj, (str, int, float, bool, type(None))):
                        return obj
                    elif hasattr(obj, '__dict__'):
                        return {k: safe_serialize(v) for k, v in obj.__dict__.items()}
                    elif isinstance(obj, dict):
                        return {k: safe_serialize(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple, set)):
                        return [safe_serialize(i) for i in obj]
                    else:
                        return str(obj)
                
                return {
                    'success': True,
                    'response': final_response,
                    'model_used': self.model_name,
                    'tokens_used': response.get('tokens_used', 0),
                    'processing_time': processing_time,
                    'timestamp': timezone.now().isoformat(),
                    'query_analysis': safe_serialize(query_analysis),
                    'followup_suggestions': safe_serialize(followup_suggestions)
                }
            else:
                # Fallback response
                fallback_response = self._get_fallback_response(message)
                processing_time = time.time() - start_time
                
                return {
                    'success': False,
                    'response': fallback_response,
                    'model_used': self.model_name,
                    'tokens_used': 0,
                    'processing_time': processing_time,
                    'timestamp': timezone.now().isoformat(),
                    'error': response.get('error', 'Error desconocido')
                }
                
        except Exception as e:
            logger.error(f"Error en chat: {str(e)}")
            processing_time = time.time() - start_time
            
            return {
                'success': False,
                'response': self._get_fallback_response(message),
                'model_used': self.model_name,
                'tokens_used': 0,
                'processing_time': processing_time,
                'timestamp': timezone.now().isoformat(),
                'error': str(e)
            }
    
    def _handle_multiple_questions(self, user_id: int, organization_id: int, original_message: str, 
                                 query_analysis: Dict, conversation_context: List, start_time: float) -> Dict[str, Any]:
        """Maneja preguntas múltiples procesando cada parte por separado."""
        try:
            # Obtener las partes de la pregunta del análisis
            if hasattr(query_analysis, 'question_parts') and query_analysis.question_parts:
                question_parts = query_analysis.question_parts
            else:
                # Fallback: dividir manualmente
                question_parts = self.query_parser._split_multiple_questions(original_message)
            
            print(f"🔍 MULTIPLE_QUESTIONS: Procesando {len(question_parts)} preguntas")
            
            responses = []
            
            for i, question in enumerate(question_parts):
                print(f"🔍 MULTIPLE_QUESTIONS: Procesando pregunta {i+1}: '{question}'")
                
                # Analizar cada pregunta individual usando el enhanced parser
                if self.enhanced_query_parser:
                    individual_analysis = self.enhanced_query_parser.parse_query(question)
                else:
                    individual_analysis = self.query_parser.parse_query(question)
                
                # Preparar contexto financiero para esta pregunta
                financial_context = self._prepare_comprehensive_financial_context(
                    user_id, organization_id, question, individual_analysis
                )
                
                # Crear prompt específico para esta pregunta
                system_prompt = self._create_enhanced_system_prompt(financial_context, individual_analysis, question)
                
                # Construir mensajes
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ]
                
                # Llamar al LLM para esta pregunta
                response = self._call_llm(messages)
                
                if response.get('success'):
                    final_response = self._force_correct_language(response['response'], question)
                    responses.append({
                        'question': question,
                        'response': final_response,
                        'analysis': individual_analysis.__dict__ if hasattr(individual_analysis, '__dict__') else individual_analysis,
                        'tokens_used': response.get('tokens_used', 0)
                    })
                else:
                    responses.append({
                        'question': question,
                        'response': self._get_fallback_response(question),
                        'analysis': individual_analysis.__dict__ if hasattr(individual_analysis, '__dict__') else individual_analysis,
                        'tokens_used': 0
                    })
            
            # Combinar todas las respuestas
            combined_response = self._combine_multiple_responses(responses)
            
            # Actualizar contexto de conversación
            conversation_context.extend([
                {"role": "user", "content": original_message},
                {"role": "assistant", "content": combined_response}
            ])
            self._save_conversation_context(user_id, organization_id, conversation_context)
            
            processing_time = time.time() - start_time
            
            # Serializar todos los campos posibles
            def safe_serialize(obj):
                if isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                elif hasattr(obj, '__dict__'):
                    return {k: safe_serialize(v) for k, v in obj.__dict__.items()}
                elif isinstance(obj, dict):
                    return {k: safe_serialize(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple, set)):
                    return [safe_serialize(i) for i in obj]
                else:
                    return str(obj)
            
            return {
                'success': True,
                'response': combined_response,
                'model_used': self.model_name,
                'tokens_used': sum(r.get('tokens_used', 0) for r in responses),
                'processing_time': processing_time,
                'timestamp': timezone.now().isoformat(),
                'query_analysis': safe_serialize(query_analysis),
                'multiple_questions': True,
                'individual_responses': safe_serialize(responses)
            }
            
        except Exception as e:
            logger.error(f"Error manejando preguntas múltiples: {str(e)}")
            processing_time = time.time() - start_time
            
            return {
                'success': False,
                'response': self._get_fallback_response(original_message),
                'model_used': self.model_name,
                'tokens_used': 0,
                'processing_time': processing_time,
                'timestamp': timezone.now().isoformat(),
                'error': str(e)
            }
    
    def _combine_multiple_responses(self, responses: List[Dict]) -> str:
        """Combina múltiples respuestas en una sola respuesta coherente y estructurada."""
        if not responses:
            return "No se pudieron procesar las preguntas."
        
        if len(responses) == 1:
            return responses[0]['response']
        
        # Crear una respuesta estructurada y coherente
        combined = []
        
        # Encabezado
        combined.append("📊 **Respuesta Completa**")
        combined.append("")
        
        # Procesar cada respuesta
        for i, resp in enumerate(responses, 1):
            question = resp['question'].strip()
            response = resp['response'].strip()
            
            # Limpiar la pregunta para mostrar
            if len(question) > 80:
                question = question[:77] + "..."
            
            # Agregar la pregunta y respuesta
            combined.append(f"**{i}. {question}**")
            
            # Si la respuesta es muy larga, truncarla inteligentemente
            if len(response) > 800:
                # Buscar el primer punto o salto de línea después de 700 caracteres
                cut_point = 700
                for j in range(700, min(len(response), 800)):
                    if response[j] in '.!?':
                        cut_point = j + 1
                        break
                response = response[:cut_point] + "..."
            
            combined.append(response)
            
            # Agregar separador si no es la última respuesta
            if i < len(responses):
                combined.append("")
        
        # Agregar resumen inteligente
        combined.append("")
        combined.append("💡 **Resumen**: He respondido a todas tus preguntas de manera detallada.")
        
        # Agregar sugerencias de seguimiento basadas en el tipo de preguntas
        question_types = []
        for resp in responses:
            analysis = resp.get('analysis', {})
            if isinstance(analysis, dict):
                intent_type = analysis.get('intent_type', 'general')
                question_types.append(intent_type)
        
        # Generar sugerencias específicas
        if 'balance_inquiry' in question_types or 'consulta_saldo' in question_types:
            combined.append("💭 **Sugerencias**: ¿Te gustaría ver un análisis de tendencias o comparar con períodos anteriores?")
        elif 'comparison' in question_types or 'comparacion' in question_types:
            combined.append("💭 **Sugerencias**: ¿Te gustaría ver recomendaciones de optimización basadas en esta comparación?")
        elif 'anomaly_detection' in question_types or 'deteccion_anomalias' in question_types:
            combined.append("💭 **Sugerencias**: ¿Te gustaría que analice patrones de comportamiento para prevenir futuras anomalías?")
        else:
            combined.append("💭 **Sugerencias**: ¿Necesitas más detalles sobre alguna de estas respuestas o tienes otras preguntas?")
        
        return "\n".join(combined)
    
    def _call_llm(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Llama al modelo LLM local."""
        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "top_k": 40,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }
            
            response = requests.post(
                f"{self.api_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'response': result.get('message', {}).get('content', ''),
                    'tokens_used': result.get('eval_count', 0)
                }
            else:
                logger.error(f"Error en llamada a LLM: {response.status_code} - {response.text}")
                return {
                    'success': False,
                    'error': f"Error HTTP {response.status_code}"
                }
                
        except requests.exceptions.Timeout:
            logger.error("Timeout en llamada a LLM")
            return {
                'success': False,
                'error': 'Timeout en la respuesta del modelo'
            }
        except Exception as e:
            logger.error(f"Error llamando al LLM: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_fallback_response(self, message: str) -> str:
        """Genera una respuesta de fallback cuando el LLM falla."""
        return """Lo siento, no puedo procesar tu consulta en este momento. 

Sin embargo, puedo ayudarte con:
- Consultas sobre ingresos, gastos y balances
- Comparaciones entre períodos
- Análisis de categorías de gasto
- Tendencias financieras
- Estadísticas de transacciones

¿Podrías reformular tu pregunta o intentar con algo más específico?"""
    
    def clear_conversation_context(self, user_id: int, organization_id: int):
        """Limpia el contexto de conversación del usuario usando ContextManager del core AI."""
        try:
            if self.context_manager:
                self.context_manager.clear_context(str(user_id), str(organization_id))
                logger.info(f"Contexto de conversación limpiado para usuario {user_id} usando ContextManager")
            else:
                # Fallback al método original
                cache_key = f"chat_context:{user_id}:{organization_id}"
                cache.delete(cache_key)
                logger.info(f"Contexto de conversación limpiado para usuario {user_id} usando cache")
        except Exception as e:
            logger.error(f"Error limpiando contexto: {str(e)}")
    
    def get_conversation_stats(self, user_id: int, organization_id: int) -> Dict[str, Any]:
        """Obtiene estadísticas de la conversación usando ContextManager del core AI."""
        try:
            if self.context_manager:
                context_history = self.context_manager.get_context_history(str(user_id), str(organization_id))
                return {
                    'message_count': len(context_history),
                    'last_message_time': context_history[-1].timestamp if context_history else None,
                    'context_size': len(str(context_history)),
                    'avg_confidence': sum(ctx.confidence_score for ctx in context_history) / len(context_history) if context_history else 0,
                    'intent_types': list(set(ctx.intent_type for ctx in context_history))
                }
            else:
                # Fallback al método original
                context = self._get_conversation_context(user_id, organization_id)
                return {
                    'message_count': len(context),
                    'last_message_time': context[-1]['timestamp'] if context else None,
                    'context_size': len(str(context))
                }
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {str(e)}")
            return {}

    def generate_report(self, user_id: int, organization_id: int, report_type: str = 'summary') -> Dict[str, Any]:
        """Genera un reporte usando ReportGenerator del core AI."""
        try:
            if self.report_generator:
                # Obtener datos financieros
                financial_data = self._get_comprehensive_financial_data(organization_id)
                
                # Crear configuración del reporte
                report_config = ReportConfig(
                    report_type=report_type,
                    language='es',
                    include_charts=True,
                    include_insights=True,
                    date_range='last_30_days'
                )
                
                # Crear datos del reporte
                report_data = ReportData(
                    financial_data=financial_data,
                    user_id=str(user_id),
                    organization_id=str(organization_id),
                    generated_at=timezone.now()
                )
                
                # Generar reporte
                report = self.report_generator.generate_report(report_data, report_config)
                
                return {
                    'success': True,
                    'report': report,
                    'report_type': report_type,
                    'generated_at': timezone.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': 'ReportGenerator no está disponible'
                }
                
        except Exception as e:
            logger.error(f"Error generando reporte: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

def get_llm_service() -> LLMService:
    """Función de fábrica para obtener una instancia del servicio LLM."""
    return LLMService() 