"""
Orquestador de IA para Sistema Financiero Completo.

Este módulo actúa como el cerebro central que coordina todos los módulos de ML
para proporcionar análisis integrales y decisiones inteligentes.
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from django.utils import timezone
from django.db.models import Q

# Importar todos los módulos de ML
from .fraud_detector import FraudDetector
from .market_predictor import MarketPredictor
from .portfolio_optimizer import PortfolioOptimizer
from .credit_scorer import CreditScorer
from .sentiment_analyzer import SentimentAnalyzer
from .risk_analyzer import RiskAnalyzer
from .recommendation_engine import RecommendationEngine
from .cash_flow_predictor import CashFlowPredictor
from .anomaly_detector import AnomalyDetector
from .nlp.text_processor import FinancialTextProcessor
from .experimentation.ab_testing import ABTesting
from .federated.federated_learning import FederatedLearning

logger = logging.getLogger('ai.orchestrator')

class AIOrchestrator:
    """
    Orquestador principal que coordina todos los módulos de IA.
    
    Funcionalidades:
    - Análisis integral de usuario
    - Decisiones inteligentes automatizadas
    - Alertas y recomendaciones unificadas
    - Dashboard de IA completo
    - Optimización de portafolio inteligente
    """
    
    def __init__(self):
        """Inicializa todos los módulos de IA."""
        self.fraud_detector = FraudDetector()
        self.market_predictor = MarketPredictor()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.credit_scorer = CreditScorer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        self.recommendation_engine = RecommendationEngine()
        self.cash_flow_predictor = CashFlowPredictor()
        self.anomaly_detector = AnomalyDetector()
        
        # Agregar módulos faltantes
        self.nlp_processor = FinancialTextProcessor()
        self.ab_testing = ABTesting()
        self.federated_learning = FederatedLearning()
        
        # Configuración del orquestador
        self.config = {
            'auto_analysis_frequency': 'daily',
            'alert_thresholds': {
                'fraud_risk': 0.7,
                'market_volatility': 0.3,
                'credit_risk': 0.6,
                'portfolio_risk': 0.4
            },
            'enable_auto_trading': False,
            'enable_auto_rebalancing': False
        }
        
        logger.info("AI Orchestrator initialized with all modules")
    
    def comprehensive_user_analysis(self, user_id: int, include_market_data: bool = True) -> Dict[str, Any]:
        """
        Realiza un análisis integral del usuario combinando todos los módulos.
        
        Args:
            user_id: ID del usuario
            include_market_data: Si incluir análisis de mercado
            
        Returns:
            dict: Análisis completo del usuario
        """
        try:
            from transactions.models import Transaction
            from accounts.models import User
            
            user = User.objects.get(id=user_id)
            transactions = list(Transaction.objects.filter(user=user).order_by('-date'))
            
            if not transactions:
                return {
                    'error': 'No transactions found for user',
                    'user_id': user_id,
                    'analysis_date': timezone.now().isoformat()
                }
            
            # Análisis de riesgo y crédito
            risk_analysis = self.risk_analyzer.analyze_user_risk(user, transactions)
            credit_features = self.credit_scorer.extract_credit_features(user_id, transactions)
            credit_analysis = self.credit_scorer.calculate_credit_score(credit_features)
            
            # Análisis de flujo de efectivo
            cash_flow_predictions = self.cash_flow_predictor.predict(transactions, days=30)
            
            # Detección de anomalías y fraude
            anomalies = self.anomaly_detector.detect_anomalies(transactions)
            fraud_analysis = self.fraud_detector.detect_fraud(transactions[0], user_id) if transactions else {}
            
            # Recomendaciones personalizadas
            recommendations = self.recommendation_engine.generate_recommendations(user_id)
            
            # Análisis de mercado (si se solicita)
            market_analysis = {}
            if include_market_data:
                market_analysis = self._get_market_analysis()
            
            # Calcular score de salud financiera
            financial_health_score = self._calculate_financial_health_score(
                risk_analysis, credit_analysis, cash_flow_predictions, anomalies
            )
            
            return {
                'user_id': user_id,
                'analysis_date': timezone.now().isoformat(),
                'financial_health_score': financial_health_score,
                'risk_analysis': risk_analysis,
                'credit_analysis': credit_analysis,
                'cash_flow_analysis': {
                    'predictions': cash_flow_predictions,
                    'summary': self._summarize_cash_flow(cash_flow_predictions)
                },
                'anomaly_analysis': {
                    'anomalies': anomalies,
                    'fraud_risk': fraud_analysis.get('combined_fraud_score', 0)
                },
                'recommendations': recommendations,
                'market_analysis': market_analysis,
                'alerts': self._generate_alerts(risk_analysis, credit_analysis, anomalies, fraud_analysis),
                'next_actions': self._suggest_next_actions(financial_health_score, recommendations)
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive user analysis: {str(e)}")
            return {
                'error': str(e),
                'user_id': user_id,
                'analysis_date': timezone.now().isoformat()
            }
    
    def intelligent_portfolio_optimization(self, user_id: int, symbols: List[str], 
                                        risk_tolerance: str = 'moderate') -> Dict[str, Any]:
        """
        Optimización inteligente de portafolio considerando múltiples factores.
        
        Args:
            user_id: ID del usuario
            symbols: Lista de símbolos de activos
            risk_tolerance: Tolerancia al riesgo ('conservative', 'moderate', 'aggressive')
            
        Returns:
            dict: Optimización de portafolio con análisis completo
        """
        try:
            # Análisis de sentimiento de mercado
            market_sentiment = self._get_market_sentiment_for_symbols(symbols)
            
            # Predicciones de mercado
            market_predictions = {}
            for symbol in symbols:
                prediction = self.market_predictor.predict_price(symbol)
                market_predictions[symbol] = prediction
            
            # Optimización de portafolio
            optimization_method = self._select_optimization_method(risk_tolerance, market_sentiment)
            portfolio_result = self.portfolio_optimizer.optimize_portfolio(
                symbols, method=optimization_method
            )
            
            # Análisis de riesgo del portafolio
            portfolio_risk = self._analyze_portfolio_risk(portfolio_result, market_predictions)
            
            # Recomendaciones de rebalanceo
            rebalancing_recommendations = self._get_rebalancing_recommendations(
                portfolio_result, market_predictions, risk_tolerance
            )
            
            return {
                'user_id': user_id,
                'optimization_date': timezone.now().isoformat(),
                'risk_tolerance': risk_tolerance,
                'market_sentiment': market_sentiment,
                'market_predictions': market_predictions,
                'portfolio_optimization': portfolio_result,
                'portfolio_risk_analysis': portfolio_risk,
                'rebalancing_recommendations': rebalancing_recommendations,
                'optimization_method_used': optimization_method,
                'confidence_score': self._calculate_optimization_confidence(
                    portfolio_result, market_predictions, market_sentiment
                )
            }
            
        except Exception as e:
            logger.error(f"Error in intelligent portfolio optimization: {str(e)}")
            return {
                'error': str(e),
                'user_id': user_id,
                'optimization_date': timezone.now().isoformat()
            }
    
    def smart_loan_assessment(self, user_id: int, loan_amount: float, 
                            loan_term: int, loan_type: str = 'personal') -> Dict[str, Any]:
        """
        Evaluación inteligente de préstamos combinando múltiples análisis.
        
        Args:
            user_id: ID del usuario
            loan_amount: Monto del préstamo
            loan_term: Plazo del préstamo
            loan_type: Tipo de préstamo
            
        Returns:
            dict: Evaluación completa del préstamo
        """
        try:
            # Evaluación de crédito
            loan_eligibility = self.credit_scorer.assess_loan_eligibility(
                user_id, loan_amount, loan_term, loan_type
            )
            
            # Análisis de riesgo
            from transactions.models import Transaction
            from accounts.models import User
            user = User.objects.get(id=user_id)
            transactions = list(Transaction.objects.filter(user=user).order_by('-date'))
            risk_analysis = self.risk_analyzer.analyze_user_risk(user, transactions)
            
            # Predicción de flujo de efectivo
            cash_flow_predictions = self.cash_flow_predictor.predict(transactions, days=loan_term)
            
            # Análisis de capacidad de pago
            payment_capacity = self._analyze_payment_capacity(
                cash_flow_predictions, loan_amount, loan_term
            )
            
            # Score de aprobación inteligente
            approval_score = self._calculate_loan_approval_score(
                loan_eligibility, risk_analysis, payment_capacity
            )
            
            # Recomendaciones de términos
            term_recommendations = self._suggest_loan_terms(
                approval_score, loan_eligibility, risk_analysis
            )
            
            return {
                'user_id': user_id,
                'loan_amount': loan_amount,
                'loan_term': loan_term,
                'loan_type': loan_type,
                'assessment_date': timezone.now().isoformat(),
                'approval_score': approval_score,
                'loan_eligibility': loan_eligibility,
                'risk_analysis': risk_analysis,
                'payment_capacity': payment_capacity,
                'cash_flow_impact': self._analyze_cash_flow_impact(
                    cash_flow_predictions, loan_amount, loan_term
                ),
                'term_recommendations': term_recommendations,
                'final_recommendation': self._get_final_loan_recommendation(approval_score)
            }
            
        except Exception as e:
            logger.error(f"Error in smart loan assessment: {str(e)}")
            return {
                'error': str(e),
                'user_id': user_id,
                'assessment_date': timezone.now().isoformat()
            }
    
    def generate_ai_insights_dashboard(self, user_id: int) -> Dict[str, Any]:
        """
        Genera un dashboard completo de insights de IA.
        
        Args:
            user_id: ID del usuario
            
        Returns:
            dict: Dashboard completo de insights
        """
        try:
            # Análisis integral
            comprehensive_analysis = self.comprehensive_user_analysis(user_id)
            
            # Métricas clave
            key_metrics = self._extract_key_metrics(comprehensive_analysis)
            
            # Tendencias
            trends = self._analyze_trends(user_id)
            
            # Alertas y notificaciones
            alerts = comprehensive_analysis.get('alerts', [])
            
            # Recomendaciones prioritarias
            priority_recommendations = self._prioritize_recommendations(
                comprehensive_analysis.get('recommendations', [])
            )
            
            # Predicciones futuras
            future_predictions = self._generate_future_predictions(user_id)
            
            return {
                'user_id': user_id,
                'dashboard_date': timezone.now().isoformat(),
                'key_metrics': key_metrics,
                'trends': trends,
                'alerts': alerts,
                'priority_recommendations': priority_recommendations,
                'future_predictions': future_predictions,
                'financial_health_score': comprehensive_analysis.get('financial_health_score', 0),
                'risk_level': comprehensive_analysis.get('risk_analysis', {}).get('risk_level', 'unknown'),
                'credit_score': comprehensive_analysis.get('credit_analysis', {}).get('credit_score', 0),
                'summary': self._generate_dashboard_summary(key_metrics, trends, alerts)
            }
            
        except Exception as e:
            logger.error(f"Error generating AI insights dashboard: {str(e)}")
            return {
                'error': str(e),
                'user_id': user_id,
                'dashboard_date': timezone.now().isoformat()
            }
    
    def _calculate_financial_health_score(self, risk_analysis: Dict, credit_analysis: Dict, 
                                        cash_flow_predictions: List, anomalies: List) -> float:
        """Calcula un score de salud financiera general."""
        try:
            # Componentes del score
            risk_score = 1 - risk_analysis.get('risk_score', 0.5)
            credit_score = credit_analysis.get('credit_score', 300) / 850
            cash_flow_score = self._calculate_cash_flow_score(cash_flow_predictions)
            anomaly_score = 1 - (len(anomalies) * 0.1)  # Reducir por anomalías
            
            # Ponderación
            weights = {
                'risk': 0.3,
                'credit': 0.3,
                'cash_flow': 0.25,
                'anomalies': 0.15
            }
            
            final_score = (
                risk_score * weights['risk'] +
                credit_score * weights['credit'] +
                cash_flow_score * weights['cash_flow'] +
                anomaly_score * weights['anomalies']
            )
            
            return min(max(final_score, 0), 1)
            
        except Exception as e:
            logger.error(f"Error calculating financial health score: {str(e)}")
            return 0.5
    
    def _get_market_analysis(self) -> Dict[str, Any]:
        """Obtiene análisis de mercado general."""
        try:
            # Análisis de sentimiento de mercado general
            market_sentiment = {
                'overall_sentiment': 'neutral',
                'confidence': 0.5,
                'trend': 'stable'
            }
            
            return {
                'market_sentiment': market_sentiment,
                'volatility_index': 0.2,
                'market_trend': 'sideways',
                'analysis_date': timezone.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting market analysis: {str(e)}")
            return {'error': str(e)}
    
    def _generate_alerts(self, risk_analysis: Dict, credit_analysis: Dict, 
                        anomalies: List, fraud_analysis: Dict) -> List[Dict[str, Any]]:
        """Genera alertas basadas en múltiples análisis."""
        alerts = []
        
        try:
            # Alertas de riesgo
            if risk_analysis.get('risk_level') == 'high':
                alerts.append({
                    'type': 'risk_alert',
                    'severity': 'high',
                    'message': 'Nivel de riesgo financiero alto detectado',
                    'recommendation': 'Revisa tus gastos y considera consultar con un asesor'
                })
            
            # Alertas de crédito
            if credit_analysis.get('default_probability', 0) > 0.7:
                alerts.append({
                    'type': 'credit_alert',
                    'severity': 'high',
                    'message': 'Alto riesgo de default detectado',
                    'recommendation': 'Revisa tu historial de pagos y mejora tu score crediticio'
                })
            
            # Alertas de anomalías
            if len(anomalies) > 3:
                alerts.append({
                    'type': 'anomaly_alert',
                    'severity': 'medium',
                    'message': f'{len(anomalies)} transacciones anómalas detectadas',
                    'recommendation': 'Revisa las transacciones marcadas como anómalas'
                })
            
            # Alertas de fraude
            if fraud_analysis.get('combined_fraud_score', 0) > 0.7:
                alerts.append({
                    'type': 'fraud_alert',
                    'severity': 'high',
                    'message': 'Posible actividad fraudulenta detectada',
                    'recommendation': 'Contacta inmediatamente con soporte'
                })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating alerts: {str(e)}")
            return []
    
    def _suggest_next_actions(self, health_score: float, recommendations: List) -> List[Dict[str, Any]]:
        """Sugiere próximas acciones basadas en el análisis."""
        actions = []
        
        try:
            if health_score < 0.3:
                actions.append({
                    'priority': 'high',
                    'action': 'review_finances',
                    'description': 'Revisa completamente tus finanzas',
                    'estimated_time': '2 hours'
                })
            
            if health_score < 0.5:
                actions.append({
                    'priority': 'medium',
                    'action': 'create_budget',
                    'description': 'Crea un presupuesto detallado',
                    'estimated_time': '1 hour'
                })
            
            # Agregar acciones basadas en recomendaciones
            for rec in recommendations[:3]:  # Top 3 recomendaciones
                actions.append({
                    'priority': 'medium',
                    'action': rec.get('action', 'review'),
                    'description': rec.get('message', 'Revisar recomendación'),
                    'estimated_time': '30 minutes'
                })
            
            return actions
            
        except Exception as e:
            logger.error(f"Error suggesting next actions: {str(e)}")
            return []
    
    def _summarize_cash_flow(self, predictions: List) -> Dict[str, Any]:
        """Resume las predicciones de flujo de efectivo."""
        try:
            if not predictions:
                return {'trend': 'stable', 'confidence': 0.0}
            
            amounts = [p.get('predicted_amount', 0) for p in predictions]
            avg_amount = sum(amounts) / len(amounts)
            
            # Determinar tendencia
            if len(amounts) > 1:
                trend = 'increasing' if amounts[-1] > amounts[0] else 'decreasing' if amounts[-1] < amounts[0] else 'stable'
            else:
                trend = 'stable'
            
            return {
                'trend': trend,
                'average_predicted_amount': avg_amount,
                'confidence': sum(p.get('confidence', 0) for p in predictions) / len(predictions)
            }
            
        except Exception as e:
            logger.error(f"Error summarizing cash flow: {str(e)}")
            return {'trend': 'stable', 'confidence': 0.0}
    
    def _get_market_sentiment_for_symbols(self, symbols: List[str]) -> Dict[str, Any]:
        """Obtiene sentimiento de mercado para símbolos específicos."""
        try:
            sentiment_data = {}
            for symbol in symbols:
                sentiment = self.market_predictor.analyze_market_sentiment(symbol)
                sentiment_data[symbol] = sentiment
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error getting market sentiment: {str(e)}")
            return {}
    
    def _select_optimization_method(self, risk_tolerance: str, market_sentiment: Dict) -> str:
        """Selecciona el método de optimización basado en tolerancia al riesgo y sentimiento."""
        if risk_tolerance == 'conservative':
            return 'risk_parity'
        elif risk_tolerance == 'aggressive':
            return 'markowitz'
        else:
            return 'black_litterman'
    
    def _analyze_portfolio_risk(self, portfolio_result: Dict, market_predictions: Dict) -> Dict[str, Any]:
        """Analiza el riesgo del portafolio optimizado."""
        try:
            if not portfolio_result.get('success'):
                return {'error': 'Portfolio optimization failed'}
            
            metrics = portfolio_result.get('metrics', {})
            
            return {
                'volatility': metrics.get('volatility', 0),
                'var_95': metrics.get('var_95', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'risk_level': self._determine_portfolio_risk_level(metrics)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio risk: {str(e)}")
            return {'error': str(e)}
    
    def _determine_portfolio_risk_level(self, metrics: Dict) -> str:
        """Determina el nivel de riesgo del portafolio."""
        volatility = metrics.get('volatility', 0)
        
        if volatility < 0.1:
            return 'low'
        elif volatility < 0.2:
            return 'medium'
        else:
            return 'high'
    
    def _get_rebalancing_recommendations(self, portfolio_result: Dict, 
                                       market_predictions: Dict, risk_tolerance: str) -> List[Dict[str, Any]]:
        """Genera recomendaciones de rebalanceo."""
        recommendations = []
        
        try:
            if not portfolio_result.get('success'):
                return recommendations
            
            # Análisis básico de rebalanceo
            weights = portfolio_result.get('weights', [])
            assets = portfolio_result.get('assets', [])
            
            for i, (weight, asset) in enumerate(zip(weights, assets)):
                if weight > 0.3:  # Más del 30% en un activo
                    recommendations.append({
                        'type': 'concentration_risk',
                        'asset': asset,
                        'current_weight': weight,
                        'recommended_weight': 0.25,
                        'reason': 'Concentración excesiva en un solo activo'
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting rebalancing recommendations: {str(e)}")
            return []
    
    def _calculate_optimization_confidence(self, portfolio_result: Dict, 
                                         market_predictions: Dict, market_sentiment: Dict) -> float:
        """Calcula la confianza en la optimización."""
        try:
            if not portfolio_result.get('success'):
                return 0.0
            
            # Factores de confianza
            portfolio_confidence = 0.7  # Base confidence
            market_confidence = 0.6    # Market prediction confidence
            
            # Ajustar por sentimiento de mercado
            sentiment_scores = [s.get('sentiment_score', 0.5) for s in market_sentiment.values()]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5
            
            final_confidence = (portfolio_confidence + market_confidence + avg_sentiment) / 3
            return min(max(final_confidence, 0), 1)
            
        except Exception as e:
            logger.error(f"Error calculating optimization confidence: {str(e)}")
            return 0.5
    
    def _analyze_payment_capacity(self, cash_flow_predictions: List, 
                                loan_amount: float, loan_term: int) -> Dict[str, Any]:
        """Analiza la capacidad de pago basada en flujo de efectivo."""
        try:
            if not cash_flow_predictions:
                return {'capacity_score': 0.0, 'risk_level': 'high'}
            
            # Calcular pago mensual estimado (simplificado)
            monthly_payment = loan_amount / loan_term
            
            # Analizar capacidad de pago
            positive_cash_flows = [p.get('predicted_amount', 0) for p in cash_flow_predictions if p.get('predicted_amount', 0) > 0]
            
            if not positive_cash_flows:
                return {'capacity_score': 0.0, 'risk_level': 'high'}
            
            avg_positive_cash_flow = sum(positive_cash_flows) / len(positive_cash_flows)
            capacity_ratio = avg_positive_cash_flow / monthly_payment if monthly_payment > 0 else 0
            
            # Determinar nivel de riesgo
            if capacity_ratio > 3:
                risk_level = 'low'
                capacity_score = 1.0
            elif capacity_ratio > 2:
                risk_level = 'medium'
                capacity_score = 0.7
            elif capacity_ratio > 1.5:
                risk_level = 'medium_high'
                capacity_score = 0.5
            else:
                risk_level = 'high'
                capacity_score = 0.2
            
            return {
                'capacity_score': capacity_score,
                'risk_level': risk_level,
                'capacity_ratio': capacity_ratio,
                'monthly_payment': monthly_payment,
                'avg_positive_cash_flow': avg_positive_cash_flow
            }
            
        except Exception as e:
            logger.error(f"Error analyzing payment capacity: {str(e)}")
            return {'capacity_score': 0.0, 'risk_level': 'high'}
    
    def _calculate_loan_approval_score(self, loan_eligibility: Dict, risk_analysis: Dict, 
                                     payment_capacity: Dict) -> float:
        """Calcula un score de aprobación de préstamo."""
        try:
            # Factores de aprobación
            eligibility_score = 1.0 if loan_eligibility.get('is_eligible', False) else 0.0
            risk_score = 1 - risk_analysis.get('risk_score', 0.5)
            capacity_score = payment_capacity.get('capacity_score', 0.0)
            credit_score = loan_eligibility.get('credit_score', 300) / 850
            
            # Ponderación
            weights = {
                'eligibility': 0.3,
                'risk': 0.25,
                'capacity': 0.25,
                'credit': 0.2
            }
            
            approval_score = (
                eligibility_score * weights['eligibility'] +
                risk_score * weights['risk'] +
                capacity_score * weights['capacity'] +
                credit_score * weights['credit']
            )
            
            return min(max(approval_score, 0), 1)
            
        except Exception as e:
            logger.error(f"Error calculating loan approval score: {str(e)}")
            return 0.0
    
    def _suggest_loan_terms(self, approval_score: float, loan_eligibility: Dict, 
                           risk_analysis: Dict) -> Dict[str, Any]:
        """Sugiere términos de préstamo basados en el análisis."""
        try:
            if approval_score > 0.8:
                terms = {
                    'recommended_rate': 'low',
                    'max_amount_multiplier': 3.0,
                    'term_flexibility': 'high',
                    'collateral_required': False
                }
            elif approval_score > 0.6:
                terms = {
                    'recommended_rate': 'standard',
                    'max_amount_multiplier': 2.0,
                    'term_flexibility': 'medium',
                    'collateral_required': False
                }
            elif approval_score > 0.4:
                terms = {
                    'recommended_rate': 'high',
                    'max_amount_multiplier': 1.5,
                    'term_flexibility': 'low',
                    'collateral_required': True
                }
            else:
                terms = {
                    'recommended_rate': 'very_high',
                    'max_amount_multiplier': 1.0,
                    'term_flexibility': 'very_low',
                    'collateral_required': True
                }
            
            return terms
            
        except Exception as e:
            logger.error(f"Error suggesting loan terms: {str(e)}")
            return {}
    
    def _analyze_cash_flow_impact(self, cash_flow_predictions: List, 
                                loan_amount: float, loan_term: int) -> Dict[str, Any]:
        """Analiza el impacto del préstamo en el flujo de efectivo."""
        try:
            if not cash_flow_predictions:
                return {'impact_level': 'unknown', 'risk_level': 'high'}
            
            monthly_payment = loan_amount / loan_term
            positive_flows = [p.get('predicted_amount', 0) for p in cash_flow_predictions if p.get('predicted_amount', 0) > 0]
            
            if not positive_flows:
                return {'impact_level': 'high', 'risk_level': 'high'}
            
            avg_positive_flow = sum(positive_flows) / len(positive_flows)
            impact_ratio = monthly_payment / avg_positive_flow
            
            if impact_ratio < 0.2:
                impact_level = 'low'
                risk_level = 'low'
            elif impact_ratio < 0.4:
                impact_level = 'medium'
                risk_level = 'medium'
            else:
                impact_level = 'high'
                risk_level = 'high'
            
            return {
                'impact_level': impact_level,
                'risk_level': risk_level,
                'impact_ratio': impact_ratio,
                'monthly_payment': monthly_payment,
                'avg_positive_flow': avg_positive_flow
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cash flow impact: {str(e)}")
            return {'impact_level': 'unknown', 'risk_level': 'high'}
    
    def _get_final_loan_recommendation(self, approval_score: float) -> Dict[str, Any]:
        """Obtiene la recomendación final del préstamo."""
        if approval_score > 0.8:
            return {
                'recommendation': 'approve',
                'confidence': 'high',
                'message': 'Préstamo recomendado con términos favorables'
            }
        elif approval_score > 0.6:
            return {
                'recommendation': 'approve_with_conditions',
                'confidence': 'medium',
                'message': 'Préstamo aprobado con términos estándar'
            }
        elif approval_score > 0.4:
            return {
                'recommendation': 'approve_with_restrictions',
                'confidence': 'low',
                'message': 'Préstamo aprobado con restricciones y términos más altos'
            }
        else:
            return {
                'recommendation': 'reject',
                'confidence': 'high',
                'message': 'Préstamo no recomendado debido al alto riesgo'
            }
    
    def _extract_key_metrics(self, analysis: Dict) -> Dict[str, Any]:
        """Extrae métricas clave del análisis."""
        try:
            return {
                'financial_health_score': analysis.get('financial_health_score', 0),
                'risk_level': analysis.get('risk_analysis', {}).get('risk_level', 'unknown'),
                'credit_score': analysis.get('credit_analysis', {}).get('credit_score', 0),
                'fraud_risk': analysis.get('anomaly_analysis', {}).get('fraud_risk', 0),
                'anomaly_count': len(analysis.get('anomaly_analysis', {}).get('anomalies', [])),
                'recommendation_count': len(analysis.get('recommendations', []))
            }
        except Exception as e:
            logger.error(f"Error extracting key metrics: {str(e)}")
            return {}
    
    def _analyze_trends(self, user_id: int) -> Dict[str, Any]:
        """Analiza tendencias del usuario."""
        try:
            # Análisis básico de tendencias
            return {
                'spending_trend': 'stable',
                'savings_trend': 'increasing',
                'risk_trend': 'decreasing',
                'confidence': 0.7
            }
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
            return {}
    
    def _prioritize_recommendations(self, recommendations: List) -> List[Dict[str, Any]]:
        """Prioriza las recomendaciones."""
        try:
            # Ordenar por prioridad (simplificado)
            priority_map = {
                'high': 3,
                'medium': 2,
                'low': 1
            }
            
            sorted_recommendations = sorted(
                recommendations,
                key=lambda x: priority_map.get(x.get('priority', 'medium'), 1),
                reverse=True
            )
            
            return sorted_recommendations[:5]  # Top 5 recomendaciones
            
        except Exception as e:
            logger.error(f"Error prioritizing recommendations: {str(e)}")
            return recommendations[:5]
    
    def _generate_future_predictions(self, user_id: int) -> Dict[str, Any]:
        """Genera predicciones futuras para el usuario."""
        try:
            return {
                'cash_flow_forecast': 'positive',
                'risk_projection': 'decreasing',
                'savings_projection': 'increasing',
                'confidence': 0.6,
                'timeframe': '3_months'
            }
        except Exception as e:
            logger.error(f"Error generating future predictions: {str(e)}")
            return {}
    
    def _generate_dashboard_summary(self, key_metrics: Dict, trends: Dict, alerts: List) -> str:
        """Genera un resumen del dashboard."""
        try:
            health_score = key_metrics.get('financial_health_score', 0)
            risk_level = key_metrics.get('risk_level', 'unknown')
            alert_count = len(alerts)
            
            if health_score > 0.7 and risk_level == 'low' and alert_count == 0:
                return "Tu salud financiera es excelente. Continúa con tus buenas prácticas."
            elif health_score > 0.5 and alert_count < 2:
                return "Tu salud financiera es buena. Algunas mejoras menores podrían optimizar tu situación."
            else:
                return "Tu salud financiera necesita atención. Revisa las alertas y recomendaciones para mejorar."
                
        except Exception as e:
            logger.error(f"Error generating dashboard summary: {str(e)}")
            return "Análisis disponible. Revisa las métricas detalladas."
    
    def _calculate_cash_flow_score(self, predictions: List) -> float:
        """Calcula un score basado en las predicciones de flujo de efectivo."""
        try:
            if not predictions:
                return 0.5
            
            positive_predictions = sum(1 for p in predictions if p.get('predicted_amount', 0) > 0)
            total_predictions = len(predictions)
            
            return positive_predictions / total_predictions if total_predictions > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating cash flow score: {str(e)}")
            return 0.5

    def analyze_text(self, text: str, method: str = 'vader'):
        return self.nlp_processor.analyze_sentiment(text, method=method)

    def extract_entities(self, text: str):
        return self.nlp_processor.extract_financial_entities(text) 