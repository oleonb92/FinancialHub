"""
Sistema de Análisis de Riesgo Personalizado para evaluar la salud financiera del usuario.
"""
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from django.utils import timezone
from ..models import AIInsight, AIPrediction
import logging

logger = logging.getLogger('ai.risk_analyzer')

class RiskAnalyzer:
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.risk_thresholds = {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
        
    def analyze_user_risk(self, user, transactions):
        """
        Analiza el riesgo financiero del usuario basado en sus transacciones.
        
        Args:
            user: Usuario a analizar
            transactions: Lista de transacciones del usuario
            
        Returns:
            dict: Análisis de riesgo con métricas y recomendaciones
        """
        try:
            if not transactions:
                return {
                    'risk_score': 0.0,
                    'risk_level': 'low',
                    'metrics': {
                        'expense_income_ratio': 0.0,
                        'monthly_expense_volatility': 0.0,
                        'savings_rate': 0.0,
                        'category_concentration': 0.0,
                        'expense_trend': 0.0,
                        'debt_ratio': 0.0
                    },
                    'anomalies': [],
                    'recommendations': [{
                        'type': 'data_insufficiency',
                        'priority': 'medium',
                        'message': 'No hay suficientes datos para realizar un análisis detallado. Se recomienda comenzar a registrar transacciones.'
                    }]
                }
            
            # Calcular métricas de riesgo
            risk_metrics = self._calculate_risk_metrics(transactions)
            
            # Detectar anomalías
            anomalies = self._detect_anomalies(transactions)
            
            # Calcular score de riesgo general
            risk_score = self._calculate_risk_score(risk_metrics, anomalies)
            
            # Generar recomendaciones
            recommendations = self._generate_risk_recommendations(risk_metrics, risk_score)
            
            # Guardar insights
            self._save_risk_insights(user, risk_metrics, risk_score, recommendations)
            
            return {
                'risk_score': risk_score,
                'risk_level': self._get_risk_level(risk_score),
                'metrics': risk_metrics,
                'anomalies': anomalies,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error analyzing user risk: {str(e)}", exc_info=True)
            raise
            
    def _calculate_risk_metrics(self, transactions):
        """Calcula métricas de riesgo financiero."""
        df = pd.DataFrame([{
            'amount': float(t.amount),  # Convertir Decimal a float
            'date': t.date,
            'category_id': t.category.id,
            'type': t.type
        } for t in transactions])
        
        # Calcular métricas básicas
        total_income = df[df['type'] == 'INCOME']['amount'].sum()
        total_expenses = df[df['type'] == 'EXPENSE']['amount'].sum()
        
        # Calcular métricas de riesgo
        metrics = {
            'expense_income_ratio': total_expenses / total_income if total_income > 0 else 0.0,
            'monthly_expense_volatility': self._calculate_volatility(df[df['type'] == 'EXPENSE']),
            'savings_rate': (total_income - total_expenses) / total_income if total_income > 0 else 0.0,
            'category_concentration': self._calculate_category_concentration(df),
            'expense_trend': self._calculate_expense_trend(df),
            'debt_ratio': 0.0  # Por ahora, no tenemos datos de deuda
        }
        
        return metrics
        
    def _detect_anomalies(self, transactions):
        """Detecta anomalías en las transacciones."""
        features = np.array([[
            float(t.amount),
            t.date.weekday(),
            getattr(t.date, 'hour', 0)
        ] for t in transactions])
        
        # Detectar anomalías
        anomaly_scores = self.anomaly_detector.fit_predict(features)
        
        # Filtrar anomalías y construir estructura serializable
        anomalies = []
        for t, score in zip(transactions, anomaly_scores):
            if score == -1:
                anomalies.append({
                    'transaction': t,
                    'amount': float(t.amount),
                    'date': t.date,
                    'category': str(t.category) if hasattr(t, 'category') else '',
                    'description': getattr(t, 'description', ''),
                    'anomaly_score': -1.0,  # IsolationForest: -1 indica anomalía
                    'reason': 'Anomalía detectada por IsolationForest'
                })
        return anomalies
        
    def _calculate_risk_score(self, metrics, anomalies):
        """Calcula un score de riesgo general."""
        weights = {
            'expense_income_ratio': 0.3,
            'monthly_expense_volatility': 0.2,
            'savings_rate': 0.2,
            'category_concentration': 0.15,
            'expense_trend': 0.15
        }
        
        # Normalizar métricas
        normalized_metrics = {
            'expense_income_ratio': min(metrics['expense_income_ratio'], 2) / 2,
            'monthly_expense_volatility': min(metrics['monthly_expense_volatility'], 1),
            'savings_rate': max(0, min(metrics['savings_rate'], 1)),
            'category_concentration': metrics['category_concentration'],
            'expense_trend': (metrics['expense_trend'] + 1) / 2
        }
        
        # Calcular score ponderado
        risk_score = sum(
            normalized_metrics[metric] * weight
            for metric, weight in weights.items()
        )
        
        # Ajustar por anomalías
        if anomalies:
            risk_score *= (1 + len(anomalies) * 0.1)
            
        return min(max(risk_score, 0), 1)
        
    def _generate_risk_recommendations(self, metrics, risk_score):
        """Genera recomendaciones basadas en el análisis de riesgo."""
        recommendations = []
        
        # Recomendaciones basadas en métricas
        if metrics['expense_income_ratio'] > 0.8:
            recommendations.append({
                'type': 'expense_control',
                'priority': 'high',
                'message': 'Considera reducir tus gastos para mejorar tu ratio de gastos/ingresos'
            })
            
        if metrics['savings_rate'] < 0.2:
            recommendations.append({
                'type': 'savings',
                'priority': 'high',
                'message': 'Intenta aumentar tu tasa de ahorro al menos al 20%'
            })
            
        if metrics['category_concentration'] > 0.7:
            recommendations.append({
                'type': 'diversification',
                'priority': 'medium',
                'message': 'Considera diversificar tus gastos entre más categorías'
            })
            
        # Recomendaciones basadas en el score de riesgo
        if risk_score > self.risk_thresholds['high']:
            recommendations.append({
                'type': 'risk_management',
                'priority': 'high',
                'message': 'Tu perfil de riesgo es alto. Considera revisar tus finanzas con un asesor'
            })
            
        return recommendations
        
    def _save_risk_insights(self, user, metrics, risk_score, recommendations):
        """Guarda insights de riesgo en la base de datos."""
        try:
            insight = AIInsight.objects.create(
                user=user,
                type='risk_analysis',
                data={
                    'metrics': metrics,
                    'risk_score': risk_score,
                    'recommendations': recommendations
                },
                created_at=timezone.now()
            )
            return insight
        except Exception as e:
            logger.error(f"Error saving risk insights: {str(e)}")
            return None
            
    def _get_risk_level(self, risk_score):
        """Determina el nivel de riesgo basado en el score."""
        if risk_score >= self.risk_thresholds['high']:
            return 'high'
        elif risk_score >= self.risk_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
            
    def _calculate_volatility(self, df):
        """Calcula la volatilidad mensual de gastos."""
        if df.empty:
            return 0
            
        monthly_expenses = df.groupby(df['date'].dt.to_period('M'))['amount'].sum()
        return monthly_expenses.std() / monthly_expenses.mean() if monthly_expenses.mean() > 0 else 0
        
    def _calculate_category_concentration(self, df):
        """Calcula la concentración de gastos por categoría."""
        if df.empty:
            return 0
            
        category_totals = df[df['type'] == 'EXPENSE'].groupby('category_id')['amount'].sum()
        total_expenses = category_totals.sum()
        
        if total_expenses == 0:
            return 0
            
        # Calcular índice de Herfindahl-Hirschman
        return sum((category_totals / total_expenses) ** 2)
        
    def _calculate_expense_trend(self, df):
        """Calcula la tendencia de gastos (-1 a 1)."""
        if df.empty:
            return 0
            
        expenses = df[df['type'] == 'EXPENSE'].groupby('date')['amount'].sum()
        if len(expenses) < 2:
            return 0
            
        # Calcular tendencia usando regresión lineal simple
        x = np.arange(len(expenses))
        y = expenses.values
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalizar la pendiente
        return slope / expenses.mean() if expenses.mean() > 0 else 0 