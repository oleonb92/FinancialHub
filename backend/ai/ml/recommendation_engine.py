"""
Motor de recomendaciones personalizadas para usuarios.

Este módulo analiza los patrones de gasto, preferencias de categorías y comportamiento
de ahorro de los usuarios para generar recomendaciones personalizadas.
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from django.db.models import Sum, Avg, Count
from transactions.models import Transaction, Category
from ai.ml.base import BaseMLModel
from django.utils import timezone
import logging

logger = logging.getLogger(__name__)

class RecommendationEngine(BaseMLModel):
    """
    Motor de recomendaciones que analiza el comportamiento financiero del usuario
    para generar sugerencias personalizadas.
    """
    
    def __init__(self):
        super().__init__('recommendation_engine')
        self.user_profiles = {}
        self.category_embeddings = {}
        self.scaler = StandardScaler()
        
    def build_user_profile(self, user, transactions):
        """
        Construye un perfil de usuario basado en sus transacciones.
        
        Args:
            user: Usuario a analizar
            transactions: Lista de transacciones del usuario
            
        Returns:
            dict: Perfil del usuario con patrones de gasto, preferencias y comportamiento
        """
        try:
            # Análisis de patrones de gasto
            spending_patterns = self._analyze_spending(transactions)
            
            # Análisis de preferencias por categoría
            category_preferences = self._analyze_categories(transactions)
            
            # Análisis de comportamiento de ahorro
            saving_behavior = self._analyze_saving(transactions)
            
            # Construir perfil completo
            profile = {
                'spending_patterns': spending_patterns,
                'category_preferences': category_preferences,
                'saving_behavior': saving_behavior,
                'last_updated': timezone.now()
            }
            
            self.user_profiles[user.id] = profile
            return profile
            
        except Exception as e:
            logger.error(f"Error building user profile: {str(e)}")
            return None
            
    def _analyze_spending(self, transactions):
        """
        Analiza patrones de gasto del usuario.
        
        Args:
            transactions: Lista de transacciones
            
        Returns:
            dict: Patrones de gasto identificados
        """
        if not transactions:
            return {
                'total_spent': 0,
                'average_transaction': 0,
                'spending_frequency': 0,
                'spending_trend': 'stable'
            }
            
        # Calcular métricas básicas
        total_spent = sum(t.amount for t in transactions)
        average_transaction = total_spent / len(transactions)
        
        # Analizar frecuencia de gastos
        dates = [t.date for t in transactions]
        date_range = (max(dates) - min(dates)).days
        spending_frequency = len(transactions) / max(date_range, 1)
        
        # Analizar tendencia de gastos
        recent_transactions = [t for t in transactions if t.date >= timezone.now() - timedelta(days=30)]
        recent_average = sum(t.amount for t in recent_transactions) / len(recent_transactions) if recent_transactions else 0
        
        spending_trend = 'increasing' if recent_average > average_transaction else 'decreasing' if recent_average < average_transaction else 'stable'
        
        return {
            'total_spent': total_spent,
            'average_transaction': average_transaction,
            'spending_frequency': spending_frequency,
            'spending_trend': spending_trend
        }
        
    def _analyze_categories(self, transactions):
        """
        Analiza preferencias de categorías del usuario.
        
        Args:
            transactions: Lista de transacciones
            
        Returns:
            dict: Preferencias por categoría
        """
        if not transactions:
            return {}
            
        # Agrupar transacciones por categoría
        category_totals = {}
        for transaction in transactions:
            category_id = transaction.category.id
            if category_id not in category_totals:
                category_totals[category_id] = 0
            category_totals[category_id] += transaction.amount
            
        # Calcular porcentajes
        total_spent = sum(category_totals.values())
        category_preferences = {
            category_id: (amount / total_spent) * 100
            for category_id, amount in category_totals.items()
        }
        
        return category_preferences
        
    def _analyze_saving(self, transactions):
        """
        Analiza comportamiento de ahorro del usuario.
        
        Args:
            transactions: Lista de transacciones
            
        Returns:
            dict: Métricas de ahorro
        """
        if not transactions:
            return {
                'saving_rate': 0,
                'saving_trend': 'stable',
                'score': 0
            }
            
        # Identificar ingresos y gastos
        incomes = [t.amount for t in transactions if t.amount > 0]
        expenses = [abs(t.amount) for t in transactions if t.amount < 0]
        
        total_income = sum(incomes)
        total_expenses = sum(expenses)
        
        # Calcular tasa de ahorro
        saving_rate = ((total_income - total_expenses) / total_income * 100) if total_income > 0 else 0
        
        # Analizar tendencia de ahorro
        recent_transactions = [t for t in transactions if t.date >= timezone.now() - timedelta(days=30)]
        recent_incomes = [t.amount for t in recent_transactions if t.amount > 0]
        recent_expenses = [abs(t.amount) for t in recent_transactions if t.amount < 0]
        
        recent_income = sum(recent_incomes)
        recent_expenses = sum(recent_expenses)
        recent_saving_rate = ((recent_income - recent_expenses) / recent_income * 100) if recent_income > 0 else 0
        
        saving_trend = 'improving' if recent_saving_rate > saving_rate else 'declining' if recent_saving_rate < saving_rate else 'stable'
        
        # Calcular score de ahorro (0-1)
        saving_score = min(saving_rate / 50, 1)  # 50% es el objetivo máximo
        
        return {
            'saving_rate': saving_rate,
            'saving_trend': saving_trend,
            'score': saving_score
        }
        
    def generate_recommendations(self, user_id):
        """
        Genera recomendaciones personalizadas para el usuario.
        
        Args:
            user_id: ID del usuario
            
        Returns:
            list: Lista de recomendaciones
        """
        try:
            if user_id not in self.user_profiles:
                return []
                
            profile = self.user_profiles[user_id]
            recommendations = []
            
            # Recomendaciones basadas en ahorro
            if profile['saving_behavior']['score'] < 0.5:
                recommendations.append({
                    'type': 'saving',
                    'action': 'increase_savings',
                    'confidence': 0.8,
                    'message': 'Considera aumentar tu tasa de ahorro para alcanzar tus metas financieras.'
                })
                
            # Recomendaciones basadas en patrones de gasto
            if profile['spending_patterns']['spending_trend'] == 'increasing':
                recommendations.append({
                    'type': 'spending',
                    'action': 'review_spending',
                    'confidence': 0.7,
                    'message': 'Tu gasto ha aumentado recientemente. Revisa tus gastos para mantener el control.'
                })
                
            # Recomendaciones basadas en categorías
            for category_id, percentage in profile['category_preferences'].items():
                if percentage > 30:  # Si más del 30% del gasto está en una categoría
                    category = Category.objects.get(id=category_id)
                    recommendations.append({
                        'type': 'category',
                        'action': 'diversify_spending',
                        'confidence': 0.6,
                        'message': f'Considera diversificar tus gastos. {category.name} representa el {percentage:.1f}% de tus gastos.'
                    })
                    
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []
            
    def update_profile(self, user, new_transactions):
        """
        Actualiza el perfil del usuario con nuevas transacciones.
        
        Args:
            user: Usuario a actualizar
            new_transactions: Lista de nuevas transacciones
            
        Returns:
            dict: Perfil actualizado
        """
        try:
            if user.id not in self.user_profiles:
                return self.build_user_profile(user, new_transactions)
                
            # Obtener transacciones existentes
            existing_transactions = Transaction.objects.filter(
                user=user,
                date__lt=min(t.date for t in new_transactions)
            )
            
            # Combinar transacciones
            all_transactions = list(existing_transactions) + new_transactions
            
            # Reconstruir perfil
            return self.build_user_profile(user, all_transactions)
            
        except Exception as e:
            logger.error(f"Error updating user profile: {str(e)}")
            return None
            
    def save(self):
        """Guarda el estado del motor de recomendaciones."""
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            import joblib
            joblib.dump({
                'user_profiles': self.user_profiles,
                'category_embeddings': self.category_embeddings
            }, self.model_path)
            self.logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
            
    def load(self):
        """Carga el estado del motor de recomendaciones."""
        try:
            if self.model_path.exists():
                import joblib
                data = joblib.load(self.model_path)
                self.user_profiles = data['user_profiles']
                self.category_embeddings = data['category_embeddings']
                self.logger.info(f"Model loaded from {self.model_path}")
            else:
                self.logger.warning(f"No saved model found at {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, *args, **kwargs):
        raise NotImplementedError("RecommendationEngine does not implement predict().")

    def train(self, *args, **kwargs):
        raise NotImplementedError("RecommendationEngine does not implement train().") 