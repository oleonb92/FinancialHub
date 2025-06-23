"""
Sistema de Optimización de Presupuesto para FinancialHub.

Este módulo implementa un sistema inteligente de optimización de presupuestos que:
- Analiza patrones de gasto históricos
- Predice necesidades futuras
- Genera sugerencias de optimización automática
- Optimiza la asignación de recursos basándose en objetivos financieros
"""
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from ai.ml.base import BaseMLModel
from django.db.models import Q, Sum, Avg, Count
from transactions.models import Transaction, Category, Budget
from datetime import datetime, timedelta
from django.utils import timezone
import joblib
import logging

logger = logging.getLogger('ai.ml.budget_optimizer')

class BudgetOptimizer(BaseMLModel):
    """
    Optimizador de presupuestos que analiza patrones de gasto y genera
    sugerencias de optimización automática.
    
    Características:
    - Análisis de patrones de gasto por categoría
    - Predicción de necesidades futuras
    - Optimización de asignación de recursos
    - Sugerencias de reasignación de presupuesto
    - Análisis de eficiencia presupuestaria
    """
    
    def __init__(self):
        super().__init__('budget_optimizer')
        self.scaler = StandardScaler()
        self.expense_predictor = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.efficiency_analyzer = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42
        )
        self.feature_names = [
            'month', 'day_of_week', 'category_id', 'historical_avg',
            'seasonal_factor', 'trend_factor', 'volatility'
        ]
        self.is_fitted = False
        
    def train(self, transactions: Union[List[Transaction], List[Dict]]) -> None:
        """
        Entrena el optimizador de presupuestos.
        
        Args:
            transactions: Lista de transacciones para entrenamiento
            
        Raises:
            ValueError: Si no hay transacciones suficientes
            RuntimeError: Si el entrenamiento falla
        """
        try:
            if not transactions:
                raise ValueError("No transactions provided for training")
                
            # Preparar datos de entrenamiento
            training_data = self._prepare_training_data(transactions)
            
            if training_data.empty:
                raise ValueError("No valid training data extracted")
            
            # Separar features y target
            X = training_data.drop(['amount', 'category_id'], axis=1)
            y = training_data['amount']
            
            # Escalar features
            X_scaled = self.scaler.fit_transform(X)
            
            # Entrenar predictor de gastos
            self.expense_predictor.fit(X_scaled, y)
            
            # Entrenar analizador de eficiencia
            efficiency_scores = self._calculate_efficiency_scores(transactions)
            self.efficiency_analyzer.fit(X_scaled, efficiency_scores)
            
            self.is_fitted = True
            self.is_trained = True
            
            # Guardar modelo entrenado
            self.save()
            
            self.logger.info(f"Budget optimizer trained on {len(transactions)} transactions")
            
        except Exception as e:
            self.logger.error(f"Error training budget optimizer: {str(e)}")
            raise RuntimeError(f"Failed to train budget optimizer: {str(e)}")
    
    def predict(self, data: Union[Transaction, Dict, List]) -> Dict[str, Any]:
        """
        Genera predicciones y optimizaciones de presupuesto.
        
        Args:
            data: Datos para predicción (transacción, diccionario o lista)
            
        Returns:
            dict: Resultados de optimización y predicciones
            
        Raises:
            RuntimeError: Si el modelo no está entrenado
        """
        try:
            if not self.is_fitted:
                raise RuntimeError("Model must be trained before making predictions")
                
            if isinstance(data, (Transaction, dict)):
                data = [data]
                
            # Preparar features
            features = self._prepare_features(data)
            
            if features.empty:
                return {
                    'predicted_expense': 0.0,
                    'optimization_suggestions': [],
                    'efficiency_score': 0.0,
                    'confidence': 0.0
                }
            
            # Escalar features (sin category_id)
            features_scaled = self.scaler.transform(features.drop(['category_id'], axis=1))
            
            # Predecir gastos
            predicted_expenses = self.expense_predictor.predict(features_scaled)
            
            # Calcular scores de eficiencia
            efficiency_scores = self.efficiency_analyzer.predict(features_scaled)
            
            # Generar sugerencias de optimización
            optimization_suggestions = self._generate_optimization_suggestions(
                data, predicted_expenses, efficiency_scores
            )
            
            return {
                'predicted_expense': float(predicted_expenses[0]) if len(predicted_expenses) == 1 else predicted_expenses.tolist(),
                'optimization_suggestions': optimization_suggestions,
                'efficiency_score': float(efficiency_scores[0]) if len(efficiency_scores) == 1 else efficiency_scores.tolist(),
                'confidence': self._calculate_confidence(features_scaled)
            }
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            raise RuntimeError(f"Failed to make prediction: {str(e)}")
    
    def optimize_budget_allocation(self, organization_id: int, total_budget: float, 
                                 period: str = None) -> Dict[str, Any]:
        """
        Optimiza la asignación de presupuesto entre categorías.
        
        Args:
            organization_id: ID de la organización
            total_budget: Presupuesto total disponible
            period: Período para optimización (YYYY-MM)
            
        Returns:
            dict: Asignación optimizada de presupuesto
        """
        try:
            if not self.is_fitted:
                raise RuntimeError("Model must be trained before optimization")
                
            # Obtener datos históricos de la organización
            transactions = self._get_organization_transactions(organization_id, period)
            
            if not transactions:
                return {
                    'error': 'No hay suficientes datos históricos para optimización',
                    'suggested_allocation': {}
                }
            
            # Analizar patrones por categoría
            category_analysis = self._analyze_category_patterns(transactions)
            
            # Calcular asignación optimizada
            optimized_allocation = self._calculate_optimal_allocation(
                category_analysis, total_budget
            )
            
            # Generar recomendaciones
            recommendations = self._generate_allocation_recommendations(
                category_analysis, optimized_allocation
            )
            
            return {
                'suggested_allocation': optimized_allocation,
                'category_analysis': category_analysis,
                'recommendations': recommendations,
                'total_budget': total_budget,
                'period': period or timezone.now().strftime('%Y-%m')
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing budget allocation: {str(e)}")
            return {'error': str(e)}
    
    def analyze_budget_efficiency(self, organization_id: int, period: str = None) -> Dict[str, Any]:
        """
        Analiza la eficiencia del presupuesto actual.
        
        Args:
            organization_id: ID de la organización
            period: Período para análisis (YYYY-MM)
            
        Returns:
            dict: Análisis de eficiencia presupuestaria
        """
        try:
            # Obtener presupuestos y gastos actuales
            budgets = Budget.objects.filter(
                organization_id=organization_id,
                period=period or timezone.now().strftime('%Y-%m')
            ).select_related('category')
            
            if not budgets:
                return {
                    'error': 'No hay presupuestos configurados para el período',
                    'efficiency_metrics': {}
                }
            
            efficiency_metrics = {}
            total_efficiency = 0
            total_budget = 0
            
            for budget in budgets:
                # Calcular métricas de eficiencia
                spent = budget.spent_amount
                allocated = float(budget.amount)
                efficiency = self._calculate_category_efficiency(budget, spent, allocated)
                
                efficiency_metrics[budget.category.name] = {
                    'allocated': allocated,
                    'spent': float(spent),
                    'remaining': float(budget.remaining_amount),
                    'percentage_used': float(budget.percentage_used),
                    'efficiency_score': efficiency,
                    'status': self._get_efficiency_status(efficiency)
                }
                
                total_efficiency += efficiency * allocated
                total_budget += allocated
            
            overall_efficiency = total_efficiency / total_budget if total_budget > 0 else 0
            
            return {
                'overall_efficiency': overall_efficiency,
                'category_efficiencies': efficiency_metrics,
                'total_budget': total_budget,
                'period': period or timezone.now().strftime('%Y-%m'),
                'recommendations': self._generate_efficiency_recommendations(efficiency_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing budget efficiency: {str(e)}")
            return {'error': str(e)}
    
    def _prepare_training_data(self, transactions: List[Union[Transaction, Dict]]) -> pd.DataFrame:
        """Prepara datos de entrenamiento."""
        try:
            data = []
            
            for transaction in transactions:
                if isinstance(transaction, dict):
                    amount = float(transaction.get('amount', 0))
                    date = transaction.get('date')
                    category_id = transaction.get('category_id')
                else:
                    amount = float(transaction.amount)
                    date = transaction.date
                    category_id = transaction.category.id if transaction.category else 0
                
                if date and category_id:
                    # Calcular features temporales
                    month = date.month
                    day_of_week = date.weekday()
                    
                    # Calcular promedio histórico para la categoría
                    historical_avg = self._calculate_historical_average(category_id, date)
                    
                    # Calcular factor estacional
                    seasonal_factor = self._calculate_seasonal_factor(month)
                    
                    # Calcular factor de tendencia
                    trend_factor = self._calculate_trend_factor(category_id, date)
                    
                    # Calcular volatilidad
                    volatility = self._calculate_volatility(category_id, date)
                    
                    data.append({
                        'amount': amount,
                        'category_id': category_id,
                        'month': month,
                        'day_of_week': day_of_week,
                        'historical_avg': historical_avg,
                        'seasonal_factor': seasonal_factor,
                        'trend_factor': trend_factor,
                        'volatility': volatility
                    })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            return pd.DataFrame()
    
    def _prepare_features(self, transactions: List[Union[Transaction, Dict]]) -> pd.DataFrame:
        """Prepara features para predicción."""
        try:
            data = []
            
            for transaction in transactions:
                if isinstance(transaction, dict):
                    date = transaction.get('date')
                    category_id = transaction.get('category_id')
                else:
                    date = transaction.date
                    category_id = transaction.category.id if transaction.category else 0
                
                if date and category_id:
                    month = date.month
                    day_of_week = date.weekday()
                    historical_avg = self._calculate_historical_average(category_id, date)
                    seasonal_factor = self._calculate_seasonal_factor(month)
                    trend_factor = self._calculate_trend_factor(category_id, date)
                    volatility = self._calculate_volatility(category_id, date)
                    
                    data.append({
                        'month': month,
                        'day_of_week': day_of_week,
                        'category_id': category_id,
                        'historical_avg': historical_avg,
                        'seasonal_factor': seasonal_factor,
                        'trend_factor': trend_factor,
                        'volatility': volatility
                    })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_historical_average(self, category_id: int, date: datetime) -> float:
        """Calcula el promedio histórico de gastos para una categoría."""
        try:
            # Obtener transacciones de los últimos 6 meses para la categoría
            start_date = date - timedelta(days=180)
            transactions = Transaction.objects.filter(
                category_id=category_id,
                type='EXPENSE',
                date__gte=start_date,
                date__lt=date
            )
            
            if transactions.exists():
                return float(transactions.aggregate(avg=Avg('amount'))['avg'])
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating historical average: {str(e)}")
            return 0.0
    
    def _calculate_seasonal_factor(self, month: int) -> float:
        """Calcula factor estacional basado en el mes."""
        # Factores estacionales simplificados
        seasonal_factors = {
            1: 1.1,   # Enero - post-navidad
            2: 0.9,   # Febrero - mes bajo
            3: 1.0,   # Marzo - normal
            4: 1.0,   # Abril - normal
            5: 1.1,   # Mayo - primavera
            6: 1.2,   # Junio - verano
            7: 1.3,   # Julio - vacaciones
            8: 1.2,   # Agosto - vacaciones
            9: 1.0,   # Septiembre - normal
            10: 1.1,  # Octubre - otoño
            11: 1.2,  # Noviembre - preparación navidad
            12: 1.4   # Diciembre - navidad
        }
        return seasonal_factors.get(month, 1.0)
    
    def _calculate_trend_factor(self, category_id: int, date: datetime) -> float:
        """Calcula factor de tendencia para una categoría."""
        try:
            # Comparar gastos de los últimos 3 meses vs 3 meses anteriores
            recent_start = date - timedelta(days=90)
            recent_end = date
            old_start = date - timedelta(days=180)
            old_end = date - timedelta(days=90)
            
            recent_avg = Transaction.objects.filter(
                category_id=category_id,
                type='EXPENSE',
                date__gte=recent_start,
                date__lt=recent_end
            ).aggregate(avg=Avg('amount'))['avg'] or 0
            
            old_avg = Transaction.objects.filter(
                category_id=category_id,
                type='EXPENSE',
                date__gte=old_start,
                date__lt=old_end
            ).aggregate(avg=Avg('amount'))['avg'] or 0
            
            if old_avg > 0:
                return float(recent_avg / old_avg)
            return 1.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating trend factor: {str(e)}")
            return 1.0
    
    def _calculate_volatility(self, category_id: int, date: datetime) -> float:
        """Calcula volatilidad de gastos para una categoría."""
        try:
            # Calcular desviación estándar de los últimos 6 meses
            start_date = date - timedelta(days=180)
            transactions = Transaction.objects.filter(
                category_id=category_id,
                type='EXPENSE',
                date__gte=start_date,
                date__lt=date
            ).values_list('amount', flat=True)
            
            if len(transactions) > 1:
                amounts = [float(amount) for amount in transactions]
                return float(np.std(amounts))
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating volatility: {str(e)}")
            return 0.0
    
    def _calculate_efficiency_scores(self, transactions: List[Union[Transaction, Dict]]) -> np.ndarray:
        """Calcula scores de eficiencia para entrenamiento."""
        try:
            scores = []
            
            for transaction in transactions:
                if isinstance(transaction, dict):
                    amount = float(transaction.get('amount', 0))
                    category_id = transaction.get('category_id')
                else:
                    amount = float(transaction.amount)
                    category_id = transaction.category.id if transaction.category else 0
                
                # Calcular eficiencia basada en relación con promedio histórico
                historical_avg = self._calculate_historical_average(category_id, transaction.date if hasattr(transaction, 'date') else transaction.get('date'))
                
                if historical_avg > 0:
                    efficiency = 1.0 - min(abs(amount - historical_avg) / historical_avg, 1.0)
                else:
                    efficiency = 0.5  # Valor neutral si no hay datos históricos
                
                scores.append(efficiency)
            
            return np.array(scores)
            
        except Exception as e:
            self.logger.warning(f"Error calculating efficiency scores: {str(e)}")
            return np.array([0.5] * len(transactions))
    
    def _generate_optimization_suggestions(self, transactions: List, 
                                         predicted_expenses: np.ndarray, 
                                         efficiency_scores: np.ndarray) -> List[Dict]:
        """Genera sugerencias de optimización."""
        suggestions = []
        
        try:
            for i, transaction in enumerate(transactions):
                if isinstance(transaction, dict):
                    amount = float(transaction.get('amount', 0))
                    category_id = transaction.get('category_id')
                else:
                    amount = float(transaction.amount)
                    category_id = transaction.category.id if transaction.category else 0
                
                predicted = predicted_expenses[i] if isinstance(predicted_expenses, np.ndarray) else predicted_expenses
                efficiency = efficiency_scores[i] if isinstance(efficiency_scores, np.ndarray) else efficiency_scores
                
                suggestion = {
                    'type': 'expense_optimization',
                    'category_id': category_id,
                    'current_amount': amount,
                    'predicted_amount': float(predicted),
                    'efficiency_score': float(efficiency),
                    'recommendation': self._get_optimization_recommendation(amount, predicted, efficiency)
                }
                
                suggestions.append(suggestion)
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating optimization suggestions: {str(e)}")
            return []
    
    def _get_optimization_recommendation(self, current: float, predicted: float, efficiency: float) -> str:
        """Genera recomendación específica de optimización."""
        if efficiency < 0.3:
            return "Considera reducir gastos en esta categoría - eficiencia muy baja"
        elif efficiency < 0.6:
            return "Evalúa si los gastos en esta categoría son necesarios"
        elif current > predicted * 1.2:
            return "Gasto actual excede la predicción - revisa si es necesario"
        elif current < predicted * 0.8:
            return "Gasto actual está por debajo de la predicción - puede ser eficiente"
        else:
            return "Gasto actual está dentro del rango esperado"
    
    def _get_organization_transactions(self, organization_id: int, period: str = None) -> List[Transaction]:
        """Obtiene transacciones de una organización."""
        try:
            queryset = Transaction.objects.filter(
                organization_id=organization_id,
                type='EXPENSE'
            ).select_related('category')
            
            if period:
                year, month = map(int, period.split('-'))
                queryset = queryset.filter(date__year=year, date__month=month)
            else:
                # Últimos 6 meses por defecto
                start_date = timezone.now() - timedelta(days=180)
                queryset = queryset.filter(date__gte=start_date)
            
            return list(queryset)
            
        except Exception as e:
            self.logger.error(f"Error getting organization transactions: {str(e)}")
            return []
    
    def _analyze_category_patterns(self, transactions: List[Transaction]) -> Dict[int, Dict]:
        """Analiza patrones de gasto por categoría."""
        try:
            category_analysis = {}
            
            for transaction in transactions:
                category_id = transaction.category.id if transaction.category else 0
                
                if category_id not in category_analysis:
                    category_analysis[category_id] = {
                        'total_spent': 0.0,
                        'transaction_count': 0,
                        'avg_amount': 0.0,
                        'max_amount': 0.0,
                        'min_amount': float('inf'),
                        'category_name': transaction.category.name if transaction.category else 'Unknown'
                    }
                
                amount = float(transaction.amount)
                category_analysis[category_id]['total_spent'] += amount
                category_analysis[category_id]['transaction_count'] += 1
                category_analysis[category_id]['max_amount'] = max(
                    category_analysis[category_id]['max_amount'], amount
                )
                category_analysis[category_id]['min_amount'] = min(
                    category_analysis[category_id]['min_amount'], amount
                )
            
            # Calcular promedios
            for category_data in category_analysis.values():
                if category_data['transaction_count'] > 0:
                    category_data['avg_amount'] = category_data['total_spent'] / category_data['transaction_count']
                if category_data['min_amount'] == float('inf'):
                    category_data['min_amount'] = 0.0
            
            return category_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing category patterns: {str(e)}")
            return {}
    
    def _calculate_optimal_allocation(self, category_analysis: Dict, total_budget: float) -> Dict[int, float]:
        """Calcula asignación óptima de presupuesto."""
        try:
            if not category_analysis:
                return {}
            
            # Calcular pesos basados en patrones históricos
            total_spent = sum(data['total_spent'] for data in category_analysis.values())
            
            if total_spent == 0:
                # Distribución equitativa si no hay datos históricos
                num_categories = len(category_analysis)
                equal_share = total_budget / num_categories
                return {category_id: equal_share for category_id in category_analysis.keys()}
            
            # Asignación proporcional basada en gastos históricos
            allocation = {}
            for category_id, data in category_analysis.items():
                proportion = data['total_spent'] / total_spent
                allocation[category_id] = total_budget * proportion
            
            return allocation
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal allocation: {str(e)}")
            return {}
    
    def _generate_allocation_recommendations(self, category_analysis: Dict, 
                                           optimized_allocation: Dict) -> List[Dict]:
        """Genera recomendaciones de asignación."""
        recommendations = []
        
        try:
            for category_id, data in category_analysis.items():
                if category_id in optimized_allocation:
                    current_spent = data['total_spent']
                    suggested_budget = optimized_allocation[category_id]
                    
                    if suggested_budget > current_spent * 1.2:
                        recommendations.append({
                            'type': 'increase_budget',
                            'category_id': category_id,
                            'category_name': data['category_name'],
                            'current_spent': current_spent,
                            'suggested_budget': suggested_budget,
                            'reason': 'Patrón de gasto sugiere mayor presupuesto'
                        })
                    elif suggested_budget < current_spent * 0.8:
                        recommendations.append({
                            'type': 'decrease_budget',
                            'category_id': category_id,
                            'category_name': data['category_name'],
                            'current_spent': current_spent,
                            'suggested_budget': suggested_budget,
                            'reason': 'Patrón de gasto sugiere menor presupuesto'
                        })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating allocation recommendations: {str(e)}")
            return []
    
    def _calculate_category_efficiency(self, budget: Budget, spent: float, allocated: float) -> float:
        """Calcula eficiencia de una categoría específica."""
        try:
            if allocated == 0:
                return 0.0
            
            # Eficiencia basada en qué tan cerca está el gasto del presupuesto
            utilization = spent / allocated
            
            if utilization <= 1.0:
                # Dentro del presupuesto - eficiencia alta
                return 1.0 - (utilization - 0.8) * 2  # Penalizar si está muy por debajo
            else:
                # Excede el presupuesto - eficiencia baja
                return max(0.0, 1.0 - (utilization - 1.0))
                
        except Exception as e:
            self.logger.warning(f"Error calculating category efficiency: {str(e)}")
            return 0.5
    
    def _get_efficiency_status(self, efficiency: float) -> str:
        """Obtiene estado de eficiencia."""
        if efficiency >= 0.8:
            return 'excellent'
        elif efficiency >= 0.6:
            return 'good'
        elif efficiency >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _generate_efficiency_recommendations(self, efficiency_metrics: Dict) -> List[Dict]:
        """Genera recomendaciones de eficiencia."""
        recommendations = []
        
        try:
            for category_name, metrics in efficiency_metrics.items():
                efficiency = metrics['efficiency_score']
                status = metrics['status']
                
                if status == 'poor':
                    recommendations.append({
                        'type': 'efficiency_improvement',
                        'category': category_name,
                        'priority': 'high',
                        'message': f'La eficiencia de {category_name} es baja. Considera revisar el presupuesto.'
                    })
                elif status == 'fair':
                    recommendations.append({
                        'type': 'efficiency_improvement',
                        'category': category_name,
                        'priority': 'medium',
                        'message': f'La eficiencia de {category_name} puede mejorarse.'
                    })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating efficiency recommendations: {str(e)}")
            return []
    
    def _calculate_confidence(self, features_scaled: np.ndarray) -> float:
        """Calcula nivel de confianza de las predicciones."""
        try:
            # Confianza basada en la varianza de las predicciones
            if hasattr(self.expense_predictor, 'estimators_'):
                predictions = []
                for estimator in self.expense_predictor.estimators_:
                    pred = estimator.predict(features_scaled)
                    predictions.append(pred)
                
                if len(predictions) > 1:
                    variance = np.var(predictions, axis=0)
                    confidence = 1.0 / (1.0 + variance)
                    return float(np.mean(confidence))
            
            return 0.7  # Confianza por defecto
            
        except Exception as e:
            self.logger.warning(f"Error calculating confidence: {str(e)}")
            return 0.5
    
    def save(self):
        """Guarda el modelo entrenado."""
        try:
            if not self.is_trained:
                raise RuntimeError("Cannot save untrained model")
                
            model_data = {
                'expense_predictor': self.expense_predictor,
                'efficiency_analyzer': self.efficiency_analyzer,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted
            }
            
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model_data, self.model_path)
            self.logger.info(f"Budget optimizer saved to {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving budget optimizer: {str(e)}")
            raise RuntimeError(f"Failed to save budget optimizer: {str(e)}")
    
    def load(self):
        """Carga el modelo entrenado."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"No saved model found at {self.model_path}")
                
            model_data = joblib.load(self.model_path)
            self.expense_predictor = model_data['expense_predictor']
            self.efficiency_analyzer = model_data['efficiency_analyzer']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_fitted = model_data['is_fitted']
            self.is_trained = True
            
            self.logger.info(f"Budget optimizer loaded from {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading budget optimizer: {str(e)}")
            raise RuntimeError(f"Failed to load budget optimizer: {str(e)}")
    
    def reset(self):
        """Resetea el modelo a su estado inicial."""
        self.is_fitted = False
        self.is_trained = False
        self.expense_predictor = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.efficiency_analyzer = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.logger.info("Budget optimizer reset")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtiene información del modelo."""
        info = super().get_model_info()
        info.update({
            'feature_names': self.feature_names,
            'expense_predictor_type': type(self.expense_predictor).__name__,
            'efficiency_analyzer_type': type(self.efficiency_analyzer).__name__
        })
        return info 