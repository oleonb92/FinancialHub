#!/usr/bin/env python3
"""
Script completo para entrenar todos los modelos de IA de FinancialHub
Incluye generación de datos sintéticos, entrenamiento y evaluación
"""

import os
import sys
import django
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Configurar Django
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'financialhub.settings')
django.setup()

from django.utils import timezone
from django.contrib.auth import get_user_model
from transactions.models import Transaction, Category, Tag
from organizations.models import Organization
from ai.services import AIService
from ai.ml.classifiers.transaction import TransactionClassifier
from ai.ml.predictors.expense import ExpensePredictor
from ai.ml.analyzers.behavior import BehaviorAnalyzer
from ai.ml.optimizers.budget_optimizer import BudgetOptimizer
from ai.ml.anomaly_detector import AnomalyDetector
from ai.ml.cash_flow_predictor import CashFlowPredictor
from ai.ml.risk_analyzer import RiskAnalyzer
from ai.ml.recommendation_engine import RecommendationEngine

User = get_user_model()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Clase para entrenar todos los modelos de IA"""
    
    def __init__(self):
        self.ai_service = AIService()
        self.organization = None
        self.user = None
        self.categories = []
        self.tags = []
        
    def setup_test_data(self):
        """Configurar datos de prueba"""
        logger.info("Configurando datos de prueba...")
        
        # Crear organización si no existe
        self.organization, created = Organization.objects.get_or_create(
            name="AI Training Organization",
            defaults={'plan': 'pro'}
        )
        
        # Crear usuario si no existe
        self.user, created = User.objects.get_or_create(
            username='ai_trainer',
            defaults={
                'email': 'ai_trainer@example.com',
                'first_name': 'AI',
                'last_name': 'Trainer',
                'organization': self.organization,
                'role': 'admin',
                'was_approved': True,
                'is_active': True
            }
        )
        
        # Crear categorías
        category_data = [
            'Food & Dining', 'Transportation', 'Shopping', 'Entertainment',
            'Utilities', 'Healthcare', 'Education', 'Travel', 'Insurance',
            'Investments', 'Salary', 'Freelance', 'Business', 'Gifts',
            'Home & Garden', 'Technology', 'Fitness', 'Pets', 'Charity'
        ]
        
        for cat_name in category_data:
            category, created = Category.objects.get_or_create(
                name=cat_name,
                organization=self.organization
            )
            self.categories.append(category)
        
        # Crear tags (sin campo organization)
        tag_data = [
            'essential', 'luxury', 'recurring', 'one-time', 'emergency',
            'planned', 'impulse', 'business', 'personal', 'tax-deductible',
            'high-priority', 'low-priority', 'seasonal', 'annual'
        ]
        
        for tag_name in tag_data:
            tag, created = Tag.objects.get_or_create(
                name=tag_name
            )
            self.tags.append(tag)
        
        logger.info(f"Configurados {len(self.categories)} categorías y {len(self.tags)} tags")
    
    def generate_synthetic_transactions(self, num_transactions=1000):
        """Generar transacciones sintéticas para entrenamiento"""
        logger.info(f"Generando {num_transactions} transacciones sintéticas...")
        
        # Patrones de transacciones
        transaction_patterns = [
            # Gastos regulares
            {'type': 'expense', 'amount_range': (50, 200), 'frequency': 0.25, 'categories': ['Food & Dining', 'Transportation', 'Utilities']},
            {'type': 'expense', 'amount_range': (200, 500), 'frequency': 0.20, 'categories': ['Shopping', 'Entertainment', 'Healthcare']},
            {'type': 'expense', 'amount_range': (500, 1000), 'frequency': 0.10, 'categories': ['Travel', 'Technology', 'Home & Garden']},
            {'type': 'expense', 'amount_range': (1000, 5000), 'frequency': 0.05, 'categories': ['Insurance', 'Education', 'Investments']},
            
            # Ingresos
            {'type': 'income', 'amount_range': (2000, 8000), 'frequency': 0.20, 'categories': ['Salary', 'Freelance', 'Business']},
            {'type': 'income', 'amount_range': (500, 2000), 'frequency': 0.20, 'categories': ['Investments', 'Gifts', 'Business']},
        ]
        
        # Descriptions por categoría
        descriptions = {
            'Food & Dining': ['Grocery shopping', 'Restaurant dinner', 'Coffee shop', 'Fast food', 'Food delivery'],
            'Transportation': ['Gas station', 'Uber ride', 'Public transport', 'Car maintenance', 'Parking'],
            'Shopping': ['Clothing store', 'Electronics', 'Home goods', 'Books', 'Sporting goods'],
            'Entertainment': ['Movie tickets', 'Concert tickets', 'Gaming', 'Streaming services', 'Hobbies'],
            'Utilities': ['Electric bill', 'Water bill', 'Internet', 'Phone bill', 'Gas bill'],
            'Healthcare': ['Doctor visit', 'Pharmacy', 'Dental care', 'Vision care', 'Medical supplies'],
            'Education': ['Tuition', 'Books', 'Online course', 'Workshop', 'Certification'],
            'Travel': ['Airline tickets', 'Hotel', 'Rental car', 'Vacation', 'Business trip'],
            'Insurance': ['Car insurance', 'Health insurance', 'Life insurance', 'Home insurance'],
            'Investments': ['Stock purchase', 'Mutual fund', 'Retirement contribution', 'Crypto'],
            'Salary': ['Monthly salary', 'Paycheck', 'Bonus', 'Commission'],
            'Freelance': ['Freelance work', 'Consulting', 'Project payment', 'Side hustle'],
            'Business': ['Business expense', 'Office supplies', 'Marketing', 'Client meeting'],
            'Gifts': ['Birthday gift', 'Wedding gift', 'Holiday gift', 'Charity donation'],
            'Home & Garden': ['Home improvement', 'Furniture', 'Garden supplies', 'Repairs'],
            'Technology': ['Software license', 'Hardware', 'Cloud services', 'Tech support'],
            'Fitness': ['Gym membership', 'Personal trainer', 'Fitness equipment', 'Sports'],
            'Pets': ['Pet food', 'Veterinary care', 'Pet supplies', 'Grooming'],
            'Charity': ['Donation', 'Fundraiser', 'Volunteer expense', 'Community support']
        }
        
        transactions_created = 0
        
        for i in range(num_transactions):
            # Seleccionar patrón
            pattern = np.random.choice(transaction_patterns, p=[p['frequency'] for p in transaction_patterns])
            
            # Generar monto
            min_amount, max_amount = pattern['amount_range']
            amount = np.random.uniform(min_amount, max_amount)
            
            # Seleccionar categoría
            category_name = np.random.choice(pattern['categories'])
            category = next((c for c in self.categories if c.name == category_name), self.categories[0])
            
            # Generar descripción
            category_descriptions = descriptions.get(category_name, ['Transaction'])
            description = np.random.choice(category_descriptions)
            
            # Generar fecha (últimos 2 años)
            days_ago = np.random.randint(0, 730)
            date = timezone.now() - timedelta(days=days_ago)
            
            # Crear transacción
            transaction = Transaction.objects.create(
                description=description,
                amount=round(amount, 2),
                date=date,
                category=category,
                type=pattern['type'],
                status='completed',
                created_by=self.user,
                organization=self.organization
            )
            
            # Agregar tags aleatorios
            num_tags = np.random.randint(0, 3)
            selected_tags = np.random.choice(self.tags, num_tags, replace=False)
            transaction.tags.set(selected_tags)
            
            transactions_created += 1
            
            if transactions_created % 100 == 0:
                logger.info(f"Creadas {transactions_created} transacciones...")
        
        logger.info(f"✅ Generadas {transactions_created} transacciones sintéticas")
        return transactions_created
    
    def train_transaction_classifier(self):
        """Entrenar clasificador de transacciones"""
        logger.info("Entrenando clasificador de transacciones...")
        
        try:
            # Obtener transacciones para entrenamiento
            transactions = Transaction.objects.filter(
                organization=self.organization,
                category__isnull=False
            ).select_related('category')
            
            if transactions.count() < 100:
                logger.warning("Pocas transacciones para entrenar el clasificador")
                return False
            
            # Entrenar modelo (solo lista de transacciones)
            self.ai_service.transaction_classifier.train(list(transactions))
            
            logger.info("✅ Clasificador de transacciones entrenado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error entrenando clasificador de transacciones: {e}")
            return False
    
    def train_expense_predictor(self):
        """Entrenar predictor de gastos"""
        logger.info("Entrenando predictor de gastos...")
        
        try:
            # Obtener transacciones de gastos
            expenses = Transaction.objects.filter(
                organization=self.organization,
                type='expense',
                category__isnull=False
            ).select_related('category')
            
            if expenses.count() < 50:
                logger.warning("Pocos gastos para entrenar el predictor")
                return False
            
            # Entrenar modelo (solo lista de transacciones)
            self.ai_service.expense_predictor.train(list(expenses))
            
            logger.info("✅ Predictor de gastos entrenado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error entrenando predictor de gastos: {e}")
            return False
    
    def train_behavior_analyzer(self):
        """Entrenar analizador de comportamiento"""
        logger.info("Entrenando analizador de comportamiento...")
        
        try:
            # Obtener todas las transacciones del usuario
            transactions = Transaction.objects.filter(
                created_by=self.user
            ).select_related('category')
            
            if transactions.count() < 100:
                logger.warning("Pocas transacciones para entrenar el analizador de comportamiento")
                return False
            
            # Entrenar modelo
            self.ai_service.behavior_analyzer.train(transactions)
            
            logger.info("✅ Analizador de comportamiento entrenado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error entrenando analizador de comportamiento: {e}")
            return False
    
    def train_budget_optimizer(self):
        """Entrenar optimizador de presupuesto"""
        logger.info("Entrenando optimizador de presupuesto...")
        
        try:
            # Obtener datos históricos
            transactions = Transaction.objects.filter(
                organization=self.organization
            ).select_related('category')
            
            if transactions.count() < 200:
                logger.warning("Pocas transacciones para entrenar el optimizador de presupuesto")
                return False
            
            # Entrenar modelo
            self.ai_service.budget_optimizer.train(transactions)
            
            logger.info("✅ Optimizador de presupuesto entrenado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error entrenando optimizador de presupuesto: {e}")
            return False
    
    def train_anomaly_detector(self):
        """Entrenar detector de anomalías"""
        logger.info("Entrenando detector de anomalías...")
        
        try:
            # Obtener transacciones
            transactions = Transaction.objects.filter(
                organization=self.organization
            ).select_related('category')
            
            if transactions.count() < 100:
                logger.warning("Pocas transacciones para entrenar el detector de anomalías")
                return False
            
            # Entrenar modelo
            self.ai_service.anomaly_detector.train(transactions)
            
            logger.info("✅ Detector de anomalías entrenado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error entrenando detector de anomalías: {e}")
            return False
    
    def train_cash_flow_predictor(self):
        """Entrenar predictor de flujo de efectivo"""
        logger.info("Entrenando predictor de flujo de efectivo...")
        
        try:
            # Obtener transacciones
            transactions = Transaction.objects.filter(
                created_by=self.user
            ).select_related('category')
            
            if transactions.count() < 100:
                logger.warning("Pocas transacciones para entrenar el predictor de flujo de efectivo")
                return False
            
            # Entrenar modelo
            self.ai_service.cash_flow_predictor.train(transactions)
            
            logger.info("✅ Predictor de flujo de efectivo entrenado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error entrenando predictor de flujo de efectivo: {e}")
            return False
    
    def train_risk_analyzer(self):
        """Entrenar analizador de riesgo"""
        logger.info("Entrenando analizador de riesgo...")
        
        try:
            # Obtener transacciones
            transactions = Transaction.objects.filter(
                created_by=self.user
            ).select_related('category')
            
            if transactions.count() < 100:
                logger.warning("Pocas transacciones para entrenar el analizador de riesgo")
                return False
            
            # Entrenar modelo
            self.ai_service.risk_analyzer.train(transactions)
            
            logger.info("✅ Analizador de riesgo entrenado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error entrenando analizador de riesgo: {e}")
            return False
    
    def train_recommendation_engine(self):
        """Entrenar motor de recomendaciones"""
        logger.info("Entrenando motor de recomendaciones...")
        
        try:
            # Obtener transacciones
            transactions = Transaction.objects.filter(
                created_by=self.user
            ).select_related('category')
            
            if transactions.count() < 100:
                logger.warning("Pocas transacciones para entrenar el motor de recomendaciones")
                return False
            
            # Entrenar modelo
            self.ai_service.recommendation_engine.train(transactions)
            
            logger.info("✅ Motor de recomendaciones entrenado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error entrenando motor de recomendaciones: {e}")
            return False
    
    def evaluate_models(self):
        """Evaluar todos los modelos entrenados"""
        logger.info("Evaluando modelos entrenados...")
        
        results = {}
        
        # Evaluar clasificador de transacciones
        try:
            test_transactions = Transaction.objects.filter(
                organization=self.organization
            )[:10]
            
            for transaction in test_transactions:
                prediction = self.ai_service.analyze_transaction(transaction)
                logger.info(f"Predicción para '{transaction.description}': {prediction.get('category_prediction', 'N/A')}")
            
            results['transaction_classifier'] = 'success'
        except Exception as e:
            logger.error(f"Error evaluando clasificador: {e}")
            results['transaction_classifier'] = 'error'
        
        # Evaluar predictor de gastos
        try:
            predictions = self.ai_service.predict_expenses(
                user=self.user,
                category_id=self.categories[0].id,
                start_date=timezone.now().date(),
                end_date=timezone.now().date() + timedelta(days=30)
            )
            logger.info(f"Predicciones de gastos generadas: {len(predictions)}")
            results['expense_predictor'] = 'success'
        except Exception as e:
            logger.error(f"Error evaluando predictor de gastos: {e}")
            results['expense_predictor'] = 'error'
        
        # Evaluar analizador de comportamiento
        try:
            analysis = self.ai_service.analyze_behavior(self.user)
            logger.info(f"Análisis de comportamiento: {analysis.get('spending_patterns', 'N/A')}")
            results['behavior_analyzer'] = 'success'
        except Exception as e:
            logger.error(f"Error evaluando analizador de comportamiento: {e}")
            results['behavior_analyzer'] = 'error'
        
        # Evaluar detector de anomalías
        try:
            anomalies = self.ai_service.anomaly_detector.detect_anomalies(
                Transaction.objects.filter(organization=self.organization)[:50]
            )
            logger.info(f"Anomalías detectadas: {len(anomalies)}")
            results['anomaly_detector'] = 'success'
        except Exception as e:
            logger.error(f"Error evaluando detector de anomalías: {e}")
            results['anomaly_detector'] = 'error'
        
        logger.info("📊 Resultados de evaluación:")
        for model, result in results.items():
            logger.info(f"  {model}: {result}")
        
        return results
    
    def run_complete_training(self):
        """Ejecutar entrenamiento completo"""
        logger.info("🚀 Iniciando entrenamiento completo de modelos de IA...")
        
        # Configurar datos
        self.setup_test_data()
        
        # Generar datos sintéticos
        num_transactions = self.generate_synthetic_transactions(2000)
        
        # Entrenar todos los modelos
        training_results = {
            'transaction_classifier': self.train_transaction_classifier(),
            'expense_predictor': self.train_expense_predictor(),
            'behavior_analyzer': self.train_behavior_analyzer(),
            'budget_optimizer': self.train_budget_optimizer(),
            'anomaly_detector': self.train_anomaly_detector(),
            'cash_flow_predictor': self.train_cash_flow_predictor(),
            'risk_analyzer': self.train_risk_analyzer(),
            'recommendation_engine': self.train_recommendation_engine(),
        }
        
        # Evaluar modelos
        evaluation_results = self.evaluate_models()
        
        # Resumen final
        logger.info("🎉 Entrenamiento completado!")
        logger.info(f"📈 Transacciones generadas: {num_transactions}")
        logger.info("📊 Resultados del entrenamiento:")
        
        successful_models = sum(1 for result in training_results.values() if result)
        total_models = len(training_results)
        
        for model, success in training_results.items():
            status = "✅" if success else "❌"
            logger.info(f"  {status} {model}")
        
        logger.info(f"🎯 Éxito: {successful_models}/{total_models} modelos entrenados correctamente")
        
        return {
            'transactions_created': num_transactions,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'success_rate': successful_models / total_models
        }

def main():
    """Función principal"""
    trainer = ModelTrainer()
    results = trainer.run_complete_training()
    
    print("\n" + "="*60)
    print("🎉 ENTRENAMIENTO DE MODELOS DE IA COMPLETADO")
    print("="*60)
    print(f"📊 Transacciones generadas: {results['transactions_created']}")
    print(f"🎯 Tasa de éxito: {results['success_rate']:.1%}")
    print("="*60)

if __name__ == '__main__':
    main() 