"""
Script para entrenar los modelos de ML inicialmente.
"""

import logging
from datetime import datetime, timedelta
from django.db.models import Q

from transactions.models import Transaction
from ai.ml import TransactionClassifier, ExpensePredictor, BehaviorAnalyzer
from ai.models import AIPrediction

logger = logging.getLogger(__name__)

def train_models():
    """
    Entrena todos los modelos de ML con datos históricos.
    """
    try:
        # Obtener transacciones de los últimos 6 meses
        start_date = datetime.now() - timedelta(days=180)
        transactions = Transaction.objects.filter(
            Q(date__gte=start_date) & 
            Q(amount__isnull=False) &
            Q(category__isnull=False)
        ).select_related('category', 'merchant')

        if not transactions:
            logger.warning("No hay suficientes transacciones para entrenar los modelos")
            return

        # Entrenar clasificador de transacciones
        classifier = TransactionClassifier()
        classifier.train(transactions)
        classifier.save_model()

        # Entrenar predictor de gastos
        predictor = ExpensePredictor()
        predictor.train(transactions)
        predictor.save_model()

        # Entrenar analizador de comportamiento
        analyzer = BehaviorAnalyzer()
        analyzer.train(transactions)
        analyzer.save_model()

        logger.info("Modelos entrenados exitosamente")

    except Exception as e:
        logger.error(f"Error entrenando modelos: {str(e)}")
        raise

if __name__ == "__main__":
    train_models() 