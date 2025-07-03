"""
Celery tasks for AI model training and evaluation.
"""
from celery import shared_task
from django.utils import timezone
from datetime import timedelta
import logging
from ai.services import AIService
from transactions.models import Transaction
from django.db.models import Q

logger = logging.getLogger(__name__)

@shared_task(bind=True, name='ai.tasks.training.train_models')
def train_models(self):
    """
    Celery task to train all AI models.
    """
    try:
        logger.info("[AI][TASK] Iniciando entrenamiento de modelos...")
        
        ai_service = AIService()
        result = ai_service.train_models()
        
        logger.info(f"[AI][TASK] Entrenamiento completado: {result}")
        
        return {
            'status': 'success',
            'timestamp': timezone.now().isoformat(),
            'results': result
        }
        
    except Exception as e:
        logger.error(f"[AI][TASK] Error en train_models: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': timezone.now().isoformat()
        }

@shared_task(bind=True, name='ai.tasks.training.evaluate_models')
def evaluate_models(self, model_name=None):
    """
    Celery task to evaluate AI model performance.
    
    Args:
        model_name: Optional specific model to evaluate
    """
    try:
        logger.info(f"[AI][TASK] Iniciando evaluación de modelos: {model_name or 'todos'}")
        
        ai_service = AIService()
        
        if model_name:
            # Evaluate specific model
            metrics = ai_service.get_model_metrics(model_name)
            result = {
                'model': model_name,
                'metrics': metrics
            }
        else:
            # Evaluate all models
            models = ['transaction_classifier', 'expense_predictor', 'behavior_analyzer']
            result = {}
            
            for model in models:
                try:
                    metrics = ai_service.get_model_metrics(model)
                    result[model] = metrics
                except Exception as e:
                    logger.error(f"[AI][TASK] Error evaluando modelo {model}: {str(e)}")
                    result[model] = {'status': 'error', 'error': str(e)}
        
        logger.info(f"[AI][TASK] Evaluación completada: {result}")
        
        return {
            'status': 'success',
            'timestamp': timezone.now().isoformat(),
            'results': result
        }
        
    except Exception as e:
        logger.error(f"[AI][TASK] Error en evaluate_models: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': timezone.now().isoformat()
        }

@shared_task(bind=True, name='ai.tasks.training.cleanup_old_versions')
def cleanup_old_versions(self, days_to_keep=30):
    """
    Celery task to cleanup old model versions.
    
    Args:
        days_to_keep: Number of days to keep model versions
    """
    try:
        logger.info(f"[AI][TASK] Iniciando limpieza de versiones antiguas (mantener {days_to_keep} días)")
        
        ai_service = AIService()
        
        # Get memory status before cleanup
        memory_before = ai_service.get_memory_status()
        
        # Perform cleanup
        cleanup_result = ai_service.cleanup_memory(force=True)
        
        # Get memory status after cleanup
        memory_after = ai_service.get_memory_status()
        
        result = {
            'cleanup_result': cleanup_result,
            'memory_before': memory_before,
            'memory_after': memory_after,
            'days_kept': days_to_keep
        }
        
        logger.info(f"[AI][TASK] Limpieza completada: {result}")
        
        return {
            'status': 'success',
            'timestamp': timezone.now().isoformat(),
            'results': result
        }
        
    except Exception as e:
        logger.error(f"[AI][TASK] Error en cleanup_old_versions: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': timezone.now().isoformat()
        }

@shared_task(bind=True, name='ai.tasks.training.retrain_low_performance_models')
def retrain_low_performance_models(self):
    """
    Celery task to automatically retrain models with low performance.
    """
    try:
        logger.info("[AI][TASK] Iniciando retraining de modelos con bajo rendimiento...")
        
        ai_service = AIService()
        result = ai_service.auto_retrain_low_performance_models()
        
        logger.info(f"[AI][TASK] Retraining completado: {result}")
        
        return {
            'status': 'success',
            'timestamp': timezone.now().isoformat(),
            'results': result
        }
        
    except Exception as e:
        logger.error(f"[AI][TASK] Error en retrain_low_performance_models: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': timezone.now().isoformat()
        }

@shared_task(bind=True, name='ai.tasks.training.analyze_transaction_ai')
def analyze_transaction_ai(self, transaction_id):
    """
    Celery task to analyze a single transaction with AI.
    
    Args:
        transaction_id: ID of the transaction to analyze
    """
    try:
        logger.info(f"[AI][TASK] Analizando transacción {transaction_id}...")
        
        transaction = Transaction.objects.get(id=transaction_id)
        ai_service = AIService()
        
        result = ai_service.analyze_transaction(transaction)
        
        logger.info(f"[AI][TASK] Análisis completado para transacción {transaction_id}")
        
        return {
            'status': 'success',
            'transaction_id': transaction_id,
            'timestamp': timezone.now().isoformat(),
            'results': result
        }
        
    except Transaction.DoesNotExist:
        logger.error(f"[AI][TASK] Transacción {transaction_id} no encontrada")
        return {
            'status': 'error',
            'error': f'Transaction {transaction_id} not found',
            'timestamp': timezone.now().isoformat()
        }
    except Exception as e:
        logger.error(f"[AI][TASK] Error analizando transacción {transaction_id}: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': timezone.now().isoformat()
        } 