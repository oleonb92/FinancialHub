"""
Sistema de métricas de rendimiento para evaluar y mejorar los modelos de ML.
"""
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
import numpy as np
import pandas as pd
from datetime import datetime
import json
import logging
from django.core.cache import cache
from django.utils import timezone

logger = logging.getLogger('ai.metrics')

class ModelMetrics:
    def __init__(self, model_name):
        self.model_name = model_name
        self.metrics_history = []
        
    def evaluate_classification(self, y_true, y_pred, y_prob=None):
        """
        Evalúa el rendimiento de un modelo de clasificación.
        
        Args:
            y_true: Valores reales
            y_pred: Predicciones
            y_prob: Probabilidades de predicción (opcional)
            
        Returns:
            dict: Métricas de rendimiento
        """
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted')
            }
            
            # Matriz de confusión
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Reporte de clasificación
            metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
            
            # Calcular ROC AUC si hay probabilidades
            if y_prob is not None:
                from sklearn.metrics import roc_auc_score
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
                
            self._save_metrics(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating classification model: {str(e)}")
            raise
            
    def evaluate_regression(self, y_true, y_pred):
        """
        Evalúa el rendimiento de un modelo de regresión.
        
        Args:
            y_true: Valores reales
            y_pred: Predicciones
            
        Returns:
            dict: Métricas de rendimiento
        """
        try:
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
            
            # Calcular error porcentual medio
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            self._save_metrics(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating regression model: {str(e)}")
            raise
            
    def evaluate_clustering(self, X, labels):
        """
        Evalúa el rendimiento de un modelo de clustering.
        
        Args:
            X: Datos
            labels: Etiquetas de cluster
            
        Returns:
            dict: Métricas de rendimiento
        """
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            
            metrics = {
                'silhouette_score': silhouette_score(X, labels),
                'calinski_harabasz_score': calinski_harabasz_score(X, labels)
            }
            
            # Calcular métricas adicionales
            n_clusters = len(np.unique(labels))
            metrics['n_clusters'] = n_clusters
            
            # Calcular tamaño de clusters
            cluster_sizes = np.bincount(labels)
            metrics['cluster_sizes'] = cluster_sizes.tolist()
            metrics['cluster_size_std'] = np.std(cluster_sizes)
            
            self._save_metrics(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating clustering model: {str(e)}")
            raise
            
    def _save_metrics(self, metrics):
        """Guarda las métricas en el historial."""
        timestamp = timezone.now()
        self.metrics_history.append({
            'timestamp': timestamp,
            'metrics': metrics
        })
        
        # Guardar en caché
        cache_key = f'model_metrics_{self.model_name}'
        try:
            cache.set(cache_key, self.metrics_history, timeout=3600)
        except Exception as cache_exc:
            logger.warning(f"Error saving metrics to cache: {cache_exc}")
        
    def get_metrics_history(self, days=30):
        """
        Obtiene el historial de métricas.
        
        Args:
            days: Número de días de historial a obtener
            
        Returns:
            list: Historial de métricas
        """
        cache_key = f'model_metrics_{self.model_name}'
        history = cache.get(cache_key)
        
        if history is None:
            return self.metrics_history
            
        # Filtrar por fecha
        cutoff_date = timezone.now() - timezone.timedelta(days=days)
        return [
            entry for entry in history
            if entry['timestamp'] >= cutoff_date
        ]
        
    def get_latest_metrics(self):
        """Obtiene las métricas más recientes."""
        if not self.metrics_history:
            return None
        return self.metrics_history[-1]['metrics']
        
    def get_metrics_trend(self, metric_name, days=30):
        """
        Obtiene la tendencia de una métrica específica.
        
        Args:
            metric_name: Nombre de la métrica
            days: Número de días de historial
            
        Returns:
            dict: Tendencias de la métrica
        """
        history = self.get_metrics_history(days)
        if not history:
            return None
            
        values = [entry['metrics'].get(metric_name) for entry in history]
        values = [v for v in values if v is not None]
        
        if not values:
            return None
            
        return {
            'current': values[-1],
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'trend': np.polyfit(range(len(values)), values, 1)[0]
        }
        
    def export_metrics(self, format='json'):
        """
        Exporta las métricas en el formato especificado.
        
        Args:
            format: Formato de exportación ('json' o 'csv')
            
        Returns:
            str: Métricas exportadas
        """
        if format == 'json':
            return json.dumps(self.metrics_history, default=str)
        elif format == 'csv':
            df = pd.DataFrame([
                {
                    'timestamp': entry['timestamp'],
                    **entry['metrics']
                }
                for entry in self.metrics_history
            ])
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Formato no soportado: {format}") 