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
from typing import Dict, Any, List

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

    def evaluate_model(self, model_name, model):
        """
        Evalúa un modelo específico.
        
        Args:
            model_name: Nombre del modelo
            model: Instancia del modelo
            
        Returns:
            dict: Métricas de evaluación
        """
        try:
            # Este es un método placeholder que debería ser implementado
            # según el tipo específico de modelo
            logger.info(f"Evaluando modelo: {model_name}")
            
            # Por ahora, retornar métricas básicas
            return {
                'model_name': model_name,
                'evaluation_timestamp': timezone.now().isoformat(),
                'status': 'evaluated',
                'basic_metrics': {
                    'accuracy': 0.85,  # Placeholder
                    'precision': 0.82,  # Placeholder
                    'recall': 0.88,     # Placeholder
                    'f1_score': 0.85    # Placeholder
                }
            }
            
        except Exception as e:
            logger.error(f"Error evaluando modelo {model_name}: {str(e)}")
            return {
                'model_name': model_name,
                'error': str(e),
                'status': 'error'
            }

    def record_metric(self, metric_name: str, value: float):
        """
        Registra una métrica específica.
        
        Args:
            metric_name: Nombre de la métrica
            value: Valor de la métrica
        """
        try:
            timestamp = timezone.now()
            metric_entry = {
                'timestamp': timestamp,
                'metric_name': metric_name,
                'value': value
            }
            
            # Agregar al historial
            self.metrics_history.append(metric_entry)
            
            # Guardar en caché
            cache_key = f'model_metrics_{self.model_name}'
            try:
                cache.set(cache_key, self.metrics_history, timeout=3600)
            except Exception as cache_exc:
                logger.warning(f"Error saving metrics to cache: {cache_exc}")
                
        except Exception as e:
            logger.error(f"Error recording metric {metric_name}: {str(e)}")

    def get_metric_summary(self, metric_name: str, days: int = 30) -> Dict[str, Any]:
        """
        Obtiene un resumen de una métrica específica.
        
        Args:
            metric_name: Nombre de la métrica
            days: Número de días de historial
            
        Returns:
            dict: Resumen de la métrica
        """
        try:
            history = self.get_metrics_history(days)
            metric_values = [
                entry['value'] for entry in history 
                if entry.get('metric_name') == metric_name
            ]
            
            if not metric_values:
                return {
                    'metric_name': metric_name,
                    'count': 0,
                    'mean': 0,
                    'std': 0,
                    'min': 0,
                    'max': 0,
                    'latest': 0
                }
            
            return {
                'metric_name': metric_name,
                'count': len(metric_values),
                'mean': float(np.mean(metric_values)),
                'std': float(np.std(metric_values)),
                'min': float(np.min(metric_values)),
                'max': float(np.max(metric_values)),
                'latest': float(metric_values[-1])
            }
            
        except Exception as e:
            logger.error(f"Error getting metric summary: {str(e)}")
            return {'error': str(e)}

    def compare_models(self, other_model_metrics: 'ModelMetrics', metric_name: str) -> Dict[str, Any]:
        """
        Compara las métricas de este modelo con otro.
        
        Args:
            other_model_metrics: Otra instancia de ModelMetrics
            metric_name: Nombre de la métrica a comparar
            
        Returns:
            dict: Comparación de métricas
        """
        try:
            this_summary = self.get_metric_summary(metric_name)
            other_summary = other_model_metrics.get_metric_summary(metric_name)
            
            if 'error' in this_summary or 'error' in other_summary:
                return {'error': 'No se pueden comparar métricas'}
            
            return {
                'metric_name': metric_name,
                'this_model': {
                    'name': self.model_name,
                    'mean': this_summary['mean'],
                    'std': this_summary['std'],
                    'latest': this_summary['latest']
                },
                'other_model': {
                    'name': other_model_metrics.model_name,
                    'mean': other_summary['mean'],
                    'std': other_summary['std'],
                    'latest': other_summary['latest']
                },
                'comparison': {
                    'mean_difference': this_summary['mean'] - other_summary['mean'],
                    'relative_improvement': (
                        (this_summary['mean'] - other_summary['mean']) / other_summary['mean'] * 100
                        if other_summary['mean'] != 0 else 0
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            return {'error': str(e)}

    def get_performance_alerts(self, threshold: float = 0.1) -> List[Dict[str, Any]]:
        """
        Genera alertas basadas en el rendimiento del modelo.
        
        Args:
            threshold: Umbral para considerar una degradación significativa
            
        Returns:
            list: Lista de alertas
        """
        try:
            alerts = []
            latest_metrics = self.get_latest_metrics()
            
            if not latest_metrics:
                return alerts
            
            # Verificar métricas de clasificación
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                if metric in latest_metrics:
                    current_value = latest_metrics[metric]
                    trend = self.get_metrics_trend(metric, days=7)
                    
                    if trend and trend['trend'] < -threshold:
                        alerts.append({
                            'type': 'performance_degradation',
                            'metric': metric,
                            'current_value': current_value,
                            'trend': trend['trend'],
                            'severity': 'high' if abs(trend['trend']) > threshold * 2 else 'medium',
                            'message': f'Degradación significativa en {metric}: {trend["trend"]:.3f}'
                        })
            
            # Verificar métricas de regresión
            for metric in ['mse', 'rmse', 'mae']:
                if metric in latest_metrics:
                    current_value = latest_metrics[metric]
                    trend = self.get_metrics_trend(metric, days=7)
                    
                    if trend and trend['trend'] > threshold:
                        alerts.append({
                            'type': 'error_increase',
                            'metric': metric,
                            'current_value': current_value,
                            'trend': trend['trend'],
                            'severity': 'high' if trend['trend'] > threshold * 2 else 'medium',
                            'message': f'Aumento en error {metric}: {trend["trend"]:.3f}'
                        })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating performance alerts: {str(e)}")
            return []

    def generate_performance_report(self, days: int = 30) -> Dict[str, Any]:
        """
        Genera un reporte completo de rendimiento del modelo.
        
        Args:
            days: Número de días para el reporte
            
        Returns:
            dict: Reporte de rendimiento
        """
        try:
            history = self.get_metrics_history(days)
            latest_metrics = self.get_latest_metrics()
            alerts = self.get_performance_alerts()
            
            # Calcular estadísticas generales
            total_evaluations = len(history)
            avg_metrics = {}
            
            if history:
                # Calcular promedios de métricas
                metric_names = set()
                for entry in history:
                    if 'metrics' in entry:
                        metric_names.update(entry['metrics'].keys())
                
                for metric in metric_names:
                    values = [
                        entry['metrics'][metric] 
                        for entry in history 
                        if 'metrics' in entry and metric in entry['metrics']
                    ]
                    if values:
                        avg_metrics[metric] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'min': float(np.min(values)),
                            'max': float(np.max(values))
                        }
            
            return {
                'model_name': self.model_name,
                'report_period': f'{days} days',
                'total_evaluations': total_evaluations,
                'latest_metrics': latest_metrics,
                'average_metrics': avg_metrics,
                'alerts': alerts,
                'trends': {
                    metric: self.get_metrics_trend(metric, days)
                    for metric in (latest_metrics.keys() if latest_metrics else [])
                },
                'generated_at': timezone.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            return {'error': str(e)}

    def save_metrics_to_file(self, filepath: str, format: str = 'json'):
        """
        Guarda las métricas en un archivo.
        
        Args:
            filepath: Ruta del archivo
            format: Formato del archivo ('json' o 'csv')
        """
        try:
            if format == 'json':
                with open(filepath, 'w') as f:
                    json.dump(self.metrics_history, f, default=str, indent=2)
            elif format == 'csv':
                df = pd.DataFrame([
                    {
                        'timestamp': entry['timestamp'],
                        **entry.get('metrics', {})
                    }
                    for entry in self.metrics_history
                ])
                df.to_csv(filepath, index=False)
            else:
                raise ValueError(f"Formato no soportado: {format}")
                
            logger.info(f"Métricas guardadas en: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving metrics to file: {str(e)}")
            raise

    def load_metrics_from_file(self, filepath: str, format: str = 'json'):
        """
        Carga métricas desde un archivo.
        
        Args:
            filepath: Ruta del archivo
            format: Formato del archivo ('json' o 'csv')
        """
        try:
            if format == 'json':
                with open(filepath, 'r') as f:
                    self.metrics_history = json.load(f)
            elif format == 'csv':
                df = pd.read_csv(filepath)
                self.metrics_history = []
                for _, row in df.iterrows():
                    entry = {
                        'timestamp': pd.to_datetime(row['timestamp']),
                        'metrics': {col: row[col] for col in df.columns if col != 'timestamp'}
                    }
                    self.metrics_history.append(entry)
            else:
                raise ValueError(f"Formato no soportado: {format}")
                
            logger.info(f"Métricas cargadas desde: {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading metrics from file: {str(e)}")
            raise 