"""
Sistema de monitoreo de recursos.

Este módulo proporciona funcionalidades para:
- Monitorear uso de CPU, memoria y disco
- Registrar métricas de rendimiento
- Alertar sobre problemas de recursos
"""
import psutil
import time
from datetime import datetime
from django.core.cache import cache
import logging
from typing import Dict, Any, List
import threading
from queue import Queue
import json

logger = logging.getLogger('ai.monitoring')

class ResourceMonitor:
    def __init__(self, alert_thresholds: Dict[str, float] = None):
        self.metrics = {}
        self.alert_thresholds = alert_thresholds or {
            'cpu_percent': 80.0,
            'memory_percent': 80.0,
            'disk_percent': 90.0
        }
        self.metrics_queue = Queue()
        self.is_monitoring = False
        self.monitor_thread = None
        
    def collect_metrics(self) -> Dict[str, Any]:
        """
        Recolecta métricas del sistema.
        
        Returns:
            Dict: Métricas recolectadas
        """
        try:
            metrics = {
                'cpu': {
                    'percent': psutil.cpu_percent(interval=1),
                    'count': psutil.cpu_count(),
                    'freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
                },
                'memory': {
                    'total': psutil.virtual_memory().total,
                    'available': psutil.virtual_memory().available,
                    'percent': psutil.virtual_memory().percent,
                    'used': psutil.virtual_memory().used,
                    'free': psutil.virtual_memory().free
                },
                'disk': {
                    'total': psutil.disk_usage('/').total,
                    'used': psutil.disk_usage('/').used,
                    'free': psutil.disk_usage('/').free,
                    'percent': psutil.disk_usage('/').percent
                },
                'network': {
                    'bytes_sent': psutil.net_io_counters().bytes_sent,
                    'bytes_recv': psutil.net_io_counters().bytes_recv,
                    'packets_sent': psutil.net_io_counters().packets_sent,
                    'packets_recv': psutil.net_io_counters().packets_recv
                },
                'timestamp': datetime.now().isoformat()
            }
            
            self.metrics = metrics
            self._check_alerts(metrics)
            self._store_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error recolectando métricas: {str(e)}")
            raise
            
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Verifica si hay alertas basadas en los umbrales"""
        alerts = []
        
        if metrics['cpu']['percent'] > self.alert_thresholds['cpu_percent']:
            alerts.append({
                'type': 'cpu',
                'level': 'warning',
                'message': f"Uso de CPU alto: {metrics['cpu']['percent']}%"
            })
            
        if metrics['memory']['percent'] > self.alert_thresholds['memory_percent']:
            alerts.append({
                'type': 'memory',
                'level': 'warning',
                'message': f"Uso de memoria alto: {metrics['memory']['percent']}%"
            })
            
        if metrics['disk']['percent'] > self.alert_thresholds['disk_percent']:
            alerts.append({
                'type': 'disk',
                'level': 'warning',
                'message': f"Uso de disco alto: {metrics['disk']['percent']}%"
            })
            
        if alerts:
            self._handle_alerts(alerts)
            
    def _handle_alerts(self, alerts: List[Dict[str, Any]]):
        """Maneja las alertas generadas"""
        for alert in alerts:
            logger.warning(f"Alerta de recursos: {alert['message']}")
            # Aquí se podría implementar notificación por email, Slack, etc.
            
    def _store_metrics(self, metrics: Dict[str, Any]):
        """Almacena las métricas en caché"""
        try:
            # Obtener métricas históricas
            historical_metrics = cache.get('system_metrics_history', [])
            
            # Agregar nuevas métricas
            historical_metrics.append(metrics)
            
            # Mantener solo las últimas 24 horas de métricas (asumiendo recolección cada 5 minutos)
            max_entries = 24 * 12  # 24 horas * 12 entradas por hora
            if len(historical_metrics) > max_entries:
                historical_metrics = historical_metrics[-max_entries:]
                
            # Guardar en caché
            cache.set('system_metrics_history', historical_metrics, timeout=86400)  # 24 horas
            cache.set('system_metrics_latest', metrics, timeout=300)  # 5 minutos
            
        except Exception as e:
            logger.error(f"Error almacenando métricas: {str(e)}")
            
    def start_monitoring(self, interval: int = 300):
        """
        Inicia el monitoreo continuo en un hilo separado.
        
        Args:
            interval: Intervalo de monitoreo en segundos (default: 5 minutos)
        """
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Detiene el monitoreo continuo"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def _monitoring_loop(self, interval: int):
        """Loop principal de monitoreo"""
        while self.is_monitoring:
            try:
                self.collect_metrics()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error en loop de monitoreo: {str(e)}")
                time.sleep(60)  # Esperar 1 minuto antes de reintentar
                
    def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Obtiene el historial de métricas.
        
        Args:
            hours: Número de horas de historial a obtener
            
        Returns:
            List: Historial de métricas
        """
        try:
            historical_metrics = cache.get('system_metrics_history', [])
            if not historical_metrics:
                return []
                
            # Filtrar por tiempo
            cutoff_time = time.time() - (hours * 3600)
            return [
                m for m in historical_metrics
                if datetime.fromisoformat(m['timestamp']).timestamp() > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error obteniendo historial de métricas: {str(e)}")
            return []
            
    def get_latest_metrics(self) -> Dict[str, Any]:
        """
        Obtiene las últimas métricas recolectadas.
        
        Returns:
            Dict: Últimas métricas
        """
        return cache.get('system_metrics_latest', {})
        
    def export_metrics(self, format: str = 'json') -> str:
        """
        Exporta las métricas en el formato especificado.
        
        Args:
            format: Formato de exportación ('json' o 'csv')
            
        Returns:
            str: Métricas en el formato especificado
        """
        try:
            metrics = self.get_metrics_history()
            
            if format == 'json':
                return json.dumps(metrics, indent=2)
            elif format == 'csv':
                if not metrics:
                    return ''
                    
                # Crear encabezados
                headers = list(metrics[0].keys())
                rows = [','.join(headers)]
                
                # Agregar datos
                for metric in metrics:
                    row = []
                    for header in headers:
                        value = metric.get(header, '')
                        if isinstance(value, (dict, list)):
                            value = json.dumps(value)
                        row.append(str(value))
                    rows.append(','.join(row))
                    
                return '\n'.join(rows)
            else:
                raise ValueError(f"Formato no soportado: {format}")
                
        except Exception as e:
            logger.error(f"Error exportando métricas: {str(e)}")
            raise 