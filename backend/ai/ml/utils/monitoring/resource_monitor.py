"""
Sistema de monitoreo de recursos.

Este módulo proporciona funcionalidades para:
- Monitorear uso de CPU, memoria y disco
- Registrar métricas de rendimiento
- Alertar sobre problemas de recursos
"""
import psutil
import time
import platform
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
        self.system = platform.system()
        
    def collect_metrics(self) -> Dict[str, Any]:
        """
        Recolecta métricas del sistema.
        
        Returns:
            Dict: Métricas recolectadas
        """
        try:
            # Métricas de CPU
            cpu_metrics = {
                'percent': psutil.cpu_percent(interval=1),
                'count': psutil.cpu_count()
            }
            
            # Intentar obtener frecuencia de CPU (compatible con macOS)
            try:
                if self.system == 'Darwin':  # macOS
                    # En macOS, usar un método alternativo o valores por defecto
                    cpu_metrics['freq'] = {
                        'current': None,
                        'min': None,
                        'max': None
                    }
                    # Intentar obtener información de CPU usando comandos del sistema
                    try:
                        import subprocess
                        result = subprocess.run(['sysctl', '-n', 'hw.cpufrequency'], 
                                              capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            freq_hz = int(result.stdout.strip())
                            cpu_metrics['freq']['current'] = freq_hz / 1000000  # Convertir a MHz
                    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
                        pass
                else:
                    # Para otros sistemas, usar psutil
                    try:
                        cpu_freq = psutil.cpu_freq()
                        if cpu_freq:
                            cpu_metrics['freq'] = {
                                'current': cpu_freq.current,
                                'min': cpu_freq.min,
                                'max': cpu_freq.max
                            }
                        else:
                            cpu_metrics['freq'] = None
                    except Exception:
                        cpu_metrics['freq'] = None
            except Exception as e:
                logger.debug(f"No se pudo obtener frecuencia de CPU: {e}")
                cpu_metrics['freq'] = None
            
            # Métricas de memoria
            memory = psutil.virtual_memory()
            memory_metrics = {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used,
                'free': memory.free
            }
            
            # Métricas de disco
            try:
                disk = psutil.disk_usage('/')
                disk_metrics = {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': disk.percent
                }
            except Exception:
                # Fallback para sistemas donde '/' no está disponible
                disk_metrics = {
                    'total': 0,
                    'used': 0,
                    'free': 0,
                    'percent': 0
                }
            
            # Métricas de red
            try:
                net_io = psutil.net_io_counters()
                network_metrics = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv
                }
            except Exception:
                network_metrics = {
                    'bytes_sent': 0,
                    'bytes_recv': 0,
                    'packets_sent': 0,
                    'packets_recv': 0
                }
            
            metrics = {
                'cpu': cpu_metrics,
                'memory': memory_metrics,
                'disk': disk_metrics,
                'network': network_metrics,
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
        try:
            latest_metrics = cache.get('system_metrics_latest')
            if latest_metrics is None:
                # Si no hay métricas en caché, recolectar nuevas
                return self.collect_metrics()
            return latest_metrics
        except Exception as e:
            logger.error(f"Error obteniendo últimas métricas: {str(e)}")
            return self.collect_metrics()
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Obtiene las métricas actuales del sistema.
        Alias para get_latest_metrics para compatibilidad.
        
        Returns:
            Dict: Métricas actuales
        """
        return self.get_latest_metrics()
        
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