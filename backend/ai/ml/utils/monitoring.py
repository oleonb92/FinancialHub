"""
Utilidades para monitorear recursos del sistema.
"""
import psutil
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ResourceMonitor:
    """
    Clase para monitorear recursos del sistema.
    """
    
    def __init__(self):
        """Inicializar el monitor de recursos."""
        self.process = psutil.Process()
        
    def get_current_metrics(self) -> Dict[str, float]:
        """
        Obtener métricas actuales del sistema.
        
        Returns:
            Dict[str, float]: Diccionario con métricas del sistema
        """
        try:
            # Métricas de CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            process_cpu = self.process.cpu_percent()
            
            # Métricas de memoria
            memory = psutil.virtual_memory()
            process_memory = self.process.memory_info()
            
            # Métricas de disco
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'process_cpu_percent': process_cpu,
                'memory_percent': memory.percent,
                'process_memory_percent': process_memory.rss / memory.total * 100,
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'memory_available_gb': memory.available / (1024**3)
            }
            
        except Exception as e:
            logger.error(f"Error al obtener métricas del sistema: {str(e)}")
            return {
                'cpu_percent': 0.0,
                'process_cpu_percent': 0.0,
                'memory_percent': 0.0,
                'process_memory_percent': 0.0,
                'disk_percent': 0.0,
                'disk_free_gb': 0.0,
                'memory_available_gb': 0.0
            }
            
    def check_resources(self, min_memory_gb: float = 1.0, min_disk_gb: float = 5.0) -> Dict[str, Any]:
        """
        Verificar si hay suficientes recursos disponibles.
        
        Args:
            min_memory_gb (float): Memoria mínima requerida en GB
            min_disk_gb (float): Espacio en disco mínimo requerido en GB
            
        Returns:
            Dict[str, Any]: Diccionario con el resultado de la verificación
        """
        metrics = self.get_current_metrics()
        
        memory_available = metrics['memory_available_gb']
        disk_free = metrics['disk_free_gb']
        
        has_resources = (
            memory_available >= min_memory_gb and
            disk_free >= min_disk_gb
        )
        
        return {
            'has_resources': has_resources,
            'memory_available_gb': memory_available,
            'disk_free_gb': disk_free,
            'metrics': metrics
        }
        
    def get_process_info(self) -> Dict[str, Any]:
        """
        Obtener información detallada del proceso actual.
        
        Returns:
            Dict[str, Any]: Diccionario con información del proceso
        """
        try:
            return {
                'pid': self.process.pid,
                'name': self.process.name(),
                'status': self.process.status(),
                'create_time': self.process.create_time(),
                'cpu_percent': self.process.cpu_percent(),
                'memory_percent': self.process.memory_percent(),
                'num_threads': self.process.num_threads(),
                'num_handles': self.process.num_handles() if hasattr(self.process, 'num_handles') else None
            }
        except Exception as e:
            logger.error(f"Error al obtener información del proceso: {str(e)}")
            return {} 