"""
Optimizador de memoria para el sistema de IA.

Este módulo implementa:
- Lazy loading de modelos
- Gestión inteligente de memoria
- Limpieza automática de caché
- Monitoreo de recursos
"""
import gc
import psutil
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps
import weakref
from threading import Lock
import time

logger = logging.getLogger('ai.memory_optimizer')

class MemoryOptimizer:
    """
    Optimizador de memoria para modelos de IA.
    
    Características:
    - Lazy loading de modelos
    - Gestión automática de memoria
    - Limpieza inteligente de caché
    - Monitoreo de recursos
    """
    
    def __init__(self, max_memory_percent: float = 80.0):
        self.max_memory_percent = max_memory_percent
        self.loaded_models = weakref.WeakValueDictionary()
        self.model_locks = {}
        self.global_lock = Lock()
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutos
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Obtiene el uso actual de memoria"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent,
            'free': memory.free
        }
    
    def is_memory_high(self) -> bool:
        """Verifica si el uso de memoria es alto"""
        memory_usage = self.get_memory_usage()
        return memory_usage['percent'] > self.max_memory_percent
    
    def cleanup_memory(self, force: bool = False) -> Dict[str, Any]:
        """
        Limpia memoria automáticamente.
        
        Args:
            force: Forzar limpieza incluso si no es necesario
            
        Returns:
            Dict con información de la limpieza
        """
        current_time = time.time()
        
        # Limpiar solo si es necesario o forzado
        if not force and (current_time - self.last_cleanup) < self.cleanup_interval:
            return {'status': 'skipped', 'reason': 'too_soon'}
        
        with self.global_lock:
            # Forzar garbage collection
            collected = gc.collect()
            
            # Limpiar modelos no utilizados
            models_before = len(self.loaded_models)
            self.loaded_models.clear()
            models_after = len(self.loaded_models)
            
            # Limpiar locks no utilizados
            locks_before = len(self.model_locks)
            self.model_locks.clear()
            locks_after = len(self.model_locks)
            
            self.last_cleanup = current_time
            
            memory_after = self.get_memory_usage()
            
            logger.info(f"Memory cleanup completed: {collected} objects collected")
            
            return {
                'status': 'success',
                'garbage_collected': collected,
                'models_cleared': models_before - models_after,
                'locks_cleared': locks_before - locks_after,
                'memory_after': memory_after
            }
    
    def lazy_load_model(self, model_name: str, loader_func: Callable) -> Any:
        """
        Carga un modelo de forma lazy (solo cuando se necesita).
        
        Args:
            model_name: Nombre del modelo
            loader_func: Función que carga el modelo
            
        Returns:
            El modelo cargado
        """
        # Verificar si el modelo ya está cargado
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        # Verificar memoria antes de cargar
        if self.is_memory_high():
            logger.warning(f"High memory usage before loading {model_name}, cleaning up...")
            self.cleanup_memory(force=True)
        
        # Obtener lock para este modelo
        if model_name not in self.model_locks:
            self.model_locks[model_name] = Lock()
        
        with self.model_locks[model_name]:
            # Verificar nuevamente después del lock
            if model_name in self.loaded_models:
                return self.loaded_models[model_name]
            
            # Cargar el modelo
            logger.info(f"Loading model: {model_name}")
            model = loader_func()
            
            # Almacenar en caché
            self.loaded_models[model_name] = model
            
            # Verificar memoria después de cargar
            if self.is_memory_high():
                logger.warning(f"High memory usage after loading {model_name}")
            
            return model
    
    def unload_model(self, model_name: str) -> bool:
        """
        Descarga un modelo de memoria.
        
        Args:
            model_name: Nombre del modelo a descargar
            
        Returns:
            True si se descargó exitosamente
        """
        with self.global_lock:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
                logger.info(f"Model unloaded: {model_name}")
                return True
            return False
    
    def get_loaded_models(self) -> Dict[str, Any]:
        """Obtiene información de los modelos cargados"""
        return {
            'count': len(self.loaded_models),
            'models': list(self.loaded_models.keys()),
            'memory_usage': self.get_memory_usage()
        }

# Instancia global del optimizador
memory_optimizer = MemoryOptimizer()

def optimize_memory(func: Callable) -> Callable:
    """
    Decorador para optimizar memoria en funciones.
    
    Args:
        func: Función a decorar
        
    Returns:
        Función decorada con optimización de memoria
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Verificar memoria antes de ejecutar
        if memory_optimizer.is_memory_high():
            logger.warning(f"High memory usage before {func.__name__}, cleaning up...")
            memory_optimizer.cleanup_memory(force=True)
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Verificar memoria después de ejecutar
            if memory_optimizer.is_memory_high():
                logger.warning(f"High memory usage after {func.__name__}")
    
    return wrapper

def lazy_model_loader(model_name: str):
    """
    Decorador para hacer lazy loading de modelos.
    
    Args:
        model_name: Nombre del modelo
        
    Returns:
        Decorador que implementa lazy loading
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return memory_optimizer.lazy_load_model(model_name, lambda: func(*args, **kwargs))
        return wrapper
    return decorator 