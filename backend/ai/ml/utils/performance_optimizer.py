"""
Optimizador de Rendimiento para Modelos de IA
Implementa técnicas avanzadas para mejorar el rendimiento de los modelos
"""

import time
import threading
import multiprocessing
from functools import wraps, lru_cache
from typing import Dict, List, Any, Callable
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import gc

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Optimizador de rendimiento para modelos de IA"""
    
    def __init__(self):
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.performance_metrics = {}
        self.model_cache = {}
        self.scaler_cache = {}
        
    def cache_result(self, key: str, result: Any, ttl: int = 3600):
        """Cachear resultado con tiempo de vida"""
        self.cache[key] = {
            'result': result,
            'timestamp': time.time(),
            'ttl': ttl
        }
    
    def get_cached_result(self, key: str) -> Any:
        """Obtener resultado cacheado"""
        if key in self.cache:
            cached = self.cache[key]
            if time.time() - cached['timestamp'] < cached['ttl']:
                self.cache_hits += 1
                return cached['result']
            else:
                del self.cache[key]
        
        self.cache_misses += 1
        return None
    
    def clear_expired_cache(self):
        """Limpiar cache expirado"""
        current_time = time.time()
        expired_keys = [
            key for key, cached in self.cache.items()
            if current_time - cached['timestamp'] > cached['ttl']
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def optimize_memory_usage(self):
        """Optimizar uso de memoria"""
        # Forzar garbage collection
        gc.collect()
        
        # Limpiar cache expirado
        self.clear_expired_cache()
        
        # Obtener información de memoria
        memory_info = psutil.virtual_memory()
        logger.info(f"Memory usage: {memory_info.percent}%")
        
        # Si el uso de memoria es alto, limpiar más agresivamente
        if memory_info.percent > 80:
            self.cache.clear()
            self.model_cache.clear()
            self.scaler_cache.clear()
            gc.collect()
            logger.warning("High memory usage detected, cleared all caches")
    
    def parallel_processing(self, func: Callable, data: List, max_workers: int = None) -> List:
        """Procesar datos en paralelo"""
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), len(data))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(func, data))
        
        return results
    
    def batch_processing(self, func: Callable, data: List, batch_size: int = 100) -> List:
        """Procesar datos en lotes"""
        results = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_results = func(batch)
            results.extend(batch_results)
        return results
    
    def optimize_data_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimizar preprocesamiento de datos"""
        # Reducir uso de memoria
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = data[col].astype('category')
        
        # Optimizar tipos numéricos
        for col in data.select_dtypes(include=['int64']).columns:
            data[col] = pd.to_numeric(data[col], downcast='integer')
        
        for col in data.select_dtypes(include=['float64']).columns:
            data[col] = pd.to_numeric(data[col], downcast='float')
        
        return data
    
    def cache_model(self, model_name: str, model: Any):
        """Cachear modelo en memoria"""
        self.model_cache[model_name] = {
            'model': model,
            'timestamp': time.time()
        }
    
    def get_cached_model(self, model_name: str) -> Any:
        """Obtener modelo cacheado"""
        if model_name in self.model_cache:
            return self.model_cache[model_name]['model']
        return None
    
    def optimize_inference(self, model: Any, data: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        """Optimizar inferencia del modelo"""
        if hasattr(model, 'predict'):
            # Procesar en lotes para evitar problemas de memoria
            predictions = []
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                batch_predictions = model.predict(batch)
                predictions.extend(batch_predictions)
            return np.array(predictions)
        return None
    
    def monitor_performance(self, func_name: str, start_time: float):
        """Monitorear rendimiento de funciones"""
        execution_time = time.time() - start_time
        
        if func_name not in self.performance_metrics:
            self.performance_metrics[func_name] = []
        
        self.performance_metrics[func_name].append(execution_time)
        
        # Mantener solo las últimas 100 mediciones
        if len(self.performance_metrics[func_name]) > 100:
            self.performance_metrics[func_name] = self.performance_metrics[func_name][-100:]
        
        logger.debug(f"{func_name} executed in {execution_time:.4f}s")
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Obtener estadísticas de rendimiento"""
        stats = {}
        for func_name, times in self.performance_metrics.items():
            if times:
                stats[func_name] = {
                    'mean': np.mean(times),
                    'median': np.median(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'count': len(times)
                }
        return stats
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Obtener estadísticas de cache"""
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'cache_size': len(self.cache),
            'model_cache_size': len(self.model_cache)
        }

# Decoradores para optimización
def performance_monitor(func: Callable) -> Callable:
    """Decorador para monitorear rendimiento"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        try:
            result = func(self, *args, **kwargs)
            # Verificar si el objeto tiene el método monitor_performance
            if hasattr(self, 'monitor_performance'):
                try:
                    # Intentar llamar con los parámetros correctos
                    if hasattr(self.monitor_performance, '__code__') and self.monitor_performance.__code__.co_argcount == 3:
                        # Si el método espera 3 argumentos (self, func_name, start_time)
                        self.monitor_performance(func.__name__, start_time)
                    else:
                        # Si el método no espera argumentos adicionales
                        self.monitor_performance()
                except Exception as e:
                    logger.warning(f"Error calling monitor_performance: {e}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper

def cache_result(ttl: int = 3600):
    """Decorador para cachear resultados"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Crear clave de cache basada en función y argumentos
            cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Intentar obtener del cache
            if hasattr(self, 'get_cached_result'):
                cached_result = self.get_cached_result(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Ejecutar función
            result = func(self, *args, **kwargs)
            
            # Cachear resultado
            if hasattr(self, 'cache_result'):
                self.cache_result(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

def optimize_memory(func: Callable) -> Callable:
    """Decorador para optimizar memoria"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Optimizar memoria antes de ejecutar
        if hasattr(self, 'optimize_memory_usage'):
            self.optimize_memory_usage()
        
        result = func(self, *args, **kwargs)
        
        # Optimizar memoria después de ejecutar
        if hasattr(self, 'optimize_memory_usage'):
            self.optimize_memory_usage()
        
        return result
    return wrapper

def parallel_execution(max_workers: int = None):
    """Decorador para ejecución paralela"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Si el primer argumento es una lista, procesar en paralelo
            if args and isinstance(args[0], list):
                data = args[0]
                if hasattr(self, 'parallel_processing'):
                    return self.parallel_processing(func, data, max_workers)
            
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

# Clase para optimización específica de modelos
class ModelOptimizer:
    """Optimizador específico para modelos de ML"""
    
    def __init__(self):
        self.optimizer = PerformanceOptimizer()
        self.model_versions = {}
        self.feature_cache = {}
    
    def optimize_model_loading(self, model_path: str) -> Any:
        """Optimizar carga de modelos"""
        # Intentar cargar desde cache
        cached_model = self.optimizer.get_cached_model(model_path)
        if cached_model is not None:
            return cached_model
        
        # Cargar modelo
        model = joblib.load(model_path)
        
        # Cachear modelo
        self.optimizer.cache_model(model_path, model)
        
        return model
    
    def optimize_feature_extraction(self, data: pd.DataFrame, features: List[str]) -> np.ndarray:
        """Optimizar extracción de características"""
        # Crear clave de cache
        cache_key = f"features_{hash(str(data.shape) + str(features))}"
        
        # Intentar obtener del cache
        cached_features = self.optimizer.get_cached_result(cache_key)
        if cached_features is not None:
            return cached_features
        
        # Extraer características
        feature_matrix = data[features].values
        
        # Cachear características
        self.optimizer.cache_result(cache_key, feature_matrix, ttl=1800)  # 30 minutos
        
        return feature_matrix
    
    def optimize_scaling(self, data: np.ndarray, scaler_name: str) -> np.ndarray:
        """Optimizar escalado de datos"""
        # Intentar obtener scaler cacheado
        if scaler_name in self.optimizer.scaler_cache:
            scaler = self.optimizer.scaler_cache[scaler_name]
        else:
            scaler = StandardScaler()
            scaler.fit(data)
            self.optimizer.scaler_cache[scaler_name] = scaler
        
        return scaler.transform(data)
    
    def optimize_prediction_pipeline(self, model: Any, data: pd.DataFrame, 
                                   features: List[str], scaler_name: str = None) -> np.ndarray:
        """Optimizar pipeline completo de predicción"""
        # Extraer características
        feature_matrix = self.optimize_feature_extraction(data, features)
        
        # Escalar si es necesario
        if scaler_name:
            feature_matrix = self.optimize_scaling(feature_matrix, scaler_name)
        
        # Optimizar inferencia
        predictions = self.optimizer.optimize_inference(model, feature_matrix)
        
        return predictions
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de optimización"""
        return {
            'performance': self.optimizer.get_performance_stats(),
            'cache': self.optimizer.get_cache_stats(),
            'memory': {
                'usage_percent': psutil.virtual_memory().percent,
                'available_gb': psutil.virtual_memory().available / (1024**3)
            }
        }

# Instancia global del optimizador
performance_optimizer = PerformanceOptimizer()
model_optimizer = ModelOptimizer() 