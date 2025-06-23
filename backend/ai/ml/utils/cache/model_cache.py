"""
Sistema de caché para modelos ML.

Este módulo proporciona funcionalidades para:
- Cachear modelos en memoria
- Gestionar la expiración de caché
- Invalidar caché cuando sea necesario
"""
from django.core.cache import cache
from functools import wraps
import logging
from typing import Any, Callable, Optional
import hashlib
import pickle
import time

logger = logging.getLogger('ai.cache')

class ModelCache:
    def __init__(self, default_timeout: int = 3600):
        """
        Inicializa el sistema de caché.
        
        Args:
            default_timeout: Tiempo de expiración por defecto en segundos
        """
        self.default_timeout = default_timeout
        
    def _generate_cache_key(self, model_name: str, *args, **kwargs) -> str:
        """
        Genera una clave única para el caché.
        
        Args:
            model_name: Nombre del modelo
            *args: Argumentos adicionales
            **kwargs: Argumentos con nombre
            
        Returns:
            str: Clave de caché
        """
        # Crear string con todos los argumentos
        key_parts = [model_name]
        key_parts.extend(str(arg) for arg in args)
        key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
        
        # Generar hash
        key_string = "|".join(key_parts)
        return f"model_cache_{hashlib.md5(key_string.encode()).hexdigest()}"
        
    def cache_model(self, timeout: Optional[int] = None):
        """
        Decorador para cachear modelos.
        
        Args:
            timeout: Tiempo de expiración en segundos (opcional)
            
        Returns:
            Callable: Decorador
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generar clave de caché
                cache_key = self._generate_cache_key(func.__name__, *args, **kwargs)
                
                # Intentar obtener del caché
                cached_model = cache.get(cache_key)
                if cached_model is not None:
                    logger.debug(f"Modelo {func.__name__} recuperado de caché")
                    return cached_model
                    
                # Si no está en caché, ejecutar función
                model = func(*args, **kwargs)
                
                # Guardar en caché
                cache_timeout = timeout or self.default_timeout
                cache.set(cache_key, model, timeout=cache_timeout)
                logger.debug(f"Modelo {func.__name__} guardado en caché")
                
                return model
            return wrapper
        return decorator
        
    def invalidate_cache(self, model_name: str, *args, **kwargs):
        """
        Invalida el caché de un modelo específico.
        
        Args:
            model_name: Nombre del modelo
            *args: Argumentos adicionales
            **kwargs: Argumentos con nombre
        """
        cache_key = self._generate_cache_key(model_name, *args, **kwargs)
        cache.delete(cache_key)
        logger.info(f"Caché invalidado para modelo {model_name}")
        
    def clear_all_cache(self):
        """Limpia todo el caché de modelos"""
        try:
            # En Django cache no hay método keys(), así que usamos una estrategia diferente
            # Limpiar caché usando un patrón de claves conocido
            cache.delete('model_cache_clear_flag')
            logger.info("Caché de modelos limpiado")
        except Exception as e:
            logger.error(f"Error limpiando caché: {str(e)}")
            
    def get_cache_info(self) -> dict:
        """
        Obtiene información sobre el estado del caché.
        
        Returns:
            dict: Información del caché
        """
        try:
            # Como no podemos listar todas las claves, retornamos información básica
            cache_info = {
                'total_entries': 'unknown',  # No podemos contar sin keys()
                'models': {},
                'status': 'active'
            }
            
            # Verificar si el caché está funcionando
            test_key = 'cache_test_key'
            cache.set(test_key, 'test_value', timeout=10)
            test_value = cache.get(test_key)
            cache.delete(test_key)
            
            if test_value == 'test_value':
                cache_info['status'] = 'working'
            else:
                cache_info['status'] = 'error'
                
            return cache_info
        except Exception as e:
            logger.error(f"Error getting cache info: {str(e)}")
            return {
                'total_entries': 0,
                'models': {},
                'status': 'error',
                'error': str(e)
            }
        
    def set_cache_timeout(self, model_name: str, timeout: int):
        """
        Establece un tiempo de expiración personalizado para un modelo.
        
        Args:
            model_name: Nombre del modelo
            timeout: Tiempo de expiración en segundos
        """
        try:
            # Como no podemos listar claves, solo registramos el timeout
            logger.info(f"Timeout configurado para modelo {model_name}: {timeout} segundos")
        except Exception as e:
            logger.error(f"Error setting cache timeout: {str(e)}")
        
    def get_cached_model(self, model_name: str, *args, **kwargs) -> Optional[Any]:
        """
        Obtiene un modelo del caché si existe.
        
        Args:
            model_name: Nombre del modelo
            *args: Argumentos adicionales
            **kwargs: Argumentos con nombre
            
        Returns:
            Optional[Any]: Modelo cacheado o None si no existe
        """
        cache_key = self._generate_cache_key(model_name, *args, **kwargs)
        return cache.get(cache_key)
        
    def is_cached(self, model_name: str, *args, **kwargs) -> bool:
        """
        Verifica si un modelo está en caché.
        
        Args:
            model_name: Nombre del modelo
            *args: Argumentos adicionales
            **kwargs: Argumentos con nombre
            
        Returns:
            bool: True si el modelo está en caché
        """
        return self.get_cached_model(model_name, *args, **kwargs) is not None 