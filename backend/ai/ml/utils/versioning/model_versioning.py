"""
Sistema de versionado de modelos ML.

Este módulo proporciona funcionalidades para:
- Guardar y cargar versiones de modelos
- Gestionar el historial de versiones
- Realizar rollback a versiones anteriores
"""
import os
import json
import joblib
from datetime import datetime
from django.conf import settings
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger('ai.model_versioning')

class ModelVersioning:
    def __init__(self):
        self.models_dir = getattr(settings, 'ML_MODELS_DIR', 'ml_models')
        self.version_file = os.path.join(self.models_dir, 'versions.json')
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Asegura que los directorios necesarios existan"""
        os.makedirs(self.models_dir, exist_ok=True)
        if not os.path.exists(self.version_file):
            self._save_version_registry({})
            
    def _load_version_registry(self) -> Dict:
        """Carga el registro de versiones"""
        try:
            with open(self.version_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
            
    def _save_version_registry(self, registry: Dict):
        """Guarda el registro de versiones"""
        with open(self.version_file, 'w') as f:
            json.dump(registry, f, indent=2)
            
    def _update_version_registry(self, model_name: str, version: str, metrics: Dict[str, Any]):
        """Actualiza el registro de versiones"""
        registry = self._load_version_registry()
        if model_name not in registry:
            registry[model_name] = {}
            
        registry[model_name][version] = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        self._save_version_registry(registry)
        
    def _get_latest_version(self, model_name: str) -> Optional[str]:
        """Obtiene la última versión de un modelo"""
        registry = self._load_version_registry()
        if model_name not in registry or not registry[model_name]:
            return None
            
        return max(registry[model_name].keys())
        
    def save_model_version(self, model_name: str, model: Any, metrics: Dict[str, Any]) -> str:
        """
        Guarda una nueva versión del modelo.
        
        Args:
            model_name: Nombre del modelo
            model: Instancia del modelo a guardar
            metrics: Métricas del modelo
            
        Returns:
            str: Versión guardada
        """
        try:
            # Crear directorio de versiones si no existe
            version_dir = os.path.join(self.models_dir, model_name, 'versions')
            os.makedirs(version_dir, exist_ok=True)
            
            # Generar nombre de versión
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = os.path.join(version_dir, f'model_{version}.joblib')
            
            # Guardar modelo
            joblib.dump(model, model_path)
            
            # Actualizar registro de versiones
            self._update_version_registry(model_name, version, metrics)
            
            logger.info(f"Modelo {model_name} versión {version} guardado exitosamente")
            return version
            
        except Exception as e:
            logger.error(f"Error guardando versión del modelo {model_name}: {str(e)}")
            raise
            
    def load_model_version(self, model_name: str, version: Optional[str] = None) -> Any:
        """
        Carga una versión específica del modelo.
        
        Args:
            model_name: Nombre del modelo
            version: Versión a cargar (opcional, por defecto la última)
            
        Returns:
            Any: Modelo cargado
        """
        try:
            if version is None:
                version = self._get_latest_version(model_name)
                if version is None:
                    raise ValueError(f"No hay versiones disponibles para el modelo {model_name}")
                    
            model_path = os.path.join(
                self.models_dir, 
                model_name, 
                'versions', 
                f'model_{version}.joblib'
            )
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No se encontró la versión {version} del modelo {model_name}")
                
            model = joblib.load(model_path)
            logger.info(f"Modelo {model_name} versión {version} cargado exitosamente")
            return model
            
        except Exception as e:
            logger.error(f"Error cargando versión del modelo {model_name}: {str(e)}")
            raise
            
    def rollback_model(self, model_name: str, version: str) -> str:
        """
        Realiza rollback a una versión anterior.
        
        Args:
            model_name: Nombre del modelo
            version: Versión a la que hacer rollback
            
        Returns:
            str: Nueva versión creada
        """
        try:
            # Cargar versión anterior
            model = self.load_model_version(model_name, version)
            
            # Guardar como versión actual
            current_version = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.save_model_version(
                model_name, 
                model, 
                {
                    'rollback_to': version,
                    'rollback_timestamp': datetime.now().isoformat()
                }
            )
            
            logger.info(f"Rollback del modelo {model_name} a la versión {version} completado")
            return current_version
            
        except Exception as e:
            logger.error(f"Error en rollback del modelo {model_name}: {str(e)}")
            raise
            
    def get_model_versions(self, model_name: str) -> Dict[str, Dict]:
        """
        Obtiene el historial de versiones de un modelo.
        
        Args:
            model_name: Nombre del modelo
            
        Returns:
            Dict: Historial de versiones
        """
        registry = self._load_version_registry()
        return registry.get(model_name, {})
        
    def delete_model_version(self, model_name: str, version: str):
        """
        Elimina una versión específica del modelo.
        
        Args:
            model_name: Nombre del modelo
            version: Versión a eliminar
        """
        try:
            # Eliminar archivo del modelo
            model_path = os.path.join(
                self.models_dir, 
                model_name, 
                'versions', 
                f'model_{version}.joblib'
            )
            if os.path.exists(model_path):
                os.remove(model_path)
                
            # Actualizar registro
            registry = self._load_version_registry()
            if model_name in registry and version in registry[model_name]:
                del registry[model_name][version]
                self._save_version_registry(registry)
                
            logger.info(f"Versión {version} del modelo {model_name} eliminada exitosamente")
            
        except Exception as e:
            logger.error(f"Error eliminando versión del modelo {model_name}: {str(e)}")
            raise 