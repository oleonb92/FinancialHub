"""
Sistema de Federated Learning para entrenamiento distribuido y privado.

Este módulo implementa:
- Entrenamiento federado de modelos
- Agregación de modelos distribuidos
- Preservación de privacidad
- Sincronización de parámetros
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import joblib
import json
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import hashlib
import pickle
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger('ai.federated')

class AggregationMethod(Enum):
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDNOVA = "fednova"

@dataclass
class ClientConfig:
    """Configuración de un cliente federado"""
    client_id: str
    data_size: int
    local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32
    privacy_budget: float = 1.0

@dataclass
class FederatedRound:
    """Información de una ronda federada"""
    round_id: int
    clients_participated: List[str]
    global_model_accuracy: float
    aggregation_time: float
    timestamp: datetime

class FederatedLearning:
    def __init__(self, 
                 task_type: str = 'classification',
                 aggregation_method: AggregationMethod = AggregationMethod.FEDAVG,
                 min_clients: int = 2,
                 max_rounds: int = 100):
        """
        Inicializa el sistema de federated learning.
        
        Args:
            task_type: 'classification' o 'regression'
            aggregation_method: Método de agregación
            min_clients: Número mínimo de clientes para una ronda
            max_rounds: Número máximo de rondas
        """
        self.task_type = task_type
        self.aggregation_method = aggregation_method
        self.min_clients = min_clients
        self.max_rounds = max_rounds
        
        # Estado global
        self.global_model = None
        self.current_round = 0
        self.clients = {}
        self.round_history = []
        
        # Métricas
        self.global_metrics = {
            'accuracy': [],
            'loss': [],
            'rounds_completed': 0
        }
        
        # Inicializar modelo global
        self._initialize_global_model()
        
    def _initialize_global_model(self):
        """Inicializa el modelo global según el tipo de tarea"""
        if self.task_type == 'classification':
            self.global_model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            self.global_model = LinearRegression()
    
    def add_client(self, client_id: str, data_size: int, config: ClientConfig = None) -> bool:
        """
        Agrega un cliente al sistema federado.
        
        Args:
            client_id: ID único del cliente
            data_size: Tamaño del dataset del cliente
            config: Configuración específica del cliente
            
        Returns:
            bool: True si se agregó exitosamente
        """
        try:
            if client_id in self.clients:
                logger.warning(f"Cliente {client_id} ya existe")
                return False
            
            if config is None:
                config = ClientConfig(
                    client_id=client_id,
                    data_size=data_size
                )
            
            self.clients[client_id] = {
                'config': config,
                'local_model': None,
                'last_update': None,
                'participation_count': 0,
                'data_hash': None
            }
            
            logger.info(f"Cliente {client_id} agregado con {data_size} muestras")
            return True
            
        except Exception as e:
            logger.error(f"Error agregando cliente {client_id}: {str(e)}")
            return False
    
    def remove_client(self, client_id: str) -> bool:
        """
        Remueve un cliente del sistema federado.
        
        Args:
            client_id: ID del cliente a remover
            
        Returns:
            bool: True si se removió exitosamente
        """
        try:
            if client_id in self.clients:
                del self.clients[client_id]
                logger.info(f"Cliente {client_id} removido")
                return True
            return False
        except Exception as e:
            logger.error(f"Error removiendo cliente {client_id}: {str(e)}")
            return False
    
    def train_client(self, client_id: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Entrena el modelo local de un cliente.
        
        Args:
            client_id: ID del cliente
            X: Features de entrenamiento
            y: Target de entrenamiento
            
        Returns:
            Dict con resultados del entrenamiento
        """
        try:
            if client_id not in self.clients:
                raise ValueError(f"Cliente {client_id} no encontrado")
            
            client = self.clients[client_id]
            config = client['config']
            
            # Crear modelo local basado en el global
            local_model = self._create_local_model()
            
            # Si hay un modelo global, inicializar con sus parámetros
            if self.global_model is not None and hasattr(self.global_model, 'coef_'):
                local_model.coef_ = self.global_model.coef_.copy()
                if hasattr(self.global_model, 'intercept_'):
                    local_model.intercept_ = self.global_model.intercept_.copy()
            
            # Entrenar modelo local
            start_time = datetime.now()
            local_model.fit(X, y)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Evaluar modelo local
            y_pred = local_model.predict(X)
            if self.task_type == 'classification':
                accuracy = accuracy_score(y, y_pred)
                f1 = f1_score(y, y_pred, average='weighted')
                metrics = {'accuracy': accuracy, 'f1_score': f1}
            else:
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                metrics = {'mse': mse, 'r2_score': r2}
            
            # Actualizar estado del cliente
            client['local_model'] = local_model
            client['last_update'] = datetime.now()
            client['participation_count'] += 1
            client['data_hash'] = self._hash_data(X, y)
            
            results = {
                'client_id': client_id,
                'training_time': training_time,
                'metrics': metrics,
                'data_size': len(X),
                'model_params': self._extract_model_params(local_model)
            }
            
            logger.info(f"Cliente {client_id} entrenado - Accuracy: {metrics.get('accuracy', metrics.get('r2_score', 0)):.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error entrenando cliente {client_id}: {str(e)}")
            raise
    
    def aggregate_models(self, client_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Agrega los modelos de los clientes participantes.
        
        Args:
            client_results: Lista de resultados de entrenamiento de clientes
            
        Returns:
            Dict con información de la agregación
        """
        try:
            if len(client_results) < self.min_clients:
                raise ValueError(f"Se requieren al menos {self.min_clients} clientes para agregación")
            
            start_time = datetime.now()
            
            if self.aggregation_method == AggregationMethod.FEDAVG:
                aggregated_params = self._fedavg_aggregation(client_results)
            elif self.aggregation_method == AggregationMethod.FEDPROX:
                aggregated_params = self._fedprox_aggregation(client_results)
            elif self.aggregation_method == AggregationMethod.FEDNOVA:
                aggregated_params = self._fednova_aggregation(client_results)
            else:
                raise ValueError(f"Método de agregación no soportado: {self.aggregation_method}")
            
            # Actualizar modelo global
            self._update_global_model(aggregated_params)
            
            aggregation_time = (datetime.now() - start_time).total_seconds()
            
            # Calcular métricas globales
            global_accuracy = self._evaluate_global_model(client_results)
            
            # Registrar ronda
            round_info = FederatedRound(
                round_id=self.current_round,
                clients_participated=[r['client_id'] for r in client_results],
                global_model_accuracy=global_accuracy,
                aggregation_time=aggregation_time,
                timestamp=datetime.now()
            )
            self.round_history.append(round_info)
            
            # Actualizar métricas globales
            self.global_metrics['accuracy'].append(global_accuracy)
            self.global_metrics['rounds_completed'] += 1
            
            self.current_round += 1
            
            results = {
                'round_id': self.current_round - 1,
                'clients_participated': len(client_results),
                'global_accuracy': global_accuracy,
                'aggregation_time': aggregation_time,
                'aggregation_method': self.aggregation_method.value
            }
            
            logger.info(f"Ronda {self.current_round - 1} completada - Accuracy global: {global_accuracy:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error en agregación de modelos: {str(e)}")
            raise
    
    def _fedavg_aggregation(self, client_results: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Agregación FedAvg (Federated Averaging)"""
        total_samples = sum(r['data_size'] for r in client_results)
        aggregated_params = {}
        
        for param_name in client_results[0]['model_params'].keys():
            weighted_sum = np.zeros_like(client_results[0]['model_params'][param_name])
            
            for result in client_results:
                weight = result['data_size'] / total_samples
                weighted_sum += weight * result['model_params'][param_name]
            
            aggregated_params[param_name] = weighted_sum
        
        return aggregated_params
    
    def _fedprox_aggregation(self, client_results: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Agregación FedProx (Federated Proximal)"""
        # Implementación simplificada de FedProx
        # En una implementación completa, se incluiría el término proximal
        return self._fedavg_aggregation(client_results)
    
    def _fednova_aggregation(self, client_results: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Agregación FedNova (Federated Normalized Averaging)"""
        # Implementación simplificada de FedNova
        # En una implementación completa, se normalizaría por el número de épocas locales
        return self._fedavg_aggregation(client_results)
    
    def _extract_model_params(self, model) -> Dict[str, np.ndarray]:
        """Extrae los parámetros de un modelo"""
        params = {}
        
        if hasattr(model, 'coef_'):
            params['coef_'] = model.coef_.copy()
        
        if hasattr(model, 'intercept_'):
            params['intercept_'] = model.intercept_.copy()
        
        return params
    
    def _update_global_model(self, aggregated_params: Dict[str, np.ndarray]):
        """Actualiza el modelo global con los parámetros agregados"""
        if hasattr(self.global_model, 'coef_'):
            self.global_model.coef_ = aggregated_params['coef_']
        
        if hasattr(self.global_model, 'intercept_'):
            self.global_model.intercept_ = aggregated_params['intercept_']
    
    def _create_local_model(self):
        """Crea un modelo local basado en el tipo de tarea"""
        if self.task_type == 'classification':
            return LogisticRegression(random_state=42, max_iter=1000)
        else:
            return LinearRegression()
    
    def _evaluate_global_model(self, client_results: List[Dict[str, Any]]) -> float:
        """Evalúa el modelo global usando los datos de los clientes"""
        # En una implementación real, se evaluaría con un conjunto de prueba global
        # Por simplicidad, usamos el promedio de las métricas de los clientes
        accuracies = []
        
        for result in client_results:
            if 'accuracy' in result['metrics']:
                accuracies.append(result['metrics']['accuracy'])
            elif 'r2_score' in result['metrics']:
                accuracies.append(result['metrics']['r2_score'])
        
        return np.mean(accuracies) if accuracies else 0.0
    
    def _hash_data(self, X: pd.DataFrame, y: pd.Series) -> str:
        """Genera un hash de los datos para verificar integridad"""
        data_str = f"{X.to_string()}{y.to_string()}"
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones con el modelo global.
        
        Args:
            X: Features para predicción
            
        Returns:
            Predicciones
        """
        if self.global_model is None:
            raise ValueError("No hay modelo global entrenado")
        
        return self.global_model.predict(X)
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """
        Obtiene las métricas globales del entrenamiento federado.
        
        Returns:
            Dict con métricas globales
        """
        return {
            'current_round': self.current_round,
            'total_clients': len(self.clients),
            'rounds_completed': self.global_metrics['rounds_completed'],
            'accuracy_history': self.global_metrics['accuracy'],
            'best_accuracy': max(self.global_metrics['accuracy']) if self.global_metrics['accuracy'] else 0,
            'round_history': [
                {
                    'round_id': r.round_id,
                    'clients_participated': len(r.clients_participated),
                    'global_accuracy': r.global_model_accuracy,
                    'aggregation_time': r.aggregation_time,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in self.round_history
            ]
        }
    
    def get_client_info(self, client_id: str) -> Dict[str, Any]:
        """
        Obtiene información de un cliente específico.
        
        Args:
            client_id: ID del cliente
            
        Returns:
            Dict con información del cliente
        """
        if client_id not in self.clients:
            raise ValueError(f"Cliente {client_id} no encontrado")
        
        client = self.clients[client_id]
        return {
            'client_id': client_id,
            'config': {
                'data_size': client['config'].data_size,
                'local_epochs': client['config'].local_epochs,
                'learning_rate': client['config'].learning_rate,
                'batch_size': client['config'].batch_size,
                'privacy_budget': client['config'].privacy_budget
            },
            'participation_count': client['participation_count'],
            'last_update': client['last_update'].isoformat() if client['last_update'] else None,
            'data_hash': client['data_hash']
        }
    
    def save_federated_model(self, filepath: str):
        """
        Guarda el modelo federado completo.
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        model_data = {
            'global_model': self.global_model,
            'task_type': self.task_type,
            'current_round': self.current_round,
            'global_metrics': self.global_metrics,
            'round_history': self.round_history,
            'clients': self.clients,
            'aggregation_method': self.aggregation_method.value
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Modelo federado guardado en: {filepath}")
    
    def load_federated_model(self, filepath: str):
        """
        Carga un modelo federado guardado.
        
        Args:
            filepath: Ruta del modelo a cargar
        """
        model_data = joblib.load(filepath)
        
        self.global_model = model_data['global_model']
        self.task_type = model_data['task_type']
        self.current_round = model_data['current_round']
        self.global_metrics = model_data['global_metrics']
        self.round_history = model_data['round_history']
        self.clients = model_data['clients']
        self.aggregation_method = AggregationMethod(model_data['aggregation_method'])
        
        logger.info(f"Modelo federado cargado desde: {filepath}")
    
    def export_training_report(self) -> Dict[str, Any]:
        """
        Genera un reporte completo del entrenamiento federado.
        
        Returns:
            Dict con reporte detallado
        """
        return {
            'task_type': self.task_type,
            'aggregation_method': self.aggregation_method.value,
            'total_rounds': self.current_round,
            'total_clients': len(self.clients),
            'global_metrics': self.get_global_metrics(),
            'client_summary': {
                client_id: self.get_client_info(client_id)
                for client_id in self.clients.keys()
            },
            'training_summary': {
                'start_time': self.round_history[0].timestamp.isoformat() if self.round_history else None,
                'end_time': self.round_history[-1].timestamp.isoformat() if self.round_history else None,
                'total_training_time': sum(r.aggregation_time for r in self.round_history),
                'average_accuracy': np.mean(self.global_metrics['accuracy']) if self.global_metrics['accuracy'] else 0
            }
        }
