"""
Sistema de AutoML avanzado para optimización automática de modelos.

Este módulo proporciona:
- Optimización automática de hiperparámetros
- Selección automática de algoritmos
- Feature engineering automático
- Ensamblaje de modelos
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import joblib
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger('ai.automl')

class AutoMLOptimizer:
    def __init__(self, task_type: str = 'classification', max_time: int = 3600):
        """
        Inicializa el optimizador de AutoML.
        
        Args:
            task_type: 'classification' o 'regression'
            max_time: Tiempo máximo de optimización en segundos
        """
        self.task_type = task_type
        self.max_time = max_time
        self.best_model = None
        self.best_score = 0
        self.best_params = {}
        self.optimization_history = []
        
        # Definir modelos base según el tipo de tarea
        if task_type == 'classification':
            self.models = {
                'random_forest': RandomForestClassifier(random_state=42),
                'gradient_boosting': GradientBoostingClassifier(random_state=42),
                'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
                'svm': SVC(random_state=42, probability=True),
                'neural_network': MLPClassifier(random_state=42, max_iter=1000)
            }
            self.scoring = 'f1_weighted'
        else:
            self.models = {
                'random_forest': RandomForestRegressor(random_state=42),
                'gradient_boosting': GradientBoostingRegressor(random_state=42),
                'linear_regression': LinearRegression(),
                'ridge': Ridge(random_state=42),
                'lasso': Lasso(random_state=42),
                'svr': SVR(),
                'neural_network': MLPRegressor(random_state=42, max_iter=1000)
            }
            self.scoring = 'r2'
            
        # Definir espacios de hiperparámetros
        self.param_spaces = self._define_param_spaces()
        
    def _define_param_spaces(self) -> Dict[str, Dict]:
        """Define los espacios de hiperparámetros para cada modelo"""
        if self.task_type == 'classification':
            return {
                'random_forest': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'gradient_boosting': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'logistic_regression': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                },
                'svm': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                },
                'neural_network': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        else:
            return {
                'random_forest': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                },
                'gradient_boosting': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                'ridge': {
                    'alpha': [0.1, 1, 10, 100]
                },
                'lasso': {
                    'alpha': [0.1, 1, 10, 100]
                },
                'svr': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                },
                'neural_network': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'alpha': [0.0001, 0.001, 0.01]
                }
            }
    
    def optimize(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, Any]:
        """
        Optimiza automáticamente el mejor modelo y hiperparámetros.
        
        Args:
            X: Features
            y: Target
            cv: Número de folds para cross-validation
            
        Returns:
            Dict con información del mejor modelo
        """
        logger.info(f"Iniciando optimización de AutoML para tarea: {self.task_type}")
        start_time = datetime.now()
        
        # Preprocesamiento automático
        X_processed = self._auto_preprocess(X)
        
        best_overall_score = 0
        best_overall_model = None
        best_overall_params = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Optimizando modelo: {model_name}")
            
            try:
                # Optimización de hiperparámetros
                param_space = self.param_spaces.get(model_name, {})
                
                if param_space:
                    # Usar RandomizedSearchCV para mayor eficiencia
                    search = RandomizedSearchCV(
                        model,
                        param_space,
                        n_iter=20,  # Número de iteraciones para búsqueda aleatoria
                        cv=cv,
                        scoring=self.scoring,
                        n_jobs=-1,
                        random_state=42,
                        verbose=0
                    )
                    
                    search.fit(X_processed, y)
                    best_score = search.best_score_
                    best_params = search.best_params_
                    best_model = search.best_estimator_
                else:
                    # Sin optimización de hiperparámetros
                    scores = cross_val_score(model, X_processed, y, cv=cv, scoring=self.scoring)
                    best_score = scores.mean()
                    best_params = {}
                    best_model = model
                
                # Registrar resultados
                self.optimization_history.append({
                    'model_name': model_name,
                    'score': best_score,
                    'params': best_params,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Actualizar mejor modelo global
                if best_score > best_overall_score:
                    best_overall_score = best_score
                    best_overall_model = best_model
                    best_overall_params = best_params
                    
                logger.info(f"Modelo {model_name}: Score = {best_score:.4f}")
                
            except Exception as e:
                logger.error(f"Error optimizando {model_name}: {str(e)}")
                continue
        
        # Guardar mejor modelo
        self.best_model = best_overall_model
        self.best_score = best_overall_score
        self.best_params = best_overall_params
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        results = {
            'best_model_name': type(best_overall_model).__name__,
            'best_score': best_overall_score,
            'best_params': best_overall_params,
            'optimization_time': optimization_time,
            'models_evaluated': len(self.optimization_history),
            'optimization_history': self.optimization_history
        }
        
        logger.info(f"Optimización completada. Mejor score: {best_overall_score:.4f}")
        return results
    
    def _auto_preprocess(self, X: pd.DataFrame) -> np.ndarray:
        """
        Preprocesamiento automático de features.
        
        Args:
            X: DataFrame con features
            
        Returns:
            Array preprocesado
        """
        # Detectar tipos de columnas
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        # Para simplicidad, solo usar columnas numéricas por ahora
        # En una implementación completa, se incluiría encoding de categóricas
        X_numeric = X[numeric_cols].fillna(0)
        
        # Escalado automático
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numeric)
        
        return X_scaled
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones con el mejor modelo.
        
        Args:
            X: Features para predicción
            
        Returns:
            Predicciones
        """
        if self.best_model is None:
            raise ValueError("No hay modelo optimizado. Ejecuta optimize() primero.")
        
        X_processed = self._auto_preprocess(X)
        return self.best_model.predict(X_processed)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones de probabilidad (solo para clasificación).
        
        Args:
            X: Features para predicción
            
        Returns:
            Probabilidades de predicción
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba solo disponible para clasificación")
        
        if self.best_model is None:
            raise ValueError("No hay modelo optimizado. Ejecuta optimize() primero.")
        
        X_processed = self._auto_preprocess(X)
        
        if hasattr(self.best_model, 'predict_proba'):
            return self.best_model.predict_proba(X_processed)
        else:
            raise ValueError("El modelo no soporta predict_proba")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Obtiene la importancia de features del mejor modelo.
        
        Returns:
            Dict con importancia de features
        """
        if self.best_model is None:
            raise ValueError("No hay modelo optimizado. Ejecuta optimize() primero.")
        
        if hasattr(self.best_model, 'feature_importances_'):
            # Para modelos basados en árboles
            feature_names = [f"feature_{i}" for i in range(len(self.best_model.feature_importances_))]
            return dict(zip(feature_names, self.best_model.feature_importances_))
        elif hasattr(self.best_model, 'coef_'):
            # Para modelos lineales
            feature_names = [f"feature_{i}" for i in range(len(self.best_model.coef_))]
            return dict(zip(feature_names, np.abs(self.best_model.coef_)))
        else:
            return {}
    
    def save_model(self, filepath: str):
        """
        Guarda el mejor modelo optimizado.
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        if self.best_model is None:
            raise ValueError("No hay modelo optimizado para guardar")
        
        model_data = {
            'model': self.best_model,
            'task_type': self.task_type,
            'best_score': self.best_score,
            'best_params': self.best_params,
            'optimization_history': self.optimization_history
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Modelo guardado en: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Carga un modelo optimizado guardado.
        
        Args:
            filepath: Ruta del modelo a cargar
        """
        model_data = joblib.load(filepath)
        
        self.best_model = model_data['model']
        self.task_type = model_data['task_type']
        self.best_score = model_data['best_score']
        self.best_params = model_data['best_params']
        self.optimization_history = model_data['optimization_history']
        
        logger.info(f"Modelo cargado desde: {filepath}")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """
        Genera un reporte detallado de la optimización.
        
        Returns:
            Dict con reporte de optimización
        """
        return {
            'task_type': self.task_type,
            'best_score': self.best_score,
            'best_params': self.best_params,
            'models_evaluated': len(self.optimization_history),
            'optimization_history': self.optimization_history,
            'feature_importance': self.get_feature_importance()
        } 