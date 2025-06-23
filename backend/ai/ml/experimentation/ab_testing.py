"""
Sistema de A/B Testing para experimentación con modelos de AI.

Este módulo implementa:
- Diseño de experimentos A/B
- Asignación aleatoria de usuarios
- Análisis estadístico de resultados
- Detección de significancia estadística
- Monitoreo en tiempo real
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import json
import logging
import hashlib
from dataclasses import dataclass, asdict
from enum import Enum
import random

logger = logging.getLogger('ai.abtesting')

class ExperimentStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"

class MetricType(Enum):
    CONTINUOUS = "continuous"  # Revenue, time spent, etc.
    BINARY = "binary"          # Conversion, click, etc.
    COUNT = "count"           # Number of actions, etc.

@dataclass
class ExperimentConfig:
    """Configuración de un experimento A/B"""
    experiment_id: str
    name: str
    description: str
    start_date: datetime
    end_date: datetime
    traffic_split: Dict[str, float]  # {'A': 0.5, 'B': 0.5}
    primary_metric: str
    secondary_metrics: List[str]
    sample_size: int
    confidence_level: float = 0.95
    power: float = 0.8
    min_detectable_effect: float = 0.05

@dataclass
class ExperimentResult:
    """Resultado de un experimento A/B"""
    experiment_id: str
    variant: str
    sample_size: int
    metric_value: float
    confidence_interval: Tuple[float, float]
    p_value: float
    effect_size: float
    timestamp: datetime

class ABTesting:
    def __init__(self):
        """Inicializa el sistema de A/B testing"""
        self.experiments = {}
        self.results = {}
        self.user_assignments = {}
        self.metrics_history = []
        
    def create_experiment(self, config: ExperimentConfig) -> str:
        """
        Crea un nuevo experimento A/B.
        
        Args:
            config: Configuración del experimento
            
        Returns:
            str: ID del experimento creado
        """
        try:
            # Validar configuración
            self._validate_experiment_config(config)
            
            # Generar ID único si no se proporciona
            if not config.experiment_id:
                config.experiment_id = self._generate_experiment_id(config.name)
            
            # Inicializar experimento
            self.experiments[config.experiment_id] = {
                'config': config,
                'status': ExperimentStatus.DRAFT,
                'created_at': datetime.now(),
                'updated_at': datetime.now(),
                'results': {},
                'user_assignments': {},
                'metrics_history': []
            }
            
            logger.info(f"Experimento {config.experiment_id} creado: {config.name}")
            return config.experiment_id
            
        except Exception as e:
            logger.error(f"Error creando experimento: {str(e)}")
            raise
    
    def _validate_experiment_config(self, config: ExperimentConfig):
        """Valida la configuración del experimento"""
        # Validar fechas
        if config.start_date >= config.end_date:
            raise ValueError("La fecha de inicio debe ser anterior a la fecha de fin")
        
        # Validar split de tráfico
        total_split = sum(config.traffic_split.values())
        if abs(total_split - 1.0) > 0.01:
            raise ValueError(f"El split de tráfico debe sumar 1.0, actual: {total_split}")
        
        # Validar métricas
        if not config.primary_metric:
            raise ValueError("Debe especificar una métrica primaria")
        
        # Validar tamaño de muestra
        if config.sample_size < 100:
            raise ValueError("El tamaño de muestra mínimo es 100")
    
    def _generate_experiment_id(self, name: str) -> str:
        """Genera un ID único para el experimento"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        return f"exp_{timestamp}_{name_hash}"
    
    def start_experiment(self, experiment_id: str) -> bool:
        """
        Inicia un experimento A/B.
        
        Args:
            experiment_id: ID del experimento
            
        Returns:
            bool: True si se inició exitosamente
        """
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experimento {experiment_id} no encontrado")
            
            experiment = self.experiments[experiment_id]
            config = experiment['config']
            
            # Verificar que la fecha de inicio haya llegado
            if datetime.now() < config.start_date:
                logger.warning(f"Experimento {experiment_id} no puede iniciar antes de {config.start_date}")
                return False
            
            # Cambiar estado a activo
            experiment['status'] = ExperimentStatus.ACTIVE
            experiment['updated_at'] = datetime.now()
            
            logger.info(f"Experimento {experiment_id} iniciado")
            return True
            
        except Exception as e:
            logger.error(f"Error iniciando experimento {experiment_id}: {str(e)}")
            return False
    
    def assign_user_to_variant(self, experiment_id: str, user_id: str) -> str:
        """
        Asigna un usuario a una variante del experimento.
        
        Args:
            experiment_id: ID del experimento
            user_id: ID del usuario
            
        Returns:
            str: Variante asignada (A, B, C, etc.)
        """
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experimento {experiment_id} no encontrado")
            
            experiment = self.experiments[experiment_id]
            
            # Verificar si el experimento está activo
            if experiment['status'] != ExperimentStatus.ACTIVE:
                return 'control'  # Asignar a control si no está activo
            
            # Verificar si el usuario ya fue asignado
            user_key = f"{experiment_id}_{user_id}"
            if user_key in self.user_assignments:
                return self.user_assignments[user_key]
            
            # Asignar variante usando hash consistente
            config = experiment['config']
            user_hash = hashlib.md5(f"{user_id}_{experiment_id}".encode()).hexdigest()
            user_hash_int = int(user_hash, 16)
            
            # Asignar basado en el split de tráfico
            cumulative_prob = 0
            for variant, probability in config.traffic_split.items():
                cumulative_prob += probability
                if user_hash_int / (2**64) <= cumulative_prob:
                    assigned_variant = variant
                    break
            else:
                assigned_variant = list(config.traffic_split.keys())[-1]
            
            # Guardar asignación
            self.user_assignments[user_key] = assigned_variant
            experiment['user_assignments'][user_key] = assigned_variant
            
            return assigned_variant
            
        except Exception as e:
            logger.error(f"Error asignando usuario {user_id} a experimento {experiment_id}: {str(e)}")
            return 'control'
    
    def record_metric(self, experiment_id: str, user_id: str, 
                     metric_name: str, value: float, 
                     metric_type: MetricType = MetricType.CONTINUOUS) -> bool:
        """
        Registra una métrica para un usuario en un experimento.
        
        Args:
            experiment_id: ID del experimento
            user_id: ID del usuario
            metric_name: Nombre de la métrica
            value: Valor de la métrica
            metric_type: Tipo de métrica
            
        Returns:
            bool: True si se registró exitosamente
        """
        try:
            if experiment_id not in self.experiments:
                return False
            
            # Obtener variante asignada
            user_key = f"{experiment_id}_{user_id}"
            if user_key not in self.user_assignments:
                return False
            
            variant = self.user_assignments[user_key]
            
            # Crear entrada de métrica
            metric_entry = {
                'experiment_id': experiment_id,
                'user_id': user_id,
                'variant': variant,
                'metric_name': metric_name,
                'value': value,
                'metric_type': metric_type.value,
                'timestamp': datetime.now()
            }
            
            # Guardar en historial
            self.metrics_history.append(metric_entry)
            
            # Actualizar resultados del experimento
            experiment = self.experiments[experiment_id]
            if metric_name not in experiment['results']:
                experiment['results'][metric_name] = {}
            
            if variant not in experiment['results'][metric_name]:
                experiment['results'][metric_name][variant] = []
            
            experiment['results'][metric_name][variant].append(value)
            
            return True
            
        except Exception as e:
            logger.error(f"Error registrando métrica: {str(e)}")
            return False
    
    def analyze_experiment(self, experiment_id: str, 
                          metric_name: str = None) -> Dict[str, Any]:
        """
        Analiza los resultados de un experimento A/B.
        
        Args:
            experiment_id: ID del experimento
            metric_name: Nombre de la métrica a analizar (si None, usa la primaria)
            
        Returns:
            Dict con resultados del análisis
        """
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experimento {experiment_id} no encontrado")
            
            experiment = self.experiments[experiment_id]
            config = experiment['config']
            
            # Usar métrica primaria si no se especifica
            if metric_name is None:
                metric_name = config.primary_metric
            
            if metric_name not in experiment['results']:
                raise ValueError(f"Métrica {metric_name} no encontrada en experimento {experiment_id}")
            
            results = experiment['results'][metric_name]
            
            # Verificar que hay suficientes datos
            min_sample_size = 30  # Mínimo para análisis estadístico
            for variant, values in results.items():
                if len(values) < min_sample_size:
                    logger.warning(f"Variante {variant} tiene solo {len(values)} muestras")
            
            # Realizar análisis estadístico
            analysis_results = self._perform_statistical_analysis(results, metric_name)
            
            # Agregar información del experimento
            analysis_results.update({
                'experiment_id': experiment_id,
                'experiment_name': config.name,
                'metric_name': metric_name,
                'status': experiment['status'].value,
                'total_users': len(experiment['user_assignments']),
                'analysis_timestamp': datetime.now().isoformat()
            })
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analizando experimento {experiment_id}: {str(e)}")
            raise
    
    def _perform_statistical_analysis(self, results: Dict[str, List[float]], 
                                    metric_name: str) -> Dict[str, Any]:
        """
        Realiza análisis estadístico de los resultados.
        
        Args:
            results: Diccionario con resultados por variante
            metric_name: Nombre de la métrica
            
        Returns:
            Dict con resultados del análisis estadístico
        """
        variants = list(results.keys())
        if len(variants) < 2:
            return {'error': 'Se requieren al menos 2 variantes para análisis'}
        
        # Determinar tipo de métrica basado en los datos
        metric_type = self._infer_metric_type(results)
        
        analysis = {
            'metric_type': metric_type.value,
            'variants': {},
            'comparisons': {},
            'overall_significance': False
        }
        
        # Calcular estadísticas por variante
        for variant, values in results.items():
            values_array = np.array(values)
            
            variant_stats = {
                'sample_size': len(values),
                'mean': np.mean(values_array),
                'std': np.std(values_array, ddof=1),
                'median': np.median(values_array),
                'min': np.min(values_array),
                'max': np.max(values_array)
            }
            
            # Calcular intervalo de confianza
            if len(values) > 1:
                ci = stats.t.interval(0.95, len(values)-1, 
                                    loc=variant_stats['mean'], 
                                    scale=stats.sem(values_array))
                variant_stats['confidence_interval'] = (ci[0], ci[1])
            else:
                variant_stats['confidence_interval'] = (variant_stats['mean'], variant_stats['mean'])
            
            analysis['variants'][variant] = variant_stats
        
        # Comparar variantes
        if metric_type == MetricType.CONTINUOUS:
            # Test t para métricas continuas
            for i in range(len(variants)):
                for j in range(i+1, len(variants)):
                    var1, var2 = variants[i], variants[j]
                    
                    # Test t independiente
                    t_stat, p_value = ttest_ind(results[var1], results[var2])
                    
                    # Test no paramétrico (Mann-Whitney U)
                    u_stat, u_p_value = mannwhitneyu(results[var1], results[var2], 
                                                    alternative='two-sided')
                    
                    # Calcular tamaño del efecto (Cohen's d)
                    pooled_std = np.sqrt(((len(results[var1])-1)*np.var(results[var1], ddof=1) + 
                                        (len(results[var2])-1)*np.var(results[var2], ddof=1)) / 
                                       (len(results[var1]) + len(results[var2]) - 2))
                    cohens_d = (np.mean(results[var1]) - np.mean(results[var2])) / pooled_std
                    
                    comparison_key = f"{var1}_vs_{var2}"
                    analysis['comparisons'][comparison_key] = {
                        't_statistic': t_stat,
                        'p_value_t_test': p_value,
                        'u_statistic': u_stat,
                        'p_value_mann_whitney': u_p_value,
                        'cohens_d': cohens_d,
                        'significant': p_value < 0.05,
                        'effect_size': self._interpret_effect_size(cohens_d)
                    }
        
        elif metric_type == MetricType.BINARY:
            # Test chi-cuadrado para métricas binarias
            for i in range(len(variants)):
                for j in range(i+1, len(variants)):
                    var1, var2 = variants[i], variants[j]
                    
                    # Crear tabla de contingencia
                    success1 = sum(1 for x in results[var1] if x > 0)
                    success2 = sum(1 for x in results[var2] if x > 0)
                    total1, total2 = len(results[var1]), len(results[var2])
                    
                    contingency_table = [[success1, total1-success1], 
                                       [success2, total2-success2]]
                    
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    
                    # Calcular odds ratio
                    odds_ratio = (success1 * (total2-success2)) / (success2 * (total1-success1))
                    
                    comparison_key = f"{var1}_vs_{var2}"
                    analysis['comparisons'][comparison_key] = {
                        'chi2_statistic': chi2,
                        'p_value': p_value,
                        'odds_ratio': odds_ratio,
                        'significant': p_value < 0.05
                    }
        
        # Determinar significancia general
        significant_comparisons = sum(1 for comp in analysis['comparisons'].values() 
                                    if comp.get('significant', False))
        analysis['overall_significance'] = significant_comparisons > 0
        
        return analysis
    
    def _infer_metric_type(self, results: Dict[str, List[float]]) -> MetricType:
        """Infere el tipo de métrica basado en los datos"""
        all_values = []
        for values in results.values():
            all_values.extend(values)
        
        # Si todos los valores son 0 o 1, es binaria
        unique_values = set(all_values)
        if unique_values.issubset({0, 1}):
            return MetricType.BINARY
        
        # Si hay valores enteros no negativos, es count
        if all(isinstance(x, (int, float)) and x >= 0 for x in all_values):
            if all(x == int(x) for x in all_values):
                return MetricType.COUNT
        
        # Por defecto, es continua
        return MetricType.CONTINUOUS
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpreta el tamaño del efecto según Cohen's d"""
        if abs(cohens_d) < 0.2:
            return "negligible"
        elif abs(cohens_d) < 0.5:
            return "small"
        elif abs(cohens_d) < 0.8:
            return "medium"
        else:
            return "large"
    
    def stop_experiment(self, experiment_id: str) -> bool:
        """
        Detiene un experimento A/B.
        
        Args:
            experiment_id: ID del experimento
            
        Returns:
            bool: True si se detuvo exitosamente
        """
        try:
            if experiment_id not in self.experiments:
                return False
            
            experiment = self.experiments[experiment_id]
            experiment['status'] = ExperimentStatus.STOPPED
            experiment['updated_at'] = datetime.now()
            
            logger.info(f"Experimento {experiment_id} detenido")
            return True
            
        except Exception as e:
            logger.error(f"Error deteniendo experimento {experiment_id}: {str(e)}")
            return False
    
    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """
        Obtiene un resumen del experimento.
        
        Args:
            experiment_id: ID del experimento
            
        Returns:
            Dict con resumen del experimento
        """
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experimento {experiment_id} no encontrado")
            
            experiment = self.experiments[experiment_id]
            config = experiment['config']
            
            summary = {
                'experiment_id': experiment_id,
                'name': config.name,
                'description': config.description,
                'status': experiment['status'].value,
                'start_date': config.start_date.isoformat(),
                'end_date': config.end_date.isoformat(),
                'traffic_split': config.traffic_split,
                'primary_metric': config.primary_metric,
                'total_users': len(experiment['user_assignments']),
                'created_at': experiment['created_at'].isoformat(),
                'updated_at': experiment['updated_at'].isoformat(),
                'metrics_recorded': list(experiment['results'].keys())
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error obteniendo resumen del experimento {experiment_id}: {str(e)}")
            raise
    
    def export_experiment_data(self, experiment_id: str) -> Dict[str, Any]:
        """
        Exporta todos los datos de un experimento.
        
        Args:
            experiment_id: ID del experimento
            
        Returns:
            Dict con todos los datos del experimento
        """
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experimento {experiment_id} no encontrado")
            
            experiment = self.experiments[experiment_id]
            
            # Filtrar métricas del experimento
            experiment_metrics = [
                metric for metric in self.metrics_history 
                if metric['experiment_id'] == experiment_id
            ]
            
            export_data = {
                'experiment': asdict(experiment['config']),
                'status': experiment['status'].value,
                'user_assignments': experiment['user_assignments'],
                'results': experiment['results'],
                'metrics_history': experiment_metrics,
                'created_at': experiment['created_at'].isoformat(),
                'updated_at': experiment['updated_at'].isoformat()
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exportando datos del experimento {experiment_id}: {str(e)}")
            raise
    
    def calculate_sample_size(self, baseline_conversion: float, 
                            min_detectable_effect: float,
                            confidence_level: float = 0.95,
                            power: float = 0.8) -> int:
        """
        Calcula el tamaño de muestra necesario para un experimento.
        
        Args:
            baseline_conversion: Tasa de conversión base
            min_detectable_effect: Efecto mínimo detectable
            confidence_level: Nivel de confianza
            power: Potencia estadística
            
        Returns:
            int: Tamaño de muestra por variante
        """
        try:
            # Usar fórmula para test de proporciones
            alpha = 1 - confidence_level
            beta = 1 - power
            
            # Valores críticos
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = stats.norm.ppf(1 - beta)
            
            # Proporciones
            p1 = baseline_conversion
            p2 = baseline_conversion + min_detectable_effect
            
            # Varianza combinada
            p_combined = (p1 + p2) / 2
            var_combined = p_combined * (1 - p_combined)
            
            # Tamaño de muestra
            n = (2 * var_combined * (z_alpha + z_beta)**2) / (min_detectable_effect**2)
            
            return int(np.ceil(n))
            
        except Exception as e:
            logger.error(f"Error calculando tamaño de muestra: {str(e)}")
            return 1000  # Valor por defecto 