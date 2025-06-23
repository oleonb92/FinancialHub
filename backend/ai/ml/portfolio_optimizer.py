"""
Sistema de Optimización de Portafolio para Fintech.

Este módulo implementa un sistema avanzado de optimización de portafolio que incluye:
- Optimización de Markowitz (Mean-Variance)
- Optimización de Black-Litterman
- Gestión de riesgo con VaR y CVaR
- Rebalanceo automático
- Análisis de correlaciones
- Optimización multi-objetivo
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    cp = None
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from django.utils import timezone
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None
import json

logger = logging.getLogger('ai.portfolio_optimizer')

class PortfolioOptimizer:
    """
    Sistema principal de optimización de portafolio.
    
    Características:
    - Optimización de Markowitz
    - Gestión de riesgo avanzada
    - Rebalanceo automático
    - Análisis de correlaciones
    - Optimización multi-objetivo
    - Backtesting de estrategias
    """
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% anual
        self.optimization_method = 'markowitz'
        self.rebalance_frequency = 'monthly'
        self.max_position_size = 0.25  # 25% máximo por activo
        self.min_position_size = 0.01  # 1% mínimo por activo
        
        # Configuración de riesgo
        self.risk_config = {
            'var_confidence': 0.95,
            'max_var': 0.05,  # 5% máximo VaR
            'max_volatility': 0.20,  # 20% máximo volatilidad
            'target_return': 0.08,  # 8% retorno objetivo
            'risk_aversion': 2.0
        }
        
        # Historial de optimizaciones
        self.optimization_history = []
        
    def fetch_portfolio_data(self, symbols: List[str], period: str = '2y') -> pd.DataFrame:
        """
        Obtiene datos históricos para optimización de portafolio.
        
        Args:
            symbols: Lista de símbolos de activos
            period: Período de datos
            
        Returns:
            DataFrame: Datos de precios ajustados
        """
        if not YFINANCE_AVAILABLE:
            logger.warning("yfinance not available. Portfolio data fetching disabled.")
            return pd.DataFrame()
            
        try:
            data = {}
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                if not hist.empty:
                    data[symbol] = hist['Adj Close']
            
            if not data:
                logger.error("No data fetched for any symbol")
                return pd.DataFrame()
            
            # Crear DataFrame con todos los activos
            portfolio_data = pd.DataFrame(data)
            portfolio_data = portfolio_data.dropna()
            
            logger.info(f"Fetched data for {len(portfolio_data.columns)} assets")
            return portfolio_data
            
        except Exception as e:
            logger.error(f"Error fetching portfolio data: {str(e)}")
            return pd.DataFrame()
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula retornos logarítmicos.
        
        Args:
            prices: DataFrame de precios
            
        Returns:
            DataFrame: Retornos logarítmicos
        """
        try:
            returns = np.log(prices / prices.shift(1))
            return returns.dropna()
        except Exception as e:
            logger.error(f"Error calculating returns: {str(e)}")
            return pd.DataFrame()
    
    def calculate_portfolio_metrics(self, returns: pd.DataFrame, weights: np.array) -> Dict[str, float]:
        """
        Calcula métricas del portafolio.
        
        Args:
            returns: DataFrame de retornos
            weights: Vector de pesos
            
        Returns:
            dict: Métricas del portafolio
        """
        try:
            # Retorno esperado
            expected_returns = returns.mean()
            portfolio_return = np.sum(expected_returns * weights)
            
            # Volatilidad
            covariance_matrix = returns.cov()
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            
            # Sharpe Ratio
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            # VaR
            portfolio_returns = np.dot(returns, weights)
            var_95 = np.percentile(portfolio_returns, (1 - self.risk_config['var_confidence']) * 100)
            
            # CVaR (Expected Shortfall)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
            
            # Beta (asumiendo mercado como primer activo)
            if len(weights) > 1:
                market_returns = returns.iloc[:, 0]
                portfolio_returns_series = pd.Series(portfolio_returns, index=returns.index)
                beta = np.cov(portfolio_returns_series, market_returns)[0, 1] / np.var(market_returns)
            else:
                beta = 1.0
            
            return {
                'expected_return': float(portfolio_return),
                'volatility': float(portfolio_volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'var_95': float(var_95),
                'cvar_95': float(cvar_95),
                'beta': float(beta),
                'max_drawdown': float(self._calculate_max_drawdown(portfolio_returns))
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return {}
    
    def _calculate_max_drawdown(self, returns: np.array) -> float:
        """
        Calcula el máximo drawdown.
        
        Args:
            returns: Array de retornos
            
        Returns:
            float: Máximo drawdown
        """
        try:
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return float(np.min(drawdown))
        except:
            return 0.0
    
    def optimize_markowitz(self, returns: pd.DataFrame, target_return: float = None) -> Dict[str, Any]:
        """
        Optimización de portafolio usando teoría de Markowitz.
        
        Args:
            returns: DataFrame de retornos
            target_return: Retorno objetivo (opcional)
            
        Returns:
            dict: Resultados de optimización
        """
        try:
            n_assets = len(returns.columns)
            expected_returns = returns.mean().values
            covariance_matrix = returns.cov().values
            
            # Función objetivo: minimizar varianza
            def objective(weights):
                return np.dot(weights.T, np.dot(covariance_matrix, weights))
            
            # Restricciones
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Suma de pesos = 1
            ]
            
            if target_return:
                constraints.append({
                    'type': 'eq', 
                    'fun': lambda x: np.sum(expected_returns * x) - target_return
                })
            
            # Límites de posición
            bounds = [(self.min_position_size, self.max_position_size) for _ in range(n_assets)]
            
            # Punto inicial (igual peso)
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Optimización
            result = minimize(
                objective, 
                initial_weights, 
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimal_weights = result.x
                portfolio_metrics = self.calculate_portfolio_metrics(returns, optimal_weights)
                
                return {
                    'success': True,
                    'weights': optimal_weights.tolist(),
                    'assets': returns.columns.tolist(),
                    'metrics': portfolio_metrics,
                    'optimization_info': {
                        'method': 'markowitz',
                        'target_return': target_return,
                        'iterations': result.nit,
                        'status': result.message
                    }
                }
            else:
                return {
                    'success': False,
                    'error': result.message
                }
                
        except Exception as e:
            logger.error(f"Error in Markowitz optimization: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def optimize_black_litterman(self, returns: pd.DataFrame, views: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Optimización usando modelo Black-Litterman.
        
        Args:
            returns: DataFrame de retornos
            views: Diccionario con vistas del inversor {asset: expected_return}
            
        Returns:
            dict: Resultados de optimización
        """
        try:
            n_assets = len(returns.columns)
            
            # Retornos de equilibrio (CAPM)
            market_cap_weights = np.array([1/n_assets] * n_assets)  # Simplificado
            covariance_matrix = returns.cov().values
            
            # Retorno de equilibrio
            risk_aversion = self.risk_config['risk_aversion']
            equilibrium_returns = risk_aversion * np.dot(covariance_matrix, market_cap_weights)
            
            # Matriz de vistas (si se proporcionan)
            if views:
                P = np.zeros((len(views), n_assets))
                Q = np.zeros(len(views))
                
                for i, (asset, view_return) in enumerate(views.items()):
                    if asset in returns.columns:
                        asset_idx = returns.columns.get_loc(asset)
                        P[i, asset_idx] = 1
                        Q[i] = view_return
                
                # Matriz de confianza en vistas
                omega = np.eye(len(views)) * 0.1  # 10% de confianza
                
                # Combinar retornos de equilibrio con vistas
                tau = 0.05  # Factor de escala
                pi = equilibrium_returns
                
                # Retornos combinados
                M1 = np.linalg.inv(tau * covariance_matrix)
                M2 = np.dot(P.T, np.dot(np.linalg.inv(omega), P))
                M3 = np.dot(P.T, np.dot(np.linalg.inv(omega), Q))
                
                combined_returns = np.linalg.inv(M1 + M2).dot(M1.dot(pi) + M3)
                
                # Matriz de covarianza combinada
                combined_covariance = np.linalg.inv(M1 + M2)
            else:
                combined_returns = equilibrium_returns
                combined_covariance = covariance_matrix
            
            # Optimización con retornos combinados
            def objective(weights):
                return np.dot(weights.T, np.dot(combined_covariance, weights))
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            
            bounds = [(self.min_position_size, self.max_position_size) for _ in range(n_assets)]
            initial_weights = np.array([1/n_assets] * n_assets)
            
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimal_weights = result.x
                portfolio_metrics = self.calculate_portfolio_metrics(returns, optimal_weights)
                
                return {
                    'success': True,
                    'weights': optimal_weights.tolist(),
                    'assets': returns.columns.tolist(),
                    'metrics': portfolio_metrics,
                    'equilibrium_returns': equilibrium_returns.tolist(),
                    'combined_returns': combined_returns.tolist() if views else None,
                    'optimization_info': {
                        'method': 'black_litterman',
                        'views': views,
                        'iterations': result.nit,
                        'status': result.message
                    }
                }
            else:
                return {
                    'success': False,
                    'error': result.message
                }
                
        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def optimize_risk_parity(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Optimización de paridad de riesgo.
        
        Args:
            returns: DataFrame de retornos
            
        Returns:
            dict: Resultados de optimización
        """
        try:
            n_assets = len(returns.columns)
            covariance_matrix = returns.cov().values
            
            # Función objetivo: minimizar la diferencia entre contribuciones de riesgo
            def objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
                risk_contributions = weights * (np.dot(covariance_matrix, weights)) / portfolio_vol
                return np.sum((risk_contributions - np.mean(risk_contributions))**2)
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            
            bounds = [(self.min_position_size, self.max_position_size) for _ in range(n_assets)]
            initial_weights = np.array([1/n_assets] * n_assets)
            
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimal_weights = result.x
                portfolio_metrics = self.calculate_portfolio_metrics(returns, optimal_weights)
                
                # Calcular contribuciones de riesgo
                portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights)))
                risk_contributions = optimal_weights * (np.dot(covariance_matrix, optimal_weights)) / portfolio_vol
                
                return {
                    'success': True,
                    'weights': optimal_weights.tolist(),
                    'assets': returns.columns.tolist(),
                    'metrics': portfolio_metrics,
                    'risk_contributions': risk_contributions.tolist(),
                    'optimization_info': {
                        'method': 'risk_parity',
                        'iterations': result.nit,
                        'status': result.message
                    }
                }
            else:
                return {
                    'success': False,
                    'error': result.message
                }
                
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def optimize_portfolio(self, symbols: List[str], method: str = 'markowitz', 
                         target_return: float = None, views: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Optimiza portafolio usando el método especificado.
        
        Args:
            symbols: Lista de símbolos de activos
            method: Método de optimización ('markowitz', 'black_litterman', 'risk_parity')
            target_return: Retorno objetivo
            views: Vistas para Black-Litterman
            
        Returns:
            dict: Resultados de optimización
        """
        try:
            # Obtener datos
            portfolio_data = self.fetch_portfolio_data(symbols)
            if portfolio_data.empty:
                return {
                    'success': False,
                    'error': 'No data available for optimization'
                }
            
            # Calcular retornos
            returns = self.calculate_returns(portfolio_data)
            if returns.empty:
                return {
                    'success': False,
                    'error': 'Unable to calculate returns'
                }
            
            # Optimización según método
            if method == 'markowitz':
                result = self.optimize_markowitz(returns, target_return)
            elif method == 'black_litterman':
                result = self.optimize_black_litterman(returns, views)
            elif method == 'risk_parity':
                result = self.optimize_risk_parity(returns)
            else:
                return {
                    'success': False,
                    'error': f'Unknown optimization method: {method}'
                }
            
            if result['success']:
                # Agregar información adicional
                result['optimization_date'] = timezone.now().isoformat()
                result['data_period'] = f"{returns.index[0].date()} to {returns.index[-1].date()}"
                result['n_assets'] = len(symbols)
                
                # Guardar en historial
                self.optimization_history.append({
                    'date': result['optimization_date'],
                    'method': method,
                    'symbols': symbols,
                    'metrics': result['metrics']
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def rebalance_portfolio(self, current_weights: Dict[str, float], 
                          target_weights: Dict[str, float], 
                          portfolio_value: float) -> Dict[str, Any]:
        """
        Calcula las transacciones necesarias para rebalancear el portafolio.
        
        Args:
            current_weights: Pesos actuales
            target_weights: Pesos objetivo
            portfolio_value: Valor total del portafolio
            
        Returns:
            dict: Transacciones de rebalanceo
        """
        try:
            transactions = []
            total_trades = 0
            
            for asset in set(current_weights.keys()) | set(target_weights.keys()):
                current_weight = current_weights.get(asset, 0)
                target_weight = target_weights.get(asset, 0)
                
                current_value = current_weight * portfolio_value
                target_value = target_weight * portfolio_value
                trade_value = target_value - current_value
                
                if abs(trade_value) > portfolio_value * 0.001:  # Mínimo 0.1% del portafolio
                    transactions.append({
                        'asset': asset,
                        'action': 'buy' if trade_value > 0 else 'sell',
                        'value': abs(trade_value),
                        'weight_change': target_weight - current_weight
                    })
                    total_trades += abs(trade_value)
            
            return {
                'transactions': transactions,
                'total_trades_value': total_trades,
                'turnover_ratio': total_trades / portfolio_value,
                'n_transactions': len(transactions)
            }
            
        except Exception as e:
            logger.error(f"Error calculating rebalancing: {str(e)}")
            return {
                'error': str(e),
                'transactions': []
            }
    
    def backtest_strategy(self, symbols: List[str], method: str = 'markowitz', 
                         rebalance_frequency: str = 'monthly', 
                         start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        Realiza backtesting de una estrategia de optimización.
        
        Args:
            symbols: Lista de símbolos
            method: Método de optimización
            rebalance_frequency: Frecuencia de rebalanceo
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            dict: Resultados del backtesting
        """
        try:
            # Obtener datos históricos
            portfolio_data = self.fetch_portfolio_data(symbols, '5y')
            if portfolio_data.empty:
                return {
                    'success': False,
                    'error': 'No data available for backtesting'
                }
            
            # Filtrar por fechas si se especifican
            if start_date:
                portfolio_data = portfolio_data[start_date:]
            if end_date:
                portfolio_data = portfolio_data[:end_date]
            
            returns = self.calculate_returns(portfolio_data)
            
            # Definir fechas de rebalanceo
            if rebalance_frequency == 'monthly':
                rebalance_dates = returns.resample('M').last().index
            elif rebalance_frequency == 'quarterly':
                rebalance_dates = returns.resample('Q').last().index
            else:
                rebalance_dates = returns.resample('W').last().index
            
            # Inicializar variables
            portfolio_values = [1000]  # $1000 inicial
            weights_history = []
            current_weights = None
            
            for i, date in enumerate(rebalance_dates[:-1]):
                # Datos hasta la fecha de rebalanceo
                historical_returns = returns[:date]
                
                if len(historical_returns) < 60:  # Mínimo 60 días de datos
                    continue
                
                # Optimizar portafolio
                if method == 'markowitz':
                    opt_result = self.optimize_markowitz(historical_returns)
                elif method == 'risk_parity':
                    opt_result = self.optimize_risk_parity(historical_returns)
                else:
                    opt_result = self.optimize_markowitz(historical_returns)
                
                if not opt_result['success']:
                    continue
                
                # Obtener pesos óptimos
                optimal_weights = np.array(opt_result['weights'])
                weights_history.append({
                    'date': date.isoformat(),
                    'weights': optimal_weights.tolist()
                })
                
                # Calcular retornos del período
                period_returns = returns[date:rebalance_dates[i+1]]
                if not period_returns.empty:
                    if current_weights is not None:
                        # Retornos del portafolio
                        portfolio_period_returns = np.dot(period_returns, current_weights)
                        cumulative_return = np.prod(1 + portfolio_period_returns)
                        portfolio_values.append(portfolio_values[-1] * cumulative_return)
                    else:
                        portfolio_values.append(portfolio_values[-1])
                
                current_weights = optimal_weights
            
            # Calcular métricas de performance
            portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            volatility = np.std(portfolio_returns) * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            
            return {
                'success': True,
                'method': method,
                'symbols': symbols,
                'total_return': float(total_return),
                'annualized_return': float(annualized_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'final_value': float(portfolio_values[-1]),
                'weights_history': weights_history,
                'portfolio_values': portfolio_values,
                'backtest_period': {
                    'start': portfolio_data.index[0].isoformat(),
                    'end': portfolio_data.index[-1].isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in backtesting: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """
        Obtiene el historial de optimizaciones.
        
        Returns:
            list: Historial de optimizaciones
        """
        return self.optimization_history
    
    def get_correlation_matrix(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Calcula la matriz de correlación entre activos.
        
        Args:
            symbols: Lista de símbolos
            
        Returns:
            dict: Matriz de correlación y análisis
        """
        try:
            portfolio_data = self.fetch_portfolio_data(symbols)
            if portfolio_data.empty:
                return {'error': 'No data available'}
            
            returns = self.calculate_returns(portfolio_data)
            correlation_matrix = returns.corr()
            
            # Encontrar correlaciones más altas y bajas
            corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    asset1 = correlation_matrix.columns[i]
                    asset2 = correlation_matrix.columns[j]
                    corr_value = correlation_matrix.iloc[i, j]
                    corr_pairs.append({
                        'asset1': asset1,
                        'asset2': asset2,
                        'correlation': float(corr_value)
                    })
            
            # Ordenar por correlación
            corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'highest_correlations': corr_pairs[:5],
                'lowest_correlations': corr_pairs[-5:],
                'average_correlation': float(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()),
                'assets': symbols
            }
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {str(e)}")
            return {'error': str(e)} 