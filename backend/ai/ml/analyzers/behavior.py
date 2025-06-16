"""
Behavior analyzer for identifying spending patterns and anomalies.
"""
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from ..base import BaseMLModel
from django.db.models import Q
from transactions.models import Transaction
from datetime import timedelta

class BehaviorAnalyzer(BaseMLModel):
    """
    Analyzer for identifying spending patterns and anomalies in transaction data.
    """
    
    def __init__(self):
        super().__init__('behavior_analyzer')
        self.scaler = StandardScaler()
        self.clustering_model = DBSCAN(
            eps=0.5,
            min_samples=5,
            metric='euclidean'
        )
    
    def _prepare_features(self, transactions):
        """
        Prepare features for behavior analysis.
        
        Args:
            transactions: List of Transaction objects
            
        Returns:
            pd.DataFrame: Prepared features
        """
        features = pd.DataFrame({
            'amount': [float(t.amount) for t in transactions],
            'day_of_week': [t.date.weekday() for t in transactions],
            'hour': [t.date.hour for t in transactions],
            'category_id': [t.category.id for t in transactions],
            'merchant_id': [hash(t.merchant) % 1000 if t.merchant else 0 for t in transactions]
        })
        
        return features
    
    def _detect_anomalies(self, features):
        """
        Detect anomalies in transaction data.
        
        Args:
            features: DataFrame of transaction features
            
        Returns:
            np.array: Boolean array indicating anomalies
        """
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Fit clustering model
        clusters = self.clustering_model.fit_predict(scaled_features)
        
        # Mark points in cluster -1 as anomalies
        return clusters == -1
    
    def analyze_spending_patterns(self, transactions):
        """
        Analyze spending patterns in transaction data.
        
        Args:
            transactions: List of Transaction objects
            
        Returns:
            dict: Dictionary containing pattern analysis results
        """
        try:
            # Prepare features
            features = self._prepare_features(transactions)
            
            # Detect anomalies
            anomalies = self._detect_anomalies(features)
            
            # Analyze patterns by category
            category_patterns = {}
            for category_id in features['category_id'].unique():
                category_transactions = transactions[features['category_id'] == category_id]
                category_features = features[features['category_id'] == category_id]
                
                pattern = {
                    'total_spent': float(category_features['amount'].sum()),
                    'avg_amount': float(category_features['amount'].mean()),
                    'frequency': len(category_transactions),
                    'anomalies': int(anomalies[features['category_id'] == category_id].sum()),
                    'preferred_days': self._get_preferred_days(category_features),
                    'preferred_hours': self._get_preferred_hours(category_features)
                }
                
                category_patterns[category_id] = pattern
            
            # Analyze overall patterns
            overall_patterns = {
                'total_transactions': len(transactions),
                'total_spent': float(features['amount'].sum()),
                'avg_transaction': float(features['amount'].mean()),
                'anomalies': int(anomalies.sum()),
                'spending_trend': self._analyze_spending_trend(transactions),
                'category_distribution': self._analyze_category_distribution(features)
            }
            
            return {
                'category_patterns': category_patterns,
                'overall_patterns': overall_patterns
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing spending patterns: {str(e)}")
            raise
    
    def _get_preferred_days(self, features):
        """
        Get preferred days of the week for transactions.
        
        Args:
            features: DataFrame of transaction features
            
        Returns:
            dict: Dictionary of day frequencies
        """
        day_counts = features['day_of_week'].value_counts()
        return {
            day: int(count)
            for day, count in day_counts.items()
        }
    
    def _get_preferred_hours(self, features):
        """
        Get preferred hours of the day for transactions.
        
        Args:
            features: DataFrame of transaction features
            
        Returns:
            dict: Dictionary of hour frequencies
        """
        hour_counts = features['hour'].value_counts()
        return {
            hour: int(count)
            for hour, count in hour_counts.items()
        }
    
    def _analyze_spending_trend(self, transactions):
        """
        Analyze spending trend over time.
        
        Args:
            transactions: List of Transaction objects
            
        Returns:
            dict: Dictionary containing trend analysis
        """
        # Group transactions by date
        daily_amounts = pd.DataFrame({
            'date': [t.date for t in transactions],
            'amount': [float(t.amount) for t in transactions]
        }).groupby('date')['amount'].sum().reset_index()
        
        # Calculate trend
        if len(daily_amounts) > 1:
            trend = np.polyfit(
                range(len(daily_amounts)),
                daily_amounts['amount'],
                deg=1
            )[0]
        else:
            trend = 0
        
        return {
            'trend_coefficient': float(trend),
            'trend_direction': 'increasing' if trend > 0 else 'decreasing',
            'daily_average': float(daily_amounts['amount'].mean())
        }
    
    def _analyze_category_distribution(self, features):
        """
        Analyze distribution of spending across categories.
        
        Args:
            features: DataFrame of transaction features
            
        Returns:
            dict: Dictionary containing category distribution
        """
        category_totals = features.groupby('category_id')['amount'].sum()
        total_spent = category_totals.sum()
        
        return {
            category_id: float(amount / total_spent)
            for category_id, amount in category_totals.items()
        } 