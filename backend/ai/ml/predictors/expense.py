"""
Expense predictor for forecasting future expenses.
"""
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from ..base import BaseMLModel
from django.db.models import Q
from transactions.models import Transaction
from datetime import timedelta

class ExpensePredictor(BaseMLModel):
    """
    Predictor for forecasting future expenses based on historical transaction data.
    """
    
    def __init__(self):
        super().__init__('expense_predictor')
        self.scaler = StandardScaler()
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    
    def _prepare_features(self, transactions):
        """
        Prepare features for training or prediction.
        
        Args:
            transactions: List of Transaction objects
            
        Returns:
            pd.DataFrame: Prepared features
        """
        features = pd.DataFrame({
            'day_of_week': [t.date.weekday() for t in transactions],
            'day_of_month': [t.date.day for t in transactions],
            'month': [t.date.month for t in transactions],
            'category_id': [t.category.id for t in transactions],
            'amount': [float(t.amount) for t in transactions]
        })
        
        return features
    
    def _prepare_sequence_features(self, transactions, sequence_length=30):
        """
        Prepare sequence features for time series prediction.
        
        Args:
            transactions: List of Transaction objects
            sequence_length: Number of days to look back
            
        Returns:
            pd.DataFrame: Prepared sequence features
        """
        # Group transactions by date
        daily_amounts = pd.DataFrame({
            'date': [t.date for t in transactions],
            'amount': [float(t.amount) for t in transactions]
        }).groupby('date')['amount'].sum().reset_index()
        
        # Create sequence features
        sequences = []
        for i in range(len(daily_amounts) - sequence_length):
            sequence = daily_amounts.iloc[i:i+sequence_length]
            target = daily_amounts.iloc[i+sequence_length]['amount']
            sequences.append({
                'sequence': sequence['amount'].values,
                'target': target
            })
        
        return pd.DataFrame(sequences)
    
    def train(self, transactions):
        """
        Train the expense predictor.
        
        Args:
            transactions: List of Transaction objects to train on
        """
        try:
            # Prepare features
            X = self._prepare_features(transactions)
            y = X['amount']
            X = X.drop('amount', axis=1)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            # Save the trained model
            self.save()
            
            self.logger.info(f"Model trained on {len(transactions)} transactions")
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, date, category_id):
        """
        Predict expenses for a given date and category.
        
        Args:
            date: Date to predict for
            category_id: Category ID to predict for
            
        Returns:
            float: Predicted expense amount
        """
        try:
            # Prepare features
            features = pd.DataFrame({
                'day_of_week': [date.weekday()],
                'day_of_month': [date.day],
                'month': [date.month],
                'category_id': [category_id]
            })
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            
            return max(0, prediction)  # Ensure non-negative prediction
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def predict_sequence(self, start_date, days=30):
        """
        Predict expenses for a sequence of days.
        
        Args:
            start_date: Start date for prediction
            days: Number of days to predict
            
        Returns:
            pd.DataFrame: Predictions for each day
        """
        try:
            predictions = []
            current_date = start_date
            
            for _ in range(days):
                # Get all categories
                categories = Transaction.objects.values_list('category_id', flat=True).distinct()
                
                # Predict for each category
                daily_total = 0
                for category_id in categories:
                    prediction = self.predict(current_date, category_id)
                    daily_total += prediction
                
                predictions.append({
                    'date': current_date,
                    'predicted_amount': daily_total
                })
                
                current_date += timedelta(days=1)
            
            return pd.DataFrame(predictions)
            
        except Exception as e:
            self.logger.error(f"Error making sequence prediction: {str(e)}")
            raise 