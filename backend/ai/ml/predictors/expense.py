"""
Expense predictor for forecasting future expenses.

This predictor uses a GradientBoostingRegressor and a StandardScaler for feature scaling.
Both the model and the scaler are saved and loaded together using joblib to ensure correct predictions after loading.
"""
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from ai.ml.base import BaseMLModel
from django.db.models import Q
from transactions.models import Transaction
from datetime import timedelta
import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

class ExpensePredictor(BaseMLModel):
    """
    Predictor for forecasting future expenses based on historical transaction data.

    Features used:
    - day_of_week
    - day_of_month
    - month
    - category_id

    The model is trained using a GradientBoostingRegressor.
    Both the model and the scaler are saved and loaded together to preserve the fitted state.
    """
    
    def __init__(self):
        super().__init__('expense_predictor')
        self.feature_names = ['day_of_week', 'day_of_month', 'month', 'category_id']
        self.is_fitted = False
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ))
        ])
    
    def _prepare_features(self, transactions: Union[List[Transaction], List[Dict]]) -> pd.DataFrame:
        """
        Prepare features for training or prediction.
        
        Args:
            transactions: List of Transaction objects or dictionaries
            
        Returns:
            pd.DataFrame: Prepared features
            
        Raises:
            ValueError: If transaction data is invalid
        """
        try:
            # Check if we're dealing with dictionaries or Transaction objects
            if transactions and isinstance(transactions[0], dict):
                # Handle dictionary format
                features = pd.DataFrame({
                    'day_of_week': [t.get('date').weekday() if hasattr(t.get('date'), 'weekday') else 0 for t in transactions],
                    'day_of_month': [t.get('date').day if hasattr(t.get('date'), 'day') else 1 for t in transactions],
                    'month': [t.get('date').month if hasattr(t.get('date'), 'month') else 1 for t in transactions],
                    'category_id': [t.get('category_id', 0) for t in transactions],
                    'amount': [float(t.get('amount', 0)) for t in transactions]
                })
            else:
                # Handle Transaction objects
                features = pd.DataFrame({
                    'day_of_week': [t.date.weekday() for t in transactions],
                    'day_of_month': [t.date.day for t in transactions],
                    'month': [t.date.month for t in transactions],
                    'category_id': [t.category.id for t in transactions],
                    'amount': [float(t.amount) for t in transactions]
                })
            return features
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid transaction data: {str(e)}")
    
    def _prepare_sequence_features(self, transactions: List[Transaction], 
                                 sequence_length: int = 30) -> pd.DataFrame:
        """
        Prepare sequence features for time series prediction.
        
        Args:
            transactions: List of Transaction objects
            sequence_length: Number of days to look back
            
        Returns:
            pd.DataFrame: Prepared sequence features
            
        Raises:
            ValueError: If transaction data is invalid or insufficient
        """
        try:
            # Group transactions by date
            daily_amounts = pd.DataFrame({
                'date': [t.date for t in transactions],
                'amount': [float(t.amount) for t in transactions]
            }).groupby('date')['amount'].sum().reset_index()
            
            if len(daily_amounts) < sequence_length + 1:
                raise ValueError(f"Insufficient data: need at least {sequence_length + 1} days")
            
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
        except Exception as e:
            raise ValueError(f"Error preparing sequence features: {str(e)}")
    
    def _prepare_target_variable(self, features_df: pd.DataFrame, transactions: List[Transaction]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare target variable for expense prediction (next month's total expenses).
        """
        try:
            # Group transactions by month and calculate total expenses
            monthly_expenses = {}
            
            for t in transactions:
                # Handle both dict and Transaction objects
                if isinstance(t, dict):
                    transaction_type = t.get('type', 'EXPENSE')
                    amount = float(t.get('amount', 0))
                    date = t.get('date')
                    if hasattr(date, 'year') and hasattr(date, 'month'):
                        year, month = date.year, date.month
                    else:
                        continue
                else:
                    transaction_type = t.type
                    amount = float(t.amount)
                    year, month = t.date.year, t.date.month
                
                if transaction_type == 'EXPENSE':
                    month_key = (year, month)
                    if month_key not in monthly_expenses:
                        monthly_expenses[month_key] = 0
                    monthly_expenses[month_key] += abs(amount)
            
            # Create target variable (next month's expenses)
            X_list = []
            y_list = []
            
            for i, (year, month) in enumerate(sorted(monthly_expenses.keys())):
                if i == 0:  # Skip first month (no previous data)
                    continue
                    
                # Get current month's features (average of all transactions in that month)
                month_transactions = []
                for t in transactions:
                    if isinstance(t, dict):
                        t_date = t.get('date')
                        if hasattr(t_date, 'year') and hasattr(t_date, 'month'):
                            if t_date.year == year and t_date.month == month:
                                month_transactions.append(t)
                    else:
                        if t.date.year == year and t.date.month == month:
                            month_transactions.append(t)
                
                if not month_transactions:
                    continue
                
                # Get features for this month
                month_features = features_df.iloc[:len(month_transactions)].mean()
                
                # Target is next month's total expenses
                next_month_key = (year, month + 1) if month < 12 else (year + 1, 1)
                target_expense = monthly_expenses.get(next_month_key, 0)
                
                X_list.append(month_features)
                y_list.append(target_expense)
            
            if not X_list:
                # Fallback: use current month's expenses as target
                X_list = [features_df.mean()]
                y_list = [features_df['amount'].sum()]
            
            X = pd.DataFrame(X_list)
            y = pd.Series(y_list)
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing target variable: {e}")
            # Fallback: use amount as target
            return features_df.drop('amount', axis=1), features_df['amount']
    
    def train(self, transactions: List[Transaction]) -> None:
        """
        Train the enhanced expense predictor with improved features and algorithms.
        """
        try:
            if not transactions:
                raise ValueError("No transactions provided for training")
                
            self.logger.info(f"Training enhanced expense predictor on {len(transactions)} transactions")
            
            # Create enhanced features
            features_df = self._create_enhanced_features(transactions)
            
            # Si el DataFrame está vacío o no tiene 'amount', crear uno artificial
            if features_df.empty or 'amount' not in features_df.columns:
                self.logger.warning("[ExpensePredictor] Features vacías o sin columna 'amount'. Usando DataFrame artificial para evitar fallo.")
                features_df = pd.DataFrame({
                    'amount': [0.0],
                    'amount_log': [0.0],
                    'amount_sqrt': [0.0],
                    'amount_category': [0],
                    'day_of_week': [0],
                    'day_of_month': [1],
                    'month': [1],
                    'quarter': [1],
                    'is_weekend': [0],
                    'is_month_start': [1],
                    'is_month_end': [0],
                    'desc_length': [0],
                    'merchant_length': [0],
                    'desc_word_count': [0],
                    'has_numbers': [0],
                    'has_currency': [0],
                    'category_id': [0],
                    'subcategory_id': [0],
                    'transaction_type': [1],
                    'payment_method': [0],
                    'is_holiday_season': [0],
                    'is_summer': [0],
                    'is_winter': [0],
                    'is_large_purchase': [0],
                    'is_small_purchase': [0],
                    'is_medium_purchase': [0],
                })
            
            # Prepare target variable (next month's expenses)
            X, y = self._prepare_target_variable(features_df, transactions)
            
            # Definir las columnas esperadas antes de usarlas
            numeric_features = ['amount', 'amount_log', 'amount_sqrt', 'day_of_week', 'day_of_month', 
                              'month', 'quarter', 'desc_length', 'merchant_length', 'desc_word_count']
            
            categorical_features = ['amount_category', 'category_id', 'subcategory_id', 
                                  'transaction_type', 'payment_method', 'is_weekend', 
                                  'is_month_start', 'is_month_end', 'has_numbers', 'has_currency',
                                  'is_holiday_season', 'is_summer', 'is_winter', 'is_large_purchase',
                                  'is_small_purchase', 'is_medium_purchase']
            
            if len(X) < 2:
                # Si no hay suficientes datos para predicción temporal, usar predicción directa
                self.logger.info("Insufficient temporal data, switching to direct prediction mode")
                if 'amount' not in features_df.columns:
                    features_df['amount'] = 0.0
                
                # Asegurar que features_df tenga todas las columnas necesarias antes del drop
                all_expected_columns = numeric_features + categorical_features
                for col in all_expected_columns:
                    if col not in features_df.columns:
                        if col in numeric_features:
                            features_df[col] = 0.0
                        else:
                            features_df[col] = 0
                
                # Verificar que 'amount' esté presente antes del drop
                if 'amount' not in features_df.columns:
                    features_df['amount'] = 0.0
                
                X = features_df.drop('amount', axis=1)
                y = features_df['amount']
                
                if len(X) < 2:
                    self.logger.warning("[ExpensePredictor] Datos insuficientes incluso en modo directo. Usando DataFrame artificial.")
                    # Crear un DataFrame artificial con todas las columnas necesarias
                    artificial_df = pd.DataFrame({
                        'amount': [0.0],
                        'amount_log': [0.0],
                        'amount_sqrt': [0.0],
                        'amount_category': [0],
                        'day_of_week': [0],
                        'day_of_month': [1],
                        'month': [1],
                        'quarter': [1],
                        'is_weekend': [0],
                        'is_month_start': [1],
                        'is_month_end': [0],
                        'desc_length': [0],
                        'merchant_length': [0],
                        'desc_word_count': [0],
                        'has_numbers': [0],
                        'has_currency': [0],
                        'category_id': [0],
                        'subcategory_id': [0],
                        'transaction_type': [1],
                        'payment_method': [0],
                        'is_holiday_season': [0],
                        'is_summer': [0],
                        'is_winter': [0],
                        'is_large_purchase': [0],
                        'is_small_purchase': [0],
                        'is_medium_purchase': [0],
                    })
                    X = artificial_df.drop('amount', axis=1)
                    y = artificial_df['amount']
            
            # Reordenar y filtrar X para que tenga exactamente las columnas esperadas
            expected_columns = numeric_features + categorical_features
            for col in expected_columns:
                if col not in X.columns:
                    X[col] = 0.0 if col in numeric_features else 0
            X = X[expected_columns]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Create enhanced pipeline with multiple algorithms
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
            from sklearn.linear_model import LinearRegression, Ridge, Lasso
            from sklearn.svm import SVR
            from sklearn.preprocessing import StandardScaler, RobustScaler
            from sklearn.pipeline import Pipeline
            from sklearn.compose import ColumnTransformer
            from sklearn.feature_selection import SelectKBest, f_regression
            
            # Create preprocessing pipeline
            preprocessor = ColumnTransformer([
                ('num', RobustScaler(), numeric_features),
                ('cat', 'passthrough', categorical_features)
            ])
            
            # Create ensemble of models
            models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=200, max_depth=15, min_samples_split=5, 
                    min_samples_leaf=2, random_state=42, n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=150, learning_rate=0.1, max_depth=8, 
                    min_samples_split=5, random_state=42
                ),
                'extra_trees': ExtraTreesRegressor(
                    n_estimators=200, max_depth=15, min_samples_split=5, 
                    random_state=42, n_jobs=-1
                ),
                'ridge': Ridge(alpha=1.0, random_state=42),
                'lasso': Lasso(alpha=0.1, random_state=42)
            }
            
            # Train and evaluate each model
            best_score = -float('inf')
            best_model = None
            
            for name, model in models.items():
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('feature_selection', SelectKBest(f_regression, k=min(20, len(X.columns)))),
                    ('regressor', model)
                ])
            
            # Train model
                pipeline.fit(X_train, y_train)
                
                # Evaluate
                from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                y_pred = pipeline.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                self.logger.info(f"{name}: R² = {r2:.3f}, MAE = {mae:.2f}, RMSE = {rmse:.2f}")
                
                if r2 > best_score:
                    best_score = r2
                    best_model = pipeline
            
            # Use the best model
            self.pipeline = best_model
            self.is_fitted = True
            
            # Final evaluation
            y_pred_final = self.pipeline.predict(X_test)
            final_r2 = r2_score(y_test, y_pred_final)
            
            self.logger.info(f"Best model selected with R² = {final_r2:.3f}")
            self.logger.info(f"Model trained on {len(X_train)} samples, tested on {len(X_test)} samples")
            
            # Save model
            self.save()
            
        except Exception as e:
            self.logger.error(f"Failed to train enhanced model: {str(e)}")
            raise RuntimeError(f"Failed to train enhanced model: {str(e)}")
    
    def predict(self, date: pd.Timestamp, category_id: int) -> float:
        """
        Predict expenses for a given date and category.
        
        Args:
            date: Date to predict for
            category_id: Category ID to predict for
            
        Returns:
            float: Predicted expense amount
            
        Raises:
            RuntimeError: If model is not trained
            ValueError: If input data is invalid
        """
        try:
            if not self.is_fitted:
                raise RuntimeError("Model must be trained before making predictions")
                
            # Prepare features
            features = pd.DataFrame({
                'day_of_week': [date.weekday()],
                'day_of_month': [date.day],
                'month': [date.month],
                'category_id': [category_id]
            })
            
            # Make prediction
            prediction = float(self.pipeline.predict(features)[0])
            
            return max(0, prediction)  # Ensure non-negative prediction
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            raise RuntimeError(f"Failed to make prediction: {str(e)}")
    
    def predict_sequence(self, start_date: pd.Timestamp, days: int = 30) -> pd.DataFrame:
        """
        Predict expenses for a sequence of days.
        
        Args:
            start_date: Start date for prediction
            days: Number of days to predict
            
        Returns:
            pd.DataFrame: Predictions for each day
            
        Raises:
            RuntimeError: If model is not trained
            ValueError: If input data is invalid
        """
        try:
            if not self.is_fitted:
                raise RuntimeError("Model must be trained before making predictions")
                
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
            raise RuntimeError(f"Failed to make sequence prediction: {str(e)}")

    def _calculate_metrics(self, predictions: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: Model predictions
            true_labels: True labels
            
        Returns:
            dict: Dictionary of metric names and values
        """
        return {
            'mse': float(mean_squared_error(true_labels, predictions)),
            'rmse': float(np.sqrt(mean_squared_error(true_labels, predictions))),
            'mae': float(mean_absolute_error(true_labels, predictions)),
            'r2': float(r2_score(true_labels, predictions))
        }
    
    def evaluate(self, test_transactions: List[Transaction]) -> Dict[str, float]:
        """
        Evaluate the model's performance.
        
        Args:
            test_transactions: List of Transaction objects to evaluate on
            
        Returns:
            dict: Dictionary containing evaluation metrics
            
        Raises:
            RuntimeError: If model is not trained
            ValueError: If test data is invalid
        """
        try:
            if not self.is_fitted:
                raise RuntimeError("Model must be trained before evaluation")
                
            if not test_transactions:
                raise ValueError("No test transactions provided")
                
            # Prepare test data
            X_test = self._prepare_features(test_transactions)
            y_test = X_test['amount']
            X_test = X_test.drop('amount', axis=1)
            
            # Get predictions
            y_pred = self.pipeline.predict(X_test)
            
            # Calculate metrics
            return self._calculate_metrics(y_pred, y_test)
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise RuntimeError(f"Failed to evaluate model: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            dict: Dictionary containing model information
        """
        info = super().get_model_info()
        info.update({
            'feature_names': self.feature_names,
            'model_params': self.pipeline.get_params()
        })
        return info

    def save(self):
        """
        Save both the trained model and the scaler to disk using joblib.
        This ensures that predictions after loading use the same scaling as during training.
        """
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            model_data = {
                'model': self.pipeline,
            }
            joblib.dump(model_data, self.model_path)
            self.logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load(self):
        """
        Load both the trained model and the scaler from disk using joblib.
        This restores the model and scaler for accurate predictions.
        """
        try:
            if self.model_path.exists():
                model_data = joblib.load(self.model_path)
                self.pipeline = model_data['model']
                self.logger.info(f"Model loaded from {self.model_path}")
                self.is_fitted = True
            else:
                self.logger.warning(f"No saved model found at {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def reset(self):
        """
        Reset the model to its initial state.
        """
        self.is_fitted = False
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
            ))
        ])
        self.logger.info("Model reset") 

    def _create_enhanced_features(self, transactions: List[Transaction]) -> pd.DataFrame:
        """
        Create enhanced features for better expense prediction.
        """
        try:
            features_list = []
            
            for t in transactions:
                # Handle both Transaction objects and dictionaries
                if isinstance(t, dict):
                    # Handle dictionary format
                    amount = float(t.get('amount', 0))
                    date = t.get('date')
                    if hasattr(date, 'weekday'):
                        day_of_week = date.weekday()
                        day_of_month = date.day
                        month = date.month
                    else:
                        day_of_week = 0
                        day_of_month = 1
                        month = 1
                    
                    description = t.get('description', '')
                    merchant = t.get('merchant', '')
                    category_id = t.get('category_id', 0)
                    transaction_type = t.get('type', 'EXPENSE')
                    payment_method = t.get('payment_method', 'unknown')
                else:
                    # Handle Transaction objects
                    amount = float(t.amount)
                    date = t.date
                    day_of_week = date.weekday()
                    day_of_month = date.day
                    month = date.month
                    description = t.description or ""
                    merchant = t.merchant or ""
                    category_id = t.category.id if t.category else 0
                    transaction_type = t.type
                    payment_method = getattr(t, 'payment_method', 'unknown')
                
                # Enhanced temporal features
                quarter = (month - 1) // 3 + 1
                is_weekend = 1 if day_of_week >= 5 else 0
                is_month_start = 1 if day_of_month <= 3 else 0
                is_month_end = 1 if day_of_month >= 28 else 0
                
                # Amount-based features
                amount_log = np.log1p(abs(amount))
                amount_sqrt = np.sqrt(abs(amount))
                amount_category = self._categorize_amount(amount)
                
                # Enhanced text features
                desc_length = len(description)
                merchant_length = len(merchant)
                desc_word_count = len(description.split())
                has_numbers = 1 if any(c.isdigit() for c in description) else 0
                has_currency = 1 if any(symbol in description for symbol in ['$', '€', '£', '¥']) else 0
                
                # Category-based features (simplified for dictionary format)
                subcategory_id = 0  # Default for dictionary format
                
                # Transaction type
                transaction_type_encoded = 1 if transaction_type == 'EXPENSE' else 0
                
                # Payment method encoding
                payment_method_encoded = hash(str(payment_method)) % 10  # Simple encoding
                
                # Seasonal features
                is_holiday_season = 1 if month in [11, 12] else 0  # Nov-Dec
                is_summer = 1 if month in [6, 7, 8] else 0
                is_winter = 1 if month in [12, 1, 2] else 0
                
                # Behavioral features
                is_large_purchase = 1 if amount > 100 else 0
                is_small_purchase = 1 if amount < 10 else 0
                is_medium_purchase = 1 if 10 <= amount <= 100 else 0
                
                features = {
                    'amount': amount,
                    'amount_log': amount_log,
                    'amount_sqrt': amount_sqrt,
                    'amount_category': amount_category,
                    'day_of_week': day_of_week,
                    'day_of_month': day_of_month,
                    'month': month,
                    'quarter': quarter,
                    'is_weekend': is_weekend,
                    'is_month_start': is_month_start,
                    'is_month_end': is_month_end,
                    'desc_length': desc_length,
                    'merchant_length': merchant_length,
                    'desc_word_count': desc_word_count,
                    'has_numbers': has_numbers,
                    'has_currency': has_currency,
                    'category_id': category_id,
                    'subcategory_id': subcategory_id,
                    'transaction_type': transaction_type_encoded,
                    'payment_method': payment_method_encoded,
                    'is_holiday_season': is_holiday_season,
                    'is_summer': is_summer,
                    'is_winter': is_winter,
                    'is_large_purchase': is_large_purchase,
                    'is_small_purchase': is_small_purchase,
                    'is_medium_purchase': is_medium_purchase,
                }
                
                features_list.append(features)
            
            return pd.DataFrame(features_list)
            
        except Exception as e:
            self.logger.error(f"Error creating enhanced features: {e}")
            return pd.DataFrame()
    
    def _categorize_amount(self, amount: float) -> int:
        """
        Categorize amount into discrete bins for better feature engineering.
        
        Args:
            amount: Transaction amount
            
        Returns:
            int: Category bin (0-4)
        """
        try:
            abs_amount = abs(amount)
            if abs_amount < 10:
                return 0  # Small
            elif abs_amount < 50:
                return 1  # Medium-small
            elif abs_amount < 200:
                return 2  # Medium
            elif abs_amount < 1000:
                return 3  # Large
            else:
                return 4  # Very large
        except Exception:
            return 0  # Default to small category 