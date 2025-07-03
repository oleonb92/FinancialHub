"""
Transaction classifier for categorizing financial transactions.

This classifier uses a scikit-learn Pipeline with a ColumnTransformer to process both text and numeric features:
- The 'description' field is vectorized using TfidfVectorizer.
- Numeric fields ('amount', 'day_of_week', 'day_of_month', 'month') are scaled with StandardScaler.

The entire pipeline is saved and loaded using joblib to preserve the fitted state.
"""
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Union
from ai.ml.base import BaseMLModel
from django.db.models import Q
from transactions.models import Transaction, Category
import re
import logging
from datetime import datetime, timedelta

logger = logging.getLogger('ai.ml.classifier')

class TransactionClassifier(BaseMLModel):
    """
    Enhanced classifier for categorizing financial transactions based on their description
    and other features.

    Features used:
    - description (text, vectorized with enhanced preprocessing)
    - amount (numeric)
    - day_of_week (numeric)
    - day_of_month (numeric)
    - month (numeric)
    - transaction_type (categorical)
    - merchant (text, vectorized)
    - payment_method (categorical)
    - amount_bins (categorical)
    - description_length (numeric)
    - merchant_length (numeric)
    - is_weekend (boolean)
    - is_month_start (boolean)
    - is_month_end (boolean)
    - amount_log (numeric)
    - hour_of_day (numeric)
    - day_of_year (numeric)
    - week_of_year (numeric)
    - quarter (numeric)
    - is_holiday (boolean)
    - description_word_count (numeric)
    - merchant_word_count (numeric)
    - has_numbers (boolean)
    - has_currency_symbol (boolean)
    - amount_category (categorical)

    The model uses ensemble methods and advanced preprocessing for better accuracy.
    """
    
    def __init__(self):
        super().__init__('transaction_classifier')
        self.feature_names = [
            'description', 'amount', 'day_of_week', 'day_of_month', 'month',
            'transaction_type', 'merchant', 'payment_method', 'amount_bins',
            'description_length', 'merchant_length', 'is_weekend', 
            'is_month_start', 'is_month_end', 'amount_log', 'hour_of_day',
            'day_of_year', 'week_of_year', 'quarter', 'is_holiday',
            'description_word_count', 'merchant_word_count', 'has_numbers',
            'has_currency_symbol', 'amount_category'
        ]
        self.text_features = ['description', 'merchant']
        self.numeric_features = [
            'amount', 'day_of_week', 'day_of_month', 'month',
            'description_length', 'merchant_length', 'is_weekend', 
            'is_month_start', 'is_month_end', 'amount_log', 'hour_of_day',
            'day_of_year', 'week_of_year', 'quarter', 'is_holiday',
            'description_word_count', 'merchant_word_count', 'has_numbers',
            'has_currency_symbol'
        ]
        self.categorical_features = ['transaction_type', 'payment_method', 'amount_bins', 'amount_category']
        
        # Label encoders for categorical features
        self.label_encoders = {}
        
        # Enhanced pipeline with better preprocessing and multiple classifiers
        self.pipeline = Pipeline([
            ('preprocessor', ColumnTransformer([
                ('text_desc', TfidfVectorizer(
                    max_features=3000,
                    stop_words='english',
                    ngram_range=(1, 4),
                    min_df=1,
                    max_df=0.9,
                    sublinear_tf=True,
                    analyzer='word'
                ), 'description'),
                ('text_merchant', TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 3),
                    min_df=1,
                    max_df=0.85,
                    sublinear_tf=True,
                    analyzer='word'
                ), 'merchant'),
                ('num', RobustScaler(), self.numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_features)
            ], sparse_threshold=0.0)),  # Force dense output
            ('classifier', VotingClassifier([
                ('rf', RandomForestClassifier(
                    n_estimators=300,
                    max_depth=20,
                    min_samples_split=3,
                    min_samples_leaf=1,
                    max_features='sqrt',
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1
                )),
                ('gb', GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    subsample=0.8,
                    random_state=42
                )),
                ('et', ExtraTreesClassifier(
                    n_estimators=300,
                    max_depth=20,
                    min_samples_split=3,
                    min_samples_leaf=1,
                    max_features='sqrt',
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1
                )),
                ('svm', SVC(
                    C=1.0,
                    kernel='rbf',
                    gamma='scale',
                    probability=True,
                    random_state=42
                )),
                ('mlp', MLPClassifier(
                    hidden_layer_sizes=(200, 100, 50),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    learning_rate='adaptive',
                    max_iter=500,
                random_state=42
            ))
            ], voting='soft'))
        ])
        
        # Advanced ensemble models for better accuracy
        self.ensemble_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=300, max_depth=20, random_state=42, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=10, random_state=42
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=300, max_depth=20, random_state=42, n_jobs=-1
            ),
            'logistic_regression': LogisticRegression(
                C=1.0, max_iter=1000, random_state=42, n_jobs=-1
            )
        }
        
        self.categories = None
        self.is_fitted = False
        self.metrics = None
        
        # US holidays for better feature engineering
        self.us_holidays = {
            'new_year': [(1, 1)],
            'independence_day': [(7, 4)],
            'christmas': [(12, 25)],
            'thanksgiving': [(11, 22), (11, 23), (11, 24), (11, 25), (11, 26), (11, 27), (11, 28)],
            'labor_day': [(9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7)],
            'memorial_day': [(5, 25), (5, 26), (5, 27), (5, 28), (5, 29), (5, 30), (5, 31)]
        }
        
    def _enhance_text_preprocessing(self, text: str) -> str:
        """
        Enhanced text preprocessing for better feature extraction.
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Add common financial terms normalization
        financial_terms = {
            'atm': 'atm withdrawal',
            'pos': 'point of sale',
            'debit': 'debit card',
            'credit': 'credit card',
            'transfer': 'bank transfer',
            'deposit': 'bank deposit',
            'withdrawal': 'cash withdrawal',
            'purchase': 'purchase transaction',
            'payment': 'payment transaction',
            'fee': 'service fee',
            'charge': 'transaction charge',
            'refund': 'refund transaction',
            'cashback': 'cash back',
            'cash back': 'cash back',
            'online': 'online transaction',
            'in store': 'in store purchase',
            'in-store': 'in store purchase',
            'grocery': 'grocery store',
            'supermarket': 'grocery store',
            'restaurant': 'dining out',
            'fast food': 'dining out',
            'gas': 'gas station',
            'fuel': 'gas station',
            'gasoline': 'gas station',
            'uber': 'transportation',
            'lyft': 'transportation',
            'taxi': 'transportation',
            'amazon': 'online shopping',
            'walmart': 'shopping',
            'target': 'shopping',
            'costco': 'shopping',
            'netflix': 'entertainment subscription',
            'spotify': 'entertainment subscription',
            'hulu': 'entertainment subscription',
            'disney': 'entertainment subscription',
            'apple': 'technology',
            'google': 'technology',
            'microsoft': 'technology',
            'verizon': 'utilities',
            'at&t': 'utilities',
            'comcast': 'utilities',
            'electric': 'utilities',
            'water': 'utilities',
            'internet': 'utilities',
            'phone': 'utilities',
            'insurance': 'insurance payment',
            'medical': 'health care',
            'dental': 'health care',
            'pharmacy': 'health care',
            'doctor': 'health care',
            'hospital': 'health care',
            'clinic': 'health care',
            'gym': 'fitness',
            'fitness': 'fitness',
            'workout': 'fitness',
            'movie': 'entertainment',
            'theater': 'entertainment',
            'cinema': 'entertainment',
            'hotel': 'travel',
            'airline': 'travel',
            'flight': 'travel',
            'rental': 'travel',
            'car rental': 'travel',
            'parking': 'transportation',
            'toll': 'transportation',
            'bus': 'transportation',
            'train': 'transportation',
            'subway': 'transportation',
            'metro': 'transportation'
        }
        
        for term, replacement in financial_terms.items():
            text = text.replace(term, replacement)
        
        return text
    
    def _is_holiday(self, date) -> int:
        """
        Check if date is a US holiday.
        """
        if not hasattr(date, 'month') or not hasattr(date, 'day'):
            return 0
            
        month_day = (date.month, date.day)
        
        for holiday_name, dates in self.us_holidays.items():
            if month_day in dates:
                return 1
        return 0
    
    def _create_enhanced_features(self, transactions: Union[Transaction, List[Transaction], List[Dict]]) -> pd.DataFrame:
        """
        Create enhanced features for better classification.
        """
        if not isinstance(transactions, list):
            transactions = [transactions]
            
        try:
            features_list = []
            
            for t in transactions:
                # Handle both dict and Transaction objects
                if isinstance(t, dict):
                    description = t.get('description', '')
                    amount = float(t.get('amount', 0))
                    date = t.get('date')
                    transaction_type = t.get('type', 'EXPENSE')
                    merchant = t.get('merchant', '')
                    payment_method = t.get('payment_method', '')
                else:
                    description = t.description or ''
                    amount = float(t.amount)
                    date = t.date
                    transaction_type = t.type
                    merchant = getattr(t, 'merchant', '') or ''
                    payment_method = getattr(t, 'payment_method', '') or ''
                
                # Enhanced text preprocessing
                description = self._enhance_text_preprocessing(description)
                merchant = self._enhance_text_preprocessing(merchant)
                
                # Advanced feature engineering
                amount_log = np.log1p(abs(amount)) if amount != 0 else 0
                hour_of_day = date.hour if hasattr(date, 'hour') else 12
                day_of_year = date.timetuple().tm_yday if hasattr(date, 'timetuple') else 1
                week_of_year = date.isocalendar()[1] if hasattr(date, 'isocalendar') else 1
                quarter = (date.month - 1) // 3 + 1 if hasattr(date, 'month') else 1
                is_holiday = self._is_holiday(date)
                
                description_word_count = len(description.split())
                merchant_word_count = len(merchant.split())
                has_numbers = 1 if re.search(r'\d', description) else 0
                has_currency_symbol = 1 if re.search(r'[\$\€\£]', description) else 0
                
                # Create enhanced features
                features = {
                    'description': description,
                    'amount': amount,
                    'day_of_week': date.weekday() if hasattr(date, 'weekday') else 0,
                    'day_of_month': date.day if hasattr(date, 'day') else 1,
                    'month': date.month if hasattr(date, 'month') else 1,
                    'transaction_type': transaction_type,
                    'merchant': merchant,
                    'payment_method': payment_method,
                    'amount_bins': self._bin_amount(amount),
                    'description_length': len(description),
                    'merchant_length': len(merchant),
                    'is_weekend': 1 if hasattr(date, 'weekday') and date.weekday() >= 5 else 0,
                    'is_month_start': 1 if hasattr(date, 'day') and date.day <= 3 else 0,
                    'is_month_end': 1 if hasattr(date, 'day') and date.day >= 28 else 0,
                    'amount_log': amount_log,
                    'hour_of_day': hour_of_day,
                    'day_of_year': day_of_year,
                    'week_of_year': week_of_year,
                    'quarter': quarter,
                    'is_holiday': is_holiday,
                    'description_word_count': description_word_count,
                    'merchant_word_count': merchant_word_count,
                    'has_numbers': has_numbers,
                    'has_currency_symbol': has_currency_symbol,
                    'amount_category': self._categorize_amount(amount)
                }
                
                features_list.append(features)
            
            df = pd.DataFrame(features_list)
            
            # Ensure all categorical columns are strings
            for col in self.categorical_features:
                if col in df.columns:
                    df[col] = df[col].astype(str)
            
            # Ensure all numeric columns are numeric
            for col in self.numeric_features:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            return df
            
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid transaction data: {str(e)}")
    
    def _bin_amount(self, amount: float) -> str:
        """
        Create amount bins for better categorization.
        """
        if amount <= 10:
            return 'very_small'
        elif amount <= 50:
            return 'small'
        elif amount <= 200:
            return 'medium'
        elif amount <= 1000:
            return 'large'
        else:
            return 'very_large'
    
    def _categorize_amount(self, amount: float) -> str:
        """
        Create more detailed amount categories for better classification.
        """
        if amount <= 5:
            return 'micro'
        elif amount <= 25:
            return 'small_personal'
        elif amount <= 100:
            return 'medium_personal'
        elif amount <= 500:
            return 'large_personal'
        elif amount <= 2000:
            return 'business_small'
        elif amount <= 10000:
            return 'business_medium'
        else:
            return 'business_large'
    
    def _prepare_features(self, transactions: Union[Transaction, List[Transaction], List[Dict]]) -> pd.DataFrame:
        """
        Prepare features for training or prediction using enhanced preprocessing.
        """
        return self._create_enhanced_features(transactions)
    
    def train(self, transactions: Union[List[Transaction], List[Dict]]) -> None:
        """
        Train the enhanced transaction classifier with cross-validation and hyperparameter optimization.
        """
        try:
            if not transactions:
                raise ValueError("No transactions provided for training")
                
            # Split data into train and test sets (80% train, 20% test)
            import random
            random.shuffle(transactions)
            split_index = int(len(transactions) * 0.8)
            train_transactions = transactions[:split_index]
            test_transactions = transactions[split_index:]
            
            self.logger.info(f"Training on {len(train_transactions)} transactions, testing on {len(test_transactions)} transactions")
            
            # Prepare features and labels for training
            X_train = self._prepare_features(train_transactions)
            
            # Extract labels based on data type
            if train_transactions and isinstance(train_transactions[0], dict):
                valid_transactions = [t for t in train_transactions if t.get('category_id')]
                if not valid_transactions:
                    raise ValueError("No valid transactions with category_id found")
                y_train = np.array([t.get('category_id') for t in valid_transactions])
                self.categories = {t.get('category_id'): t.get('category_name', '') for t in valid_transactions}
                X_train = self._prepare_features(valid_transactions)
            else:
                valid_transactions = [t for t in train_transactions if hasattr(t, 'category') and t.category]
                if not valid_transactions:
                    raise ValueError("No valid transactions with category found")
                y_train = np.array([t.category.id for t in valid_transactions])
                self.categories = {t.category.id: t.category.name for t in valid_transactions}
                X_train = self._prepare_features(valid_transactions)
            
            if len(X_train) != len(y_train):
                raise ValueError(f"Inconsistent data: {len(X_train)} features vs {len(y_train)} labels")
            
            # Perform cross-validation to get baseline performance
            self.logger.info("Performing cross-validation...")
            cv_scores = cross_val_score(
                self.pipeline, 
                X_train, 
                y_train, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='accuracy',
                n_jobs=-1
            )
            
            self.logger.info(f"Cross-validation scores: {cv_scores}")
            self.logger.info(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Train the main pipeline
            self.logger.info("Training main pipeline...")
            self.pipeline.fit(X_train, y_train)
            self.is_fitted = True
            
            # Train ensemble models for better accuracy
            self._train_ensemble_models(X_train, y_train)
            
            # Evaluate the model
            if test_transactions:
                test_metrics = self.evaluate(test_transactions)
                self.metrics = test_metrics
                self.logger.info(f"Model evaluation completed. Accuracy: {test_metrics.get('accuracy', 'N/A'):.3f}")
                
                # If accuracy is below threshold, try hyperparameter optimization
                accuracy_threshold = 0.85
                if test_metrics.get('accuracy', 0) < accuracy_threshold:
                    self.logger.info(f"Accuracy below {accuracy_threshold}, attempting hyperparameter optimization...")
                    self._optimize_hyperparameters(X_train, y_train)
            
            # Calculate training metrics
            y_pred_train = self.pipeline.predict(X_train)
            self.metrics_train = self._calculate_metrics(y_pred_train, y_train)
            
            self.save()
            self.logger.info(f"Enhanced model trained on {len(train_transactions)} transactions")
            
        except Exception as e:
            self.logger.error(f"Error training enhanced model: {str(e)}")
            raise RuntimeError(f"Failed to train enhanced model: {str(e)}")
    
    def _optimize_hyperparameters(self, X_train, y_train) -> Dict[str, Any]:
        """
        Optimize hyperparameters using a more efficient search space.
        """
        try:
            self.logger.info("Starting hyperparameter optimization...")
            
            # Create a simpler pipeline for optimization (single classifier instead of voting)
            from sklearn.ensemble import RandomForestClassifier
            
            simple_pipeline = Pipeline([
                ('preprocessor', ColumnTransformer([
                    ('text_desc', TfidfVectorizer(
                        max_features=2000,
                        stop_words='english',
                        ngram_range=(1, 3),
                        min_df=1,
                        max_df=0.9
                    ), 'description'),
                    ('text_merchant', TfidfVectorizer(
                        max_features=1000,
                        stop_words='english',
                        ngram_range=(1, 2),
                        min_df=1,
                        max_df=0.85
                    ), 'merchant'),
                    ('num', RobustScaler(), self.numeric_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_features)
                ], sparse_threshold=0.0)),
                ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
            ])
            
            # Reduced parameter grid for faster optimization
            param_grid = {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__max_features': ['sqrt', 'log2', None],
                'preprocessor__text_desc__max_features': [1000, 2000],
                'preprocessor__text_merchant__max_features': [500, 1000],
            }
            
            # Use RandomizedSearchCV for faster search
            random_search = RandomizedSearchCV(
                simple_pipeline,
                param_distributions=param_grid,
                n_iter=15,  # Only try 15 random combinations
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
            
            random_search.fit(X_train, y_train)
            
            self.logger.info(f"Best parameters: {random_search.best_params_}")
            self.logger.info(f"Best cross-validation score: {random_search.best_score_:.3f}")
            
            # Update the main pipeline with optimized preprocessor
            best_preprocessor = random_search.best_estimator_.named_steps['preprocessor']
            self.pipeline.named_steps['preprocessor'] = best_preprocessor
            
            return random_search.best_params_
            
        except Exception as e:
            self.logger.error(f"Hyperparameter optimization failed: {e}")
            return {}
    
    def _train_ensemble_models(self, X: pd.DataFrame, y: np.ndarray):
        """
        Train ensemble models for better accuracy.
        """
        try:
            # Prepare features for ensemble models (simplified)
            X_ensemble = X[['amount', 'day_of_week', 'day_of_month', 'month', 'description_length', 
                           'amount_log', 'hour_of_day', 'quarter', 'is_holiday']].copy()
            
            for name, model in self.ensemble_models.items():
                self.logger.info(f"Training ensemble model: {name}")
                model.fit(X_ensemble, y)
                
        except Exception as e:
            self.logger.warning(f"Error training ensemble models: {str(e)}")
    
    def predict(self, transaction: Transaction) -> Tuple[int, float]:
        """
        Enhanced prediction with ensemble voting and improved confidence calculation.
        """
        try:
            if not self.is_fitted:
                raise RuntimeError("Model must be trained before making predictions")
                
            # Prepare features
            features = self._prepare_features([transaction])
            
            # Get main prediction
            main_prediction = self.pipeline.predict(features)[0]
            main_probabilities = self.pipeline.predict_proba(features)[0]
            main_probability = np.max(main_probabilities)
            
            # Get ensemble predictions
            X_ensemble = features[['amount', 'day_of_week', 'day_of_month', 'month', 'description_length', 
                                 'amount_log', 'hour_of_day', 'quarter', 'is_holiday']].copy()
            ensemble_predictions = []
            ensemble_probabilities = []
            
            for name, model in self.ensemble_models.items():
                try:
                    pred = model.predict(X_ensemble)[0]
                    prob = model.predict_proba(X_ensemble)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
                    ensemble_predictions.append(pred)
                    ensemble_probabilities.append(np.max(prob))
                except:
                    continue
            
            # Enhanced voting mechanism with confidence weighting
            if ensemble_predictions:
                # Count votes
                from collections import Counter
                vote_counts = Counter(ensemble_predictions)
                most_common_prediction = vote_counts.most_common(1)[0][0]
                agreement_ratio = vote_counts.most_common(1)[0][1] / len(ensemble_predictions)
                
                # Calculate ensemble confidence
                ensemble_confidence = np.mean(ensemble_probabilities) if ensemble_probabilities else 0.5
                
                # Weighted decision based on agreement and confidence
                if most_common_prediction == main_prediction:
                    # Ensemble agrees with main prediction
                    final_prediction = main_prediction
                    confidence = (main_probability * 0.6) + (ensemble_confidence * 0.4)
                else:
                    # Ensemble disagrees with main prediction
                    if agreement_ratio >= 0.7:  # 70% agreement threshold
                        final_prediction = most_common_prediction
                        confidence = ensemble_confidence
                    elif agreement_ratio >= 0.5:  # 50% agreement threshold
                        # Use weighted average
                        final_prediction = most_common_prediction
                        confidence = (ensemble_confidence * 0.7) + (main_probability * 0.3)
                    else:
                        # Use main prediction with reduced confidence
                        final_prediction = main_prediction
                        confidence = main_probability * 0.8
            else:
                final_prediction = main_prediction
                confidence = main_probability
            
            # Apply confidence boosting for high-probability predictions
            if confidence > 0.8:
                confidence = min(confidence * 1.1, 0.99)  # Boost confidence but cap at 0.99
            
            return int(final_prediction), float(confidence)
            
        except Exception as e:
            self.logger.error(f"Error making enhanced prediction: {str(e)}")
            raise RuntimeError(f"Failed to make enhanced prediction: {str(e)}")
    
    def _calculate_metrics(self, predictions: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: Model predictions
            true_labels: True labels
            
        Returns:
            dict: Dictionary of metric names and values
        """
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, 
            predictions, 
            average='weighted'
        )
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
    
    def evaluate(self, test_transactions: List[Transaction]) -> Dict[str, float]:
        """
        Evaluate the model's performance.
        Args:
            test_transactions: List of Transaction objects or dicts to evaluate on
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
            X_test = self._prepare_features(test_transactions)
            # Soporte para dict y modelo
            if isinstance(test_transactions[0], dict):
                y_test = np.array([t.get('category_id') for t in test_transactions])
            else:
                y_test = np.array([t.category.id for t in test_transactions])
            y_pred = self.pipeline.predict(X_test)
            metrics = self._calculate_metrics(y_pred, y_test)
            return metrics
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
            'categories': self.categories,
            'pipeline_steps': [step[0] for step in self.pipeline.steps]
        })
        return info

    def save(self):
        """
        Save the trained pipeline to disk using joblib.
        This ensures that the entire preprocessing and model pipeline, including fitted parameters, is preserved.
        """
        import joblib
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            # Guardar pipeline y métricas
            joblib.dump({'pipeline': self.pipeline, 'metrics': self.metrics, 'categories': self.categories}, self.model_path)
            self.logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load(self):
        """
        Load the trained pipeline from disk using joblib.
        This restores the entire preprocessing and model pipeline, ready for predictions.
        """
        import joblib
        try:
            if self.model_path.exists():
                data = joblib.load(self.model_path)
                if isinstance(data, dict):
                    self.pipeline = data.get('pipeline', self.pipeline)
                    self.metrics = data.get('metrics', None)
                    self.categories = data.get('categories', None)
                else:
                    self.pipeline = data
                self.logger.info(f"Model loaded from {self.model_path}")
                self.is_fitted = True
            else:
                self.logger.warning(f"No saved model found at {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def reset(self):
        """
        Reset the classifier to its initial state.
        """
        self.is_fitted = False
        self.categories = None
        self.pipeline = Pipeline([
            ('preprocessor', ColumnTransformer([
                ('text', TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                ), 'description'),
                ('num', StandardScaler(), ['amount', 'day_of_week', 'day_of_month', 'month'])
            ])),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ))
        ])
        self.save()
        self.logger.info("Model reset") 