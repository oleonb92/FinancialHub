"""
Intent Classifier Module

This module provides intelligent intent classification using embeddings and machine learning.
It replaces basic keyword-matching with a more sophisticated approach using SentenceTransformers
and Logistic Regression for better accuracy and confidence scoring.
"""

import os
import pickle
import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Try to import sentence-transformers, fallback to basic embeddings if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available, using basic embeddings")

logger = logging.getLogger(__name__)


@dataclass
class IntentPrediction:
    """Represents an intent prediction with confidence score"""
    intent: str
    confidence: float
    all_scores: Dict[str, float]
    embedding: Optional[np.ndarray] = None


class IntentClassifier:
    """
    Intelligent intent classifier using embeddings and machine learning.
    
    This classifier uses SentenceTransformers to generate embeddings and
    Logistic Regression for classification, providing better accuracy than
    keyword-based approaches.
    """
    
    def __init__(self, model_path: str = None, language: str = 'en'):
        """
        Initialize the intent classifier.
        
        Args:
            model_path: Path to save/load the trained model
            language: Language for the model ('en' or 'es')
        """
        self.model_path = model_path or f"ml_models/intent_classifier_{language}.joblib"
        self.language = language
        self.model = None
        self.label_encoder = LabelEncoder()
        self.embedding_model = None
        self.is_trained = False
        
        # Intent categories for financial queries
        self.intent_categories = {
            'en': [
                'balance_inquiry',
                'spending_analysis', 
                'savings_planning',
                'trend_analysis',
                'anomaly_detection',
                'goal_tracking',
                'comparison',
                'prediction',
                'budget_analysis',
                'expense_categorization',
                'income_analysis',
                'debt_management',
                'investment_advice',
                'tax_planning',
                'clarification_request',
                'general_inquiry'
            ],
            'es': [
                'consulta_saldo',
                'analisis_gastos',
                'planificacion_ahorro',
                'analisis_tendencias',
                'deteccion_anomalias',
                'seguimiento_metas',
                'comparacion',
                'prediccion',
                'analisis_presupuesto',
                'categorizacion_gastos',
                'analisis_ingresos',
                'gestion_deudas',
                'consejos_inversion',
                'planificacion_impuestos',
                'solicitud_aclaracion',
                'consulta_general'
            ]
        }
        
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model based on availability."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Use a multilingual model for better language support
                self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                logger.info("Initialized SentenceTransformer embedding model")
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer: {e}")
                self.embedding_model = None
        else:
            logger.info("Using basic TF-IDF embeddings")
            self.embedding_model = None
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for the given text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        if self.embedding_model is not None:
            try:
                return self.embedding_model.encode([text])[0]
            except Exception as e:
                logger.warning(f"Error generating embedding: {e}")
                return self._basic_embedding(text)
        else:
            return self._basic_embedding(text)
    
    def _basic_embedding(self, text: str) -> np.ndarray:
        """
        Generate basic TF-IDF style embedding as fallback.
        
        Args:
            text: Input text
            
        Returns:
            Basic embedding vector
        """
        # Simple character-level features as fallback
        text_lower = text.lower()
        
        # Character frequency features
        char_features = np.zeros(26)
        for char in text_lower:
            if 'a' <= char <= 'z':
                char_features[ord(char) - ord('a')] += 1
        
        # Normalize
        if np.sum(char_features) > 0:
            char_features = char_features / np.sum(char_features)
        
        # Length feature
        length_feature = min(len(text) / 100.0, 1.0)
        
        # Combine features
        return np.concatenate([char_features, [length_feature]])
    
    def train_intent_classifier(self, dataset: List[Tuple[str, str]], 
                              test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the intent classifier with the provided dataset.
        
        Args:
            dataset: List of (text, intent_label) tuples
            test_size: Fraction of data to use for testing
            
        Returns:
            Training results dictionary
        """
        if not dataset:
            raise ValueError("Dataset cannot be empty")
        
        logger.info(f"Training intent classifier with {len(dataset)} samples")
        
        # Prepare data
        texts, labels = zip(*dataset)
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        
        X = np.array(embeddings)
        y = np.array(labels)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Train model
        logger.info("Training Logistic Regression model...")
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Generate classification report
        y_test_decoded = self.label_encoder.inverse_transform(y_test)
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred)
        report = classification_report(y_test_decoded, y_pred_decoded, output_dict=True)
        
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'n_samples': len(dataset),
            'n_classes': len(self.label_encoder.classes_),
            'classes': self.label_encoder.classes_.tolist(),
            'training_date': datetime.now().isoformat()
        }
        
        logger.info(f"Training completed. Accuracy: {accuracy:.3f}")
        return results
    
    def predict_intent(self, query: str) -> Tuple[str, float]:
        """
        Predict intent for the given query.
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (intent, confidence_score)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Generate embedding
        embedding = self._get_embedding(query)
        
        # Make prediction
        prediction_proba = self.model.predict_proba([embedding])[0]
        predicted_class_idx = np.argmax(prediction_proba)
        confidence = prediction_proba[predicted_class_idx]
        
        # Decode intent
        intent = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        return intent, confidence
    
    def predict_intent_detailed(self, query: str) -> IntentPrediction:
        """
        Predict intent with detailed information.
        
        Args:
            query: User query string
            
        Returns:
            IntentPrediction object with detailed scores
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Generate embedding
        embedding = self._get_embedding(query)
        
        # Make prediction
        prediction_proba = self.model.predict_proba([embedding])[0]
        predicted_class_idx = np.argmax(prediction_proba)
        confidence = prediction_proba[predicted_class_idx]
        
        # Get all scores
        all_scores = {}
        for i, prob in enumerate(prediction_proba):
            intent_name = self.label_encoder.inverse_transform([i])[0]
            all_scores[intent_name] = float(prob)
        
        # Decode intent
        intent = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        return IntentPrediction(
            intent=intent,
            confidence=confidence,
            all_scores=all_scores,
            embedding=embedding
        )
    
    def save_model(self, path: str = None):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise RuntimeError("No trained model to save")
        
        save_path = path or self.model_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model components
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'language': self.language,
            'intent_categories': self.intent_categories,
            'training_date': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, path: str = None):
        """Load a trained model from disk."""
        load_path = path or self.model_path
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        try:
            model_data = joblib.load(load_path)
            
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.language = model_data.get('language', 'en')
            self.intent_categories = model_data.get('intent_categories', self.intent_categories)
            self.is_trained = True
            
            logger.info(f"Model loaded from {load_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_training_data_template(self) -> List[Tuple[str, str]]:
        """
        Get a template of training data for the current language.
        
        Returns:
            List of (text, intent) tuples for training
        """
        categories = self.intent_categories.get(self.language, self.intent_categories['en'])
        
        # Template training data
        template_data = {
            'en': [
                ("What's my current balance?", "balance_inquiry"),
                ("How much money do I have?", "balance_inquiry"),
                ("Show me my account balance", "balance_inquiry"),
                ("How much did I spend this month?", "spending_analysis"),
                ("What are my expenses?", "spending_analysis"),
                ("Show my spending breakdown", "spending_analysis"),
                ("I want to save money", "savings_planning"),
                ("Help me create a savings plan", "savings_planning"),
                ("How can I save more?", "savings_planning"),
                ("Show me spending trends", "trend_analysis"),
                ("How has my spending changed?", "trend_analysis"),
                ("Are there any unusual transactions?", "anomaly_detection"),
                ("Show me suspicious spending", "anomaly_detection"),
                ("How am I doing with my goals?", "goal_tracking"),
                ("What's my progress on savings?", "goal_tracking"),
                ("Compare this month to last month", "comparison"),
                ("How does this compare to last year?", "comparison"),
                ("What will my expenses be next month?", "prediction"),
                ("Can you predict my future spending?", "prediction"),
                ("Analyze my budget", "budget_analysis"),
                ("How is my budget performing?", "budget_analysis"),
                ("Categorize my expenses", "expense_categorization"),
                ("What categories are my expenses in?", "expense_categorization"),
                ("Show me my income analysis", "income_analysis"),
                ("How much did I earn?", "income_analysis"),
                ("Help me manage my debt", "debt_management"),
                ("What's my debt situation?", "debt_management"),
                ("Give me investment advice", "investment_advice"),
                ("How should I invest my money?", "investment_advice"),
                ("Help with tax planning", "tax_planning"),
                ("What about my taxes?", "tax_planning"),
                ("I don't understand", "clarification_request"),
                ("Can you explain that?", "clarification_request"),
                ("What can you help me with?", "general_inquiry"),
                ("Tell me about your features", "general_inquiry")
            ],
            'es': [
                ("¿Cuál es mi saldo actual?", "consulta_saldo"),
                ("¿Cuánto dinero tengo?", "consulta_saldo"),
                ("Muéstrame mi balance de cuenta", "consulta_saldo"),
                ("¿Cuánto gasté este mes?", "analisis_gastos"),
                ("¿Cuáles son mis gastos?", "analisis_gastos"),
                ("Muéstrame el desglose de mis gastos", "analisis_gastos"),
                ("Quiero ahorrar dinero", "planificacion_ahorro"),
                ("Ayúdame a crear un plan de ahorro", "planificacion_ahorro"),
                ("¿Cómo puedo ahorrar más?", "planificacion_ahorro"),
                ("Muéstrame las tendencias de gastos", "analisis_tendencias"),
                ("¿Cómo han cambiado mis gastos?", "analisis_tendencias"),
                ("¿Hay transacciones inusuales?", "deteccion_anomalias"),
                ("Muéstrame gastos sospechosos", "deteccion_anomalias"),
                ("¿Cómo voy con mis metas?", "seguimiento_metas"),
                ("¿Cuál es mi progreso en ahorros?", "seguimiento_metas"),
                ("Compara este mes con el anterior", "comparacion"),
                ("¿Cómo se compara con el año pasado?", "comparacion"),
                ("¿Cuáles serán mis gastos el próximo mes?", "prediccion"),
                ("¿Puedes predecir mis gastos futuros?", "prediccion"),
                ("Analiza mi presupuesto", "analisis_presupuesto"),
                ("¿Cómo está funcionando mi presupuesto?", "analisis_presupuesto"),
                ("Categoriza mis gastos", "categorizacion_gastos"),
                ("¿En qué categorías están mis gastos?", "categorizacion_gastos"),
                ("Muéstrame mi análisis de ingresos", "analisis_ingresos"),
                ("¿Cuánto gané?", "analisis_ingresos"),
                ("Ayúdame a gestionar mi deuda", "gestion_deudas"),
                ("¿Cuál es mi situación de deuda?", "gestion_deudas"),
                ("Dame consejos de inversión", "consejos_inversion"),
                ("¿Cómo debo invertir mi dinero?", "consejos_inversion"),
                ("Ayuda con la planificación fiscal", "planificacion_impuestos"),
                ("¿Qué hay de mis impuestos?", "planificacion_impuestos"),
                ("No entiendo", "solicitud_aclaracion"),
                ("¿Puedes explicar eso?", "solicitud_aclaracion"),
                ("¿En qué puedes ayudarme?", "consulta_general"),
                ("Cuéntame sobre tus funciones", "consulta_general")
            ]
        }
        
        return template_data.get(self.language, template_data['en'])
    
    def evaluate_model(self, test_dataset: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Evaluate the trained model on a test dataset.
        
        Args:
            test_dataset: List of (text, intent_label) tuples for testing
            
        Returns:
            Evaluation results dictionary
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")
        
        texts, labels = zip(*test_dataset)
        
        # Generate embeddings
        embeddings = [self._get_embedding(text) for text in texts]
        X_test = np.array(embeddings)
        
        # Encode labels
        y_test = self.label_encoder.transform(labels)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Decode for classification report
        y_test_decoded = self.label_encoder.inverse_transform(y_test)
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred)
        
        report = classification_report(y_test_decoded, y_pred_decoded, output_dict=True)
        
        # Calculate confidence statistics
        max_confidences = np.max(y_pred_proba, axis=1)
        avg_confidence = np.mean(max_confidences)
        min_confidence = np.min(max_confidences)
        max_confidence = np.max(max_confidences)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confidence_stats': {
                'average': avg_confidence,
                'minimum': min_confidence,
                'maximum': max_confidence
            },
            'n_test_samples': len(test_dataset),
            'predictions': list(zip(texts, y_pred_decoded, max_confidences))
        }
        
        return results 