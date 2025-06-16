"""
Base class for all ML models in the system.
"""
from abc import ABC, abstractmethod
import joblib
from pathlib import Path
from django.conf import settings
import logging

logger = logging.getLogger('ai.ml')

class BaseMLModel(ABC):
    """
    Base class for all machine learning models.
    Provides common functionality for model training, prediction, and persistence.
    """
    
    def __init__(self, model_name):
        """
        Initialize the base ML model.
        
        Args:
            model_name (str): Name of the model, used for saving/loading
        """
        self.model_name = model_name
        self.model = None
        self.model_path = Path(settings.ML_MODELS_DIR) / f"{model_name}.joblib"
        self.logger = logger.getChild(model_name)
    
    @abstractmethod
    def train(self, data):
        """
        Train the model with the provided data.
        
        Args:
            data: Training data in the format expected by the specific model
        """
        pass
    
    @abstractmethod
    def predict(self, data):
        """
        Make predictions using the trained model.
        
        Args:
            data: Data to make predictions on
            
        Returns:
            Model predictions
        """
        pass
    
    def save(self):
        """
        Save the trained model to disk.
        """
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, self.model_path)
            self.logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load(self):
        """
        Load a trained model from disk.
        """
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                self.logger.info(f"Model loaded from {self.model_path}")
            else:
                self.logger.warning(f"No saved model found at {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def evaluate(self, test_data, test_labels):
        """
        Evaluate the model's performance.
        
        Args:
            test_data: Test data
            test_labels: True labels for test data
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        pass 