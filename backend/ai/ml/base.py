"""
Base class for all ML models in the system.
"""
from abc import ABC, abstractmethod
import joblib
from pathlib import Path
from django.conf import settings
import logging
from typing import Any, Dict, Optional, Union, List
import numpy as np
import pandas as pd

logger = logging.getLogger('ai.ml')

class BaseMLModel(ABC):
    """
    Base class for all machine learning models.
    Provides common functionality for model training, prediction, and persistence.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the base ML model.
        
        Args:
            model_name (str): Name of the model, used for saving/loading
        """
        self.model_name = model_name
        self.model = None
        self.model_path = Path(settings.ML_MODELS_DIR) / f"{model_name}.joblib"
        self.logger = logger.getChild(model_name)
        self.is_trained = False
    
    @abstractmethod
    def train(self, data: Union[List[Any], pd.DataFrame, np.ndarray]) -> None:
        """
        Train the model with the provided data.
        
        Args:
            data: Training data in the format expected by the specific model
            
        Raises:
            ValueError: If the data is invalid or insufficient
            RuntimeError: If there's an error during training
        """
        pass
    
    @abstractmethod
    def predict(self, data: Union[List[Any], pd.DataFrame, np.ndarray]) -> Any:
        """
        Make predictions using the trained model.
        
        Args:
            data: Data to make predictions on
            
        Returns:
            Model predictions
            
        Raises:
            RuntimeError: If the model is not trained
            ValueError: If the input data is invalid
        """
        pass
    
    def save(self) -> None:
        """
        Save the trained model to disk.
        
        Raises:
            RuntimeError: If there's an error saving the model
        """
        try:
            if not self.is_trained:
                raise RuntimeError("Cannot save untrained model")
                
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, self.model_path)
            self.logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise RuntimeError(f"Failed to save model: {str(e)}")
    
    def load(self) -> None:
        """
        Load a trained model from disk.
        
        Raises:
            FileNotFoundError: If no saved model exists
            RuntimeError: If there's an error loading the model
        """
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"No saved model found at {self.model_path}")
                
            self.model = joblib.load(self.model_path)
            self.is_trained = True
            self.logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def evaluate(self, test_data: Union[List[Any], pd.DataFrame, np.ndarray], 
                test_labels: Union[List[Any], np.ndarray]) -> Dict[str, float]:
        """
        Evaluate the model's performance.
        
        Args:
            test_data: Test data
            test_labels: True labels for test data
            
        Returns:
            dict: Dictionary containing evaluation metrics
            
        Raises:
            RuntimeError: If the model is not trained
            ValueError: If the test data or labels are invalid
        """
        if not self.is_trained:
            raise RuntimeError("Cannot evaluate untrained model")
            
        try:
            predictions = self.predict(test_data)
            return self._calculate_metrics(predictions, test_labels)
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise RuntimeError(f"Failed to evaluate model: {str(e)}")
    
    def _calculate_metrics(self, predictions: Any, true_labels: Any) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: Model predictions
            true_labels: True labels
            
        Returns:
            dict: Dictionary of metric names and values
        """
        raise NotImplementedError("Subclasses must implement _calculate_metrics")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            dict: Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "is_trained": self.is_trained,
            "model_type": self.__class__.__name__,
            "model_path": str(self.model_path)
        } 