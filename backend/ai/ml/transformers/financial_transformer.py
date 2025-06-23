"""
Sistema de Transformers personalizado para análisis financiero.

Este módulo implementa:
- Arquitectura de transformer personalizada
- Embeddings específicos para finanzas
- Análisis de secuencias temporales
- Predicción de series financieras
- Análisis de sentimiento avanzado
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import math
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import json
import joblib
from dataclasses import dataclass

logger = logging.getLogger('ai.transformers')

@dataclass
class TransformerConfig:
    """Configuración del transformer"""
    vocab_size: int = 10000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_length: int = 512
    dropout: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 100

class PositionalEncoding(nn.Module):
    """Codificación posicional para transformers"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """Atención multi-cabeza"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Atención de producto escalado"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Proyectar a Q, K, V
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Aplicar atención
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenar y proyectar
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.w_o(attention_output)
        
        return output, attention_weights

class FeedForward(nn.Module):
    """Capa feed-forward"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Bloque de transformer"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Atención con residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward con residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class FinancialTransformer(nn.Module):
    """Transformer personalizado para análisis financiero"""
    
    def __init__(self, config: TransformerConfig, num_classes: int = 1):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_length)
        
        # Capas de transformer
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # Capas de salida
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.d_model, num_classes)
        
        # Inicializar pesos
        self._init_weights()
        
    def _init_weights(self):
        """Inicializa los pesos del modelo"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def create_padding_mask(self, seq):
        """Crea máscara de padding"""
        return (seq != 0).unsqueeze(1).unsqueeze(2)
    
    def forward(self, x, mask=None):
        # Embeddings + posicional encoding
        x = self.embedding(x) * math.sqrt(self.config.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Crear máscara si no se proporciona
        if mask is None:
            mask = self.create_padding_mask(x)
        
        # Pasar por bloques de transformer
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # Pooling global (promedio)
        x = x.mean(dim=1)
        
        # Clasificación
        output = self.fc(x)
        
        return output

class FinancialDataset(Dataset):
    """Dataset para datos financieros"""
    
    def __init__(self, texts: List[str], labels: List[float] = None, 
                 tokenizer=None, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenizar
        if self.tokenizer:
            tokens = self.tokenizer.encode(text, max_length=self.max_length, 
                                         truncation=True, padding='max_length')
        else:
            # Tokenización simple
            tokens = self._simple_tokenize(text)
        
        # Convertir a tensor
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float)
            return input_ids, label
        else:
            return input_ids
    
    def _simple_tokenize(self, text: str) -> List[int]:
        """Tokenización simple basada en palabras"""
        # Implementación básica - en producción usar un tokenizer real
        words = text.lower().split()
        tokens = []
        
        for word in words:
            # Hash simple de palabras
            token_id = hash(word) % 10000
            tokens.append(token_id)
        
        # Padding
        if len(tokens) < self.max_length:
            tokens.extend([0] * (self.max_length - len(tokens)))
        else:
            tokens = tokens[:self.max_length]
        
        return tokens

class FinancialTokenizer:
    """Tokenizador personalizado para texto financiero"""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_to_id = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.id_to_word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        self.word_freq = {}
        
    def fit(self, texts: List[str]):
        """Entrena el tokenizador con los textos"""
        # Contar frecuencia de palabras
        for text in texts:
            words = text.lower().split()
            for word in words:
                self.word_freq[word] = self.word_freq.get(word, 0) + 1
        
        # Tomar las palabras más frecuentes
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        
        for i, (word, freq) in enumerate(sorted_words[:self.vocab_size - 4]):
            word_id = i + 4
            self.word_to_id[word] = word_id
            self.id_to_word[word_id] = word
    
    def encode(self, text: str, max_length: int = None, 
              truncation: bool = True, padding: str = 'max_length') -> List[int]:
        """Codifica texto a tokens"""
        words = text.lower().split()
        tokens = [self.word_to_id.get(word, self.word_to_id['<UNK>']) for word in words]
        
        if truncation and max_length:
            tokens = tokens[:max_length]
        
        if padding == 'max_length' and max_length:
            if len(tokens) < max_length:
                tokens.extend([self.word_to_id['<PAD>']] * (max_length - len(tokens)))
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Decodifica tokens a texto"""
        words = []
        for token in tokens:
            if token in self.id_to_word:
                word = self.id_to_word[token]
                if word not in ['<PAD>', '<START>', '<END>']:
                    words.append(word)
        
        return ' '.join(words)

class FinancialTransformerTrainer:
    """Entrenador para el transformer financiero"""
    
    def __init__(self, model: FinancialTransformer, config: TransformerConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizador y función de pérdida
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss() if config.num_classes == 1 else nn.CrossEntropyLoss()
        
        # Historial de entrenamiento
        self.train_losses = []
        self.val_losses = []
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Entrena el modelo"""
        self.model.train()
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    logger.info(f'Epoch {epoch+1}/{self.config.epochs}, '
                              f'Batch {batch_idx}/{len(train_loader)}, '
                              f'Loss: {loss.item():.4f}')
            
            # Calcular pérdida promedio
            avg_loss = epoch_loss / len(train_loader)
            self.train_losses.append(avg_loss)
            
            # Validación
            if val_loader:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                logger.info(f'Epoch {epoch+1}/{self.config.epochs}, '
                          f'Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}')
            else:
                logger.info(f'Epoch {epoch+1}/{self.config.epochs}, '
                          f'Train Loss: {avg_loss:.4f}')
    
    def validate(self, val_loader: DataLoader) -> float:
        """Valida el modelo"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), targets)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def predict(self, test_loader: DataLoader) -> np.ndarray:
        """Realiza predicciones"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for inputs in test_loader:
                if isinstance(inputs, tuple):
                    inputs = inputs[0]
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predictions.extend(outputs.cpu().numpy())
        
        return np.array(predictions)
    
    def save_model(self, filepath: str):
        """Guarda el modelo"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)
        logger.info(f"Modelo guardado en {filepath}")
    
    def load_model(self, filepath: str):
        """Carga el modelo"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        logger.info(f"Modelo cargado desde {filepath}")

class FinancialTransformerService:
    """Servicio principal para el transformer financiero"""
    
    def __init__(self, config: TransformerConfig = None):
        self.config = config or TransformerConfig()
        self.tokenizer = FinancialTokenizer(self.config.vocab_size)
        self.model = None
        self.trainer = None
        self.scaler = StandardScaler()
        
    def prepare_data(self, texts: List[str], labels: List[float] = None) -> Tuple[DataLoader, DataLoader]:
        """Prepara los datos para entrenamiento"""
        # Entrenar tokenizador
        self.tokenizer.fit(texts)
        
        # Dividir datos
        if labels is not None:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42
            )
        else:
            train_texts, val_texts = train_test_split(texts, test_size=0.2, random_state=42)
            train_labels = val_labels = None
        
        # Crear datasets
        train_dataset = FinancialDataset(
            train_texts, train_labels, self.tokenizer, self.config.max_seq_length
        )
        val_dataset = FinancialDataset(
            val_texts, val_labels, self.tokenizer, self.config.max_seq_length
        )
        
        # Crear dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        
        return train_loader, val_loader
    
    def train_model(self, texts: List[str], labels: List[float]):
        """Entrena el modelo transformer"""
        try:
            # Preparar datos
            train_loader, val_loader = self.prepare_data(texts, labels)
            
            # Crear modelo
            self.model = FinancialTransformer(self.config)
            self.trainer = FinancialTransformerTrainer(self.model, self.config)
            
            # Entrenar
            self.trainer.train(train_loader, val_loader)
            
            logger.info("Modelo transformer entrenado exitosamente")
            
        except Exception as e:
            logger.error(f"Error entrenando modelo transformer: {str(e)}")
            raise
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Realiza predicciones"""
        if self.model is None:
            raise ValueError("Modelo no entrenado. Ejecuta train_model() primero.")
        
        try:
            # Preparar datos
            dataset = FinancialDataset(texts, tokenizer=self.tokenizer, 
                                     max_length=self.config.max_seq_length)
            dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
            
            # Predecir
            predictions = self.trainer.predict(dataloader)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error realizando predicciones: {str(e)}")
            raise
    
    def analyze_sentiment(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analiza sentimiento usando el transformer"""
        try:
            predictions = self.predict(texts)
            
            results = []
            for pred in predictions:
                # Normalizar predicción a rango [-1, 1]
                sentiment_score = np.tanh(pred[0])
                
                # Clasificar sentimiento
                if sentiment_score > 0.1:
                    sentiment = 'positive'
                elif sentiment_score < -0.1:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                results.append({
                    'sentiment': sentiment,
                    'score': float(sentiment_score),
                    'confidence': abs(sentiment_score)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error analizando sentimiento: {str(e)}")
            raise
    
    def save_model(self, filepath_prefix: str):
        """Guarda el modelo y tokenizador"""
        try:
            # Guardar modelo
            if self.trainer:
                self.trainer.save_model(f"{filepath_prefix}_transformer.pt")
            
            # Guardar tokenizador
            tokenizer_data = {
                'word_to_id': self.tokenizer.word_to_id,
                'id_to_word': self.tokenizer.id_to_word,
                'vocab_size': self.tokenizer.vocab_size
            }
            joblib.dump(tokenizer_data, f"{filepath_prefix}_tokenizer.joblib")
            
            logger.info(f"Modelo y tokenizador guardados con prefijo: {filepath_prefix}")
            
        except Exception as e:
            logger.error(f"Error guardando modelo: {str(e)}")
            raise
    
    def load_model(self, filepath_prefix: str):
        """Carga el modelo y tokenizador"""
        try:
            # Cargar tokenizador
            tokenizer_data = joblib.load(f"{filepath_prefix}_tokenizer.joblib")
            self.tokenizer.word_to_id = tokenizer_data['word_to_id']
            self.tokenizer.id_to_word = tokenizer_data['id_to_word']
            self.tokenizer.vocab_size = tokenizer_data['vocab_size']
            
            # Cargar modelo
            self.model = FinancialTransformer(self.config)
            self.trainer = FinancialTransformerTrainer(self.model, self.config)
            self.trainer.load_model(f"{filepath_prefix}_transformer.pt")
            
            logger.info(f"Modelo y tokenizador cargados desde prefijo: {filepath_prefix}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtiene información del modelo"""
        return {
            'config': {
                'vocab_size': self.config.vocab_size,
                'd_model': self.config.d_model,
                'n_heads': self.config.n_heads,
                'n_layers': self.config.n_layers,
                'max_seq_length': self.config.max_seq_length
            },
            'model_loaded': self.model is not None,
            'tokenizer_vocab_size': len(self.tokenizer.word_to_id),
            'trainer_available': self.trainer is not None
        } 