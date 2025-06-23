"""
Sistema de Análisis de Sentimiento para Fintech.

Este módulo implementa un sistema avanzado de análisis de sentimiento que incluye:
- Análisis de sentimiento de noticias financieras
- Tracking de sentimiento en redes sociales
- Indicadores de sentimiento de mercado
- Análisis de sentimiento de transacciones
- Machine Learning para clasificación de sentimiento
- Alertas basadas en cambios de sentimiento
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from django.utils import timezone
from textblob import TextBlob
import re
import requests
import json

logger = logging.getLogger('ai.sentiment_analyzer')

class SentimentAnalyzer:
    """
    Sistema principal de análisis de sentimiento.
    
    Características:
    - Análisis de noticias financieras
    - Tracking de redes sociales
    - Indicadores de mercado
    - Análisis de transacciones
    - Machine Learning adaptativo
    - Alertas de sentimiento
    """
    
    def __init__(self, model_path: str = 'backend/ml_models/sentiment_analyzer.joblib'):
        self.model_path = model_path
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.sentiment_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        self.naive_bayes = MultinomialNB()
        self.is_trained = False
        
        # Configuración de análisis
        self.analysis_config = {
            'confidence_threshold': 0.7,
            'update_frequency': 'hourly',
            'max_text_length': 1000,
            'min_text_length': 10,
            'sentiment_thresholds': {
                'very_positive': 0.8,
                'positive': 0.6,
                'neutral': 0.4,
                'negative': 0.2,
                'very_negative': 0.0
            }
        }
        
        # Palabras clave financieras
        self.financial_keywords = {
            'positive': [
                'bullish', 'rally', 'surge', 'gain', 'profit', 'growth', 'strong',
                'positive', 'up', 'higher', 'increase', 'rise', 'boost', 'recovery'
            ],
            'negative': [
                'bearish', 'crash', 'drop', 'loss', 'decline', 'weak', 'negative',
                'down', 'lower', 'decrease', 'fall', 'plunge', 'recession'
            ],
            'market_terms': [
                'stock', 'market', 'trading', 'investment', 'portfolio', 'asset',
                'bond', 'currency', 'commodity', 'crypto', 'etf', 'mutual fund'
            ]
        }
        
        # Cargar modelo si existe
        self.load()
        
    def analyze_text_sentiment(self, text: str, method: str = 'ml') -> Dict[str, Any]:
        """
        Analiza el sentimiento de un texto.
        
        Args:
            text: Texto a analizar
            method: Método de análisis ('ml', 'textblob', 'keyword')
            
        Returns:
            dict: Resultados del análisis de sentimiento
        """
        try:
            if not text or len(text.strip()) < self.analysis_config['min_text_length']:
                return {
                    'sentiment': 'neutral',
                    'score': 0.5,
                    'confidence': 0.0,
                    'method': method,
                    'error': 'Text too short'
                }
            
            # Limpiar texto
            cleaned_text = self._clean_text(text)
            
            if method == 'ml' and self.is_trained:
                return self._analyze_ml_sentiment(cleaned_text)
            elif method == 'textblob':
                return self._analyze_textblob_sentiment(cleaned_text)
            elif method == 'keyword':
                return self._analyze_keyword_sentiment(cleaned_text)
            else:
                # Fallback a TextBlob
                return self._analyze_textblob_sentiment(cleaned_text)
                
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {str(e)}")
            return {
                'sentiment': 'neutral',
                'score': 0.5,
                'confidence': 0.0,
                'method': method,
                'error': str(e)
            }
    
    def _clean_text(self, text: str) -> str:
        """
        Limpia y preprocesa el texto.
        
        Args:
            text: Texto original
            
        Returns:
            str: Texto limpio
        """
        try:
            # Convertir a minúsculas
            text = text.lower()
            
            # Remover URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remover caracteres especiales pero mantener palabras
            text = re.sub(r'[^\w\s]', ' ', text)
            
            # Remover espacios extra
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Limitar longitud
            if len(text) > self.analysis_config['max_text_length']:
                text = text[:self.analysis_config['max_text_length']]
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return text
    
    def _analyze_ml_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analiza sentimiento usando Machine Learning.
        
        Args:
            text: Texto limpio
            
        Returns:
            dict: Resultados de ML
        """
        try:
            # Vectorizar texto
            text_vectorized = self.vectorizer.transform([text])
            
            # Predicción
            sentiment_proba = self.sentiment_classifier.predict_proba(text_vectorized)[0]
            predicted_class = self.sentiment_classifier.predict(text_vectorized)[0]
            
            # Mapear clases a scores
            class_mapping = {
                0: 0.0,    # Very negative
                1: 0.25,   # Negative
                2: 0.5,    # Neutral
                3: 0.75,   # Positive
                4: 1.0     # Very positive
            }
            
            sentiment_score = class_mapping.get(predicted_class, 0.5)
            confidence = max(sentiment_proba)
            
            return {
                'sentiment': self._get_sentiment_label(sentiment_score),
                'score': float(sentiment_score),
                'confidence': float(confidence),
                'method': 'ml',
                'predicted_class': int(predicted_class),
                'class_probabilities': sentiment_proba.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in ML sentiment analysis: {str(e)}")
            return {
                'sentiment': 'neutral',
                'score': 0.5,
                'confidence': 0.0,
                'method': 'ml',
                'error': str(e)
            }
    
    def _analyze_textblob_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analiza sentimiento usando TextBlob.
        
        Args:
            text: Texto limpio
            
        Returns:
            dict: Resultados de TextBlob
        """
        try:
            blob = TextBlob(text)
            sentiment_score = (blob.sentiment.polarity + 1) / 2  # Normalizar a 0-1
            
            return {
                'sentiment': self._get_sentiment_label(sentiment_score),
                'score': float(sentiment_score),
                'confidence': float(abs(blob.sentiment.subjectivity)),
                'method': 'textblob',
                'polarity': float(blob.sentiment.polarity),
                'subjectivity': float(blob.sentiment.subjectivity)
            }
            
        except Exception as e:
            logger.error(f"Error in TextBlob sentiment analysis: {str(e)}")
            return {
                'sentiment': 'neutral',
                'score': 0.5,
                'confidence': 0.0,
                'method': 'textblob',
                'error': str(e)
            }
    
    def _analyze_keyword_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analiza sentimiento usando palabras clave.
        
        Args:
            text: Texto limpio
            
        Returns:
            dict: Resultados de análisis por palabras clave
        """
        try:
            words = text.lower().split()
            
            positive_count = sum(1 for word in words if word in self.financial_keywords['positive'])
            negative_count = sum(1 for word in words if word in self.financial_keywords['negative'])
            
            total_keywords = positive_count + negative_count
            
            if total_keywords == 0:
                return {
                    'sentiment': 'neutral',
                    'score': 0.5,
                    'confidence': 0.0,
                    'method': 'keyword',
                    'positive_keywords': 0,
                    'negative_keywords': 0
                }
            
            sentiment_score = positive_count / total_keywords
            confidence = min(total_keywords / 10, 1.0)  # Normalizar confianza
            
            return {
                'sentiment': self._get_sentiment_label(sentiment_score),
                'score': float(sentiment_score),
                'confidence': float(confidence),
                'method': 'keyword',
                'positive_keywords': positive_count,
                'negative_keywords': negative_count,
                'total_keywords': total_keywords
            }
            
        except Exception as e:
            logger.error(f"Error in keyword sentiment analysis: {str(e)}")
            return {
                'sentiment': 'neutral',
                'score': 0.5,
                'confidence': 0.0,
                'method': 'keyword',
                'error': str(e)
            }
    
    def _get_sentiment_label(self, score: float) -> str:
        """
        Convierte score numérico a etiqueta de sentimiento.
        
        Args:
            score: Score de sentimiento (0-1)
            
        Returns:
            str: Etiqueta de sentimiento
        """
        thresholds = self.analysis_config['sentiment_thresholds']
        
        if score >= thresholds['very_positive']:
            return 'very_positive'
        elif score >= thresholds['positive']:
            return 'positive'
        elif score >= thresholds['neutral']:
            return 'neutral'
        elif score >= thresholds['negative']:
            return 'negative'
        else:
            return 'very_negative'
    
    def analyze_news_sentiment(self, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analiza sentimiento de noticias financieras.
        
        Args:
            news_data: Lista de noticias con título y contenido
            
        Returns:
            dict: Análisis agregado de sentimiento
        """
        try:
            if not news_data:
                return {
                    'overall_sentiment': 'neutral',
                    'average_score': 0.5,
                    'confidence': 0.0,
                    'news_count': 0,
                    'sentiment_distribution': {},
                    'top_positive_news': [],
                    'top_negative_news': []
                }
            
            sentiment_results = []
            positive_news = []
            negative_news = []
            
            for news in news_data:
                title = news.get('title', '')
                content = news.get('content', '')
                text = f"{title} {content}"
                
                sentiment = self.analyze_text_sentiment(text, 'ml')
                sentiment['news_id'] = news.get('id')
                sentiment['title'] = title
                sentiment['published_at'] = news.get('published_at')
                
                sentiment_results.append(sentiment)
                
                # Categorizar noticias
                if sentiment['sentiment'] in ['positive', 'very_positive']:
                    positive_news.append(sentiment)
                elif sentiment['sentiment'] in ['negative', 'very_negative']:
                    negative_news.append(sentiment)
            
            # Calcular estadísticas agregadas
            scores = [r['score'] for r in sentiment_results if 'score' in r]
            average_score = np.mean(scores) if scores else 0.5
            
            # Distribución de sentimientos
            sentiment_distribution = {}
            for result in sentiment_results:
                sentiment = result.get('sentiment', 'neutral')
                sentiment_distribution[sentiment] = sentiment_distribution.get(sentiment, 0) + 1
            
            # Ordenar noticias por score
            positive_news.sort(key=lambda x: x.get('score', 0), reverse=True)
            negative_news.sort(key=lambda x: x.get('score', 0))
            
            return {
                'overall_sentiment': self._get_sentiment_label(average_score),
                'average_score': float(average_score),
                'confidence': float(np.mean([r.get('confidence', 0) for r in sentiment_results])),
                'news_count': len(news_data),
                'sentiment_distribution': sentiment_distribution,
                'top_positive_news': positive_news[:5],
                'top_negative_news': negative_news[:5],
                'analyzed_at': timezone.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {str(e)}")
            return {
                'error': str(e),
                'overall_sentiment': 'neutral',
                'average_score': 0.5
            }
    
    def analyze_social_media_sentiment(self, posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analiza sentimiento de posts de redes sociales.
        
        Args:
            posts: Lista de posts de redes sociales
            
        Returns:
            dict: Análisis de sentimiento de redes sociales
        """
        try:
            if not posts:
                return {
                    'overall_sentiment': 'neutral',
                    'average_score': 0.5,
                    'confidence': 0.0,
                    'posts_count': 0,
                    'platform_sentiment': {},
                    'trending_topics': []
                }
            
            sentiment_results = []
            platform_sentiment = {}
            
            for post in posts:
                text = post.get('text', '')
                platform = post.get('platform', 'unknown')
                
                sentiment = self.analyze_text_sentiment(text, 'textblob')
                sentiment['post_id'] = post.get('id')
                sentiment['platform'] = platform
                sentiment['created_at'] = post.get('created_at')
                
                sentiment_results.append(sentiment)
                
                # Agrupar por plataforma
                if platform not in platform_sentiment:
                    platform_sentiment[platform] = []
                platform_sentiment[platform].append(sentiment)
            
            # Calcular sentimiento por plataforma
            platform_averages = {}
            for platform, sentiments in platform_sentiment.items():
                scores = [s.get('score', 0.5) for s in sentiments]
                platform_averages[platform] = {
                    'average_score': float(np.mean(scores)),
                    'sentiment': self._get_sentiment_label(np.mean(scores)),
                    'posts_count': len(sentiments)
                }
            
            # Calcular sentimiento general
            all_scores = [r.get('score', 0.5) for r in sentiment_results]
            overall_score = np.mean(all_scores)
            
            # Extraer temas trending (simplificado)
            trending_topics = self._extract_trending_topics(posts)
            
            return {
                'overall_sentiment': self._get_sentiment_label(overall_score),
                'average_score': float(overall_score),
                'confidence': float(np.mean([r.get('confidence', 0) for r in sentiment_results])),
                'posts_count': len(posts),
                'platform_sentiment': platform_averages,
                'trending_topics': trending_topics,
                'analyzed_at': timezone.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing social media sentiment: {str(e)}")
            return {
                'error': str(e),
                'overall_sentiment': 'neutral',
                'average_score': 0.5
            }
    
    def _extract_trending_topics(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extrae temas trending de los posts.
        
        Args:
            posts: Lista de posts
            
        Returns:
            list: Temas trending con sentimiento
        """
        try:
            # Palabras comunes a ignorar
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            
            # Contar palabras
            word_counts = {}
            word_sentiments = {}
            
            for post in posts:
                text = post.get('text', '').lower()
                words = re.findall(r'\b\w+\b', text)
                
                # Filtrar palabras
                words = [w for w in words if w not in stop_words and len(w) > 3]
                
                # Analizar sentimiento del post
                sentiment = self.analyze_text_sentiment(post.get('text', ''))
                
                for word in words:
                    if word in self.financial_keywords['market_terms']:
                        if word not in word_counts:
                            word_counts[word] = 0
                            word_sentiments[word] = []
                        
                        word_counts[word] += 1
                        word_sentiments[word].append(sentiment.get('score', 0.5))
            
            # Crear lista de temas trending
            trending_topics = []
            for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                avg_sentiment = np.mean(word_sentiments[word])
                trending_topics.append({
                    'topic': word,
                    'frequency': count,
                    'sentiment': self._get_sentiment_label(avg_sentiment),
                    'sentiment_score': float(avg_sentiment)
                })
            
            return trending_topics
            
        except Exception as e:
            logger.error(f"Error extracting trending topics: {str(e)}")
            return []
    
    def get_market_sentiment_indicator(self, news_sentiment: Dict[str, Any], 
                                     social_sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera indicador de sentimiento de mercado combinando fuentes.
        
        Args:
            news_sentiment: Análisis de sentimiento de noticias
            social_sentiment: Análisis de sentimiento de redes sociales
            
        Returns:
            dict: Indicador de sentimiento de mercado
        """
        try:
            # Pesos para diferentes fuentes
            news_weight = 0.6
            social_weight = 0.4
            
            # Scores ponderados
            news_score = news_sentiment.get('average_score', 0.5)
            social_score = social_sentiment.get('average_score', 0.5)
            
            # Calcular score combinado
            combined_score = (news_score * news_weight + social_score * social_weight)
            
            # Calcular confianza combinada
            news_confidence = news_sentiment.get('confidence', 0.0)
            social_confidence = social_sentiment.get('confidence', 0.0)
            combined_confidence = (news_confidence * news_weight + social_confidence * social_weight)
            
            # Determinar señal de mercado
            if combined_score > 0.7:
                market_signal = 'bullish'
                signal_strength = 'strong'
            elif combined_score > 0.6:
                market_signal = 'bullish'
                signal_strength = 'moderate'
            elif combined_score < 0.3:
                market_signal = 'bearish'
                signal_strength = 'strong'
            elif combined_score < 0.4:
                market_signal = 'bearish'
                signal_strength = 'moderate'
            else:
                market_signal = 'neutral'
                signal_strength = 'weak'
            
            return {
                'market_sentiment': self._get_sentiment_label(combined_score),
                'sentiment_score': float(combined_score),
                'confidence': float(combined_confidence),
                'market_signal': market_signal,
                'signal_strength': signal_strength,
                'news_contribution': {
                    'score': float(news_score),
                    'weight': news_weight
                },
                'social_contribution': {
                    'score': float(social_score),
                    'weight': social_weight
                },
                'generated_at': timezone.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating market sentiment indicator: {str(e)}")
            return {
                'error': str(e),
                'market_sentiment': 'neutral',
                'sentiment_score': 0.5
            }
    
    def train(self, training_data: List[Dict[str, Any]], labels: List[int] = None):
        """
        Entrena el modelo de análisis de sentimiento.
        
        Args:
            training_data: Lista de textos de entrenamiento
            labels: Etiquetas de sentimiento (0-4)
        """
        try:
            if not training_data or not labels:
                logger.warning("No training data or labels provided")
                return
            
            # Limpiar textos
            cleaned_texts = [self._clean_text(item.get('text', '')) for item in training_data]
            
            # Filtrar textos válidos
            valid_indices = [i for i, text in enumerate(cleaned_texts) if len(text) >= self.analysis_config['min_text_length']]
            valid_texts = [cleaned_texts[i] for i in valid_indices]
            valid_labels = [labels[i] for i in valid_indices]
            
            if not valid_texts:
                logger.warning("No valid texts after cleaning")
                return
            
            # Vectorizar textos
            X = self.vectorizer.fit_transform(valid_texts)
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, valid_labels, test_size=0.2, random_state=42, stratify=valid_labels
            )
            
            # Entrenar clasificador
            self.sentiment_classifier.fit(X_train, y_train)
            
            # Entrenar Naive Bayes como respaldo
            self.naive_bayes.fit(X_train, y_train)
            
            # Evaluar modelo
            y_pred = self.sentiment_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Sentiment model trained - Accuracy: {accuracy:.3f}")
            
            # Guardar reporte
            report = classification_report(y_test, y_pred, output_dict=True)
            logger.info(f"Classification report: {report}")
            
            self.is_trained = True
            self.save()
            
            logger.info("Sentiment analysis model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training sentiment analysis model: {str(e)}")
            raise
    
    def save(self):
        """Guarda el modelo entrenado."""
        try:
            model_data = {
                'vectorizer': self.vectorizer,
                'sentiment_classifier': self.sentiment_classifier,
                'naive_bayes': self.naive_bayes,
                'is_trained': self.is_trained,
                'analysis_config': self.analysis_config,
                'financial_keywords': self.financial_keywords
            }
            
            joblib.dump(model_data, self.model_path)
            logger.info(f"Sentiment analysis model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving sentiment analysis model: {str(e)}")
            raise
    
    def load(self):
        """Carga el modelo entrenado."""
        try:
            model_data = joblib.load(self.model_path)
            
            self.vectorizer = model_data['vectorizer']
            self.sentiment_classifier = model_data['sentiment_classifier']
            self.naive_bayes = model_data['naive_bayes']
            self.is_trained = model_data['is_trained']
            self.analysis_config = model_data.get('analysis_config', self.analysis_config)
            self.financial_keywords = model_data.get('financial_keywords', self.financial_keywords)
            
            logger.info("Sentiment analysis model loaded successfully")
            
        except FileNotFoundError:
            logger.info("No pre-trained sentiment analysis model found")
        except Exception as e:
            logger.error(f"Error loading sentiment analysis model: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtiene información del modelo.
        
        Returns:
            dict: Información del modelo
        """
        return {
            'is_trained': self.is_trained,
            'analysis_config': self.analysis_config,
            'financial_keywords_count': {
                'positive': len(self.financial_keywords['positive']),
                'negative': len(self.financial_keywords['negative']),
                'market_terms': len(self.financial_keywords['market_terms'])
            },
            'model_path': self.model_path
        } 