"""
Sistema de Procesamiento de Lenguaje Natural para análisis financiero.

Este módulo implementa:
- Análisis de sentimientos financieros
- Extracción de entidades financieras
- Clasificación de documentos
- Resumen automático de texto
- Análisis de temas (Topic Modeling)
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import re
import string
import os
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import spacy
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import json
import joblib
from collections import Counter, defaultdict

logger = logging.getLogger('ai.nlp')

class FinancialTextProcessor:
    def __init__(self, language: str = 'en', use_spacy: bool = True):
        """
        Inicializa el procesador de texto financiero.
        
        Args:
            language: Idioma del texto ('en', 'es')
            use_spacy: Si usar spaCy para procesamiento avanzado
        """
        self.language = language
        self.use_spacy = use_spacy
        
        # Descargar recursos NLTK si no están disponibles
        self._download_nltk_resources()
        
        # Inicializar procesadores
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Cargar spaCy si está disponible
        if use_spacy:
            try:
                self.nlp = spacy.load('en_core_web_sm')
                logger.info("spaCy cargado exitosamente")
            except OSError:
                logger.warning("spaCy no disponible, usando NLTK")
                self.use_spacy = False
                self.nlp = None
        
        # Vocabulario financiero específico
        self.financial_terms = self._load_financial_vocabulary()
        
        # Modelos entrenados
        self.sentiment_model = None
        self.topic_model = None
        self.classifier_model = None
        
    def _download_nltk_resources(self):
        """Descarga recursos necesarios de NLTK"""
        try:
            # Descargar recursos básicos
            resources = [
                'punkt',
                'stopwords', 
                'wordnet',
                'averaged_perceptron_tagger',
                'maxent_ne_chunker',
                'words',
                'vader_lexicon'
            ]
            
            for resource in resources:
                try:
                    nltk.download(resource, quiet=True)
                except Exception as e:
                    logger.warning(f"Error descargando {resource}: {e}")
                    
        except Exception as e:
            logger.warning(f"Error descargando recursos NLTK: {e}")
    
    def _load_financial_vocabulary(self) -> Dict[str, List[str]]:
        """Carga vocabulario financiero específico"""
        return {
            'positive_terms': [
                'profit', 'growth', 'revenue', 'increase', 'positive', 'strong',
                'bullish', 'gain', 'surge', 'rally', 'outperform', 'beat',
                'exceed', 'rise', 'climb', 'soar', 'jump', 'boost', 'improve'
            ],
            'negative_terms': [
                'loss', 'decline', 'decrease', 'negative', 'weak', 'bearish',
                'drop', 'fall', 'crash', 'plunge', 'underperform', 'miss',
                'decline', 'fall', 'drop', 'crash', 'plunge', 'downgrade'
            ],
            'financial_entities': [
                'revenue', 'profit', 'earnings', 'income', 'expense', 'cost',
                'investment', 'asset', 'liability', 'equity', 'debt', 'cash',
                'stock', 'bond', 'dividend', 'interest', 'tax', 'budget'
            ],
            'currencies': ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF'],
            'time_periods': ['quarter', 'year', 'month', 'week', 'daily', 'annual']
        }
    
    def preprocess_text(self, text: str, 
                       remove_stopwords: bool = True,
                       lemmatize: bool = True,
                       remove_punctuation: bool = True) -> str:
        """
        Preprocesa el texto para análisis.
        
        Args:
            text: Texto a procesar
            remove_stopwords: Si remover palabras vacías
            lemmatize: Si aplicar lematización
            remove_punctuation: Si remover puntuación
            
        Returns:
            str: Texto preprocesado
        """
        try:
            # Convertir a minúsculas
            text = text.lower()
            
            # Remover puntuación
            if remove_punctuation:
                text = re.sub(r'[^\w\s]', '', text)
            
            # Tokenización robusta
            try:
                tokens = word_tokenize(text)
            except Exception as tokenize_error:
                logger.warning(f"Error en tokenización NLTK, usando split simple: {tokenize_error}")
                # Fallback: tokenización simple
                tokens = text.split()
            
            # Remover palabras vacías
            if remove_stopwords:
                try:
                    stop_words = set(stopwords.words('english'))
                    tokens = [token for token in tokens if token not in stop_words]
                except Exception as stopwords_error:
                    logger.warning(f"Error cargando stopwords, saltando: {stopwords_error}")
            
            # Lematización
            if lemmatize:
                try:
                    tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
                except Exception as lemmatize_error:
                    logger.warning(f"Error en lematización, saltando: {lemmatize_error}")
            
            # Unir tokens
            processed_text = ' '.join(tokens)
            
            return processed_text
            
        except Exception as e:
            logger.error(f"Error preprocesando texto: {str(e)}")
            return text
    
    def extract_financial_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extrae entidades financieras del texto.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Dict con entidades extraídas
        """
        entities = {
            'currencies': [],
            'amounts': [],
            'percentages': [],
            'companies': [],
            'dates': [],
            'financial_terms': []
        }
        
        try:
            # Patrones regex para entidades financieras
            currency_pattern = r'\$[\d,]+(?:\.\d{2})?|\d+(?:\.\d{2})?\s*(?:USD|EUR|GBP|JPY|CAD|AUD|CHF)'
            amount_pattern = r'\$[\d,]+(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?'
            percentage_pattern = r'\d+(?:\.\d+)?%'
            date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}'
            
            # Extraer usando regex
            entities['currencies'] = re.findall(currency_pattern, text, re.IGNORECASE)
            entities['amounts'] = re.findall(amount_pattern, text)
            entities['percentages'] = re.findall(percentage_pattern, text)
            entities['dates'] = re.findall(date_pattern, text)
            
            # Extraer términos financieros
            try:
                tokens = word_tokenize(text.lower())
            except Exception as tokenize_error:
                logger.warning(f"Error en tokenización para entidades, usando split: {tokenize_error}")
                tokens = text.lower().split()
            
            entities['financial_terms'] = [
                token for token in tokens 
                if token in self.financial_terms['financial_entities']
            ]
            
            # Usar spaCy para extracción de entidades nombradas
            if self.use_spacy and self.nlp:
                doc = self.nlp(text)
                
                for ent in doc.ents:
                    if ent.label_ == 'ORG':
                        entities['companies'].append(ent.text)
                    elif ent.label_ == 'MONEY':
                        entities['amounts'].append(ent.text)
                    elif ent.label_ == 'PERCENT':
                        entities['percentages'].append(ent.text)
                    elif ent.label_ == 'DATE':
                        entities['dates'].append(ent.text)
            
            # Remover duplicados
            for key in entities:
                entities[key] = list(set(entities[key]))
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extrayendo entidades: {str(e)}")
            return entities
    
    def analyze_sentiment(self, text: str, method: str = 'vader') -> Dict[str, float]:
        """
        Analiza el sentimiento del texto.
        
        Args:
            text: Texto a analizar
            method: Método de análisis ('vader', 'financial', 'custom')
            
        Returns:
            Dict con scores de sentimiento
        """
        try:
            if method == 'vader':
                return self._vader_sentiment(text)
            elif method == 'financial':
                return self._financial_sentiment(text)
            elif method == 'custom':
                return self._custom_sentiment(text)
            else:
                raise ValueError(f"Método de sentimiento no válido: {method}")
                
        except Exception as e:
            logger.error(f"Error analizando sentimiento: {str(e)}")
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
    
    def _vader_sentiment(self, text: str) -> Dict[str, float]:
        """Análisis de sentimiento usando VADER"""
        scores = self.sentiment_analyzer.polarity_scores(text)
        return scores
    
    def _financial_sentiment(self, text: str) -> Dict[str, float]:
        """Análisis de sentimiento específico para finanzas"""
        text_lower = text.lower()
        tokens = word_tokenize(text_lower)
        
        positive_count = sum(1 for token in tokens if token in self.financial_terms['positive_terms'])
        negative_count = sum(1 for token in tokens if token in self.financial_terms['negative_terms'])
        total_terms = len(tokens)
        
        if total_terms == 0:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        positive_score = positive_count / total_terms
        negative_score = negative_count / total_terms
        neutral_score = 1.0 - positive_score - negative_score
        
        # Calcular score compuesto
        compound_score = positive_score - negative_score
        
        return {
            'compound': compound_score,
            'positive': positive_score,
            'negative': negative_score,
            'neutral': neutral_score
        }
    
    def _custom_sentiment(self, text: str) -> Dict[str, float]:
        """Análisis de sentimiento personalizado"""
        # Combinar VADER con análisis financiero
        vader_scores = self._vader_sentiment(text)
        financial_scores = self._financial_sentiment(text)
        
        # Ponderar scores
        compound = (vader_scores['compound'] * 0.7 + financial_scores['compound'] * 0.3)
        positive = (vader_scores['positive'] * 0.7 + financial_scores['positive'] * 0.3)
        negative = (vader_scores['negative'] * 0.7 + financial_scores['negative'] * 0.3)
        neutral = (vader_scores['neutral'] * 0.7 + financial_scores['neutral'] * 0.3)
        
        return {
            'compound': compound,
            'positive': positive,
            'negative': negative,
            'neutral': neutral
        }
    
    def extract_topics(self, texts: List[str], n_topics: int = 5, 
                      method: str = 'lda') -> Dict[str, Any]:
        """
        Extrae temas de una colección de textos.
        
        Args:
            texts: Lista de textos
            n_topics: Número de temas a extraer
            method: Método de extracción ('lda', 'nmf')
            
        Returns:
            Dict con temas extraídos
        """
        try:
            # Preprocesar textos
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Vectorizar
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            X = vectorizer.fit_transform(processed_texts)
            
            if method == 'lda':
                model = LatentDirichletAllocation(
                    n_components=n_topics,
                    random_state=42,
                    max_iter=100
                )
            elif method == 'nmf':
                model = NMF(
                    n_components=n_topics,
                    random_state=42,
                    max_iter=200
                )
            else:
                raise ValueError(f"Método de extracción no válido: {method}")
            
            # Entrenar modelo
            model.fit(X)
            
            # Extraer temas
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(model.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': topic[top_words_idx].tolist()
                })
            
            # Asignar temas a documentos
            doc_topics = model.transform(X)
            document_topics = []
            
            for i, doc_topic in enumerate(doc_topics):
                dominant_topic = doc_topic.argmax()
                document_topics.append({
                    'document_id': i,
                    'dominant_topic': dominant_topic,
                    'topic_distribution': doc_topic.tolist()
                })
            
            return {
                'topics': topics,
                'document_topics': document_topics,
                'model_type': method,
                'n_topics': n_topics
            }
            
        except Exception as e:
            logger.error(f"Error extrayendo temas: {str(e)}")
            return {'error': str(e)}
    
    def train_topic_model(self, texts: List[str], categories: List[str] = None, 
                         n_topics: int = 10, method: str = 'lda') -> Dict[str, Any]:
        """
        Entrena un modelo de topic modeling.
        
        Args:
            texts: Lista de textos para entrenar
            categories: Categorías asociadas (opcional)
            n_topics: Número de temas
            method: Método de topic modeling
            
        Returns:
            Dict con información del modelo entrenado
        """
        try:
            logger.info(f"Entrenando topic model con {len(texts)} textos...")
            
            # Extraer temas
            result = self.extract_topics(texts, n_topics, method)
            
            if 'error' in result:
                return result
            
            # Guardar modelo
            self.topic_model = {
                'model': result,
                'method': method,
                'n_topics': n_topics,
                'topics': result.get('topics', [])
            }
            
            logger.info(f"Topic model entrenado exitosamente con {n_topics} temas")
            return {
                'status': 'success',
                'n_topics': n_topics,
                'method': method,
                'topics': result.get('topics', [])
            }
            
        except Exception as e:
            logger.error(f"Error entrenando topic model: {str(e)}")
            return {'error': str(e)}
    
    def save_topic_model(self, filepath: str = None):
        """
        Guarda el modelo de topic modeling.
        """
        try:
            if not self.topic_model:
                logger.warning("No hay modelo de topic para guardar")
                return
            
            if filepath is None:
                filepath = 'backend/ml_models/topic_model.joblib'
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Guardar modelo
            joblib.dump(self.topic_model, filepath)
            logger.info(f"Topic model guardado en {filepath}")
            
        except Exception as e:
            logger.error(f"Error guardando topic model: {str(e)}")
    
    def load_topic_model(self, filepath: str = None):
        """
        Carga el modelo de topic modeling.
        """
        try:
            if filepath is None:
                filepath = 'backend/ml_models/topic_model.joblib'
            
            if os.path.exists(filepath):
                self.topic_model = joblib.load(filepath)
                logger.info(f"Topic model cargado desde {filepath}")
            else:
                logger.warning(f"Archivo de modelo no encontrado: {filepath}")
                
        except Exception as e:
            logger.error(f"Error cargando topic model: {str(e)}")
    
    def classify_documents(self, texts: List[str], labels: List[str] = None,
                          test_size: float = 0.2) -> Dict[str, Any]:
        """
        Clasifica documentos en categorías.
        
        Args:
            texts: Lista de textos
            labels: Etiquetas de los textos (opcional)
            test_size: Proporción de datos de prueba
            
        Returns:
            Dict con resultados de clasificación
        """
        try:
            # Preprocesar textos
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Vectorizar
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            X = vectorizer.fit_transform(processed_texts)
            
            if labels is None:
                # Clustering no supervisado
                n_clusters = min(5, len(texts) // 10)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(X)
                
                return {
                    'method': 'clustering',
                    'n_clusters': n_clusters,
                    'cluster_labels': cluster_labels.tolist(),
                    'cluster_centers': kmeans.cluster_centers_.tolist()
                }
            else:
                # Clasificación supervisada
                X_train, X_test, y_train, y_test = train_test_split(
                    X, labels, test_size=test_size, random_state=42
                )
                
                # Entrenar múltiples modelos
                models = {
                    'naive_bayes': MultinomialNB(),
                    'logistic_regression': LogisticRegression(random_state=42),
                    'random_forest': RandomForestClassifier(random_state=42)
                }
                
                results = {}
                for name, model in models.items():
                    # Entrenar modelo
                    model.fit(X_train, y_train)
                    
                    # Predecir
                    y_pred = model.predict(X_test)
                    
                    # Evaluar
                    accuracy = accuracy_score(y_test, y_pred)
                    report = classification_report(y_test, y_pred, output_dict=True)
                    
                    results[name] = {
                        'accuracy': accuracy,
                        'classification_report': report,
                        'model': model
                    }
                
                # Guardar mejor modelo
                best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
                self.classifier_model = results[best_model_name]['model']
                
                return {
                    'method': 'supervised',
                    'best_model': best_model_name,
                    'results': results,
                    'vectorizer': vectorizer
                }
                
        except Exception as e:
            logger.error(f"Error clasificando documentos: {str(e)}")
            return {'error': str(e)}
    
    def generate_summary(self, text: str, max_sentences: int = 3) -> str:
        """
        Genera un resumen del texto.
        
        Args:
            text: Texto a resumir
            max_sentences: Número máximo de oraciones en el resumen
            
        Returns:
            str: Resumen generado
        """
        try:
            # Tokenizar en oraciones
            sentences = sent_tokenize(text)
            
            if len(sentences) <= max_sentences:
                return text
            
            # Preprocesar oraciones
            processed_sentences = [self.preprocess_text(sent) for sent in sentences]
            
            # Vectorizar
            vectorizer = TfidfVectorizer(stop_words='english')
            sentence_vectors = vectorizer.fit_transform(processed_sentences)
            
            # Calcular importancia de oraciones (basado en TF-IDF)
            sentence_scores = sentence_vectors.sum(axis=1).A1
            
            # Seleccionar oraciones con mayor puntuación
            top_indices = sentence_scores.argsort()[-max_sentences:][::-1]
            top_indices = sorted(top_indices)  # Mantener orden original
            
            # Construir resumen
            summary = ' '.join([sentences[i] for i in top_indices])
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generando resumen: {str(e)}")
            return text[:200] + "..." if len(text) > 200 else text
    
    def extract_keywords(self, text: str, n_keywords: int = 10) -> List[Tuple[str, float]]:
        """
        Extrae palabras clave del texto.
        
        Args:
            text: Texto a analizar
            n_keywords: Número de palabras clave a extraer
            
        Returns:
            Lista de tuplas (palabra, score)
        """
        try:
            # Preprocesar texto
            processed_text = self.preprocess_text(text)
            
            # Vectorizar
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            X = vectorizer.fit_transform([processed_text])
            
            # Obtener scores TF-IDF
            feature_names = vectorizer.get_feature_names_out()
            scores = X.toarray()[0]
            
            # Crear lista de (palabra, score)
            word_scores = list(zip(feature_names, scores))
            
            # Ordenar por score y tomar top n
            word_scores.sort(key=lambda x: x[1], reverse=True)
            
            return word_scores[:n_keywords]
            
        except Exception as e:
            logger.error(f"Error extrayendo palabras clave: {str(e)}")
            return []
    
    def analyze_text_complexity(self, text: str) -> Dict[str, float]:
        """
        Analiza la complejidad del texto.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Dict con métricas de complejidad
        """
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text.lower())
            
            # Remover puntuación
            words = [word for word in words if word.isalpha()]
            
            # Calcular métricas
            avg_sentence_length = len(words) / len(sentences) if sentences else 0
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            
            # Calcular índice de Flesch-Kincaid
            syllables = sum(self._count_syllables(word) for word in words)
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (syllables / len(words))) if words else 0
            
            # Vocabulario único
            unique_words = len(set(words))
            lexical_diversity = unique_words / len(words) if words else 0
            
            return {
                'avg_sentence_length': avg_sentence_length,
                'avg_word_length': avg_word_length,
                'flesch_reading_ease': flesch_score,
                'lexical_diversity': lexical_diversity,
                'total_words': len(words),
                'total_sentences': len(sentences),
                'unique_words': unique_words
            }
            
        except Exception as e:
            logger.error(f"Error analizando complejidad: {str(e)}")
            return {}
    
    def _count_syllables(self, word: str) -> int:
        """Cuenta sílabas en una palabra (aproximación)"""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        on_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel
        
        if word.endswith('e'):
            count -= 1
        if count == 0:
            count = 1
            
        return count
    
    def save_models(self, filepath_prefix: str):
        """
        Guarda los modelos entrenados.
        
        Args:
            filepath_prefix: Prefijo para los archivos de modelo
        """
        try:
            models_to_save = {
                'sentiment_model': self.sentiment_model,
                'topic_model': self.topic_model,
                'classifier_model': self.classifier_model
            }
            
            for name, model in models_to_save.items():
                if model is not None:
                    # Guardar en la ubicación estándar
                    filepath = f"{filepath_prefix}/nlp_{name}.joblib"
                    joblib.dump(model, filepath)
                    logger.info(f"Modelo {name} guardado en {filepath}")
                    
        except Exception as e:
            logger.error(f"Error guardando modelos: {str(e)}")
    
    def load_models(self, filepath_prefix: str):
        """
        Carga modelos guardados.
        
        Args:
            filepath_prefix: Prefijo para los archivos de modelo
        """
        try:
            models_to_load = ['sentiment_model', 'topic_model', 'classifier_model']
            
            for name in models_to_load:
                # Buscar en diferentes ubicaciones posibles
                possible_paths = [
                    f"{filepath_prefix}/nlp_{name}.joblib",  # backend/ml_models/nlp_sentiment_model.joblib
                    f"{filepath_prefix}_{name}.joblib",      # backend/ml_models_sentiment_model.joblib (legacy)
                    f"{filepath_prefix}/{name}.joblib"       # backend/ml_models/sentiment_model.joblib
                ]
                
                model_loaded = False
                for filepath in possible_paths:
                    try:
                        if os.path.exists(filepath):
                            model = joblib.load(filepath)
                            setattr(self, name, model)
                            logger.info(f"Modelo {name} cargado desde {filepath}")
                            model_loaded = True
                            break
                    except Exception as e:
                        logger.debug(f"No se pudo cargar {filepath}: {str(e)}")
                        continue
                
                if not model_loaded:
                    logger.warning(f"No se encontró modelo {name} en ninguna ubicación esperada")
                    
        except Exception as e:
            logger.error(f"Error cargando modelos: {str(e)}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del procesamiento.
        
        Returns:
            Dict con estadísticas
        """
        return {
            'language': self.language,
            'use_spacy': self.use_spacy,
            'financial_terms_count': len(self.financial_terms['financial_entities']),
            'models_loaded': {
                'sentiment_model': self.sentiment_model is not None,
                'topic_model': self.topic_model is not None,
                'classifier_model': self.classifier_model is not None
            }
        } 