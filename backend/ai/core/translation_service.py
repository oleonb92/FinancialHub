"""
Translation Service Module

This module provides a centralized translation service using deep-translator
for runtime translations, replacing Django's locale-based i18n system.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import locale

# Try to import deep-translator
try:
    from deep_translator import GoogleTranslator
    DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    DEEP_TRANSLATOR_AVAILABLE = False
    logging.warning("deep-translator not available, translations disabled")

# Try to import langdetect
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0  # For consistent results
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect not available, language detection disabled")

logger = logging.getLogger(__name__)


class TranslationService:
    """
    Centralized translation service using deep-translator for runtime translations.
    
    This service replaces Django's locale-based i18n system with a more flexible
    runtime translation approach using Google Translate API via deep-translator.
    """
    
    def __init__(self, default_language: str = 'en'):
        """
        Initialize the translation service.
        
        Args:
            default_language: Default language code ('en' or 'es')
        """
        self.default_language = default_language
        self.supported_languages = ['en', 'es']
        
        # Translation cache to avoid repeated API calls
        self.translation_cache = {}
        
        # Currency and formatting configurations
        self.currency_formats = {
            'en': {
                'USD': {'symbol': '$', 'position': 'before', 'decimal_places': 2},
                'EUR': {'symbol': '€', 'position': 'before', 'decimal_places': 2},
                'GBP': {'symbol': '£', 'position': 'before', 'decimal_places': 2}
            },
            'es': {
                'USD': {'symbol': '$', 'position': 'before', 'decimal_places': 2},
                'EUR': {'symbol': '€', 'position': 'before', 'decimal_places': 2},
                'MXN': {'symbol': '$', 'position': 'before', 'decimal_places': 2}
            }
        }
        
        self.date_formats = {
            'en': {
                'short': '%m/%d/%Y',
                'long': '%B %d, %Y',
                'month_year': '%B %Y'
            },
            'es': {
                'short': '%d/%m/%Y',
                'long': '%d de %B de %Y',
                'month_year': '%B de %Y'
            }
        }
        
        self.number_formats = {
            'en': {'decimal_separator': '.', 'thousands_separator': ','},
            'es': {'decimal_separator': ',', 'thousands_separator': '.'}
        }
    
    def translate(self, text: str, target_lang: str = None) -> str:
        """
        Translate text to the target language using deep-translator.
        
        Args:
            text: Text to translate
            target_lang: Target language code ('en' or 'es')
            
        Returns:
            Translated text or original text if translation fails
        """
        if not DEEP_TRANSLATOR_AVAILABLE:
            logger.warning("deep-translator not available, returning original text")
            return text
        
        if not text or not text.strip():
            return text
        
        # Use default language if not specified
        if target_lang is None:
            target_lang = self.default_language
        
        # Don't translate if target language is not supported
        if target_lang not in self.supported_languages:
            logger.warning(f"Unsupported target language: {target_lang}, falling back to {self.default_language}")
            target_lang = self.default_language
        
        # Check cache first
        cache_key = f"{text}_{target_lang}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        try:
            # Detect source language
            source_lang = self.detect_language(text)
            
            # Don't translate if source and target are the same
            if source_lang == target_lang:
                self.translation_cache[cache_key] = text
                return text
            
            # Translate using Google Translator
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            translated_text = translator.translate(text)
            
            # Cache the result
            self.translation_cache[cache_key] = translated_text
            
            logger.debug(f"Translated '{text}' from {source_lang} to {target_lang}: '{translated_text}'")
            return translated_text
            
        except Exception as e:
            logger.error(f"Translation error for '{text}' to {target_lang}: {e}")
            # Return original text on error
            return text
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the given text.
        
        Args:
            text: Text to detect language for
            
        Returns:
            Language code ('en' or 'es') or 'en' as fallback
        """
        if not LANGDETECT_AVAILABLE:
            logger.warning("langdetect not available, defaulting to English")
            return 'en'
        
        if not text or not text.strip():
            return self.default_language
        
        try:
            detected_lang = detect(text)
            
            # Map detected language to supported languages
            if detected_lang in ['en', 'es']:
                return detected_lang
            else:
                # For other languages, default to English
                logger.debug(f"Unsupported detected language: {detected_lang}, defaulting to English")
                return 'en'
                
        except Exception as e:
            logger.error(f"Language detection error for '{text}': {e}")
            return self.default_language
    
    def format_currency(self, amount: float, language: str = None, currency: str = 'USD') -> str:
        """
        Format currency amount according to language-specific formatting.
        
        Args:
            amount: Amount to format
            language: Language code ('en' or 'es')
            currency: Currency code
            
        Returns:
            Formatted currency string
        """
        if language is None:
            language = self.default_language
        
        if language not in self.supported_languages:
            language = self.default_language
        
        # Get currency format configuration
        currency_config = self.currency_formats.get(language, {}).get(currency, {})
        symbol = currency_config.get('symbol', '$')
        position = currency_config.get('position', 'before')
        decimal_places = currency_config.get('decimal_places', 2)
        
        # Format the number
        formatted_number = self.format_number(amount, language, decimal_places)
        
        # Add currency symbol
        if position == 'before':
            return f"{symbol}{formatted_number}"
        else:
            return f"{formatted_number}{symbol}"
    
    def format_number(self, number: float, language: str = None, decimal_places: int = 2) -> str:
        """
        Format number according to language-specific formatting.
        
        Args:
            number: Number to format
            language: Language code ('en' or 'es')
            decimal_places: Number of decimal places
            
        Returns:
            Formatted number string
        """
        if language is None:
            language = self.default_language
        
        if language not in self.supported_languages:
            language = self.default_language
        
        # Get number format configuration
        number_config = self.number_formats.get(language, {})
        decimal_sep = number_config.get('decimal_separator', '.')
        thousands_sep = number_config.get('thousands_separator', ',')
        
        # Format the number
        try:
            # Use locale for proper formatting
            if language == 'en':
                locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
            elif language == 'es':
                locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')
            
            formatted = locale.format_string(f"%.{decimal_places}f", number, grouping=True)
            
            # Replace separators if needed
            if decimal_sep != '.':
                formatted = formatted.replace('.', decimal_sep)
            if thousands_sep != ',':
                formatted = formatted.replace(',', thousands_sep)
            
            return formatted
            
        except Exception as e:
            logger.error(f"Number formatting error: {e}")
            # Fallback to simple formatting
            return f"{number:.{decimal_places}f}"
    
    def format_date(self, date: datetime, format_type: str = 'long', language: str = None) -> str:
        """
        Format date according to language-specific formatting.
        
        Args:
            date: Date to format
            format_type: Format type ('short', 'long', 'month_year')
            language: Language code ('en' or 'es')
            
        Returns:
            Formatted date string
        """
        if language is None:
            language = self.default_language
        
        if language not in self.supported_languages:
            language = self.default_language
        
        # Get date format configuration
        date_format = self.date_formats.get(language, {}).get(format_type, '%Y-%m-%d')
        
        try:
            # Use locale for proper formatting
            if language == 'en':
                locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
            elif language == 'es':
                locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')
            
            return date.strftime(date_format)
            
        except Exception as e:
            logger.error(f"Date formatting error: {e}")
            # Fallback to simple formatting
            return date.strftime('%Y-%m-%d')
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.supported_languages.copy()
    
    def clear_cache(self):
        """Clear the translation cache."""
        self.translation_cache.clear()
        logger.debug("Translation cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get translation cache statistics."""
        return {
            'cache_size': len(self.translation_cache),
            'supported_languages': self.supported_languages,
            'default_language': self.default_language
        }


# Global instance for convenience
_translation_service = None


def get_translation_service() -> TranslationService:
    """Get the global translation service instance."""
    global _translation_service
    if _translation_service is None:
        _translation_service = TranslationService()
    return _translation_service


def translate(text: str, target_lang: str = None) -> str:
    """
    Convenience function for translation.
    
    Args:
        text: Text to translate
        target_lang: Target language code
        
    Returns:
        Translated text
    """
    return get_translation_service().translate(text, target_lang)


def detect_language(text: str) -> str:
    """
    Convenience function for language detection.
    
    Args:
        text: Text to detect language for
        
    Returns:
        Language code
    """
    return get_translation_service().detect_language(text)


def format_currency(amount: float, language: str = None, currency: str = 'USD') -> str:
    """
    Convenience function for currency formatting.
    
    Args:
        amount: Amount to format
        language: Language code
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    return get_translation_service().format_currency(amount, language, currency)


def format_date(date: datetime, format_type: str = 'long', language: str = None) -> str:
    """
    Convenience function for date formatting.
    
    Args:
        date: Date to format
        format_type: Format type
        language: Language code
        
    Returns:
        Formatted date string
    """
    return get_translation_service().format_date(date, format_type, language) 