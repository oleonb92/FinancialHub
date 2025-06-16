import logging
import os
from logging.handlers import RotatingFileHandler
from django.conf import settings

def setup_logging():
    """Configure logging for the application"""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(settings.BASE_DIR, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler for general logs
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'financialhub.log'),
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)

    # File handler for error logs
    error_handler = RotatingFileHandler(
        os.path.join(log_dir, 'error.log'),
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    error_handler.setLevel(logging.ERROR)
    error_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
        'Path: %(pathname)s\n'
        'Line: %(lineno)d\n'
        'Function: %(funcName)s\n'
        'Exception: %(exc_info)s'
    )
    error_handler.setFormatter(error_format)
    root_logger.addHandler(error_handler)

    # Set specific loggers
    loggers = {
        'django': logging.INFO,
        'django.request': logging.ERROR,
        'django.db.backends': logging.WARNING,
        'stripe': logging.INFO,
        'payments': logging.INFO,
        'access_control': logging.INFO,
    }

    for logger_name, level in loggers.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)

    return root_logger 