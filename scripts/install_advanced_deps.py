#!/usr/bin/env python3
"""
Script de instalación de dependencias avanzadas para el sistema de AI.

Este script instala todas las dependencias necesarias para:
- AutoML y optimización automática
- Federated Learning
- A/B Testing y experimentación
- NLP avanzado
- Transformers personalizados
- Monitoreo de recursos
"""

import subprocess
import sys
import os
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Ejecuta un comando y maneja errores"""
    logger.info(f"Ejecutando: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error en {description}: {e}")
        logger.error(f"Salida de error: {e.stderr}")
        return False

def install_python_packages():
    """Instala paquetes de Python necesarios"""
    packages = [
        # Core ML
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        
        # Deep Learning
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "tensorflow>=2.13.0",
        
        # AutoML
        "optuna>=3.2.0",
        "hyperopt>=0.2.7",
        "autosklearn>=0.15.0",
        "flaml>=1.2.0",
        
        # NLP
        "nltk>=3.8.1",
        "spacy>=3.6.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.2",
        "textblob>=0.17.1",
        
        # Experimentación
        "mlflow>=2.5.0",
        "wandb>=0.15.0",
        "optuna-dashboard>=0.10.0",
        
        # Monitoreo
        "psutil>=5.9.0",
        "prometheus-client>=0.17.0",
        "grafana-api>=1.0.3",
        
        # Visualización
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "bokeh>=3.1.0",
        
        # Utilidades
        "joblib>=1.3.0",
        "pickle5>=0.0.12",
        "tqdm>=4.65.0",
        "click>=8.1.0",
        "rich>=13.3.0",
        
        # Caché y almacenamiento
        "redis>=4.6.0",
        "pymongo>=4.3.0",
        "elasticsearch>=8.8.0",
        
        # Seguridad y privacidad
        "cryptography>=41.0.0",
        "pycryptodome>=3.18.0",
        
        # Testing
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.11.0",
        
        # Desarrollo
        "black>=23.7.0",
        "flake8>=6.0.0",
        "mypy>=1.5.0",
        "pre-commit>=3.3.0"
    ]
    
    logger.info("Instalando paquetes de Python...")
    
    for package in packages:
        success = run_command(f"pip install {package}", f"Instalando {package}")
        if not success:
            logger.warning(f"⚠️ No se pudo instalar {package}, continuando...")

def install_spacy_models():
    """Instala modelos de spaCy"""
    models = [
        "en_core_web_sm",
        "en_core_web_md",
        "es_core_news_sm"
    ]
    
    logger.info("Instalando modelos de spaCy...")
    
    for model in models:
        success = run_command(f"python -m spacy download {model}", f"Instalando modelo {model}")
        if not success:
            logger.warning(f"⚠️ No se pudo instalar modelo {model}, continuando...")

def install_nltk_data():
    """Descarga datos de NLTK"""
    logger.info("Descargando datos de NLTK...")
    
    nltk_commands = [
        "python -c \"import nltk; nltk.download('punkt')\"",
        "python -c \"import nltk; nltk.download('stopwords')\"",
        "python -c \"import nltk; nltk.download('wordnet')\"",
        "python -c \"import nltk; nltk.download('averaged_perceptron_tagger')\"",
        "python -c \"import nltk; nltk.download('maxent_ne_chunker')\"",
        "python -c \"import nltk; nltk.download('words')\"",
        "python -c \"import nltk; nltk.download('vader_lexicon')\"",
        "python -c \"import nltk; nltk.download('omw-1.4')\""
    ]
    
    for command in nltk_commands:
        success = run_command(command, "Descargando datos de NLTK")
        if not success:
            logger.warning("⚠️ Error descargando datos de NLTK, continuando...")

def setup_redis():
    """Configura Redis"""
    logger.info("Configurando Redis...")
    
    # Verificar si Redis está instalado
    if sys.platform.startswith('linux'):
        run_command("sudo apt-get update", "Actualizando repositorios")
        run_command("sudo apt-get install -y redis-server", "Instalando Redis")
        run_command("sudo systemctl start redis-server", "Iniciando Redis")
        run_command("sudo systemctl enable redis-server", "Habilitando Redis")
    elif sys.platform.startswith('darwin'):
        run_command("brew install redis", "Instalando Redis con Homebrew")
        run_command("brew services start redis", "Iniciando Redis")
    else:
        logger.warning("⚠️ Instalación automática de Redis no soportada en este sistema")

def setup_celery():
    """Configura Celery"""
    logger.info("Configurando Celery...")
    
    # Instalar dependencias adicionales para Celery
    celery_packages = [
        "celery>=5.3.0",
        "flower>=2.0.0",
        "django-celery-beat>=2.5.0",
        "django-celery-results>=2.5.0"
    ]
    
    for package in celery_packages:
        run_command(f"pip install {package}", f"Instalando {package}")

def create_directories():
    """Crea directorios necesarios"""
    logger.info("Creando directorios necesarios...")
    
    directories = [
        "backend/ml_models",
        "backend/ml_models/nlp",
        "backend/ml_models/transformer",
        "backend/ml_models/automl",
        "backend/ml_models/federated",
        "backend/logs",
        "backend/cache",
        "backend/temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Directorio creado: {directory}")

def setup_environment():
    """Configura variables de entorno"""
    logger.info("Configurando variables de entorno...")
    
    env_content = """
# Configuración de AI
AI_ENABLED=true
AI_MODELS_PATH=backend/ml_models
AI_CACHE_ENABLED=true
AI_MONITORING_ENABLED=true

# Configuración de Redis
REDIS_URL=redis://localhost:6379/0
REDIS_CACHE_TIMEOUT=3600

# Configuración de Celery
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# Configuración de MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=financial_hub_ai

# Configuración de monitoreo
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true

# Configuración de privacidad
FEDERATED_LEARNING_ENABLED=true
PRIVACY_BUDGET=1.0

# Configuración de experimentación
AB_TESTING_ENABLED=true
EXPERIMENT_TRACKING_ENABLED=true
"""
    
    with open(".env", "a") as f:
        f.write(env_content)
    
    logger.info("✅ Variables de entorno configuradas")

def run_tests():
    """Ejecuta tests básicos"""
    logger.info("Ejecutando tests básicos...")
    
    test_commands = [
        "python -c \"import torch; print(f'PyTorch version: {torch.__version__}')\"",
        "python -c \"import sklearn; print(f'scikit-learn version: {sklearn.__version__}')\"",
        "python -c \"import spacy; print('spaCy installed successfully')\"",
        "python -c \"import nltk; print('NLTK installed successfully')\"",
        "python -c \"import redis; print('Redis client installed successfully')\"",
        "python -c \"import celery; print('Celery installed successfully')\""
    ]
    
    for command in test_commands:
        success = run_command(command, "Ejecutando test")
        if not success:
            logger.warning("⚠️ Test falló, pero continuando...")

def main():
    """Función principal"""
    logger.info("🚀 Iniciando instalación de dependencias avanzadas de AI...")
    
    # Verificar Python
    if sys.version_info < (3, 8):
        logger.error("❌ Se requiere Python 3.8 o superior")
        sys.exit(1)
    
    logger.info(f"✅ Python {sys.version} detectado")
    
    # Instalar paquetes
    install_python_packages()
    
    # Instalar modelos de spaCy
    install_spacy_models()
    
    # Descargar datos de NLTK
    install_nltk_data()
    
    # Configurar Redis
    setup_redis()
    
    # Configurar Celery
    setup_celery()
    
    # Crear directorios
    create_directories()
    
    # Configurar variables de entorno
    setup_environment()
    
    # Ejecutar tests
    run_tests()
    
    logger.info("🎉 Instalación completada exitosamente!")
    logger.info("📋 Próximos pasos:")
    logger.info("   1. Configura las variables de entorno en .env")
    logger.info("   2. Ejecuta las migraciones de Django")
    logger.info("   3. Inicia Redis y Celery")
    logger.info("   4. Ejecuta el script de entrenamiento de modelos")

if __name__ == "__main__":
    main() 