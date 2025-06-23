#!/usr/bin/env python3
"""
Script para automatizar el entrenamiento de modelos de IA.

Este script puede ser ejecutado por cron o un scheduler para mantener
los modelos actualizados automáticamente.
"""

import os
import sys
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path

# --- INICIO: Configuración de Django ---
# Añadir el path del proyecto para poder importar módulos de Django
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.financialhub.settings.dev')

import django
django.setup()

from django.db.models import Avg
from backend.ai.models import AIPrediction
# --- FIN: Configuración de Django ---

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ai_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AITrainingScheduler:
    """Scheduler para entrenamiento automático de modelos de IA"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.backend_dir = self.project_root / 'backend'
        self.logs_dir = self.project_root / 'logs'
        self.performance_threshold = 0.65 # Umbral de precisión del 65%
        
        # Crear directorio de logs si no existe
        self.logs_dir.mkdir(exist_ok=True)
        
        # Configuración de entrenamiento
        self.training_config = {
            'weekly': {
                'enabled': True,
                'command': ['python', 'manage.py', 'train_ai_models', '--include-nlp'],
                'description': 'Entrenamiento semanal completo'
            },
            'monthly': {
                'enabled': True,
                'command': ['python', 'manage.py', 'train_ai_models', '--include-nlp', '--force'],
                'description': 'Entrenamiento mensual forzado'
            },
            'nlp_only': {
                'enabled': False,  # Solo cuando sea necesario
                'command': ['python', 'manage.py', 'train_ai_models', '--nlp-only', '--force'],
                'description': 'Solo modelos de NLP'
            },
            'evaluate': {
                'enabled': True,
                'command': ['python', 'manage.py', 'evaluate_predictions', '--force'],
                'description': 'Evaluación de rendimiento de modelos'
            }
        }
    
    def _execute_command(self, command_key: str) -> bool:
        """Ejecuta un comando de gestión de Django."""
        config = self.training_config.get(command_key)
        if not config or not config['enabled']:
            logger.info(f"Comando '{command_key}' deshabilitado o no encontrado.")
            return False
        
        logger.info(f"Iniciando: {config['description']}...")
        
        try:
            # Asegurarse de estar en el directorio correcto
            os.chdir(self.backend_dir)
            
            result = subprocess.run(
                config['command'],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hora de timeout
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {config['description']} completado exitosamente.")
                logger.debug(f"STDOUT: {result.stdout}")
                return True
            else:
                logger.error(f"❌ Error en {config['description']}.")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Timeout en {config['description']}.")
            return False
        except Exception as e:
            logger.error(f"❌ Error ejecutando '{config['description']}': {e}")
            return False

    def get_model_performance_scores(self) -> dict:
        """Consulta la BD y devuelve el score de precisión promedio por tipo de modelo."""
        logger.info("Consultando scores de precisión de los modelos...")
        try:
            scores = AIPrediction.objects.filter(
                accuracy_score__isnull=False
            ).values('type').annotate(
                average_accuracy=Avg('accuracy_score')
            )
            
            performance_map = {item['type']: item['average_accuracy'] for item in scores}
            logger.info(f"Scores de rendimiento obtenidos: {performance_map}")
            return performance_map
        except Exception as e:
            logger.error(f"No se pudo obtener el rendimiento de los modelos: {e}")
            return {}

    def run_intelligent_training_cycle(self):
        """
        Ejecuta el ciclo de IA completo: Evaluar -> Decidir -> Entrenar.
        """
        logger.info("🤖 Iniciando ciclo de entrenamiento inteligente...")

        # 1. Evaluar el rendimiento actual de todos los modelos
        evaluation_success = self._execute_command('evaluate')
        if not evaluation_success:
            logger.error("El ciclo no puede continuar porque la evaluación de modelos falló.")
            return

        # 2. Obtener los scores de la base de datos
        performance_scores = self.get_model_performance_scores()
        if not performance_scores:
            logger.warning("No se encontraron scores de rendimiento. Se procederá con entrenamiento estándar.")
            self.execute_training('weekly')
            return
            
        # 3. Decidir si algún modelo necesita re-entrenamiento
        models_to_retrain = []
        for model_type, avg_accuracy in performance_scores.items():
            if avg_accuracy < self.performance_threshold:
                logger.warning(f"📉 El modelo '{model_type}' está por debajo del umbral ({avg_accuracy:.2%} < {self.performance_threshold:.0%}). Necesita re-entrenamiento.")
                models_to_retrain.append(model_type)

        # 4. Ejecutar re-entrenamiento si es necesario
        if models_to_retrain:
            logger.info(f"Se re-entrenarán los modelos debido al bajo rendimiento de: {', '.join(models_to_retrain)}")
            training_success = self.execute_training('weekly') # Ejecuta el entrenamiento completo
            if training_success:
                self._record_training_date('weekly')
        else:
            logger.info("✅ Todos los modelos están funcionando por encima del umbral de rendimiento. No se necesita re-entrenamiento.")

    def execute_training(self, training_type: str) -> bool:
        """Ejecuta el entrenamiento especificado"""
        return self._execute_command(training_type)
    
    def _record_training_date(self, training_type: str):
        """Registra la fecha del entrenamiento"""
        try:
            timestamp_file = self.logs_dir / f'last_{training_type}_training.txt'
            with open(timestamp_file, 'w') as f:
                f.write(datetime.now().isoformat())
        except Exception as e:
            logger.warning(f"Error registrando fecha de entrenamiento: {e}")
    
    def run_scheduled_training(self):
        """DEPRECATED: Reemplazado por run_intelligent_training_cycle"""
        logger.warning("El método run_scheduled_training está obsoleto. Usando run_intelligent_training_cycle en su lugar.")
        self.run_intelligent_training_cycle()
    
    def run_manual_training(self, training_type: str = 'nlp_only'):
        """Ejecuta entrenamiento manual"""
        logger.info(f"🔧 Ejecutando entrenamiento manual: {training_type}")
        return self.execute_training(training_type)
    
    def get_training_status(self) -> dict:
        """Obtiene el estado de los entrenamientos"""
        status = {}
        
        for training_type in ['weekly', 'monthly']:
            try:
                timestamp_file = self.logs_dir / f'last_{training_type}_training.txt'
                
                if timestamp_file.exists():
                    with open(timestamp_file, 'r') as f:
                        last_date_str = f.read().strip()
                        last_date = datetime.fromisoformat(last_date_str)
                        days_ago = (datetime.now() - last_date).days
                        
                    status[training_type] = {
                        'last_training': last_date.isoformat(),
                        'days_ago': days_ago,
                        'next_due': max(0, 7 if training_type == 'weekly' else 30 - days_ago)
                    }
                else:
                    status[training_type] = {
                        'last_training': 'Nunca',
                        'days_ago': float('inf'),
                        'next_due': 0
                    }
                    
            except Exception as e:
                status[training_type] = {
                    'error': str(e)
                }
        
        return status

def main():
    """Función principal"""
    scheduler = AITrainingScheduler()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'scheduled':
            # Ejecutar el nuevo ciclo inteligente
            scheduler.run_intelligent_training_cycle()
            
        elif command == 'manual':
            # Entrenamiento manual
            training_type = sys.argv[2] if len(sys.argv) > 2 else 'nlp_only'
            scheduler.run_manual_training(training_type)
            
        elif command == 'status':
            # Mostrar estado
            status = scheduler.get_training_status()
            print("📊 Estado de Entrenamientos:")
            for training_type, info in status.items():
                print(f"  {training_type.upper()}:")
                for key, value in info.items():
                    print(f"    {key}: {value}")
                    
        elif command == 'weekly':
            # Entrenamiento semanal manual
            scheduler.run_manual_training('weekly')
            
        elif command == 'monthly':
            # Entrenamiento mensual manual
            scheduler.run_manual_training('monthly')
            
        else:
            print("Comandos disponibles:")
            print("  scheduled  - Ejecutar entrenamientos programados")
            print("  manual     - Entrenamiento manual (nlp_only por defecto)")
            print("  status     - Mostrar estado de entrenamientos")
            print("  weekly     - Entrenamiento semanal manual")
            print("  monthly    - Entrenamiento mensual manual")
    else:
        # Por defecto, ejecutar el ciclo inteligente
        scheduler.run_intelligent_training_cycle()

if __name__ == "__main__":
    main() 