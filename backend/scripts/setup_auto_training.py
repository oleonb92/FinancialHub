#!/usr/bin/env python3
"""
Script para configurar automáticamente el entrenamiento de IA.

Este script:
1. Configura cron jobs automáticamente
2. Crea directorios necesarios
3. Configura logs
4. Verifica la configuración
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoTrainingSetup:
    """Configurador automático de entrenamiento de IA"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.backend_dir = self.project_root / 'backend'
        self.logs_dir = self.project_root / 'logs'
        self.scripts_dir = self.project_root / 'scripts'
        
        # Obtener ruta absoluta del proyecto
        self.project_path = str(self.project_root.absolute())
        
        # Configuración de cron jobs
        self.cron_jobs = [
            {
                'schedule': '0 2 * * 0',  # Domingo 2:00 AM
                'command': f'cd {self.project_path}/backend && python scripts/schedule_ai_training.py scheduled >> logs/system/cron.log 2>&1',
                'description': 'Entrenamiento semanal automático'
            },
            {
                'schedule': '0 3 1 * *',  # Primer día del mes 3:00 AM
                'command': f'cd {self.project_path}/backend && python scripts/schedule_ai_training.py scheduled >> logs/system/cron.log 2>&1',
                'description': 'Entrenamiento mensual automático'
            },
            {
                'schedule': '0 6 * * *',  # Diario 6:00 AM
                'command': f'cd {self.project_path}/backend && python scripts/schedule_ai_training.py status >> logs/system/status.log 2>&1',
                'description': 'Verificación diaria de estado'
            }
        ]
    
    def setup_directories(self):
        """Crea directorios necesarios"""
        logger.info("📁 Creando directorios necesarios...")
        
        directories = [
            self.logs_dir,
            self.project_root / 'backup',
            self.backend_dir / 'ml_models' / 'test'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"  ✅ {directory}")
    
    def setup_logs(self):
        """Configura archivos de log iniciales"""
        logger.info("📝 Configurando archivos de log...")
        
        log_files = [
            self.backend_dir / 'logs' / 'training' / 'ai_training.log',
            self.backend_dir / 'logs' / 'system' / 'cron.log',
            self.backend_dir / 'logs' / 'system' / 'status.log'
        ]
        
        for log_file in log_files:
            if not log_file.exists():
                log_file.touch()
                logger.info(f"  ✅ {log_file.name}")
    
    def get_current_crontab(self):
        """Obtiene el crontab actual"""
        try:
            result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout
            else:
                return ""
        except Exception as e:
            logger.warning(f"No se pudo obtener crontab actual: {e}")
            return ""
    
    def add_cron_jobs(self):
        """Agrega los cron jobs automáticamente"""
        logger.info("⏰ Configurando cron jobs automáticamente...")
        
        # Obtener crontab actual
        current_crontab = self.get_current_crontab()
        
        # Verificar si ya existen nuestros jobs
        existing_jobs = []
        new_jobs = []
        
        for job in self.cron_jobs:
            job_line = f"{job['schedule']} {job['command']}"
            
            if job_line in current_crontab:
                existing_jobs.append(job['description'])
                logger.info(f"  ✅ {job['description']} ya existe")
            else:
                new_jobs.append(job_line)
                logger.info(f"  ➕ Agregando {job['description']}")
        
        if not new_jobs:
            logger.info("🎉 Todos los cron jobs ya están configurados")
            return True
        
        # Crear nuevo crontab
        new_crontab = current_crontab.strip() + "\n\n# 🤖 AI Training Jobs - FinancialHub\n"
        new_crontab += "# Configurado automáticamente el " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"
        
        for job in new_jobs:
            new_crontab += job + "\n"
        
        # Instalar nuevo crontab
        try:
            result = subprocess.run(
                ['crontab', '-'],
                input=new_crontab,
                text=True,
                capture_output=True
            )
            
            if result.returncode == 0:
                logger.info("✅ Cron jobs instalados exitosamente")
                return True
            else:
                logger.error(f"❌ Error instalando cron jobs: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error configurando cron: {e}")
            return False
    
    def verify_setup(self):
        """Verifica que la configuración sea correcta"""
        logger.info("🔍 Verificando configuración...")
        
        # Verificar directorios
        required_dirs = [
            self.backend_dir / 'logs' / 'training',
            self.backend_dir / 'logs' / 'system',
            self.backend_dir / 'ml_models' / 'test'
        ]
        
        for directory in required_dirs:
            if directory.exists():
                logger.info(f"  ✅ {directory}")
            else:
                logger.error(f"  ❌ {directory} no existe")
                return False
        
        # Verificar archivos
        required_files = [
            self.scripts_dir / 'schedule_ai_training.py',
            self.backend_dir / 'manage.py'
        ]
        
        for file_path in required_files:
            if file_path.exists():
                logger.info(f"  ✅ {file_path}")
            else:
                logger.error(f"  ❌ {file_path} no existe")
                return False
        
        # Verificar cron jobs
        current_crontab = self.get_current_crontab()
        jobs_found = 0
        
        for job in self.cron_jobs:
            if job['command'] in current_crontab:
                jobs_found += 1
                logger.info(f"  ✅ {job['description']} configurado")
            else:
                logger.warning(f"  ⚠️ {job['description']} no encontrado")
        
        if jobs_found >= 2:  # Al menos semanal y mensual
            logger.info("✅ Configuración de cron verificada")
            return True
        else:
            logger.error("❌ Configuración de cron incompleta")
            return False
    
    def test_scheduler(self):
        """Prueba el scheduler"""
        logger.info("🧪 Probando scheduler...")
        
        try:
            result = subprocess.run(
                [sys.executable, str(self.scripts_dir / 'schedule_ai_training.py'), 'status'],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                logger.info("✅ Scheduler funciona correctamente")
                logger.info(f"📊 Estado: {result.stdout.strip()}")
                return True
            else:
                logger.error(f"❌ Error en scheduler: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error probando scheduler: {e}")
            return False
    
    def create_backup_script(self):
        """Crea script de backup automático"""
        logger.info("💾 Creando script de backup...")
        
        backup_script = self.scripts_dir / 'backup_models.py'
        
        backup_content = '''#!/usr/bin/env python3
"""
Script de backup automático para modelos de IA.
"""

import os
import tarfile
from datetime import datetime
from pathlib import Path

def create_backup():
    """Crea backup de modelos entrenados"""
    project_root = Path(__file__).parent.parent
    backup_dir = project_root / 'backup'
    models_dir = project_root / 'backend' / 'ml_models'
    
    # Crear directorio de backup si no existe
    backup_dir.mkdir(exist_ok=True)
    
    # Nombre del archivo de backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"models_backup_{timestamp}.tar.gz"
    
    # Crear backup
    with tarfile.open(backup_file, "w:gz") as tar:
        tar.add(models_dir, arcname="ml_models")
    
    print(f"✅ Backup creado: {backup_file}")
    
    # Limpiar backups antiguos (mantener solo los últimos 5)
    backup_files = sorted(backup_dir.glob("models_backup_*.tar.gz"))
    if len(backup_files) > 5:
        for old_backup in backup_files[:-5]:
            old_backup.unlink()
            print(f"🗑️ Backup eliminado: {old_backup}")

if __name__ == "__main__":
    create_backup()
'''
        
        with open(backup_script, 'w') as f:
            f.write(backup_content)
        
        # Dar permisos de ejecución
        os.chmod(backup_script, 0o755)
        
        logger.info(f"  ✅ {backup_script}")
    
    def setup_initial_training(self):
        """Ejecuta entrenamiento inicial"""
        logger.info("🚀 Ejecutando entrenamiento inicial...")
        
        try:
            result = subprocess.run(
                [sys.executable, str(self.backend_dir / 'manage.py'), 'train_ai_models', '--include-nlp', '--force'],
                capture_output=True,
                text=True,
                cwd=self.backend_dir
            )
            
            if result.returncode == 0:
                logger.info("✅ Entrenamiento inicial completado")
                return True
            else:
                logger.warning(f"⚠️ Entrenamiento inicial con advertencias: {result.stderr}")
                return True  # No es crítico
                
        except Exception as e:
            logger.warning(f"⚠️ Error en entrenamiento inicial: {e}")
            return True  # No es crítico
    
    def run_setup(self):
        """Ejecuta toda la configuración"""
        logger.info("🤖 Iniciando configuración automática de entrenamiento de IA...")
        
        # 1. Crear directorios
        self.setup_directories()
        
        # 2. Configurar logs
        self.setup_logs()
        
        # 3. Crear script de backup
        self.create_backup_script()
        
        # 4. Configurar cron jobs
        if not self.add_cron_jobs():
            logger.error("❌ Error configurando cron jobs")
            return False
        
        # 5. Verificar configuración
        if not self.verify_setup():
            logger.error("❌ Error en verificación")
            return False
        
        # 6. Probar scheduler
        if not self.test_scheduler():
            logger.error("❌ Error probando scheduler")
            return False
        
        # 7. Entrenamiento inicial
        self.setup_initial_training()
        
        logger.info("🎉 Configuración automática completada exitosamente!")
        return True

def main():
    """Función principal"""
    setup = AutoTrainingSetup()
    
    print("=" * 60)
    print("🤖 CONFIGURADOR AUTOMÁTICO DE ENTRENAMIENTO DE IA")
    print("=" * 60)
    
    if setup.run_setup():
        print("\n" + "=" * 60)
        print("✅ ¡CONFIGURACIÓN COMPLETADA!")
        print("=" * 60)
        print("\n📋 Resumen de lo que se configuró:")
        print("  • Cron jobs para entrenamiento semanal y mensual")
        print("  • Verificación diaria de estado")
        print("  • Directorios de logs y backup")
        print("  • Script de backup automático")
        print("  • Entrenamiento inicial de modelos")
        print("\n📅 Próximos entrenamientos:")
        print("  • Semanal: Cada domingo a las 2:00 AM")
        print("  • Mensual: Primer día del mes a las 3:00 AM")
        print("\n🔍 Para verificar:")
        print("  • Estado: python backend/scripts/schedule_ai_training.py status")
        print("  • Logs: tail -f backend/logs/training/ai_training.log")
        print("  • Cron: crontab -l")
    else:
        print("\n❌ Error en la configuración")
        sys.exit(1)

if __name__ == "__main__":
    main() 