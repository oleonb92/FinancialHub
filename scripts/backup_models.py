#!/usr/bin/env python3
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
