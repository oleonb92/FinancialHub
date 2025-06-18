#!/usr/bin/env python
import os
import sys
import django
from django.core.management import call_command
from django.conf import settings

def setup_test_db():
    """
    Configura la base de datos de test y aplica las migraciones.
    """
    print("Configurando base de datos de test...")
    
    # Asegurarse de que estamos en el entorno de test
    os.environ['DJANGO_SETTINGS_MODULE'] = 'financialhub.settings'
    os.environ['TESTING'] = 'True'
    
    django.setup()
    
    try:
        # Aplicar migraciones
        call_command('migrate', '--noinput')
        print("✅ Migraciones aplicadas correctamente")
        
        # Verificar que las tablas se crearon
        from django.db import connection
        with connection.cursor() as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"📊 Tablas creadas: {len(tables)}")
            for table in tables:
                print(f"  - {table[0]}")
                
    except Exception as e:
        print(f"❌ Error al configurar la base de datos de test: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    setup_test_db() 