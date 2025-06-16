import os
from django.core.management.base import BaseCommand
from elasticsearch import Elasticsearch
from elasticsearch_dsl import connections
from transactions.documents import TransactionDocument

class Command(BaseCommand):
    help = 'Prueba la conexión con Elasticsearch y el índice de transacciones'

    def handle(self, *args, **options):
        # Configurar la conexión
        connections.configure(
            default={
                'hosts': os.getenv('ELASTICSEARCH_HOST', 'http://localhost:9200'),
                'timeout': 30,
                'retry_on_timeout': True,
                'max_retries': 3,
            }
        )

        # Probar conexión básica
        es = Elasticsearch(os.getenv('ELASTICSEARCH_HOST', 'http://localhost:9200'))
        self.stdout.write('Probando conexión con Elasticsearch...')
        
        try:
            info = es.info()
            self.stdout.write(self.style.SUCCESS(f'Conexión exitosa con Elasticsearch v{info["version"]["number"]}'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error al conectar con Elasticsearch: {str(e)}'))
            return

        # Probar índice de transacciones
        self.stdout.write('Probando índice de transacciones...')
        try:
            # Crear el índice si no existe
            TransactionDocument.init()
            self.stdout.write(self.style.SUCCESS('Índice de transacciones creado/verificado correctamente'))
            
            # Intentar indexar una transacción de prueba
            doc = TransactionDocument()
            doc.update(doc.get_queryset())
            self.stdout.write(self.style.SUCCESS('Transacciones indexadas correctamente'))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error al trabajar con el índice: {str(e)}')) 