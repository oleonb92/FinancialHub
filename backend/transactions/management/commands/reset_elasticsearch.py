from django.core.management.base import BaseCommand
from elasticsearch_dsl import connections
from transactions.documents import TransactionDocument

class Command(BaseCommand):
    help = 'Resetea el índice de Elasticsearch y reindexa las transacciones'

    def handle(self, *args, **options):
        # Configurar la conexión
        connections.configure(
            default={
                'hosts': 'http://localhost:9200',
                'timeout': 30,
                'retry_on_timeout': True,
                'max_retries': 3,
            }
        )

        # Eliminar el índice si existe
        if TransactionDocument._index.exists():
            self.stdout.write('Eliminando índice existente...')
            TransactionDocument._index.delete()
            self.stdout.write(self.style.SUCCESS('Índice eliminado correctamente'))

        # Crear el índice con la nueva configuración
        self.stdout.write('Creando nuevo índice...')
        TransactionDocument.init()
        self.stdout.write(self.style.SUCCESS('Índice creado correctamente'))

        # Reindexar las transacciones
        self.stdout.write('Reindexando transacciones...')
        doc = TransactionDocument()
        success, errors = doc.update(doc.get_queryset())
        
        if errors:
            self.stdout.write(self.style.WARNING(f'Se encontraron {len(errors)} errores durante la indexación'))
            for error in errors:
                self.stdout.write(self.style.ERROR(f'Error: {error}'))
        else:
            self.stdout.write(self.style.SUCCESS('Transacciones indexadas correctamente')) 