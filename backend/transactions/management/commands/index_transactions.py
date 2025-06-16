from django.core.management.base import BaseCommand
from elasticsearch_dsl import connections
from transactions.documents import TransactionDocument
from transactions.models import Transaction

class Command(BaseCommand):
    help = 'Indexa todas las transacciones en Elasticsearch'

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

        # Asegurarse de que el índice existe
        TransactionDocument.init()
        
        # Obtener todas las transacciones
        transactions = Transaction.objects.all()
        total = transactions.count()
        
        self.stdout.write(f'Indexando {total} transacciones...')
        
        # Indexar las transacciones
        doc = TransactionDocument()
        success, errors = doc.update(transactions)
        
        if errors:
            self.stdout.write(self.style.WARNING(f'Se encontraron {len(errors)} errores durante la indexación'))
            for error in errors:
                self.stdout.write(self.style.ERROR(f'Error: {error}'))
        else:
            self.stdout.write(self.style.SUCCESS(f'Se indexaron {total} transacciones correctamente')) 