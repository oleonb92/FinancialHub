from django.core.management.base import BaseCommand
from elasticsearch_dsl import connections
from transactions.services import TransactionSearchService
from datetime import datetime, timedelta

class Command(BaseCommand):
    help = 'Prueba todas las funcionalidades de búsqueda de transacciones'

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

        # 1. Búsqueda básica con fuzzy matching
        self.stdout.write("\n=== Búsqueda básica con fuzzy matching ===")
        service = TransactionSearchService()
        response = service.search_by_description('UHC').execute()
        self.print_results(response)

        # 2. Búsqueda con filtros de monto
        self.stdout.write("\n=== Búsqueda con filtros de monto ===")
        response = (service
            .filter_by_amount_range(min_amount=1000, max_amount=20000)
            .execute())
        self.print_results(response)

        # 3. Búsqueda con filtros de fecha
        self.stdout.write("\n=== Búsqueda con filtros de fecha ===")
        response = (service
            .filter_by_date_range(days=30)
            .execute())
        self.print_results(response)

        # 4. Búsqueda combinada
        self.stdout.write("\n=== Búsqueda combinada ===")
        response = (service
            .search_by_description('UHC')
            .filter_by_amount_range(min_amount=1000)
            .filter_by_date_range(days=30)
            .execute())
        self.print_results(response)

        # 5. Búsqueda con ordenamiento
        self.stdout.write("\n=== Búsqueda con ordenamiento ===")
        response = (service
            .sort_by('amount', 'desc')
            .paginate(page=1, size=5)
            .execute())
        self.print_results(response)

        # 6. Búsqueda con agregaciones
        self.stdout.write("\n=== Búsqueda con agregaciones ===")
        response = (service
            .add_aggregations()
            .execute())
        
        # Mostrar agregaciones
        self.stdout.write("\nAgregación por tipo:")
        for bucket in response.aggregations.por_tipo.buckets:
            self.stdout.write(f"Tipo: {bucket.key}, Cantidad: {bucket.doc_count}")
        
        self.stdout.write("\nEstadísticas de monto:")
        stats = response.aggregations.estadisticas_monto
        self.stdout.write(f"Min: {stats.min}, Max: {stats.max}, Avg: {stats.avg}")
        
        self.stdout.write("\nAgregación por mes:")
        for bucket in response.aggregations.por_mes.buckets:
            self.stdout.write(f"Mes: {bucket.key}, Cantidad: {bucket.doc_count}")
        
        self.stdout.write("\nAgregación por tags:")
        for bucket in response.aggregations.por_tags.buckets:
            self.stdout.write(f"Tag: {bucket.key}, Cantidad: {bucket.doc_count}")

        # 7. Prueba de sugerencias
        self.stdout.write("\n=== Prueba de sugerencias ===")
        suggestions = service.get_suggestions('UH')
        self.stdout.write("Sugerencias para 'UH':")
        for option in suggestions.suggest.suggestions[0].options:
            self.stdout.write(f"- {option.text}")

    def print_results(self, response):
        """Imprimir resultados de búsqueda"""
        self.stdout.write(f"Total de resultados: {response.hits.total.value}")
        for hit in response:
            self.stdout.write(f"ID: {hit.meta.id}")
            self.stdout.write(f"Descripción: {hit.description}")
            self.stdout.write(f"Monto: {hit.amount}")
            self.stdout.write(f"Fecha: {hit.date}")
            self.stdout.write(f"Tipo: {hit.type}")
            if hasattr(hit, 'tags') and hit.tags:
                self.stdout.write(f"Tags: {', '.join(hit.tags)}")
            self.stdout.write("---") 