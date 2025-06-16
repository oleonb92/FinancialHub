from datetime import datetime, timedelta
from elasticsearch_dsl import Q, Search
from elasticsearch_dsl.query import MultiMatch, Range, Terms, Match
from elasticsearch_dsl.aggs import Terms as TermsAgg, Stats, DateHistogram
from .documents import TransactionDocument

class TransactionSearchService:
    def __init__(self):
        self.search = Search(index='transactions')

    def search_by_description(self, query, fuzzy=True):
        """Búsqueda por descripción con opción de fuzzy matching"""
        if fuzzy:
            self.search = self.search.query(
                MultiMatch(
                    query=query,
                    fields=['description^3', 'description.raw'],
                    fuzziness='AUTO'
                )
            )
        else:
            self.search = self.search.query(
                Match(description=query)
            )
        return self

    def filter_by_amount_range(self, min_amount=None, max_amount=None):
        """Filtrar por rango de monto"""
        amount_range = {}
        if min_amount is not None:
            amount_range['gte'] = float(min_amount)
        if max_amount is not None:
            amount_range['lte'] = float(max_amount)
        if amount_range:
            self.search = self.search.filter('range', amount=amount_range)
        return self

    def filter_by_date_range(self, start_date=None, end_date=None, days=None):
        """Filtrar por rango de fechas"""
        if days is not None:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
        
        date_range = {}
        if start_date:
            date_range['gte'] = start_date
        if end_date:
            date_range['lte'] = end_date
        if date_range:
            self.search = self.search.filter('range', date=date_range)
        return self

    def filter_by_types(self, types):
        """Filtrar por tipos de transacción"""
        if types:
            self.search = self.search.filter('terms', type=types)
        return self

    def filter_by_accounts(self, account_ids):
        """Filtrar por IDs de cuentas"""
        if account_ids:
            self.search = self.search.filter('terms', account_id=[str(id) for id in account_ids])
        return self

    def filter_by_tags(self, tags):
        """Filtrar por tags"""
        if tags:
            self.search = self.search.filter('terms', tags=tags)
        return self

    def sort_by(self, field, order='asc'):
        """Ordenar resultados"""
        self.search = self.search.sort({field: {'order': order}})
        return self

    def paginate(self, page=1, size=10):
        """Paginación de resultados"""
        self.search = self.search[(page - 1) * size:page * size]
        return self

    def add_aggregations(self):
        """Agregar agregaciones útiles"""
        # Agregación por tipo
        self.search.aggs.bucket(
            'por_tipo',
            TermsAgg(field='type')
        )
        
        # Estadísticas de monto
        self.search.aggs.metric(
            'estadisticas_monto',
            Stats(field='amount')
        )
        
        # Agregación por mes
        self.search.aggs.bucket(
            'por_mes',
            DateHistogram(
                field='date',
                calendar_interval='month',
                format='yyyy-MM'
            )
        )
        
        # Agregación por tags
        self.search.aggs.bucket(
            'por_tags',
            TermsAgg(field='tags')
        )
        
        return self

    def get_suggestions(self, prefix):
        """Obtener sugerencias para autocompletado"""
        return self.search.suggest(
            'suggestions',
            prefix,
            completion={
                'field': 'description.suggest',
                'fuzzy': {
                    'fuzziness': 2
                }
            }
        ).execute()

    def execute(self):
        """Ejecutar la búsqueda"""
        return self.search.execute()

    def get_total(self):
        """Obtener el total de resultados sin paginación"""
        return self.search.count() 