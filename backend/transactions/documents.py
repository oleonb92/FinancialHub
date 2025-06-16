from django_elasticsearch_dsl import Document, Index, fields
from django_elasticsearch_dsl.registries import registry
from .models import Transaction
from decimal import Decimal

@registry.register_document
class TransactionDocument(Document):
    # Campos para búsqueda y agregaciones
    type = fields.KeywordField()  # Para agregaciones y filtros exactos
    description = fields.TextField(
        fields={
            'raw': fields.KeywordField(),  # Para agregaciones y ordenamiento
            'suggest': fields.CompletionField(),  # Para autocompletado
        }
    )
    amount = fields.FloatField()  # Para rangos y agregaciones numéricas
    date = fields.DateField()  # Para rangos de fechas
    source_account_id = fields.KeywordField()  # Para filtros por cuenta origen
    destination_account_id = fields.KeywordField()  # Para filtros por cuenta destino
    tags = fields.KeywordField(multi=True)  # Para filtros por tags
    
    class Index:
        name = 'transactions'
        settings = {
            'number_of_shards': 1,
            'number_of_replicas': 0,
            'analysis': {
                'analyzer': {
                    'custom_analyzer': {
                        'type': 'custom',
                        'tokenizer': 'standard',
                        'filter': ['lowercase', 'asciifolding']
                    }
                }
            }
        }
    
    class Django:
        model = Transaction
        ignore_signals = False
        auto_refresh = True

    def prepare_source_account_id(self, instance):
        return str(instance.source_account.id) if instance.source_account else None

    def prepare_destination_account_id(self, instance):
        return str(instance.destination_account.id) if instance.destination_account else None

    def prepare_amount(self, instance):
        return float(instance.amount) if instance.amount else 0.0

    def prepare_tags(self, instance):
        return [tag.name for tag in instance.tags.all()] if instance.tags.exists() else [] 