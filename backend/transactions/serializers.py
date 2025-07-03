# from rest_framework import serializers
# from .models import Transaction, Tag

# class TransactionSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = Transaction
#         fields = [
#             'id',
#             'type',
#             'amount',
#             'date',
#             'description',
#             'category',
#             'source_account',
#             'destination_account',
#             'is_imported',
#             'bank_transaction_id',
#             'status',
#             'created_at',
#             'modified_at'
#         ]
#         read_only_fields = ['created_by', 'created_at', 'modified_at']

from rest_framework import serializers
from .models import Tag, Transaction, Category, Budget
from chartofaccounts.serializers import AccountSerializer
from chartofaccounts.models import Account
from ai.services import AIService
import logging
from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger(__name__)

# Singleton para el servicio de IA
_ai_service_instance = None

def get_ai_service():
    """Obtiene una instancia singleton del servicio de IA."""
    global _ai_service_instance
    if _ai_service_instance is None:
        _ai_service_instance = AIService()
    return _ai_service_instance

class CategorySerializer(serializers.ModelSerializer):
    subcategories = serializers.SerializerMethodField()
    parent_name = serializers.CharField(source='parent.name', read_only=True)
    transaction_count = serializers.SerializerMethodField()

    class Meta:
        model = Category
        fields = ['id', 'name', 'icon', 'description', 'parent', 'parent_name', 
                 'organization', 'created_by', 'created_at', 'modified_at', 
                 'subcategories', 'transaction_count']
        read_only_fields = ['created_by', 'created_at', 'modified_at']

    def get_subcategories(self, obj):
        """Obtiene las subcategorías de esta categoría."""
        subcategories = Category.objects.filter(parent=obj)
        return CategorySerializer(subcategories, many=True, context=self.context).data
    
    def get_transaction_count(self, obj):
        """Obtiene el número de transacciones en esta categoría y sus subcategorías."""
        # Obtener IDs de esta categoría y todas sus subcategorías
        category_ids = self._get_all_category_ids(obj)
        return Transaction.objects.filter(category_id__in=category_ids).count()
    
    def _get_all_category_ids(self, category):
        """Obtiene recursivamente todos los IDs de categoría y subcategorías."""
        if not category:
            return []

        ids = [category.id]
        for subcat in category.children.all():
            ids += self._get_all_category_ids(subcat)
        return ids

class TagSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tag
        fields = ['id', 'name']

class TransactionSerializer(serializers.ModelSerializer):
    tags = serializers.SerializerMethodField()
    tag_names = serializers.ListField(child=serializers.CharField(), write_only=True, required=False)
    source_account = AccountSerializer(read_only=True)
    destination_account = AccountSerializer(read_only=True)
    source_account_id = serializers.PrimaryKeyRelatedField(queryset=Account.objects.all(), source='source_account', write_only=True, required=False)
    destination_account_id = serializers.PrimaryKeyRelatedField(queryset=Account.objects.all(), source='destination_account', write_only=True, required=False)
    category = CategorySerializer(read_only=True)
    category_id = serializers.PrimaryKeyRelatedField(queryset=Category.objects.all(), source='category', write_only=True, required=False, allow_null=True)
    classification = serializers.SerializerMethodField()
    
    class Meta:
        model = Transaction
        fields = [
            'id',
            'description',
            'amount',
            'date',
            'category',
            'category_id',
            'status',
            'type',
            'tags',
            'tag_names',
            'source_account',
            'destination_account',
            'source_account_id',
            'destination_account_id',
            'created_at',
            'modified_at',
            'is_imported',
            'bank_transaction_id',
            'merchant',
            'classification',
        ]
        extra_kwargs = {
            'source_account': {'read_only': True},
            'destination_account': {'read_only': True},
        }
        
    def get_tags(self, obj):
        return [tag.name for tag in obj.tags.all()]
        
    def get_classification(self, obj):
        # Optimización: Solo devolver estado pendiente para evitar análisis síncrono
        return {'status': 'pending'}
        
    def create(self, validated_data):
        tag_names = validated_data.pop('tag_names', [])
        transaction = Transaction.objects.create(**validated_data)
        for tag_name in tag_names:
            tag, _ = Tag.objects.get_or_create(name=tag_name)
            transaction.tags.add(tag)
        return transaction
    
    def update(self, instance, validated_data):
        tag_names = validated_data.pop('tag_names', None)
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        if tag_names is not None:
            instance.tags.clear()
            for tag_name in tag_names:
                tag, _ = Tag.objects.get_or_create(name=tag_name)
                instance.tags.add(tag)
        instance.save()
        return instance

    def to_representation(self, instance):
        """Personaliza la representación de la transacción optimizada para rendimiento."""
        data = super().to_representation(instance)
        
        # Agregar información de análisis de IA desde caché o campos del modelo
        if instance.ai_analyzed:
            data['ai_analysis'] = {
                'analyzed': True,
                'confidence': instance.ai_confidence,
                'suggested_category_id': instance.ai_category_suggestion.id if instance.ai_category_suggestion else None,
                'notes': instance.ai_notes,
                'quality_status': 'high' if instance.ai_confidence and instance.ai_confidence >= 0.85 else 'fallback'
            }
        else:
            data['ai_analysis'] = {
                'analyzed': False,
                'confidence': None,
                'suggested_category_id': None,
                'notes': None,
                'quality_status': 'not_analyzed'
            }
        
        # Obtener sugerencias desde caché en lugar de análisis síncrono
        cache_key = f"transaction_suggestion_{instance.id}"
        cached_suggestion = cache.get(cache_key)
        
        if cached_suggestion:
            data['category_suggestion'] = cached_suggestion
        else:
            # Si no hay caché, devolver estado básico
            data['category_suggestion'] = {
                'status': 'pending',
                'message': 'Análisis en progreso...',
                'needs_update': False,
                'current_category_id': instance.category.id if instance.category else None,
                'suggested_category_id': None,
                'already_approved': False
            }
        
        return data

class BudgetSerializer(serializers.ModelSerializer):
    category_name = serializers.CharField(source='category.name', read_only=True)
    spent_amount = serializers.DecimalField(max_digits=12, decimal_places=2, read_only=True)
    remaining_amount = serializers.DecimalField(max_digits=12, decimal_places=2, read_only=True)
    percentage_used = serializers.FloatField(read_only=True)

    class Meta:
        model = Budget
        fields = [
            'id', 'category', 'category_name', 'organization', 'amount',
            'period', 'spent_amount', 'remaining_amount', 'percentage_used',
            'created_by', 'created_at', 'modified_at'
        ]
        read_only_fields = ['created_by', 'created_at', 'modified_at']