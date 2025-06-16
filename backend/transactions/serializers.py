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

class CategorySerializer(serializers.ModelSerializer):
    children = serializers.SerializerMethodField()
    parent_name = serializers.SerializerMethodField()

    class Meta:
        model = Category
        fields = ['id', 'name', 'description', 'parent', 'parent_name', 'children', 'organization', 'created_by', 'created_at', 'modified_at']
        read_only_fields = ['created_by', 'created_at', 'modified_at']

    def get_children(self, obj):
        # Solo serializar subcategorías si estamos en el nivel superior
        if obj.parent is None:
            children = Category.objects.filter(parent=obj, organization=obj.organization)
            return CategorySerializer(children, many=True).data
        return []

    def get_parent_name(self, obj):
        return obj.parent.name if obj.parent else None

    def validate(self, data):
        # Validar que una categoría no sea su propia subcategoría
        if 'parent' in data and data['parent']:
            if data['parent'].id == self.instance.id if self.instance else None:
                raise serializers.ValidationError("Una categoría no puede ser su propia subcategoría")
            
            # Validar que no se cree un ciclo en la jerarquía
            parent = data['parent']
            while parent:
                if parent.id == self.instance.id if self.instance else None:
                    raise serializers.ValidationError("No se puede crear un ciclo en la jerarquía de categorías")
                parent = parent.parent

        return data

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
        ]
        extra_kwargs = {
            'source_account': {'read_only': True},
            'destination_account': {'read_only': True},
        }
        
    def get_tags(self, obj):
        return [tag.name for tag in obj.tags.all()]
        
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