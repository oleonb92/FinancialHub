from rest_framework import serializers
from .models import Account

class AccountSerializer(serializers.ModelSerializer):
    parent = serializers.PrimaryKeyRelatedField(
        queryset=Account.objects.all(),
        required=False,
        allow_null=True
    )

    class Meta:
        model = Account
        fields = [
            'id',
            'name',
            'code',
            'type',
            'functional_type',
            'currency',
            'is_default',
            'parent',
            'organization',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['created_at', 'updated_at']