from rest_framework import serializers
from .models import Incentive

class IncentiveSerializer(serializers.ModelSerializer):
    accountant_username = serializers.CharField(source='accountant.username', read_only=True)
    client_username = serializers.CharField(source='client.username', read_only=True)
    organization_name = serializers.CharField(source='organization.name', read_only=True)

    class Meta:
        model = Incentive
        fields = [
            'id', 'accountant', 'accountant_username', 'client', 'client_username',
            'organization', 'organization_name', 'amount', 'description', 'created_at'
        ]
        read_only_fields = ['id', 'created_at', 'accountant', 'accountant_username', 'client_username', 'organization_name'] 