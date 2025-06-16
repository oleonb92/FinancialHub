from rest_framework import serializers
from .models import Organization, OrganizationMembership
from accounts.serializers import UserSerializer

class OrganizationSerializer(serializers.ModelSerializer):
    members_count = serializers.SerializerMethodField()
    sponsor_details = UserSerializer(source='sponsor', read_only=True)

    class Meta:
        model = Organization
        fields = ['id', 'name', 'created_at', 'plan', 'sponsor', 'sponsor_details', 'members_count']
        read_only_fields = ['created_at', 'sponsor']

    def get_members_count(self, obj):
        return obj.memberships.count()

class OrganizationMembershipSerializer(serializers.ModelSerializer):
    user_details = UserSerializer(source='user', read_only=True)
    organization_details = OrganizationSerializer(source='organization', read_only=True)
    invited_by_details = UserSerializer(source='invited_by', read_only=True)

    class Meta:
        model = OrganizationMembership
        fields = [
            'id', 'user', 'user_details', 'organization', 'organization_details',
            'role', 'pro_features_for_accountant', 'joined_at', 'invited_by',
            'invited_by_details'
        ]
        read_only_fields = ['joined_at']

class OrganizationInviteSerializer(serializers.Serializer):
    email = serializers.EmailField()
    role = serializers.ChoiceField(choices=OrganizationMembership.ROLE_CHOICES)
    pro_features_for_accountant = serializers.BooleanField(default=False)

class OrganizationRoleUpdateSerializer(serializers.Serializer):
    role = serializers.ChoiceField(choices=OrganizationMembership.ROLE_CHOICES)
    pro_features_for_accountant = serializers.BooleanField(required=False) 