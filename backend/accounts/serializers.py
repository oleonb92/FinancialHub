from rest_framework import serializers
from .models import User, PendingInvitation
from django.contrib.auth import get_user_model
from organizations.models import Organization
from django.db import transaction

User = get_user_model()

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True)
    organization = serializers.CharField(write_only=True, required=False)
    role = serializers.ChoiceField(choices=[
        ('owner', 'Owner'),
        ('admin', 'Admin'),
        ('member', 'Member'),
        ('accountant', 'Accountant'),
        ('bookkeeper', 'Bookkeeper'),
        ('advisor', 'Financial Advisor'),
    ], default='member')
    birthdate = serializers.DateField(required=False, allow_null=True)
    organization_name = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = User
        fields = [
            'username', 'password', 'email', 'first_name', 'last_name',
            'birthdate', 'role', 'organization', 'organization_name'
        ]
        extra_kwargs = {
            'password': {'write_only': True},
            'email': {'required': True},
            'first_name': {'required': True},
            'last_name': {'required': True},
            'birthdate': {'required': False},
            'role': {'required': False},
        }

    def validate(self, data):
        role = data.get('role', 'member')
        organization_name = data.get('organization')

        # Validar organization según el rol
        if role == 'admin':
            if organization_name and Organization.objects.filter(name=organization_name).exists():
                raise serializers.ValidationError({
                    'organization': 'An organization with this name already exists.'
                })
        elif role == 'member':
            if organization_name and not Organization.objects.filter(name=organization_name).exists():
                raise serializers.ValidationError({
                    'organization': 'Organization does not exist.'
                })

        return data

    def get_organization_name(self, obj):
        return obj.organization.name if obj.organization else None

    @transaction.atomic
    def create(self, validated_data):
        organization_name = validated_data.pop('organization', None)
        role = validated_data.pop('role', 'member')
        birthdate = validated_data.pop('birthdate', None)

        # Si es admin, debe ser aprobado automáticamente
        was_approved = True if role == 'admin' else False
        
        # Crear el usuario con campos estándar
        user = User.objects.create_user(
            username=validated_data['username'],
            email=validated_data['email'],
            password=validated_data['password'],
            first_name=validated_data['first_name'],
            last_name=validated_data['last_name'],
        )

        # Asignar campos personalizados
        user.role = role
        user.was_approved = was_approved
        if birthdate:
            user.birthdate = birthdate

        # Manejar organization según el rol
        if organization_name:
            if role == 'admin':
                organization = Organization.objects.create(name=organization_name)
            else:
                organization = Organization.objects.get(name=organization_name)
            user.organization = organization

        # Guardar el usuario
        user.save()

        return user


# Serializer for updating user information from the frontend (e.g., EditUserModal.jsx)
class UserSerializer(serializers.ModelSerializer):
    organization_name = serializers.SerializerMethodField(read_only=True)
    password = serializers.CharField(write_only=True, required=False)
    organization = serializers.CharField(write_only=True, required=False)

    class Meta:
        model = User
        fields = (
            'id', 'email', 'username', 'first_name', 'last_name',
            'birthdate', 'role', 'organization', 'organization_name'
        )
        read_only_fields = ('id',)

    def get_organization_name(self, obj):
        return obj.organization.name if obj.organization else None

    def validate(self, data):
        organization_name = data.get('organization')

        # Validar organization según el rol
        if data.get('role') == 'admin':
            if organization_name and Organization.objects.filter(name=organization_name).exists():
                raise serializers.ValidationError({
                    'organization': 'An organization with this name already exists.'
                })
        else:
            if organization_name and not Organization.objects.filter(name=organization_name).exists():
                raise serializers.ValidationError({
                    'organization': 'Organization does not exist.'
                })

        return data

    def create(self, validated_data):
        password = validated_data.pop('password', None)
        organization_name = validated_data.pop('organization', None)
        user = User.objects.create(**validated_data)

        if password:
            user.set_password(password)

        # Manejar organization según el rol
        if organization_name:
            if user.role == 'admin':
                organization = Organization.objects.create(name=organization_name)
            else:
                organization = Organization.objects.get(name=organization_name)
            user.organization = organization

        user.save()
        return user

class UserApprovalSerializer(serializers.ModelSerializer):
    organization_name = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = (
            'id', 'email', 'username', 'first_name', 'last_name',
            'is_active', 'was_approved', 'organization_name'
        )

    def get_organization_name(self, obj):
        return obj.organization.name if obj.organization else None