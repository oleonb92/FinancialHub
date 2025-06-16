from django.shortcuts import render
from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from django.db import transaction
from .models import Organization, OrganizationMembership
from .serializers import (
    OrganizationSerializer,
    OrganizationMembershipSerializer,
    OrganizationInviteSerializer,
    OrganizationRoleUpdateSerializer
)
from accounts.models import User
from accounts.access_control import require_access, has_pro_access
from django.utils import timezone
from datetime import timedelta

class OrganizationViewSet(viewsets.ModelViewSet):
    serializer_class = OrganizationSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        return Organization.objects.filter(memberships__user=user).distinct()

    def perform_create(self, serializer):
        with transaction.atomic():
            organization = serializer.save()
            # Crear membresía para el creador como owner
            OrganizationMembership.objects.create(
                user=self.request.user,
                organization=organization,
                role='owner'
            )

    @require_access(required_roles=['admin', 'accountant'], require_pro=True, allow_accountant_always=True)
    @action(detail=True, methods=['get'])
    def members(self, request, pk=None):
        organization = self.get_object()
        memberships = organization.memberships.all()
        serializer = OrganizationMembershipSerializer(memberships, many=True)
        return Response(serializer.data)

    @require_access(required_roles=['admin'], require_pro=True)
    @action(detail=True, methods=['post'])
    def invite(self, request, pk=None):
        organization = self.get_object()
        serializer = OrganizationInviteSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        # Verificar que el usuario que invita tiene permisos
        membership = get_object_or_404(
            OrganizationMembership,
            organization=organization,
            user=request.user
        )
        if membership.role not in ['owner', 'admin']:
            return Response(
                {'detail': 'No tienes permisos para invitar miembros'},
                status=status.HTTP_403_FORBIDDEN
            )

        # Buscar usuario por email
        try:
            user = User.objects.get(email=serializer.validated_data['email'])
        except User.DoesNotExist:
            return Response(
                {'detail': 'Usuario no encontrado'},
                status=status.HTTP_404_NOT_FOUND
            )

        # Crear membresía
        membership, created = OrganizationMembership.objects.get_or_create(
            user=user,
            organization=organization,
            defaults={
                'role': serializer.validated_data['role'],
                'pro_features_for_accountant': serializer.validated_data['pro_features_for_accountant'],
                'invited_by': request.user
            }
        )

        # Trial automático para contadores invitados a organización Pro
        if created and serializer.validated_data['role'] == 'accountant':
            if organization.plan == 'pro' and not user.pro_features and (not user.pro_trial_until or user.pro_trial_until < timezone.now()):
                user.pro_trial_until = timezone.now() + timedelta(days=30)
                user.save()

        if not created:
            return Response(
                {'detail': 'El usuario ya es miembro de esta organización'},
                status=status.HTTP_400_BAD_REQUEST
            )

        return Response(
            OrganizationMembershipSerializer(membership).data,
            status=status.HTTP_201_CREATED
        )

    @require_access(required_roles=['admin'], require_pro=True)
    @action(detail=True, methods=['post'])
    def update_member_role(self, request, pk=None):
        organization = self.get_object()
        serializer = OrganizationRoleUpdateSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        # Verificar que el usuario que actualiza tiene permisos
        membership = get_object_or_404(
            OrganizationMembership,
            organization=organization,
            user=request.user
        )
        if membership.role not in ['owner', 'admin']:
            return Response(
                {'detail': 'No tienes permisos para actualizar roles'},
                status=status.HTTP_403_FORBIDDEN
            )

        # Obtener el miembro a actualizar
        member_id = request.data.get('member_id')
        if not member_id:
            return Response(
                {'detail': 'Se requiere member_id'},
                status=status.HTTP_400_BAD_REQUEST
            )

        member_membership = get_object_or_404(
            OrganizationMembership,
            organization=organization,
            id=member_id
        )

        # Actualizar rol
        member_membership.role = serializer.validated_data['role']
        if 'pro_features_for_accountant' in serializer.validated_data:
            member_membership.pro_features_for_accountant = serializer.validated_data['pro_features_for_accountant']
        member_membership.save()

        return Response(OrganizationMembershipSerializer(member_membership).data)

    @require_access(required_roles=['admin'], require_pro=True)
    @action(detail=True, methods=['post'])
    def remove_member(self, request, pk=None):
        organization = self.get_object()
        
        # Verificar que el usuario que elimina tiene permisos
        membership = get_object_or_404(
            OrganizationMembership,
            organization=organization,
            user=request.user
        )
        if membership.role not in ['owner', 'admin']:
            return Response(
                {'detail': 'No tienes permisos para eliminar miembros'},
                status=status.HTTP_403_FORBIDDEN
            )

        # Obtener el miembro a eliminar
        member_id = request.data.get('member_id')
        if not member_id:
            return Response(
                {'detail': 'Se requiere member_id'},
                status=status.HTTP_400_BAD_REQUEST
            )

        member_membership = get_object_or_404(
            OrganizationMembership,
            organization=organization,
            id=member_id
        )

        # No permitir eliminar al último owner
        if member_membership.role == 'owner':
            owner_count = OrganizationMembership.objects.filter(
                organization=organization,
                role='owner'
            ).count()
            if owner_count <= 1:
                return Response(
                    {'detail': 'No se puede eliminar al último owner'},
                    status=status.HTTP_400_BAD_REQUEST
                )

        member_membership.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
