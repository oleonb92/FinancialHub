from rest_framework import generics, permissions, status, viewsets
from .serializers import RegisterSerializer, UserSerializer, UserApprovalSerializer
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes, action
from .models import PendingInvitation
from django.contrib.auth import get_user_model
import logging
from django.db import transaction
from accounts.access_control import require_access
from organizations.models import Organization

logger = logging.getLogger(__name__)
User = get_user_model()

class RegisterView(generics.CreateAPIView):
    serializer_class = RegisterSerializer
    permission_classes = [permissions.AllowAny]

    def create(self, request, *args, **kwargs):
        logger.info("Received registration data: %s", request.data)
        serializer = self.get_serializer(data=request.data)
        
        if not serializer.is_valid():
            logger.error("Validation errors: %s", serializer.errors)
            return Response(
                {
                    'error': 'Validation failed',
                    'details': serializer.errors
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            with transaction.atomic():
                user = serializer.save()
                logger.info("User created successfully: %s", user.username)
                
                # Serializar la respuesta usando UserSerializer
                response_serializer = UserSerializer(user)
                return Response(
                    {
                        'message': 'User registered successfully',
                        'user': response_serializer.data
                    },
                    status=status.HTTP_201_CREATED
                )
        except Exception as e:
            logger.error("Error creating user: %s", str(e))
            return Response(
                {
                    'error': 'Error creating user',
                    'details': str(e)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

@require_access(required_roles=['admin', 'accountant'], require_pro=True, allow_accountant_always=True)
def list_pending_invitations(request):
    user = request.user
    if user.role != 'admin':
        return Response(
            {"detail": "Only admins can view pending invitations."},
            status=status.HTTP_403_FORBIDDEN
        )

    invitations = PendingInvitation.objects.filter(
        household=user.household,
        is_approved=False
    )
    data = [
        {
            "username": inv.username,
            "requested_at": inv.requested_at,
            "role": inv.role
        }
        for inv in invitations
    ]
    return Response(data)

class UserViewSet(viewsets.ModelViewSet):
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return User.objects.filter(organization=self.request.user.organization)

    def perform_create(self, serializer):
        serializer.save(organization=self.request.user.organization)

    def perform_update(self, serializer):
        if serializer.instance.organization != self.request.user.organization:
            return Response(status=status.HTTP_403_FORBIDDEN)
        serializer.save()

    @action(detail=False, methods=['get'])
    def me(self, request):
        serializer = self.get_serializer(request.user)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def pending_approvals(self, request):
        if not request.user.role == 'admin':
            return Response(status=status.HTTP_403_FORBIDDEN)
        
        pending_users = User.objects.filter(
            organization=request.user.organization,
            was_approved=False
        )
        serializer = UserApprovalSerializer(pending_users, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def approve(self, request, pk=None):
        if not request.user.role == 'admin':
            return Response(status=status.HTTP_403_FORBIDDEN)
        
        user = self.get_object()
        if user.organization != request.user.organization:
            return Response(status=status.HTTP_403_FORBIDDEN)
        
        user.was_approved = True
        user.is_active = True
        user.save()
        
        serializer = UserApprovalSerializer(user)
        return Response(serializer.data)

# class UserUpdateView(generics.UpdateAPIView):
#     queryset = User.objects.all()
#     serializer_class = UserSerializer
#     permission_classes = [permissions.IsAdminUser]
#     lookup_field = 'id'