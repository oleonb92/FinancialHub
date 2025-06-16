import logging
from django.contrib.auth import get_user_model
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework_simplejwt.views import TokenObtainPairView
from .permissions import admin_required
from audit.models import AuditLog
from accounts.access_control import require_access, has_pro_access
from organizations.models import Organization
from accounts.constants import PRO_FEATURES_ACCOUNTANT, PRO_FEATURES_MEMBER
from django.utils import timezone
from rest_framework import viewsets, permissions
from rest_framework.decorators import action
from accounts.serializers import UserSerializer, UserApprovalSerializer


User = get_user_model()
logger = logging.getLogger(__name__)

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def profile_view(request):
    user = request.user
    logger.info("🔍 Profile requested by: %s", user)
    logger.debug("User.is_authenticated: %s", user.is_authenticated)
    logger.debug("User role: %s", getattr(user, "role", "undefined"))
    logger.debug("User household: %s", getattr(user.household, "name", None) if hasattr(user, "household") else "None")

    # Estado Pro y features activos
    now = timezone.now()
    pro_status = {
        "pro_global": user.pro_features,
        "pro_trial_active": bool(user.pro_trial_until and user.pro_trial_until > now),
        "pro_trial_until": user.pro_trial_until,
        "pro_features_list": user.pro_features_list,
        "account_type": user.account_type,
        "features_accountant": PRO_FEATURES_ACCOUNTANT,
        "features_member": PRO_FEATURES_MEMBER,
    }

    try:
        return Response({
            "id": user.id,
            "username": user.username,
            "role": getattr(user, "role", None),
            "household": user.household.name if hasattr(user, "household") and user.household else None,
            "pro_status": pro_status,
        })
    except Exception as e:
        logger.error("🔥 Error in profile_view: %s", str(e))
        return Response(
            {"error": "Error fetching profile."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    def validate(self, attrs):
        logger.info("🔑 Token login attempt for: %s", attrs.get("username"))
        try:
            data = super().validate(attrs)
            data['username'] = self.user.username
            data['role'] = getattr(self.user, 'role', None)
            data['household'] = self.user.household.name if hasattr(self.user, 'household') and self.user.household else None
            logger.info("✅ Token issued for user: %s", self.user.username)
            return data
        except Exception as e:
            logger.error("❌ Login failed for user %s: %s", attrs.get("username"), str(e))
            raise

class MyTokenObtainPairView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer


@api_view(["GET"])
@require_access(required_roles=["admin"], require_pro=True)
def list_users(request):
    if getattr(request.user, "role", "") != "admin":
        return Response(
            {"detail": "Not authorized."},
            status=status.HTTP_403_FORBIDDEN
        )

    users = User.objects.all()
    user_data = []
    for user in users:
        if user.role == "admin":
            status_display = "admin"
        elif user.is_active:
            status_display = "approved"
        else:
            status_display = "pending" if not user.was_approved else "paused"

        user_data.append({
            "id": user.id,
            "username": user.username,
            "is_active": user.is_active,
            "is_staff": user.is_staff,
            "date_joined": user.date_joined,
            "role": user.role,
            "household_name": getattr(user.household, "name", None) if hasattr(user, "household") else None,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "email": user.email,
            "birthdate": user.birthdate,
            "was_approved": user.was_approved,
            "status": status_display,
            "show_pause": True if user.role == "member" and user.was_approved and user.is_active else False,
        })

    return Response(user_data)

@api_view(["PATCH"])
@require_access(required_roles=["admin"], require_pro=True)
def approve_user(request, user_id):
    try:
        user = User.objects.get(id=user_id)
        user.is_active = True
        user.was_approved = True
        user.save()

        AuditLog.objects.create(
            action="approve",
            performed_by=request.user,
            target=user.username,
            details="User approved via admin panel"
        )

        return Response({"message": f"User '{user.username}' approved."})
    except User.DoesNotExist:
        return Response(
            {"error": "User not found."},
            status=status.HTTP_404_NOT_FOUND
        )

@api_view(["PATCH"])
@require_access(required_roles=["admin"], require_pro=True)
def deny_user(request, user_id):
    try:
        user = User.objects.get(id=user_id)
        user.is_active = False
        user.save()

        AuditLog.objects.create(
            action="reject",
            performed_by=request.user,
            target=user.username,
            details="User rejected via admin panel"
        )

        return Response({"message": f"User '{user.username}' denied."})
    except User.DoesNotExist:
        return Response(
            {"error": "User not found."},
            status=status.HTTP_404_NOT_FOUND
        )

@api_view(["PATCH"])
@require_access(required_roles=["admin"], require_pro=True)
def edit_user(request, user_id):
    try:
        user = User.objects.get(id=user_id)
        logger.info("🛠 PATCH user data for %s:", user.username)
        logger.info(" - username: %s", request.data.get("username"))
        logger.info(" - first_name: %s", request.data.get("first_name"))
        logger.info(" - last_name: %s", request.data.get("last_name"))
        logger.info(" - email: %s", request.data.get("email"))
        logger.info(" - role: %s", request.data.get("role"))
        logger.info(" - birthdate: %s", request.data.get("birthdate"))

        user.username = request.data.get("username", user.username)
        user.first_name = request.data.get("first_name", user.first_name)
        user.last_name = request.data.get("last_name", user.last_name)
        user.email = request.data.get("email", user.email)
        user.role = request.data.get("role", user.role)
        user.birthdate = request.data.get("birthdate", user.birthdate)
        user.save()

        AuditLog.objects.create(
            action="edit",
            performed_by=request.user,
            target=user.username,
            details="User profile edited by admin"
        )

        return Response({"message": "User updated."})
    except User.DoesNotExist:
        return Response(
            {"error": "User not found."},
            status=status.HTTP_404_NOT_FOUND
        )

@api_view(["PATCH"])
@require_access(required_roles=["admin"], require_pro=True)
def pause_user(request, user_id):
    try:
        user = User.objects.get(id=user_id)
        user.is_active = False
        user.save()

        AuditLog.objects.create(
            action="pause",
            performed_by=request.user,
            target=user.username,
            details=f"User {user.username} was paused by {request.user.username}"
        )

        return Response({"message": f"User '{user.username}' paused."})
    except User.DoesNotExist:
        return Response(
            {"error": "User not found."},
            status=status.HTTP_404_NOT_FOUND
        )

@api_view(["PATCH"])
@require_access(required_roles=["admin"], require_pro=True)
def delete_user(request, user_id):
    try:
        user = User.objects.get(id=user_id)
        user.delete()

        AuditLog.objects.create(
            action="delete",
            performed_by=request.user,
            target=user.username,
            details="User deleted from system"
        )

        return Response({"message": f"User '{user.username}' deleted successfully."})
    except User.DoesNotExist:
        return Response({"error": "User not found."}, status=404)
    
@api_view(["GET"])
@require_access(required_roles=["admin", "accountant"], require_pro=True, allow_accountant_always=True)
def get_user_detail(request, user_id):
    try:
        user = User.objects.get(id=user_id)
        data = {
            "id": user.id,
            "username": user.username,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "email": user.email,
            "birthdate": user.birthdate,
            "role": user.role,
            "household_name": user.household.name if user.household else None,
            "status": "approved" if user.is_active else ("paused" if user.was_approved else "pending"),
            "is_active": user.is_active,
            "was_approved": user.was_approved,
            "show_pause": user.was_approved and user.is_active and user.role == "member",
        }
        return Response(data)
    except User.DoesNotExist:
        return Response({"error": "User not found."}, status=status.HTTP_404_NOT_FOUND)


# Health check endpoint
@api_view(["GET"])
def health_check(request):
    """
    A simple endpoint to check if the API service is up.
    """
    return Response({"status": "healthy", "message": "Backend API is running"}, status=status.HTTP_200_OK)

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
        user = request.user
        data = {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'role': user.role,
            'organization_name': user.organization.name if user.organization else None,
        }
        return Response(data)

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