from rest_framework import viewsets, permissions
from rest_framework.exceptions import PermissionDenied
from .models import Account
from .serializers import AccountSerializer
from accounts.access_control import require_access

class AccountViewSet(viewsets.ModelViewSet):
    serializer_class = AccountSerializer
    permission_classes = [permissions.IsAuthenticated]

    @require_access(required_roles=["admin", "accountant"], allow_accountant_always=True)
    def get_queryset(self):
        if getattr(self, 'swagger_fake_view', False):
            return Account.objects.none()
        if not self.request.user.is_authenticated:
            return Account.objects.none()
        if not hasattr(self.request, 'organization'):
            raise PermissionDenied("No se ha especificado una organización. Por favor, incluye el header 'X-Organization-ID' en tu solicitud.")
        return Account.objects.filter(organization=self.request.organization)

    @require_access(required_roles=["admin", "accountant"], allow_accountant_always=True)
    def perform_create(self, serializer):
        if not hasattr(self.request, 'organization'):
            raise PermissionDenied("No se ha especificado una organización. Por favor, incluye el header 'X-Organization-ID' en tu solicitud.")
        serializer.save(organization=self.request.organization)