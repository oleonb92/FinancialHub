import logging
from functools import wraps
from rest_framework.response import Response
from rest_framework import status
from organizations.models import OrganizationMembership
from django.utils.decorators import method_decorator
from accounts.constants import PRO_FEATURES_ACCOUNTANT, PRO_FEATURES_MEMBER
from django.utils import timezone
from django.http import JsonResponse

logger = logging.getLogger(__name__)

class AccessControlError(Exception):
    """Excepción personalizada para errores de control de acceso"""
    pass

def get_user_role_in_org(user, org):
    try:
        membership = OrganizationMembership.objects.get(user=user, organization=org)
        return membership.role
    except OrganizationMembership.DoesNotExist:
        return None

def require_access(required_roles=None, require_pro=False, allow_accountant_always=False, sponsor_only=False):
    """
    Decorador para controlar acceso basado en rol, plan, sponsor y reglas especiales.
    Soporta views DRF y CBV (usando method_decorator).
    """
    required_roles = required_roles or []

    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(self_or_request, *args, **kwargs):
            # Soporta tanto CBV (ViewSet) como FBV
            if hasattr(self_or_request, 'request'):
                request = self_or_request.request
            else:
                request = self_or_request
            user = request.user
            org = getattr(request, 'organization', None)
            if not org:
                logger.warning(f"Acceso denegado: organización no especificada para usuario {user}")
                # Check if user has multiple organizations
                memberships = OrganizationMembership.objects.filter(user=user)
                if memberships.count() > 1:
                    return JsonResponse({
                        'detail': 'Se requiere especificar una organización.',
                        'code': 'organization_required',
                        'organizations': [{'id': m.organization.id, 'name': m.organization.name} for m in memberships]
                    }, status=400)
                else:
                    raise AccessControlError("Organization not specified.")

            # Check sponsor access
            if sponsor_only and not getattr(user, 'is_sponsor', False):
                logger.warning(f"Access denied: Sponsor only access required for user {user.id}")
                raise AccessControlError("Only sponsors can perform this action.")

            # Check role access
            user_role = get_user_role_in_org(user, org)
            if required_roles and user_role not in required_roles:
                logger.warning(f"Access denied: Insufficient role {user_role} for user {user.id}")
                raise AccessControlError("Insufficient role permissions.")

            # Check Pro access
            if require_pro and not has_pro_access(user, org):
                logger.warning(f"Access denied: Pro access required for user {user.id}")
                raise AccessControlError("Pro access required.")

            return view_func(self_or_request, *args, **kwargs)
        return _wrapped_view
    return decorator

def has_pro_access(user, organization=None, feature=None):
    now = timezone.now()
    # Acceso Pro global o trial
    if getattr(user, 'pro_features', False) or (getattr(user, 'pro_trial_until', None) and user.pro_trial_until and user.pro_trial_until > now):
        if feature:
            if user.account_type == "accountant" and feature in PRO_FEATURES_ACCOUNTANT:
                return True
            if user.account_type == "personal" and feature in PRO_FEATURES_MEMBER:
                return True
        else:
            return True
    # Acceso por feature específica en la lista
    if feature and feature in getattr(user, 'pro_features_list', []):
        return True
    # Acceso Pro por organización
    if organization:
        if getattr(organization, 'plan', None) == 'pro':
            if feature:
                if user.account_type == "accountant" and feature in PRO_FEATURES_ACCOUNTANT:
                    return True
                if user.account_type == "personal" and feature in PRO_FEATURES_MEMBER:
                    return True
            else:
                return True
        # Membership con pro_features_for_accountant
        if hasattr(organization, 'memberships') and organization.memberships.filter(user=user, pro_features_for_accountant=True).exists():
            return True
    return False

# Para CBV, puedes usar method_decorator(require_access(...)) en dispatch o métodos específicos. 