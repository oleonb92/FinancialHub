import logging
from functools import wraps
from django.core.cache import cache
from django.utils import timezone
from organizations.models import Organization, OrganizationMembership
from accounts.constants import PRO_FEATURES_ACCOUNTANT, PRO_FEATURES_MEMBER
from core.exceptions import AccessControlError
from django.http import JsonResponse

logger = logging.getLogger('access_control')

class AccessControlService:
    """Service for handling access control and permissions"""
    
    CACHE_TIMEOUT = 300  # 5 minutes
    CACHE_KEY_PREFIX = 'access_control:'
    
    @classmethod
    def get_cache_key(cls, user_id, org_id, feature=None):
        """Generate cache key for access control checks"""
        key = f"{cls.CACHE_KEY_PREFIX}{user_id}:{org_id}"
        if feature:
            key += f":{feature}"
        return key

    @classmethod
    def clear_user_cache(cls, user_id):
        """Clear all access control cache entries for a user"""
        pattern = f"{cls.CACHE_KEY_PREFIX}{user_id}:*"
        keys = cache.keys(pattern)
        if keys:
            cache.delete_many(keys)

    @classmethod
    def clear_org_cache(cls, org_id):
        """Clear all access control cache entries for an organization"""
        pattern = f"{cls.CACHE_KEY_PREFIX}*:{org_id}:*"
        keys = cache.keys(pattern)
        if keys:
            cache.delete_many(keys)

    @classmethod
    def get_user_role_in_org(cls, user, org):
        """Get user's role in an organization with caching"""
        cache_key = f"{cls.CACHE_KEY_PREFIX}role:{user.id}:{org.id}"
        role = cache.get(cache_key)
        
        if role is None:
            try:
                membership = OrganizationMembership.objects.get(user=user, organization=org)
                role = membership.role
                cache.set(cache_key, role, cls.CACHE_TIMEOUT)
            except OrganizationMembership.DoesNotExist:
                role = None
                cache.set(cache_key, role, cls.CACHE_TIMEOUT)
        
        return role

    @classmethod
    def has_pro_access(cls, user, organization=None, feature=None):
        """
        Check if user has Pro access with clear hierarchy:
        1. Global Pro access
        2. Trial access
        3. Organization Pro access
        4. Feature-specific access
        """
        cache_key = cls.get_cache_key(user.id, organization.id if organization else 'global', feature)
        cached_result = cache.get(cache_key)
        
        if cached_result is not None:
            return cached_result

        now = timezone.now()
        result = False

        # 1. Check global Pro access
        if getattr(user, 'pro_features', False):
            result = True
            logger.info(f"Global Pro access granted for user {user.id}")
        
        # 2. Check trial access
        elif (getattr(user, 'pro_trial_until', None) and 
              user.pro_trial_until and 
              user.pro_trial_until > now):
            result = True
            logger.info(f"Trial access granted for user {user.id}")
        
        # 3. Check organization Pro access
        elif organization:
            if getattr(organization, 'plan', None) == 'pro':
                if feature:
                    if (user.account_type == "accountant" and feature in PRO_FEATURES_ACCOUNTANT) or \
                       (user.account_type == "personal" and feature in PRO_FEATURES_MEMBER):
                        result = True
                        logger.info(f"Organization Pro access granted for user {user.id} in org {organization.id}")
                else:
                    result = True
                    logger.info(f"Organization Pro access granted for user {user.id} in org {organization.id}")
            
            # Check accountant-specific Pro features
            elif (hasattr(organization, 'memberships') and 
                  organization.memberships.filter(user=user, pro_features_for_accountant=True).exists()):
                result = True
                logger.info(f"Accountant Pro features granted for user {user.id} in org {organization.id}")
        
        # 4. Check feature-specific access
        elif feature and feature in getattr(user, 'pro_features_list', []):
            result = True
            logger.info(f"Feature-specific access granted for user {user.id}")

        cache.set(cache_key, result, cls.CACHE_TIMEOUT)
        return result

    @classmethod
    def require_access(cls, required_roles=None, require_pro=False, allow_accountant_always=False, sponsor_only=False):
        """
        Decorator for access control with caching
        """
        required_roles = required_roles or []

        def decorator(view_func):
            @wraps(view_func)
            def _wrapped_view(request, *args, **kwargs):
                user = request.user
                org = getattr(request, 'organization', None)
                
                if not org:
                    logger.warning(f"Access denied: No organization specified for user {user.id}")
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
                user_role = cls.get_user_role_in_org(user, org)
                if required_roles and user_role not in required_roles:
                    logger.warning(f"Access denied: Insufficient role {user_role} for user {user.id}")
                    raise AccessControlError("Insufficient role permissions.")

                # Check Pro access
                if require_pro and not cls.has_pro_access(user, org):
                    logger.warning(f"Access denied: Pro access required for user {user.id}")
                    raise AccessControlError("Pro subscription required.")

                # Special case for accountants
                if allow_accountant_always and user_role == 'accountant':
                    logger.info(f"Special access granted for accountant {user.id}")
                    return view_func(request, *args, **kwargs)

                logger.info(f"Access granted for user {user.id} in organization {org.id}")
                return view_func(request, *args, **kwargs)
            
            return _wrapped_view
        return decorator 