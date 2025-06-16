import logging
from organizations.models import Organization, OrganizationMembership
from django.http import JsonResponse
from core.exceptions import OrganizationError
from rest_framework_simplejwt.authentication import JWTAuthentication
from django.utils.deprecation import MiddlewareMixin

logger = logging.getLogger(__name__)

class OrganizationMiddleware(MiddlewareMixin):
    """
    Middleware para manejar la autenticación y la organización actual.
    """
    def __init__(self, get_response):
        self.get_response = get_response
        self.jwt_auth = JWTAuthentication()

    def __call__(self, request):
        # Rutas que no requieren organización
        exempt_paths = [
            '/api/token/',
            '/api/token/refresh/',
            '/api/token/verify/',
            '/api/register/',
            '/api/auth/register/',
            '/api/password-reset/',
            '/api/password-reset/confirm/',
            '/api/organizations/',
            '/api/organizations/create/',
            '/api/payments/webhook/',
            '/api/accounts/profile/',
            '/api/accounts/me/',
            '/api/profile/',
            '/admin/',
            '/admin/login/',
        ]
        
        # Log de la ruta actual para depuración
        logger.debug(f"Procesando request para ruta: {request.path}")
        
        if any(request.path.startswith(path) for path in exempt_paths):
            logger.debug(f"Ruta exenta de organización: {request.path}")
            return self.get_response(request)

        # Verificar autenticación
        try:
            auth_tuple = self.jwt_auth.authenticate(request)
            if auth_tuple is not None:
                request.user, request.auth = auth_tuple
                logger.debug(f"Usuario autenticado: {request.user}")
            else:
                logger.warning(f"Request no autenticado para ruta: {request.path}")
                return JsonResponse({
                    'detail': 'Authentication credentials were not provided.',
                    'code': 'not_authenticated'
                }, status=401)
        except Exception as e:
            logger.error(f"Error en autenticación: {str(e)}")
            return JsonResponse({
                'detail': 'Error en autenticación.',
                'code': 'authentication_error'
            }, status=401)

        # Procesar organización
        org_id = request.headers.get('X-Organization-ID') or request.GET.get('organization_id')
        org = None

        if org_id:
            try:
                org = Organization.objects.get(id=org_id)
                # Verificar que el usuario pertenece a la organización
                if not OrganizationMembership.objects.filter(user=request.user, organization=org).exists():
                    logger.error(f"Usuario {request.user} no pertenece a la organización {org_id}")
                    return JsonResponse({
                        'detail': 'No perteneces a esta organización.',
                        'code': 'organization_not_member'
                    }, status=403)
                request.organization = org
                logger.info(f"Organización {org_id} inyectada por header/query en request de {request.user}")
            except Organization.DoesNotExist:
                logger.error(f"Organización {org_id} no encontrada para request de {request.user}")
                return JsonResponse({
                    'detail': 'Organización no encontrada.',
                    'code': 'organization_not_found'
                }, status=404)
        else:
            memberships = OrganizationMembership.objects.filter(user=request.user)
            if memberships.count() == 1:
                org = memberships.first().organization
                request.organization = org
                logger.info(f"Organización {org.id} inyectada por membresía única en request de {request.user}")
            elif memberships.count() > 1:
                # Si el usuario tiene múltiples organizaciones, requerimos que especifique una
                logger.warning(f"Usuario {request.user} tiene múltiples organizaciones. Se requiere especificar una.")
                return JsonResponse({
                    'detail': 'Se requiere especificar una organización.',
                    'code': 'organization_required',
                    'organizations': [{'id': m.organization.id, 'name': m.organization.name} for m in memberships],
                    'debug_info': {
                        'user_id': request.user.id,
                        'username': request.user.username,
                        'path': request.path,
                        'method': request.method
                    }
                }, status=400)
            else:
                # Si el usuario no tiene organizaciones, permitimos continuar
                request.organization = None
                logger.info(f"Usuario {request.user} no tiene organizaciones.")

        # Log de la organización asignada para depuración
        logger.debug(f"Organización asignada: {getattr(request, 'organization', None)}")
        
        return self.get_response(request) 