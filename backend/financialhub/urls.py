from django.contrib import admin
from django.urls import path, include
from api.views import MyTokenObtainPairView, health_check
from rest_framework_simplejwt.views import TokenRefreshView
from django.conf import settings
from django.conf.urls.static import static
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from django.shortcuts import redirect
from django.views.generic import TemplateView
from django.http import HttpResponseServerError
from django.http import HttpResponse

# API Info for Swagger
api_info = openapi.Info(
    title="Financial Hub API",
    default_version='v1',
    description="API for Financial Hub application",
    terms_of_service="https://www.google.com/policies/terms/",
    contact=openapi.Contact(email="contact@financialhub.com"),
    license=openapi.License(name="BSD License"),
)

# Swagger Schema View
schema_view = get_schema_view(
    api_info,
    public=True,
    permission_classes=(permissions.AllowAny,),
)

def canary_view(request):
    return HttpResponse("😊 Canary OK", content_type="text/plain")

def redirect_to_swagger(request):
    return redirect('/swagger/')

def handler500(request):
    return HttpResponseServerError(
        """
        <html>
            <head>
                <title>500 Server Error</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .error-container { max-width: 600px; margin: 0 auto; }
                    h1 { color: #e74c3c; }
                </style>
            </head>
            <body>
                <div class="error-container">
                    <h1>500 Server Error</h1>
                    <p>Lo sentimos, ha ocurrido un error en el servidor.</p>
                    <p>Por favor, intenta acceder a la <a href="/swagger/">documentación de la API</a>.</p>
                </div>
            </body>
        </html>
        """
    )

urlpatterns = [
    path("canary/", canary_view),
    path("", redirect_to_swagger, name="home"),
    path("admin/", admin.site.urls),
    path("api/health-check/", health_check, name="health-check"),

    # API grouped by app routers
    path("api/", include("api.urls")),
    path("api/auth/", include("accounts.urls")),
    path("api/transactions/", include("transactions.urls")),
    path("api/organizations/", include("organizations.urls")),
    path("api/chat/", include("chat.urls")),
    path("api/notifications/", include("notifications.urls")),
    path("api/ai/", include("ai.urls")),
    path("api/goals/", include("goals.urls")),
    path("api/chartofaccounts/", include("chartofaccounts.urls")),
    path('api/payments/', include('payments.urls')),
    path('api/incentives/', include('incentives.urls')),

    # JWT tokens
    path("api/token/", MyTokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("api/token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),

    # Swagger/OpenAPI URLs
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
]

def test_error_view(request):
    raise Exception("🔥 Error forzado para probar el middleware Show500ErrorMiddleware")

urlpatterns += [
    path("test-error/", test_error_view, name="test_error"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)