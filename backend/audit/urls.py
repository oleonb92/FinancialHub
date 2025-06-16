from django.urls import path
from .views import list_audit_logs

urlpatterns = [
    path("audit-logs/", list_audit_logs),
]