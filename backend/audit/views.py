from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from audit.models import AuditLog

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_audit_logs(request):
    if getattr(request.user, "role", "") != "admin":
        return Response({"detail": "Not authorized."}, status=status.HTTP_403_FORBIDDEN)

    logs = AuditLog.objects.select_related("performed_by").order_by("-timestamp")[:200]
    data = []
    for log in logs:
        data.append({
            "id": log.id,
            "action": log.action,
            "performed_by": log.performed_by.username,
            "target": log.target,
            "timestamp": log.timestamp.isoformat(),
            "details": log.details,
        })
    return Response(data)