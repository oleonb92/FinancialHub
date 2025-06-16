from rest_framework.response import Response
from rest_framework import status
from functools import wraps

def admin_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if getattr(request.user, "role", "") != "admin":
            return Response({"error": "Not authorized."}, status=status.HTTP_403_FORBIDDEN)
        return view_func(request, *args, **kwargs)
    return wrapper