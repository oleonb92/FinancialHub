from django.urls import path
from .views import (
    profile_view,
    MyTokenObtainPairView,
    list_users,
    approve_user,
    deny_user,
    edit_user,
    pause_user,
    delete_user,
    # list_audit_logs,
    get_user_detail,
    health_check,
)

urlpatterns = [
    path("profile/", profile_view, name="profile"),
    path("token/", MyTokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("users/", list_users, name="list_users"),
    path("users/<int:user_id>/", get_user_detail, name="get_user_detail"),
    path("users/<int:user_id>/approve/", approve_user, name="approve_user"),
    path("users/<int:user_id>/deny/", deny_user, name="deny_user"),
    path("users/<int:user_id>/edit/", edit_user, name="edit_user"),
    path("users/<int:user_id>/pause/", pause_user, name="pause_user"),
    path("users/<int:user_id>/delete/", delete_user, name="delete_user"),
    path("health/", health_check, name="health_check"),
    # path("audit-logs/", list_audit_logs, name="list-audit-logs"),
]