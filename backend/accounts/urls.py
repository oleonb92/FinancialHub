from django.urls import path
from .views import RegisterView, list_pending_invitations

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('pending-invitations/', list_pending_invitations, name='pending-invitations'),
]