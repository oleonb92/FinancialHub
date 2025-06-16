from django.urls import path
from payments.views import (
    CreateStripeSubscriptionView,
    CancelSubscriptionView,
    UpdateSubscriptionView,
    GetSubscriptionStatusView,
    StripeWebhookView
)

urlpatterns = [
    path('stripe/create-subscription/', CreateStripeSubscriptionView.as_view(), name='stripe-create-subscription'),
    path('stripe/cancel-subscription/', CancelSubscriptionView.as_view(), name='stripe-cancel-subscription'),
    path('stripe/update-subscription/', UpdateSubscriptionView.as_view(), name='stripe-update-subscription'),
    path('stripe/subscription-status/', GetSubscriptionStatusView.as_view(), name='stripe-subscription-status'),
    path('stripe/webhook/', StripeWebhookView.as_view(), name='stripe-webhook'),
] 