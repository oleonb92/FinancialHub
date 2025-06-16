from django.db import models
from organizations.models import Organization

class Subscription(models.Model):
    organization = models.OneToOneField(Organization, on_delete=models.CASCADE, related_name='subscription')
    stripe_customer_id = models.CharField(max_length=128)
    stripe_subscription_id = models.CharField(max_length=128)
    plan = models.CharField(max_length=32)
    status = models.CharField(max_length=32)
    current_period_end = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class Payment(models.Model):
    subscription = models.ForeignKey(Subscription, on_delete=models.CASCADE, related_name='payments')
    amount = models.DecimalField(max_digits=8, decimal_places=2)
    currency = models.CharField(max_length=8)
    paid_at = models.DateTimeField()
    stripe_payment_intent = models.CharField(max_length=128)
    status = models.CharField(max_length=32) 