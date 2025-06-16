from django.db import models
from accounts.models import User
from organizations.models import Organization

class Incentive(models.Model):
    accountant = models.ForeignKey(User, on_delete=models.CASCADE, related_name='incentives')
    client = models.ForeignKey(User, on_delete=models.CASCADE, related_name='client_incentives')
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE)
    amount = models.DecimalField(max_digits=8, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)
    description = models.TextField(blank=True) 