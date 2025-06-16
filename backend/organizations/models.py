from django.db import models
from accounts.models import User

class Organization(models.Model):
    """
    Representa una organización (empresa, familia, negocio, etc.)
    Puede tener diferentes planes y un sponsor (quien paga la suscripción principal).
    """
    PLAN_CHOICES = [
        ('free', 'Free'),
        ('pro', 'Pro'),
        ('enterprise', 'Enterprise'),
    ]
    name = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    plan = models.CharField(max_length=20, choices=PLAN_CHOICES, default='free')
    sponsor = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL, related_name='sponsored_organizations')
    # Puedes agregar más campos: fecha de expiración, datos de facturación, etc.

    def __str__(self):
        return self.name

class OrganizationMembership(models.Model):
    """
    Relación muchos-a-muchos entre usuario y organización.
    Permite múltiples roles y flags de acceso pro.
    """
    ROLE_CHOICES = [
        ('owner', 'Owner'),
        ('admin', 'Admin'),
        ('member', 'Member'),
        ('accountant', 'Accountant'),
        ('bookkeeper', 'Bookkeeper'),
        ('advisor', 'Financial Advisor'),
    ]
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='organization_memberships')
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name='memberships')
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    pro_features_for_accountant = models.BooleanField(default=False)
    joined_at = models.DateTimeField(auto_now_add=True)
    invited_by = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL, related_name='organization_invitations_sent')

    class Meta:
        unique_together = ('user', 'organization')
        verbose_name = 'Organization Membership'
        verbose_name_plural = 'Organization Memberships'

    def __str__(self):
        return f"{self.user.username} in {self.organization.name} as {self.get_role_display()}"
