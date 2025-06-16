# from django.db import models
# from households.models import Household

# class Account(models.Model):
#     ACCOUNT_TYPES = [
#         ('ASSET', 'Activo'),
#         ('LIABILITY', 'Pasivo'),
#         ('EQUITY', 'Patrimonio'),
#         ('INCOME', 'Ingreso'),
#         ('EXPENSE', 'Gasto'),
#         ('BANK', 'Cuenta Bancaria'),
#         ('CREDIT', 'Tarjeta de Crédito'),
#     ]

#     name = models.CharField(max_length=100)
#     type = models.CharField(max_length=20, choices=ACCOUNT_TYPES)
#     household = models.ForeignKey(Household, on_delete=models.CASCADE, related_name='accounts')
#     is_default = models.BooleanField(default=False)

#     def __str__(self):
#         return f"{self.name} ({self.type})"


from django.db import models
from organizations.models import Organization

class Account(models.Model):
    ACCOUNT_TYPES = [
        ('ASSET', 'Activo'),
        ('LIABILITY', 'Pasivo'),
        ('EQUITY', 'Patrimonio'),
        ('INCOME', 'Ingreso'),
        ('EXPENSE', 'Gasto'),
        ('BANK', 'Cuenta Bancaria'),
        ('CREDIT', 'Tarjeta de Crédito'),
    ]

    FUNCTIONAL_TYPES = [
        ('BANK', 'Cuenta Bancaria'),
        ('CREDIT_CARD', 'Tarjeta de Crédito'),
        ('WALLET', 'Billetera Digital'),
        ('CASH', 'Efectivo'),
        ('OTHER', 'Otro'),
    ]

    name = models.CharField(max_length=100)
    code = models.CharField(max_length=20, unique=True)
    type = models.CharField(max_length=50, choices=ACCOUNT_TYPES)
    functional_type = models.CharField(max_length=20, choices=FUNCTIONAL_TYPES, default='OTHER')
    currency = models.CharField(max_length=10, default='USD')
    is_default = models.BooleanField(default=False)
    parent = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True, related_name='children')
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name='accounts')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.code} - {self.name}"

    class Meta:
        unique_together = ('code', 'organization')
        ordering = ['code']