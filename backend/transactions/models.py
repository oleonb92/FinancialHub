from django.db import models
from accounts.models import User
from organizations.models import Organization
from chartofaccounts.models import Account
from django.utils import timezone
from django.db.models import Sum
from django.db.models.functions import ExtractYear, ExtractMonth

class Tag(models.Model):
    name = models.CharField(max_length=255, unique=True)
    color = models.CharField(max_length=7, default='#000000')  # Hex color code
    icon = models.CharField(max_length=50, blank=True, null=True)  # Icon identifier
    
    def __str__(self):
        return self.name

class Category(models.Model):
    name = models.CharField(max_length=100)
    icon = models.CharField(max_length=50, blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    parent = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True, related_name='children')
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name='categories')
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "Categories"
        unique_together = ('name', 'organization', 'parent')

    def __str__(self):
        return self.name

class Transaction(models.Model):
    TRANSACTION_TYPES = [
        ('INCOME', 'Income'),
        ('EXPENSE', 'Expense'),
        ('TRANSFER', 'Transfer'),
        ('INVESTMENT', 'Investment'),
        ('LOAN', 'Loan'),
        ('REFUND', 'Refund'),
    ]

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('confirmed', 'Confirmed'),
        ('reconciled', 'Reconciled'),
        ('void', 'Void'),
        ('disputed', 'Disputed'),
    ]

    type = models.CharField(max_length=20, choices=TRANSACTION_TYPES)
    amount = models.DecimalField(max_digits=12, decimal_places=2)
    date = models.DateField()
    description = models.TextField(blank=True, null=True)
    category = models.ForeignKey(Category, on_delete=models.SET_NULL, null=True, blank=True, related_name='transactions')
    subcategory = models.ForeignKey(Category, on_delete=models.SET_NULL, null=True, blank=True, related_name='subcategory_transactions')

    source_account = models.ForeignKey(
        Account, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='source_transactions'
    )
    destination_account = models.ForeignKey(
        Account, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='destination_transactions'
    )

    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name='transactions')
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    approved_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='approved_transactions')

    is_imported = models.BooleanField(default=False)
    bank_transaction_id = models.CharField(max_length=255, blank=True, null=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    tags = models.ManyToManyField(Tag, related_name="transactions", blank=True)

    # Additional fields
    receipt = models.FileField(upload_to='receipts/', blank=True, null=True)
    notes = models.TextField(blank=True, null=True)
    location = models.CharField(max_length=255, blank=True, null=True)
    merchant = models.CharField(max_length=255, blank=True, null=True)
    payment_method = models.CharField(max_length=50, blank=True, null=True)
    recurring = models.BooleanField(default=False)
    recurring_frequency = models.CharField(max_length=50, blank=True, null=True)
    recurring_end_date = models.DateField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)
    reconciled_at = models.DateTimeField(null=True, blank=True)
    voided_at = models.DateTimeField(null=True, blank=True)

    # AI Analysis fields
    ai_analyzed = models.BooleanField(default=False)
    ai_confidence = models.FloatField(null=True, blank=True)
    ai_category_suggestion = models.ForeignKey(Category, on_delete=models.SET_NULL, null=True, blank=True, related_name='ai_suggested_transactions')
    ai_notes = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ['-date', '-created_at']
        indexes = [
            models.Index(fields=['organization', 'date']),
            models.Index(fields=['type', 'status']),
            models.Index(fields=['category']),
            models.Index(fields=['merchant']),
        ]

    def __str__(self):
        return f"{self.type} - {self.amount} on {self.date}"

    def save(self, *args, **kwargs):
        if self.status == 'reconciled' and not self.reconciled_at:
            self.reconciled_at = timezone.now()
        elif self.status == 'void' and not self.voided_at:
            self.voided_at = timezone.now()
        super().save(*args, **kwargs)

class Budget(models.Model):
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='budgets')
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name='budgets')
    amount = models.DecimalField(max_digits=12, decimal_places=2)
    period = models.CharField(max_length=7)  # Format: YYYY-MM
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('category', 'organization', 'period')
        ordering = ['-period', 'category__name']

    def __str__(self):
        return f"{self.category.name} - {self.period}"

    def get_all_subcategory_ids(self, category):
        """Recursively get all subcategory IDs for a given category."""
        subcategories = list(category.children.all())
        ids = [category.id]
        for subcat in subcategories:
            ids += self.get_all_subcategory_ids(subcat)
        return ids

    @property
    def spent_amount(self):
        """Calculate the total spent amount for this budget's category and all subcategories in the current period"""
        year, month = map(int, self.period.split('-'))
        category_ids = self.get_all_subcategory_ids(self.category)
        return Transaction.objects.filter(
            category_id__in=category_ids,
            organization=self.organization,
            type='EXPENSE',
            date__year=year,
            date__month=month
        ).aggregate(total=Sum('amount'))['total'] or 0

    @property
    def remaining_amount(self):
        """Calculate the remaining amount in the budget"""
        return self.amount - self.spent_amount

    @property
    def percentage_used(self):
        """Calculate the percentage of budget used"""
        if self.amount == 0:
            return 0
        return (self.spent_amount / self.amount) * 100
    
    