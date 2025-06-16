from django.db import models
from django.conf import settings
from django.utils.translation import gettext_lazy as _
from django.utils import timezone
from organizations.models import Organization

class FinancialGoal(models.Model):
    GOAL_TYPES = (
        ('saving', 'Saving'),
        ('debt_payoff', 'Debt Payoff'),
        ('investment', 'Investment'),
        ('purchase', 'Purchase'),
        ('income', 'Income'),
        ('expense_reduction', 'Expense Reduction'),
        ('emergency_fund', 'Emergency Fund'),
        ('retirement', 'Retirement'),
        ('education', 'Education'),
        ('vacation', 'Vacation'),
        ('other', 'Other'),
    )

    STATUS_CHOICES = (
        ('not_started', 'Not Started'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('paused', 'Paused'),
        ('cancelled', 'Cancelled'),
    )

    PRIORITY_CHOICES = (
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('urgent', 'Urgent'),
    )

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='financial_goals')
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name='goals')
    type = models.CharField(max_length=20, choices=GOAL_TYPES)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    target_amount = models.DecimalField(max_digits=12, decimal_places=2)
    current_amount = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    start_date = models.DateField()
    target_date = models.DateField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='not_started')
    priority = models.CharField(max_length=10, choices=PRIORITY_CHOICES, default='medium')
    
    # Progress tracking
    progress_percentage = models.FloatField(default=0)
    last_updated = models.DateTimeField(auto_now=True)
    completion_date = models.DateField(null=True, blank=True)
    
    # AI Analysis
    ai_confidence = models.FloatField(null=True, blank=True)
    ai_recommendations = models.TextField(blank=True, null=True)
    ai_risk_assessment = models.CharField(max_length=20, blank=True, null=True)
    
    # Notifications
    reminder_frequency = models.CharField(max_length=20, blank=True, null=True)
    last_reminder_sent = models.DateTimeField(null=True, blank=True)
    
    # Additional metadata
    tags = models.ManyToManyField('transactions.Tag', blank=True)
    related_accounts = models.ManyToManyField('chartofaccounts.Account', blank=True)
    attachments = models.FileField(upload_to='goal_attachments/', blank=True, null=True)
    notes = models.TextField(blank=True, null=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['organization', 'type']),
            models.Index(fields=['target_date']),
        ]

    def __str__(self):
        return f"{self.name} - {self.type}"

    def update_progress(self):
        if self.target_amount > 0:
            self.progress_percentage = (self.current_amount / self.target_amount) * 100
            if self.progress_percentage >= 100:
                self.status = 'completed'
                self.completion_date = timezone.now().date()
        self.save()

    def save(self, *args, **kwargs):
        if self.current_amount >= self.target_amount:
            self.current_amount = self.target_amount
        super().save(*args, **kwargs)

class GoalMilestone(models.Model):
    goal = models.ForeignKey(FinancialGoal, on_delete=models.CASCADE, related_name='milestones')
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    target_amount = models.DecimalField(max_digits=12, decimal_places=2)
    target_date = models.DateField()
    completed = models.BooleanField(default=False)
    completed_date = models.DateField(null=True, blank=True)
    order = models.IntegerField(default=0)

    class Meta:
        ordering = ['order', 'target_date']
        unique_together = ('goal', 'order')

    def __str__(self):
        return f"{self.name} - {self.goal.name}"

class GoalContribution(models.Model):
    goal = models.ForeignKey(FinancialGoal, on_delete=models.CASCADE, related_name='contributions')
    amount = models.DecimalField(max_digits=12, decimal_places=2)
    date = models.DateField()
    description = models.TextField(blank=True, null=True)
    transaction = models.ForeignKey('transactions.Transaction', on_delete=models.SET_NULL, null=True, blank=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

    class Meta:
        ordering = ['-date']
        indexes = [
            models.Index(fields=['goal', 'date']),
        ]

    def __str__(self):
        return f"Contribution of {self.amount} to {self.goal.name}"

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        self.goal.current_amount += self.amount
        self.goal.update_progress() 