from django.contrib.auth.models import AbstractUser, Group, Permission
from django.db import models
from django.contrib.postgres.fields import ArrayField
from django.utils import timezone
from django.conf import settings
# Translation handled by TranslationService

class User(AbstractUser):
    ROLE_CHOICES = [
        ('owner', 'Owner'),
        ('admin', 'Admin'),
        ('member', 'Member'),
        ('accountant', 'Accountant'),
        ('bookkeeper', 'Bookkeeper'),
        ('advisor', 'Financial Advisor'),
    ]

    LANGUAGE_CHOICES = [
        ('en', 'English'),
        ('es', 'Spanish'),
    ]

    ACCOUNT_TYPE_CHOICES = [
        ('personal', 'Personal'),
        ('accountant', 'Accountant'),
    ]

    role = models.CharField(
        max_length=20,
        choices=ROLE_CHOICES,
        null=True,
        blank=True
    )
    organization = models.ForeignKey(
        'organizations.Organization',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='members'
    )

    was_approved = models.BooleanField(default=False)
    birthdate = models.DateField(null=True, blank=True)
    preferred_language = models.CharField(
        max_length=2,
        choices=LANGUAGE_CHOICES,
        default='en'
    )
    notification_preferences = models.JSONField(default=dict, blank=True)
    ai_assistant_enabled = models.BooleanField(default=True)

    account_type = models.CharField(
        max_length=20,
        choices=ACCOUNT_TYPE_CHOICES,
        default='personal',
        help_text='Account type: personal or accountant.'
    )
    pro_features = models.BooleanField(
        default=False,
        help_text='Indicates if the user has global access to Pro features (e.g., pro accountant).'
    )
    pro_trial_until = models.DateTimeField(
        null=True,
        blank=True,
        help_text='Date until which the user has an active Pro trial.'
    )
    pro_features_list = models.JSONField(
        default=list,
        blank=True,
        help_text='List of active Pro features for this user.'
    )

    groups = models.ManyToManyField(
        Group,
        related_name='custom_user_groups',
        blank=True,
        help_text='The groups this user belongs to.',
        verbose_name='groups'
    )

    user_permissions = models.ManyToManyField(
        Permission,
        related_name='custom_user_permissions',
        blank=True,
        help_text='Specific permissions for this user.',
        verbose_name='user permissions'
    )
    
    avatar = models.ImageField(upload_to='avatars/', blank=True, null=True)

    class Meta:
        verbose_name = 'user'
        verbose_name_plural = 'users'

    def __str__(self):
        return f"{self.username} ({self.get_role_display()})"

    def has_pro_access(self):
        """Check if user has access to Pro features"""
        if self.pro_features:
            return True
        if self.pro_trial_until and self.pro_trial_until > timezone.now():
            return True
        return False

    def has_pro_feature(self, feature):
        """Check if user has access to a specific Pro feature"""
        if self.has_pro_access():
            return True
        return feature in self.pro_features_list

class PendingInvitation(models.Model):
    username = models.CharField(max_length=150)
    organization = models.ForeignKey('organizations.Organization', on_delete=models.CASCADE, null=True, blank=True)
    requested_at = models.DateTimeField(auto_now_add=True)
    is_approved = models.BooleanField(default=False)
    role = models.CharField(max_length=20, choices=User.ROLE_CHOICES, default='member')

    def __str__(self):
        return f"{self.username} requested to join {self.organization.name if self.organization else 'Unknown'} as {self.get_role_display()}"