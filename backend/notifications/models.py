from django.db import models
from django.conf import settings
from django.utils.translation import gettext_lazy as _

class Notification(models.Model):
    NOTIFICATION_TYPES = (
        ('alert', 'Alert'),
        ('warning', 'Warning'),
        ('info', 'Information'),
        ('success', 'Success'),
        ('ai_insight', 'AI Insight'),
    )

    PRIORITY_LEVELS = (
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('urgent', 'Urgent'),
    )

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='notifications')
    type = models.CharField(max_length=20, choices=NOTIFICATION_TYPES)
    priority = models.CharField(max_length=10, choices=PRIORITY_LEVELS, default='medium')
    title = models.CharField(max_length=255)
    message = models.TextField()
    data = models.JSONField(default=dict, blank=True)  # Additional context data
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    scheduled_for = models.DateTimeField(null=True, blank=True)  # For scheduled notifications
    expires_at = models.DateTimeField(null=True, blank=True)
    action_url = models.URLField(blank=True, null=True)  # URL for action button
    action_text = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'is_read']),
            models.Index(fields=['type', 'priority']),
            models.Index(fields=['scheduled_for']),
        ]

    def __str__(self):
        return f"{self.get_type_display()} for {self.user.username}: {self.title}" 