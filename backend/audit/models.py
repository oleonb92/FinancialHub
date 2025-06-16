from django.db import models
from django.conf import settings

class AuditLog(models.Model):
    ACTION_CHOICES = [
        ('approve', 'Approve'),
        ('pause', 'Pause'),
        ('resume', 'Resume'),
        ('delete', 'Delete'),
        ('edit', 'Edit'),
        ('reject', 'Reject'),
        ('login', 'Login'),
        ('transaction', 'Transaction Change'),
        ('budget_edit', 'Budget Edit'),
    ]

    action = models.CharField(max_length=30, choices=ACTION_CHOICES)
    performed_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="logs_performed")
    target = models.CharField(max_length=255, help_text="What the action was performed on (username, module, transaction ID, etc.)")
    timestamp = models.DateTimeField(auto_now_add=True)
    details = models.TextField(blank=True)

    def __str__(self):
        return f"{self.performed_by.username} {self.action} {self.target} @ {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"