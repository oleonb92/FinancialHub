from django.db import models
from django.conf import settings
from django.utils.translation import gettext_lazy as _

class AIInteraction(models.Model):
    INTERACTION_TYPES = (
        ('query', 'Query'),
        ('analysis', 'Analysis'),
        ('prediction', 'Prediction'),
        ('recommendation', 'Recommendation'),
        ('alert', 'Alert'),
    )

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='ai_interactions')
    type = models.CharField(max_length=20, choices=INTERACTION_TYPES)
    query = models.TextField()
    response = models.TextField()
    context = models.JSONField(default=dict, blank=True)  # Additional context data
    created_at = models.DateTimeField(auto_now_add=True)
    confidence_score = models.FloatField(null=True, blank=True)
    feedback = models.BooleanField(null=True, blank=True)  # User feedback on response
    feedback_comment = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'type']),
            models.Index(fields=['created_at']),
        ]

    def __str__(self):
        return f"AI {self.get_type_display()} for {self.user.username}"

class AIInsight(models.Model):
    INSIGHT_TYPES = (
        ('budget', 'Budget'),
        ('spending', 'Spending'),
        ('saving', 'Saving'),
        ('investment', 'Investment'),
        ('risk', 'Risk'),
        ('opportunity', 'Opportunity'),
    )

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='ai_insights')
    type = models.CharField(max_length=20, choices=INSIGHT_TYPES)
    title = models.CharField(max_length=255)
    description = models.TextField()
    data = models.JSONField(default=dict, blank=True)  # Supporting data
    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)
    action_taken = models.BooleanField(default=False)
    action_description = models.TextField(blank=True, null=True)
    impact_score = models.FloatField(null=True, blank=True)  # Estimated impact score

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'type']),
            models.Index(fields=['is_read']),
        ]

    def __str__(self):
        return f"AI Insight: {self.title} for {self.user.username}"

class AIPrediction(models.Model):
    PREDICTION_TYPES = (
        ('budget', 'Budget'),
        ('cash_flow', 'Cash Flow'),
        ('spending', 'Spending'),
        ('saving', 'Saving'),
        ('risk', 'Risk'),
    )

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='ai_predictions')
    type = models.CharField(max_length=20, choices=PREDICTION_TYPES)
    prediction = models.JSONField()  # Structured prediction data
    confidence_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    prediction_date = models.DateField()  # Date for which prediction is made
    actual_result = models.JSONField(null=True, blank=True)  # Actual result when available
    accuracy_score = models.FloatField(null=True, blank=True)  # Accuracy score when actual result is available

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'type']),
            models.Index(fields=['prediction_date']),
        ]

    def __str__(self):
        return f"AI Prediction: {self.get_type_display()} for {self.user.username} on {self.prediction_date}" 