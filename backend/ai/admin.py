from django.contrib import admin
from django.utils.html import format_html
from .models import AIInteraction, AIInsight, AIPrediction

@admin.register(AIInteraction)
class AIInteractionAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'organization', 'type', 'created_at', 'confidence_score', 'feedback']
    list_filter = ['type', 'created_at', 'confidence_score', 'feedback', 'organization']
    search_fields = ['user__username', 'user__email', 'query', 'response']
    readonly_fields = ['created_at']
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('user', 'organization', 'type', 'created_at')
        }),
        ('Interaction Details', {
            'fields': ('query', 'response', 'context')
        }),
        ('Analysis', {
            'fields': ('confidence_score', 'feedback', 'feedback_comment')
        }),
    )

@admin.register(AIInsight)
class AIInsightAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'organization', 'type', 'title', 'created_at', 'is_read', 'action_taken', 'impact_score']
    list_filter = ['type', 'created_at', 'is_read', 'action_taken', 'organization']
    search_fields = ['user__username', 'user__email', 'title', 'description']
    readonly_fields = ['created_at']
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('user', 'organization', 'type', 'title', 'created_at')
        }),
        ('Insight Details', {
            'fields': ('description', 'data')
        }),
        ('Status', {
            'fields': ('is_read', 'action_taken', 'action_description', 'impact_score')
        }),
    )

@admin.register(AIPrediction)
class AIPredictionAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'organization', 'type', 'prediction_date', 'created_at', 'confidence_score', 'accuracy_score']
    list_filter = ['type', 'created_at', 'prediction_date', 'organization']
    search_fields = ['user__username', 'user__email']
    readonly_fields = ['created_at']
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('user', 'organization', 'type', 'prediction_date', 'created_at')
        }),
        ('Prediction Details', {
            'fields': ('prediction', 'confidence_score')
        }),
        ('Results', {
            'fields': ('actual_result', 'accuracy_score')
        }),
    )

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user', 'organization')

# Customize admin site
admin.site.site_header = "FinancialHub AI Administration"
admin.site.site_title = "FinancialHub AI Admin"
admin.site.index_title = "Welcome to FinancialHub AI Administration" 