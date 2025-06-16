from django.contrib import admin
from .models import AuditLog

@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ('id', 'performed_by', 'action', 'target', 'timestamp')
    list_filter = ('action', 'timestamp')
    search_fields = ('performed_by__username', 'target')
    ordering = ('-timestamp',)
    date_hierarchy = 'timestamp'
    readonly_fields = ('performed_by', 'action', 'target', 'timestamp', 'details')
