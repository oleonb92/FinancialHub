from django.contrib import admin
from .models import Account

@admin.register(Account)
class AccountAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'code', 'type', 'organization', 'created_at', 'updated_at')
    list_filter = ('type', 'organization')
    search_fields = ('name', 'code')
    ordering = ('organization', 'type', 'name')