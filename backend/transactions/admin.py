from django.contrib import admin
from .models import Transaction, Tag, Category

@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    list_display = ('id', 'description', 'amount', 'date', 'type', 'organization', 'source_account', 'status')
    list_filter = ('type', 'organization', 'status', 'date')
    search_fields = ('description',)
    ordering = ('-date',)
    filter_horizontal = ('tags',)
    date_hierarchy = 'date'

@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ('id', 'name')
    search_fields = ('name',)
    ordering = ('name',)

@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'organization', 'parent')
    list_filter = ('organization', 'parent')
    search_fields = ('name',)
    ordering = ('name',)
