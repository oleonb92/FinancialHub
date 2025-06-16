from django.contrib import admin
from django.contrib.auth import get_user_model
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.utils.html import format_html
from .models import User, PendingInvitation

User = get_user_model()

class CustomUserAdmin(BaseUserAdmin):
    list_display = (
        "username", "email", "first_name", "last_name", "organization", "role", "is_staff"
    )
    list_filter = ("role", "organization", "is_staff", "is_superuser", "is_active", "groups")
    search_fields = ("username", "first_name", "last_name", "email")
    ordering = ("username",)
    fieldsets = (
        (None, {"fields": ("username", "password")}),
        ("Personal info", {"fields": ("first_name", "last_name", "email", "birthdate", "avatar")}),
        ("Organization", {"fields": ("organization", "role")}),
        ("Preferences", {"fields": ("preferred_language", "notification_preferences", "ai_assistant_enabled")}),
        ("Account Type", {"fields": ("account_type", "pro_features", "pro_trial_until", "pro_features_list")}),
        ("Permissions", {"fields": ("is_active", "is_staff", "is_superuser", "groups", "user_permissions")}),
        ("Important dates", {"fields": ("last_login", "date_joined")}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('username', 'email', 'password1', 'password2', 'organization', 'role'),
        }),
    )

    def avatar_tag(self, obj):
        if obj.avatar:
            return format_html('<img src="{}" style="width:40px;height:40px;border-radius:50%;" />', obj.avatar.url)
        return "-"
    avatar_tag.short_description = "Avatar"

# Solo registramos si aún no ha sido registrado
try:
    admin.site.register(User, CustomUserAdmin)
except admin.sites.AlreadyRegistered:
    pass

admin.site.register(PendingInvitation)