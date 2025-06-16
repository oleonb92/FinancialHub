from rest_framework import serializers
from .models import Notification

class NotificationSerializer(serializers.ModelSerializer):
    type_display = serializers.CharField(source='get_type_display', read_only=True)
    priority_display = serializers.CharField(source='get_priority_display', read_only=True)

    class Meta:
        model = Notification
        fields = [
            'id', 'type', 'type_display', 'priority', 'priority_display',
            'title', 'message', 'data', 'is_read', 'created_at',
            'scheduled_for', 'expires_at', 'action_url', 'action_text'
        ]
        read_only_fields = ['created_at']

class NotificationBulkUpdateSerializer(serializers.Serializer):
    notification_ids = serializers.ListField(
        child=serializers.IntegerField(),
        min_length=1
    )
    is_read = serializers.BooleanField(required=False)
    action_taken = serializers.BooleanField(required=False) 