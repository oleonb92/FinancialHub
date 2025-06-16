from django.utils import timezone
from django.core.mail import send_mail
from django.conf import settings
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from .models import Notification

class NotificationService:
    def create_notification(
        self,
        user,
        type,
        title,
        message,
        priority='medium',
        data=None,
        scheduled_for=None,
        expires_at=None,
        action_url=None,
        action_text=None
    ):
        """
        Create a new notification for a user
        """
        notification = Notification.objects.create(
            user=user,
            type=type,
            title=title,
            message=message,
            priority=priority,
            data=data or {},
            scheduled_for=scheduled_for,
            expires_at=expires_at,
            action_url=action_url,
            action_text=action_text
        )

        # Send real-time notification
        self._send_realtime_notification(notification)
        
        # Send email if enabled
        if self._should_send_email(user):
            self._send_email_notification(notification)

        return notification

    def _send_realtime_notification(self, notification):
        """
        Send notification through WebSocket
        """
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            f"user_{notification.user.id}",
            {
                "type": "notification.message",
                "message": {
                    "id": notification.id,
                    "type": notification.type,
                    "title": notification.title,
                    "message": notification.message,
                    "created_at": notification.created_at.isoformat(),
                }
            }
        )

    def _send_email_notification(self, notification):
        """
        Send notification via email
        """
        try:
            send_mail(
                subject=notification.title,
                message=notification.message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[notification.user.email],
                fail_silently=True
            )
        except Exception as e:
            print(f"Error sending email notification: {e}")

    def _should_send_email(self, user):
        """
        Check if email notifications are enabled for the user
        """
        preferences = getattr(user, 'notification_preferences', {})
        return preferences.get('email_enabled', True)

    def process_scheduled_notifications(self):
        """
        Process notifications that are scheduled for now
        """
        now = timezone.now()
        scheduled_notifications = Notification.objects.filter(
            scheduled_for__lte=now,
            is_read=False
        )

        for notification in scheduled_notifications:
            self._send_realtime_notification(notification)
            if self._should_send_email(notification.user):
                self._send_email_notification(notification)

    def clean_expired_notifications(self):
        """
        Clean up expired notifications
        """
        now = timezone.now()
        Notification.objects.filter(
            expires_at__lte=now,
            is_read=True
        ).delete() 