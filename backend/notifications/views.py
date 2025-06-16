from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.utils import timezone
from .models import Notification
from .serializers import (
    NotificationSerializer,
    NotificationBulkUpdateSerializer
)
from .services import NotificationService

class NotificationViewSet(viewsets.ModelViewSet):
    serializer_class = NotificationSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Notification.objects.filter(user=self.request.user)

    @action(detail=False, methods=['post'])
    def bulk_update(self, request):
        serializer = NotificationBulkUpdateSerializer(data=request.data)
        if serializer.is_valid():
            notification_ids = serializer.validated_data['notification_ids']
            is_read = serializer.validated_data.get('is_read')
            action_taken = serializer.validated_data.get('action_taken')

            notifications = self.get_queryset().filter(id__in=notification_ids)
            
            if is_read is not None:
                notifications.update(is_read=is_read)
            
            if action_taken is not None:
                notifications.update(action_taken=action_taken)

            return Response({'status': 'success'})
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['post'])
    def mark_all_read(self, request):
        self.get_queryset().update(is_read=True)
        return Response({'status': 'success'})

    @action(detail=False, methods=['get'])
    def unread_count(self, request):
        count = self.get_queryset().filter(is_read=False).count()
        return Response({'unread_count': count}) 