from django.shortcuts import render
from rest_framework import viewsets, permissions, status
from .models import Chat, Message
from .serializers import ChatSerializer, MessageSerializer
from rest_framework.decorators import action
from rest_framework.response import Response
from django.core.exceptions import ObjectDoesNotExist
import logging
import traceback

logger = logging.getLogger(__name__)

# Create your views here.

class ChatViewSet(viewsets.ModelViewSet):
    serializer_class = ChatSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        if getattr(self, 'swagger_fake_view', False):
            return Chat.objects.none()
        if not self.request.user.is_authenticated:
            return Chat.objects.none()
        queryset = Chat.objects.filter(participants=self.request.user).distinct()
        logger.info(f"User {self.request.user.username} requesting chats. Found {queryset.count()} chats")
        return queryset

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context['request'] = self.request
        return context

    def get_object(self):
        try:
            obj = super().get_object()
            logger.info(f"User {self.request.user.username} accessing chat {obj.id}")
            logger.info(f"Chat participants: {[u.username for u in obj.participants.all()]}")
            return obj
        except Exception as e:
            logger.error(f"Error in get_object: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    @action(detail=True, methods=['get'])
    def messages(self, request, pk=None):
        try:
            logger.info(f"Attempting to fetch messages for chat {pk}")
            logger.info(f"User: {request.user.username}")
            
            chat = self.get_object()
            logger.info(f"Successfully retrieved chat {chat.id}")
            
            if not chat.participants.filter(id=request.user.id).exists():
                logger.warning(f"User {request.user.username} tried to access chat {pk} without permission")
                return Response(
                    {"detail": "You don't have permission to access this chat"},
                    status=status.HTTP_403_FORBIDDEN
                )

            messages = chat.messages.order_by('created_at')
            logger.info(f"Found {messages.count()} messages for chat {chat.id}")
            
            serializer = MessageSerializer(messages, many=True)
            return Response(serializer.data)
            
        except ObjectDoesNotExist:
            logger.error(f"Chat {pk} not found")
            return Response(
                {"detail": "Chat not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Error fetching messages for chat {pk}: {str(e)}")
            logger.error(traceback.format_exc())
            return Response(
                {"detail": f"An error occurred while fetching messages: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class MessageViewSet(viewsets.ModelViewSet):
    serializer_class = MessageSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        if getattr(self, 'swagger_fake_view', False):
            return Message.objects.none()
        if not self.request.user.is_authenticated:
            return Message.objects.none()
        return Message.objects.filter(chat__participants=self.request.user)
    
    def perform_create(self, serializer):
        serializer.save(sender=self.request.user)

    @action(detail=True, methods=['patch'], url_path='delete_for_me')
    def delete_for_me(self, request, pk=None):
        message = self.get_object()
        user = request.user
        message.deleted_for.add(user)
        return Response({'status': 'deleted for you'})

    @action(detail=True, methods=['patch'], url_path='delete_for_everyone')
    def delete_for_everyone(self, request, pk=None):
        message = self.get_object()
        user = request.user
        if message.sender != user:
            return Response({'error': 'Solo el autor puede borrar para todos.'}, status=status.HTTP_403_FORBIDDEN)
        message.deleted_for_everyone = True
        message.save()
        return Response({'status': 'deleted for everyone'})

    @action(detail=True, methods=['patch'], url_path='react')
    def react(self, request, pk=None):
        message = self.get_object()
        user = request.user
        emoji = request.data.get('emoji')
        add = request.data.get('add', True)
        if not emoji:
            return Response({'error': 'Emoji is required.'}, status=400)
        reactions = message.reactions or {}
        user_id = str(user.id)
        if add:
            reactions.setdefault(emoji, [])
            if user_id not in reactions[emoji]:
                reactions[emoji].append(user_id)
        else:
            if emoji in reactions and user_id in reactions[emoji]:
                reactions[emoji].remove(user_id)
                if not reactions[emoji]:
                    del reactions[emoji]
        message.reactions = reactions
        message.save()
        return Response({'status': 'reacted', 'reactions': reactions})

    @action(detail=True, methods=['patch'], url_path='pin')
    def pin(self, request, pk=None):
        message = self.get_object()
        pin = request.data.get('pinned', True)
        message.pinned = bool(pin)
        message.save()
        return Response({'status': 'pinned', 'pinned': message.pinned})

    @action(detail=True, methods=['patch'], url_path='star')
    def star(self, request, pk=None):
        message = self.get_object()
        user = request.user
        star = request.data.get('star', True)
        if star:
            message.starred_by.add(user)
        else:
            message.starred_by.remove(user)
        message.save()
        return Response({'status': 'starred', 'starred_by': list(message.starred_by.values_list('id', flat=True))})

    @action(detail=True, methods=['patch'], url_path='forward')
    def forward(self, request, pk=None):
        message = self.get_object()
        chat_id = request.data.get('chat')
        if not chat_id:
            return Response({'error': 'Chat ID is required.'}, status=400)
        try:
            chat = Chat.objects.get(id=chat_id)
        except Chat.DoesNotExist:
            return Response({'error': 'Chat not found.'}, status=404)
        new_msg = Message.objects.create(
            chat=chat,
            sender=request.user,
            text=message.text,
            attachment=message.attachment,
            forwarded_from=message
        )
        serializer = MessageSerializer(new_msg, context={'request': request})
        return Response({'status': 'forwarded', 'message': serializer.data})
