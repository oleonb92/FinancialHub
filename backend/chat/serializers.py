from rest_framework import serializers
from .models import Chat, Message
from transactions.models import Transaction
from django.contrib.auth import get_user_model
from django.utils import timezone

User = get_user_model()

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username']

class MessageSerializer(serializers.ModelSerializer):
    sender = UserSerializer(read_only=True)
    attachment = serializers.FileField(required=False, allow_null=True)
    reply_to = serializers.PrimaryKeyRelatedField(queryset=Message.objects.all(), required=False, allow_null=True)
    edited_at = serializers.DateTimeField(read_only=True)
    deleted_for = serializers.PrimaryKeyRelatedField(many=True, queryset=User.objects.all(), required=False)
    deleted_for_everyone = serializers.BooleanField(required=False)
    reactions = serializers.JSONField(required=False)
    pinned = serializers.BooleanField(required=False)
    starred_by = serializers.PrimaryKeyRelatedField(many=True, queryset=User.objects.all(), required=False)
    forwarded_from = serializers.PrimaryKeyRelatedField(queryset=Message.objects.all(), required=False, allow_null=True)

    class Meta:
        model = Message
        fields = [
            'id', 'chat', 'sender', 'text', 'attachment', 'created_at', 'reply_to', 'edited_at',
            'deleted_for', 'deleted_for_everyone', 'reactions', 'pinned', 'starred_by', 'forwarded_from'
        ]

    def update(self, instance, validated_data):
        user = self.context['request'].user
        # Solo el autor puede editar
        if instance.sender != user:
            raise serializers.ValidationError('Solo puedes editar tus propios mensajes.')
        instance.text = validated_data.get('text', instance.text)
        instance.edited_at = timezone.now()
        # WhatsApp-style features
        if 'reactions' in validated_data:
            instance.reactions = validated_data['reactions']
        if 'pinned' in validated_data:
            instance.pinned = validated_data['pinned']
        if 'starred_by' in validated_data:
            instance.starred_by.set(validated_data['starred_by'])
        if 'forwarded_from' in validated_data:
            instance.forwarded_from = validated_data['forwarded_from']
        instance.save()
        return instance

class ChatSerializer(serializers.ModelSerializer):
    participants = UserSerializer(many=True, read_only=True)
    participant_ids = serializers.PrimaryKeyRelatedField(
        many=True, queryset=User.objects.all(), write_only=True, source='participants'
    )
    transactions = serializers.PrimaryKeyRelatedField(
        many=True, queryset=Transaction.objects.all(), required=False
    )
    last_message = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = Chat
        fields = [
            'id', 'participants', 'participant_ids', 'transactions',
            'created_at', 'updated_at', 'last_message'
        ]

    def get_last_message(self, obj):
        last = obj.messages.order_by('-created_at').first()
        return MessageSerializer(last).data if last else None

    def create(self, validated_data):
        # Get the current user from the context
        current_user = self.context['request'].user
        participants = validated_data.pop('participants', [])
        transactions = validated_data.pop('transactions', [])

        # Asegúrate de incluir al usuario actual
        all_participants = list(participants)
        if current_user not in all_participants:
            all_participants.append(current_user)
        all_participants_ids = sorted([u.id for u in all_participants])

        # Buscar si ya existe un chat con exactamente estos participantes
        existing_chats = Chat.objects.annotate(num_participants=serializers.Count('participants')).filter(num_participants=len(all_participants))
        for chat in existing_chats:
            ids = sorted([u.id for u in chat.participants.all()])
            if ids == all_participants_ids:
                # Si existe, retorna ese chat
                return chat

        # Si no existe, crea uno nuevo
        chat = Chat.objects.create(**validated_data)
        chat.participants.set(all_participants)
        if transactions:
            chat.transactions.set(transactions)
        return chat 