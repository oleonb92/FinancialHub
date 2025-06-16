from django.db import models
from django.contrib.auth import get_user_model
from django.conf import settings
from organizations.models import Organization

User = get_user_model()

class Chat(models.Model):
    participants = models.ManyToManyField(User, related_name='chats')
    transactions = models.ManyToManyField('transactions.Transaction', blank=True, related_name='chats')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Chat ({self.id}) - {', '.join([u.username for u in self.participants.all()])}"

class Message(models.Model):
    chat = models.ForeignKey(Chat, on_delete=models.CASCADE, related_name='messages')
    sender = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    text = models.TextField(blank=True)
    attachment = models.FileField(upload_to='chat_attachments/', blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    reply_to = models.ForeignKey('self', null=True, blank=True, on_delete=models.SET_NULL, related_name='replies')
    edited_at = models.DateTimeField(null=True, blank=True)
    deleted_for = models.ManyToManyField(User, related_name='deleted_messages', blank=True)
    deleted_for_everyone = models.BooleanField(default=False)
    # WhatsApp-style features
    reactions = models.JSONField(default=dict, blank=True)  # {emoji: [user_id, ...]}
    pinned = models.BooleanField(default=False)
    starred_by = models.ManyToManyField(User, related_name='starred_messages', blank=True)
    forwarded_from = models.ForeignKey('self', null=True, blank=True, on_delete=models.SET_NULL, related_name='forwards')
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.sender.username}: {self.text[:50]}"

    class Meta:
        ordering = ['-created_at']
