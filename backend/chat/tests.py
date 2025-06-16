from django.test import TestCase
from django.contrib.auth import get_user_model
from .models import Chat, Message

User = get_user_model()

class ChatModelTest(TestCase):
    def setUp(self):
        self.user1 = User.objects.create_user(username='user1', password='pass')
        self.user2 = User.objects.create_user(username='user2', password='pass')

    def test_create_chat_and_message(self):
        chat = Chat.objects.create()
        chat.participants.set([self.user1, self.user2])
        msg = Message.objects.create(chat=chat, sender=self.user1, text='Hello!')
        self.assertEqual(chat.participants.count(), 2)
        self.assertEqual(chat.messages.count(), 1)
        self.assertEqual(msg.text, 'Hello!')
