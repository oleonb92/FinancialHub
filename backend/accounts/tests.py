from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status
from django.contrib.auth import get_user_model

User = get_user_model()

class RegistrationTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def test_admin_registration(self):
        response = self.client.post(reverse('register'), {
            "username": "adminuser",
            "password": "testpass123",
            "role": "admin",
            "household": "AlphaFamily"
        })
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(User.objects.filter(username="adminuser", is_active=True).exists())

    def test_member_registration(self):
        response = self.client.post(reverse('register'), {
            "username": "memberuser",
            "password": "testpass123",
            "role": "member",
            "household": "AlphaFamily"
        })
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(User.objects.filter(username="memberuser", is_active=False).exists())

    def test_member_invalid_household(self):
        response = self.client.post(reverse('register'), {
            "username": "ghost",
            "password": "testpass123",
            "role": "member",
            "household": "UnknownFam"
        })
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_admin_duplicate_household(self):
        response = self.client.post(reverse('register'), {
            "username": "adminx",
            "password": "testpass123",
            "role": "admin",
            "household": "MyFamily"
        })
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)