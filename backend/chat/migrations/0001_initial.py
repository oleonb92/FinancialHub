# Generated by Django 5.1.9 on 2025-06-17 21:06

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ("organizations", "0001_initial"),
        ("transactions", "0001_initial"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="Chat",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "participants",
                    models.ManyToManyField(
                        related_name="chats", to=settings.AUTH_USER_MODEL
                    ),
                ),
                (
                    "transactions",
                    models.ManyToManyField(
                        blank=True, related_name="chats", to="transactions.transaction"
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Message",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("text", models.TextField(blank=True)),
                (
                    "attachment",
                    models.FileField(
                        blank=True, null=True, upload_to="chat_attachments/"
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("edited_at", models.DateTimeField(blank=True, null=True)),
                ("deleted_for_everyone", models.BooleanField(default=False)),
                ("reactions", models.JSONField(blank=True, default=dict)),
                ("pinned", models.BooleanField(default=False)),
                (
                    "chat",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="messages",
                        to="chat.chat",
                    ),
                ),
                (
                    "deleted_for",
                    models.ManyToManyField(
                        blank=True,
                        related_name="deleted_messages",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    "forwarded_from",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="forwards",
                        to="chat.message",
                    ),
                ),
                (
                    "organization",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="organizations.organization",
                    ),
                ),
                (
                    "reply_to",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="replies",
                        to="chat.message",
                    ),
                ),
                (
                    "sender",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    "starred_by",
                    models.ManyToManyField(
                        blank=True,
                        related_name="starred_messages",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
    ]
