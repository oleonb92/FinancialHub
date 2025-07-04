# Generated by Django 5.1.9 on 2025-06-01 14:53

import django.contrib.postgres.fields
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0002_user_account_type_user_pro_features"),
    ]

    operations = [
        migrations.AddField(
            model_name="user",
            name="pro_features_list",
            field=django.contrib.postgres.fields.ArrayField(
                base_field=models.CharField(max_length=50),
                blank=True,
                default=list,
                help_text="Lista de features Pro activas para este usuario.",
                size=None,
            ),
        ),
        migrations.AddField(
            model_name="user",
            name="pro_trial_until",
            field=models.DateTimeField(
                blank=True,
                help_text="Fecha hasta la que el usuario tiene trial Pro activo.",
                null=True,
            ),
        ),
    ]
