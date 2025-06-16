from django.apps import AppConfig

class AIConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ai'
    verbose_name = 'AI Services'

    def ready(self):
        try:
            import ai.signals  # noqa
        except ImportError:
            pass 