from django.apps import AppConfig

class GoalsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'goals'
    verbose_name = 'Financial Goals'

    def ready(self):
        try:
            import goals.signals  # noqa
        except ImportError:
            pass 