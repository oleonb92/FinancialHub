from django.db.models.signals import post_save, pre_save
from django.dispatch import receiver
from django.utils import timezone
from .models import FinancialGoal, GoalMilestone, GoalContribution
from notifications.services import NotificationService

@receiver(post_save, sender=FinancialGoal)
def notify_goal_status_change(sender, instance, created, **kwargs):
    """
    Notificar cuando el estado de un objetivo cambia
    """
    if not created and instance.status == 'completed':
        notification_service = NotificationService()
        notification_service.create_notification(
            user=instance.user,
            type='goal',
            title='¡Objetivo Completado!',
            message=f'¡Felicidades! Has completado tu objetivo: {instance.name}',
            priority='high'
        )

@receiver(post_save, sender=GoalMilestone)
def notify_milestone_completion(sender, instance, created, **kwargs):
    """
    Notificar cuando se completa un hito
    """
    if not created and instance.is_completed:
        notification_service = NotificationService()
        notification_service.create_notification(
            user=instance.goal.user,
            type='goal',
            title='Hito Completado',
            message=f'Has completado el hito "{instance.title}" de tu objetivo {instance.goal.name}',
            priority='medium'
        )

@receiver(pre_save, sender=FinancialGoal)
def check_goal_completion(sender, instance, **kwargs):
    """
    Verificar si el objetivo se ha completado
    """
    if instance.current_amount >= instance.target_amount:
        instance.status = 'completed'
        instance.completed_at = timezone.now() 