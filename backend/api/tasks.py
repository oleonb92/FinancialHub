from celery import shared_task

@shared_task
def add(x, y):
    print(f"Sumando {x} + {y}")
    return x + y 