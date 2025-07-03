import csv
from django.core.wsgi import get_wsgi_application
import os
import sys

# Configura el entorno Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.financialhub.settings.dev')
get_wsgi_application()

from transactions.models import Transaction

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'transactions_report.csv')

with open(output_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'description', 'type', 'category', 'subcategory', 'date', 'amount'])
    for tx in Transaction.objects.all().iterator():
        writer.writerow([
            tx.id,
            tx.description,
            tx.type,
            getattr(tx.category, 'name', '') if tx.category else '',
            getattr(tx, 'subcategory', ''),  # Solo si existe el campo
            tx.date,
            tx.amount
        ])
print(f"Reporte generado: {output_path}") 