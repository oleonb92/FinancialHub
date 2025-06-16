import os
import django
import random
from datetime import datetime, timedelta
from decimal import Decimal

# Configurar Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'financialhub.settings')
django.setup()

# Importar modelos
from accounts.models import User
from households.models import Household
from chartofaccounts.models import Account
from transactions.models import Category, Tag, Transaction
from django.utils import timezone
from organizations.models import Organization

def create_sample_data():
    # Obtener usuario y household/organización
    user = User.objects.first()
    try:
        organization = Organization.objects.first()
    except Exception:
        organization = None
    if not user or not organization:
        print("❌ Error: No se encontró usuario u organización")
        return

    print(f"👤 Usuario: {user.username}")
    print(f"🏢 Organización: {organization.name}")

    # Crear cuentas si no existen
    accounts = []
    account_names = ['Checking', 'Savings', 'Credit Card']
    for name in account_names:
        account, created = Account.objects.get_or_create(
            name=name,
            household=organization,
            type='asset'
        )
        accounts.append(account)
        print(f"✅ Cuenta {'creada' if created else 'existente'}: {name}")

    # Usar categorías existentes
    categories = list(Category.objects.filter(organization=organization))
    if not categories:
        print("❌ No hay categorías en la organización. Crea algunas primero.")
        return

    # Crear tags si no existen
    tags = []
    tag_names = ['urgent', 'monthly', 'food', 'fun', 'ai-analyzed']
    for name in tag_names:
        tag, created = Tag.objects.get_or_create(name=name)
        tags.append(tag)
        print(f"✅ Tag {'creado' if created else 'existente'}: {name}")

    transaction_types = ['EXPENSE', 'INCOME', 'TRANSFER', 'INVESTMENT', 'LOAN', 'REFUND']
    merchants = ['Amazon', 'Walmart', 'Netflix', 'Apple', 'Uber', 'Starbucks', 'Target', 'Shell', 'BestBuy', 'McDonalds']
    payment_methods = ['Credit Card', 'Debit Card', 'Cash', 'Bank Transfer', 'Paypal']
    locations = ['New York', 'Los Angeles', 'Chicago', 'Miami', 'San Francisco']

    year = 2025
    total_transactions = 120
    transactions_per_month = total_transactions // 12

    for month in range(1, 13):
        for i in range(transactions_per_month):
            day = random.randint(1, 28)  # Para evitar problemas con febrero
            date = datetime(year, month, day).date()
            category = random.choice(categories)
            tx_type = random.choices(
                population=['EXPENSE', 'INCOME', 'TRANSFER'],
                weights=[0.7, 0.2, 0.1],
                k=1
            )[0]
            amount = Decimal(str(round(random.uniform(10, 800), 2)))
            merchant = random.choice(merchants)
            payment_method = random.choice(payment_methods)
            location = random.choice(locations)
            status = random.choice(['pending', 'confirmed', 'reconciled'])

            transaction = Transaction.objects.create(
                type=tx_type,
                amount=amount,
                date=date,
                description=f'Transacción de ejemplo {month}-{i+1}',
                category=category,
                source_account=random.choice(accounts),
                destination_account=random.choice(accounts),
                organization=organization,
                created_by=user,
                status=status,
                is_imported=bool(i % 2),
                bank_transaction_id=f'BANKTXN{month}{i+1}',
                notes=f'Nota de ejemplo {month}-{i+1}',
                location=location,
                merchant=merchant,
                payment_method=payment_method,
                recurring=bool(i % 3 == 0),
                recurring_frequency='monthly' if i % 3 == 0 else '',
                recurring_end_date=date + timedelta(days=30) if i % 3 == 0 else None,
                ai_analyzed=bool(i % 2),
                ai_confidence=round(random.uniform(0.5, 1.0), 2),
                ai_notes=f'AI nota {month}-{i+1}'
            )
            # Asignar tags aleatorios
            num_tags = random.randint(1, len(tags))
            transaction.tags.set(random.sample(tags, num_tags))
            print(f"✅ Transacción creada: {transaction.description} | {tx_type} | {category.name} | {date} | ${amount}")

    print("\n✨ ¡120 transacciones de ejemplo creadas exitosamente para todo el año 2025!")

if __name__ == '__main__':
    create_sample_data() 