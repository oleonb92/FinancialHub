#!/usr/bin/env python
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'financialhub.settings.dev')
django.setup()

from transactions.models import Transaction
from django.db.models import Sum, Q
from datetime import datetime

def debug_financial_data():
    org_id = 1
    
    print("=== DEBUG FINANCIAL DATA ===")
    
    # Totales generales
    total_income = sum(t.amount for t in Transaction.objects.filter(organization_id=org_id, type='INCOME'))
    total_expenses = sum(t.amount for t in Transaction.objects.filter(organization_id=org_id, type='EXPENSE'))
    total_transactions = Transaction.objects.filter(organization_id=org_id).count()
    
    print(f"Total Income: ${total_income:,.2f}")
    print(f"Total Expenses: ${total_expenses:,.2f}")
    print(f"Net Balance: ${total_income - total_expenses:,.2f}")
    print(f"Total Transactions: {total_transactions}")
    
    # Datos por mes - versión simplificada
    print("\n=== MONTHLY BREAKDOWN ===")
    
    # Obtener todos los meses únicos
    transactions = Transaction.objects.filter(organization_id=org_id).order_by('date')
    months_data = {}
    
    for t in transactions:
        year = t.date.year
        month = t.date.month
        key = f"{year}-{month:02d}"
        
        if key not in months_data:
            months_data[key] = {'year': year, 'month': month, 'income': 0, 'expenses': 0}
        
        if t.type == 'INCOME':
            months_data[key]['income'] += float(t.amount)
        elif t.type == 'EXPENSE':
            months_data[key]['expenses'] += float(t.amount)
    
    # Mostrar datos por mes
    for key in sorted(months_data.keys()):
        data = months_data[key]
        income = data['income']
        expenses = data['expenses']
        balance = income - expenses
        print(f"{key}: Income=${income:,.2f}, Expenses=${expenses:,.2f}, Balance=${balance:,.2f}")
    
    # Calcular promedios mensuales correctos
    months_count = len(months_data)
    if months_count > 0:
        avg_monthly_income = total_income / months_count
        avg_monthly_expenses = total_expenses / months_count
        print(f"\n=== AVERAGES ===")
        print(f"Months with data: {months_count}")
        print(f"Average monthly income: ${avg_monthly_income:,.2f}")
        print(f"Average monthly expenses: ${avg_monthly_expenses:,.2f}")
        print(f"Average monthly balance: ${avg_monthly_income - avg_monthly_expenses:,.2f}")

if __name__ == "__main__":
    debug_financial_data() 