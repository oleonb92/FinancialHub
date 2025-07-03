#!/usr/bin/env python3
"""
Script para asignar automáticamente categorías a transacciones de TODAS las organizaciones
basándose en la descripción y tipo de transacción.
"""

import os
import sys
import json
import re
from collections import defaultdict

# Configurar Django
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.financialhub.settings.dev')

import django
django.setup()

from transactions.models import Transaction, Category
from organizations.models import Organization

def load_categories():
    """Carga las categorías desde categories_en.json"""
    categories_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                  'backend', 'categories_en.json')
    with open(categories_path, 'r') as f:
        return json.load(f)

def create_keyword_mapping():
    """Crea el mapeo de palabras clave a categorías principales y subcategorías"""
    return {
        # Income
        'salary': ('Income', 'Salary'),
        'payroll': ('Income', 'Salary'),
        'wage': ('Income', 'Salary'),
        'freelance': ('Income', 'Freelance Income'),
        'consulting': ('Income', 'Freelance Income'),
        'side hustle': ('Income', 'Side Hustles'),
        'sales': ('Income', 'Sales Revenue'),
        'revenue': ('Income', 'Sales Revenue'),
        'rental': ('Income', 'Rental Income'),
        'interest': ('Income', 'Interest Earned'),
        'dividend': ('Income', 'Dividend Income'),
        'refund': ('Income', 'Refunds'),
        'gift': ('Income', 'Cash Gifts'),
        'grant': ('Income', 'Grants'),
        
        # Business Expenses
        'advertising': ('Business Expenses', 'Advertising & Marketing'),
        'marketing': ('Business Expenses', 'Advertising & Marketing'),
        'promotion': ('Business Expenses', 'Advertising & Marketing'),
        'bank fee': ('Business Expenses', 'Bank & Payment Fees'),
        'payment fee': ('Business Expenses', 'Bank & Payment Fees'),
        'transaction fee': ('Business Expenses', 'Bank & Payment Fees'),
        'car expense': ('Business Expenses', 'Car & Vehicle Expenses'),
        'vehicle': ('Business Expenses', 'Car & Vehicle Expenses'),
        'client meal': ('Business Expenses', 'Client Meals'),
        'business lunch': ('Business Expenses', 'Client Meals'),
        'contractor': ('Business Expenses', 'Contractor Payments'),
        'education': ('Business Expenses', 'Education & Courses'),
        'course': ('Business Expenses', 'Education & Courses'),
        'training': ('Business Expenses', 'Education & Courses'),
        'employee': ('Business Expenses', 'Employee Wages'),
        'wage': ('Business Expenses', 'Employee Wages'),
        'freelancer': ('Business Expenses', 'Freelancer Payments'),
        'home office': ('Business Expenses', 'Home Office Expenses'),
        'office': ('Business Expenses', 'Home Office Expenses'),
        'insurance': ('Business Expenses', 'Insurance'),
        'legal': ('Business Expenses', 'Legal & Professional Fees'),
        'professional': ('Business Expenses', 'Legal & Professional Fees'),
        'license': ('Business Expenses', 'Licensing & Permits'),
        'permit': ('Business Expenses', 'Licensing & Permits'),
        'office supply': ('Business Expenses', 'Office Supplies'),
        'supply': ('Business Expenses', 'Office Supplies'),
        'phone': ('Business Expenses', 'Phone & Internet'),
        'internet': ('Business Expenses', 'Phone & Internet'),
        'software': ('Business Expenses', 'Software & Subscriptions'),
        'subscription': ('Business Expenses', 'Software & Subscriptions'),
        'tax': ('Business Expenses', 'Taxes & Licenses'),
        'travel': ('Business Expenses', 'Travel & Transportation'),
        'transportation': ('Business Expenses', 'Travel & Transportation'),
        'utility': ('Business Expenses', 'Utilities'),
        'website': ('Business Expenses', 'Website & Hosting'),
        'hosting': ('Business Expenses', 'Website & Hosting'),
        
        # Personal Expenses
        'grocery': ('Personal Expenses', 'Groceries'),
        'supermarket': ('Personal Expenses', 'Groceries'),
        'food store': ('Personal Expenses', 'Groceries'),
        'restaurant': ('Personal Expenses', 'Dining Out'),
        'dining': ('Personal Expenses', 'Dining Out'),
        'cafe': ('Personal Expenses', 'Dining Out'),
        'coffee': ('Personal Expenses', 'Dining Out'),
        'takeout': ('Personal Expenses', 'Dining Out'),
        'rent': ('Personal Expenses', 'Rent or Mortgage'),
        'mortgage': ('Personal Expenses', 'Rent or Mortgage'),
        'utility': ('Personal Expenses', 'Utilities'),
        'electric': ('Personal Expenses', 'Utilities'),
        'water': ('Personal Expenses', 'Utilities'),
        'gas': ('Personal Expenses', 'Utilities'),
        'phone': ('Personal Expenses', 'Phone & Internet'),
        'internet': ('Personal Expenses', 'Phone & Internet'),
        'health': ('Personal Expenses', 'Health & Dental'),
        'dental': ('Personal Expenses', 'Health & Dental'),
        'medical': ('Personal Expenses', 'Health & Dental'),
        'car payment': ('Personal Expenses', 'Car Payment'),
        'auto loan': ('Personal Expenses', 'Car Payment'),
        'fuel': ('Personal Expenses', 'Gas & Fuel'),
        'gasoline': ('Personal Expenses', 'Gas & Fuel'),
        'insurance': ('Personal Expenses', 'Insurance (Home, Auto)'),
        'clothing': ('Personal Expenses', 'Clothing & Accessories'),
        'accessory': ('Personal Expenses', 'Clothing & Accessories'),
        'gift': ('Personal Expenses', 'Gifts & Donations'),
        'donation': ('Personal Expenses', 'Gifts & Donations'),
        'subscription': ('Personal Expenses', 'Subscriptions'),
        'childcare': ('Personal Expenses', 'Childcare & School'),
        'school': ('Personal Expenses', 'Childcare & School'),
        'tuition': ('Personal Expenses', 'Childcare & School'),
        'personal care': ('Personal Expenses', 'Personal Care'),
        'gym': ('Personal Expenses', 'Gym & Fitness'),
        'fitness': ('Personal Expenses', 'Gym & Fitness'),
        'entertainment': ('Personal Expenses', 'Entertainment'),
        'movie': ('Personal Expenses', 'Entertainment'),
        'vacation': ('Personal Expenses', 'Vacations & Travel'),
        'travel': ('Personal Expenses', 'Vacations & Travel'),
        'pet': ('Personal Expenses', 'Pet Expenses'),
        'veterinary': ('Personal Expenses', 'Pet Expenses'),
        
        # Assets
        'bank account': ('Assets', 'Bank Accounts'),
        'checking': ('Assets', 'Bank Accounts'),
        'savings': ('Assets', 'Bank Accounts'),
        'cash': ('Assets', 'Cash on Hand'),
        'investment': ('Assets', 'Investments'),
        'stock': ('Assets', 'Investments'),
        'property': ('Assets', 'Property'),
        'real estate': ('Assets', 'Property'),
        'vehicle': ('Assets', 'Vehicle'),
        'car': ('Assets', 'Vehicle'),
        'equipment': ('Assets', 'Business Equipment'),
        'business equipment': ('Assets', 'Business Equipment'),
        
        # Liabilities
        'credit card': ('Liabilities', 'Credit Cards'),
        'student loan': ('Liabilities', 'Student Loans'),
        'personal loan': ('Liabilities', 'Personal Loans'),
        'business loan': ('Liabilities', 'Business Loans'),
        'mortgage': ('Liabilities', 'Mortgage'),
        'car loan': ('Liabilities', 'Car Loan'),
        'auto loan': ('Liabilities', 'Car Loan'),
        'tax owed': ('Liabilities', 'Taxes Owed'),
        
        # Savings & Goals
        'emergency fund': ('Savings & Goals', 'Emergency Fund'),
        'vacation fund': ('Savings & Goals', 'Vacation Fund'),
        'down payment': ('Savings & Goals', 'Down Payment'),
        'retirement': ('Savings & Goals', 'Retirement'),
        '401k': ('Savings & Goals', 'Retirement'),
        'ira': ('Savings & Goals', 'Retirement'),
        'education savings': ('Savings & Goals', 'Education Savings'),
        'investment savings': ('Savings & Goals', 'Investment Savings'),
        
        # Taxes
        'federal tax': ('Taxes', 'Federal Taxes'),
        'state tax': ('Taxes', 'State Taxes'),
        'sales tax': ('Taxes', 'Sales Tax Paid'),
        'self employment': ('Taxes', 'Self-Employment Tax'),
        'estimated tax': ('Taxes', 'Estimated Taxes'),
    }

def get_default_category_by_type(transaction_type):
    """Retorna la categoría por defecto según el tipo de transacción"""
    defaults = {
        'EXPENSE': ('Personal Expenses', 'Other Personal Expenses'),
        'INCOME': ('Income', 'Other Income'),
        'TRANSFER': ('Assets', 'Bank Accounts'),
    }
    return defaults.get(transaction_type, ('Personal Expenses', 'Other Personal Expenses'))

def find_best_category_match(description, transaction_type):
    """Encuentra la mejor coincidencia de categoría basada en la descripción"""
    description_lower = description.lower()
    keyword_mapping = create_keyword_mapping()
    
    # Buscar coincidencias exactas primero
    for keyword, (category, subcategory) in keyword_mapping.items():
        if keyword in description_lower:
            # Verificar que la categoría sea apropiada para el tipo de transacción
            if transaction_type == 'EXPENSE' and category in ['Income', 'Assets', 'Liabilities', 'Savings & Goals']:
                continue
            if transaction_type == 'INCOME' and category in ['Business Expenses', 'Personal Expenses', 'Assets', 'Liabilities', 'Savings & Goals']:
                continue
            if transaction_type == 'TRANSFER' and category not in ['Assets', 'Liabilities']:
                continue
            return category, subcategory
    
    # Si no hay coincidencia, usar categoría por defecto según el tipo
    return get_default_category_by_type(transaction_type)

def get_or_create_category_hierarchy(category_name, subcategory_name, organization):
    """Obtiene o crea la jerarquía de categorías (categoría padre y subcategoría)"""
    # Obtener o crear categoría padre
    parent_category, created = Category.objects.get_or_create(
        name=category_name,
        organization=organization,
        parent=None  # Categoría padre no tiene parent
    )
    if created:
        print(f"  ✓ Categoría padre creada: {category_name}")
    
    # Obtener o crear subcategoría (hija)
    subcategory, created = Category.objects.get_or_create(
        name=subcategory_name,
        organization=organization,
        parent=parent_category
    )
    if created:
        print(f"  ✓ Subcategoría creada: {subcategory_name}")
    
    return subcategory  # Retornamos la subcategoría para asignar a la transacción

def process_organization(organization, stats, total_updated, total_count):
    """Procesa todas las transacciones de una organización específica"""
    print(f"\n🏢 Procesando organización: {organization.name}")
    print("-" * 50)
    
    # Obtener todas las transacciones de la organización
    transactions = Transaction.objects.filter(organization=organization).select_related('category')
    org_transaction_count = transactions.count()
    
    print(f"📊 Transacciones en {organization.name}: {org_transaction_count}")
    
    org_updated = 0
    org_count = 0
    
    for i, transaction in enumerate(transactions, 1):
        if i % 500 == 0:
            print(f"  Procesando transacción {i}/{org_transaction_count}...")
        
        org_count += 1
        total_count += 1
        
        # Encontrar la mejor categoría
        best_category, best_subcategory = find_best_category_match(
            transaction.description, 
            transaction.type
        )
        
        # Obtener o crear categoría y subcategoría
        category = get_or_create_category_hierarchy(
            best_category, 
            best_subcategory,
            organization
        )
        
        # Verificar si necesita actualización
        current_category = transaction.category.name if transaction.category else None
        
        if current_category != best_subcategory:  # Comparamos con la subcategoría
            # Actualizar transacción
            transaction.category = category
            transaction.save()
            
            org_updated += 1
            total_updated += 1
            stats[transaction.type][best_category] += 1
            
            if org_updated <= 5:  # Mostrar solo las primeras 5 actualizaciones por organización
                print(f"  🔄 Actualizada: '{transaction.description[:50]}...'")
                print(f"     Tipo: {transaction.type} | Categoría: {best_category} | Subcategoría: {best_subcategory}")
    
    print(f"✅ {organization.name}: {org_updated}/{org_count} transacciones actualizadas")
    return org_updated, org_count

def main():
    """Función principal del script"""
    print("🚀 Iniciando asignación automática de categorías para TODAS las organizaciones...")
    print("=" * 70)
    
    # Obtener todas las organizaciones
    organizations = Organization.objects.all()
    total_organizations = organizations.count()
    
    print(f"📊 Total de organizaciones encontradas: {total_organizations}")
    for org in organizations:
        count = Transaction.objects.filter(organization=org).count()
        print(f"  - {org.name}: {count} transacciones")
    
    # Estadísticas globales
    stats = defaultdict(lambda: defaultdict(int))
    total_updated = 0
    total_count = 0
    
    # Procesar cada organización
    for i, organization in enumerate(organizations, 1):
        print(f"\n🔄 Procesando organización {i}/{total_organizations}")
        org_updated, org_count = process_organization(organization, stats, total_updated, total_count)
        total_updated += org_updated
        total_count += org_count
    
    # Mostrar resumen final
    print("\n" + "=" * 70)
    print("📈 RESUMEN FINAL DE ASIGNACIONES")
    print("=" * 70)
    print(f"Total de organizaciones procesadas: {total_organizations}")
    print(f"Total de transacciones procesadas: {total_count}")
    print(f"Total de transacciones actualizadas: {total_updated}")
    print(f"Total de transacciones sin cambios: {total_count - total_updated}")
    print()
    
    print("📊 Desglose por tipo de transacción:")
    for transaction_type, categories in stats.items():
        print(f"\n{transaction_type}:")
        for category, count in categories.items():
            print(f"  - {category}: {count} transacciones")
    
    print("\n✅ ¡Asignación automática completada para todas las organizaciones!")
    print("\n💡 Próximos pasos:")
    print("1. Verifica las categorías asignadas en el admin de Django")
    print("2. Ajusta manualmente las categorías incorrectas si es necesario")
    print("3. Prueba el chatbot financiero - ahora debería mostrar tus gastos correctamente")

if __name__ == '__main__':
    main() 