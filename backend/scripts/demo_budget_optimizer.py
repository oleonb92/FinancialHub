#!/usr/bin/env python3
"""
Script de demostración del BudgetOptimizer.

Este script muestra cómo usar el optimizador de presupuestos para:
- Entrenar el modelo con datos históricos
- Optimizar la asignación de presupuesto
- Analizar la eficiencia presupuestaria
- Generar predicciones y recomendaciones
"""

import os
import sys
import django
from datetime import datetime, timedelta
from decimal import Decimal

# Configurar Django
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'financialhub.settings')
django.setup()

from ai.ml.optimizers.budget_optimizer import BudgetOptimizer
from transactions.models import Transaction, Category, Budget
from organizations.models import Organization
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.db import models
import json

User = get_user_model()

def create_demo_data():
    """Crea datos de demostración para el optimizador."""
    print("🔧 Creando datos de demostración...")
    
    # Crear organización de demostración
    org, created = Organization.objects.get_or_create(
        name="Demo Organization",
        defaults={'description': 'Organización para demostración del BudgetOptimizer'}
    )
    
    if created:
        print(f"✅ Organización creada: {org.name}")
    else:
        print(f"📋 Usando organización existente: {org.name}")
    
    # Crear usuario de demostración
    user, created = User.objects.get_or_create(
        username='demo_user',
        defaults={
            'email': 'demo@example.com',
            'organization': org
        }
    )
    user.set_password('demo123')
    user.save()
    
    if created:
        print(f"✅ Usuario creado: {user.username}")
    else:
        print(f"📋 Usando usuario existente: {user.username}")
    
    # Crear categorías de demostración
    categories = {
        'Alimentación': {'color': '#FF6B6B', 'icon': '🍽️'},
        'Transporte': {'color': '#4ECDC4', 'icon': '🚗'},
        'Entretenimiento': {'color': '#45B7D1', 'icon': '🎬'},
        'Servicios': {'color': '#96CEB4', 'icon': '💡'},
        'Salud': {'color': '#FFEAA7', 'icon': '🏥'}
    }
    
    created_categories = {}
    for name, attrs in categories.items():
        category, created = Category.objects.get_or_create(
            name=name,
            organization=org,
            defaults={'color': attrs['color']}
        )
        created_categories[name] = category
        if created:
            print(f"✅ Categoría creada: {attrs['icon']} {name}")
    
    # Crear transacciones de demostración (últimos 6 meses)
    base_date = timezone.now() - timedelta(days=180)
    
    # Patrones de gasto por categoría
    spending_patterns = {
        'Alimentación': {
            'base_amount': 50,
            'variation': 20,
            'frequency': 3,  # veces por semana
            'seasonal_factor': 1.1  # más gasto en verano
        },
        'Transporte': {
            'base_amount': 30,
            'variation': 15,
            'frequency': 5,  # diario
            'seasonal_factor': 0.9  # menos en verano
        },
        'Entretenimiento': {
            'base_amount': 80,
            'variation': 40,
            'frequency': 1,  # semanal
            'seasonal_factor': 1.3  # más en vacaciones
        },
        'Servicios': {
            'base_amount': 120,
            'variation': 30,
            'frequency': 1,  # mensual
            'seasonal_factor': 1.0  # constante
        },
        'Salud': {
            'base_amount': 60,
            'variation': 25,
            'frequency': 2,  # quincenal
            'seasonal_factor': 0.8  # menos en verano
        }
    }
    
    transactions_created = 0
    for day in range(180):
        current_date = base_date + timedelta(days=day)
        month = current_date.month
        
        for category_name, pattern in spending_patterns.items():
            category = created_categories[category_name]
            
            # Determinar si crear transacción basado en frecuencia
            if category_name == 'Servicios' and day % 30 == 0:  # Mensual
                should_create = True
            elif category_name == 'Entretenimiento' and day % 7 == 0:  # Semanal
                should_create = True
            elif category_name == 'Salud' and day % 14 == 0:  # Quincenal
                should_create = True
            elif category_name in ['Alimentación', 'Transporte'] and day % pattern['frequency'] == 0:
                should_create = True
            else:
                should_create = False
            
            if should_create:
                # Calcular monto con variación y factor estacional
                base_amount = pattern['base_amount']
                variation = pattern['variation']
                seasonal_factor = pattern['seasonal_factor']
                
                # Ajustar por estación
                if month in [6, 7, 8]:  # Verano
                    amount = base_amount * seasonal_factor
                elif month in [12, 1, 2]:  # Invierno
                    amount = base_amount * (2 - seasonal_factor)
                else:
                    amount = base_amount
                
                # Agregar variación aleatoria
                import random
                amount += random.uniform(-variation, variation)
                amount = max(amount, 5)  # Mínimo $5
                
                # Crear transacción
                Transaction.objects.get_or_create(
                    amount=Decimal(str(round(amount, 2))),
                    type='EXPENSE',
                    description=f'{category_name} - {current_date.strftime("%Y-%m-%d")}',
                    category=category,
                    organization=org,
                    created_by=user,
                    date=current_date,
                    ai_analyzed=True
                )
                transactions_created += 1
    
    print(f"✅ {transactions_created} transacciones de demostración creadas")
    
    # Crear presupuestos de demostración
    budgets_created = 0
    current_period = timezone.now().strftime('%Y-%m')
    
    budget_allocations = {
        'Alimentación': 1500,
        'Transporte': 800,
        'Entretenimiento': 600,
        'Servicios': 400,
        'Salud': 300
    }
    
    for category_name, amount in budget_allocations.items():
        category = created_categories[category_name]
        
        # Calcular gasto real para este período
        period_start = datetime.strptime(current_period, '%Y-%m').replace(tzinfo=timezone.utc)
        if period_start.month == 12:
            period_end = period_start.replace(year=period_start.year + 1, month=1)
        else:
            period_end = period_start.replace(month=period_start.month + 1)
        
        spent = Transaction.objects.filter(
            category=category,
            organization=org,
            date__gte=period_start,
            date__lt=period_end
        ).aggregate(total=models.Sum('amount'))['total'] or 0
        
        Budget.objects.get_or_create(
            category=category,
            organization=org,
            period=current_period,
            defaults={
                'amount': Decimal(str(amount)),
                'spent_amount': spent
            }
        )
        budgets_created += 1
    
    print(f"✅ {budgets_created} presupuestos de demostración creados")
    
    return org, user, created_categories

def demo_budget_optimizer():
    """Demuestra las capacidades del BudgetOptimizer."""
    print("\n" + "="*60)
    print("🚀 DEMOSTRACIÓN DEL BUDGET OPTIMIZER")
    print("="*60)
    
    # Crear datos de demostración
    org, user, categories = create_demo_data()
    
    # Inicializar optimizador
    print("\n🔧 Inicializando BudgetOptimizer...")
    optimizer = BudgetOptimizer()
    
    # Obtener transacciones para entrenamiento
    transactions = Transaction.objects.filter(
        organization=org,
        type='EXPENSE',
        date__gte=timezone.now() - timedelta(days=180)
    ).select_related('category')
    
    print(f"📊 Transacciones disponibles para entrenamiento: {transactions.count()}")
    
    # Preparar datos de entrenamiento
    transaction_data = []
    for t in transactions:
        transaction_data.append({
            'amount': float(t.amount),
            'date': t.date,
            'category_id': t.category.id if t.category else 0
        })
    
    # Entrenar modelo
    print("\n🎯 Entrenando modelo...")
    try:
        optimizer.train(transaction_data)
        print("✅ Modelo entrenado exitosamente")
    except Exception as e:
        print(f"❌ Error entrenando modelo: {str(e)}")
        return
    
    # 1. Optimización de asignación de presupuesto
    print("\n" + "-"*40)
    print("💰 OPTIMIZACIÓN DE ASIGNACIÓN DE PRESUPUESTO")
    print("-"*40)
    
    total_budget = 5000.0
    optimization_result = optimizer.optimize_budget_allocation(org.id, total_budget)
    
    if 'suggested_allocation' in optimization_result:
        print(f"\n📋 Presupuesto total: ${total_budget:,.2f}")
        print("\n🎯 Asignación optimizada por categoría:")
        
        for category_id, amount in optimization_result['suggested_allocation'].items():
            category = Category.objects.get(id=category_id)
            percentage = (amount / total_budget) * 100
            print(f"   • {category.name}: ${amount:,.2f} ({percentage:.1f}%)")
        
        if 'recommendations' in optimization_result:
            print(f"\n💡 Recomendaciones ({len(optimization_result['recommendations'])}):")
            for i, rec in enumerate(optimization_result['recommendations'][:3], 1):
                category = Category.objects.get(id=rec['category_id'])
                print(f"   {i}. {rec['type'].replace('_', ' ').title()} en {category.name}: {rec['reason']}")
    else:
        print(f"❌ Error en optimización: {optimization_result.get('error', 'Unknown error')}")
    
    # 2. Análisis de eficiencia presupuestaria
    print("\n" + "-"*40)
    print("📊 ANÁLISIS DE EFICIENCIA PRESUPUESTARIA")
    print("-"*40)
    
    efficiency_result = optimizer.analyze_budget_efficiency(org.id)
    
    if 'overall_efficiency' in efficiency_result:
        overall_eff = efficiency_result['overall_efficiency']
        print(f"\n📈 Eficiencia general: {overall_eff:.1%}")
        
        if 'category_efficiencies' in efficiency_result:
            print("\n📋 Eficiencia por categoría:")
            for category_name, metrics in efficiency_result['category_efficiencies'].items():
                efficiency = metrics['efficiency_score']
                status = metrics['status']
                allocated = metrics['allocated']
                spent = metrics['spent']
                remaining = metrics['remaining']
                
                status_icon = {
                    'excellent': '🟢',
                    'good': '🟡',
                    'fair': '🟠',
                    'poor': '🔴'
                }.get(status, '⚪')
                
                print(f"   {status_icon} {category_name}:")
                print(f"      Asignado: ${allocated:,.2f} | Gastado: ${spent:,.2f} | Restante: ${remaining:,.2f}")
                print(f"      Eficiencia: {efficiency:.1%} ({status})")
        
        if 'recommendations' in efficiency_result:
            print(f"\n💡 Recomendaciones de eficiencia ({len(efficiency_result['recommendations'])}):")
            for i, rec in enumerate(efficiency_result['recommendations'][:3], 1):
                print(f"   {i}. {rec['message']}")
    else:
        print(f"❌ Error en análisis de eficiencia: {efficiency_result.get('error', 'Unknown error')}")
    
    # 3. Predicción de necesidades presupuestarias
    print("\n" + "-"*40)
    print("🔮 PREDICCIÓN DE NECESIDADES PRESUPUESTARIAS")
    print("-"*40)
    
    prediction_result = optimizer.predict_budget_needs(org.id)
    
    if 'total_predicted' in prediction_result:
        total_predicted = prediction_result['total_predicted']
        confidence = prediction_result['confidence']
        
        print(f"\n📊 Predicción para el próximo período:")
        print(f"   Total predicho: ${total_predicted:,.2f}")
        print(f"   Confianza: {confidence:.1%}")
        
        if 'category_predictions' in prediction_result:
            print("\n📋 Predicciones por categoría:")
            for category_id, pred in prediction_result['category_predictions'].items():
                category = Category.objects.get(id=category_id)
                predicted_amount = pred['predicted_amount']
                pred_confidence = pred['confidence']
                
                print(f"   • {category.name}: ${predicted_amount:,.2f} (confianza: {pred_confidence:.1%})")
    else:
        print(f"❌ Error en predicción: {prediction_result.get('error', 'Unknown error')}")
    
    # 4. Insights presupuestarios
    print("\n" + "-"*40)
    print("🧠 INSIGHTS PRESUPUESTARIOS")
    print("-"*40)
    
    insights_result = optimizer.get_budget_insights(org.id)
    
    if 'insights' in insights_result:
        insights = insights_result['insights']
        print(f"\n💡 Insights generados ({len(insights)}):")
        
        for i, insight in enumerate(insights[:5], 1):
            icon = {
                'warning': '⚠️',
                'info': 'ℹ️',
                'success': '✅',
                'prediction': '🔮'
            }.get(insight['type'], '📌')
            
            priority_icon = {
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢'
            }.get(insight['priority'], '⚪')
            
            print(f"   {i}. {icon} {priority_icon} {insight['title']}")
            print(f"      {insight['message']}")
    else:
        print(f"❌ Error generando insights: {insights_result.get('error', 'Unknown error')}")
    
    # 5. Información del modelo
    print("\n" + "-"*40)
    print("🔧 INFORMACIÓN DEL MODELO")
    print("-"*40)
    
    model_info = optimizer.get_model_info()
    print(f"\n📋 Información del modelo:")
    print(f"   Nombre: {model_info['model_name']}")
    print(f"   Entrenado: {'Sí' if model_info['is_trained'] else 'No'}")
    print(f"   Features: {len(model_info['feature_names'])}")
    print(f"   Predictor de gastos: {model_info['expense_predictor_type']}")
    print(f"   Analizador de eficiencia: {model_info['efficiency_analyzer_type']}")
    
    print("\n" + "="*60)
    print("✅ DEMOSTRACIÓN COMPLETADA")
    print("="*60)
    print("\n🎯 El BudgetOptimizer está listo para usar en tu aplicación!")
    print("📚 Consulta la documentación para más detalles sobre la API.")

if __name__ == '__main__':
    try:
        demo_budget_optimizer()
    except KeyboardInterrupt:
        print("\n\n⏹️ Demostración interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante la demostración: {str(e)}")
        import traceback
        traceback.print_exc() 