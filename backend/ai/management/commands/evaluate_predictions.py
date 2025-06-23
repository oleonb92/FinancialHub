from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
from ai.models import AIPrediction
from transactions.models import Transaction
import json

class Command(BaseCommand):
    help = 'Evalúa la precisión de las predicciones de IA que ya han concluido.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Forzar la evaluación de TODAS las predicciones no evaluadas.',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('🚀 Iniciando la evaluación de predicciones de IA...'))
        
        # Buscar predicciones que ya deberían haber ocurrido y no han sido evaluadas
        now = timezone.now().date()
        
        if options['force']:
            self.stdout.write(self.style.WARNING('⚡ Forzando la evaluación de TODAS las predicciones no evaluadas.'))
            predictions_to_evaluate = AIPrediction.objects.filter(accuracy_score__isnull=True)
        else:
            predictions_to_evaluate = AIPrediction.objects.filter(
                prediction_date__lt=now,
                accuracy_score__isnull=True
            )
        
        if not predictions_to_evaluate.exists():
            self.stdout.write(self.style.WARNING('No se encontraron predicciones para evaluar.'))
            return

        evaluated_count = 0
        for prediction in predictions_to_evaluate:
            self.stdout.write(f"  - Evaluando predicción #{prediction.id} para {prediction.organization.name}...")
            
            try:
                period_days = prediction.prediction.get('days_predicted', 30)
                start_date = prediction.prediction_date - timedelta(days=period_days)
                end_date = prediction.prediction_date

                if prediction.type == 'spending':
                    # --- Lógica para predicciones de GASTOS ---
                    actual_transactions = Transaction.objects.filter(
                        organization=prediction.organization,
                        date__gte=start_date,
                        date__lt=end_date,
                        type='EXPENSE'
                    )
                    actual_total = sum(t.amount for t in actual_transactions)
                    predicted_total = prediction.prediction.get('total_predicted', 0.0)
                    
                    accuracy = 0.0
                    if actual_total > 0:
                        error = abs(float(actual_total) - predicted_total) / float(actual_total)
                        accuracy = 1 - error

                    prediction.actual_result = {
                        'total_actual_spending': float(actual_total),
                        'start_date': str(start_date),
                        'end_date': str(end_date),
                        'transaction_count': actual_transactions.count()
                    }

                elif prediction.type == 'cash_flow':
                    # --- Lógica para predicciones de FLUJO DE CAJA ---
                    incomes = Transaction.objects.filter(
                        organization=prediction.organization,
                        date__gte=start_date,
                        date__lt=end_date,
                        type='INCOME'
                    )
                    expenses = Transaction.objects.filter(
                        organization=prediction.organization,
                        date__gte=start_date,
                        date__lt=end_date,
                        type='EXPENSE'
                    )
                    total_income = sum(t.amount for t in incomes)
                    total_expense = sum(t.amount for t in expenses)
                    
                    actual_cash_flow = total_income - total_expense
                    predicted_cash_flow = prediction.prediction.get('total_predicted', 0.0)

                    accuracy = 0.0
                    if abs(actual_cash_flow) > 0.01:
                        error = abs(float(actual_cash_flow) - predicted_cash_flow) / abs(float(actual_cash_flow))
                        accuracy = 1 - error
                    elif abs(float(actual_cash_flow) - predicted_cash_flow) < 1.0:
                        accuracy = 1.0
                    
                    prediction.actual_result = {
                        'actual_cash_flow': float(actual_cash_flow),
                        'total_income': float(total_income),
                        'total_expense': float(total_expense),
                        'start_date': str(start_date),
                        'end_date': str(end_date)
                    }
                
                else:
                    self.stdout.write(self.style.WARNING(f"    ⚠️ Tipo de predicción '{prediction.type}' no soportado para evaluación."))
                    continue
                
                prediction.accuracy_score = max(0, min(1, accuracy)) # Asegurar que esté entre 0 y 1
                prediction.save()

                self.stdout.write(self.style.SUCCESS(f'    ✅ Evaluación completada para [{prediction.type}]. Precisión: {prediction.accuracy_score:.2%}'))
                evaluated_count += 1

            except Exception as e:
                self.stderr.write(self.style.ERROR(f'    ❌ Error evaluando predicción #{prediction.id}: {e}'))

        self.stdout.write(self.style.SUCCESS(f'\n🎉 Proceso completado. Se evaluaron {evaluated_count} predicciones.')) 