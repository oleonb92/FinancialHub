import logging
from datetime import date, timedelta
from django.core.management.base import BaseCommand
from django.db import transaction
from django.contrib.auth import get_user_model
from organizations.models import Organization
from ai.services import AIService
from ai.models import AIPrediction

logger = logging.getLogger(__name__)
User = get_user_model()

class Command(BaseCommand):
    help = 'Genera predicciones de IA para todas las organizaciones y usuarios existentes.'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('🚀 Iniciando la generación de predicciones de IA...'))
        
        ai_service = AIService()
        organizations = Organization.objects.all()
        
        if not organizations.exists():
            self.stdout.write(self.style.WARNING('No se encontraron organizaciones. Creando una de prueba.'))
            admin_user = User.objects.filter(is_superuser=True).first()
            if not admin_user:
                self.stderr.write(self.style.ERROR('No se encontró un superusuario para crear una organización de prueba.'))
                return
            organization = Organization.objects.create(name="Default Organization", owner=admin_user)
            organization.members.add(admin_user)
            organizations = [organization]
        else:
            organizations = list(organizations)

        total_orgs = len(organizations)
        self.stdout.write(f'🔍 Se procesarán {total_orgs} organizaciones.')

        predictions_generadas = 0
        errores = 0

        for i, org in enumerate(organizations):
            self.stdout.write(self.style.HTTP_INFO(f'\nProcesando Organización: {org.name} ({i+1}/{total_orgs})'))
            
            # Asignar al primer superusuario como responsable de la predicción
            admin_user = User.objects.filter(is_superuser=True).first()
            if not admin_user:
                self.stderr.write(self.style.ERROR('No se encontró un superusuario. Abortando.'))
                return

            # --- Generar Predicción de Gastos ---
            try:
                self.stdout.write('  💰 Generando predicción de gastos...')
                expense_prediction = ai_service.predict_expenses_simple(organization=org)
                
                if expense_prediction and expense_prediction.get('status') == 'success':
                    # Calcular el promedio de confianza de las predicciones diarias
                    daily_predictions = expense_prediction.get('predictions', [])
                    if daily_predictions:
                        total_confidence = sum(p.get('confidence', 0.0) for p in daily_predictions)
                        avg_confidence = total_confidence / len(daily_predictions)
                    else:
                        avg_confidence = 0.0

                    with transaction.atomic():
                        AIPrediction.objects.create(
                            user=admin_user,
                            organization=org,
                            type='spending',
                            prediction_date=date.today() + timedelta(days=30),
                            prediction=expense_prediction,
                            confidence_score=avg_confidence
                        )
                    predictions_generadas += 1
                    self.stdout.write(self.style.SUCCESS('  ✅ Predicción de gastos generada.'))
                else:
                    self.stdout.write(self.style.WARNING(f"  ⚠️ No se pudo generar la predicción de gastos: {expense_prediction.get('message', 'Sin mensaje')}"))

            except Exception as e:
                errores += 1
                self.stderr.write(self.style.ERROR(f'  ❌ Error en predicción de gastos: {e}'))

            # --- Generar Predicción de Flujo de Caja ---
            try:
                self.stdout.write('  🌊 Generando predicción de flujo de caja...')
                cash_flow_prediction = ai_service.predict_cash_flow(organization=org)
                
                if cash_flow_prediction and cash_flow_prediction.get('status') == 'success':
                    # Calcular el promedio de confianza si existe en los datos
                    daily_cash_flow = cash_flow_prediction.get('predictions', [])
                    if daily_cash_flow:
                        total_confidence = sum(p.get('confidence', 0.0) for p in daily_cash_flow)
                        avg_confidence = total_confidence / len(daily_cash_flow)
                    else:
                        avg_confidence = cash_flow_prediction.get('confidence', 0.0)

                    with transaction.atomic():
                        AIPrediction.objects.create(
                            user=admin_user,
                            organization=org,
                            type='cash_flow',
                            prediction_date=date.today() + timedelta(days=30),
                            prediction=cash_flow_prediction,
                            confidence_score=avg_confidence
                        )
                    predictions_generadas += 1
                    self.stdout.write(self.style.SUCCESS('  ✅ Predicción de flujo de caja generada.'))
                else:
                     self.stdout.write(self.style.WARNING(f"  ⚠️ No se pudo generar la predicción de flujo de caja: {cash_flow_prediction.get('message', 'Sin mensaje')}"))

            except Exception as e:
                errores += 1
                self.stderr.write(self.style.ERROR(f'  ❌ Error en predicción de flujo de caja: {e}'))
        
        self.stdout.write(self.style.SUCCESS('\n🎉 Proceso completado.'))
        self.stdout.write(f'   - Predicciones generadas: {predictions_generadas}')
        self.stdout.write(f'   - Errores encontrados: {errores}') 