from django.core.management.base import BaseCommand
from django.db import transaction
from transactions.models import Transaction
from ai.services import AIService
from ai.models import AIInteraction, AIInsight
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Genera insights para todas las transacciones existentes en la base de datos.'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('🚀 Iniciando la generación de insights para transacciones existentes...'))
        
        ai_service = AIService()
        transactions = Transaction.objects.all()
        
        if not transactions.exists():
            self.stdout.write(self.style.WARNING('No se encontraron transacciones para analizar.'))
            return

        total_transactions = transactions.count()
        self.stdout.write(f'🔍 Se encontraron {total_transactions} transacciones.')

        insights_generados = 0
        errores = 0

        with transaction.atomic():
            for i, trans in enumerate(transactions):
                try:
                    # 1. Simular una interacción de análisis
                    interaction = AIInteraction.objects.create(
                        user=trans.created_by,
                        organization=trans.organization,
                        type='transaction_analysis',
                        query=f"Analizar transacción {trans.id}",
                        context={'transaction_id': trans.id}
                    )
                    
                    # 2. Analizar la transacción
                    analysis_result = ai_service.analyze_transaction(trans)
                    
                    interaction.response = analysis_result
                    interaction.confidence_score = analysis_result.get('confidence_score', 0.8)
                    interaction.save()
                    
                    # 3. Generar el insight
                    ai_service._generate_insights(trans.created_by, interaction)
                    
                    insights_generados += 1
                    self.stdout.write(self.style.SUCCESS(f'✅ Insight generado para transacción {trans.id} ({i+1}/{total_transactions})'))

                except Exception as e:
                    errores += 1
                    self.stderr.write(self.style.ERROR(f'❌ Error analizando transacción {trans.id}: {e}'))
        
        self.stdout.write(self.style.SUCCESS('\n🎉 Proceso completado.'))
        self.stdout.write(f'   - Insights generados: {insights_generados}')
        self.stdout.write(f'   - Errores encontrados: {errores}') 