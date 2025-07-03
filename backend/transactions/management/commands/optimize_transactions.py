from django.core.management.base import BaseCommand
from django.db import connection
from django.core.cache import cache
from transactions.models import Transaction
from ai.services import AIService
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Optimiza el rendimiento de las transacciones y análisis de IA'

    def add_arguments(self, parser):
        parser.add_argument(
            '--organization-id',
            type=int,
            help='ID de la organización específica a optimizar',
        )
        parser.add_argument(
            '--clear-cache',
            action='store_true',
            help='Limpiar caché antes de optimizar',
        )
        parser.add_argument(
            '--preload-ai',
            action='store_true',
            help='Precargar análisis de IA para transacciones recientes',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('🚀 Iniciando optimización de transacciones...'))
        
        org_id = options.get('organization_id')
        clear_cache = options.get('clear_cache')
        preload_ai = options.get('preload_ai')
        
        # Limpiar caché si se solicita
        if clear_cache:
            self.stdout.write('🧹 Limpiando caché...')
            cache.clear()
            self.stdout.write(self.style.SUCCESS('✅ Caché limpiado'))
        
        # Optimizar consultas de base de datos
        self.optimize_database_queries(org_id)
        
        # Precargar análisis de IA si se solicita
        if preload_ai:
            self.preload_ai_analysis(org_id)
        
        self.stdout.write(self.style.SUCCESS('✅ Optimización completada'))

    def optimize_database_queries(self, org_id=None):
        """Optimiza las consultas de base de datos."""
        self.stdout.write('📊 Optimizando consultas de base de datos...')
        
        with connection.cursor() as cursor:
            # Analizar tablas para optimizar el plan de consultas
            cursor.execute("ANALYZE transactions_transaction;")
            cursor.execute("ANALYZE transactions_category;")
            cursor.execute("ANALYZE chartofaccounts_account;")
            
            # Verificar índices existentes
            cursor.execute("""
                SELECT indexname, tablename 
                FROM pg_indexes 
                WHERE tablename IN ('transactions_transaction', 'transactions_category', 'chartofaccounts_account')
                ORDER BY tablename, indexname;
            """)
            
            indexes = cursor.fetchall()
            self.stdout.write(f'📋 Índices encontrados: {len(indexes)}')
            
            for index in indexes:
                self.stdout.write(f'   - {index[0]} en {index[1]}')
        
        self.stdout.write(self.style.SUCCESS('✅ Consultas optimizadas'))

    def preload_ai_analysis(self, org_id=None):
        """Precarga análisis de IA para transacciones recientes."""
        self.stdout.write('🤖 Precargando análisis de IA...')
        
        # Obtener transacciones recientes sin análisis
        filters = {'ai_analyzed': False}
        if org_id:
            filters['organization_id'] = org_id
        
        recent_transactions = Transaction.objects.filter(**filters).order_by('-date')[:100]
        
        if not recent_transactions:
            self.stdout.write('ℹ️ No hay transacciones recientes para analizar')
            return
        
        self.stdout.write(f'📈 Analizando {len(recent_transactions)} transacciones...')
        
        ai_service = AIService()
        processed = 0
        
        for transaction in recent_transactions:
            try:
                # Analizar transacción
                analysis_result = ai_service.analyze_transaction(transaction, force_reanalysis=False)
                
                # Actualizar campos del modelo
                if analysis_result.get('classification'):
                    transaction.ai_analyzed = True
                    transaction.ai_confidence = analysis_result['classification'].get('confidence', 0)
                    
                    # Asignar categoría sugerida si existe
                    suggested_category_id = analysis_result['classification'].get('category_id')
                    if suggested_category_id:
                        try:
                            from transactions.models import Category
                            suggested_category = Category.objects.get(id=suggested_category_id)
                            transaction.ai_category_suggestion = suggested_category
                        except Category.DoesNotExist:
                            pass
                    
                    transaction.save()
                
                processed += 1
                if processed % 10 == 0:
                    self.stdout.write(f'   Procesadas: {processed}/{len(recent_transactions)}')
                    
            except Exception as e:
                logger.error(f"Error analizando transacción {transaction.id}: {e}")
                continue
        
        self.stdout.write(self.style.SUCCESS(f'✅ Análisis precargado para {processed} transacciones'))

    def get_performance_stats(self, org_id=None):
        """Obtiene estadísticas de rendimiento."""
        self.stdout.write('📊 Obteniendo estadísticas de rendimiento...')
        
        filters = {}
        if org_id:
            filters['organization_id'] = org_id
        
        total_transactions = Transaction.objects.filter(**filters).count()
        analyzed_transactions = Transaction.objects.filter(**filters, ai_analyzed=True).count()
        cached_suggestions = 0
        
        # Contar sugerencias en caché
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM cache WHERE key LIKE 'transaction_suggestion_%';")
            cached_suggestions = cursor.fetchone()[0]
        
        self.stdout.write(f'📈 Estadísticas:')
        self.stdout.write(f'   - Total de transacciones: {total_transactions}')
        self.stdout.write(f'   - Transacciones analizadas: {analyzed_transactions}')
        self.stdout.write(f'   - Sugerencias en caché: {cached_suggestions}')
        self.stdout.write(f'   - Porcentaje analizado: {(analyzed_transactions/total_transactions*100):.1f}%' if total_transactions > 0 else '   - Porcentaje analizado: 0%') 