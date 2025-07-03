from django.core.management.base import BaseCommand
from django.db import transaction
from transactions.models import Transaction
from django.contrib.auth import get_user_model

User = get_user_model()

class Command(BaseCommand):
    help = 'Limpia el cache de IA de todas las transacciones para forzar reanálisis'

    def add_arguments(self, parser):
        parser.add_argument(
            '--organization',
            type=int,
            help='ID de la organización específica (opcional)',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Mostrar qué se haría sin ejecutar cambios',
        )

    def handle(self, *args, **options):
        org_id = options.get('organization')
        dry_run = options.get('dry_run')
        
        # Construir el queryset base
        queryset = Transaction.objects.all()
        
        if org_id:
            queryset = queryset.filter(organization_id=org_id)
            self.stdout.write(f"Limpiando cache de IA para organización {org_id}")
        else:
            self.stdout.write("Limpiando cache de IA para todas las organizaciones")
        
        # Contar transacciones que serán afectadas
        total_transactions = queryset.count()
        analyzed_transactions = queryset.filter(ai_analyzed=True).count()
        pending_transactions = queryset.filter(ai_category_suggestion__isnull=False).count()
        
        self.stdout.write(f"Total de transacciones: {total_transactions}")
        self.stdout.write(f"Transacciones analizadas (ai_analyzed=True): {analyzed_transactions}")
        self.stdout.write(f"Transacciones con sugerencias: {pending_transactions}")
        
        if dry_run:
            self.stdout.write(
                self.style.WARNING(
                    "DRY RUN - No se ejecutarán cambios. "
                    "Ejecuta sin --dry-run para aplicar los cambios."
                )
            )
            return
        
        # Confirmar antes de proceder
        confirm = input(f"\n¿Estás seguro de que quieres limpiar el cache de IA para {total_transactions} transacciones? (y/N): ")
        if confirm.lower() != 'y':
            self.stdout.write(self.style.WARNING("Operación cancelada."))
            return
        
        # Ejecutar la limpieza
        with transaction.atomic():
            updated_count = queryset.update(
                ai_analyzed=False,
                ai_confidence=None,
                ai_category_suggestion=None,
                ai_notes=None
            )
        
        self.stdout.write(
            self.style.SUCCESS(
                f"✅ Cache de IA limpiado exitosamente para {updated_count} transacciones"
            )
        )
        
        # Mostrar estadísticas finales
        final_analyzed = Transaction.objects.filter(ai_analyzed=True).count()
        final_pending = Transaction.objects.filter(ai_category_suggestion__isnull=False).count()
        
        self.stdout.write(f"Transacciones analizadas después: {final_analyzed}")
        self.stdout.write(f"Transacciones con sugerencias después: {final_pending}")
        
        self.stdout.write(
            self.style.SUCCESS(
                "🎯 Ahora puedes ejecutar el análisis masivo desde el frontend "
                "para regenerar todas las sugerencias de IA."
            )
        ) 