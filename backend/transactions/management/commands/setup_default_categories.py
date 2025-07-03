from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from transactions.models import Category
from organizations.models import Organization
import logging

logger = logging.getLogger(__name__)
User = get_user_model()

class Command(BaseCommand):
    help = 'Configura categorías y subcategorías por defecto para todas las organizaciones'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('🚀 Configurando categorías por defecto...'))
        
        # Categorías principales por defecto
        default_categories = [
            {
                'name': 'Alimentación',
                'color': '#FF6B6B',
                'subcategories': [
                    'Supermercado', 'Restaurantes', 'Café', 'Delivery', 'Snacks'
                ]
            },
            {
                'name': 'Transporte',
                'color': '#4ECDC4',
                'subcategories': [
                    'Gasolina', 'Transporte público', 'Taxi/Uber', 'Mantenimiento', 'Estacionamiento'
                ]
            },
            {
                'name': 'Vivienda',
                'color': '#45B7D1',
                'subcategories': [
                    'Renta', 'Hipoteca', 'Servicios públicos', 'Mantenimiento', 'Seguros'
                ]
            },
            {
                'name': 'Salud',
                'color': '#96CEB4',
                'subcategories': [
                    'Médico', 'Farmacia', 'Seguro médico', 'Dental', 'Óptica'
                ]
            },
            {
                'name': 'Entretenimiento',
                'color': '#FFEAA7',
                'subcategories': [
                    'Cine', 'Conciertos', 'Deportes', 'Juegos', 'Hobbies'
                ]
            },
            {
                'name': 'Educación',
                'color': '#DDA0DD',
                'subcategories': [
                    'Matrícula', 'Libros', 'Cursos', 'Material escolar', 'Tecnología educativa'
                ]
            },
            {
                'name': 'Ropa y Accesorios',
                'color': '#FFB6C1',
                'subcategories': [
                    'Ropa', 'Zapatos', 'Accesorios', 'Joyería', 'Cosméticos'
                ]
            },
            {
                'name': 'Tecnología',
                'color': '#87CEEB',
                'subcategories': [
                    'Electrónicos', 'Software', 'Servicios digitales', 'Reparaciones', 'Accesorios'
                ]
            },
            {
                'name': 'Servicios Financieros',
                'color': '#98FB98',
                'subcategories': [
                    'Comisiones bancarias', 'Seguros', 'Inversiones', 'Préstamos', 'Tarjetas de crédito'
                ]
            },
            {
                'name': 'Ingresos',
                'color': '#32CD32',
                'subcategories': [
                    'Salario', 'Freelance', 'Inversiones', 'Ventas', 'Bonos'
                ]
            },
            {
                'name': 'Otros',
                'color': '#D3D3D3',
                'subcategories': [
                    'Regalos', 'Donaciones', 'Impuestos', 'Multas', 'Otros gastos'
                ]
            }
        ]
        
        organizations = Organization.objects.all()
        total_created = 0
        
        for organization in organizations:
            self.stdout.write(f'📊 Configurando organización: {organization.name}')
            
            # Obtener el primer usuario de la organización como created_by
            first_user = organization.users.first()
            if not first_user:
                self.stdout.write(self.style.WARNING(f'⚠️  No hay usuarios en la organización {organization.name}'))
                continue
            
            for cat_data in default_categories:
                # Crear categoría principal si no existe
                category, created = Category.objects.get_or_create(
                    name=cat_data['name'],
                    organization=organization,
                    parent=None,  # Categoría principal
                    defaults={
                        'created_by': first_user
                    }
                )
                
                if created:
                    total_created += 1
                    self.stdout.write(f'  ✅ Categoría creada: {category.name}')
                
                # Crear subcategorías
                for subcat_name in cat_data['subcategories']:
                    subcategory, sub_created = Category.objects.get_or_create(
                        name=subcat_name,
                        organization=organization,
                        parent=category,  # Subcategoría de la categoría principal
                        defaults={
                            'created_by': first_user
                        }
                    )
                    
                    if sub_created:
                        total_created += 1
                        self.stdout.write(f'    📝 Subcategoría creada: {subcategory.name}')
        
        self.stdout.write(self.style.SUCCESS(f'🎉 Configuración completada! Se crearon {total_created} elementos.'))
        self.stdout.write(self.style.SUCCESS('💡 Las categorías por defecto están listas para que la IA las use como referencia.')) 