# from rest_framework import viewsets, permissions
# from rest_framework.exceptions import PermissionDenied
# from .models import Transaction
# from .serializers import TransactionSerializer

# class TransactionViewSet(viewsets.ModelViewSet):
#     serializer_class = TransactionSerializer
#     permission_classes = [permissions.IsAuthenticated]

#     def get_queryset(self):
#         return Transaction.objects.filter(household=self.request.user.household)

#     def perform_create(self, serializer):
#         serializer.save(created_by=self.request.user, household=self.request.user.household)

#     def get_object(self):
#         obj = super().get_object()
#         if obj.household != self.request.user.household:
#             raise PermissionDenied("You do not have permission to access this transaction.")
#         return obj

#     def perform_update(self, serializer):
#         obj = self.get_object()
#         if obj.created_by != self.request.user:
#             raise PermissionDenied("You can only edit your own transactions.")
#         serializer.save()

#     def perform_destroy(self, instance):
#         if instance.created_by != self.request.user:
#             raise PermissionDenied("You can only delete your own transactions.")
#         instance.delete()

from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.db.models import Sum, Count, Q
from django.db.models.functions import ExtractYear, ExtractMonth
from django.utils import timezone
from .models import Transaction, Category, Tag, Budget
from .serializers import TransactionSerializer, CategorySerializer, TagSerializer, BudgetSerializer
from organizations.models import Organization
from accounts.access_control import require_access, has_pro_access
from rest_framework.pagination import PageNumberPagination

class StandardResultsSetPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100

# Transacción ViewSet
class TransactionViewSet(viewsets.ModelViewSet):
    serializer_class = TransactionSerializer
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = StandardResultsSetPagination

    @require_access(required_roles=["admin", "accountant"], allow_accountant_always=True)
    def get_queryset(self):
        if getattr(self, 'swagger_fake_view', False):
            return Transaction.objects.none()
        
        queryset = Transaction.objects.filter(organization=self.request.organization)
        
        # Filtros
        transaction_type = self.request.query_params.get('type', None)
        if transaction_type:
            queryset = queryset.filter(type=transaction_type)
            
        category_ids = self.request.query_params.get('category_ids', None)
        if category_ids:
            category_ids = category_ids.split(',')
            queryset = queryset.filter(category_id__in=category_ids)
            
        start_date = self.request.query_params.get('start_date', None)
        if start_date:
            queryset = queryset.filter(date__gte=start_date)
            
        end_date = self.request.query_params.get('end_date', None)
        if end_date:
            queryset = queryset.filter(date__lte=end_date)
            
        search = self.request.query_params.get('search', None)
        if search:
            queryset = queryset.filter(
                Q(description__icontains=search) |
                Q(merchant__icontains=search)
            )
            
        return queryset.order_by('-date', '-id')

    @require_access(required_roles=["admin", "accountant"], allow_accountant_always=True)
    def perform_create(self, serializer):
        serializer.save(
            organization=self.request.organization,
            created_by=self.request.user
        )

    @require_access(required_roles=["admin", "accountant"], allow_accountant_always=True)
    def get_object(self):
        if not hasattr(self.request, 'organization'):
            raise PermissionDenied("No se ha especificado una organización. Por favor, incluye el header 'X-Organization-ID' en tu solicitud.")
        obj = super().get_object()
        if obj.organization != self.request.organization:
            raise PermissionDenied("You do not have permission to access this transaction.")
        return obj

    @require_access(required_roles=["admin", "accountant"], allow_accountant_always=True)
    def perform_update(self, serializer):
        if not hasattr(self.request, 'organization'):
            raise PermissionDenied("No se ha especificado una organización. Por favor, incluye el header 'X-Organization-ID' en tu solicitud.")
        obj = self.get_object()
        if obj.created_by != self.request.user:
            raise PermissionDenied("You can only edit your own transactions.")
        serializer.save()

    @require_access(required_roles=["admin", "accountant"], allow_accountant_always=True)
    def perform_destroy(self, instance):
        if not hasattr(self.request, 'organization'):
            raise PermissionDenied("No se ha especificado una organización. Por favor, incluye el header 'X-Organization-ID' en tu solicitud.")
        if instance.created_by != self.request.user:
            raise PermissionDenied("You can only delete your own transactions.")
        instance.delete()

    @action(detail=False, methods=['get'])
    def summary(self, request):
        queryset = self.get_queryset()
        
        # Calcular totales
        income = queryset.filter(type='INCOME').aggregate(total=Sum('amount'))['total'] or 0
        expenses = queryset.filter(type='EXPENSE').aggregate(total=Sum('amount'))['total'] or 0
        net = income - expenses
        
        return Response({
            'income': income,
            'expenses': expenses,
            'net': net,
            'total_transactions': queryset.count()
        })

# Tag ViewSet
class TagViewSet(viewsets.ModelViewSet):
    queryset = Tag.objects.all()
    serializer_class = TagSerializer
    @require_access(required_roles=["admin", "accountant"], require_pro=True, allow_accountant_always=True)
    def get_queryset(self):
        return Tag.objects.all()

# Category ViewSet
class CategoryViewSet(viewsets.ModelViewSet):
    serializer_class = CategorySerializer
    permission_classes = [permissions.IsAuthenticated]
    queryset = Category.objects.all()

    @require_access(required_roles=["admin", "accountant"], allow_accountant_always=True)
    def get_queryset(self):
        if not hasattr(self.request, 'organization'):
            raise PermissionDenied("No se ha especificado una organización. Por favor, incluye el header 'X-Organization-ID' en tu solicitud.")
        
        queryset = Category.objects.filter(organization=self.request.organization)
        
        # Si se solicita solo categorías principales
        if self.request.query_params.get('top_level', 'false').lower() == 'true':
            queryset = queryset.filter(parent__isnull=True)
        
        # Si se solicita subcategorías de una categoría específica
        parent_id = self.request.query_params.get('parent_id')
        if parent_id:
            queryset = queryset.filter(parent_id=parent_id)
        
        return queryset.select_related('parent')

    @require_access(required_roles=["admin", "accountant"], allow_accountant_always=True)
    def perform_create(self, serializer):
        if not hasattr(self.request, 'organization'):
            raise PermissionDenied("No se ha especificado una organización. Por favor, incluye el header 'X-Organization-ID' en tu solicitud.")
        serializer.save(created_by=self.request.user, organization=self.request.organization)

    @require_access(required_roles=["admin", "accountant"], allow_accountant_always=True)
    def perform_update(self, serializer):
        if not hasattr(self.request, 'organization'):
            raise PermissionDenied("No se ha especificado una organización. Por favor, incluye el header 'X-Organization-ID' en tu solicitud.")
        serializer.save()

    @require_access(required_roles=["admin", "accountant"], allow_accountant_always=True)
    def perform_destroy(self, instance):
        if not hasattr(self.request, 'organization'):
            raise PermissionDenied("No se ha especificado una organización. Por favor, incluye el header 'X-Organization-ID' en tu solicitud.")
        
        # Verificar si hay subcategorías
        if instance.children.exists():
            raise PermissionDenied("No se puede eliminar una categoría que tiene subcategorías. Elimine primero las subcategorías.")
        
        # Verificar si hay transacciones usando esta categoría
        if instance.transactions.exists():
            raise PermissionDenied("No se puede eliminar una categoría que está siendo usada en transacciones.")
        
        instance.delete()

    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

class BudgetViewSet(viewsets.ModelViewSet):
    serializer_class = BudgetSerializer
    permission_classes = [permissions.IsAuthenticated]

    @require_access(required_roles=["admin", "accountant"], allow_accountant_always=True)
    def get_queryset(self):
        if not hasattr(self.request, 'organization'):
            raise PermissionDenied("No se ha especificado una organización. Por favor, incluye el header 'X-Organization-ID' en tu solicitud.")
        
        queryset = Budget.objects.filter(organization=self.request.organization)
        
        # Filtrar por período si se especifica
        period = self.request.query_params.get('period')
        if period:
            queryset = queryset.filter(period=period)
        
        # Filtrar por categoría si se especifica
        category_id = self.request.query_params.get('category_id')
        if category_id:
            queryset = queryset.filter(category_id=category_id)
        
        return queryset.select_related('category', 'organization', 'created_by')

    @require_access(required_roles=["admin", "accountant"], allow_accountant_always=True)
    def perform_create(self, serializer):
        if not hasattr(self.request, 'organization'):
            raise PermissionDenied("No se ha especificado una organización. Por favor, incluye el header 'X-Organization-ID' en tu solicitud.")
        serializer.save(created_by=self.request.user, organization=self.request.organization)

    @require_access(required_roles=["admin", "accountant"], allow_accountant_always=True)
    def perform_update(self, serializer):
        if not hasattr(self.request, 'organization'):
            raise PermissionDenied("No se ha especificado una organización. Por favor, incluye el header 'X-Organization-ID' en tu solicitud.")
        serializer.save()