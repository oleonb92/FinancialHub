import logging
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
from ai.services import AIService
from django.core.cache import cache

logger = logging.getLogger(__name__)

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

    @action(detail=True, methods=['post'])
    def analyze_with_ai(self, request, pk=None):
        """Re-analiza una transacción con IA y sugiere categorías."""
        try:
            transaction = self.get_object()
            
            # Verificar permisos
            if not request.user.has_perm('transactions.view_transaction', transaction):
                return Response({'error': 'No tienes permisos para analizar esta transacción'}, 
                              status=status.HTTP_403_FORBIDDEN)
            
            # Forzar re-análisis
            ai_service = AIService()
            analysis_result = ai_service.analyze_transaction(transaction, force_reanalysis=True)
            
            if 'error' in analysis_result:
                return Response({'error': analysis_result['error']}, 
                              status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # Actualizar la transacción con el nuevo análisis
            transaction.ai_analyzed = True
            transaction.ai_confidence = analysis_result['classification']['confidence']
            
            # Asignar la categoría sugerida
            if analysis_result['classification']['category_id']:
                try:
                    suggested_category = Category.objects.get(
                        id=analysis_result['classification']['category_id'],
                        organization=transaction.organization
                    )
                    transaction.ai_category_suggestion = suggested_category
                except Category.DoesNotExist:
                    transaction.ai_category_suggestion = None
            else:
                transaction.ai_category_suggestion = None
            
            transaction.save()
            
            return Response({
                'message': 'Análisis completado exitosamente',
                'analysis': analysis_result
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error analizando transacción {pk}: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=['post'])
    def approve_suggestion(self, request, pk=None):
        """Aprueba una sugerencia de categoría de la IA."""
        try:
            transaction = self.get_object()
            suggested_category_id = request.data.get('suggested_category_id')
            
            if not suggested_category_id:
                return Response({'error': 'suggested_category_id es requerido'}, 
                              status=status.HTTP_400_BAD_REQUEST)
            
            # Verificar permisos
            if not request.user.has_perm('transactions.change_transaction', transaction):
                return Response({'error': 'No tienes permisos para modificar esta transacción'}, 
                              status=status.HTTP_403_FORBIDDEN)
            
            # Obtener la categoría sugerida
            try:
                suggested_category = Category.objects.get(
                    id=suggested_category_id,
                    organization=transaction.organization
                )
            except Category.DoesNotExist:
                return Response({'error': 'Categoría sugerida no encontrada'}, 
                              status=status.HTTP_404_NOT_FOUND)
            
            # Actualizar la transacción
            old_category = transaction.category
            transaction.category = suggested_category
            transaction.save()
            
            # Registrar la aprobación en el historial
            logger.info(f"Usuario {request.user.id} aprobó sugerencia de categoría: "
                       f"Transacción {transaction.id} cambió de '{old_category.name if old_category else 'Sin categoría'}' "
                       f"a '{suggested_category.name}'")
            
            return Response({
                'message': f'Categoría actualizada exitosamente a "{suggested_category.name}"',
                'old_category': old_category.name if old_category else 'Sin categoría',
                'new_category': suggested_category.name
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error aprobando sugerencia para transacción {pk}: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=['post'])
    def reject_suggestion(self, request, pk=None):
        """Rechaza una sugerencia de categoría de la IA."""
        try:
            transaction = self.get_object()
            suggested_category_id = request.data.get('suggested_category_id')
            reason = request.data.get('reason', 'Usuario rechazó la sugerencia')
            
            if not suggested_category_id:
                return Response({'error': 'suggested_category_id es requerido'}, 
                              status=status.HTTP_400_BAD_REQUEST)
            
            # Verificar permisos
            if not request.user.has_perm('transactions.change_transaction', transaction):
                return Response({'error': 'No tienes permisos para modificar esta transacción'}, 
                              status=status.HTTP_403_FORBIDDEN)
            
            # Registrar el rechazo en el historial
            logger.info(f"Usuario {request.user.id} rechazó sugerencia de categoría: "
                       f"Transacción {transaction.id}, categoría sugerida ID {suggested_category_id}, "
                       f"razón: {reason}")
            
            return Response({
                'message': 'Sugerencia rechazada exitosamente',
                'reason': reason
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error rechazando sugerencia para transacción {pk}: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=False, methods=['post'])
    def bulk_analyze(self, request):
        """Re-analiza múltiples transacciones con IA."""
        try:
            transaction_ids = request.data.get('transaction_ids', [])
            force_reanalysis = request.data.get('force_reanalysis', True)
            
            if not transaction_ids:
                return Response({'error': 'transaction_ids es requerido'}, 
                              status=status.HTTP_400_BAD_REQUEST)
            
            # Verificar permisos para todas las transacciones
            transactions = Transaction.objects.filter(
                id__in=transaction_ids,
                organization=request.organization
            )
            
            if len(transactions) != len(transaction_ids):
                return Response({'error': 'No tienes permisos para algunas transacciones'}, 
                              status=status.HTTP_403_FORBIDDEN)
            
            # Analizar cada transacción
            ai_service = AIService()
            results = []
            
            for transaction in transactions:
                try:
                    analysis_result = ai_service.analyze_transaction(transaction, force_reanalysis=force_reanalysis)
                    logger.info(f"[AI][BULK_ANALYZE] tx {transaction.id} - analysis_result: {analysis_result}")
                    
                    if 'error' not in analysis_result:
                        # Actualizar la transacción
                        transaction.ai_analyzed = True
                        transaction.ai_confidence = analysis_result['classification']['confidence']
                        
                        # Asignar la categoría sugerida
                        if analysis_result['classification']['category_id']:
                            try:
                                suggested_category = Category.objects.get(
                                    id=analysis_result['classification']['category_id'],
                                    organization=transaction.organization
                                )
                                transaction.ai_category_suggestion = suggested_category
                            except Category.DoesNotExist:
                                transaction.ai_category_suggestion = None
                        else:
                            transaction.ai_category_suggestion = None
                        
                        transaction.save()
                        logger.info(f"[AI][BULK_ANALYZE] tx {transaction.id} - ai_category_suggestion: {transaction.ai_category_suggestion}, ai_confidence: {transaction.ai_confidence}")
                        
                        results.append({
                            'transaction_id': transaction.id,
                            'status': 'success',
                            'analysis': analysis_result
                        })
                    else:
                        results.append({
                            'transaction_id': transaction.id,
                            'status': 'error',
                            'error': analysis_result['error']
                        })
                        
                except Exception as e:
                    results.append({
                        'transaction_id': transaction.id,
                        'status': 'error',
                        'error': str(e)
                    })
            
            return Response({
                'message': f'Análisis completado para {len(results)} transacciones',
                'results': results
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error en análisis masivo: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=False, methods=['get'])
    def ai_suggestions(self, request):
        """Obtiene sugerencias de IA para transacciones de forma asíncrona."""
        try:
            transaction_ids = request.query_params.get('transaction_ids', '').split(',')
            transaction_ids = [int(tid) for tid in transaction_ids if tid.isdigit()]
            
            if not transaction_ids:
                return Response({'error': 'transaction_ids es requerido'}, 
                              status=status.HTTP_400_BAD_REQUEST)
            
            # Verificar permisos
            transactions = Transaction.objects.filter(
                id__in=transaction_ids,
                organization=request.organization
            )
            
            if len(transactions) != len(transaction_ids):
                return Response({'error': 'No tienes permisos para algunas transacciones'}, 
                              status=status.HTTP_403_FORBIDDEN)
            
            # Obtener sugerencias desde caché o generar nuevas
            suggestions = {}
            ai_service = AIService()
            
            for transaction in transactions:
                cache_key = f"transaction_suggestion_{transaction.id}"
                cached_suggestion = cache.get(cache_key)
                
                if cached_suggestion:
                    suggestions[transaction.id] = cached_suggestion
                else:
                    # Generar sugerencia y cachear
                    try:
                        analysis_result = ai_service.analyze_transaction(transaction, force_reanalysis=False)
                        if 'suggestion' in analysis_result:
                            suggestion = analysis_result['suggestion']
                            # Cachear por 1 hora
                            cache.set(cache_key, suggestion, timeout=3600)
                            suggestions[transaction.id] = suggestion
                        else:
                            suggestions[transaction.id] = {
                                'status': 'error',
                                'message': 'Error al analizar sugerencias',
                                'needs_update': False,
                                'current_category_id': transaction.category.id if transaction.category else None,
                                'suggested_category_id': None,
                                'already_approved': False
                            }
                    except Exception as e:
                        logger.warning(f"Error obteniendo sugerencias para transacción {transaction.id}: {e}")
                        suggestions[transaction.id] = {
                            'status': 'error',
                            'message': f'Error: {str(e)}',
                            'needs_update': False,
                            'current_category_id': transaction.category.id if transaction.category else None,
                            'suggested_category_id': None,
                            'already_approved': False
                        }
            
            return Response({
                'suggestions': suggestions,
                'total_processed': len(suggestions)
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error obteniendo sugerencias de IA: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=False, methods=['get'])
    def categories_hierarchy(self, request):
        """Obtiene todas las categorías organizadas jerárquicamente."""
        try:
            # Obtener solo categorías principales (sin parent)
            main_categories = Category.objects.filter(
                organization__users=request.user,
                parent=None
            ).prefetch_related('children')
            
            # Serializar con subcategorías
            serializer = CategorySerializer(main_categories, many=True, context={'request': request})
            
            return Response({
                'categories': serializer.data,
                'total_categories': main_categories.count()
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error obteniendo jerarquía de categorías: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=False, methods=['get'])
    def categories_flat(self, request):
        """Obtiene todas las categorías en formato plano (útil para dropdowns)."""
        try:
            categories = Category.objects.filter(
                organization__users=request.user
            ).select_related('parent')
            
            # Formatear para dropdown
            flat_categories = []
            for cat in categories:
                if cat.parent:
                    # Subcategoría
                    flat_categories.append({
                        'id': cat.id,
                        'name': f"{cat.parent.name} > {cat.name}",
                        'parent_id': cat.parent.id,
                        'parent_name': cat.parent.name,
                        'is_subcategory': True
                    })
                else:
                    # Categoría principal
                    flat_categories.append({
                        'id': cat.id,
                        'name': cat.name,
                        'parent_id': None,
                        'parent_name': None,
                        'is_subcategory': False
                    })
            
            return Response({
                'categories': flat_categories,
                'total_categories': len(flat_categories)
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error obteniendo categorías planas: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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