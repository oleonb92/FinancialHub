from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import TransactionViewSet, TagViewSet, CategoryViewSet, BudgetViewSet

router = DefaultRouter()
router.register(r'categories', CategoryViewSet, basename='categories')
router.register(r'tags', TagViewSet)
router.register(r'budgets', BudgetViewSet, basename='budgets')
router.register(r'', TransactionViewSet, basename='transactions')

urlpatterns = [
    path('', include(router.urls)),
]