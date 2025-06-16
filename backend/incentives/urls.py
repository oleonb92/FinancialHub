from django.urls import path
from incentives.views import IncentiveListCreateView, IncentiveDetailView

urlpatterns = [
    path('', IncentiveListCreateView.as_view(), name='incentive-list-create'),
    path('<int:pk>/', IncentiveDetailView.as_view(), name='incentive-detail'),
] 