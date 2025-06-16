from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from incentives.models import Incentive
from incentives.serializers import IncentiveSerializer
from accounts.models import User
from organizations.models import Organization
from notifications.models import Notification
from accounts.access_control import has_pro_access

class IncentiveListCreateView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        # Controlar acceso a feature Pro 'incentives'
        org_id = request.query_params.get('organization_id')
        org = Organization.objects.get(id=org_id) if org_id else None
        if not has_pro_access(request.user, org, feature='incentives'):
            return Response({'detail': 'Acceso a incentivos solo para usuarios Pro.'}, status=status.HTTP_402_PAYMENT_REQUIRED)
        if getattr(request.user, 'role', None) == 'accountant':
            incentives = Incentive.objects.filter(accountant=request.user).order_by('-created_at')
        else:
            incentives = Incentive.objects.filter(client=request.user).order_by('-created_at')
        serializer = IncentiveSerializer(incentives, many=True)
        return Response(serializer.data)

    def post(self, request):
        # Controlar acceso a feature Pro 'incentives'
        org_id = request.data.get('organization')
        org = Organization.objects.get(id=org_id) if org_id else None
        if not has_pro_access(request.user, org, feature='incentives'):
            return Response({'detail': 'Acceso a incentivos solo para usuarios Pro.'}, status=status.HTTP_402_PAYMENT_REQUIRED)
        if getattr(request.user, 'role', None) != 'accountant':
            return Response({'detail': 'Solo los contadores pueden crear incentivos.'}, status=status.HTTP_403_FORBIDDEN)
        serializer = IncentiveSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(accountant=request.user)
            # Notificación interna al cliente
            client = User.objects.get(id=request.data['client'])
            Notification.objects.create(
                user=client,
                message=f"Has recibido un nuevo incentivo de {request.user.username}: {serializer.validated_data['amount']}"
            )
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class IncentiveDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, pk):
        try:
            incentive = Incentive.objects.get(pk=pk)
            # Solo el contador o el cliente pueden ver el detalle
            if incentive.accountant != request.user and incentive.client != request.user:
                return Response({'detail': 'No autorizado.'}, status=status.HTTP_403_FORBIDDEN)
            serializer = IncentiveSerializer(incentive)
            return Response(serializer.data)
        except Incentive.DoesNotExist:
            return Response({'detail': 'No encontrado.'}, status=status.HTTP_404_NOT_FOUND)

    def delete(self, request, pk):
        try:
            incentive = Incentive.objects.get(pk=pk)
            # Solo el contador puede eliminar
            if incentive.accountant != request.user:
                return Response({'detail': 'No autorizado.'}, status=status.HTTP_403_FORBIDDEN)
            incentive.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        except Incentive.DoesNotExist:
            return Response({'detail': 'No encontrado.'}, status=status.HTTP_404_NOT_FOUND) 