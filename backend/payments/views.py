import stripe
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.core.exceptions import ValidationError
from payments.services import (
    create_stripe_customer,
    create_stripe_subscription,
    cancel_stripe_subscription,
    get_stripe_subscription,
    update_stripe_subscription,
    attach_subscription_to_org,
    get_subscription_status
)
from payments.models import Subscription, Payment
from organizations.models import Organization, OrganizationMembership
from django.utils import timezone
from incentives.models import Incentive
from accounts.models import User
from django.http import HttpResponse
from core.services.subscription import SubscriptionService
from core.exceptions import StripeError
from core.services.access_control import AccessControlService

class CreateStripeSubscriptionView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            org_id = request.data.get('organization_id')
            price_id = request.data.get('price_id')
            trial_period_days = request.data.get('trial_period_days')

            if not org_id or not price_id:
                return Response(
                    {'error': 'organization_id and price_id are required'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            org = Organization.objects.get(id=org_id)
            
            # Verificar si ya existe una suscripción
            if hasattr(org, 'subscription'):
                return Response(
                    {'error': 'Organization already has a subscription'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            customer_id = create_stripe_customer(
                email=request.user.email,
                name=org.name,
                metadata={'organization_id': str(org.id)}
            )
            
            subscription = create_stripe_subscription(
                customer_id=customer_id,
                price_id=price_id,
                trial_period_days=trial_period_days
            )
            
            attach_subscription_to_org(
                org=org,
                stripe_customer_id=customer_id,
                stripe_subscription_id=subscription['id'],
                plan=price_id,
                status=subscription['status'],
                current_period_end=timezone.datetime.fromtimestamp(
                    subscription['current_period_end'],
                    tz=timezone.utc
                )
            )

            return Response({
                'client_secret': subscription['latest_invoice']['payment_intent']['client_secret'],
                'subscription_id': subscription['id'],
                'status': subscription['status']
            })

        except Organization.DoesNotExist:
            return Response(
                {'error': 'Organization not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        except ValidationError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class CancelSubscriptionView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            org_id = request.data.get('organization_id')
            if not org_id:
                return Response(
                    {'error': 'organization_id is required'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            org = Organization.objects.get(id=org_id)
            subscription = Subscription.objects.get(organization=org)
            
            # Cancelar en Stripe
            stripe_subscription = cancel_stripe_subscription(subscription.stripe_subscription_id)
            
            # Actualizar en nuestra base de datos
            subscription.status = 'canceled'
            subscription.save()

            return Response({
                'status': 'success',
                'message': 'Subscription canceled successfully'
            })

        except (Organization.DoesNotExist, Subscription.DoesNotExist):
            return Response(
                {'error': 'Organization or subscription not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        except ValidationError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class UpdateSubscriptionView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            org_id = request.data.get('organization_id')
            new_price_id = request.data.get('price_id')

            if not org_id or not new_price_id:
                return Response(
                    {'error': 'organization_id and price_id are required'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            org = Organization.objects.get(id=org_id)
            subscription = Subscription.objects.get(organization=org)
            
            # Actualizar en Stripe
            stripe_subscription = update_stripe_subscription(
                subscription.stripe_subscription_id,
                new_price_id
            )
            
            # Actualizar en nuestra base de datos
            subscription.plan = new_price_id
            subscription.status = stripe_subscription['status']
            subscription.current_period_end = timezone.datetime.fromtimestamp(
                stripe_subscription['current_period_end'],
                tz=timezone.utc
            )
            subscription.save()

            return Response({
                'status': 'success',
                'message': 'Subscription updated successfully',
                'subscription': {
                    'status': subscription.status,
                    'plan': subscription.plan,
                    'current_period_end': subscription.current_period_end
                }
            })

        except (Organization.DoesNotExist, Subscription.DoesNotExist):
            return Response(
                {'error': 'Organization or subscription not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        except ValidationError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class GetSubscriptionStatusView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            org_id = request.query_params.get('organization_id')
            if not org_id:
                return Response(
                    {'error': 'organization_id is required'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            org = Organization.objects.get(id=org_id)
            status_data = get_subscription_status(org)
            
            if not status_data:
                return Response(
                    {'error': 'No subscription found'},
                    status=status.HTTP_404_NOT_FOUND
                )

            return Response(status_data)

        except Organization.DoesNotExist:
            return Response(
                {'error': 'Organization not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        except ValidationError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class StripeWebhookView(APIView):
    """
    Handle Stripe webhook events
    """
    permission_classes = []  # No authentication required for webhooks
    
    def post(self, request, *args, **kwargs):
        payload = request.body
        sig_header = request.META.get('HTTP_STRIPE_SIGNATURE')
        
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
            )
        except ValueError as e:
            return Response(
                {'error': 'Invalid payload'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        except stripe.error.SignatureVerificationError as e:
            return Response(
                {'error': 'Invalid signature'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            SubscriptionService.handle_webhook_event(event)
            return HttpResponse(status=200)
        except StripeError as e:
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response(
                {'error': 'Internal server error'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            ) 