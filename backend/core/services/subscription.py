import logging
import stripe
from django.conf import settings
from django.core.cache import cache
from django.utils import timezone
from payments.models import Subscription
from organizations.models import Organization
from core.exceptions import StripeError, SubscriptionError, ValidationError

logger = logging.getLogger('payments')

class SubscriptionService:
    """Service for handling subscription management and Stripe integration"""
    
    CACHE_TIMEOUT = 300  # 5 minutes
    CACHE_KEY_PREFIX = 'subscription:'
    
    @classmethod
    def get_cache_key(cls, org_id):
        """Generate cache key for subscription status"""
        return f"{cls.CACHE_KEY_PREFIX}{org_id}"

    @classmethod
    def clear_subscription_cache(cls, org_id):
        """Clear subscription cache for an organization"""
        cache_key = cls.get_cache_key(org_id)
        cache.delete(cache_key)

    @classmethod
    def create_stripe_customer(cls, email, name=None, metadata=None):
        """Create a Stripe customer with error handling"""
        try:
            customer = stripe.Customer.create(
                email=email,
                name=name,
                metadata=metadata
            )
            logger.info(f"Created Stripe customer {customer.id} for {email}")
            return customer['id']
        except stripe.error.StripeError as e:
            logger.error(f"Error creating Stripe customer: {str(e)}")
            raise StripeError(f"Error creating customer: {str(e)}")

    @classmethod
    def create_subscription(cls, customer_id, price_id, trial_period_days=None):
        """Create a Stripe subscription with error handling"""
        try:
            subscription_data = {
                'customer': customer_id,
                'items': [{'price': price_id}],
                'payment_behavior': 'default_incomplete',
                'expand': ['latest_invoice.payment_intent'],
            }
            
            if trial_period_days:
                subscription_data['trial_period_days'] = trial_period_days
                
            subscription = stripe.Subscription.create(**subscription_data)
            logger.info(f"Created Stripe subscription {subscription.id} for customer {customer_id}")
            return subscription
        except stripe.error.StripeError as e:
            logger.error(f"Error creating Stripe subscription: {str(e)}")
            raise StripeError(f"Error creating subscription: {str(e)}")

    @classmethod
    def cancel_subscription(cls, subscription_id):
        """Cancel a Stripe subscription with error handling"""
        try:
            subscription = stripe.Subscription.delete(subscription_id)
            logger.info(f"Cancelled Stripe subscription {subscription_id}")
            return subscription
        except stripe.error.StripeError as e:
            logger.error(f"Error cancelling Stripe subscription: {str(e)}")
            raise StripeError(f"Error cancelling subscription: {str(e)}")

    @classmethod
    def get_subscription_status(cls, organization):
        """Get subscription status with caching"""
        cache_key = cls.get_cache_key(organization.id)
        cached_status = cache.get(cache_key)
        
        if cached_status is not None:
            return cached_status

        try:
            subscription = Subscription.objects.get(organization=organization)
            stripe_sub = stripe.Subscription.retrieve(subscription.stripe_subscription_id)
            
            status_data = {
                'status': stripe_sub['status'],
                'current_period_end': timezone.datetime.fromtimestamp(
                    stripe_sub['current_period_end'], 
                    tz=timezone.utc
                ),
                'plan': subscription.plan,
                'cancel_at_period_end': stripe_sub['cancel_at_period_end']
            }
            
            cache.set(cache_key, status_data, cls.CACHE_TIMEOUT)
            return status_data
        except Subscription.DoesNotExist:
            logger.warning(f"No subscription found for organization {organization.id}")
            return None
        except stripe.error.StripeError as e:
            logger.error(f"Error getting subscription status: {str(e)}")
            raise StripeError(f"Error getting subscription status: {str(e)}")

    @classmethod
    def handle_webhook_event(cls, event):
        """Handle Stripe webhook events"""
        try:
            if event.type == 'customer.subscription.updated':
                cls._handle_subscription_updated(event.data.object)
            elif event.type == 'customer.subscription.deleted':
                cls._handle_subscription_deleted(event.data.object)
            elif event.type == 'invoice.payment_succeeded':
                cls._handle_payment_succeeded(event.data.object)
            elif event.type == 'invoice.payment_failed':
                cls._handle_payment_failed(event.data.object)
            
            logger.info(f"Successfully processed webhook event {event.type}")
        except Exception as e:
            logger.error(f"Error processing webhook event {event.type}: {str(e)}")
            raise StripeError(f"Error processing webhook: {str(e)}")

    @classmethod
    def _handle_subscription_updated(cls, subscription_data):
        """Handle subscription update webhook"""
        try:
            subscription = Subscription.objects.get(
                stripe_subscription_id=subscription_data['id']
            )
            
            subscription.status = subscription_data['status']
            subscription.current_period_end = timezone.datetime.fromtimestamp(
                subscription_data['current_period_end'],
                tz=timezone.utc
            )
            subscription.save()
            
            # Update organization plan
            org = subscription.organization
            if subscription_data['status'] == 'active':
                org.plan = 'pro'
            else:
                org.plan = 'free'
            org.save()
            
            # Clear caches
            cls.clear_subscription_cache(org.id)
            from core.services.access_control import AccessControlService
            AccessControlService.clear_org_cache(org.id)
            
            logger.info(f"Updated subscription {subscription.id} for organization {org.id}")
        except Subscription.DoesNotExist:
            logger.error(f"Subscription {subscription_data['id']} not found")
            raise ValidationError("Subscription not found")

    @classmethod
    def _handle_subscription_deleted(cls, subscription_data):
        """Handle subscription deletion webhook"""
        try:
            subscription = Subscription.objects.get(
                stripe_subscription_id=subscription_data['id']
            )
            org = subscription.organization
            
            # Update organization plan
            org.plan = 'free'
            org.save()
            
            # Clear caches
            cls.clear_subscription_cache(org.id)
            from core.services.access_control import AccessControlService
            AccessControlService.clear_org_cache(org.id)
            
            logger.info(f"Deleted subscription {subscription.id} for organization {org.id}")
        except Subscription.DoesNotExist:
            logger.error(f"Subscription {subscription_data['id']} not found")
            raise ValidationError("Subscription not found")

    @classmethod
    def _handle_payment_succeeded(cls, invoice_data):
        """Handle successful payment webhook"""
        try:
            subscription = Subscription.objects.get(
                stripe_subscription_id=invoice_data['subscription']
            )
            logger.info(f"Payment succeeded for subscription {subscription.id}")
        except Subscription.DoesNotExist:
            logger.error(f"Subscription {invoice_data['subscription']} not found")
            raise ValidationError("Subscription not found")

    @classmethod
    def _handle_payment_failed(cls, invoice_data):
        """Handle failed payment webhook"""
        try:
            subscription = Subscription.objects.get(
                stripe_subscription_id=invoice_data['subscription']
            )
            logger.warning(f"Payment failed for subscription {subscription.id}")
        except Subscription.DoesNotExist:
            logger.error(f"Subscription {invoice_data['subscription']} not found")
            raise ValidationError("Subscription not found") 