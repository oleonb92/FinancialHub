import stripe
from django.conf import settings
from payments.models import Subscription
from organizations.models import Organization
from django.utils import timezone
from django.core.exceptions import ValidationError

stripe.api_key = settings.STRIPE_SECRET_KEY

def create_stripe_customer(email, name=None, metadata=None):
    try:
        customer = stripe.Customer.create(
            email=email,
            name=name,
            metadata=metadata
        )
        return customer['id']
    except stripe.error.StripeError as e:
        raise ValidationError(f"Error creating Stripe customer: {str(e)}")

def create_stripe_subscription(customer_id, price_id, trial_period_days=None):
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
        return subscription
    except stripe.error.StripeError as e:
        raise ValidationError(f"Error creating Stripe subscription: {str(e)}")

def cancel_stripe_subscription(subscription_id):
    try:
        subscription = stripe.Subscription.delete(subscription_id)
        return subscription
    except stripe.error.StripeError as e:
        raise ValidationError(f"Error canceling Stripe subscription: {str(e)}")

def get_stripe_subscription(subscription_id):
    try:
        subscription = stripe.Subscription.retrieve(subscription_id)
        return subscription
    except stripe.error.StripeError as e:
        raise ValidationError(f"Error retrieving Stripe subscription: {str(e)}")

def update_stripe_subscription(subscription_id, price_id):
    try:
        subscription = stripe.Subscription.retrieve(subscription_id)
        subscription = stripe.Subscription.modify(
            subscription_id,
            items=[{
                'id': subscription['items']['data'][0]['id'],
                'price': price_id,
            }]
        )
        return subscription
    except stripe.error.StripeError as e:
        raise ValidationError(f"Error updating Stripe subscription: {str(e)}")

def attach_subscription_to_org(org: Organization, stripe_customer_id, stripe_subscription_id, plan, status, current_period_end):
    try:
        subscription, created = Subscription.objects.update_or_create(
            organization=org,
            defaults={
                'stripe_customer_id': stripe_customer_id,
                'stripe_subscription_id': stripe_subscription_id,
                'plan': plan,
                'status': status,
                'current_period_end': current_period_end,
            }
        )
        return subscription
    except Exception as e:
        raise ValidationError(f"Error attaching subscription to organization: {str(e)}")

def get_subscription_status(organization):
    try:
        subscription = Subscription.objects.get(organization=organization)
        stripe_sub = get_stripe_subscription(subscription.stripe_subscription_id)
        return {
            'status': stripe_sub['status'],
            'current_period_end': timezone.datetime.fromtimestamp(stripe_sub['current_period_end'], tz=timezone.utc),
            'plan': subscription.plan,
            'cancel_at_period_end': stripe_sub['cancel_at_period_end']
        }
    except Subscription.DoesNotExist:
        return None
    except stripe.error.StripeError as e:
        raise ValidationError(f"Error getting subscription status: {str(e)}") 