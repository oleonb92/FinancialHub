from rest_framework.exceptions import APIException
from rest_framework import status

class BaseFinancialHubException(APIException):
    """Base exception for all FinancialHub custom exceptions"""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    default_detail = 'An unexpected error occurred.'
    default_code = 'error'

class SubscriptionError(BaseFinancialHubException):
    """Exception for subscription-related errors"""
    status_code = status.HTTP_402_PAYMENT_REQUIRED
    default_detail = 'Subscription error occurred.'
    default_code = 'subscription_error'

class AccessControlError(BaseFinancialHubException):
    """Exception for access control related errors"""
    status_code = status.HTTP_403_FORBIDDEN
    default_detail = 'Access denied.'
    default_code = 'access_denied'

class StripeError(BaseFinancialHubException):
    """Exception for Stripe-related errors"""
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = 'Stripe operation failed.'
    default_code = 'stripe_error'

class OrganizationError(BaseFinancialHubException):
    """Exception for organization-related errors"""
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = 'Organization operation failed.'
    default_code = 'organization_error'

class ValidationError(BaseFinancialHubException):
    """Exception for validation errors"""
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = 'Validation error occurred.'
    default_code = 'validation_error' 