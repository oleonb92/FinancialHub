import pytest
from django.utils import timezone
from accounts.models import User
from organizations.models import Organization
from accounts.access_control import has_pro_access
from datetime import timedelta

@pytest.mark.django_db
def test_has_pro_access_global():
    user = User.objects.create(username="pro_user", pro_features=True, account_type="accountant")
    assert has_pro_access(user, feature="multi_org_panel") is True

@pytest.mark.django_db
def test_has_pro_access_trial():
    user = User.objects.create(username="trial_user", pro_trial_until=timezone.now() + timedelta(days=5), account_type="accountant")
    assert has_pro_access(user, feature="multi_org_panel") is True

@pytest.mark.django_db
def test_has_pro_access_org_plan():
    user = User.objects.create(username="org_user", account_type="accountant")
    org = Organization.objects.create(name="Org Pro", plan="pro")
    assert has_pro_access(user, org, feature="multi_org_panel") is True

@pytest.mark.django_db
def test_has_pro_access_feature_list():
    user = User.objects.create(username="feature_user", pro_features_list=["multi_org_panel"], account_type="accountant")
    assert has_pro_access(user, feature="multi_org_panel") is True 