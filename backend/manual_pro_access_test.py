import os
import django
from django.utils import timezone
from datetime import timedelta

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'financialhub.settings')
django.setup()

from accounts.models import User
from organizations.models import Organization
from accounts.access_control import has_pro_access


def print_result(test_name, result):
    print(f"{test_name}: {'OK' if result else 'FAIL'}")


def test_has_pro_access_global():
    User.objects.filter(username="pro_user_manual").delete()
    user = User.objects.create(username="pro_user_manual", pro_features=True, account_type="accountant")
    result = has_pro_access(user, feature="multi_org_panel") is True
    print_result("test_has_pro_access_global", result)
    user.delete()

def test_has_pro_access_trial():
    User.objects.filter(username="trial_user_manual").delete()
    user = User.objects.create(username="trial_user_manual", pro_trial_until=timezone.now() + timedelta(days=5), account_type="accountant")
    result = has_pro_access(user, feature="multi_org_panel") is True
    print_result("test_has_pro_access_trial", result)
    user.delete()

def test_has_pro_access_org_plan():
    User.objects.filter(username="org_user_manual").delete()
    Organization.objects.filter(name="Org Pro Manual").delete()
    user = User.objects.create(username="org_user_manual", account_type="accountant")
    org = Organization.objects.create(name="Org Pro Manual", plan="pro")
    result = has_pro_access(user, org, feature="multi_org_panel") is True
    print_result("test_has_pro_access_org_plan", result)
    user.delete()
    org.delete()

def test_has_pro_access_feature_list():
    User.objects.filter(username="feature_user_manual").delete()
    user = User.objects.create(username="feature_user_manual", pro_features_list=["multi_org_panel"], account_type="accountant")
    result = has_pro_access(user, feature="multi_org_panel") is True
    print_result("test_has_pro_access_feature_list", result)
    user.delete()

if __name__ == "__main__":
    test_has_pro_access_global()
    test_has_pro_access_trial()
    test_has_pro_access_org_plan()
    test_has_pro_access_feature_list() 