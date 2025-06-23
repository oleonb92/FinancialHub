#!/usr/bin/env python3
"""
Script para probar endpoints de IA con diferentes configuraciones
Permite probar con y sin autenticación para desarrollo y testing
"""

import os
import sys
import django
import requests
import json
from datetime import datetime

# Configurar Django
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'financialhub.settings')
django.setup()

from django.contrib.auth import get_user_model
from organizations.models import Organization
from transactions.models import Transaction

class AIEndpointTester:
    """Clase para probar endpoints de IA"""
    
    def __init__(self, base_url='http://localhost:8000', auth_required=False):
        self.base_url = base_url
        self.auth_required = auth_required
        self.session = requests.Session()
        self.token = None
        
        if auth_required:
            self._setup_authentication()
    
    def _setup_authentication(self):
        """Configurar autenticación"""
        try:
            # Crear usuario de prueba si no existe
            User = get_user_model()
            organization = Organization.objects.first()
            
            if not organization:
                organization = Organization.objects.create(
                    name='Test Organization',
                    slug='test-org'
                )
            
            user, created = User.objects.get_or_create(
                username='testuser',
                defaults={
                    'email': 'test@example.com',
                    'organization': organization
                }
            )
            
            if created:
                user.set_password('testpass123')
                user.save()
            
            # Obtener token
            login_url = f"{self.base_url}/api/auth/login/"
            login_data = {
                'username': 'testuser',
                'password': 'testpass123'
            }
            
            response = self.session.post(login_url, json=login_data)
            if response.status_code == 200:
                self.token = response.json().get('access')
                self.session.headers.update({
                    'Authorization': f'Bearer {self.token}'
                })
                print("✅ Autenticación configurada correctamente")
            else:
                print("❌ Error en autenticación")
                
        except Exception as e:
            print(f"❌ Error configurando autenticación: {e}")
    
    def test_endpoint(self, endpoint, method='GET', data=None, expected_status=200):
        """Probar un endpoint específico"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data)
            elif method.upper() == 'PUT':
                response = self.session.put(url, json=data)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url)
            else:
                print(f"❌ Método {method} no soportado")
                return False
            
            status_ok = response.status_code == expected_status
            status_icon = "✅" if status_ok else "❌"
            
            print(f"{status_icon} {method} {endpoint} - Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    if 'status' in result:
                        print(f"   📊 Status: {result['status']}")
                    if 'message' in result:
                        print(f"   💬 Message: {result['message']}")
                except:
                    pass
            
            return status_ok
            
        except Exception as e:
            print(f"❌ Error probando {endpoint}: {e}")
            return False
    
    def test_all_endpoints(self):
        """Probar todos los endpoints de IA"""
        print(f"\n🚀 Probando endpoints de IA en {self.base_url}")
        print(f"🔐 Autenticación requerida: {self.auth_required}")
        print("=" * 60)
        
        # Endpoints básicos
        endpoints = [
            # Health check (siempre público)
            ('/api/ai/health/', 'GET'),
            
            # Endpoints principales
            ('/api/ai/analyze-transaction/', 'POST', {
                'description': 'Test transaction',
                'amount': 100.00,
                'type': 'expense'
            }),
            ('/api/ai/predict-expenses/', 'POST', {'days_ahead': 30}),
            ('/api/ai/analyze-behavior/', 'POST', {'user_id': 1}),
            ('/api/ai/detect-anomalies/', 'POST', {}),
            ('/api/ai/optimize-budget/', 'POST', {
                'categories': ['food', 'transport'],
                'total_budget': 1000.00
            }),
            ('/api/ai/predict-cash-flow/', 'POST', {'months_ahead': 3}),
            ('/api/ai/analyze-risk/', 'POST', {}),
            ('/api/ai/recommendations/', 'POST', {'user_id': 1}),
            
            # Endpoints de modelos
            ('/api/ai/train-models/', 'POST', {}),
            ('/api/ai/models-status/', 'GET'),
            ('/api/ai/evaluate-models/', 'POST', {}),
            ('/api/ai/update-models/', 'POST', {'model_type': 'all'}),
            
            # Endpoints de monitoreo
            ('/api/ai/monitor-performance/', 'GET'),
            ('/api/ai/metrics/', 'GET'),
            ('/api/ai/config/', 'GET'),
            ('/api/ai/config/', 'POST', {'test_config': 'value'}),
            
            # Endpoints avanzados
            ('/api/ai/experiments/', 'POST', {
                'experiment_name': 'test',
                'parameters': {'test': 'value'}
            }),
            ('/api/ai/federated-learning/', 'POST', {
                'model_type': 'test',
                'data': {'test': 'data'}
            }),
            
            # Endpoints NLP
            ('/api/ai/nlp/analyze/', 'POST', {'text': 'Test transaction'}),
            ('/api/ai/nlp/sentiment/', 'POST', {'text': 'I am happy'}),
            ('/api/ai/nlp/extract/', 'POST', {'text': 'Paid $50 for groceries'}),
            
            # Endpoints AutoML
            ('/api/ai/automl/optimize/', 'POST', {
                'model_type': 'classifier',
                'dataset': {'features': [], 'target': []}
            }),
            ('/api/ai/automl/status/', 'GET'),
            
            # Endpoints A/B Testing
            ('/api/ai/ab-testing/', 'POST', {
                'test_name': 'test',
                'variants': ['A', 'B'],
                'metrics': ['accuracy']
            }),
            ('/api/ai/ab-testing/results/', 'GET'),
            
            # ViewSets
            ('/api/ai/interactions/', 'GET'),
            ('/api/ai/interactions/', 'POST', {
                'interaction_type': 'analysis',
                'input_data': {'test': 'data'},
                'output_data': {'result': 'test'}
            }),
            ('/api/ai/insights/', 'GET'),
            ('/api/ai/insights/', 'POST', {
                'insight_type': 'pattern',
                'description': 'Test insight',
                'confidence': 0.85
            }),
            ('/api/ai/predictions/', 'GET'),
            ('/api/ai/predictions/', 'POST', {
                'prediction_type': 'expense',
                'predicted_value': 150.00,
                'confidence': 0.90
            }),
        ]
        
        results = []
        for endpoint_info in endpoints:
            if len(endpoint_info) == 2:
                endpoint, method = endpoint_info
                data = None
            else:
                endpoint, method, data = endpoint_info
            
            success = self.test_endpoint(endpoint, method, data)
            results.append((endpoint, success))
        
        # Resumen
        print("\n" + "=" * 60)
        print("📊 RESUMEN DE PRUEBAS")
        print("=" * 60)
        
        successful = sum(1 for _, success in results if success)
        total = len(results)
        
        print(f"✅ Exitosos: {successful}/{total}")
        print(f"❌ Fallidos: {total - successful}/{total}")
        print(f"📈 Tasa de éxito: {(successful/total)*100:.1f}%")
        
        if total - successful > 0:
            print("\n❌ Endpoints fallidos:")
            for endpoint, success in results:
                if not success:
                    print(f"   - {endpoint}")
        
        return successful == total

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Probar endpoints de IA')
    parser.add_argument('--base-url', default='http://localhost:8000', 
                       help='URL base del servidor')
    parser.add_argument('--auth', action='store_true', 
                       help='Requerir autenticación')
    parser.add_argument('--no-auth', action='store_true', 
                       help='No requerir autenticación')
    
    args = parser.parse_args()
    
    # Determinar si usar autenticación
    auth_required = args.auth
    if not args.auth and not args.no_auth:
        # Por defecto, usar autenticación en producción
        auth_required = os.getenv('DEBUG', 'True') != 'True'
    
    # Crear tester y ejecutar pruebas
    tester = AIEndpointTester(args.base_url, auth_required)
    success = tester.test_all_endpoints()
    
    if success:
        print("\n🎉 ¡Todas las pruebas pasaron!")
        sys.exit(0)
    else:
        print("\n💥 Algunas pruebas fallaron")
        sys.exit(1)

if __name__ == '__main__':
    main() 