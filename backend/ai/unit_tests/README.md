# Tests de IA - FinancialHub

Este directorio contiene todos los tests para el sistema de IA de FinancialHub, incluyendo tests unitarios, de integración y end-to-end.

## 🏗️ Estructura de Tests

```
ai/tests/
├── README.md                           # Este archivo
├── test_complete_ai_system.py         # Tests completos del sistema
├── unit/                              # Tests unitarios
│   ├── test_expense_predictor.py
│   ├── test_transaction_classifier.py
│   ├── test_behavior_analyzer.py
│   └── test_budget_optimizer.py
├── integration/                       # Tests de integración
│   └── test_model_training.py
└── manual/                           # Tests manuales
    └── test_pro_access.py
```

## 🚀 Configuraciones de Testing

### 1. **Configuración de Test (Recomendada)**
```bash
# Sin autenticación - Ideal para desarrollo y CI/CD
python scripts/run_ai_tests.py --config test
```

### 2. **Configuración de Desarrollo**
```bash
# Sin autenticación - Para desarrollo local
python scripts/run_ai_tests.py --config dev
```

### 3. **Configuración de Producción**
```bash
# Con autenticación - Para simular producción
python scripts/run_ai_tests.py --config prod
```

## 🛠️ Scripts Disponibles

### Script Principal de Tests
```bash
# Ejecutar todos los tests
python scripts/run_ai_tests.py --all

# Ejecutar tests específicos
python scripts/run_ai_tests.py --specific

# Ejecutar tests con cobertura
python scripts/run_ai_tests.py --coverage

# Ejecutar tests de rendimiento
python scripts/run_ai_tests.py --performance

# Ejecutar tests con patrón específico
python scripts/run_ai_tests.py --pattern "test_analyze"

# Modo verbose
python scripts/run_ai_tests.py --verbose
```

### Script de Prueba de Endpoints
```bash
# Probar endpoints sin autenticación
python scripts/test_ai_endpoints.py --no-auth

# Probar endpoints con autenticación
python scripts/test_ai_endpoints.py --auth

# Probar en servidor específico
python scripts/test_ai_endpoints.py --base-url http://localhost:8000
```

## 🔧 Configuración de Autenticación

### Variable de Entorno
```bash
# Habilitar autenticación en tests
export AI_TEST_ENDPOINTS_AUTH=True

# Deshabilitar autenticación en tests
export AI_TEST_ENDPOINTS_AUTH=False
```

### Configuración en Settings
```python
# settings.py
AI_TEST_ENDPOINTS_AUTH = os.getenv('AI_TEST_ENDPOINTS_AUTH', 'False') == 'True'
```

### Lógica de Permisos
```python
def get_ai_permissions():
    """Obtener permisos según el entorno"""
    if getattr(settings, 'AI_TEST_ENDPOINTS_AUTH', False):
        return [IsAuthenticated]
    return [AllowAny] if settings.DEBUG else [IsAuthenticated]
```

## 📊 Tipos de Tests

### 1. **Tests Unitarios**
- **Ubicación**: `ai/tests/unit/`
- **Propósito**: Probar componentes individuales
- **Ejecución**: `pytest ai/tests/unit/`

### 2. **Tests de Integración**
- **Ubicación**: `ai/tests/integration/`
- **Propósito**: Probar interacciones entre componentes
- **Ejecución**: `pytest ai/tests/integration/`

### 3. **Tests Completos del Sistema**
- **Archivo**: `test_complete_ai_system.py`
- **Propósito**: Probar todo el sistema end-to-end
- **Ejecución**: `pytest ai/tests/test_complete_ai_system.py`

### 4. **Tests de Rendimiento**
- **Marcador**: `@pytest.mark.performance`
- **Propósito**: Probar rendimiento y escalabilidad
- **Ejecución**: `pytest -m performance`

## 🎯 Casos de Uso por Entorno

### **Desarrollo Local**
```bash
# Configuración recomendada para desarrollo
export AI_TEST_ENDPOINTS_AUTH=False
export DEBUG=True
python scripts/run_ai_tests.py --config dev --verbose
```

### **CI/CD Pipeline**
```bash
# Configuración para integración continua
export AI_TEST_ENDPOINTS_AUTH=False
python scripts/run_ai_tests.py --config test --coverage
```

### **Testing de Producción**
```bash
# Simular entorno de producción
export AI_TEST_ENDPOINTS_AUTH=True
export DEBUG=False
python scripts/run_ai_tests.py --config prod
```

### **Testing Manual**
```bash
# Probar endpoints manualmente
python scripts/test_ai_endpoints.py --base-url http://localhost:8000 --no-auth
```

## 🔍 Endpoints de Test

### **Endpoints Públicos (Siempre Accesibles)**
- `GET /api/ai/health/` - Health check del sistema

### **Endpoints con Autenticación Flexible**
- `POST /api/ai/analyze-transaction/` - Análisis de transacciones
- `POST /api/ai/predict-expenses/` - Predicción de gastos
- `POST /api/ai/analyze-behavior/` - Análisis de comportamiento
- `POST /api/ai/detect-anomalies/` - Detección de anomalías
- `POST /api/ai/optimize-budget/` - Optimización de presupuesto
- `POST /api/ai/predict-cash-flow/` - Predicción de flujo de efectivo
- `POST /api/ai/analyze-risk/` - Análisis de riesgo
- `POST /api/ai/recommendations/` - Recomendaciones

### **Endpoints de Gestión de Modelos**
- `POST /api/ai/train-models/` - Entrenar modelos
- `GET /api/ai/models-status/` - Estado de modelos
- `POST /api/ai/evaluate-models/` - Evaluar modelos
- `POST /api/ai/update-models/` - Actualizar modelos

### **Endpoints de Monitoreo**
- `GET /api/ai/monitor-performance/` - Monitorear rendimiento
- `GET /api/ai/metrics/` - Métricas de IA
- `GET /api/ai/config/` - Configuración de IA
- `POST /api/ai/config/` - Actualizar configuración

## 🚨 Troubleshooting

### **Error: "No such file or directory (originated from sysctl(HW_CPU_FREQ))"**
```bash
# Este error es normal en macOS, no afecta la funcionalidad
# Se puede ignorar o suprimir en logs
```

### **Error: "AbstractConnection.__init__() got an unexpected keyword argument 'CLIENT_CLASS'"**
```bash
# Error de Redis, verificar configuración
export REDIS_URL=redis://localhost:6379/0
```

### **Error: "'AIService' object has no attribute 'get_models'"**
```bash
# Verificar que el servicio de IA esté actualizado
# Ejecutar migraciones si es necesario
python manage.py migrate
```

### **Tests Fallan con Autenticación**
```bash
# Usar configuración sin autenticación para tests
export AI_TEST_ENDPOINTS_AUTH=False
python scripts/run_ai_tests.py --config test
```

## 📈 Métricas de Cobertura

### **Objetivo de Cobertura**
- **Mínimo**: 80%
- **Recomendado**: 90%
- **Ideal**: 95%

### **Generar Reporte de Cobertura**
```bash
python scripts/run_ai_tests.py --coverage
```

### **Ver Reporte HTML**
```bash
open htmlcov/index.html
```

## 🔄 Workflow de Testing

### **1. Desarrollo Diario**
```bash
# Ejecutar tests rápidos
python scripts/run_ai_tests.py --config test --pattern "test_analyze"
```

### **2. Antes de Commit**
```bash
# Ejecutar todos los tests
python scripts/run_ai_tests.py --all
```

### **3. En CI/CD**
```bash
# Tests con cobertura
python scripts/run_ai_tests.py --config test --coverage
```

### **4. Antes de Deploy**
```bash
# Simular producción
python scripts/run_ai_tests.py --config prod
```

## 📝 Mejores Prácticas

### **1. Nomenclatura de Tests**
```python
def test_functionality_description():
    """Descripción clara del test"""
    pass
```

### **2. Organización de Tests**
```python
class TestComponentName(TestCase):
    def setUp(self):
        """Configuración común"""
        pass
    
    def test_specific_functionality(self):
        """Test específico"""
        pass
```

### **3. Manejo de Datos de Test**
```python
# Usar factories o fixtures
from django.test import TestCase
from model_bakery import baker

class TestExample(TestCase):
    def setUp(self):
        self.user = baker.make('accounts.User')
        self.organization = baker.make('organizations.Organization')
```

### **4. Mocking de Servicios Externos**
```python
from unittest.mock import patch

@patch('ai.services.AIService.external_api_call')
def test_with_mock(self, mock_api):
    mock_api.return_value = {'result': 'success'}
    # Test implementation
```

## 🎯 Próximos Pasos

1. **Agregar más tests de rendimiento**
2. **Implementar tests de carga**
3. **Agregar tests de seguridad**
4. **Mejorar cobertura de código**
5. **Automatizar tests en CI/CD**

## 📞 Soporte

Para problemas con tests:
1. Revisar logs en `logs/test/`
2. Verificar configuración de entorno
3. Ejecutar tests con `--verbose`
4. Consultar documentación de Django/Pytest 