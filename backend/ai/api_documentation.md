# 🤖 API de Inteligencia Artificial - FinancialHub

## 📋 Descripción General

La API de IA de FinancialHub proporciona capacidades avanzadas de machine learning para análisis financiero, predicción, optimización y automatización. Esta API está diseñada para ser escalable, eficiente y fácil de integrar.

## 🚀 Características Principales

- **Análisis de Transacciones**: Clasificación automática y análisis de patrones
- **Predicción de Gastos**: Forecasting de gastos futuros
- **Optimización de Presupuestos**: Recomendaciones inteligentes
- **Detección de Anomalías**: Identificación de transacciones sospechosas
- **Análisis de Sentimientos**: Procesamiento de lenguaje natural
- **AutoML**: Optimización automática de modelos
- **Federated Learning**: Entrenamiento distribuido
- **A/B Testing**: Experimentación controlada

## 🔧 Configuración

### Autenticación
```python
# Headers requeridos
headers = {
    'Authorization': 'Bearer <your_token>',
    'Content-Type': 'application/json'
}
```

### Base URL
```
Development: http://localhost:8000/api/ai/
Production: https://your-domain.com/api/ai/
```

## 📊 Endpoints Principales

### 1. Análisis de Transacciones

#### `POST /analyze-transaction/`
Analiza una transacción específica y proporciona insights.

**Request:**
```json
{
    "transaction_id": 123,
    "include_sentiment": true,
    "include_anomaly_detection": true
}
```

**Response:**
```json
{
    "status": "success",
    "analysis": {
        "category_prediction": "Groceries",
        "confidence": 0.95,
        "sentiment": "neutral",
        "anomaly_score": 0.02,
        "risk_level": "low",
        "insights": [
            "Esta transacción sigue el patrón típico de gastos en alimentos",
            "El monto está dentro del rango esperado para esta categoría"
        ]
    }
}
```

#### `POST /classify-transactions/`
Clasifica múltiples transacciones de forma batch.

**Request:**
```json
{
    "transactions": [
        {"id": 123, "description": "Walmart", "amount": 150.00},
        {"id": 124, "description": "Gas Station", "amount": 45.00}
    ]
}
```

### 2. Predicción de Gastos

#### `POST /predict-expenses/`
Predice gastos futuros basado en datos históricos.

**Request:**
```json
{
    "organization_id": 1,
    "category_id": 5,
    "months_ahead": 3,
    "confidence_level": 0.95
}
```

**Response:**
```json
{
    "status": "success",
    "predictions": {
        "next_month": 1250.00,
        "next_2_months": 1180.00,
        "next_3_months": 1320.00
    },
    "confidence_intervals": {
        "next_month": [1100.00, 1400.00],
        "next_2_months": [1050.00, 1310.00],
        "next_3_months": [1200.00, 1440.00]
    },
    "trend": "increasing",
    "seasonality": "monthly"
}
```

#### `POST /predict-cash-flow/`
Predice el flujo de efectivo futuro.

**Request:**
```json
{
    "organization_id": 1,
    "months_ahead": 6,
    "include_scenarios": true
}
```

### 3. Optimización de Presupuestos

#### `POST /optimize-budget/`
Optimiza la distribución del presupuesto.

**Request:**
```json
{
    "organization_id": 1,
    "total_budget": 50000,
    "period": "monthly",
    "constraints": {
        "min_emergency_fund": 5000,
        "max_entertainment": 2000
    }
}
```

**Response:**
```json
{
    "status": "success",
    "optimized_budget": {
        "groceries": 8000,
        "transportation": 6000,
        "entertainment": 2000,
        "utilities": 4000,
        "emergency_fund": 5000,
        "savings": 25000
    },
    "efficiency_score": 0.92,
    "recommendations": [
        "Considera reducir gastos en entretenimiento",
        "Aumenta el fondo de emergencia"
    ]
}
```

### 4. Análisis de Comportamiento

#### `POST /analyze-behavior/`
Analiza patrones de comportamiento financiero.

**Request:**
```json
{
    "user_id": 123,
    "time_period": "90_days",
    "include_insights": true
}
```

**Response:**
```json
{
    "status": "success",
    "behavior_analysis": {
        "spending_patterns": {
            "primary_category": "Groceries",
            "spending_frequency": "weekly",
            "average_amount": 150.00
        },
        "savings_behavior": {
            "savings_rate": 0.25,
            "consistency_score": 0.85
        },
        "risk_profile": "conservative",
        "financial_health_score": 0.78
    },
    "insights": [
        "Tu gasto en alimentos es 15% mayor que el promedio",
        "Excelente consistencia en ahorros"
    ]
}
```

### 5. Detección de Anomalías

#### `POST /detect-anomalies/`
Detecta transacciones anómalas o sospechosas.

**Request:**
```json
{
    "organization_id": 1,
    "time_period": "30_days",
    "threshold": 0.8
}
```

**Response:**
```json
{
    "status": "success",
    "anomalies": [
        {
            "transaction_id": 456,
            "anomaly_score": 0.95,
            "reason": "Amount significantly higher than usual",
            "risk_level": "high",
            "recommended_action": "review"
        }
    ],
    "summary": {
        "total_anomalies": 3,
        "high_risk": 1,
        "medium_risk": 2
    }
}
```

### 6. Procesamiento de Lenguaje Natural

#### `POST /analyze-sentiment/`
Analiza el sentimiento de texto financiero.

**Request:**
```json
{
    "text": "La empresa reportó ganancias récord este trimestre",
    "method": "transformer"
}
```

**Response:**
```json
{
    "status": "success",
    "sentiment": {
        "score": 0.85,
        "label": "positive",
        "confidence": 0.92
    },
    "entities": [
        {"text": "empresa", "type": "organization"},
        {"text": "ganancias", "type": "financial_term"}
    ]
}
```

#### `POST /extract-entities/`
Extrae entidades financieras del texto.

**Request:**
```json
{
    "text": "Compré acciones de AAPL por $150.50",
    "entity_types": ["company", "amount", "currency"]
}
```

### 7. AutoML y Optimización

#### `POST /automl-optimize/`
Optimiza automáticamente modelos de ML.

**Request:**
```json
{
    "task_type": "classification",
    "data_config": {
        "features": ["amount", "category", "day_of_week"],
        "target": "fraud_detection"
    },
    "optimization_goal": "accuracy"
}
```

**Response:**
```json
{
    "status": "success",
    "optimization_result": {
        "best_model": "RandomForest",
        "best_score": 0.94,
        "parameters": {
            "n_estimators": 100,
            "max_depth": 10
        },
        "training_time": "45.2s"
    }
}
```

### 8. Federated Learning

#### `POST /federated-setup/`
Configura entrenamiento federado.

**Request:**
```json
{
    "task_type": "classification",
    "aggregation_method": "fedavg",
    "min_clients": 3,
    "rounds": 10
}
```

#### `POST /federated-train/`
Entrena modelo federado.

**Request:**
```json
{
    "client_id": "client_001",
    "data_size": 1000,
    "model_update": "base64_encoded_model"
}
```

### 9. A/B Testing

#### `POST /ab-testing/create/`
Crea experimento A/B.

**Request:**
```json
{
    "experiment_name": "new_budget_algorithm",
    "variants": ["control", "treatment"],
    "traffic_split": [0.5, 0.5],
    "metrics": ["conversion_rate", "user_satisfaction"]
}
```

#### `POST /ab-testing/assign/`
Asigna usuario a experimento.

**Request:**
```json
{
    "experiment_id": "exp_123",
    "user_id": 456
}
```

## 🔍 Endpoints de Monitoreo

### `GET /health/`
Verifica el estado del sistema de IA.

**Response:**
```json
{
    "status": "healthy",
    "models": {
        "transaction_classifier": "loaded",
        "expense_predictor": "loaded",
        "behavior_analyzer": "loaded"
    },
    "memory_usage": {
        "percent": 65.2,
        "available": "5.2GB"
    },
    "performance": {
        "average_response_time": "0.15s",
        "requests_per_minute": 120
    }
}
```

### `GET /models/status/`
Obtiene el estado de todos los modelos.

### `GET /metrics/`
Obtiene métricas de rendimiento.

### `POST /memory/cleanup/`
Limpia memoria del sistema.

## ⚠️ Códigos de Error

| Código | Descripción |
|--------|-------------|
| 400 | Bad Request - Datos inválidos |
| 401 | Unauthorized - Token inválido |
| 403 | Forbidden - Sin permisos |
| 404 | Not Found - Recurso no encontrado |
| 422 | Unprocessable Entity - Datos insuficientes |
| 429 | Too Many Requests - Rate limit excedido |
| 500 | Internal Server Error - Error del servidor |
| 503 | Service Unavailable - Servicio no disponible |

## 📈 Rate Limits

- **Standard**: 100 requests/minute
- **Premium**: 1000 requests/minute
- **Enterprise**: Sin límite

## 🔒 Seguridad

- **Autenticación**: JWT Bearer tokens
- **Autorización**: Basada en roles y organizaciones
- **Encriptación**: TLS 1.3 en tránsito
- **Validación**: Sanitización de inputs
- **Auditoría**: Logs de todas las operaciones

## 📝 Ejemplos de Uso

### Python
```python
import requests

# Configurar cliente
base_url = "https://api.financialhub.com/api/ai"
headers = {
    'Authorization': 'Bearer your_token_here',
    'Content-Type': 'application/json'
}

# Analizar transacción
response = requests.post(
    f"{base_url}/analyze-transaction/",
    headers=headers,
    json={"transaction_id": 123}
)

if response.status_code == 200:
    analysis = response.json()
    print(f"Categoría predicha: {analysis['analysis']['category_prediction']}")
```

### JavaScript
```javascript
// Configurar cliente
const baseUrl = 'https://api.financialhub.com/api/ai';
const headers = {
    'Authorization': 'Bearer your_token_here',
    'Content-Type': 'application/json'
};

// Predicción de gastos
async function predictExpenses(orgId, categoryId) {
    const response = await fetch(`${baseUrl}/predict-expenses/`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
            organization_id: orgId,
            category_id: categoryId,
            months_ahead: 3
        })
    });
    
    const result = await response.json();
    return result.predictions;
}
```

### cURL
```bash
# Analizar transacción
curl -X POST \
  https://api.financialhub.com/api/ai/analyze-transaction/ \
  -H 'Authorization: Bearer your_token_here' \
  -H 'Content-Type: application/json' \
  -d '{
    "transaction_id": 123,
    "include_sentiment": true
  }'
```

## 🚀 Optimizaciones de Rendimiento

### Lazy Loading
Los modelos se cargan solo cuando se necesitan, reduciendo el uso de memoria inicial.

### Caché Inteligente
- **Resultados**: Cacheados por 30 minutos
- **Modelos**: Cacheados en memoria
- **Predicciones**: Cacheadas por 1 hora

### Batch Processing
Para múltiples transacciones, usa endpoints batch para mejor rendimiento.

### Async Processing
Operaciones largas se procesan de forma asíncrona con webhooks.

## 📊 Métricas y Monitoreo

### Métricas Disponibles
- **Latencia**: Tiempo de respuesta promedio
- **Throughput**: Requests por segundo
- **Accuracy**: Precisión de predicciones
- **Memory Usage**: Uso de memoria
- **Error Rate**: Tasa de errores

### Alertas
- Uso de memoria > 80%
- Latencia > 2 segundos
- Error rate > 5%
- Modelos no disponibles

## 🔄 Versionado

La API usa versionado semántico (v1, v2, etc.). La versión actual es **v1**.

Para especificar versión:
```
https://api.financialhub.com/api/v1/ai/analyze-transaction/
```

## 📞 Soporte

- **Documentación**: https://docs.financialhub.com/api
- **GitHub**: https://github.com/financialhub/api
- **Email**: api-support@financialhub.com
- **Slack**: #api-support

---

**Última actualización**: 2025-01-21
**Versión de la API**: v1.0.0 