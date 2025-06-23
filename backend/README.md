# FinancialHub Backend

Sistema de gestión financiera empresarial con capacidades de IA avanzadas.

## Características Principales

### 🤖 Sistema de IA Integrado
- **Clasificación Automática de Transacciones**: Clasifica automáticamente las transacciones en categorías apropiadas
- **Predicción de Gastos**: Predice gastos futuros basándose en patrones históricos
- **Análisis de Comportamiento**: Analiza patrones de gasto y comportamiento financiero
- **Detección de Anomalías**: Identifica transacciones inusuales o sospechosas
- **Predicción de Flujo de Efectivo**: Predice el flujo de efectivo futuro
- **Análisis de Riesgo Personalizado**: Evalúa el riesgo financiero individual
- **Optimización de Presupuestos**: Optimiza la asignación de presupuestos entre categorías
- **Recomendaciones Personalizadas**: Genera recomendaciones financieras personalizadas

### 💰 Optimización de Presupuestos (NUEVO)
El sistema incluye un optimizador de presupuestos inteligente que:

- **Análisis de Patrones**: Analiza patrones de gasto históricos por categoría
- **Predicción de Necesidades**: Predice necesidades presupuestarias futuras
- **Optimización Automática**: Genera sugerencias de optimización automática
- **Análisis de Eficiencia**: Evalúa la eficiencia del presupuesto actual
- **Reasignación Inteligente**: Sugiere reasignación de presupuesto basada en datos
- **Insights Presupuestarios**: Genera insights y recomendaciones específicas

### 🔧 Funcionalidades Técnicas
- **Entrenamiento Automático**: Los modelos se entrenan automáticamente con datos recientes
- **Métricas de Rendimiento**: Seguimiento continuo del rendimiento de los modelos
- **Persistencia de Modelos**: Los modelos entrenados se guardan y cargan automáticamente
- **Escalabilidad**: Arquitectura preparada para escalar con más datos y usuarios

## Instalación

### Requisitos
- Python 3.8+
- Django 4.2+
- PostgreSQL
- Redis (para Celery)

### Configuración
```bash
# Clonar el repositorio
git clone <repository-url>
cd FinancialHub_backup/backend

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus configuraciones

# Ejecutar migraciones
python manage.py migrate

# Crear superusuario
python manage.py createsuperuser

# Entrenar modelos de IA
python manage.py train_ai_models

# Ejecutar el servidor
python manage.py runserver
```

## Uso del Sistema de IA

### Entrenamiento de Modelos
```bash
# Entrenar todos los modelos
python manage.py train_ai_models

# Entrenar con datos específicos
python manage.py train_ai_models --days 180 --user-id 1

# Forzar entrenamiento
python manage.py train_ai_models --force
```

### API de IA

#### Optimización de Presupuestos
```python
# Optimizar asignación de presupuesto
POST /api/ai/optimize-budget/
{
    "total_budget": 10000.0,
    "period": "2024-01"
}

# Analizar eficiencia presupuestaria
GET /api/ai/analyze-budget-efficiency/?period=2024-01

# Predecir necesidades presupuestarias
GET /api/ai/predict-budget-needs/?period=2024-01

# Obtener insights presupuestarios
GET /api/ai/get-budget-insights/?period=2024-01
```

#### Análisis de Transacciones
```python
# Analizar transacción
POST /api/ai/analyze-transaction/
{
    "transaction_id": 123
}

# Predecir gastos
POST /api/ai/predict-expenses/
{
    "category_id": 1,
    "start_date": "2024-01-01",
    "end_date": "2024-01-31"
}
```

#### Análisis de Comportamiento
```python
# Analizar comportamiento
GET /api/ai/analyze-behavior/

# Obtener recomendaciones
GET /api/ai/recommendations/

# Analizar riesgo
GET /api/ai/analyze-risk/
```

### Uso Programático

#### BudgetOptimizer
```python
from ai.ml.optimizers.budget_optimizer import BudgetOptimizer

# Inicializar optimizador
optimizer = BudgetOptimizer()

# Entrenar con datos históricos
transactions = Transaction.objects.filter(organization=org)
optimizer.train(transactions)

# Optimizar presupuesto
result = optimizer.optimize_budget_allocation(org_id, 10000.0)

# Analizar eficiencia
efficiency = optimizer.analyze_budget_efficiency(org_id)

# Predecir necesidades
predictions = optimizer.predict_budget_needs(org_id)
```

#### Servicio de IA Unificado
```python
from ai.services import AIService

# Inicializar servicio
ai_service = AIService()

# Optimizar presupuesto
optimization = ai_service.optimize_budget(org_id, 10000.0)

# Obtener insights
insights = ai_service.get_budget_insights(org_id)

# Analizar transacción
analysis = ai_service.analyze_transaction(transaction)
```

## Estructura del Proyecto

```
backend/
├── ai/                          # Sistema de IA
│   ├── ml/                      # Modelos de Machine Learning
│   │   ├── optimizers/          # Optimizadores
│   │   │   └── budget_optimizer.py  # Optimizador de presupuestos
│   │   ├── classifiers/         # Clasificadores
│   │   ├── predictors/          # Predictores
│   │   └── analyzers/           # Analizadores
│   ├── services.py              # Servicio unificado de IA
│   ├── views.py                 # Vistas de la API
│   └── management/              # Comandos de gestión
├── transactions/                # Gestión de transacciones
├── organizations/               # Gestión de organizaciones
└── config/                      # Configuración
```

## Características del BudgetOptimizer

### Funcionalidades Principales
1. **Análisis de Patrones Históricos**: Analiza patrones de gasto de los últimos 6 meses
2. **Factores Estacionales**: Considera variaciones estacionales en el gasto
3. **Predicción de Tendencias**: Identifica tendencias de gasto por categoría
4. **Optimización de Asignación**: Distribuye el presupuesto de manera óptima
5. **Análisis de Eficiencia**: Evalúa qué tan bien se está usando el presupuesto
6. **Generación de Insights**: Proporciona recomendaciones específicas

### Algoritmos Utilizados
- **Gradient Boosting Regressor**: Para predicción de gastos
- **Random Forest Regressor**: Para análisis de eficiencia
- **Standard Scaler**: Para normalización de features
- **Análisis Temporal**: Para patrones estacionales y tendencias

### Métricas de Eficiencia
- **Utilización del Presupuesto**: Qué tan bien se usa el presupuesto asignado
- **Eficiencia por Categoría**: Análisis individual por categoría
- **Tendencias de Gasto**: Comparación con períodos anteriores
- **Predicciones de Necesidades**: Estimaciones para períodos futuros

## Monitoreo y Métricas

### Métricas de Modelos
```python
# Obtener métricas de un modelo
GET /api/ai/get-model-metrics/?model_name=budget_optimizer

# Exportar métricas
GET /api/ai/export-metrics/?model_name=budget_optimizer&format=json
```

### Logs y Monitoreo
- Los modelos registran su rendimiento automáticamente
- Métricas de precisión, recall y F1-score
- Seguimiento de tendencias de rendimiento
- Alertas automáticas para degradación de rendimiento

## Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Soporte

Para soporte técnico o preguntas sobre el sistema de IA, contacta al equipo de desarrollo o consulta la documentación técnica detallada. 