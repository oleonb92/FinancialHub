# FinancialHub - Sistema de Gestión Financiera

Un sistema completo de gestión financiera personal y empresarial con capacidades de IA y análisis avanzado.

## 🚀 Características Principales

- **Gestión de transacciones** con categorización automática
- **Análisis de gastos** con predicciones de IA
- **Sistema de presupuestos** inteligente
- **Búsqueda avanzada** con Elasticsearch
- **Tareas asíncronas** con Celery + Redis
- **API REST** completa con Django REST Framework
- **Frontend React** moderno y responsivo con tecnologías avanzadas
- **Sistema de pagos** con Stripe
- **Sistema de usuarios** y organizaciones
- **Auditoría completa** de cambios
- **Notificaciones** en tiempo real
- **Monitoreo avanzado** de recursos del sistema

## 🏗️ Arquitectura del Sistema

### Backend (Django) - Sistema Principal
El backend Django es el núcleo del sistema, proporcionando todas las funcionalidades principales:

#### **Framework y Tecnologías**
- **Framework**: Django 5.1.9 con Django REST Framework
- **Autenticación**: JWT (JSON Web Tokens)
- **Base de datos**: PostgreSQL/MySQL con ORM de Django
- **Búsqueda**: Elasticsearch para búsqueda avanzada
- **Cache**: Redis para caché y sesiones
- **Tareas asíncronas**: Celery + Redis para procesamiento en background
- **API**: REST API completa con documentación automática

#### **Módulos Principales**
1. **accounts/**: Gestión de usuarios, autenticación y permisos
2. **transactions/**: Gestión completa de transacciones financieras
3. **ai/**: Sistema de Machine Learning y IA
4. **organizations/**: Gestión de organizaciones y multi-tenancy
5. **goals/**: Sistema de metas financieras
6. **budgets/**: Gestión de presupuestos
7. **payments/**: Integración con Stripe para pagos
8. **notifications/**: Sistema de notificaciones
9. **audit/**: Auditoría completa de cambios

#### **Sistema de IA (Módulo ai/)**
El módulo de IA incluye:
- **Clasificadores**: Categorización automática de transacciones
- **Predictores**: Predicción de gastos futuros
- **Analizadores**: Análisis de comportamiento financiero
- **Detectores**: Detección de anomalías
- **Optimizadores**: Optimización de presupuestos
- **NLP**: Procesamiento de lenguaje natural para descripciones

### Frontend (React) - Interfaz Web Avanzada
- **Framework**: React 18 con hooks modernos
- **Estilos**: Tailwind CSS para diseño responsivo
- **Estado**: Context API para gestión de estado global
- **HTTP**: Axios para comunicación con la API
- **Componentes**: Sistema modular de componentes reutilizables
- **Visualización**: Chart.js, D3.js, Recharts para gráficos avanzados
- **UI/UX**: Material-UI, Ant Design para componentes modernos
- **Tiempo Real**: WebSockets para actualizaciones en vivo
- **Estado Avanzado**: React Query para caché y sincronización

## 🔄 Cómo Funciona el Sistema

### 1. **Flujo de Datos Principal**

```
Usuario → Frontend React → API Django → Base de Datos
    ↓
Sistema IA → Procesamiento → Predicciones
    ↓
Celery Tasks → Procesamiento Asíncrono → Notificaciones
```

### 2. **Procesamiento de Transacciones**

1. **Entrada de Datos**: Usuario ingresa transacción vía React
2. **Validación**: Django valida datos y permisos
3. **Clasificación IA**: Sistema de ML categoriza automáticamente
4. **Almacenamiento**: Transacción se guarda en PostgreSQL
5. **Indexación**: Elasticsearch indexa para búsqueda
6. **Análisis**: Celery ejecuta análisis en background
7. **Notificación**: Usuario recibe confirmación
8. **Actualización UI**: React actualiza interfaz en tiempo real

### 3. **Sistema de IA en Acción**

#### **Entrenamiento de Modelos**
```python
# Los modelos se entrenan automáticamente con:
- Datos históricos de transacciones
- Patrones de comportamiento del usuario
- Categorías y etiquetas existentes
- Métricas de rendimiento continuas
```

#### **Predicciones en Tiempo Real**
```python
# El sistema predice:
- Categoría de nueva transacción
- Gastos futuros basados en patrones
- Anomalías en transacciones
- Recomendaciones de presupuesto
```

### 4. **Procesamiento Asíncrono (Celery)**

#### **Tareas Automáticas**
- **Entrenamiento de modelos**: Cada 24 horas
- **Análisis de comportamiento**: Cada 6 horas
- **Monitoreo de recursos**: Cada hora
- **Limpieza de datos**: Diariamente
- **Generación de reportes**: Semanalmente

#### **Colas de Procesamiento**
- **training**: Entrenamiento de modelos de IA
- **evaluation**: Evaluación de rendimiento
- **monitoring**: Monitoreo del sistema
- **maintenance**: Mantenimiento automático

## 📦 Instalación y Configuración

### Prerrequisitos
- Python 3.9+
- Node.js 16+
- Redis
- PostgreSQL/MySQL
- Elasticsearch

### Instalación Completa

#### **1. Backend Django**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r ../requirements.txt
python manage.py migrate
python manage.py createsuperuser
```

#### **2. Frontend React**
```bash
cd frontend
npm install
npm start
```

#### **3. Servicios de Soporte**
```bash
# Redis
redis-server

# Elasticsearch
elasticsearch

# Celery Worker
cd backend
celery -A financialhub worker -l info

# Celery Beat (scheduler)
cd backend
celery -A financialhub beat -l info
```

### Configuración de Variables de Entorno
Crea un archivo `.env` en el directorio `backend`:

```env
DEBUG=True
SECRET_KEY=your-secret-key
DATABASE_URL=postgresql://user:pass@localhost:5432/financialhub
REDIS_URL=redis://localhost:6379/0
ELASTICSEARCH_URL=http://localhost:9200
OPENAI_API_KEY=your-openai-key
STRIPE_SECRET_KEY=sk_test_...
STRIPE_PUBLISHABLE_KEY=pk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
```

## 🚀 Uso del Sistema

### Iniciar el Sistema Completo
```bash
# Terminal 1: Backend Django
cd backend
python manage.py runserver

# Terminal 2: Frontend React
cd frontend
npm start

# Terminal 3: Celery Worker
cd backend
celery -A financialhub worker -l info

# Terminal 4: Celery Beat
cd backend
celery -A financialhub beat -l info
```

### Acceso a las Interfaces

#### **Frontend React**
- **URL**: http://localhost:3000
- **Funcionalidades**: Gestión completa de transacciones, usuarios, presupuestos
- **Características**: Interfaz moderna, responsiva, tiempo real, visualizaciones avanzadas

#### **API Django**
- **URL**: http://localhost:8000/api/
- **Documentación**: http://localhost:8000/api/docs/
- **Funcionalidades**: Endpoints REST para integración

### Verificar Estado del Sistema
```bash
# Verificar conexión de Celery
celery -A financialhub inspect ping

# Ver tareas registradas
celery -A financialhub inspect registered

# Ver colas activas
celery -A financialhub inspect active_queues

# Verificar logs
tail -f backend/logs/financialhub.log
```

## 📊 Monitoreo y Mantenimiento

### Monitoreo de Celery
- **Flower**: Monitor web para Celery (puerto 5555)
- **Comandos de inspección**: `celery -A financialhub inspect`
- **Monitoreo automático**: Tareas de monitoreo de recursos cada hora

### Monitoreo del Sistema
- **Logs**: `backend/logs/financialhub.log`
- **Métricas**: Prometheus endpoints
- **Health checks**: `/health/` endpoint
- **Monitoreo de recursos**: CPU, memoria, disco, red

### Monitoreo de IA
- **Rendimiento de modelos**: Seguimiento de métricas
- **Drift detection**: Detección de cambios en datos
- **Resource monitoring**: Monitoreo de recursos del sistema
- **Alertas**: Notificaciones automáticas

## 🧪 Testing y Desarrollo

### Ejecutar Tests
```bash
# Backend tests
cd backend
python manage.py test

# Frontend tests
cd frontend
npm test

# Coverage
cd backend
coverage run --source='.' manage.py test
coverage report
```

### Herramientas de Desarrollo
```bash
# Generar diagrama ER de la base de datos
python tools/generate_erd.py

# Insertar datos de prueba
python tools/seed_data.py

# Instalar dependencias avanzadas
python scripts/install_advanced_deps.py

# Configurar IA avanzada
python scripts/setup_advanced_ai.py
```

## 📁 Estructura Detallada del Proyecto

```
FinancialHub/
├── backend/                 # Django backend (Sistema Principal)
│   ├── financialhub/       # Configuración principal
│   ├── accounts/          # Gestión de usuarios y autenticación
│   ├── transactions/      # Gestión de transacciones
│   ├── ai/               # Sistema de Machine Learning
│   │   ├── ml/          # Modelos de ML
│   │   │   ├── classifiers/    # Clasificadores
│   │   │   ├── predictors/     # Predictores
│   │   │   ├── analyzers/      # Analizadores
│   │   │   └── optimizers/     # Optimizadores
│   │   ├── tasks/       # Tareas Celery
│   │   └── utils/       # Utilidades de ML
│   ├── organizations/    # Gestión de organizaciones
│   ├── goals/           # Metas financieras
│   ├── budgets/         # Presupuestos
│   ├── payments/        # Integración con Stripe
│   ├── notifications/   # Sistema de notificaciones
│   ├── audit/          # Auditoría de cambios
│   └── scripts/         # Scripts de sistema
├── frontend/             # React frontend (Interfaz Web Avanzada)
│   ├── src/
│   │   ├── components/   # Componentes React
│   │   ├── pages/       # Páginas
│   │   ├── hooks/       # Custom hooks
│   │   ├── charts/      # Componentes de visualización
│   │   └── utils/       # Utilidades
│   └── public/
├── requirements.txt    # Dependencias Python
└── docs/               # Documentación
```

## 🔍 Funcionalidades Avanzadas

### Búsqueda con Elasticsearch
- Búsqueda por texto libre en transacciones
- Filtros avanzados por fecha, categoría, monto
- Búsqueda semántica y fuzzy matching
- Historial de búsquedas personalizado

### Sistema de IA Completo
- **Clasificador de transacciones**: Categorización automática
- **Predictor de gastos**: Predicción de gastos futuros
- **Analizador de comportamiento**: Análisis de patrones
- **Motor de recomendaciones**: Sugerencias inteligentes
- **Detector de anomalías**: Detección de transacciones inusuales
- **Optimizador de presupuestos**: Optimización automática

### Visualizaciones Avanzadas en React
- **Gráficos interactivos**: Chart.js, D3.js, Recharts
- **Dashboards en tiempo real**: Actualizaciones automáticas
- **Métricas clave**: KPIs visuales y dinámicos
- **Análisis de tendencias**: Gráficos de líneas y barras
- **Comparativas**: Gráficos de comparación entre períodos
- **Mapas de calor**: Visualización de patrones de gastos

### Sistema de Pagos (Stripe)
- **Suscripciones**: Gestión de planes y suscripciones
- **Pagos**: Procesamiento de pagos seguros
- **Webhooks**: Eventos en tiempo real
- **Facturación**: Generación automática de facturas

## 🔧 Mantenimiento y Actualizaciones

### Comandos de Mantenimiento
```bash
# Verificar estado del sistema
cd backend
python manage.py check

# Limpiar cache
python manage.py clearcache

# Verificar logs
tail -f logs/financialhub.log

# Reiniciar servicios
pkill -f celery
pkill -f redis-server
```

### Actualizaciones Recientes
- **Corrección de errores de Celery**: Solucionados problemas de conexión y monitoreo
- **Mejora del sistema de monitoreo**: Monitoreo robusto de recursos del sistema
- **Actualización de dependencias**: Todas las dependencias actualizadas a versiones estables
- **Optimización de rendimiento**: Mejoras en el procesamiento de tareas asíncronas
- **Eliminación de Streamlit**: Simplificación de arquitectura con React avanzado

## 📈 Roadmap y Futuras Mejoras

- [x] Sistema de monitoreo avanzado
- [x] Corrección de errores de Celery
- [x] Actualización de dependencias
- [x] Eliminación de Streamlit (simplificación)
- [ ] Dashboard avanzado con métricas en tiempo real
- [ ] Integración con bancos (Open Banking)
- [ ] Análisis de inversiones
- [ ] Reportes personalizados
- [ ] API pública para desarrolladores
- [ ] App móvil nativa

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🆘 Soporte

Para soporte técnico o preguntas:
- Crear un issue en GitHub
- Revisar la documentación en `/docs`
- Verificar los logs en `backend/logs/`
- Consultar el estado de Celery con `celery -A financialhub inspect`


