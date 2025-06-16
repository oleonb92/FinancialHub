# FinancialHub

**FinancialHub** es un sistema integral de gestión financiera personal y empresarial, robusto, seguro y preparado para producción, con todos los servicios críticos orquestados en Docker.

---

## **Características principales**

- Gestión de usuarios, households y empresas
- Cuentas y transacciones
- Etiquetas y categorías
- Tareas asíncronas con Celery + RabbitMQ
- Cache y sesiones con Redis
- Búsqueda avanzada con Elasticsearch
- Monitoreo con Prometheus y Grafana
- Logs y errores con Sentry
- Backups automáticos de base de datos y media
- Documentación Swagger/OpenAPI
- CI/CD con GitHub Actions
- Seguridad y buenas prácticas (CORS, CSRF, HTTPS, .env centralizado)
- Listo para escalar y servir estáticos/media con Nginx

---

## **Estructura del proyecto**

```
/backend         # Código Django
/frontend        # Frontend (React, etc.)
/docker
  ├── nginx/     # Configuración Nginx
  ├── certbot/   # Let's Encrypt (opcional)
docker-compose.yml
docker-compose.prod.yml
.env             # Variables de entorno (dev)
.env.prod        # Variables de entorno (prod)
```

---

## **Servicios principales (Docker Compose Prod)**

- **web**: Django + Gunicorn
- **nginx**: Servidor web/proxy, sirve static/media y API
- **db**: PostgreSQL
- **redis**: Cache y backend Celery
- **rabbitmq**: Broker de tareas Celery
- **elasticsearch**: Búsqueda avanzada
- **certbot**: Certificados SSL (opcional)
- **pgbackups**: Backups automáticos de PostgreSQL
- **mediabackups**: Backups automáticos de archivos media
- **prometheus/grafana**: Monitoreo y dashboards

---

## **Comandos clave para desarrollo**

```bash
# Instalar dependencias backend
cd backend
pip install -r requirements.txt

# Migrar base de datos
python manage.py migrate

# Crear superusuario
python manage.py createsuperuser

# Levantar servidor de desarrollo
python manage.py runserver

# Ejecutar tests
python manage.py test

# Levantar servicios externos (dev)
docker compose -f docker-compose.celery.yml up -d
docker compose -f docker-compose.monitoring.yml up -d

# Ejecutar Celery worker
celery -A financialhub worker -l info

# Ejecutar Flower (monitor Celery)
celery -A financialhub flower --port=5555
```

---

## **Comandos clave para producción**

```bash
# Build y levantar todo en modo producción
docker compose -f docker-compose.prod.yml up -d --build

# Ver logs de un servicio
docker compose -f docker-compose.prod.yml logs -f web

# Ejecutar migraciones en el contenedor web
docker compose -f docker-compose.prod.yml exec web python manage.py migrate

# Ejecutar collectstatic en el contenedor web
docker compose -f docker-compose.prod.yml exec web python manage.py collectstatic --noinput

# Restaurar un backup de base de datos
# (Copia el archivo dump al host y ejecuta)
docker compose -f docker-compose.prod.yml run --rm -v $(pwd)/restore:/restore postgres:15-alpine \
  pg_restore -U $DB_USER -d $DB_NAME /restore/backup-YYYY-MM-DD/backup-HH-MM-SS.dump

# Restaurar un backup de media
tar xzf media-YYYY-MM-DD/media-HH-MM-SS.tar.gz -C /ruta/a/media
```

---

## **Comandos de monitoreo y observabilidad**

```bash
# Acceder a Prometheus (por defecto en http://localhost:9090)
# Acceder a Grafana (por defecto en http://localhost:3001, user: admin, pass: admin)
# Acceder a Flower (por defecto en http://localhost:5555)
```

---

## **Calidad de Código y Pre-commit Hooks**

El proyecto usa pre-commit hooks para mantener la calidad del código. Para instalarlos:

```bash
# Instalar pre-commit
pip install pre-commit

# Instalar los hooks
pre-commit install

# Ejecutar manualmente en todo el código
pre-commit run --all-files
```

Los hooks incluyen:
- **Black**: Formateo automático de Python
- **isort**: Ordenamiento de imports
- **Flake8**: Linting y análisis estático
- **Bandit**: Análisis de seguridad
- **Otros**: Verificación de YAML, JSON, conflictos de merge, etc.

## **Documentación de la API**

La API está documentada usando Swagger/OpenAPI. Puedes acceder a la documentación en:

- **Swagger UI**: `http://localhost:8000/swagger/`
- **ReDoc**: `http://localhost:8000/redoc/`
- **JSON Schema**: `http://localhost:8000/swagger.json`

La documentación incluye:
- Todos los endpoints disponibles
- Esquemas de request/response
- Autenticación y autorización
- Ejemplos de uso
- Interfaz interactiva para probar endpoints

---

## **Backups automáticos**

- **Base de datos**: dumps diarios en el volumen `pgbackups`
- **Media**: archivos tar.gz diarios en el volumen `mediabackups`
- Los backups se organizan por fecha y hora.

---

## **Buenas prácticas y seguridad**

- Usa `.env` para desarrollo y `.env.prod` para producción (no los subas al repo).
- Cambia todas las contraseñas y claves por valores seguros en producción.
- Usa HTTPS y configura Nginx con Let's Encrypt cuando tengas dominio.
- Los archivos estáticos y media se sirven por Nginx, no por Django.
- Todos los servicios críticos están orquestados y aislados en Docker.
- Pre-commit hooks recomendados (`black`, `isort`, `flake8`, `bandit`).

---

## **CI/CD y calidad**

- Linting, formateo y tests automáticos con GitHub Actions.
- Pre-commit hooks para asegurar calidad antes de cada commit.

---

## **Onboarding rápido**

1. Clona el repo y copia `.env` y `.env.prod` (ajusta valores).
2. Instala dependencias y migra la base de datos.
3. Levanta los servicios con Docker Compose.
4. Accede a la API, admin y frontend según corresponda.
5. Consulta los backups y la documentación Swagger.
6. Usa los comandos de monitoreo y Celery según necesidad.

---

## **Notas adicionales**

- Para restaurar backups, consulta la sección de comandos de producción.
- Para escalar, puedes adaptar los servicios a Kubernetes fácilmente.
- Si tienes dudas, revisa los archivos de configuración y este README.

---

## 🛠️ Makefile: Comandos útiles para desarrollo y producción

Para facilitar la gestión de entornos y servicios, este proyecto incluye un `Makefile` con comandos prácticos:

### **Comandos disponibles**

- `make dev-env` — Prepara el entorno de desarrollo (`.env.dev` → `.env`)
- `make prod-env` — Prepara el entorno de producción (`.env.prod` → `.env`)
- `make up` — Levanta todos los servicios con Docker Compose
- `make down` — Detiene todos los servicios
- `make logs` — Muestra los logs en tiempo real
- `make bash` — Acceso bash al contenedor web
- `make migrate` — Ejecuta migraciones de Django
- `make createsuperuser` — Crea un superusuario de Django
- `make test` — Ejecuta los tests

### **Ejemplo de uso**

```bash
# Preparar entorno de desarrollo
yarn dev-env

# Levantar servicios
yarn up

# Ver logs
yarn logs

# Detener servicios
yarn down
```

> **Nota:** Asegúrate de tener el archivo `.env.dev` o `.env.prod` configurado antes de levantar los servicios.

---

## **Integración con Stripe: Pagos, Suscripciones y Webhooks**

### **1. Configuración de variables de entorno**
Asegúrate de tener en tu archivo `.env` las siguientes variables:

```
# Stripe
STRIPE_SECRET_KEY=sk_test_...
STRIPE_PUBLISHABLE_KEY=pk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PRICE_ID=price_...
```

Puedes obtener estos valores desde tu [Stripe Dashboard](https://dashboard.stripe.com/test/apikeys) y en la sección de Webhooks.

---

### **2. Endpoints disponibles**

- `POST /api/payments/stripe/create-subscription/` — Crea una suscripción para una organización.
- `POST /api/payments/stripe/cancel-subscription/` — Cancela una suscripción activa.
- `POST /api/payments/stripe/update-subscription/` — Cambia el plan de una suscripción.
- `GET /api/payments/stripe/subscription-status/` — Consulta el estado de la suscripción.
- `POST /api/payments/stripe/webhook/` — Endpoint para recibir eventos de Stripe (webhooks).

**Todos los endpoints requieren autenticación JWT, excepto el webhook.**

---

### **3. Configuración de Webhooks en Stripe (Desarrollo Local)**

Stripe no puede acceder a tu localhost directamente. Para pruebas locales, usa [ngrok](https://ngrok.com/):

1. Instala ngrok:
   ```bash
   brew install --cask ngrok
   # o descarga desde https://ngrok.com/download
   ```
2. Autentica ngrok:
   ```bash
   ngrok config add-authtoken TU_AUTHTOKEN
   ```
3. Inicia tu servidor Django:
   ```bash
   python manage.py runserver
   ```
4. En otra terminal, ejecuta:
   ```bash
   ngrok http 8000
   ```
5. Copia la URL pública que te da ngrok (ej: `https://abcd1234.ngrok.io`).
6. En Stripe Dashboard → Developers → Webhooks → "Add endpoint":
   - URL: `https://abcd1234.ngrok.io/api/payments/stripe/webhook/`
   - Selecciona los eventos:
     - `invoice.payment_succeeded`
     - `customer.subscription.updated`
     - `customer.subscription.deleted`
   - Copia el **Signing secret** y ponlo en tu `.env` como `STRIPE_WEBHOOK_SECRET`.

---

### **4. Pruebas de Webhooks con Stripe CLI**

Stripe CLI permite simular eventos fácilmente:

1. Instala Stripe CLI:
   ```bash
   brew install stripe/stripe-cli/stripe
   # o descarga desde https://stripe.com/docs/stripe-cli#install
   ```
2. Haz login:
   ```bash
   stripe login
   ```
3. Escucha y reenvía eventos a tu backend:
   ```bash
   stripe listen --forward-to localhost:8000/api/payments/stripe/webhook/
   ```
4. Dispara eventos de prueba:
   ```bash
   stripe trigger invoice.payment_succeeded
   stripe trigger customer.subscription.updated
   stripe trigger customer.subscription.deleted
   ```
5. Verifica en la terminal de Django que los eventos se reciben correctamente.

---

### **5. Ejemplo de uso de endpoints (con curl o Postman)**

**Crear suscripción:**
```bash
curl -X POST http://localhost:8000/api/payments/stripe/create-subscription/ \
  -H "Authorization: Bearer TU_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"organization_id": 1, "price_id": "price_xxx"}'
```

**Cancelar suscripción:**
```bash
curl -X POST http://localhost:8000/api/payments/stripe/cancel-subscription/ \
  -H "Authorization: Bearer TU_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"organization_id": 1}'
```

**Consultar estado de suscripción:**
```bash
curl -X GET "http://localhost:8000/api/payments/stripe/subscription-status/?organization_id=1" \
  -H "Authorization: Bearer TU_JWT_TOKEN"
```

---

### **6. Notas y buenas prácticas**
- Siempre reinicia el servidor Django si cambias el `.env`.
- Usa Stripe CLI y ngrok para pruebas locales de webhooks.
- No subas tus claves de Stripe al repositorio.
- En producción, usa HTTPS y configura el webhook con tu dominio real.
- Consulta la [documentación oficial de Stripe](https://stripe.com/docs) para más detalles.

---

## **Features Pro y lógica de acceso**

### **Features Pro para contadores**
- Panel multi-organización
- Reportes avanzados de clientes
- Incentivos por upgrades de clientes
- Exportación masiva de datos
- Herramientas de conciliación y auditoría

### **Features Pro para miembros regulares (clientes/empresas)**
- Reportes avanzados de su propia organización
- Integraciones bancarias
- Soporte prioritario
- Más miembros en la organización
- Automatizaciones y reglas personalizadas

### **Lógica de acceso Pro**
- Un usuario tiene acceso Pro si:
  - Tiene Pro global (`pro_features=True`), o
  - Tiene trial activo (`pro_trial_until` en el futuro), o
  - Es miembro de una organización con plan Pro, o
  - Tiene features Pro asignadas en `pro_features_list`, o
  - Es contador y la organización le habilitó `pro_features_for_accountant`.
- Al invitar a un contador a una organización Pro, si no tiene Pro ni trial, se le asigna un trial automático de 30 días.
- Los incentivos se asignan automáticamente al contador cuando una organización hace upgrade a Pro.

---


