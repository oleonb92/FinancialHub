# Celery: Ejecución y Solución de Problemas

## Estructura del Proyecto
Asegúrate de estar en el directorio correcto antes de ejecutar cualquier comando relacionado con Celery. El directorio correcto es:

```
/Users/osmanileon/Desktop/FinancialHub_backup/backend
```

El módulo principal de Django y Celery es `financialhub`, ubicado en `backend/financialhub`.

---

## Cómo iniciar Celery Worker y Beat

**Desde el directorio `backend` ejecuta:**

### Iniciar el worker
```bash
celery -A financialhub worker -l info -n worker1@%h
```

### Iniciar el beat
```bash
celery -A financialhub beat -l info
```

### Verificar el estado de los workers
```bash
celery -A financialhub inspect ping
```

---

## Errores comunes

### Error: `Unable to load celery application. The module financialhub was not found.`

**Causa:**
- Este error ocurre si ejecutas los comandos desde el directorio equivocado, por ejemplo, `backend/backend`.

**Solución:**
- Asegúrate de estar en el directorio `backend` antes de ejecutar cualquier comando de Celery.

---

## Scripts útiles

Puedes usar los scripts proporcionados para iniciar los servicios:

- `./start_celery.sh` — Inicia el worker
- `./start_celery_beat.sh` — Inicia el beat

Ambos scripts configuran correctamente el `PYTHONPATH` para evitar problemas de importación.

---

## Resumen
- Siempre ejecuta los comandos desde el directorio `backend`.
- No cambies a `backend/backend`.
- Si ves errores de módulo no encontrado, revisa tu directorio actual. 