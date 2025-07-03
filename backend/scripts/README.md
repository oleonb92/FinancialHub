# 🤖 Sistema de Entrenamiento Automático de IA - FinancialHub

## 📋 Descripción General

Este directorio contiene todos los scripts y herramientas para el entrenamiento automático de modelos de Inteligencia Artificial en FinancialHub. El sistema está diseñado para ser completamente automatizado, robusto y profesional.

## 🚀 Configuración Inicial

### Configuración Automática (Recomendado)

Para configurar todo el sistema automáticamente, ejecuta:

```bash
cd /Users/osmanileon/Desktop/FinancialHub_backup
python scripts/setup_auto_training.py
```

Este comando:
- ✅ Configura cron jobs automáticamente
- ✅ Crea directorios necesarios
- ✅ Configura logs
- ✅ Crea script de backup
- ✅ Ejecuta entrenamiento inicial
- ✅ Verifica toda la configuración

### Configuración Manual

Si prefieres configurar manualmente:

1. **Crear directorios necesarios:**
   ```bash
   mkdir -p logs backup backend/ml_models/test
   ```

2. **Configurar cron jobs:**
   ```bash
   crontab -e
   ```
   
   Agregar las siguientes líneas:
   ```bash
   # 🤖 AI Training Jobs - FinancialHub
   # Entrenamiento semanal (domingo 2:00 AM)
   0 2 * * 0 cd /Users/osmanileon/Desktop/FinancialHub_backup && python scripts/schedule_ai_training.py scheduled >> logs/cron.log 2>&1
   
   # Entrenamiento mensual (primer día del mes 3:00 AM)
   0 3 1 * * cd /Users/osmanileon/Desktop/FinancialHub_backup && python scripts/schedule_ai_training.py scheduled >> logs/cron.log 2>&1
   
   # Verificación diaria (6:00 AM)
   0 6 * * * cd /Users/osmanileon/Desktop/FinancialHub_backup && python scripts/schedule_ai_training.py status >> logs/status.log 2>&1
   ```

## 📁 Estructura de Archivos

```
scripts/
├── README.md                           # Este archivo
├── setup_auto_training.py              # Configurador automático principal
├── schedule_ai_training.py             # Scheduler inteligente
├── backup_models.py                    # Script de backup automático
├── setup_advanced_ai.py                # Configuración avanzada de IA
├── install_advanced_deps.py            # Instalación de dependencias
├── test_advanced_ai.py                 # Tests de IA avanzada
└── cron_setup.md                       # Documentación de cron
```

## 🤖 Scripts Principales

### 1. `setup_auto_training.py` - Configurador Automático

**Propósito:** Configuración completa y automática del sistema de entrenamiento.

**Funcionalidades:**
- Configura cron jobs automáticamente
- Crea directorios y archivos necesarios
- Configura logs
- Crea script de backup
- Ejecuta entrenamiento inicial
- Verifica toda la configuración

**Uso:**
```bash
python scripts/setup_auto_training.py
```

### 2. `schedule_ai_training.py` - Scheduler Inteligente

**Propósito:** Gestiona el entrenamiento automático de modelos de IA.

**Funcionalidades:**
- Entrenamiento semanal automático
- Entrenamiento mensual automático
- Verificación de estado
- Gestión inteligente de recursos
- Logs detallados

**Comandos disponibles:**
```bash
# Ver estado actual
python scripts/schedule_ai_training.py status

# Ejecutar entrenamiento programado
python scripts/schedule_ai_training.py scheduled

# Entrenamiento manual
python scripts/schedule_ai_training.py manual

# Limpiar logs antiguos
python scripts/schedule_ai_training.py cleanup
```

### 3. `backup_models.py` - Backup Automático

**Propósito:** Crea backups automáticos de los modelos entrenados.

**Funcionalidades:**
- Backup comprimido de modelos
- Rotación automática (mantiene últimos 5 backups)
- Timestamp en nombres de archivo
- Limpieza automática

**Uso:**
```bash
python scripts/backup_models.py
```

## 📅 Programación de Entrenamientos

### Frecuencia Automática

| **Tipo** | **Frecuencia** | **Hora** | **Descripción** |
|----------|----------------|----------|-----------------|
| **Semanal** | Cada domingo | 2:00 AM | Entrenamiento regular con datos recientes |
| **Mensual** | Primer día del mes | 3:00 AM | Entrenamiento profundo con todos los datos |
| **Verificación** | Diario | 6:00 AM | Verificación de estado y recursos |

### Entrenamiento Manual

Para entrenamiento manual inmediato:

```bash
# Entrenamiento completo (incluyendo NLP)
cd backend
python manage.py train_ai_models --include-nlp --force

# Solo modelos NLP
python manage.py train_ai_models --nlp-only --force

# Entrenamiento específico
python manage.py train_ai_models --user-id 1 --days 30
```

## 📊 Monitoreo y Logs

### Archivos de Log

```
logs/
├── ai_training.log          # Logs de entrenamiento
├── cron.log                 # Logs de cron jobs
└── status.log               # Logs de verificación de estado
```

### Comandos de Monitoreo

```bash
# Ver logs en tiempo real
tail -f logs/ai_training.log

# Ver estado actual
python scripts/schedule_ai_training.py status

# Ver cron jobs configurados
crontab -l

# Ver últimos backups
ls -la backup/
```

## 🔧 Configuración Avanzada

### Variables de Entorno

```bash
# Configurar en .env o exportar
export AI_TRAINING_DEBUG=true          # Modo debug
export AI_TRAINING_FORCE=false         # Forzar entrenamiento
export AI_TRAINING_BACKUP_RETENTION=5  # Número de backups a mantener
```

### Personalización de Horarios

Para cambiar los horarios de entrenamiento:

1. Editar `scripts/setup_auto_training.py`
2. Modificar la sección `cron_jobs`
3. Ejecutar nuevamente el configurador

```python
self.cron_jobs = [
    {
        'schedule': '0 2 * * 0',  # Cambiar horario aquí
        'command': '...',
        'description': 'Entrenamiento semanal automático'
    },
    # ... más jobs
]
```

## 🧪 Testing y Verificación

### Verificar Configuración

```bash
# Verificar que todo esté configurado
python scripts/setup_auto_training.py

# Verificar estado del scheduler
python scripts/schedule_ai_training.py status

# Verificar cron jobs
crontab -l | grep "AI Training"
```

### Tests de IA

```bash
# Ejecutar tests completos de IA
cd backend
python manage.py test ai.unit_tests.test_complete_ai_system --settings=financialhub.settings.test -v 2

# Tests específicos
python manage.py test ai.unit_tests.unit.test_behavior_analyzer
python manage.py test ai.unit_tests.unit.test_budget_optimizer
python manage.py test ai.unit_tests.unit.test_expense_predictor
```

## 🚨 Solución de Problemas

### Problemas Comunes

#### 1. Error: "Unknown command: 'train_ai_models'"
**Solución:**
```bash
cd backend
python manage.py help | grep train
# Verificar el nombre exacto del comando
```

#### 2. Error: "Worker exited prematurely: signal 11 (SIGSEGV)"
**Solución:**
- Reiniciar Celery workers
- Verificar uso de memoria
- Limpiar caché de Redis

#### 3. Error: "'RedisCache' object has no attribute 'keys'"
**Solución:**
- Verificar configuración de Redis
- Actualizar dependencias de Django

### Logs de Error

```bash
# Ver errores de cron
tail -f logs/cron.log

# Ver errores de entrenamiento
tail -f logs/ai_training.log

# Ver errores de Celery
tail -f logs/celery.log
```

## 📈 Métricas y Rendimiento

### Monitoreo de Recursos

El sistema incluye monitoreo automático de:
- Uso de CPU
- Uso de memoria
- Uso de disco
- Actividad de red

### Alertas Automáticas

- Alerta cuando uso de memoria > 80%
- Alerta cuando uso de CPU > 90%
- Alerta cuando espacio en disco < 10%

## 🔄 Mantenimiento

### Limpieza Automática

```bash
# Limpiar logs antiguos
python scripts/schedule_ai_training.py cleanup

# Limpiar backups antiguos
python scripts/backup_models.py

# Limpiar caché de modelos
cd backend
python manage.py train_ai_models --cleanup
```

### Actualización del Sistema

```bash
# Actualizar dependencias
pip install -r requirements.txt

# Actualizar modelos
cd backend
python manage.py train_ai_models --include-nlp --force

# Verificar configuración
python scripts/setup_auto_training.py
```

## 📚 Referencias

### Documentación Relacionada

- [Django Management Commands](https://docs.djangoproject.com/en/stable/ref/django-admin/)
- [Celery Documentation](https://docs.celeryproject.org/)
- [Cron Documentation](https://man7.org/linux/man-pages/man5/crontab.5.html)

### Archivos de Configuración

- `backend/ai/management/commands/train_ai_models.py` - Comando principal de entrenamiento
- `backend/ai/tasks/training/train_models.py` - Tareas de Celery
- `backend/ai/tasks/monitoring/monitor_resources.py` - Monitoreo de recursos

## 🤝 Contribución

Para contribuir al sistema de entrenamiento automático:

1. Crear una rama para tu feature
2. Implementar cambios
3. Agregar tests
4. Actualizar documentación
5. Crear pull request

## 📞 Soporte

Para soporte técnico o preguntas:

1. Revisar logs en `logs/`
2. Verificar configuración con `python scripts/setup_auto_training.py`
3. Consultar este README
4. Revisar documentación de Django y Celery

---

**Última actualización:** $(date)
**Versión del sistema:** 1.0.0
**Mantenido por:** Equipo de IA - FinancialHub 