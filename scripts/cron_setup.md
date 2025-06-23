# 🤖 Configuración de Entrenamiento Automático de IA

## 📅 Configuración de Cron

### 1. **Entrenamiento Semanal (Recomendado)**
```bash
# Ejecutar cada domingo a las 2:00 AM
0 2 * * 0 cd /ruta/a/tu/proyecto && python scripts/schedule_ai_training.py scheduled >> logs/cron.log 2>&1
```

### 2. **Entrenamiento Mensual**
```bash
# Ejecutar el primer día de cada mes a las 3:00 AM
0 3 1 * * cd /ruta/a/tu/proyecto && python scripts/schedule_ai_training.py scheduled >> logs/cron.log 2>&1
```

### 3. **Verificación Diaria (Opcional)**
```bash
# Verificar estado diariamente a las 6:00 AM
0 6 * * * cd /ruta/a/tu/proyecto && python scripts/schedule_ai_training.py status >> logs/status.log 2>&1
```

## 🔧 Configuración Manual

### Agregar a crontab:
```bash
# Editar crontab
crontab -e

# Agregar las líneas de arriba
```

### Verificar configuración:
```bash
# Ver crontab actual
crontab -l

# Ver logs de cron
tail -f logs/cron.log
```

## 📊 Comandos del Scheduler

### Verificar estado:
```bash
python scripts/schedule_ai_training.py status
```

### Entrenamiento manual:
```bash
# Solo NLP
python scripts/schedule_ai_training.py manual nlp_only

# Semanal
python scripts/schedule_ai_training.py weekly

# Mensual
python scripts/schedule_ai_training.py monthly
```

### Entrenamiento programado:
```bash
python scripts/schedule_ai_training.py scheduled
```

## ⚙️ Configuración Avanzada

### Variables de entorno:
```bash
# En tu .env o configuración del sistema
AI_TRAINING_ENABLED=true
AI_TRAINING_WEEKLY=true
AI_TRAINING_MONTHLY=true
AI_TRAINING_TIMEOUT=3600
```

### Logs:
- `logs/ai_training.log` - Logs del entrenamiento
- `logs/cron.log` - Logs de cron
- `logs/status.log` - Logs de estado

## 🚨 Monitoreo

### Verificar que funcione:
```bash
# Verificar logs
tail -f logs/ai_training.log

# Verificar archivos de timestamp
ls -la logs/last_*_training.txt

# Verificar modelos actualizados
ls -la backend/ml_models/*.joblib
```

### Alertas (opcional):
```bash
# Script para enviar notificaciones si falla el entrenamiento
python scripts/check_training_health.py
```

## 📈 Métricas de Rendimiento

### Tiempos típicos:
- **Entrenamiento completo**: 15-30 minutos
- **Solo NLP**: 5-10 minutos
- **Timeout configurado**: 1 hora

### Recursos utilizados:
- **CPU**: Alto durante entrenamiento
- **Memoria**: 2-4 GB durante entrenamiento
- **Disco**: Los modelos ocupan ~50-100 MB

## 🔄 Mantenimiento

### Limpieza de logs:
```bash
# Limpiar logs antiguos (más de 30 días)
find logs/ -name "*.log" -mtime +30 -delete
```

### Backup de modelos:
```bash
# Crear backup de modelos entrenados
tar -czf backup/models_$(date +%Y%m%d).tar.gz backend/ml_models/
```

## 🆘 Solución de Problemas

### Error común: Timeout
```bash
# Aumentar timeout en el script
timeout=7200  # 2 horas en lugar de 1
```

### Error común: Permisos
```bash
# Dar permisos de ejecución
chmod +x scripts/schedule_ai_training.py
```

### Error común: Entorno virtual
```bash
# Usar ruta completa al Python del entorno virtual
/usr/bin/env python3 scripts/schedule_ai_training.py scheduled
``` 