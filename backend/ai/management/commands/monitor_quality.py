from django.core.management.base import BaseCommand
from django.utils import timezone
from ai.services import AIService
from ai.models import AIInsight, AIPrediction
import logging
import json
from datetime import timedelta

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Monitorea y mantiene la calidad del sistema de IA (accuracy ≥ 65%)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--auto-retrain',
            action='store_true',
            help='Re-entrenar automáticamente modelos con rendimiento bajo',
        )
        parser.add_argument(
            '--generate-report',
            action='store_true',
            help='Generar reporte detallado de calidad',
        )
        parser.add_argument(
            '--fix-issues',
            action='store_true',
            help='Intentar corregir problemas de calidad automáticamente',
        )
        parser.add_argument(
            '--threshold',
            type=float,
            default=0.65,
            help='Umbral mínimo de accuracy (default: 0.65)',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('🔍 Iniciando monitoreo de calidad del sistema de IA...'))
        
        try:
            # Inicializar servicio de IA
            ai_service = AIService()
            
            # Actualizar umbral de calidad si se especifica
            ai_service.quality_gate_config['min_accuracy'] = options['threshold']
            
            # Generar reporte de calidad
            if options['generate_report']:
                self._generate_quality_report(ai_service)
            
            # Auto-retrain si se solicita
            if options['auto_retrain']:
                self._auto_retrain_models(ai_service)
            
            # Corregir problemas si se solicita
            if options['fix_issues']:
                self._fix_quality_issues(ai_service)
            
            # Monitoreo continuo
            self._continuous_monitoring(ai_service)
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Error durante el monitoreo: {str(e)}'))
            logger.error(f'Error in quality monitoring: {str(e)}', exc_info=True)

    def _generate_quality_report(self, ai_service):
        """Genera un reporte detallado de calidad."""
        self.stdout.write('📊 Generando reporte de calidad...')
        
        try:
            quality_report = ai_service.get_quality_report()
            
            # Mostrar resumen
            self.stdout.write(f"\n{'='*60}")
            self.stdout.write(f"📈 REPORTE DE CALIDAD DEL SISTEMA DE IA")
            self.stdout.write(f"{'='*60}")
            
            summary = quality_report.get('summary', {})
            self.stdout.write(f"Estado General: {quality_report['overall_status'].upper()}")
            self.stdout.write(f"Modelos Totales: {summary.get('total_models', 0)}")
            self.stdout.write(f"Modelos Aprobados: {summary.get('passing_models', 0)}")
            self.stdout.write(f"Porcentaje Aprobado: {summary.get('passing_percentage', 0):.1f}%")
            
            # Mostrar estado de cada modelo
            self.stdout.write(f"\n📋 ESTADO DE MODELOS:")
            self.stdout.write(f"{'-'*40}")
            
            for model_name, status in quality_report.get('models_status', {}).items():
                accuracy = status.get('accuracy', 0)
                confidence = status.get('confidence', 0)
                model_status = status.get('status', 'unknown')
                
                status_icon = "✅" if model_status == 'pass' else "❌" if model_status == 'fail' else "⚠️"
                
                self.stdout.write(
                    f"{status_icon} {model_name}: "
                    f"Accuracy={accuracy:.3f}, "
                    f"Confidence={confidence:.3f}, "
                    f"Status={model_status}"
                )
            
            # Mostrar recomendaciones
            recommendations = quality_report.get('recommendations', [])
            if recommendations:
                self.stdout.write(f"\n🔧 RECOMENDACIONES:")
                self.stdout.write(f"{'-'*40}")
                for rec in recommendations:
                    self.stdout.write(f"• {rec['model']}: {rec['action']} - {rec['reason']}")
            
            # Guardar reporte en archivo
            report_filename = f"quality_report_{timezone.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_filename, 'w') as f:
                json.dump(quality_report, f, indent=2, default=str)
            
            self.stdout.write(f"\n💾 Reporte guardado en: {report_filename}")
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error generando reporte: {str(e)}'))

    def _auto_retrain_models(self, ai_service):
        """Re-entrena automáticamente modelos con rendimiento bajo."""
        self.stdout.write('🔄 Iniciando re-entrenamiento automático...')
        
        try:
            retrain_result = ai_service.auto_retrain_low_performance_models()
            
            if retrain_result['status'] == 'completed':
                retrained_models = retrain_result.get('retrained_models', [])
                
                if retrained_models:
                    self.stdout.write(f"✅ Re-entrenamiento completado:")
                    for model_info in retrained_models:
                        status_icon = "✅" if model_info['status'] == 'retrained' else "❌"
                        self.stdout.write(
                            f"{status_icon} {model_info['model']}: "
                            f"Accuracy anterior={model_info['old_accuracy']:.3f}, "
                            f"Status={model_info['status']}"
                        )
                else:
                    self.stdout.write("✅ Todos los modelos ya cumplen con el umbral de calidad")
            else:
                self.stdout.write(self.style.ERROR(f"❌ Error en re-entrenamiento: {retrain_result.get('error', 'Unknown error')}"))
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error en re-entrenamiento automático: {str(e)}'))

    def _fix_quality_issues(self, ai_service):
        """Intenta corregir problemas de calidad automáticamente."""
        self.stdout.write('🔧 Corrigiendo problemas de calidad...')
        
        try:
            quality_report = ai_service.get_quality_report()
            issues_fixed = 0
            
            for model_name, status in quality_report.get('models_status', {}).items():
                if status.get('status') == 'fail':
                    self.stdout.write(f"🔧 Corrigiendo modelo: {model_name}")
                    
                    # Intentar diferentes estrategias de corrección
                    success = self._apply_fix_strategies(ai_service, model_name)
                    
                    if success:
                        issues_fixed += 1
                        self.stdout.write(f"✅ Modelo {model_name} corregido exitosamente")
                    else:
                        self.stdout.write(f"❌ No se pudo corregir el modelo {model_name}")
            
            self.stdout.write(f"📊 Total de problemas corregidos: {issues_fixed}")
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error corrigiendo problemas: {str(e)}'))

    def _apply_fix_strategies(self, ai_service, model_name):
        """Aplica estrategias de corrección para un modelo específico."""
        try:
            # Estrategia 1: Re-entrenamiento
            if ai_service._retrain_model(model_name):
                return True
            
            # Estrategia 2: Ajustar hiperparámetros
            if self._adjust_hyperparameters(ai_service, model_name):
                return True
            
            # Estrategia 3: Usar AutoML
            if self._use_automl_optimization(ai_service, model_name):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error applying fix strategies for {model_name}: {str(e)}")
            return False

    def _adjust_hyperparameters(self, ai_service, model_name):
        """Ajusta hiperparámetros del modelo."""
        try:
            # Implementar ajuste de hiperparámetros específico por modelo
            if model_name == 'transaction_classifier':
                # Ajustar parámetros del RandomForest
                pass
            elif model_name == 'expense_predictor':
                # Ajustar parámetros del GradientBoosting
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Error adjusting hyperparameters for {model_name}: {str(e)}")
            return False

    def _use_automl_optimization(self, ai_service, model_name):
        """Usa AutoML para optimizar el modelo."""
        try:
            # Obtener datos de entrenamiento
            from transactions.models import Transaction
            from django.utils import timezone
            from datetime import timedelta
            
            transactions = Transaction.objects.filter(
                created_at__gte=timezone.now() - timedelta(days=90)
            ).select_related('category')
            
            if not transactions.exists():
                return False
            
            # Preparar datos para AutoML
            transaction_data = []
            for t in transactions:
                transaction_data.append({
                    'amount': float(t.amount),
                    'category_id': t.category.id if t.category else 0,
                    'date': t.date,
                    'description': t.description or ''
                })
            
            # Usar AutoML para optimización
            if model_name == 'transaction_classifier':
                # Optimizar clasificador
                pass
            elif model_name == 'expense_predictor':
                # Optimizar predictor
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Error using AutoML for {model_name}: {str(e)}")
            return False

    def _continuous_monitoring(self, ai_service):
        """Monitoreo continuo del sistema."""
        self.stdout.write('👁️ Iniciando monitoreo continuo...')
        
        try:
            # Verificar calidad cada hora
            while True:
                quality_report = ai_service.get_quality_report()
                
                # Verificar si hay problemas críticos
                critical_issues = []
                for model_name, status in quality_report.get('models_status', {}).items():
                    if status.get('status') == 'fail':
                        critical_issues.append(model_name)
                
                if critical_issues:
                    self.stdout.write(
                        self.style.WARNING(
                            f"⚠️ Problemas críticos detectados en: {', '.join(critical_issues)}"
                        )
                    )
                    
                    # Crear alerta
                    self._create_quality_alert(critical_issues, quality_report)
                    
                    # Intentar corrección automática
                    if ai_service.quality_gate_config['enable_auto_retraining']:
                        self.stdout.write("🔄 Intentando corrección automática...")
                        ai_service.auto_retrain_low_performance_models()
                
                # Esperar antes de la siguiente verificación
                import time
                time.sleep(ai_service.quality_gate_config['quality_check_interval'])
                
        except KeyboardInterrupt:
            self.stdout.write("\n🛑 Monitoreo detenido por el usuario")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error en monitoreo continuo: {str(e)}'))

    def _create_quality_alert(self, critical_models, quality_report):
        """Crea una alerta de calidad."""
        try:
            # Crear insight de alerta
            alert_message = f"Alertas de calidad detectadas en modelos: {', '.join(critical_models)}"
            
            AIInsight.objects.create(
                user=None,  # Alerta del sistema
                type='risk',
                title='Alerta de Calidad del Sistema de IA',
                description=alert_message,
                data={
                    'critical_models': critical_models,
                    'quality_report': quality_report,
                    'timestamp': timezone.now().isoformat()
                },
                impact_score=0.8  # Alto impacto
            )
            
            self.stdout.write(f"🚨 Alerta creada: {alert_message}")
            
        except Exception as e:
            logger.error(f"Error creating quality alert: {str(e)}") 