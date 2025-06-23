#!/usr/bin/env python3
"""
Script para ejecutar tests de IA con diferentes configuraciones
Permite probar con autenticación habilitada/deshabilitada
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_tests_with_config(config_type, test_pattern=None, verbose=False):
    """Ejecutar tests con configuración específica"""
    
    # Configurar variables de entorno
    env = os.environ.copy()
    
    if config_type == 'test':
        env['DJANGO_SETTINGS_MODULE'] = 'financialhub.settings.test'
        env['AI_TEST_ENDPOINTS_AUTH'] = 'False'
        env['DEBUG'] = 'True'
        print("🧪 Ejecutando tests con configuración de TEST (sin autenticación)")
    elif config_type == 'dev':
        env['DJANGO_SETTINGS_MODULE'] = 'financialhub.settings'
        env['DEBUG'] = 'True'
        env['AI_TEST_ENDPOINTS_AUTH'] = 'False'
        print("🔧 Ejecutando tests con configuración de DESARROLLO (sin autenticación)")
    elif config_type == 'prod':
        env['DJANGO_SETTINGS_MODULE'] = 'financialhub.settings'
        env['DEBUG'] = 'False'
        env['AI_TEST_ENDPOINTS_AUTH'] = 'True'
        print("🚀 Ejecutando tests con configuración de PRODUCCIÓN (con autenticación)")
    else:
        print(f"❌ Configuración '{config_type}' no válida")
        return False
    
    # Construir comando de pytest
    cmd = [
        sys.executable, '-m', 'pytest',
        'ai/tests/',
        '-v' if verbose else '',
        '--tb=short',
        '--strict-markers',
        '--disable-warnings',
        '--color=yes'
    ]
    
    # Agregar patrón de test si se especifica
    if test_pattern:
        cmd.append(f'-k {test_pattern}')
    
    # Filtrar argumentos vacíos
    cmd = [arg for arg in cmd if arg]
    
    print(f"📋 Comando: {' '.join(cmd)}")
    print(f"🔧 Configuración: {env['DJANGO_SETTINGS_MODULE']}")
    print(f"🔐 AI_TEST_ENDPOINTS_AUTH: {env.get('AI_TEST_ENDPOINTS_AUTH', 'Not set')}")
    print(f"🐛 DEBUG: {env.get('DEBUG', 'Not set')}")
    print("=" * 60)
    
    try:
        # Ejecutar tests
        result = subprocess.run(cmd, env=env, cwd='.', check=False)
        
        print("=" * 60)
        if result.returncode == 0:
            print("✅ Tests completados exitosamente")
        else:
            print("❌ Tests fallaron")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error ejecutando tests: {e}")
        return False

def run_specific_test_files():
    """Ejecutar archivos de test específicos"""
    
    test_files = [
        'ai/tests/test_complete_ai_system.py',
        'ai/tests/unit/test_expense_predictor.py',
        'ai/tests/unit/test_transaction_classifier.py',
        'ai/tests/unit/test_behavior_analyzer.py',
        'ai/tests/unit/test_budget_optimizer.py',
        'ai/tests/integration/test_model_training.py',
    ]
    
    results = {}
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n🧪 Ejecutando {test_file}")
            print("-" * 40)
            
            cmd = [
                sys.executable, '-m', 'pytest',
                test_file,
                '-v',
                '--tb=short',
                '--strict-markers',
                '--disable-warnings',
                '--color=yes'
            ]
            
            env = os.environ.copy()
            env['DJANGO_SETTINGS_MODULE'] = 'financialhub.settings.test'
            env['AI_TEST_ENDPOINTS_AUTH'] = 'False'
            env['DEBUG'] = 'True'
            
            try:
                result = subprocess.run(cmd, env=env, cwd='.', check=False)
                results[test_file] = result.returncode == 0
                
                if result.returncode == 0:
                    print(f"✅ {test_file} - PASÓ")
                else:
                    print(f"❌ {test_file} - FALLÓ")
                    
            except Exception as e:
                print(f"❌ Error ejecutando {test_file}: {e}")
                results[test_file] = False
        else:
            print(f"⚠️  Archivo no encontrado: {test_file}")
            results[test_file] = False
    
    return results

def run_coverage_tests():
    """Ejecutar tests con cobertura"""
    
    print("📊 Ejecutando tests con cobertura")
    print("=" * 60)
    
    env = os.environ.copy()
    env['DJANGO_SETTINGS_MODULE'] = 'financialhub.settings.test'
    env['AI_TEST_ENDPOINTS_AUTH'] = 'False'
    env['DEBUG'] = 'True'
    
    cmd = [
        sys.executable, '-m', 'pytest',
        'ai/tests/',
        '--cov=ai',
        '--cov-report=html',
        '--cov-report=term-missing',
        '--cov-report=xml',
        '--cov-fail-under=80',
        '-v',
        '--tb=short',
        '--strict-markers',
        '--disable-warnings',
        '--color=yes'
    ]
    
    try:
        result = subprocess.run(cmd, env=env, cwd='.', check=False)
        
        if result.returncode == 0:
            print("✅ Tests con cobertura completados")
            print("📁 Reporte HTML generado en htmlcov/")
            print("📄 Reporte XML generado en coverage.xml")
        else:
            print("❌ Tests con cobertura fallaron")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error ejecutando tests con cobertura: {e}")
        return False

def run_performance_tests():
    """Ejecutar tests de rendimiento"""
    
    print("⚡ Ejecutando tests de rendimiento")
    print("=" * 60)
    
    env = os.environ.copy()
    env['DJANGO_SETTINGS_MODULE'] = 'financialhub.settings.test'
    env['AI_TEST_ENDPOINTS_AUTH'] = 'False'
    env['DEBUG'] = 'True'
    
    cmd = [
        sys.executable, '-m', 'pytest',
        'ai/tests/',
        '-m', 'performance',
        '--durations=10',
        '-v',
        '--tb=short',
        '--strict-markers',
        '--disable-warnings',
        '--color=yes'
    ]
    
    try:
        result = subprocess.run(cmd, env=env, cwd='.', check=False)
        
        if result.returncode == 0:
            print("✅ Tests de rendimiento completados")
        else:
            print("❌ Tests de rendimiento fallaron")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error ejecutando tests de rendimiento: {e}")
        return False

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Ejecutar tests de IA con diferentes configuraciones')
    
    parser.add_argument('--config', choices=['test', 'dev', 'prod'], default='test',
                       help='Configuración a usar (test, dev, prod)')
    parser.add_argument('--pattern', type=str,
                       help='Patrón de test a ejecutar (ej: "test_analyze")')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Modo verbose')
    parser.add_argument('--specific', action='store_true',
                       help='Ejecutar archivos de test específicos')
    parser.add_argument('--coverage', action='store_true',
                       help='Ejecutar tests con cobertura')
    parser.add_argument('--performance', action='store_true',
                       help='Ejecutar tests de rendimiento')
    parser.add_argument('--all', action='store_true',
                       help='Ejecutar todos los tipos de tests')
    
    args = parser.parse_args()
    
    print("🚀 Ejecutor de Tests de IA - FinancialHub")
    print("=" * 60)
    
    success = True
    
    if args.all:
        # Ejecutar todos los tipos de tests
        print("\n1️⃣ Ejecutando tests básicos...")
        success &= run_tests_with_config('test', verbose=args.verbose)
        
        print("\n2️⃣ Ejecutando tests específicos...")
        specific_results = run_specific_test_files()
        success &= all(specific_results.values())
        
        print("\n3️⃣ Ejecutando tests con cobertura...")
        success &= run_coverage_tests()
        
        print("\n4️⃣ Ejecutando tests de rendimiento...")
        success &= run_performance_tests()
        
    elif args.specific:
        specific_results = run_specific_test_files()
        success = all(specific_results.values())
        
    elif args.coverage:
        success = run_coverage_tests()
        
    elif args.performance:
        success = run_performance_tests()
        
    else:
        success = run_tests_with_config(args.config, args.pattern, args.verbose)
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ¡Todos los tests pasaron exitosamente!")
        sys.exit(0)
    else:
        print("💥 Algunos tests fallaron")
        sys.exit(1)

if __name__ == '__main__':
    main() 