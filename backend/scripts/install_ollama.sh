#!/bin/bash

# Script para instalar Ollama y configurar Llama 3 para el chatbot financiero

set -e

echo "🚀 Instalando Ollama y configurando Llama 3 para FinancialHub..."

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para logging
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Detectar sistema operativo
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        error "Sistema operativo no soportado: $OSTYPE"
        exit 1
    fi
    log "Sistema operativo detectado: $OS"
}

# Instalar Ollama en Linux
install_ollama_linux() {
    log "Instalando Ollama en Linux..."
    
    # Descargar e instalar Ollama
    curl -fsSL https://ollama.ai/install.sh | sh
    
    # Iniciar servicio Ollama
    sudo systemctl enable ollama
    sudo systemctl start ollama
    
    success "Ollama instalado en Linux"
}

# Instalar Ollama en macOS
install_ollama_macos() {
    log "Instalando Ollama en macOS..."
    
    # Verificar si Homebrew está instalado
    if ! command -v brew &> /dev/null; then
        error "Homebrew no está instalado. Instálalo desde https://brew.sh"
        exit 1
    fi
    
    # Instalar Ollama con Homebrew
    brew install ollama
    
    # Iniciar Ollama
    ollama serve &
    
    success "Ollama instalado en macOS"
}

# Verificar requisitos del sistema
check_requirements() {
    log "Verificando requisitos del sistema..."
    
    # Verificar memoria RAM (mínimo 8GB)
    if [[ "$OS" == "linux" ]]; then
        total_ram=$(free -g | awk '/^Mem:/{print $2}')
    else
        total_ram=$(sysctl -n hw.memsize | awk '{print $0/1024/1024/1024}')
    fi
    
    if (( $(echo "$total_ram < 8" | bc -l) )); then
        warning "Se recomienda al menos 8GB de RAM para Llama 3 8B"
        warning "RAM disponible: ${total_ram}GB"
    else
        success "RAM suficiente: ${total_ram}GB"
    fi
    
    # Verificar espacio en disco (mínimo 10GB)
    if [[ "$OS" == "linux" ]]; then
        free_space=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
    else
        free_space=$(df -g / | awk 'NR==2 {print $4}')
    fi
    
    if (( $(echo "$free_space < 10" | bc -l) )); then
        warning "Se recomienda al menos 10GB de espacio libre"
        warning "Espacio disponible: ${free_space}GB"
    else
        success "Espacio en disco suficiente: ${free_space}GB"
    fi
}

# Descargar modelo Llama 3
download_llama3() {
    log "Descargando modelo Llama 3 8B..."
    
    # Esperar a que Ollama esté listo
    log "Esperando a que Ollama esté disponible..."
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            break
        fi
        sleep 2
    done
    
    # Descargar modelo
    ollama pull llama3:8b
    
    success "Modelo Llama 3 8B descargado"
}

# Crear modelo fine-tuned (opcional)
create_finetuned_model() {
    log "Creando modelo fine-tuned para chatbot financiero..."
    
    # Crear Modelfile para fine-tuning
    cat > Modelfile << EOF
FROM llama3:8b

# Configuración del sistema
SYSTEM """Eres un asistente financiero personal experto. Solo puedes responder preguntas sobre finanzas personales del usuario actual.

INSTRUCCIONES:
1. Responde solo preguntas relacionadas con finanzas personales
2. Usa los datos financieros proporcionados para dar respuestas precisas
3. Si no tienes información suficiente, indícalo claramente
4. No des consejos de inversión específicos
5. No accedas a información de otros usuarios
6. Mantén respuestas concisas pero informativas
7. Si te preguntan algo no relacionado con finanzas, responde educadamente que solo puedes ayudar con temas financieros

CATEGORÍAS DISPONIBLES:
- supermercado, restaurantes, transporte, hogar, salud, entretenimiento, viajes, suscripciones

FORMATO DE RESPUESTA:
- Usa números con formato de moneda ($XXX.XX)
- Incluye porcentajes cuando sea relevante
- Proporciona contexto útil pero conciso"""

# Template para conversaciones
TEMPLATE """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
{{ .Response }}<|end|>
"""
EOF
    
    # Crear modelo fine-tuned
    ollama create financial-assistant -f Modelfile
    
    success "Modelo fine-tuned 'financial-assistant' creado"
}

# Configurar variables de entorno
setup_environment() {
    log "Configurando variables de entorno..."
    
    # Agregar variables al archivo .env si existe
    if [[ -f ".env" ]]; then
        echo "" >> .env
        echo "# Configuración del LLM" >> .env
        echo "LLM_API_URL=http://localhost:11434" >> .env
        echo "LLM_MODEL_NAME=financial-assistant" >> .env
        echo "LLM_ENABLED=true" >> .env
        success "Variables de entorno agregadas a .env"
    else
        warning "Archivo .env no encontrado. Agrega manualmente:"
        echo "LLM_API_URL=http://localhost:11434"
        echo "LLM_MODEL_NAME=financial-assistant"
        echo "LLM_ENABLED=true"
    fi
}

# Probar la instalación
test_installation() {
    log "Probando la instalación..."
    
    # Verificar que Ollama esté corriendo
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        success "Ollama está corriendo correctamente"
    else
        error "Ollama no está respondiendo"
        exit 1
    fi
    
    # Verificar que el modelo esté disponible
    if ollama list | grep -q "llama3:8b"; then
        success "Modelo Llama 3 8B está disponible"
    else
        error "Modelo Llama 3 8B no está disponible"
        exit 1
    fi
    
    # Probar una consulta simple
    log "Probando consulta de prueba..."
    response=$(ollama run llama3:8b "Hola, ¿cómo estás?" 2>/dev/null | head -c 100)
    if [[ -n "$response" ]]; then
        success "Prueba de consulta exitosa"
    else
        warning "No se pudo probar la consulta"
    fi
}

# Función principal
main() {
    log "Iniciando instalación de Ollama y Llama 3..."
    
    detect_os
    check_requirements
    
    # Instalar Ollama según el sistema operativo
    if [[ "$OS" == "linux" ]]; then
        install_ollama_linux
    else
        install_ollama_macos
    fi
    
    # Descargar modelo
    download_llama3
    
    # Crear modelo fine-tuned (opcional)
    read -p "¿Quieres crear un modelo fine-tuned para el chatbot financiero? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        create_finetuned_model
    fi
    
    # Configurar entorno
    setup_environment
    
    # Probar instalación
    test_installation
    
    success "¡Instalación completada!"
    echo ""
    echo "📋 Próximos pasos:"
    echo "1. Reinicia tu aplicación Django"
    echo "2. El chatbot estará disponible en el botón flotante"
    echo "3. Para usar el modelo fine-tuned, cambia LLM_MODEL_NAME a 'financial-assistant'"
    echo ""
    echo "🔧 Comandos útiles:"
    echo "- ollama list                    # Ver modelos disponibles"
    echo "- ollama run llama3:8b 'texto'   # Probar modelo"
    echo "- ollama serve                   # Iniciar servidor Ollama"
    echo ""
}

# Ejecutar función principal
main "$@" 