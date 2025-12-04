#!/bin/bash

# Dia TTS Docker Helper Script
# This script simplifies common Docker operations for the Dia TTS project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
}

# Check if nvidia-docker is available
check_gpu() {
    if docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        print_info "GPU support detected"
        return 0
    else
        print_warning "GPU support not available. The container will run but inference will be slow."
        return 1
    fi
}

# Build the Docker image
build() {
    print_info "Building Dia TTS Docker image..."
    docker-compose build
    print_info "Build complete!"
}

# Run simple example
run_example() {
    print_info "Running simple example..."
    docker-compose run --rm dia-tts python example/simple.py
}

# Run Gradio web interface
run_gradio() {
    print_info "Starting Gradio web interface..."
    print_info "Access the interface at http://localhost:7860"
    docker-compose run --rm --service-ports dia-tts python app.py
}

# Run FastAPI server
run_api() {
    print_info "Starting FastAPI server..."
    print_info "Access API docs at http://localhost:8000/docs"
    docker-compose run --rm --service-ports dia-tts uvicorn api:app --host 0.0.0.0 --port 8000
}

# Run CLI
run_cli() {
    print_info "Opening Dia TTS CLI..."
    docker-compose run --rm dia-tts python cli.py "$@"
}

# Interactive shell
shell() {
    print_info "Starting interactive shell..."
    docker-compose run --rm dia-tts bash
}

# Clean up
clean() {
    print_info "Cleaning up Docker resources..."
    docker-compose down -v
    print_info "Cleanup complete!"
}

# Show help
show_help() {
    cat << EOF
Dia TTS Docker Helper Script

Usage: ./docker-helper.sh [command]

Commands:
    build       Build the Dia TTS Docker image
    example     Run the simple example
    gradio      Start the Gradio web interface
    api         Start the FastAPI server
    cli [args]  Run the CLI with optional arguments
    shell       Start an interactive shell in the container
    clean       Clean up Docker resources
    check       Check Docker and GPU support
    help        Show this help message

Examples:
    ./docker-helper.sh build
    ./docker-helper.sh example
    ./docker-helper.sh gradio
    ./docker-helper.sh api
    ./docker-helper.sh cli --help
    ./docker-helper.sh shell
    ./docker-helper.sh clean

EOF
}

# Main script
main() {
    check_docker

    case "${1:-help}" in
        build)
            build
            ;;
        example)
            run_example
            ;;
        gradio)
            run_gradio
            ;;
        api)
            run_api
            ;;
        cli)
            shift
            run_cli "$@"
            ;;
        shell)
            shell
            ;;
        clean)
            clean
            ;;
        check)
            check_gpu
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
