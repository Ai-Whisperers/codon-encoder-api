#!/bin/bash
# Codon Encoder API Deployment Script
# ====================================
# Usage: ./scripts/deploy.sh [command]
#
# Commands:
#   build     Build Docker image
#   start     Start the API server
#   stop      Stop the API server
#   logs      View logs
#   restart   Restart the server
#   tunnel    Start with Cloudflare Tunnel
#   health    Check API health

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.yml"
SERVICE_NAME="api"
API_URL="http://localhost:8765"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_model() {
    if [ ! -f "server/model/codon_encoder.pt" ]; then
        log_error "Model file not found: server/model/codon_encoder.pt"
        log_error "Please place the model file before starting the server."
        exit 1
    fi
    log_info "Model file found: server/model/codon_encoder.pt"
}

cmd_build() {
    log_info "Building Docker image..."
    docker compose build --no-cache
    log_info "Build complete."
}

cmd_start() {
    check_model
    log_info "Starting Codon Encoder API..."
    docker compose up -d
    log_info "API starting at ${API_URL}"
    log_info "Use 'docker compose logs -f' to view logs"
}

cmd_stop() {
    log_info "Stopping Codon Encoder API..."
    docker compose down
    log_info "API stopped."
}

cmd_logs() {
    docker compose logs -f ${SERVICE_NAME}
}

cmd_restart() {
    cmd_stop
    cmd_start
}

cmd_health() {
    log_info "Checking API health..."
    if curl -s "${API_URL}/api/metadata" > /dev/null 2>&1; then
        log_info "API is healthy"
        curl -s "${API_URL}/api/metadata" | python -m json.tool 2>/dev/null || cat
    else
        log_error "API is not responding"
        exit 1
    fi
}

cmd_tunnel() {
    check_model
    if [ -z "$CLOUDFLARE_TUNNEL_TOKEN" ]; then
        log_warn "CLOUDFLARE_TUNNEL_TOKEN not set"
        log_warn "Starting API without tunnel. Use cloudflared manually:"
        log_warn "  cloudflared tunnel --url ${API_URL}"
    fi
    log_info "Starting API with Cloudflare Tunnel..."
    docker compose --profile tunnel up -d
}

cmd_help() {
    echo "Codon Encoder API Deployment"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build     Build Docker image"
    echo "  start     Start the API server"
    echo "  stop      Stop the API server"
    echo "  logs      View logs (follow mode)"
    echo "  restart   Restart the server"
    echo "  tunnel    Start with Cloudflare Tunnel"
    echo "  health    Check API health"
    echo "  help      Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 build && $0 start"
    echo "  $0 health"
    echo "  $0 logs"
}

# Main
case "${1:-help}" in
    build)   cmd_build ;;
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    logs)    cmd_logs ;;
    restart) cmd_restart ;;
    tunnel)  cmd_tunnel ;;
    health)  cmd_health ;;
    help)    cmd_help ;;
    *)
        log_error "Unknown command: $1"
        cmd_help
        exit 1
        ;;
esac
