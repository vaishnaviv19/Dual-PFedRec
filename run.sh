#!/bin/bash
# run.sh - One-click PFedRec startup (Git Bash compatible)

set -e  # Exit on error

echo "🚀 PFedRec: Dual Personalization Federated Recommendation"
echo "=========================================================="

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
LOGS_DIR="${PROJECT_ROOT}/logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker Desktop first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose not found."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon not running. Start Docker Desktop."
        exit 1
    fi
    
    log_info "✓ Prerequisites satisfied"
}

# Prepare data directory
prepare_data() {
    log_info "Preparing data directory..."
    
    mkdir -p "${DATA_DIR}/ml-100k"
    mkdir -p "${LOGS_DIR}"
    
    # Check if MovieLens data exists
    if [ ! -f "${DATA_DIR}/ml-100k/u.data" ]; then
        log_warn "MovieLens-100K data not found at ${DATA_DIR}/ml-100k/u.data"
        log_info "Download from: https://grouplens.org/datasets/movielens/100k/"
        log_info "Then run: python data/prepare.py"
        return 1
    fi
    
    # Prepare client data splits
    if [ ! -f "${DATA_DIR}/client_1.csv" ]; then
        log_info "Generating client data splits..."
        python "${PROJECT_ROOT}/data/prepare.py"
    fi
    
    log_info "✓ Data ready"
}

# Build Docker images
build_images() {
    log_info "Building Docker images (this may take 5-10 minutes)..."
    
    cd "${PROJECT_ROOT}"
    docker-compose build --no-cache
    
    log_info "✓ Images built successfully"
}

# Start services
start_services() {
    log_info "Starting PFedRec services..."
    
    # Start with monitoring profile for TensorBoard
    docker-compose --profile monitoring up -d
    
    log_info "✓ Services started"
    echo ""
    echo "📊 Access Points:"
    echo "   Server API:  http://localhost:8000"
    echo "   Health:      http://localhost:8000/health"
    echo "   TensorBoard: http://localhost:6006"
    echo ""
    echo "📋 Monitor logs:"
    echo "   docker-compose logs -f server"
    echo "   docker-compose logs -f client_1"
    echo ""
    echo "🛑 Stop services: docker-compose down"
}

# Main execution
main() {
    check_prerequisites
    prepare_data || {
        log_warn "Skipping data preparation - ensure data is ready manually"
    }
    build_images
    start_services
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi