#!/bin/bash
# setup.sh - Development Environment Setup Script
# This script sets up the complete development environment for the Tree Models project

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Project information
PROJECT_NAME="Tree Models"
PYTHON_MIN_VERSION="3.8"
VENV_NAME="venv"

print_banner() {
    echo "=================================="
    echo "  $PROJECT_NAME Setup Script"
    echo "=================================="
    echo ""
}

check_python() {
    log_info "Checking Python installation..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed!"
        echo "Please install Python 3.8 or higher from https://python.org"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    log_info "Found Python $PYTHON_VERSION"
    
    # Simple version comparison
    if [[ "$PYTHON_VERSION" < "$PYTHON_MIN_VERSION" ]]; then
        log_error "Python $PYTHON_MIN_VERSION or higher is required"
        exit 1
    fi
    
    log_success "Python version check passed"
}

check_git() {
    log_info "Checking Git installation..."
    
    if ! command -v git &> /dev/null; then
        log_error "Git is not installed!"
        echo "Please install Git from https://git-scm.com"
        exit 1
    fi
    
    GIT_VERSION=$(git --version | cut -d' ' -f3)
    log_info "Found Git $GIT_VERSION"
    log_success "Git check passed"
}

create_virtual_environment() {
    log_info "Creating virtual environment..."
    
    if [ -d "$VENV_NAME" ]; then
        log_warning "Virtual environment already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_NAME"
            log_info "Removed existing virtual environment"
        else
            log_info "Using existing virtual environment"
            return
        fi
    fi
    
    python3 -m venv "$VENV_NAME"
    log_success "Virtual environment created"
}

activate_virtual_environment() {
    log_info "Activating virtual environment..."
    source "$VENV_NAME/bin/activate"
    log_success "Virtual environment activated"
}

upgrade_pip() {
    log_info "Upgrading pip, setuptools, and wheel..."
    python -m pip install --upgrade pip setuptools wheel
    log_success "Package managers upgraded"
}

install_dependencies() {
    log_info "Installing project dependencies..."
    
    # Install in development mode with all extras
    pip install -e ".[dev,docs,all]"
    
    log_success "Dependencies installed"
}

install_pre_commit_hooks() {
    log_info "Installing pre-commit hooks..."
    
    if command -v pre-commit &> /dev/null; then
        pre-commit install
        pre-commit install --hook-type commit-msg
        log_success "Pre-commit hooks installed"
    else
        log_warning "pre-commit not found, skipping hook installation"
    fi
}

create_directories() {
    log_info "Creating project directories..."
    
    directories=(
        "reports"
        "logs" 
        "data/raw"
        "data/processed"
        "models/saved"
        "outputs"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        log_info "Created directory: $dir"
    done
    
    log_success "Project directories created"
}

setup_git_hooks() {
    log_info "Setting up Git configuration..."
    
    # Set up Git hooks directory if it doesn't exist
    if [ -d ".git" ]; then
        mkdir -p .git/hooks
        
        # Create simple pre-push hook
        cat > .git/hooks/pre-push << 'EOF'
#!/bin/bash
# Simple pre-push hook to run tests

echo "Running pre-push checks..."
make test-unit
if [ $? -ne 0 ]; then
    echo "Tests failed. Push aborted."
    exit 1
fi
echo "Pre-push checks passed!"
EOF
        chmod +x .git/hooks/pre-push
        log_success "Git hooks configured"
    else
        log_warning "Not a Git repository, skipping Git hooks setup"
    fi
}

create_env_file() {
    log_info "Creating environment configuration..."
    
    if [ ! -f ".env.example" ]; then
        cat > .env.example << 'EOF'
# Tree Models Environment Configuration
# Copy this file to .env and update values as needed

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT=tree_models_dev

# Data Configuration
TRAIN_PATH=data/raw/train.csv
TEST_PATH=data/raw/test.csv
OUTPUT_DIR=outputs

# Model Configuration
MODEL_TYPE=xgboost
N_ESTIMATORS=200
MAX_DEPTH=6
LEARNING_RATE=0.1

# Development Settings
DEBUG=true
VERBOSE=true
RANDOM_SEED=42
EOF
        log_success "Created .env.example file"
    fi
    
    if [ ! -f ".env" ]; then
        cp .env.example .env
        log_success "Created .env file from example"
    fi
}

run_initial_tests() {
    log_info "Running initial test suite..."
    
    # Run quick unit tests to verify setup
    if python -m pytest tests/ -m "unit and not slow" --tb=short -q; then
        log_success "Initial tests passed!"
    else
        log_warning "Some tests failed, but setup is complete"
        log_info "Run 'make test' to see detailed test results"
    fi
}

show_next_steps() {
    log_success "Setup completed successfully!"
    echo ""
    echo "🎉 $PROJECT_NAME development environment is ready!"
    echo ""
    echo "Next steps:"
    echo "  1. Activate the virtual environment:"
    echo "     source $VENV_NAME/bin/activate"
    echo ""
    echo "  2. Verify the installation:"
    echo "     make test"
    echo ""
    echo "  3. Try the configuration demo:"
    echo "     make demo-config"
    echo ""
    echo "  4. Run code quality checks:"
    echo "     make check"
    echo ""
    echo "  5. Build documentation:"
    echo "     make docs"
    echo ""
    echo "Available commands (run 'make help' for full list):"
    echo "  make test          - Run all tests"
    echo "  make lint          - Run code linting"
    echo "  make format        - Format code"
    echo "  make build         - Build package"
    echo "  make docs          - Build documentation"
    echo ""
    echo "Configuration files:"
    echo "  - .env            - Environment variables"
    echo "  - pyproject.toml  - Project configuration"
    echo "  - Makefile        - Development commands"
    echo ""
}

cleanup_on_error() {
    log_error "Setup failed!"
    log_info "Cleaning up..."
    
    if [ -d "$VENV_NAME" ]; then
        rm -rf "$VENV_NAME"
        log_info "Removed virtual environment"
    fi
    
    exit 1
}

main() {
    print_banner
    
    # Set up error handling
    trap cleanup_on_error ERR
    
    # Run setup steps
    check_python
    check_git
    create_virtual_environment
    activate_virtual_environment
    upgrade_pip
    install_dependencies
    install_pre_commit_hooks
    create_directories
    setup_git_hooks
    create_env_file
    
    # Optional steps
    log_info "Running optional verification steps..."
    run_initial_tests || true  # Don't fail on test failures
    
    show_next_steps
}

# Check if running with specific flags
case "${1:-}" in
    --minimal)
        log_info "Running minimal setup (no optional components)..."
        print_banner
        check_python
        create_virtual_environment
        activate_virtual_environment
        upgrade_pip
        install_dependencies
        log_success "Minimal setup complete!"
        ;;
    --help|-h)
        echo "Usage: $0 [OPTION]"
        echo ""
        echo "Setup script for $PROJECT_NAME development environment"
        echo ""
        echo "Options:"
        echo "  --minimal    Run minimal setup without optional components"
        echo "  --help, -h   Show this help message"
        echo ""
        echo "Default behavior runs full setup including:"
        echo "  - Python and Git verification"
        echo "  - Virtual environment creation"
        echo "  - Dependency installation"
        echo "  - Pre-commit hooks"
        echo "  - Directory structure"
        echo "  - Git hooks"
        echo "  - Environment files"
        echo "  - Initial test run"
        ;;
    *)
        main
        ;;
esac