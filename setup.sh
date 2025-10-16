#!/bin/bash

# setup.sh - Tree Models Project Setup Script
# This script sets up the development environment for the tree-models package

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project information
PROJECT_NAME="tree-models"
PYTHON_MIN_VERSION="3.8"

# Functions
print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}                        🌳 Tree Models Setup${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_step() {
    echo -e "${GREEN}▶ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ Error: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
check_python() {
    print_step "Checking Python version..."
    
    if ! command_exists python3; then
        print_error "Python 3 is not installed. Please install Python ${PYTHON_MIN_VERSION}+ first."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    REQUIRED_VERSION=$(echo -e "${PYTHON_VERSION}\n${PYTHON_MIN_VERSION}" | sort -V | head -n1)
    
    if [[ "${REQUIRED_VERSION}" != "${PYTHON_MIN_VERSION}" ]]; then
        print_error "Python ${PYTHON_MIN_VERSION}+ is required. Found: ${PYTHON_VERSION}"
        exit 1
    fi
    
    print_success "Python ${PYTHON_VERSION} found"
}

# Setup virtual environment
setup_venv() {
    print_step "Setting up virtual environment..."
    
    if [[ -d "venv" ]]; then
        print_info "Virtual environment already exists"
        if [[ "$FORCE_RECREATE" == "true" ]]; then
            print_info "Recreating virtual environment..."
            rm -rf venv
        else
            return 0
        fi
    fi
    
    python3 -m venv venv
    print_success "Virtual environment created"
}

# Activate virtual environment
activate_venv() {
    print_step "Activating virtual environment..."
    
    # Check if we're already in a virtual environment
    if [[ -z "${VIRTUAL_ENV}" ]]; then
        source venv/bin/activate
        print_success "Virtual environment activated"
    else
        print_info "Already in virtual environment: ${VIRTUAL_ENV}"
    fi
}

# Upgrade pip and install build tools
upgrade_pip() {
    print_step "Upgrading pip and installing build tools..."
    
    pip install --upgrade pip setuptools wheel
    print_success "Build tools updated"
}

# Install project dependencies
install_dependencies() {
    local install_type="$1"
    
    print_step "Installing dependencies (${install_type} mode)..."
    
    case "${install_type}" in
        "minimal")
            pip install -e ".[minimal]"
            ;;
        "development")
            pip install -e ".[dev,test,docs,ml]"
            ;;
        "production")
            pip install -e .
            ;;
        "ml")
            pip install -e ".[ml]"
            ;;
        *)
            pip install -e ".[dev,test,docs]"
            ;;
    esac
    
    print_success "Dependencies installed"
}

# Install pre-commit hooks
setup_pre_commit() {
    print_step "Setting up pre-commit hooks..."
    
    if command_exists pre-commit; then
        pre-commit install
        pre-commit install --hook-type commit-msg
        print_success "Pre-commit hooks installed"
    else
        print_info "Installing pre-commit..."
        pip install pre-commit
        pre-commit install
        pre-commit install --hook-type commit-msg
        print_success "Pre-commit hooks installed"
    fi
}

# Setup development tools configuration
setup_dev_config() {
    print_step "Setting up development configuration..."
    
    # Create .vscode settings if it doesn't exist
    if [[ ! -d ".vscode" ]]; then
        mkdir -p .vscode
        cat > .vscode/settings.json << EOF
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "120"],
    "python.sortImports.args": ["--profile", "black", "--line-length", "120"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/node_modules": true
    }
}
EOF
        print_success "VS Code configuration created"
    fi
    
    # Create .editorconfig if it doesn't exist
    if [[ ! -f ".editorconfig" ]]; then
        cat > .editorconfig << EOF
root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true
indent_style = space
indent_size = 4

[*.{yml,yaml}]
indent_size = 2

[*.{json,js,ts}]
indent_size = 2

[*.md]
trim_trailing_whitespace = false
EOF
        print_success ".editorconfig created"
    fi
}

# Run initial tests
run_initial_tests() {
    print_step "Running initial tests..."
    
    # Run a quick test to verify installation
    if python -c "import tree_models; print('✓ Package imports successfully')" 2>/dev/null; then
        print_success "Package import test passed"
    else
        print_error "Package import failed"
        return 1
    fi
    
    # Run quick unit tests if they exist
    if [[ -d "tests" ]]; then
        if command_exists pytest; then
            pytest tests/unit/ -q --tb=no --maxfail=5 || {
                print_error "Some tests failed, but setup continues..."
            }
        fi
    fi
}

# Install optional ML libraries
install_ml_libraries() {
    print_step "Installing optional ML libraries..."
    
    local libraries=("xgboost" "lightgbm" "catboost" "shap" "mlflow" "plotly")
    local failed_libraries=()
    
    for lib in "${libraries[@]}"; do
        print_info "Installing ${lib}..."
        if pip install "${lib}" >/dev/null 2>&1; then
            print_success "${lib} installed"
        else
            print_error "Failed to install ${lib}"
            failed_libraries+=("${lib}")
        fi
    done
    
    if [[ ${#failed_libraries[@]} -gt 0 ]]; then
        print_info "Some libraries failed to install: ${failed_libraries[*]}"
        print_info "You can install them manually later with: pip install ${failed_libraries[*]}"
    fi
}

# Generate example configuration files
setup_examples() {
    print_step "Setting up example configurations..."
    
    # Create examples directory
    mkdir -p examples/configs
    
    # Basic configuration example
    cat > examples/configs/basic_config.yaml << 'EOF'
# Basic Tree Models Configuration
experiment:
  name: basic_example
  description: Basic configuration example

model:
  model_type: xgboost
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1

data:
  target_column: target
  test_size: 0.2
  validation_size: 0.2
  random_state: 42

preprocessing:
  scaling_strategy: standard
  missing_value_strategy:
    numeric: median
    categorical: most_frequent
EOF
    
    # Advanced configuration example
    cat > examples/configs/advanced_config.yaml << 'EOF'
# Advanced Tree Models Configuration
experiment:
  name: advanced_fraud_detection
  description: Advanced fraud detection pipeline

model:
  model_type: xgboost
  n_estimators: 200
  max_depth: 8
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
  reg_alpha: 0.1
  reg_lambda: 0.1

data:
  train_path: data/train.csv
  test_path: data/test.csv
  target_column: is_fraud
  test_size: 0.2
  validation_size: 0.2
  random_state: 42

features:
  numeric_features: [amount, balance, age, transaction_count]
  categorical_features: [merchant_category, card_type, channel]
  date_features: [transaction_date]

preprocessing:
  scaling_strategy: standard
  outlier_detection: true
  outlier_method: isolation
  missing_value_strategy:
    numeric: median
    categorical: most_frequent

feature_engineering:
  log_transform_cols: [amount, balance]
  create_ratios:
    - [amount, balance]
    - [transaction_count, age]
  extract_date_features: [transaction_date]
EOF

    print_success "Example configurations created in examples/configs/"
}

# Display completion message
show_completion() {
    echo
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}                    🎉 Setup Complete!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    echo -e "${YELLOW}Next steps:${NC}"
    echo -e "  1. Activate the virtual environment: ${BLUE}source venv/bin/activate${NC}"
    echo -e "  2. Run tests: ${BLUE}make test${NC}"
    echo -e "  3. Check code quality: ${BLUE}make lint${NC}"
    echo -e "  4. View examples: ${BLUE}ls examples/${NC}"
    echo -e "  5. Read documentation: ${BLUE}make docs${NC}"
    echo
    echo -e "${YELLOW}Available commands:${NC}"
    echo -e "  • ${BLUE}make help${NC} - Show all available commands"
    echo -e "  • ${BLUE}make test${NC} - Run all tests"
    echo -e "  • ${BLUE}make test-coverage${NC} - Run tests with coverage"
    echo -e "  • ${BLUE}make lint${NC} - Run code linting"
    echo -e "  • ${BLUE}make format${NC} - Format code"
    echo -e "  • ${BLUE}make docs${NC} - Build documentation"
    echo
    echo -e "${GREEN}Happy coding! 🚀${NC}"
    echo
}

# Parse command line arguments
INSTALL_TYPE="development"
SKIP_ML_LIBS=false
FORCE_RECREATE=false
SKIP_TESTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --minimal)
            INSTALL_TYPE="minimal"
            shift
            ;;
        --production)
            INSTALL_TYPE="production"
            shift
            ;;
        --ml-only)
            INSTALL_TYPE="ml"
            shift
            ;;
        --skip-ml)
            SKIP_ML_LIBS=true
            shift
            ;;
        --force-recreate)
            FORCE_RECREATE=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        -h|--help)
            echo "Tree Models Setup Script"
            echo
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  --minimal        Install minimal dependencies only"
            echo "  --production     Install production dependencies"
            echo "  --ml-only        Install ML libraries only"
            echo "  --skip-ml        Skip ML library installation"
            echo "  --force-recreate Force recreate virtual environment"
            echo "  --skip-tests     Skip running initial tests"
            echo "  -h, --help       Show this help message"
            echo
            echo "Examples:"
            echo "  $0                    # Full development setup"
            echo "  $0 --minimal         # Minimal installation"
            echo "  $0 --production      # Production installation"
            echo "  $0 --skip-ml         # Skip ML libraries"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_header
    
    # Core setup steps
    check_python
    setup_venv
    activate_venv
    upgrade_pip
    install_dependencies "${INSTALL_TYPE}"
    
    # Development-specific setup
    if [[ "${INSTALL_TYPE}" == "development" ]]; then
        setup_pre_commit
        setup_dev_config
        setup_examples
    fi
    
    # Optional ML libraries
    if [[ "${SKIP_ML_LIBS}" == false ]] && [[ "${INSTALL_TYPE}" != "minimal" ]]; then
        install_ml_libraries
    fi
    
    # Initial validation
    if [[ "${SKIP_TESTS}" == false ]]; then
        run_initial_tests
    fi
    
    show_completion
}

# Run main function
main "$@"