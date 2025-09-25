#!/bin/bash
set -e

echo "🧪 Running STL-PINN Processor Test Suite"
echo "========================================"

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Run code quality checks
echo "🔍 Running code quality checks..."

echo "  - Black formatting check..."
black --check src/ tests/ || {
    echo "❌ Code formatting issues found. Run: black src/ tests/"
    exit 1
}

echo "  - Import sorting check..."
isort --check-only src/ tests/ || {
    echo "❌ Import sorting issues found. Run: isort src/ tests/"
    exit 1
}

echo "  - Flake8 linting..."
flake8 src/ tests/ || {
    echo "❌ Linting issues found"
    exit 1
}

echo "  - MyPy type checking..."
mypy src/ || {
    echo "⚠️ Type checking issues found (continuing...)"
}

echo "  - Bandit security scan..."
bandit -r src/ -x tests/ || {
    echo "⚠️ Security issues found (continuing...)"
}

# Run tests
echo "🚀 Running tests..."

# Unit tests
echo "  - Unit tests..."
pytest tests/ -m "unit" -v --cov=src --cov-report=term-missing

# Integration tests
echo "  - Integration tests..."
pytest tests/ -m "integration" -v

# All tests with coverage
echo "  - Full test suite with coverage..."
pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=80

echo "✅ All tests passed!"
echo "📊 Coverage report available at: htmlcov/index.html"

# ===== scripts/docker-build.sh =====
#!/bin/bash
set -e

echo "🐳 Building STL-PINN Processor Docker Images"
echo "============================================"

# Get version from setup.py or default
VERSION=${1:-"latest"}
IMAGE_NAME="stl-pinn-processor"
REGISTRY=${REGISTRY:-"your-org"}

echo "📦 Building version: $VERSION"

# Build production image
echo "🏭 Building production image..."
docker build -f deployment/docker/Dockerfile -t $REGISTRY/$IMAGE_NAME:$VERSION .
docker build -f deployment/docker/Dockerfile -t $REGISTRY/$IMAGE_NAME:latest .

# Build development image
echo "🔧 Building development image..."
docker build -f deployment/docker/Dockerfile.dev -t $REGISTRY/$IMAGE_NAME:dev .

# Test the images
echo "🧪 Testing images..."
docker run --rm $REGISTRY/$IMAGE_NAME:$VERSION python -c "import src; print('✅ Production image works')"
docker run --rm $REGISTRY/$IMAGE_NAME:dev python -c "import src; print('✅ Development image works')"

echo "✅ Docker images built successfully!"
echo "🚀 To push to registry:"
echo "  docker push $REGISTRY/$IMAGE_NAME:$VERSION"
echo "  docker push $REGISTRY/$IMAGE_NAME:latest"
echo "  docker push $REGISTRY/$IMAGE_NAME:dev"