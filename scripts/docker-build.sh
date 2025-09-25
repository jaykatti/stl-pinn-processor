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