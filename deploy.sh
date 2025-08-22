#!/bin/bash

# Deployment script for Investment Engine

set -e

# Configuration
ENV=${1:-production}
VERSION=${2:-latest}

echo "Deploying Investment Engine - Environment: $ENV, Version: $VERSION"

# Load environment variables
if [ -f .env.$ENV ]; then
    export $(cat .env.$ENV | grep -v '^#' | xargs)
fi

# Build Docker images
echo "Building Docker images..."
docker-compose build

# Run database migrations
echo "Running database migrations..."
docker-compose run --rm app alembic upgrade head

# Deploy services
echo "Starting services..."
docker-compose up -d

# Health check
echo "Waiting for services to be healthy..."
sleep 10

# Check dashboard
curl -f http://localhost:8050/health || exit 1
echo "Dashboard is healthy"

# Check WebSocket
curl -f http://localhost:8765/health || exit 1
echo "WebSocket server is healthy"

echo "Deployment completed successfully!"

# Show logs
docker-compose logs --tail=50
