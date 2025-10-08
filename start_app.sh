#!/bin/bash

echo "========================================"
echo " Digital Media & Mental Health Assessment"
echo " Docker Container Startup Script"
echo "========================================"
echo

# Check if Docker is running
if ! docker version >/dev/null 2>&1; then
    echo "ERROR: Docker is not running or not installed!"
    echo "Please start Docker and try again."
    exit 1
fi

echo "[1/4] Stopping any existing containers..."
docker-compose down

echo "[2/4] Building Docker image..."
docker-compose build --no-cache

echo "[3/4] Starting the application container..."
docker-compose up -d

echo "[4/4] Waiting for application to start..."
sleep 10

# Check if container is running
if ! docker-compose ps | grep -q "Up"; then
    echo "ERROR: Container failed to start!"
    echo "Checking logs..."
    docker-compose logs
    exit 1
fi

echo
echo "========================================"
echo " SUCCESS! Application is running!"
echo "========================================"
echo
echo "Application URL: http://localhost:5000"
echo

# Try to open browser (works on macOS and Linux)
if command -v open >/dev/null 2>&1; then
    echo "Opening browser..."
    open http://localhost:5000
elif command -v xdg-open >/dev/null 2>&1; then
    echo "Opening browser..."
    xdg-open http://localhost:5000
else
    echo "Please open your browser and go to: http://localhost:5000"
fi

echo
echo "Container Status:"
docker-compose ps

echo
echo "To stop the application, run: docker-compose down"
echo "To view logs, run: docker-compose logs -f"
echo
