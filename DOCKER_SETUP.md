# Docker Setup Guide

This guide will help you run the Digital Media & Mental Health Assessment application using Docker.

## Prerequisites

### 1. Install Docker Desktop

**For Windows:**
1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop/
2. Install Docker Desktop
3. Start Docker Desktop
4. Verify installation by opening Command Prompt and running:
   ```cmd
   docker --version
   ```

**For macOS:**
1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop/
2. Install Docker Desktop
3. Start Docker Desktop
4. Verify installation by opening Terminal and running:
   ```bash
   docker --version
   ```

**For Linux:**
1. Follow the official Docker installation guide: https://docs.docker.com/engine/install/
2. Start Docker service:
   ```bash
   sudo systemctl start docker
   sudo systemctl enable docker
   ```
3. Verify installation:
   ```bash
   docker --version
   ```

## Quick Start

### Option 1: Using Startup Scripts (Recommended)

**Windows:**
```cmd
# Double-click start_app.bat or run in Command Prompt
start_app.bat
```

**macOS/Linux:**
```bash
# Make script executable and run
chmod +x start_app.sh
./start_app.sh
```

**PowerShell (Windows):**
```powershell
# Run PowerShell script
.\start_app.ps1
```

### Option 2: Manual Docker Commands

1. **Build the Docker image:**
   ```bash
   docker-compose build
   ```

2. **Start the application:**
   ```bash
   docker-compose up -d
   ```

3. **Open your browser and go to:**
   ```
   http://localhost:5000
   ```

4. **Stop the application:**
   ```bash
   docker-compose down
   ```

## What the Scripts Do

### Startup Scripts (`start_app.*`)
1. Check if Docker is running
2. Stop any existing containers
3. Build the Docker image
4. Start the application container
5. Wait for the application to start
6. Open your browser automatically
7. Show container status

### Stop Scripts (`stop_app.*`)
1. Stop and remove the Docker containers
2. Clean up resources

## Docker Files Explained

### `Dockerfile`
- Uses Python 3.9 slim image
- Installs system dependencies
- Copies requirements and installs Python packages
- Sets up the application environment
- Exposes port 5000
- Runs the Flask application

### `docker-compose.yml`
- Defines the service configuration
- Maps port 5000 from container to host
- Sets up environment variables
- Includes health checks
- Enables automatic restart

### `.dockerignore`
- Excludes unnecessary files from Docker build context
- Reduces build time and image size

## Troubleshooting

### Docker Not Found
- Make sure Docker Desktop is installed and running
- Restart your terminal/command prompt
- Check if Docker is in your system PATH

### Port Already in Use
- Stop any existing Flask applications running on port 5000
- Or modify the port in `docker-compose.yml`:
  ```yaml
  ports:
    - "5001:5000"  # Use port 5001 instead
  ```

### Container Won't Start
- Check Docker logs:
  ```bash
  docker-compose logs
  ```
- Rebuild the image:
  ```bash
  docker-compose build --no-cache
  ```

### Permission Issues (Linux/macOS)
- Make sure your user is in the docker group:
  ```bash
  sudo usermod -aG docker $USER
  ```
- Log out and log back in

## Development Mode

For development with live code changes:

1. **Start with volume mounting:**
   ```bash
   docker-compose up -d
   ```

2. **View logs in real-time:**
   ```bash
   docker-compose logs -f
   ```

3. **Rebuild after code changes:**
   ```bash
   docker-compose build
   docker-compose up -d
   ```

## Production Deployment

For production deployment:

1. **Update `docker-compose.yml`:**
   ```yaml
   environment:
     - FLASK_ENV=production
     - FLASK_DEBUG=0
   ```

2. **Use a reverse proxy (nginx) for better performance**

3. **Set up proper logging and monitoring**

## Container Management

### View Running Containers
```bash
docker-compose ps
```

### View Container Logs
```bash
docker-compose logs -f
```

### Access Container Shell
```bash
docker-compose exec social-assessment bash
```

### Remove Everything
```bash
docker-compose down -v --rmi all
```

## Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Verify Docker is running: `docker --version`
3. Check container logs: `docker-compose logs`
4. Ensure all files are present in the project directory
