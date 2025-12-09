# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY ml_model/requirements.txt ml_model/requirements.txt

# Install Python dependencies (main app)
RUN pip install --no-cache-dir -r requirements.txt

# Install ML model dependencies
RUN pip install --no-cache-dir -r ml_model/requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p ml_model/artifacts ml_model/reports ml_model/visualizations instance

# Initialize database (will be created on first run)
# Note: Database file will be created in /app/instance/health_assessment.db

# Expose port 5000
EXPOSE 5000

# Set the default command
CMD ["python", "app.py"]
