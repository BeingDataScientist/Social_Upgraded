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

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p ml_model/artifacts ml_model/reports ml_model/visualizations

# Train the ML model during build (if data is available)
RUN if [ -f "questionnaire_data.csv" ]; then python ml_model/ml_training.py; fi

# Expose port 5000
EXPOSE 5000

# Set the default command
CMD ["python", "app.py"]
