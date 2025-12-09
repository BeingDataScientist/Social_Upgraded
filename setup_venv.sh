#!/bin/bash

# Setup Virtual Environment in Root Folder
# This script creates a venv in the root directory and installs all dependencies

echo "========================================"
echo " Setting Up Virtual Environment"
echo "========================================"
echo

# Check if venv already exists
if [ -d "venv" ]; then
    echo "Virtual environment already exists at: venv"
    read -p "Do you want to recreate it? (y/n): " response
    if [ "$response" = "y" ]; then
        echo "Removing existing virtual environment..."
        rm -rf venv
    else
        echo "Using existing virtual environment."
        echo
        echo "To activate it, run:"
        echo "  source venv/bin/activate"
        exit 0
    fi
fi

echo "[1/3] Creating virtual environment in root folder..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment!"
    echo "Make sure Python 3 is installed and in your PATH."
    exit 1
fi

echo "Virtual environment created successfully!"

echo
echo "[2/3] Activating virtual environment..."
source venv/bin/activate

echo
echo "[3/3] Installing all dependencies from requirements.txt..."
echo "This may take a few minutes..."

# Upgrade pip first
python -m pip install --upgrade pip

# Install all requirements
pip install --no-cache-dir -r requirements.txt

if [ $? -eq 0 ]; then
    echo
    echo "========================================"
    echo " SUCCESS! Virtual environment is ready!"
    echo "========================================"
    echo
    echo "To activate the virtual environment, run:"
    echo "  source venv/bin/activate"
    echo
    echo "To run the application:"
    echo "  python app.py"
    echo
else
    echo
    echo "ERROR: Failed to install some dependencies!"
    echo "Please check the error messages above."
    exit 1
fi

