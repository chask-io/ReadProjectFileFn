#!/bin/bash
set -e

echo "Setting up ReadProjectFileFn..."

# Check Python version
python3 --version || { echo "Python 3 is required"; exit 1; }

# Install pre-commit if not installed
if ! command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit..."
    pip install pre-commit
fi

# Initialize git if not already
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
fi

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Install any project dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing project dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "Setup complete!"
echo "Pre-commit hooks will now run automatically before each commit."
