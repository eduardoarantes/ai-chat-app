#!/bin/bash
#
# Quick code formatting script
# Runs black and isort to format Python code
#

echo "🔧 Formatting Python code..."

echo "Running Black (code formatter)..."
black .

echo "Running isort (import sorter)..."  
isort .

echo "✅ Code formatting complete!"
echo ""
echo "To check formatting without fixing:"
echo "  black --check --diff ."
echo "  isort --check-only --diff ."