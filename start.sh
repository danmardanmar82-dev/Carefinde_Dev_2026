#!/bin/bash

# Start script for local HTML + Python Data App development

set -e  # Exit on error

# Set default port
APP_PORT=${APP_PORT:-3001}

echo "Starting HTML + Python Data App..."

# Sync dependencies from pyproject.toml
echo "Syncing dependencies..."
uv sync

# Start local development server
echo "Starting local dev server on http://0.0.0.0:${APP_PORT}"
echo ""
uv run uvicorn app:asgi --host 0.0.0.0 --port ${APP_PORT} --reload \
  --reload-exclude ".venv" --reload-exclude ".git" --reload-exclude "__pycache__" --reload-exclude "*.pyc"
