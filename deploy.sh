#!/bin/bash

# Deploy script for Modal

set -e  # Exit on error

echo "Deploying to Modal..."

# Sync dependencies
echo "Syncing dependencies..."
uv sync

# Deploy to Modal
echo "Deploying app.py to Modal..."
uv run modal deploy app.py

echo ""
echo "Deployment complete!"
echo "Your app is now live on Modal."
