#!/bin/bash

# Setup script for FNO project on PACE
# Installs uv and prepares the virtual environment

# CRITICAL: Detect and remove Windows virtual environment if uploaded
if [ -d ".venv/Scripts" ]; then
    echo "Detected Windows .venv. Removing..."
    rm -rf .venv
fi

echo "Checking for uv..."

# Install uv if not in path
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing to ~/.local/bin..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Refresh path for current session
    export PATH="$HOME/.local/bin:$PATH"
fi

# Sync dependencies using pyproject.toml
echo "Syncing dependencies with uv..."
uv sync

echo "Environment setup complete."
