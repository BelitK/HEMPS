#!/usr/bin/env bash
set -e

# Step 1: Ensure uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Install it with:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Step 2: Create virtual environment if missing
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Step 3: Activate environment
source .venv/bin/activate

# Step 4: Sync dependencies using lockfile
if [ -f "uv.lock" ]; then
    echo "Syncing environment from uv.lock..."
    uv sync --locked
else
    echo "No uv.lock found. Installing from pyproject.toml..."
    uv pip install .
fi

echo "✅ Setup complete. Run your project with: python main.py"

## TODO: Add ollama installer