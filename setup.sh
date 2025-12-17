#!/usr/bin/env bash
set -e

# -------------------------------
# Help
# -------------------------------
show_help() {
    cat <<EOF
Usage: ./setup.sh [MODEL]

Sets up the Python environment and optionally installs an Ollama model.

Arguments:
  MODEL            Ollama model to pull (e.g. mistral:7b, llama3:8b)
                   Use "none" to skip model download
                   If omitted, you will be prompted interactively

Options:
  -h, --help       Show this help message and exit

Examples:
  ./setup.sh
  ./setup.sh mistral:7b
  ./setup.sh llama3:8b
  ./setup.sh none
EOF
}

case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
esac

MODEL_CHOICE="${1:-ask}"

# -------------------------------
# Step 0: Ensure Ollama is installed
# -------------------------------
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Installing..."

    OS="$(uname -s)"

    case "$OS" in
        Linux*)
            echo "Detected Linux"
            curl -fsSL https://ollama.com/install.sh | sh
            ;;
        Darwin*)
            echo "Detected macOS"
            echo "Installing Ollama via Homebrew..."
            if ! command -v brew &> /dev/null; then
                echo "Homebrew is required on macOS."
                echo "Install it from https://brew.sh"
                exit 1
            fi
            brew install ollama
            ;;
        *)
            echo "Unsupported OS: $OS"
            echo "Install Ollama manually from https://ollama.com"
            exit 1
            ;;
    esac

    echo "Ollama installed successfully."
else
    echo "Ollama already installed."
fi

# -------------------------------
# Step 1: Ensure uv is installed
# -------------------------------
if ! command -v uv &> /dev/null; then
    echo "uv is not installed."
    echo "Install it with:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# -------------------------------
# Step 2: Create virtual environment if missing
# -------------------------------
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# -------------------------------
# Step 3: Activate environment
# -------------------------------
source .venv/bin/activate

# -------------------------------
# Step 4: Sync dependencies
# -------------------------------
if [ -f "uv.lock" ]; then
    echo "Syncing environment from uv.lock..."
    uv sync --locked
else
    echo "No uv.lock found. Installing from pyproject.toml..."
    uv pip install .
fi

# -------------------------------
# Step 5: Optional Ollama model download
# -------------------------------
if [ "$MODEL_CHOICE" = "ask" ]; then
    echo
    echo "Choose an Ollama model to download:"
    echo "  1) mistral:7b (default)"
    echo "  2) llama3:8b"
    echo "  3) nous-hermes2:mistral"
    echo "  4) Skip model download"
    read -rp "Enter choice [1-4]: " choice

    case "$choice" in
        1|"") MODEL="mistral:7b" ;;
        2) MODEL="llama3:8b" ;;
        3) MODEL="nous-hermes2:mistral" ;;
        4) MODEL="none" ;;
        *) echo "Invalid choice. Skipping model download."; MODEL="none" ;;
    esac
else
    MODEL="$MODEL_CHOICE"
fi

if [ "$MODEL" != "none" ]; then
    if ! ollama list | grep -q "^$MODEL"; then
        echo "Pulling Ollama model: $MODEL"
        ollama pull "$MODEL"
    else
        echo "Model $MODEL already present."
    fi
else
    echo "Skipping Ollama model download."
fi

echo "✅ Setup complete. Run your project with: python main.py"
