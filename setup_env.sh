#!/usr/bin/env bash
# Creates .venv and installs all Pollex dependencies.
# Usage: bash setup_env.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

echo "[*] Activating virtual environment ..."
source "$VENV_DIR/bin/activate"

echo "[*] Upgrading pip ..."
pip install --upgrade pip setuptools wheel

echo "[*] Installing project dependencies ..."
pip install -e ".[dev]"

echo "[+] Done. Activate with:  source .venv/bin/activate"
