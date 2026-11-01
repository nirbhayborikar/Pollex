#!/usr/bin/env bash
# =============================================================================
# Pollex — Full Environment Setup
# =============================================================================
# Creates .venv, installs Pollex + external deps (LeRobot, LIBERO).
# Run once from the project root:  bash setup_env.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}[1/6]${NC} Creating virtual environment ..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo -e "${GREEN}[2/6]${NC} Upgrading pip ..."
pip install --upgrade pip setuptools wheel

# ── Clone LeRobot if missing ────────────────────────────────────────────────
LEROBOT_DIR="$PARENT_DIR/lerobot"
echo -e "${GREEN}[3/6]${NC} Setting up LeRobot ..."
if [ ! -d "$LEROBOT_DIR" ]; then
    echo "       Cloning huggingface/lerobot → $LEROBOT_DIR"
    git clone https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
fi
pip install -e "$LEROBOT_DIR"

# ── Clone LIBERO if missing ─────────────────────────────────────────────────
LIBERO_DIR="$PARENT_DIR/vla/LIBERO"
echo -e "${GREEN}[4/6]${NC} Setting up LIBERO ..."
if [ ! -d "$LIBERO_DIR" ]; then
    mkdir -p "$PARENT_DIR/vla"
    echo "       Cloning Lifelong-Robot-Learning/LIBERO → $LIBERO_DIR"
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"
fi
pip install -e "$LIBERO_DIR"

# ── Install Pollex itself ───────────────────────────────────────────────────
echo -e "${GREEN}[5/6]${NC} Installing Pollex and dependencies ..."
pip install -e ".[dev]"

# ── Display / MuJoCo hint ───────────────────────────────────────────────────
echo -e "${GREEN}[6/6]${NC} Done!"
echo ""
echo -e "${YELLOW}Before running any simulation or rollout command, set the display:${NC}"
echo ""
echo "  source $VENV_DIR/bin/activate"
echo '  export PYTHONPATH="$HOME/Documents/vla/LIBERO:$HOME/Documents/Pollex:$PYTHONPATH"'
echo '  export DISPLAY=:10'
echo '  export XAUTHORITY="$HOME/.Xauthority"'
echo '  unset MUJOCO_GL QT_DEBUG_PLUGINS QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH'
echo ""
echo -e "${GREEN}[+] Environment ready. Activate with:  source .venv/bin/activate${NC}"
