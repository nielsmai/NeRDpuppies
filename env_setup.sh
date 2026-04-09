#!/bin/bash
# =============================================================================
# Lightning AI Environment Setup Script
# Mirrors the Google Colab notebook for neural-robot-dynamics
# Run this once from your Lightning AI studio terminal:  bash setup_lightning.sh
# =============================================================================

set -e  # Exit immediately on any error

echo "============================================="
echo " Neural Robot Dynamics — Lightning AI Setup"
echo "============================================="

# -----------------------------------------------------------------------------
# 1. System dependencies (GUI / OpenGL / headless rendering)
# -----------------------------------------------------------------------------
echo ""
echo "[1/5] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y \
    freeglut3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    xvfb \
    --no-install-recommends
echo "      Done."

# -----------------------------------------------------------------------------
# 2. Python version check
#    Lightning AI studios typically ship Python 3.10+.
#    neural-robot-dynamics targets 3.8, but 3.10 is generally compatible.
#    If you hit issues, uncomment the conda block below to pin 3.8.
# -----------------------------------------------------------------------------
echo ""
echo "[2/5] Python version:"
python --version

# -- Optional: pin Python 3.8 via conda (uncomment if needed) ----------------
# conda create -n nrd python=3.8.20 -y
# conda activate nrd
# ---------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 3. Clone the repository (skip if already present)
# -----------------------------------------------------------------------------
echo ""
echo "[3/5] Cloning neural-robot-dynamics..."
if [ ! -d "neural-robot-dynamics" ]; then
    git clone https://github.com/NVlabs/neural-robot-dynamics.git
else
    echo "      Repo already exists, skipping clone."
fi
cd neural-robot-dynamics

# -----------------------------------------------------------------------------
# 4. PyTorch + Warp
#    cu121 matches CUDA 12.1 — Lightning AI studios default to CUDA 12.x.
#    Run `nvcc --version` to confirm; swap the index URL if needed.
# -----------------------------------------------------------------------------
echo ""
echo "[4/5] Installing PyTorch 2.2.2 (cu121) and Warp..."
pip install --quiet \
    torch==2.2.2 \
    torchvision==0.17.2 \
    torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121

pip install --quiet warp-lang
echo "      Done."

# -----------------------------------------------------------------------------
# 5. Project requirements
# -----------------------------------------------------------------------------
echo ""
echo "[5/5] Installing project requirements..."
pip install --quiet -r requirements.txt
echo "      Done."

# -----------------------------------------------------------------------------
# Quick smoke-test: headless visualizer (same as Colab cell 3)
# -----------------------------------------------------------------------------
echo ""
echo "============================================="
echo " Setup complete. Running headless smoke test..."
echo "============================================="
cd utils
xvfb-run -a python visualize_env.py --env-name Cartpole --num-envs 1

echo ""
echo "All done! Your Lightning AI environment is ready."