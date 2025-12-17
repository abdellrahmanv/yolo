#!/bin/bash

###############################################################################
# Raspberry Pi YOLOv5 PyTorch Detection - Setup Script
# Creates separate virtual environment (env_pt) for PyTorch inference
# Uses best.pt model
###############################################################################

set -e

echo "=============================================="
echo "  YOLOv5 PyTorch Setup"
echo "  Glasses Detection - Raspberry Pi"
echo "=============================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}Project: $PROJECT_ROOT${NC}"
echo -e "${BLUE}Virtual env: env_pt (PyTorch)${NC}"
echo ""

# ============================================
# Step 1: Check Python
# ============================================
echo -e "${GREEN}[1/7] Checking Python...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Python $PYTHON_VERSION"

# ============================================
# Step 2: Update system
# ============================================
echo ""
echo -e "${GREEN}[2/7] Updating system...${NC}"
sudo apt update
sudo apt upgrade -y

# ============================================
# Step 3: Install dependencies
# ============================================
echo ""
echo -e "${GREEN}[3/7] Installing system dependencies...${NC}"
sudo apt install -y \
    python3-dev \
    python3-venv \
    python3-pip \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    build-essential \
    cmake \
    git

# Camera
sudo apt install -y python3-picamera2 libcamera-apps || true
sudo apt install -y python3-picamera 2>/dev/null || true

# ============================================
# Step 4: Create virtual environment
# ============================================
echo ""
echo -e "${GREEN}[4/7] Creating virtual environment (env_pt)...${NC}"

if [ -d "env_pt" ]; then
    echo -e "${YELLOW}env_pt exists. Recreating...${NC}"
    rm -rf env_pt
fi

python3 -m venv env_pt --system-site-packages
echo -e "${GREEN}✓ env_pt created${NC}"

# ============================================
# Step 5: Activate and upgrade pip
# ============================================
echo ""
echo -e "${GREEN}[5/7] Activating environment...${NC}"
source env_pt/bin/activate

pip install --upgrade pip setuptools wheel

# ============================================
# Step 6: Install PyTorch and YOLOv5
# ============================================
echo ""
echo -e "${GREEN}[6/7] Installing PyTorch and YOLOv5...${NC}"

echo -e "${BLUE}Installing NumPy...${NC}"
pip install numpy>=1.21.0

echo -e "${BLUE}Installing PyTorch (CPU)...${NC}"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo -e "${BLUE}Installing OpenCV (with GUI)...${NC}"
pip install opencv-python>=4.5.0

echo -e "${BLUE}Installing YOLOv5 dependencies...${NC}"
pip install ultralytics
pip install Pillow pyyaml tqdm matplotlib seaborn pandas

# ============================================
# Step 7: Verify installation
# ============================================
echo ""
echo -e "${GREEN}[7/7] Verifying installation...${NC}"

python3 << 'EOF'
import sys
print("Testing packages...")

try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"  ✗ PyTorch: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"  ✓ OpenCV {cv2.__version__}")
except ImportError as e:
    print(f"  ✗ OpenCV: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"  ✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"  ✗ NumPy: {e}")
    sys.exit(1)

print("\n✓ All packages installed!")
EOF

# Check model
echo ""
if [ -f "$PROJECT_ROOT/model/best.pt" ]; then
    MODEL_SIZE=$(du -h "$PROJECT_ROOT/model/best.pt" | cut -f1)
    echo -e "${GREEN}✓ Model found: best.pt ($MODEL_SIZE)${NC}"
else
    echo -e "${RED}✗ Model not found: model/best.pt${NC}"
fi

# Create logs
mkdir -p "$PROJECT_ROOT/logs"

# ============================================
# Done
# ============================================
echo ""
echo "=============================================="
echo -e "${GREEN}✓ PyTorch setup complete!${NC}"
echo "=============================================="
echo ""
echo -e "${BLUE}Virtual Environment:${NC}"
echo "  Location: $PROJECT_ROOT/env_pt"
echo "  Activate: source $PROJECT_ROOT/env_pt/bin/activate"
echo ""
echo -e "${BLUE}Model:${NC}"
echo "  PyTorch: model/best.pt"
echo ""
echo -e "${BLUE}Run detection:${NC}"
echo "  ./scripts/run_detection_pt.sh"
echo ""
echo -e "${BLUE}Or manually:${NC}"
echo "  source env_pt/bin/activate"
echo "  python3 src/main_pt.py"
echo ""
