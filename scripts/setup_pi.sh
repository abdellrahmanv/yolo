#!/bin/bash

###############################################################################
# Raspberry Pi YOLOv5 Detection Pipeline - Automated Installation Script
# Compatible with Raspberry Pi OS (Bullseye/Bookworm)
# Camera: Raspberry Pi Camera Module 1.3
###############################################################################

set -e  # Exit on error

echo "=============================================="
echo "  YOLOv5 Detection Pipeline Setup"
echo "  Raspberry Pi Camera 1.3"
echo "=============================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${GREEN}[1/10] Updating system packages...${NC}"
sudo apt update
sudo apt upgrade -y

echo -e "${GREEN}[2/10] Installing system dependencies...${NC}"
sudo apt install -y \
    python3-dev \
    python3-venv \
    python3-pip \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libcamera-dev \
    libcamera-apps \
    libcamera-tools \
    libopencv-dev \
    build-essential \
    cmake \
    libhdf5-dev \
    libhdf5-serial-dev \
    python3-h5py

echo -e "${GREEN}[3/10] Creating virtual environment...${NC}"
if [ -d "env" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Removing...${NC}"
    rm -rf env
fi
python3 -m venv env

echo -e "${GREEN}[4/10] Activating virtual environment...${NC}"
source env/bin/activate

echo -e "${GREEN}[5/10] Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

echo -e "${GREEN}[6/10] Installing core Python packages...${NC}"
# Install compatible versions for Python 3.13
pip install numpy
pip install opencv-python
pip install pillow

echo -e "${GREEN}[7/10] Installing Picamera (legacy)...${NC}"
pip install "picamera[array]"

echo -e "${GREEN}[8/10] Installing PyTorch (CPU version for Raspberry Pi)...${NC}"
# Install PyTorch CPU-only build optimized for ARM
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo -e "${GREEN}[9/10] Installing YOLOv5 dependencies...${NC}"
pip install ultralytics
pip install pyyaml
pip install tqdm
pip install matplotlib
pip install seaborn
pip install pandas

echo -e "${GREEN}[10/10] Verifying installation...${NC}"

# Test camera
echo -e "${YELLOW}Testing camera connection...${NC}"
if rpicam-still --timeout 100 -o "$PROJECT_ROOT/test.jpg" 2>/dev/null; then
    echo -e "${GREEN}✓ Camera detected and working!${NC}"
    rm -f "$PROJECT_ROOT/test.jpg"
else
    echo -e "${RED}✗ Camera not detected. Please check connections.${NC}"
    echo -e "${YELLOW}Enable camera using: sudo raspi-config${NC}"
fi

# Test Python imports
echo -e "${YELLOW}Testing Python imports...${NC}"
python3 << EOF
import sys
try:
    import torch
    print(f"✓ PyTorch {torch.__version__} installed")
    import cv2
    print(f"✓ OpenCV {cv2.__version__} installed")
    import numpy
    print(f"✓ NumPy {numpy.__version__} installed")
    from picamera import PiCamera
    print("✓ Picamera (legacy) installed")
    print("\n✓ All packages installed successfully!")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
EOF

# Check for model file
echo ""
echo -e "${YELLOW}Checking for YOLOv5 model...${NC}"
if [ -f "$PROJECT_ROOT/model/best.pt" ]; then
    echo -e "${GREEN}✓ Model file found: model/best.pt${NC}"
else
    echo -e "${RED}✗ Model file not found!${NC}"
    echo -e "${YELLOW}Please copy your trained best.pt file to:${NC}"
    echo -e "${YELLOW}  $PROJECT_ROOT/model/best.pt${NC}"
    echo ""
    echo -e "${YELLOW}Example: scp best.pt pi@raspberrypi:$PROJECT_ROOT/model/${NC}"
fi

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

# Save installed packages
pip freeze > "$PROJECT_ROOT/requirements.txt"

echo ""
echo "=============================================="
echo -e "${GREEN}✓ Installation completed successfully!${NC}"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Copy your trained model:"
echo "   scp best.pt pi@raspberrypi:$PROJECT_ROOT/model/"
echo ""
echo "2. Start detection:"
echo "   ./scripts/run_detection.sh"
echo ""
echo "Virtual environment: $PROJECT_ROOT/env"
echo "Activate with: source $PROJECT_ROOT/env/bin/activate"
echo ""
