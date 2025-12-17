#!/bin/bash

###############################################################################
# YOLOv5 PyTorch Detection - Runtime Launcher
# Uses original best.pt model for reliable detection
###############################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "  YOLOv5 PyTorch Glasses Detection"
echo "=============================================="
echo ""

# Check venv
if [ ! -d "env" ]; then
    echo -e "${RED}✗ Virtual environment not found!${NC}"
    echo -e "${YELLOW}Run: ./scripts/setup_pi.sh${NC}"
    exit 1
fi

# Activate
echo -e "${GREEN}Activating virtual environment...${NC}"
source env/bin/activate

# Check model
if [ ! -f "model/best.pt" ]; then
    echo -e "${RED}✗ Model not found: model/best.pt${NC}"
    exit 1
fi

# Verify torch
python3 << 'EOF'
import sys
try:
    import torch
    import cv2
    print("✓ Dependencies OK")
except ImportError as e:
    print(f"✗ Missing: {e}")
    print("Run: pip install torch torchvision")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""
echo -e "Model: model/best.pt (PyTorch)"
echo -e "${YELLOW}Press 'q' or Ctrl+C to stop${NC}"
echo ""

# Parse args
ARGS=""
if [ "$1" == "--headless" ] || [ "$1" == "-H" ]; then
    ARGS="--headless"
    echo "Running headless..."
fi

# Run
python3 src/main_pytorch.py $ARGS

echo ""
echo -e "${GREEN}Done.${NC}"
