#!/bin/bash

###############################################################################
# YOLOv5 PyTorch Detection - Runtime Launcher
# Uses env_pt virtual environment and best.pt model
###############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "  YOLOv5 PyTorch Glasses Detection"
echo "=============================================="
echo ""

# Check env exists
if [ ! -d "env_pt" ]; then
    echo -e "${RED}✗ env_pt not found!${NC}"
    echo -e "${YELLOW}Run setup first: ./scripts/setup_pi_pt.sh${NC}"
    exit 1
fi

# Activate
echo -e "${GREEN}Activating env_pt...${NC}"
source env_pt/bin/activate

# Check model
if [ ! -f "model/best.pt" ]; then
    echo -e "${RED}✗ Model not found: model/best.pt${NC}"
    exit 1
fi

# Verify PyTorch
echo -e "${GREEN}Verifying PyTorch...${NC}"
python3 << 'EOF'
import sys
try:
    import torch
    import cv2
    print("✓ Dependencies OK")
except ImportError as e:
    print(f"✗ Missing: {e}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""
echo -e "${BLUE}Model: model/best.pt${NC}"
echo -e "${BLUE}Resolution: 320x320${NC}"
echo ""
echo "Starting detection..."
echo -e "${YELLOW}Press 'q' or Ctrl+C to stop${NC}"
echo ""

# Parse args
ARGS=""
if [ "$1" == "--headless" ] || [ "$1" == "-H" ]; then
    ARGS="--headless"
    echo -e "${BLUE}Running headless${NC}"
fi

# Run
python3 src/main_pt.py $ARGS

echo ""
echo -e "${GREEN}Stopped.${NC}"
