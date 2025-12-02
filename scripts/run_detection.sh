#!/bin/bash

###############################################################################
# YOLOv5 Detection Pipeline - Runtime Launcher
# Activates virtual environment and starts detection
###############################################################################

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "  Starting YOLOv5 Detection Pipeline"
echo "=============================================="
echo ""

# Check virtual environment exists
if [ ! -d "env" ]; then
    echo -e "${RED}✗ Virtual environment not found!${NC}"
    echo -e "${YELLOW}Run setup first: ./scripts/setup_pi.sh${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source env/bin/activate

# Check model exists
if [ ! -f "model/best.pt" ]; then
    echo -e "${RED}✗ Model file not found: model/best.pt${NC}"
    echo -e "${YELLOW}Copy your model: scp best.pt pi@raspberrypi:$PROJECT_ROOT/model/${NC}"
    exit 1
fi

# Verify dependencies
echo -e "${GREEN}Verifying dependencies...${NC}"
python3 << EOF
import sys
try:
    import torch
    import cv2
    print("✓ All dependencies verified")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("Run setup script: ./scripts/setup_pi.sh")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""
echo "Starting detection pipeline..."
echo "Press Ctrl+C to stop"
echo ""

# Run main detection script
python3 src/main.py

echo ""
echo -e "${GREEN}Detection pipeline stopped.${NC}"
