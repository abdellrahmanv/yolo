#!/bin/bash

###############################################################################
# YOLOv5 TFLite Detection - Runtime Launcher
# Activates virtual environment and starts glasses detection
###############################################################################

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "  YOLOv5 TFLite Glasses Detection"
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
if [ ! -f "model/best-int8.tflite" ]; then
    echo -e "${RED}✗ TFLite model not found: model/best-int8.tflite${NC}"
    echo -e "${YELLOW}Please ensure the model file is in place${NC}"
    exit 1
fi

# Verify TFLite runtime
echo -e "${GREEN}Verifying dependencies...${NC}"
python3 << 'EOF'
import sys
try:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
    import cv2
    import numpy
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
echo -e "${BLUE}Model: model/best-int8.tflite${NC}"
echo -e "${BLUE}Resolution: 320x320${NC}"
echo ""
echo "Starting detection pipeline..."
echo -e "${YELLOW}Press 'q' or Ctrl+C to stop${NC}"
echo ""

# Parse arguments
ARGS=""
if [ "$1" == "--headless" ] || [ "$1" == "-H" ]; then
    ARGS="--headless"
    echo -e "${BLUE}Running in headless mode (no display)${NC}"
fi

# Run main detection script
python3 src/main.py $ARGS

echo ""
echo -e "${GREEN}Detection pipeline stopped.${NC}"
