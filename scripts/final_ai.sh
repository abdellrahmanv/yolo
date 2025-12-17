#!/bin/bash

###############################################################################
# YOLOv5 Final AI - Glasses Detection with LCD and Buzzer
# Complete system with hardware feedback
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
echo "  YOLOv5 Final AI - Glasses Detection"
echo "  With LCD Display and Buzzer Feedback"
echo "=============================================="
echo ""

# Check env exists
if [ ! -d "env_pt" ]; then
    echo -e "${RED}✗ env_pt not found!${NC}"
    echo -e "${YELLOW}Run setup first: ./scripts/setup_pi_pt.sh${NC}"
    exit 1
fi

# Activate environment
echo "Activating env_pt..."
source env_pt/bin/activate

# Check for hardware dependencies
echo "Checking hardware dependencies..."

# Check RPi.GPIO
if ! python -c "import RPi.GPIO" 2>/dev/null; then
    echo -e "${YELLOW}Installing RPi.GPIO...${NC}"
    pip install RPi.GPIO
fi

# Check RPLCD for LCD
if ! python -c "from RPLCD.i2c import CharLCD" 2>/dev/null; then
    echo -e "${YELLOW}Installing RPLCD...${NC}"
    pip install RPLCD
fi

# Check smbus for I2C
if ! python -c "import smbus2" 2>/dev/null; then
    echo -e "${YELLOW}Installing smbus2...${NC}"
    pip install smbus2
fi

echo -e "${GREEN}✓ Dependencies OK${NC}"
echo ""

# Enable I2C if not already
if [ ! -e /dev/i2c-1 ]; then
    echo -e "${YELLOW}Warning: I2C not enabled. Run: sudo raspi-config -> Interface -> I2C${NC}"
fi

# Check model
if [ ! -f "model/best.pt" ]; then
    echo -e "${RED}✗ Model not found: model/best.pt${NC}"
    exit 1
fi

echo "Model: model/best.pt"
echo ""
echo "Hardware Configuration:"
echo "  LCD 16x2 I2C: SDA=GPIO2, SCL=GPIO3"
echo "  Buzzer: GPIO18"
echo ""

# Parse arguments
HEADLESS=""
NO_HW=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --headless|-H)
            HEADLESS="--headless"
            echo "Mode: Headless (no display)"
            shift
            ;;
        --no-hardware|-N)
            NO_HW="--no-hardware"
            echo "Mode: No hardware"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo ""
echo "Starting Final AI detection..."
echo "Press 'q' or Ctrl+C to stop"
echo ""

# Run detection
python src/main_final.py $HEADLESS $NO_HW

echo ""
echo -e "${GREEN}Detection stopped${NC}"
