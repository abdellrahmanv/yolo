#!/bin/bash

###############################################################################
# Raspberry Pi YOLOv5 TFLite Detection - Automated Installation Script
# Optimized for TFLite FP16 inference with Pi Camera support
# Compatible with Raspberry Pi OS Bullseye/Bookworm
###############################################################################

set -e  # Exit on error

echo "=============================================="
echo "  YOLOv5 TFLite Detection Setup"
echo "  Human Detection - Raspberry Pi"
echo "=============================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}Project directory: $PROJECT_ROOT${NC}"
echo ""

# ============================================
# Step 1: Check Python Version
# ============================================
echo -e "${GREEN}[1/9] Checking Python version...${NC}"

PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

echo "Found Python $PYTHON_VERSION"

# TFLite runtime works best with Python 3.9-3.11
if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 9 ] && [ "$PYTHON_MINOR" -le 11 ]; then
    echo -e "${GREEN}âœ“ Python version compatible with TFLite${NC}"
else
    echo -e "${YELLOW}âš  Python $PYTHON_VERSION detected. TFLite works best with Python 3.9-3.11${NC}"
    echo -e "${YELLOW}  Consider installing Python 3.11 if you encounter issues${NC}"
fi

# ============================================
# Step 2: Update System
# ============================================
echo ""
echo -e "${GREEN}[2/9] Updating system packages...${NC}"
sudo apt update
sudo apt upgrade -y

# ============================================
# Step 3: Install System Dependencies
# ============================================
echo ""
echo -e "${GREEN}[3/9] Installing system dependencies...${NC}"
sudo apt install -y \
    python3-dev \
    python3-venv \
    python3-pip \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    build-essential \
    cmake

# Install camera libraries
echo -e "${GREEN}Installing camera libraries...${NC}"
sudo apt install -y \
    python3-picamera2 \
    libcamera-apps \
    libcamera-dev \
    || echo -e "${YELLOW}Some camera packages may not be available (this is OK for USB cameras)${NC}"

# Try to install legacy picamera for Camera Module 1.3
sudo apt install -y python3-picamera 2>/dev/null || true

# ============================================
# Step 4: Create Virtual Environment
# ============================================
echo ""
echo -e "${GREEN}[4/9] Creating virtual environment...${NC}"

if [ -d "env" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Recreating...${NC}"
    rm -rf env
fi

# Create venv with system site packages (for picamera2 access)
python3 -m venv env --system-site-packages

echo -e "${GREEN}âœ“ Virtual environment created${NC}"

# ============================================
# Step 5: Activate Virtual Environment
# ============================================
echo ""
echo -e "${GREEN}[5/9] Activating virtual environment...${NC}"
source env/bin/activate

# Verify activation
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${RED}âœ— Failed to activate virtual environment${NC}"
    exit 1
fi

echo -e "${GREEN}âœ“ Virtual environment activated: $VIRTUAL_ENV${NC}"

# ============================================
# Step 6: Upgrade pip
# ============================================
echo ""
echo -e "${GREEN}[6/9] Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

# ============================================
# Step 7: Install TFLite Runtime & Dependencies
# ============================================
echo ""
echo -e "${GREEN}[7/9] Installing TFLite runtime and dependencies...${NC}"

# Install numpy first (required for TFLite)
pip install numpy>=1.21.0

# Install TFLite Runtime
# Python 3.12+ removed the 'imp' module, so tflite-runtime won't work.
# Use ai-edge-litert (Google's official replacement) for Python 3.12+.
echo -e "${BLUE}Installing TFLite Runtime...${NC}"

PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MINOR" -ge 12 ]; then
    echo -e "${BLUE}Python 3.12+ detected â€” installing ai-edge-litert...${NC}"
    pip install ai-edge-litert || {
        echo -e "${YELLOW}ai-edge-litert failed, trying tflite-runtime anyway...${NC}"
        pip install tflite-runtime || {
            echo -e "${YELLOW}Falling back to full TensorFlow...${NC}"
            pip install tensorflow
        }
    }
else
    echo -e "${BLUE}Python 3.${PYTHON_MINOR} â€” installing tflite-runtime...${NC}"
    pip install tflite-runtime || {
        echo -e "${YELLOW}tflite-runtime failed, trying ai-edge-litert...${NC}"
        pip install ai-edge-litert || {
            echo -e "${YELLOW}Falling back to full TensorFlow...${NC}"
            pip install tensorflow
        }
    }
fi

# Note: PyTorch is NOT installed here - use env_pt for PyTorch
# This keeps the TFLite environment lightweight

# Install OpenCV (full version with GUI support for display window)
echo -e "${BLUE}Installing OpenCV (with GUI support)...${NC}"
pip install opencv-python>=4.5.0

# Install Pillow for image processing
pip install Pillow>=9.0.0

# ============================================
# Step 8: Verify Installation
# ============================================
echo ""
echo -e "${GREEN}[8/9] Verifying installation...${NC}"

# Test Python imports
python3 << 'EOF'
import sys

print("Testing Python packages...")

# Test NumPy
try:
    import numpy as np
    print(f"  âœ“ NumPy {np.__version__}")
except ImportError as e:
    print(f"  âœ— NumPy: {e}")
    sys.exit(1)

# Test TFLite
try:
    try:
        from ai_edge_litert.interpreter import Interpreter
        print(f"  âœ“ ai-edge-litert (Python 3.12+ compatible)")
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
            print(f"  âœ“ TFLite Runtime (optimized)")
        except ImportError:
            import tensorflow as tf
            print(f"  âœ“ TensorFlow {tf.__version__}")
except ImportError as e:
    print(f"  âœ— TFLite/TensorFlow: {e}")
    print(f"    For Python 3.12+: pip install ai-edge-litert")
    print(f"    For Python 3.9-3.11: pip install tflite-runtime")
    sys.exit(1)

# Test OpenCV
try:
    import cv2
    print(f"  âœ“ OpenCV {cv2.__version__}")
except ImportError as e:
    print(f"  âœ— OpenCV: {e}")
    sys.exit(1)

# Test Picamera2 (optional)
try:
    from picamera2 import Picamera2
    print(f"  âœ“ Picamera2 available")
except ImportError:
    print(f"  âš  Picamera2 not available (will use fallback)")

# Test legacy picamera (optional)
try:
    import picamera
    print(f"  âœ“ Legacy Picamera available")
except ImportError:
    print(f"  âš  Legacy Picamera not available")

print("\nâœ“ All required packages installed successfully!")
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}âœ— Package verification failed${NC}"
    exit 1
fi

# ============================================
# Step 9: Verify Model and Test
# ============================================
echo ""
echo -e "${GREEN}[9/9] Checking model and running test...${NC}"

# Check for TFLite model
if [ -f "$PROJECT_ROOT/model/yolov8n-fp16.tflite" ]; then
    echo -e "${GREEN}âœ“ TFLite model found: model/yolov8n-fp16.tflite${NC}"
    
    # Get model size
    MODEL_SIZE=$(du -h "$PROJECT_ROOT/model/yolov8n-fp16.tflite" | cut -f1)
    echo -e "${BLUE}  Model size: $MODEL_SIZE${NC}"
    
    # Test model loading
    echo -e "${BLUE}Testing model loading...${NC}"
    python3 << 'EOF'
try:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter
    
    interpreter = Interpreter(model_path="model/yolov8n-fp16.tflite", num_threads=4)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    print(f"  âœ“ Model loaded successfully")
    print(f"    Input shape: {input_details['shape']}")
    print(f"    Output shape: {output_details['shape']}")
except Exception as e:
    print(f"  âœ— Model loading failed: {e}")
EOF
else
    echo -e "${RED}âœ— TFLite model not found!${NC}"
    echo -e "${YELLOW}Expected: $PROJECT_ROOT/model/yolov8n-fp16.tflite${NC}"
fi

# Check for PyTorch model (optional backup)
if [ -f "$PROJECT_ROOT/model/best.pt" ]; then
    echo -e "${BLUE}â„¹ PyTorch model also present: model/best.pt (not used by TFLite pipeline)${NC}"
fi

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

# Test camera (optional)
echo ""
echo -e "${BLUE}Testing camera (optional)...${NC}"
if command -v rpicam-still &> /dev/null; then
    if rpicam-still --timeout 100 -o "$PROJECT_ROOT/test_camera.jpg" 2>/dev/null; then
        echo -e "${GREEN}âœ“ Camera detected and working!${NC}"
        rm -f "$PROJECT_ROOT/test_camera.jpg"
    else
        echo -e "${YELLOW}âš  Camera not detected (may need to enable in raspi-config)${NC}"
    fi
else
    echo -e "${YELLOW}âš  rpicam-still not found (USB camera may still work)${NC}"
fi

# ============================================
# Complete!
# ============================================
echo ""
echo "=============================================="
echo -e "${GREEN}âœ“ Installation completed successfully!${NC}"
echo "=============================================="
echo ""
echo -e "${BLUE}Virtual Environment:${NC}"
echo "  Location: $PROJECT_ROOT/env"
echo "  Activate: source $PROJECT_ROOT/env/bin/activate"
echo ""
echo -e "${BLUE}Model:${NC}"
echo "  TFLite: model/yolov8n-fp16.tflite (float16, human detection)"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. Activate environment:"
echo "     source env/bin/activate"
echo ""
echo "  2. Run detection:"
echo "     ./scripts/run_detection.sh"
echo ""
echo "  3. Or run directly:"
echo "     python3 src/main.py"
echo ""
echo "  4. Headless mode (SSH):"
echo "     python3 src/main.py --headless"
echo ""
echo -e "${BLUE}Troubleshooting:${NC}"
echo "  - Enable camera: sudo raspi-config â†’ Interface Options â†’ Camera"
echo "  - Check logs: tail -f logs/detections.log"
echo ""
