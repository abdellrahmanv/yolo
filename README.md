# Raspberry Pi YOLOv5 Detection Pipeline

Complete real-time object detection system for Raspberry Pi using YOLOv5 and Picamera2.

## System Requirements
- Raspberry Pi (3/4/5 recommended)
- Raspberry Pi Camera Module 1.3 or compatible
- Raspberry Pi OS (Bullseye or Bookworm)
- 4GB+ RAM recommended
- Python 3.9+

## Architecture Overview

```
Camera → Legacy Camera Stack → Picamera → YOLOv5 → Detection Output
```

## Quick Start

### 1. Transfer Project to Raspberry Pi
```bash
# Copy entire folder to Raspberry Pi
scp -r yolo_pi_detection/ pi@raspberrypi.local:~/
```

### 2. Copy Your Model
```bash
# Place your trained best.pt file in the model directory
scp best.pt pi@raspberrypi.local:~/yolo_pi_detection/model/
```

### 3. Run Installation
```bash
cd ~/yolo_pi_detection
chmod +x scripts/setup_pi.sh
./scripts/setup_pi.sh
```

### 4. Start Detection
```bash
chmod +x scripts/run_detection.sh
./scripts/run_detection.sh
```

## Project Structure
```
yolo_pi_detection/
├── env/                    # Virtual environment (created during setup)
├── scripts/
│   ├── setup_pi.sh        # Automated installation
│   └── run_detection.sh   # Launch detection pipeline
├── model/
│   └── best.pt            # Your trained YOLOv5 model
├── src/
│   ├── capture.py         # Camera interface module
│   ├── detector.py        # YOLOv5 inference engine
│   └── main.py            # Main runtime controller
├── logs/
│   └── detections.log     # Detection logs
└── README.md
```

## Pipeline Flow

1. **Camera Initialization** - Legacy Picamera initializes camera hardware
2. **Frame Capture** - Continuous frame grabbing from camera
3. **Preprocessing** - Resize and normalize for YOLOv5
4. **Inference** - Model prediction using best.pt
5. **Postprocessing** - Draw bounding boxes and labels
6. **Output** - Display results or log detections

## Manual Installation Steps

If automated script fails, follow these steps:

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install system dependencies
sudo apt install -y python3-dev python3-venv python3-pip \
    libatlas-base-dev libopenblas-dev liblapack-dev \
    libjpeg-dev libcamera-dev libcamera-apps \
    libopencv-dev

# 3. Create virtual environment
cd ~/yolo_pi_detection
python3 -m venv env

# 4. Activate environment
source env/bin/activate

# 5. Install Python packages
pip install --upgrade pip
pip install numpy opencv-python pillow
pip install picamera2

# 6. Install PyTorch (Raspberry Pi compatible)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 7. Install YOLOv5
pip install ultralytics

# 8. Test camera
rpicam-still -o test.jpg
```

## Configuration

Edit `src/main.py` to adjust:
- `CONFIDENCE_THRESHOLD` - Detection confidence (default: 0.5)
- `CAMERA_RESOLUTION` - Frame size (default: 640x480)
- `DISPLAY_OUTPUT` - Show preview window (default: True)

## Troubleshooting

### Camera not detected
```bash
# Test camera
rpicam-still -o test.jpg
# Enable camera interface
sudo raspi-config
# Navigate to Interface Options → Camera → Enable
```

### Out of memory errors
- Reduce camera resolution in `src/main.py`
- Use YOLOv5s or YOLOv5n (smaller models)
- Increase swap space

### Slow inference
- Use YOLOv5n (nano) model
- Reduce input resolution
- Enable hardware acceleration if available

## Performance Tips

1. Use YOLOv5n or YOLOv5s for better FPS
2. Lower camera resolution (320x240 or 416x416)
3. Use Raspberry Pi 4/5 with 4GB+ RAM
4. Overclock CPU (with proper cooling)
5. Reduce confidence threshold only if needed

## Advanced Usage

### Headless Mode (No Display)
```python
# In src/main.py, set:
DISPLAY_OUTPUT = False
```

### MQTT Integration
Add MQTT publishing in `src/main.py` to send detections to external systems.

### Save Detections
Uncomment logging sections in `src/detector.py` to save annotated frames.

## License
MIT License - Use freely for personal and commercial projects.

## Support
For issues with:
- YOLOv5 model: https://github.com/ultralytics/yolov5
- Picamera2: https://github.com/raspberrypi/picamera2
- Raspberry Pi Camera: https://www.raspberrypi.com/documentation/
