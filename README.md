# Raspberry Pi YOLOv5 TFLite Glasses Detection

Real-time glasses detection on Raspberry Pi using YOLOv5 with TensorFlow Lite INT8 quantization for optimal performance.

## Features

- **TFLite INT8 Quantized Model** - 3-5x faster inference than PyTorch
- **Low Memory Footprint** - ~200MB RAM vs ~800MB for PyTorch
- **Multiple Camera Support** - Picamera2, legacy Picamera, USB cameras
- **Headless Mode** - Run via SSH without display
- **Auto Camera Detection** - Automatically selects best available camera backend

## System Requirements

| Component | Requirement |
|-----------|-------------|
| Hardware | Raspberry Pi 3B/4/5 (4GB+ RAM recommended) |
| Camera | Pi Camera Module 1.3/2/3 or USB camera |
| OS | Raspberry Pi OS Bullseye/Bookworm |
| Python | 3.9 - 3.11 (TFLite compatibility) |

## Model Specifications

| Property | Value |
|----------|-------|
| Model | `best-int8.tflite` |
| Input | 320×320×3 RGB (uint8) |
| Output | 6300 predictions × 6 values |
| Size | ~1.9 MB |
| Classes | glasses |

## Expected Performance

| Raspberry Pi | Resolution | Expected FPS |
|--------------|------------|--------------|
| Pi 3B        | 320×320    | 3-5 FPS      |
| Pi 4 (4GB)   | 320×320    | 8-12 FPS     |
| Pi 5 (8GB)   | 320×320    | 15-20 FPS    |

## Quick Start

### 1. Transfer to Raspberry Pi

```bash
# From your PC
scp -r yolo pi@raspberrypi.local:~/
```

### 2. Run Installation

```bash
cd ~/yolo
chmod +x scripts/setup_pi.sh
./scripts/setup_pi.sh
```

### 3. Start Detection

```bash
./scripts/run_detection.sh
```

**Press 'q' or Ctrl+C to stop**

## Project Structure

```
yolo/
├── model/
│   ├── best-int8.tflite    # TFLite INT8 quantized model (used)
│   └── best.pt              # PyTorch model (backup)
├── src/
│   ├── capture.py           # Camera capture (multi-backend)
│   ├── detector.py          # TFLite inference engine
│   └── main.py              # Main pipeline controller
├── scripts/
│   ├── setup_pi.sh          # Installation script
│   └── run_detection.sh     # Runtime launcher
├── logs/
│   └── detections.log       # Detection logs
├── requirements.txt         # Python dependencies
└── README.md
```

## Usage Options

### With Display
```bash
./scripts/run_detection.sh
```

### Headless Mode (SSH)
```bash
./scripts/run_detection.sh --headless
# or
python3 src/main.py --headless
```

### Custom Confidence Threshold
```bash
python3 src/main.py --confidence 0.6
```

### Custom Model Path
```bash
python3 src/main.py --model /path/to/model.tflite
```

## Manual Installation

If the setup script fails:

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install dependencies
sudo apt install -y python3-dev python3-venv python3-pip \
    libopenblas-dev libatlas-base-dev python3-picamera2

# 3. Create virtual environment
python3 -m venv env --system-site-packages
source env/bin/activate

# 4. Install Python packages
pip install --upgrade pip
pip install tflite-runtime numpy opencv-python-headless Pillow

# 5. Run detection
python3 src/main.py
```

## Camera Setup

### Enable Camera
```bash
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable
sudo reboot
```

### Test Camera
```bash
rpicam-still -o test.jpg
```

## Troubleshooting

### Camera Not Detected
1. Check physical connection
2. Enable camera in `raspi-config`
3. Reboot the Pi

### TFLite Import Error
```bash
# Install tflite-runtime
pip install tflite-runtime

# Or fallback to full TensorFlow
pip install tensorflow
```

### Low FPS
- Ensure using TFLite model (not PyTorch)
- Close other applications
- Use headless mode if possible

### Out of Memory
```bash
# Increase swap
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set: CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

## Dependencies

| Package | Purpose |
|---------|---------|
| tflite-runtime | TFLite inference (~5MB) |
| numpy | Array operations |
| opencv-python-headless | Image processing |
| picamera2 | Camera capture (optional) |

## License

MIT License

## Resources

- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [Picamera2 Documentation](https://github.com/raspberrypi/picamera2)
