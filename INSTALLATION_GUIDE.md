# Raspberry Pi YOLOv5 - Complete Installation Guide

## Prerequisites

### Hardware Requirements
- Raspberry Pi 3B/3B+/4/5 (4GB+ RAM recommended)
- Raspberry Pi Camera Module 1.3 or compatible
- MicroSD card (16GB+ recommended)
- Power supply (official recommended)
- Monitor + HDMI cable (for initial setup)
- Keyboard + mouse

### Software Requirements
- Raspberry Pi OS (Bullseye or Bookworm)
- Fresh OS installation recommended

---

## Step-by-Step Installation

### 1. Prepare Raspberry Pi

```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Enable camera interface
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable → Reboot
```

### 2. Connect Camera

1. Power off Raspberry Pi
2. Locate camera connector (between HDMI and USB ports)
3. Gently lift connector clip
4. Insert ribbon cable (blue side facing USB ports)
5. Push down connector clip
6. Power on Raspberry Pi

### 3. Verify Camera

```bash
# Test camera
rpicam-still -o test.jpg

# View image
display test.jpg

# If successful, you should see a captured image
```

### 4. Transfer Project to Raspberry Pi

#### Option A: Using SCP (from your Windows laptop)

```powershell
# Open PowerShell on your laptop
cd C:\Users\Asus\Desktop

# Copy entire project folder
scp -r yolo_pi_detection pi@raspberrypi.local:~/

# If hostname doesn't work, use IP address:
scp -r yolo_pi_detection pi@192.168.1.XXX:~/
```

#### Option B: Using USB Drive

1. Copy `yolo_pi_detection` folder to USB drive
2. Insert USB into Raspberry Pi
3. Copy from USB to home directory:
```bash
cp -r /media/pi/USB_DRIVE/yolo_pi_detection ~/
```

#### Option C: Using Git

```bash
# On Raspberry Pi
cd ~
git clone <your-repository-url>
```

### 5. Copy Your Trained Model

```bash
# From Windows laptop (PowerShell):
scp C:\path\to\your\best.pt pi@raspberrypi.local:~/yolo_pi_detection/model/

# Or copy manually via USB drive
```

### 6. Run Automated Installation

```bash
# On Raspberry Pi
cd ~/yolo_pi_detection

# Make scripts executable
chmod +x scripts/setup_pi.sh
chmod +x scripts/run_detection.sh

# Run installation (takes 15-30 minutes)
./scripts/setup_pi.sh
```

The installation script will:
- ✓ Update system packages
- ✓ Install system dependencies
- ✓ Create Python virtual environment
- ✓ Install PyTorch (CPU version)
- ✓ Install YOLOv5 and dependencies
- ✓ Install Picamera2
- ✓ Verify camera connection
- ✓ Test all imports

### 7. Start Detection Pipeline

```bash
# Launch detection
./scripts/run_detection.sh
```

You should see:
- Camera preview window
- Real-time detections with bounding boxes
- FPS counter
- Detection count

**Press 'q' or Ctrl+C to stop**

---

## Troubleshooting

### Camera Not Detected

```bash
# Check camera connection
vcgencmd get_camera

# Should output: supported=1 detected=1

# If not detected:
# 1. Check physical connection
# 2. Enable camera in raspi-config
# 3. Reboot
sudo reboot
```

### Out of Memory Errors

```bash
# Increase swap space
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Change: CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Slow Performance

**Optimize detection settings** in `src/main.py`:

```python
# Reduce resolution
CAMERA_RESOLUTION = (416, 416)  # Instead of (640, 480)

# Lower confidence threshold
CONFIDENCE_THRESHOLD = 0.6  # Higher = fewer detections
```

**Use smaller YOLOv5 model:**
- YOLOv5n (nano) - fastest
- YOLOv5s (small) - balanced
- YOLOv5m (medium) - more accurate, slower

### Import Errors

```bash
# Activate virtual environment
cd ~/yolo_pi_detection
source env/bin/activate

# Verify installations
python3 -c "import torch; print(torch.__version__)"
python3 -c "import cv2; print(cv2.__version__)"
python3 -c "from picamera2 import Picamera2"

# If any fail, reinstall
pip install torch torchvision opencv-python picamera2
```

### Display Issues (Headless Mode)

If running without monitor:

```python
# Edit src/main.py
DISPLAY_OUTPUT = False
```

Then access via SSH and check logs:
```bash
tail -f logs/detections.log
```

---

## Performance Optimization

### 1. Overclock Raspberry Pi (with cooling)

```bash
sudo nano /boot/config.txt

# Add:
over_voltage=6
arm_freq=2000
gpu_freq=600
```

### 2. Reduce Camera Resolution

Lower resolution = faster inference:
- 320x240 - very fast, low accuracy
- 416x416 - balanced
- 640x480 - default
- 640x640 - best accuracy, slowest

### 3. Use GPU Acceleration (Pi 4/5)

```bash
# Install TFLite or ONNX runtime for better performance
pip install onnxruntime
```

Convert model to ONNX format for faster inference.

---

## Running at Boot (Auto-Start)

### Create systemd service:

```bash
sudo nano /etc/systemd/system/yolo-detection.service
```

Add:
```ini
[Unit]
Description=YOLOv5 Detection Pipeline
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/yolo_pi_detection
ExecStart=/home/pi/yolo_pi_detection/scripts/run_detection.sh
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable yolo-detection
sudo systemctl start yolo-detection

# Check status
sudo systemctl status yolo-detection
```

---

## Testing Individual Components

### Test Camera Only

```bash
cd ~/yolo_pi_detection
source env/bin/activate
python3 src/capture.py
```

### Test Model Loading

```bash
cd ~/yolo_pi_detection
source env/bin/activate
python3 src/detector.py
```

### Full Pipeline Test

```bash
cd ~/yolo_pi_detection
source env/bin/activate
python3 src/main.py
```

---

## Expected Performance

| Raspberry Pi | Resolution | YOLOv5 Model | Expected FPS |
|--------------|------------|--------------|--------------|
| Pi 3B        | 416x416    | YOLOv5n      | 1-2 FPS      |
| Pi 4 (4GB)   | 416x416    | YOLOv5n      | 3-5 FPS      |
| Pi 4 (8GB)   | 640x480    | YOLOv5s      | 2-4 FPS      |
| Pi 5 (8GB)   | 640x640    | YOLOv5s      | 5-8 FPS      |

---

## Uninstallation

```bash
# Stop service (if enabled)
sudo systemctl stop yolo-detection
sudo systemctl disable yolo-detection
sudo rm /etc/systemd/system/yolo-detection.service

# Remove project
cd ~
rm -rf yolo_pi_detection

# Remove packages (optional)
sudo apt remove --purge python3-picamera2 libcamera-apps
sudo apt autoremove
```

---

## Support & Resources

- YOLOv5: https://github.com/ultralytics/yolov5
- Picamera2: https://github.com/raspberrypi/picamera2
- Raspberry Pi Camera: https://www.raspberrypi.com/documentation/accessories/camera.html

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Low FPS | Reduce resolution, use smaller model |
| High CPU usage | Normal, YOLOv5 is computationally intensive |
| Camera timeout | Check connection, increase timeout in code |
| Out of memory | Increase swap, close other programs |
| Model not loading | Verify model path, check PyTorch version |

---

**Installation complete! Your Raspberry Pi is now running real-time YOLO detection.**
