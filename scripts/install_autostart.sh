#!/bin/bash

###############################################################################
# Install autostart service for YOLOv5 Glasses Detection
# Runs final_ai.sh automatically on boot
###############################################################################

set -e

echo "=============================================="
echo "  Installing Autostart Service"
echo "=============================================="
echo ""

# Copy service file
echo "Installing systemd service..."
sudo cp /home/pi/yolo/scripts/glasses-detector.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable glasses-detector.service

echo ""
echo "✓ Service installed!"
echo ""
echo "Commands:"
echo "  Start now:     sudo systemctl start glasses-detector"
echo "  Stop:          sudo systemctl stop glasses-detector"
echo "  Status:        sudo systemctl status glasses-detector"
echo "  View logs:     journalctl -u glasses-detector -f"
echo "  Disable:       sudo systemctl disable glasses-detector"
echo ""
echo "The detector will start automatically on next boot!"
echo ""

# Ask to start now
read -p "Start the detector now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo systemctl start glasses-detector
    echo "Started! Check status with: sudo systemctl status glasses-detector"
fi
