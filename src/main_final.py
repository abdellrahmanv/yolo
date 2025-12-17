#!/usr/bin/env python3
"""
YOLOv5 Final AI Pipeline - With LCD and Buzzer
Glasses detection with hardware feedback:
- LCD: Shows glasses count
- Buzzer: Radar-style beeping (closer = faster)
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Set display before cv2
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':0'

print("Starting YOLOv5 Final AI...")
_start_time = time.time()

import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent))

from capture_threaded import ThreadedCamera as CameraCapture
from detector_pt import PyTorchDetector
from hardware import HardwareManager

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Config
MODEL_PATH = PROJECT_ROOT / "model" / "best.pt"
CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.45

# Camera settings
CAMERA_RESOLUTION = (320, 320)
CAMERA_FRAMERATE = 30
CAMERA_RESET_INTERVAL = 0  # Disable reset for stability

# Performance
SKIP_FRAMES = 2
INPUT_SIZE = 320

# Hardware pins
BUZZER_PIN = 18  # GPIO 18
LCD_ADDRESS = 0x27  # I2C address

WINDOW_NAME = "Glasses Detection - Final AI"


class FinalAIPipeline:
    """
    Complete detection pipeline with hardware feedback
    """
    
    def __init__(self, headless=False, enable_hardware=True):
        self.camera = None
        self.detector = None
        self.hardware = None
        self.is_running = False
        self.headless = headless
        self.enable_hardware = enable_hardware
        
        # Stats
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_update = time.time()
        self.total_detections = 0
    
    def setup(self):
        """Initialize all components"""
        # Camera
        print("Initializing camera...")
        self.camera = CameraCapture(
            resolution=CAMERA_RESOLUTION,
            framerate=CAMERA_FRAMERATE
        )
        if not self.camera.initialize():
            print("ERROR: Camera init failed")
            return False
        
        # Detector
        print("Loading model...")
        self.detector = PyTorchDetector(
            model_path=MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD
        )
        if not self.detector.load_model():
            print("ERROR: Model load failed")
            self.camera.release()
            return False
        
        # Hardware (LCD + Buzzer)
        if self.enable_hardware:
            print("Initializing hardware...")
            self.hardware = HardwareManager(
                buzzer_pin=BUZZER_PIN,
                lcd_address=LCD_ADDRESS
            )
            self.hardware.start()
            self.hardware.lcd.show_message("AI Ready!", f"Model: {MODEL_PATH.name[:10]}")
            time.sleep(1)
        
        return True
    
    def run(self):
        """Main detection loop"""
        print(f"\nReady in {time.time() - _start_time:.2f}s")
        print("=" * 40)
        print("YOLOv5 Glasses Detection - Final AI")
        print("=" * 40)
        print(f"Confidence: {CONFIDENCE_THRESHOLD}")
        print(f"Hardware: {'Enabled' if self.enable_hardware else 'Disabled'}")
        print("Press 'q' to quit")
        print("=" * 40)
        
        self.is_running = True
        display_enabled = not self.headless
        
        if display_enabled:
            try:
                if sys.platform.startswith('linux'):
                    os.environ['QT_QPA_PLATFORM'] = 'xcb'
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            except Exception as e:
                print(f"Warning: No display: {e}")
                display_enabled = False
        
        frame_counter = 0
        last_detections = []
        
        try:
            while self.is_running:
                # Capture frame
                frame = self.camera.capture_frame()
                if frame is None:
                    continue
                
                frame_counter += 1
                
                # Detection (with frame skipping)
                if frame_counter % SKIP_FRAMES == 0:
                    # Resize if needed
                    if frame.shape[0] == INPUT_SIZE and frame.shape[1] == INPUT_SIZE:
                        frame_resized = frame
                    else:
                        frame_resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
                    
                    # Detect
                    detections = self.detector.detect(frame_resized)
                    last_detections = detections
                    self.total_detections += len(detections)
                    
                    # Update hardware
                    if self.hardware:
                        self.hardware.update_detection(detections, self.fps)
                else:
                    detections = last_detections
                    if frame.shape[0] == INPUT_SIZE and frame.shape[1] == INPUT_SIZE:
                        frame_resized = frame
                    else:
                        frame_resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
                
                # Draw detections
                annotated = self.detector.draw_detections(frame_resized, detections)
                
                # FPS calculation
                self.frame_count += 1
                now = time.time()
                elapsed = now - self.last_fps_update
                if elapsed >= 0.5:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.last_fps_update = now
                
                # Draw stats overlay
                self._draw_stats(annotated, len(detections))
                
                # Display
                if display_enabled:
                    try:
                        disp = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                        cv2.imshow(WINDOW_NAME, disp)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:
                            break
                    except Exception as e:
                        print(f"Display error: {e}")
                        display_enabled = False
        
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            self.cleanup()
    
    def _draw_stats(self, frame, det_count):
        """Draw stats overlay on frame"""
        # Background
        cv2.rectangle(frame, (5, 5), (180, 85), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (180, 85), (0, 255, 0), 1)
        
        # Text
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        inf_ms = self.detector.get_average_inference_time() * 1000
        cv2.putText(frame, f"Inference: {inf_ms:.0f}ms", (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.putText(frame, f"Glasses: {det_count}", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        hw_status = "HW: ON" if self.hardware else "HW: OFF"
        cv2.putText(frame, hw_status, (10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def cleanup(self):
        """Cleanup all resources"""
        print("\nCleaning up...")
        self.is_running = False
        
        if self.hardware:
            self.hardware.lcd.show_message("Shutting down...")
            self.hardware.cleanup()
        
        if self.camera:
            self.camera.release()
        
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
        # Print final stats
        if self.detector and self.detector.inference_times:
            avg_ms = self.detector.get_average_inference_time() * 1000
            print(f"\n--- Final Stats ---")
            print(f"Avg inference: {avg_ms:.1f}ms")
            print(f"Model FPS: {self.detector.get_fps():.1f}")
            print(f"Total detections: {self.total_detections}")


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv5 Final AI - Glasses Detection with Hardware')
    parser.add_argument('--headless', '-H', action='store_true', help='No display output')
    parser.add_argument('--no-hardware', '-N', action='store_true', help='Disable LCD and buzzer')
    parser.add_argument('--confidence', '-c', type=float, default=0.50, help='Confidence threshold')
    parser.add_argument('--model', '-m', type=str, default=None, help='Model path')
    return parser.parse_args()


def main():
    args = parse_args()
    
    global MODEL_PATH, CONFIDENCE_THRESHOLD
    if args.model:
        MODEL_PATH = Path(args.model)
    CONFIDENCE_THRESHOLD = args.confidence
    
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        sys.exit(1)
    
    print(f"Model: {MODEL_PATH.name}")
    print(f"Mode: {'headless' if args.headless else 'display'}")
    print(f"Hardware: {'disabled' if args.no_hardware else 'enabled'}")
    
    pipeline = FinalAIPipeline(
        headless=args.headless,
        enable_hardware=not args.no_hardware
    )
    
    if not pipeline.setup():
        sys.exit(1)
    
    pipeline.run()


if __name__ == "__main__":
    main()
