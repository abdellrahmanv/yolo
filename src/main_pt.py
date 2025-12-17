#!/usr/bin/env python3
"""
YOLOv5 PyTorch Detection Pipeline
Uses best.pt model for glasses detection
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Set display before cv2
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':0'

import cv2
import numpy as np
import logging

sys.path.append(str(Path(__file__).parent))

from capture import CameraCapture
from detector_pt import PyTorchDetector

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Config
MODEL_PATH = PROJECT_ROOT / "model" / "best.pt"
CONFIDENCE_THRESHOLD = 0.25  # Same as TFLite pipeline
IOU_THRESHOLD = 0.45
CAMERA_RESOLUTION = (320, 320)
CAMERA_FRAMERATE = 30
WINDOW_NAME = "Glasses Detection (PyTorch)"


def setup_logging(verbose=False):
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_DIR / 'detections_pt.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class DetectionPipeline:
    def __init__(self, headless=False):
        self.camera = None
        self.detector = None
        self.is_running = False
        self.headless = headless
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_update = time.time()
        self.total_detections = 0

    def setup(self):
        # Camera
        self.camera = CameraCapture(
            resolution=CAMERA_RESOLUTION,
            framerate=CAMERA_FRAMERATE
        )
        if not self.camera.initialize():
            print("ERROR: Camera init failed")
            return False

        # Detector
        self.detector = PyTorchDetector(
            model_path=MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD
        )
        if not self.detector.load_model():
            print("ERROR: Model load failed")
            self.camera.release()
            return False

        return True

    def run(self):
        print(f"Starting PyTorch detection... (Press 'q' to quit)")
        
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

        try:
            for frame in self.camera.capture_continuous():
                if not self.is_running:
                    break

                # Detect
                detections = self.detector.detect(frame)
                self.total_detections += len(detections)

                # Draw
                annotated = self.detector.draw_detections(frame, detections)

                # FPS
                self.frame_count += 1
                now = time.time()
                elapsed = now - self.last_fps_update
                if elapsed >= 0.5:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.last_fps_update = now

                # Stats overlay
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
            print("\nStopped")
        finally:
            self.cleanup()

    def _draw_stats(self, frame, det_count):
        cv2.rectangle(frame, (5, 5), (180, 70), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (180, 70), (0, 255, 0), 1)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        inf_ms = self.detector.get_average_inference_time() * 1000
        cv2.putText(frame, f"Inf: {inf_ms:.0f}ms", (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Det: {det_count}", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def cleanup(self):
        self.is_running = False
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        if self.detector and self.detector.inference_times:
            avg_ms = self.detector.get_average_inference_time() * 1000
            print(f"\n--- Stats ---")
            print(f"Avg inference: {avg_ms:.1f}ms")
            print(f"Model FPS: {self.detector.get_fps():.1f}")
            print(f"Total detections: {self.total_detections}")


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv5 PyTorch Glasses Detection')
    parser.add_argument('--headless', '-H', action='store_true', help='No display')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose')
    parser.add_argument('--confidence', '-c', type=float, default=0.25, help='Threshold')
    parser.add_argument('--model', '-m', type=str, default=None, help='Model path')
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    global MODEL_PATH, CONFIDENCE_THRESHOLD
    if args.model:
        MODEL_PATH = Path(args.model)
    CONFIDENCE_THRESHOLD = args.confidence

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        sys.exit(1)

    print(f"Model: {MODEL_PATH.name}")
    print(f"Mode: {'headless' if args.headless else 'display'}")

    pipeline = DetectionPipeline(headless=args.headless)
    if not pipeline.setup():
        sys.exit(1)
    pipeline.run()


if __name__ == "__main__":
    main()
