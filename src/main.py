#!/usr/bin/env python3
"""
YOLOv5 TFLite Detection Pipeline - FAST STARTUP
Raspberry Pi Camera + TFLite INT8 Real-time Human Detection
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Set display BEFORE importing cv2 (critical for Raspberry Pi)
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':0'

print("Starting YOLOv5 TFLite human detector...")
_start_time = time.time()

import cv2
import numpy as np

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from capture_threaded import ThreadedCamera as CameraCapture
from detector import TFLiteDetector

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ============================================
# CONFIGURATION
# ============================================

# Model settings
MODEL_PATH = PROJECT_ROOT / "model" / "yolov5n-int8.tflite"
CONFIDENCE_THRESHOLD = 0.35  # Lowered for INT8 quantized models
IOU_THRESHOLD = 0.45

# Camera settings (matches model input for optimal performance)
CAMERA_RESOLUTION = (320, 320)
CAMERA_FRAMERATE = 30
CAMERA_RESET_INTERVAL = 6  # Reset camera every N seconds (0 to disable)

# FPS Optimizations
SKIP_FRAMES = 2  # Process every Nth frame (1=no skip, 2=skip every other, 3=process 1/3)
HEADLESS_BOOST = False  # True = no display output, maximum FPS
INPUT_SIZE = 320  # Model input size (smaller = faster: 160, 224, 320)
USE_MJPG = True  # Use MJPG camera format (faster on Pi)

# Display settings
DISPLAY_OUTPUT = True
WINDOW_NAME = "Human Detection"

# Performance settings
SHOW_FPS = True
FPS_UPDATE_INTERVAL = 0.5


def setup_logging(verbose=False):
    """Setup logging with configurable verbosity"""
    import logging
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_DIR / 'detections.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class DetectionPipeline:
    """
    Main detection pipeline controller
    Integrates camera capture and TFLite detection
    """

    def __init__(self, headless=False, logger=None):
        import logging
        self.camera = None
        self.detector = None
        self.is_running = False
        self.headless = headless
        self.logger = logger or logging.getLogger(__name__)

        # Performance tracking
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_update = time.time()
        self.total_detections = 0

    def setup(self):
        """Setup camera and detector"""
        # Initialize camera
        self.camera = CameraCapture(
            resolution=CAMERA_RESOLUTION,
            framerate=CAMERA_FRAMERATE
        )

        if not self.camera.initialize():
            print("ERROR: Failed to initialize camera")
            return False

        # Initialize detector
        self.detector = TFLiteDetector(
            model_path=MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD
        )

        if not self.detector.load_model():
            print("ERROR: Failed to load model")
            self.camera.release()
            return False

        return True

    def run(self):
        """Main detection loop"""
        print(f"Ready in {time.time() - _start_time:.2f}s")
        print(f"Starting detection... (Press 'q' to quit)")
        if CAMERA_RESET_INTERVAL > 0:
            print(f"Camera will reset every {CAMERA_RESET_INTERVAL}s")
        
        self.is_running = True
        display_enabled = DISPLAY_OUTPUT and not self.headless

        # Setup display window
        if display_enabled:
            try:
                # Force X11 backend on Linux
                if sys.platform.startswith('linux'):
                    os.environ['QT_QPA_PLATFORM'] = 'xcb'
                
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            except Exception as e:
                print(f"Warning: Could not create window: {e}")
                display_enabled = False

        last_camera_reset = time.time()
        frame_counter = 0
        last_detections = []

        try:
            while self.is_running:
                # Check if camera needs reset
                if CAMERA_RESET_INTERVAL > 0 and time.time() - last_camera_reset >= CAMERA_RESET_INTERVAL:
                    print("Resetting camera...")
                    if not self.camera.reset():
                        print("ERROR: Camera reset failed")
                        break
                    last_camera_reset = time.time()
                    print("Camera reset complete")

                # Capture frame
                frame = self.camera.capture_frame()
                if frame is None:
                    continue

                frame_counter += 1
                
                # Frame skipping - only run detection every Nth frame
                if frame_counter % SKIP_FRAMES == 0:
                    # Run detection
                    detections = self.detector.detect(frame)
                    last_detections = detections
                    self.total_detections += len(detections)
                else:
                    # Use cached detections on skipped frames
                    detections = last_detections

                # Draw detections on frame
                annotated_frame = self.detector.draw_detections(frame, detections)

                # Update FPS counter
                self.frame_count += 1
                current_time = time.time()
                elapsed = current_time - self.last_fps_update

                if elapsed >= FPS_UPDATE_INTERVAL:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.last_fps_update = current_time

                # Draw stats
                if SHOW_FPS:
                    self._draw_stats(annotated_frame, len(detections))

                # Display frame
                if display_enabled:
                    try:
                        # Convert RGB to BGR for OpenCV display
                        display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                        cv2.imshow(WINDOW_NAME, display_frame)

                        # Check for quit key (waitKey is required for window update)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:  # q or ESC
                            break
                    except Exception as e:
                        print(f"Display error: {e}")
                        display_enabled = False

        except KeyboardInterrupt:
            print("\nStopped by user")

        finally:
            self.cleanup()

    def _draw_stats(self, frame, detection_count):
        """Draw performance stats on frame"""
        h, w = frame.shape[:2]
        
        # Background box
        cv2.rectangle(frame, (5, 5), (180, 70), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (180, 70), (0, 255, 0), 1)

        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Inference time
        inf_ms = self.detector.get_average_inference_time() * 1000
        cv2.putText(frame, f"Inf: {inf_ms:.0f}ms", (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Detection count
        cv2.putText(frame, f"Det: {detection_count}", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def cleanup(self):
        """Release resources"""
        self.is_running = False

        if self.camera:
            self.camera.release()

        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Ensure windows close

        # Print final stats
        if self.detector and self.detector.inference_times:
            avg_ms = self.detector.get_average_inference_time() * 1000
            fps = self.detector.get_fps()
            print(f"\n--- Stats ---")
            print(f"Avg inference: {avg_ms:.1f}ms")
            print(f"Model FPS: {fps:.1f}")
            print(f"Total detections: {self.total_detections}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='YOLOv5 TFLite Human Detection'
    )
    parser.add_argument(
        '--headless', '-H',
        action='store_true',
        help='Run without display (SSH mode)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose logging'
    )
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.35,
        help='Confidence threshold (default: 0.35)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Path to TFLite model'
    )
    return parser.parse_args()


def main():
    """Main entry point - optimized for fast startup"""
    args = parse_args()

    # Setup logging (minimal by default for speed)
    logger = setup_logging(verbose=args.verbose)

    # Update config from args
    global MODEL_PATH, CONFIDENCE_THRESHOLD
    if args.model:
        MODEL_PATH = Path(args.model)
    CONFIDENCE_THRESHOLD = args.confidence

    # Quick model check
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        sys.exit(1)

    print(f"Model: {MODEL_PATH.name}")
    print(f"Mode: {'headless' if args.headless else 'display'}")

    # Create and run pipeline
    pipeline = DetectionPipeline(headless=args.headless, logger=logger)

    if not pipeline.setup():
        sys.exit(1)

    pipeline.run()


if __name__ == "__main__":
    main()
