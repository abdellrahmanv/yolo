"""
YOLOv5 TFLite Detection Pipeline - Main Runtime Controller
Raspberry Pi Camera + TFLite INT8 Real-time Glasses Detection
Optimized for low-power edge deployment
"""

import os
import cv2
import numpy as np
import logging
import time
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from capture import CameraCapture
from detector import TFLiteDetector

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'detections.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================

# Model settings
MODEL_PATH = PROJECT_ROOT / "model" / "best-int8.tflite"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Camera settings (matches model input for optimal performance)
CAMERA_RESOLUTION = (320, 320)
CAMERA_FRAMERATE = 30

# Display settings
DISPLAY_OUTPUT = True  # Set to False for headless mode
WINDOW_NAME = "Glasses Detection (TFLite)"

# Performance settings
SHOW_FPS = True
FPS_UPDATE_INTERVAL = 0.5  # Update FPS every 0.5 seconds


class DetectionPipeline:
    """
    Main detection pipeline controller
    Integrates camera capture and TFLite detection
    """

    def __init__(self, headless=False):
        """
        Initialize detection pipeline
        
        Args:
            headless: Run without display output
        """
        self.camera = None
        self.detector = None
        self.is_running = False
        self.headless = headless

        # Performance tracking
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_update = time.time()
        self.total_detections = 0

        logger.info("Detection pipeline initialized")

    def setup(self):
        """
        Setup camera and detector

        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Setting up detection pipeline...")

        # Initialize camera
        logger.info("Initializing camera...")
        self.camera = CameraCapture(
            resolution=CAMERA_RESOLUTION,
            framerate=CAMERA_FRAMERATE
        )

        if not self.camera.initialize():
            logger.error("Failed to initialize camera")
            return False

        camera_info = self.camera.get_camera_info()
        logger.info(f"Camera initialized: {camera_info}")

        # Initialize detector
        logger.info("Initializing TFLite detector...")
        self.detector = TFLiteDetector(
            model_path=MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD
        )

        if not self.detector.load_model():
            logger.error("Failed to load detection model")
            self.camera.release()
            return False

        logger.info("Detector initialized successfully")

        return True

    def run(self):
        """
        Main detection loop
        Captures frames, runs detection, and displays results
        """
        logger.info("=" * 50)
        logger.info("Starting Glasses Detection Pipeline")
        logger.info("=" * 50)
        logger.info(f"Model: {MODEL_PATH.name}")
        logger.info(f"Resolution: {CAMERA_RESOLUTION}")
        logger.info(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
        logger.info(f"Display: {'Enabled' if not self.headless else 'Disabled (headless)'}")
        logger.info("Press 'q' or Ctrl+C to stop")
        logger.info("=" * 50)

        self.is_running = True
        display_enabled = DISPLAY_OUTPUT and not self.headless

        # Setup display window if needed
        if display_enabled:
            try:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, 640, 480)
            except Exception as e:
                logger.warning(f"Could not create display window: {e}")
                display_enabled = False

        try:
            # Main loop
            for frame in self.camera.capture_continuous():
                if not self.is_running:
                    break

                # Run detection
                detections = self.detector.detect(frame)
                self.total_detections += len(detections)

                # Draw detections
                annotated_frame = self.detector.draw_detections(frame, detections)

                # Update FPS
                self.frame_count += 1
                current_time = time.time()
                elapsed = current_time - self.last_fps_update
                
                if elapsed >= FPS_UPDATE_INTERVAL:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.last_fps_update = current_time

                # Draw stats on frame
                if SHOW_FPS:
                    self._draw_stats(annotated_frame, len(detections))

                # Log detections
                if detections:
                    det_info = [f"{d['class_name']}({d['confidence']:.2f})" for d in detections]
                    logger.info(f"Detected: {', '.join(det_info)}")

                # Display frame
                if display_enabled:
                    try:
                        # Convert RGB to BGR for OpenCV display
                        display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                        cv2.imshow(WINDOW_NAME, display_frame)

                        # Check for quit key
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            logger.info("Quit key pressed")
                            break
                    except Exception as e:
                        logger.warning(f"Display error: {e}")
                        display_enabled = False

        except KeyboardInterrupt:
            logger.info("Detection interrupted by user (Ctrl+C)")

        except Exception as e:
            logger.error(f"Error during detection: {e}", exc_info=True)

        finally:
            self.cleanup()

    def _draw_stats(self, frame, detection_count):
        """Draw performance stats on frame"""
        # Background for text
        cv2.rectangle(frame, (5, 5), (200, 75), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (200, 75), (0, 255, 0), 2)

        # FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Inference time
        inf_time = self.detector.get_average_inference_time() * 1000
        time_text = f"Inference: {inf_time:.1f}ms"
        cv2.putText(frame, time_text, (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Detection count
        det_text = f"Detections: {detection_count}"
        cv2.putText(frame, det_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def cleanup(self):
        """Release resources"""
        logger.info("Cleaning up resources...")

        self.is_running = False

        if self.camera:
            self.camera.release()

        cv2.destroyAllWindows()

        # Log final statistics
        if self.detector:
            avg_inference_time = self.detector.get_average_inference_time()
            model_fps = self.detector.get_fps()

            logger.info("=" * 50)
            logger.info("Pipeline Statistics")
            logger.info("=" * 50)
            logger.info(f"Average inference time: {avg_inference_time*1000:.2f}ms")
            logger.info(f"Model FPS: {model_fps:.2f}")
            logger.info(f"Total detections: {self.total_detections}")
            logger.info("=" * 50)

        logger.info("Pipeline stopped successfully")


def verify_prerequisites():
    """
    Verify all prerequisites before starting

    Returns:
        bool: True if all checks pass
    """
    logger.info("Verifying prerequisites...")

    # Check model file
    if not MODEL_PATH.exists():
        logger.error(f"Model file not found: {MODEL_PATH}")
        logger.error("Expected: model/best-int8.tflite")
        return False
    logger.info(f"✓ Model file found: {MODEL_PATH.name}")

    # Check TFLite runtime
    try:
        try:
            import tflite_runtime
            logger.info("✓ TFLite Runtime installed")
        except ImportError:
            import tensorflow as tf
            logger.info(f"✓ TensorFlow {tf.__version__} (consider using tflite-runtime for Pi)")
    except ImportError:
        logger.error("✗ Neither tflite-runtime nor tensorflow found")
        logger.error("Install with: pip install tflite-runtime")
        return False

    # Check OpenCV
    try:
        import cv2
        logger.info(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        logger.error("✗ OpenCV not installed")
        return False

    # Check NumPy
    try:
        import numpy as np
        logger.info(f"✓ NumPy {np.__version__}")
    except ImportError:
        logger.error("✗ NumPy not installed")
        return False

    logger.info("All prerequisites verified")
    return True


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='YOLOv5 TFLite Glasses Detection for Raspberry Pi'
    )
    parser.add_argument(
        '--headless', '-H',
        action='store_true',
        help='Run without display output (for SSH/headless operation)'
    )
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.5,
        help='Detection confidence threshold (default: 0.5)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Path to TFLite model (default: model/best-int8.tflite)'
    )
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Update globals from args
    global MODEL_PATH, CONFIDENCE_THRESHOLD
    if args.model:
        MODEL_PATH = Path(args.model)
    CONFIDENCE_THRESHOLD = args.confidence

    logger.info("=" * 50)
    logger.info("YOLOv5 TFLite Glasses Detection")
    logger.info("Optimized for Raspberry Pi")
    logger.info("=" * 50)

    # Verify prerequisites
    if not verify_prerequisites():
        logger.error("Prerequisites check failed")
        sys.exit(1)

    # Create and setup pipeline
    pipeline = DetectionPipeline(headless=args.headless)

    if not pipeline.setup():
        logger.error("Pipeline setup failed")
        sys.exit(1)

    # Run detection
    try:
        pipeline.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Exiting...")
    sys.exit(0)


if __name__ == "__main__":
    main()
