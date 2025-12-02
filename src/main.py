"""
YOLOv5 Detection Pipeline - Main Runtime Controller
Raspberry Pi Camera 1.3 + YOLOv5 Real-time Object Detection
"""

import cv2
import numpy as np
import logging
import time
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from capture import CameraCapture
from detector import YOLOv5Detector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/detections.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================

# Model settings
MODEL_PATH = "../model/best.pt"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Camera settings
CAMERA_RESOLUTION = (640, 480)
CAMERA_FRAMERATE = 30

# Display settings
DISPLAY_OUTPUT = True
WINDOW_NAME = "YOLOv5 Detection"

# Performance settings
SHOW_FPS = True
FPS_UPDATE_INTERVAL = 1.0  # Update FPS every N seconds


class DetectionPipeline:
    """
    Main detection pipeline controller
    Integrates camera capture and YOLO detection
    """
    
    def __init__(self):
        """Initialize detection pipeline"""
        self.camera = None
        self.detector = None
        self.is_running = False
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_update = time.time()
        
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
        
        logger.info("Camera initialized successfully")
        
        # Initialize detector
        logger.info("Initializing detector...")
        self.detector = YOLOv5Detector(
            model_path=MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            device='cpu'
        )
        
        if not self.detector.load_model():
            logger.error("Failed to load detection model")
            self.camera.release()
            return False
        
        logger.info("Detector initialized successfully")
        logger.info(f"Detected classes: {self.detector.class_names}")
        
        return True
    
    def run(self):
        """
        Main detection loop
        Captures frames, runs detection, and displays results
        """
        logger.info("Starting detection pipeline...")
        logger.info(f"Model: {MODEL_PATH}")
        logger.info(f"Resolution: {CAMERA_RESOLUTION}")
        logger.info(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
        logger.info("Press 'q' or Ctrl+C to stop")
        
        self.is_running = True
        
        try:
            # Main loop
            for frame in self.camera.capture_continuous():
                if not self.is_running:
                    break
                
                # Run detection
                detections = self.detector.detect(frame)
                
                # Draw detections
                annotated_frame = self.detector.draw_detections(frame, detections)
                
                # Update FPS
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_update >= FPS_UPDATE_INTERVAL:
                    self.fps = self.frame_count / (current_time - self.last_fps_update)
                    self.frame_count = 0
                    self.last_fps_update = current_time
                
                # Draw FPS on frame
                if SHOW_FPS:
                    fps_text = f"FPS: {self.fps:.1f}"
                    cv2.putText(annotated_frame, fps_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw detection count
                det_count_text = f"Detections: {len(detections)}"
                cv2.putText(annotated_frame, det_count_text, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Log detections
                if detections:
                    det_info = [f"{d['class_name']}({d['confidence']:.2f})" for d in detections]
                    logger.info(f"Frame detections: {', '.join(det_info)}")
                
                # Display frame
                if DISPLAY_OUTPUT:
                    # Convert RGB to BGR for OpenCV display
                    display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow(WINDOW_NAME, display_frame)
                    
                    # Check for quit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Quit key pressed")
                        break
        
        except KeyboardInterrupt:
            logger.info("Detection interrupted by user")
        
        except Exception as e:
            logger.error(f"Error during detection: {e}", exc_info=True)
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        logger.info("Cleaning up resources...")
        
        self.is_running = False
        
        if self.camera:
            self.camera.release()
        
        if DISPLAY_OUTPUT:
            cv2.destroyAllWindows()
        
        # Log final statistics
        avg_inference_time = self.detector.get_average_inference_time()
        model_fps = self.detector.get_fps()
        
        logger.info("=" * 50)
        logger.info("Detection Pipeline Statistics")
        logger.info("=" * 50)
        logger.info(f"Average inference time: {avg_inference_time*1000:.2f}ms")
        logger.info(f"Model FPS: {model_fps:.2f}")
        logger.info(f"Overall FPS: {self.fps:.2f}")
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
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path.absolute()}")
        logger.error("Please copy your trained best.pt file to the model directory")
        return False
    
    logger.info(f"✓ Model file found: {model_path.absolute()}")
    
    # Check dependencies
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        logger.error("✗ PyTorch not installed")
        return False
    
    try:
        import cv2
        logger.info(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        logger.error("✗ OpenCV not installed")
        return False
    
    # Camera check moved to runtime initialization
    logger.info("✓ Camera will be initialized at runtime")
    
    logger.info("All prerequisites verified")
    return True


def main():
    """Main entry point"""
    logger.info("=" * 50)
    logger.info("YOLOv5 Detection Pipeline for Raspberry Pi")
    logger.info("=" * 50)
    
    # Verify prerequisites
    if not verify_prerequisites():
        logger.error("Prerequisites check failed. Please run setup script.")
        sys.exit(1)
    
    # Create and setup pipeline
    pipeline = DetectionPipeline()
    
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
