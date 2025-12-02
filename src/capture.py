"""
Camera Capture Module for Raspberry Pi
Handles frame acquisition from Raspberry Pi Camera using OpenCV
Compatible with both legacy and modern camera stacks
"""

import numpy as np
import cv2
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraCapture:
    """
    Camera capture interface using OpenCV
    Works with Raspberry Pi Camera via V4L2 driver
    """
    
    def __init__(self, resolution=(640, 480), framerate=30):
        """
        Initialize camera capture
        
        Args:
            resolution: Tuple (width, height) for camera resolution
            framerate: Target frames per second
        """
        self.resolution = resolution
        self.framerate = framerate
        self.camera = None
        self.is_initialized = False
        
        logger.info(f"Initializing camera with resolution {resolution} @ {framerate}fps")
        
    def initialize(self):
        """
        Initialize camera pipeline
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Try different camera indices
            for camera_index in [0, 1, 2]:
                logger.info(f"Trying camera index {camera_index}...")
                self.camera = cv2.VideoCapture(camera_index)
                
                if self.camera.isOpened():
                    logger.info(f"Camera opened on index {camera_index}")
                    break
                else:
                    self.camera.release()
                    self.camera = None
            
            if self.camera is None or not self.camera.isOpened():
                logger.error("Could not open camera on any index")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.camera.set(cv2.CAP_PROP_FPS, self.framerate)
            
            # Set buffer size to 1 for low latency
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Wait for camera to stabilize
            time.sleep(2)
            
            # Test capture
            ret, _ = self.camera.read()
            if not ret:
                logger.error("Camera opened but cannot read frames")
                self.camera.release()
                return False
            
            self.is_initialized = True
            logger.info("Camera initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            self.is_initialized = False
            return False
    
    def capture_frame(self):
        """
        Capture single frame from camera
        
        Returns:
            numpy.ndarray: RGB frame or None if capture fails
        """
        if not self.is_initialized:
            logger.error("Camera not initialized. Call initialize() first.")
            return None
        
        try:
            ret, frame = self.camera.read()
            
            if not ret or frame is None:
                logger.error("Failed to read frame from camera")
                return None
            
            # Convert BGR to RGB (OpenCV uses BGR, YOLO expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            return frame_rgb
            
        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
            return None
    
    def capture_continuous(self):
        """
        Generator for continuous frame capture
        
        Yields:
            numpy.ndarray: RGB frames
        """
        if not self.is_initialized:
            logger.error("Camera not initialized. Call initialize() first.")
            return
        
        logger.info("Starting continuous capture...")
        
        try:
            while True:
                frame = self.capture_frame()
                if frame is not None:
                    yield frame
                else:
                    logger.warning("Skipping empty frame")
                    time.sleep(0.01)
                    
        except KeyboardInterrupt:
            logger.info("Continuous capture interrupted by user")
        except Exception as e:
            logger.error(f"Error during continuous capture: {e}")
    
    def get_camera_info(self):
        """
        Get camera information
        
        Returns:
            dict: Camera properties and settings
        """
        if not self.is_initialized:
            return {"error": "Camera not initialized"}
        
        try:
            info = {
                "resolution": self.resolution,
                "framerate": self.framerate,
                "camera_model": "Raspberry Pi Camera"
            }
            return info
        except Exception as e:
            logger.error(f"Failed to get camera info: {e}")
            return {"error": str(e)}
    
    def release(self):
        """
        Release camera resources
        """
        if self.camera is not None:
            try:
                self.camera.release()
                self.is_initialized = False
                logger.info("Camera released successfully")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()


def test_camera():
    """Test camera functionality"""
    logger.info("Testing camera capture...")
    
    with CameraCapture(resolution=(640, 480), framerate=30) as capture:
        if not capture.is_initialized:
            logger.error("Camera initialization failed")
            return False
        
        # Get camera info
        info = capture.get_camera_info()
        logger.info(f"Camera info: {info}")
        
        # Capture test frame
        frame = capture.capture_frame()
        if frame is not None:
            logger.info(f"Captured frame shape: {frame.shape}, dtype: {frame.dtype}")
            logger.info("Camera test successful!")
            return True
        else:
            logger.error("Failed to capture frame")
            return False


if __name__ == "__main__":
    # Run camera test
    test_camera()
