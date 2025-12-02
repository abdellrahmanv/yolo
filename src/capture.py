"""
Camera Capture Module for Raspberry Pi
Handles frame acquisition from Raspberry Pi Camera using Picamera2
"""

import numpy as np
from picamera2 import Picamera2
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraCapture:
    """
    Camera capture interface using Picamera2 library
    Optimized for Raspberry Pi Camera Module 1.3
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
            # Create Picamera2 instance
            self.camera = Picamera2()
            
            # Configure camera
            config = self.camera.create_preview_configuration(
                main={"size": self.resolution, "format": "RGB888"},
                controls={"FrameRate": self.framerate}
            )
            
            self.camera.configure(config)
            
            # Set autofocus (if supported by camera model)
            try:
                # Try to set continuous autofocus using Picamera2's controls
                self.camera.set_controls({"AfMode": 2})  # 2 = Continuous AF
            except Exception as e:
                logger.info(f"Autofocus not available on this camera: {e}")
            
            # Start camera
            self.camera.start()
            
            # Wait for camera to stabilize
            time.sleep(2)
            
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
            # Capture frame as numpy array
            frame = self.camera.capture_array()
            return frame
            
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
                "sensor_resolution": self.camera.sensor_resolution,
                "camera_properties": self.camera.camera_properties
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
                self.camera.stop()
                self.camera.close()
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
