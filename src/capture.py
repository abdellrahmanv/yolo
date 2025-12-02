"""
Camera Capture Module for Raspberry Pi
Handles frame acquisition from Raspberry Pi Camera using legacy picamera
"""

import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraCapture:
    """
    Camera capture interface using legacy picamera library
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
        self.raw_capture = None
        self.is_initialized = False
        
        logger.info(f"Initializing camera with resolution {resolution} @ {framerate}fps")
        
    def initialize(self):
        """
        Initialize camera pipeline
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create PiCamera instance
            self.camera = PiCamera()
            
            # Configure camera
            self.camera.resolution = self.resolution
            self.camera.framerate = self.framerate
            
            # Create RGB array buffer for captures
            self.raw_capture = PiRGBArray(self.camera, size=self.resolution)
            
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
            # Clear the stream
            self.raw_capture.truncate(0)
            
            # Capture frame to array
            self.camera.capture(self.raw_capture, format='rgb', use_video_port=True)
            frame = self.raw_capture.array
            
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
