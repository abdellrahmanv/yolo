"""
Camera Capture Module for Raspberry Pi
Supports both Pi Camera Module 1.3 (legacy) and newer cameras
Auto-detects available camera interface
"""

import numpy as np
import cv2
import time
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraCapture:
    """
    Camera capture interface with multiple backend support:
    1. Picamera2 (recommended for Bookworm OS)
    2. Picamera (legacy, for older OS versions)
    3. OpenCV (fallback, for USB cameras)
    """

    def __init__(self, resolution=(320, 320), framerate=30):
        """
        Initialize camera capture

        Args:
            resolution: Tuple (width, height) for camera resolution
            framerate: Target frames per second
        """
        self.resolution = resolution
        self.framerate = framerate
        self.camera = None
        self.backend = None
        self.is_initialized = False

        logger.info(f"Initializing camera with resolution {resolution} @ {framerate}fps")

    def initialize(self):
        """
        Initialize camera with auto-detection of available backend

        Returns:
            bool: True if successful, False otherwise
        """
        # Try backends in order of preference
        backends = [
            ('picamera2', self._init_picamera2),
            ('picamera', self._init_picamera_legacy),
            ('opencv', self._init_opencv),
        ]

        for backend_name, init_func in backends:
            try:
                logger.info(f"Trying {backend_name} backend...")
                if init_func():
                    self.backend = backend_name
                    self.is_initialized = True
                    logger.info(f"Camera initialized successfully with {backend_name}")
                    return True
            except Exception as e:
                logger.debug(f"{backend_name} failed: {e}")
                continue

        logger.error("Failed to initialize camera with any backend")
        return False

    def _init_picamera2(self):
        """
        Initialize using Picamera2 (recommended for Raspberry Pi OS Bookworm)
        
        Returns:
            bool: True if successful
        """
        try:
            from picamera2 import Picamera2
            
            self.camera = Picamera2()
            
            # Configure for still/video capture
            config = self.camera.create_preview_configuration(
                main={"size": self.resolution, "format": "RGB888"},
                buffer_count=4
            )
            self.camera.configure(config)
            
            # Start camera
            self.camera.start()
            
            # Wait for camera to warm up
            time.sleep(1)
            
            logger.info("Picamera2 initialized")
            return True
            
        except ImportError:
            logger.debug("Picamera2 not installed")
            return False
        except Exception as e:
            logger.debug(f"Picamera2 init failed: {e}")
            if self.camera:
                try:
                    self.camera.close()
                except:
                    pass
                self.camera = None
            return False

    def _init_picamera_legacy(self):
        """
        Initialize using legacy Picamera (for Pi Camera Module 1.3)
        
        Returns:
            bool: True if successful
        """
        try:
            import picamera
            import picamera.array
            
            self.camera = picamera.PiCamera()
            self.camera.resolution = self.resolution
            self.camera.framerate = self.framerate
            
            # Wait for camera to warm up
            time.sleep(2)
            
            logger.info("Legacy Picamera initialized")
            return True
            
        except ImportError:
            logger.debug("Legacy picamera not installed")
            return False
        except Exception as e:
            logger.debug(f"Legacy picamera init failed: {e}")
            if self.camera:
                try:
                    self.camera.close()
                except:
                    pass
                self.camera = None
            return False

    def _init_opencv(self):
        """
        Initialize using OpenCV (fallback for USB cameras)
        
        Returns:
            bool: True if successful
        """
        try:
            # Try different camera indices
            for idx in [0, 1, -1]:
                self.camera = cv2.VideoCapture(idx)
                if self.camera.isOpened():
                    # Set resolution
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                    self.camera.set(cv2.CAP_PROP_FPS, self.framerate)
                    
                    # Test capture
                    ret, _ = self.camera.read()
                    if ret:
                        logger.info(f"OpenCV camera initialized on index {idx}")
                        return True
                    
                self.camera.release()
            
            self.camera = None
            return False
            
        except Exception as e:
            logger.debug(f"OpenCV init failed: {e}")
            if self.camera:
                try:
                    self.camera.release()
                except:
                    pass
                self.camera = None
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
            if self.backend == 'picamera2':
                # Picamera2 returns RGB directly
                frame = self.camera.capture_array()
                return frame

            elif self.backend == 'picamera':
                # Legacy picamera - capture to numpy array
                import picamera.array
                with picamera.array.PiRGBArray(self.camera) as output:
                    self.camera.capture(output, format='rgb', use_video_port=True)
                    return output.array

            elif self.backend == 'opencv':
                ret, frame = self.camera.read()
                if ret:
                    # OpenCV returns BGR, convert to RGB
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return None

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
            if self.backend == 'picamera2':
                while True:
                    frame = self.camera.capture_array()
                    if frame is not None:
                        yield frame

            elif self.backend == 'picamera':
                import picamera.array
                with picamera.array.PiRGBArray(self.camera, size=self.resolution) as output:
                    for _ in self.camera.capture_continuous(output, format='rgb', use_video_port=True):
                        yield output.array
                        output.truncate(0)

            elif self.backend == 'opencv':
                while True:
                    ret, frame = self.camera.read()
                    if ret:
                        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
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

        return {
            "resolution": self.resolution,
            "framerate": self.framerate,
            "backend": self.backend,
            "status": "running"
        }

    def release(self):
        """
        Release camera resources
        """
        if self.camera is None:
            return

        try:
            if self.backend == 'picamera2':
                self.camera.stop()
                self.camera.close()
            elif self.backend == 'picamera':
                self.camera.close()
            elif self.backend == 'opencv':
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

    with CameraCapture(resolution=(320, 320), framerate=30) as capture:
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
            
            # Save test image
            cv2.imwrite("test_capture.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            logger.info("Test image saved to test_capture.jpg")
            
            logger.info("Camera test successful!")
            return True
        else:
            logger.error("Failed to capture frame")
            return False


if __name__ == "__main__":
    test_camera()
