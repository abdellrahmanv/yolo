"""
Camera Capture Module for Raspberry Pi
Handles frame acquisition from Raspberry Pi Camera using libcamera subprocess
Compatible with Raspberry Pi Camera Module 1.3
"""

import numpy as np
import cv2
import time
import logging
import subprocess
import threading
from queue import Queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraCapture:
    """
    Camera capture interface using libcamera-vid subprocess
    Works with Raspberry Pi Camera via libcamera
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
        self.process = None
        self.frame_queue = Queue(maxsize=2)
        self.is_initialized = False
        self.capture_thread = None
        self.running = False
        
        logger.info(f"Initializing camera with resolution {resolution} @ {framerate}fps")
        
    def initialize(self):
        """
        Initialize camera pipeline using rpicam-vid or libcamera-vid
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Try rpicam-vid first (newer), then libcamera-vid (older)
            camera_commands = ['rpicam-vid', 'libcamera-vid']
            
            cmd = None
            for cam_cmd in camera_commands:
                try:
                    # Test if command exists
                    subprocess.run([cam_cmd, '--version'], 
                                 stdout=subprocess.DEVNULL, 
                                 stderr=subprocess.DEVNULL,
                                 timeout=1)
                    
                    # Build command
                    cmd = [
                        cam_cmd,
                        '--width', str(self.resolution[0]),
                        '--height', str(self.resolution[1]),
                        '--framerate', str(self.framerate),
                        '--codec', 'yuv420',
                        '--timeout', '0',  # Run indefinitely
                        '-o', '-',  # Output to stdout
                        '--nopreview'
                    ]
                    logger.info(f"Using {cam_cmd}")
                    break
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    continue
            
            if cmd is None:
                logger.error("Neither rpicam-vid nor libcamera-vid found")
                return False
            
            logger.info(f"Starting camera: {' '.join(cmd)}")
            
            # Start subprocess
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**8
            )
            
            # Wait for camera to initialize
            time.sleep(2)
            
            # Check if process is running
            if self.process.poll() is not None:
                logger.error("Camera process failed to start")
                return False
            
            self.running = True
            self.is_initialized = True
            
            # Start capture thread
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            logger.info("Camera initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            self.is_initialized = False
            return False
    
    def _capture_loop(self):
        """Background thread to continuously capture frames"""
        frame_size = self.resolution[0] * self.resolution[1] * 3 // 2  # YUV420 size
        
        while self.running:
            try:
                # Read YUV420 frame
                raw_frame = self.process.stdout.read(frame_size)
                
                if len(raw_frame) != frame_size:
                    logger.warning("Incomplete frame received")
                    continue
                
                # Convert YUV420 to RGB
                yuv_frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                    (self.resolution[1] * 3 // 2, self.resolution[0])
                )
                
                # Convert YUV to BGR then to RGB
                bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                
                # Put frame in queue (drop old frames if full)
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                
                self.frame_queue.put(rgb_frame)
                
            except Exception as e:
                if self.running:
                    logger.error(f"Error in capture loop: {e}")
                break
    
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
            # Get frame from queue (with timeout)
            frame = self.frame_queue.get(timeout=1.0)
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
        self.running = False
        
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2)
        
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
                self.is_initialized = False
                logger.info("Camera released successfully")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
                try:
                    self.process.kill()
                except:
                    pass
    
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
