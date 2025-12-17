"""
Threaded Camera Capture Module - High FPS for Raspberry Pi
Background thread continuously captures frames for maximum throughput
"""

import numpy as np
import cv2
import time
import logging
import threading
from collections import deque

logger = logging.getLogger(__name__)


class ThreadedCamera:
    """
    Threaded camera capture - background thread grabs frames continuously.
    Main thread always gets the latest frame without waiting.
    """

    def __init__(self, resolution=(320, 320), framerate=30, buffer_size=2, use_mjpg=True):
        self.resolution = resolution
        self.framerate = framerate
        self.buffer_size = buffer_size
        self.use_mjpg = use_mjpg
        
        self.camera = None
        self.backend = None
        self.is_initialized = False
        
        # Threading
        self.frame_buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.latest_frame = None

    def initialize(self):
        """Initialize camera and start capture thread"""
        backends = [
            ('picamera2', self._init_picamera2),
            ('picamera', self._init_picamera_legacy),
            ('opencv', self._init_opencv),
        ]

        for backend_name, init_func in backends:
            try:
                if init_func():
                    self.backend = backend_name
                    self.is_initialized = True
                    logger.info(f"Camera initialized with {backend_name}")
                    
                    # Start capture thread
                    self._start_thread()
                    return True
            except Exception as e:
                logger.debug(f"{backend_name} failed: {e}")
                continue

        logger.error("Failed to initialize camera")
        return False

    def _init_picamera2(self):
        """Initialize Picamera2 (fast)"""
        try:
            from picamera2 import Picamera2
            
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": self.resolution, "format": "RGB888"},
                buffer_count=2
            )
            self.camera.configure(config)
            self.camera.start()
            time.sleep(0.05)  # Minimal warmup
            return True
        except:
            return False

    def _init_picamera_legacy(self):
        """Initialize legacy Picamera"""
        try:
            import picamera
            self.camera = picamera.PiCamera()
            self.camera.resolution = self.resolution
            self.camera.framerate = self.framerate
            time.sleep(0.1)  # Minimal warmup
            return True
        except:
            return False

    def _init_opencv(self):
        """Initialize OpenCV camera (fast - no test frame)"""
        try:
            for idx in [0, 1, -1]:
                self.camera = cv2.VideoCapture(idx)
                if self.camera.isOpened():
                    # Use MJPG format for faster capture on Pi
                    if self.use_mjpg:
                        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                    self.camera.set(cv2.CAP_PROP_FPS, self.framerate)
                    self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    return True  # Skip test frame - thread grabs first
            return False
        except:
            return False

    def _start_thread(self):
        """Start background capture thread"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        # Wait for first frame (short timeout for fast startup)
        timeout = 0.3
        start = time.time()
        while self.latest_frame is None and time.time() - start < timeout:
            time.sleep(0.005)
        
        if self.latest_frame is not None:
            print(f"Camera ready in {time.time()-start:.2f}s")
        else:
            print("Camera starting (first frame pending)...")

    def _capture_loop(self):
        """Background capture loop"""
        while self.running:
            try:
                frame = self._grab_frame()
                if frame is not None:
                    with self.lock:
                        self.latest_frame = frame
                        self.frame_buffer.append(frame)
            except Exception as e:
                logger.error(f"Capture error: {e}")
                time.sleep(0.01)

    def _grab_frame(self):
        """Grab single frame from camera"""
        try:
            if self.backend == 'picamera2':
                return self.camera.capture_array()
            
            elif self.backend == 'picamera':
                import picamera.array
                with picamera.array.PiRGBArray(self.camera) as output:
                    self.camera.capture(output, format='rgb', use_video_port=True)
                    return output.array.copy()
            
            elif self.backend == 'opencv':
                ret, frame = self.camera.read()
                if ret:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None
        except:
            return None

    def capture_frame(self):
        """Get latest frame (non-blocking)"""
        with self.lock:
            return self.latest_frame

    def get_frame_nowait(self):
        """Alias for capture_frame"""
        return self.capture_frame()

    def capture_continuous(self):
        """Generator yielding latest frames"""
        while self.running and self.is_initialized:
            frame = self.capture_frame()
            if frame is not None:
                yield frame
            else:
                time.sleep(0.001)

    def reset(self):
        """Reset camera (stop thread, reinit, restart thread)"""
        logger.info("Resetting camera...")
        
        # Stop thread
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        # Release camera
        self._release_camera()
        time.sleep(0.1)
        
        # Clear buffer
        with self.lock:
            self.frame_buffer.clear()
            self.latest_frame = None
        
        # Reinitialize
        self.camera = None
        self.backend = None
        self.is_initialized = False
        
        return self.initialize()

    def _release_camera(self):
        """Release camera hardware"""
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
        except:
            pass

    def release(self):
        """Stop thread and release camera"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self._release_camera()
        self.is_initialized = False
        logger.info("Camera released")

    def get_camera_info(self):
        """Get camera info"""
        return {
            "resolution": self.resolution,
            "framerate": self.framerate,
            "backend": self.backend,
            "threaded": True,
            "buffer_size": self.buffer_size
        }

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, *args):
        self.release()


# Alias for drop-in replacement
CameraCapture = ThreadedCamera


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing threaded camera...")
    
    with ThreadedCamera(resolution=(320, 320)) as cam:
        if cam.is_initialized:
            print(f"Camera: {cam.get_camera_info()}")
            
            # FPS test
            start = time.time()
            count = 0
            while time.time() - start < 3.0:
                frame = cam.capture_frame()
                if frame is not None:
                    count += 1
            
            fps = count / 3.0
            print(f"Captured {count} frames in 3s = {fps:.1f} FPS")
