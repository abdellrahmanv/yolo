"""
YOLOv5 Detection Module
Handles model loading, inference, and postprocessing
"""

import torch
import cv2
import numpy as np
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOv5Detector:
    """
    YOLOv5 object detection inference engine
    Optimized for Raspberry Pi deployment
    """
    
    def __init__(self, model_path, confidence_threshold=0.5, iou_threshold=0.45, device='cpu'):
        """
        Initialize YOLOv5 detector
        
        Args:
            model_path: Path to best.pt model file
            confidence_threshold: Minimum confidence for detections (0-1)
            iou_threshold: IoU threshold for NMS
            device: 'cpu' or 'cuda' (use 'cpu' for Raspberry Pi)
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        self.class_names = None
        self.is_loaded = False
        
        # Performance tracking
        self.inference_times = []
        
        logger.info(f"Initializing YOLOv5 detector with model: {model_path}")
        
    def load_model(self):
        """
        Load YOLOv5 model from file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            logger.info("Loading YOLOv5 model...")
            
            # Load model using torch.hub or ultralytics
            try:
                # Method 1: Using torch.hub (recommended)
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                           path=str(self.model_path), 
                                           force_reload=False)
            except Exception as e:
                logger.warning(f"torch.hub method failed: {e}")
                # Method 2: Direct loading
                from ultralytics import YOLO
                self.model = YOLO(str(self.model_path))
            
            # Configure model
            self.model.conf = self.confidence_threshold
            self.model.iou = self.iou_threshold
            self.model.to(self.device)
            
            # Extract class names
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
            elif hasattr(self.model, 'module') and hasattr(self.model.module, 'names'):
                self.class_names = self.model.module.names
            else:
                self.class_names = {i: f"class_{i}" for i in range(80)}  # Default COCO classes
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully with {len(self.class_names)} classes")
            logger.info(f"Classes: {self.class_names}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
            return False
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for YOLOv5 inference
        
        Args:
            frame: Input frame (RGB numpy array)
            
        Returns:
            numpy.ndarray: Preprocessed frame
        """
        # YOLOv5 expects RGB format (Picamera2 already provides RGB)
        # No need to convert from BGR to RGB
        return frame
    
    def detect(self, frame):
        """
        Run object detection on frame
        
        Args:
            frame: Input frame (RGB numpy array)
            
        Returns:
            list: List of detections, each containing:
                  - bbox: [x1, y1, x2, y2]
                  - confidence: detection confidence
                  - class_id: predicted class ID
                  - class_name: predicted class name
        """
        if not self.is_loaded:
            logger.error("Model not loaded. Call load_model() first.")
            return []
        
        try:
            # Start timing
            start_time = time.time()
            
            # Preprocess
            processed_frame = self.preprocess_frame(frame)
            
            # Run inference
            results = self.model(processed_frame)
            
            # End timing
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Parse results
            detections = self._parse_results(results)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def _parse_results(self, results):
        """
        Parse YOLOv5 results into standardized format
        
        Args:
            results: YOLOv5 detection results
            
        Returns:
            list: Parsed detections
        """
        detections = []
        
        try:
            # Handle different result formats
            if hasattr(results, 'xyxy'):
                # torch.hub format
                predictions = results.xyxy[0].cpu().numpy()
            elif hasattr(results, 'boxes'):
                # ultralytics format
                predictions = results[0].boxes.data.cpu().numpy()
            else:
                # Pandas dataframe format
                predictions = results.pandas().xyxy[0]
                predictions = predictions.values
            
            for pred in predictions:
                x1, y1, x2, y2, conf, cls = pred[:6]
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class_id': int(cls),
                    'class_name': self.class_names.get(int(cls), f"class_{int(cls)}")
                }
                
                detections.append(detection)
            
        except Exception as e:
            logger.error(f"Failed to parse results: {e}")
        
        return detections
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame (numpy array)
            detections: List of detections from detect()
            
        Returns:
            numpy.ndarray: Annotated frame
        """
        annotated_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            
            # Generate color based on class_id
            color = self._get_color(det['class_id'])
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = f"{class_name}: {conf:.2f}"
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_frame
    
    def _get_color(self, class_id):
        """
        Generate consistent color for class ID
        
        Args:
            class_id: Class ID
            
        Returns:
            tuple: BGR color
        """
        np.random.seed(class_id)
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        return color
    
    def get_average_inference_time(self):
        """
        Get average inference time
        
        Returns:
            float: Average inference time in seconds
        """
        if not self.inference_times:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times)
    
    def get_fps(self):
        """
        Get current FPS based on inference times
        
        Returns:
            float: Frames per second
        """
        avg_time = self.get_average_inference_time()
        if avg_time > 0:
            return 1.0 / avg_time
        return 0.0


if __name__ == "__main__":
    # Test detector with dummy image
    logger.info("Testing YOLOv5 detector...")
    
    # Create dummy frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Initialize detector (will fail without model, but tests imports)
    detector = YOLOv5Detector("../model/best.pt")
    logger.info("Detector initialization successful (model loading test)")
