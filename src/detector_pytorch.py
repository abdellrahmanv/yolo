"""
YOLOv5 PyTorch Detection Module
Uses original best.pt model for reliable detection
"""

import numpy as np
import cv2
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOv5Detector:
    """
    YOLOv5 PyTorch detector
    Uses torch.hub to load and run YOLOv5 model
    """

    def __init__(self, model_path, confidence_threshold=0.5, iou_threshold=0.45):
        """
        Initialize YOLOv5 detector

        Args:
            model_path: Path to best.pt model file
            confidence_threshold: Minimum confidence (0-1)
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.is_loaded = False
        self.class_names = {0: 'glasses'}
        self.inference_times = []

    def load_model(self):
        """Load YOLOv5 model using torch.hub"""
        try:
            if not self.model_path.exists():
                print(f"ERROR: Model not found: {self.model_path}")
                return False

            print("Loading YOLOv5 model (this may take a moment)...")

            import torch

            # Force CPU for Raspberry Pi
            torch.set_num_threads(4)

            # Load model via torch.hub
            self.model = torch.hub.load(
                'ultralytics/yolov5',
                'custom',
                path=str(self.model_path),
                force_reload=False,
                device='cpu'
            )

            # Configure model
            self.model.conf = self.confidence_threshold
            self.model.iou = self.iou_threshold
            self.model.max_det = 10  # Max detections per image

            # Get class names from model
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names

            self.is_loaded = True
            print(f"Model loaded: {self.model_path.name}")
            print(f"Classes: {self.class_names}")
            return True

        except Exception as e:
            print(f"ERROR loading model: {e}")
            return False

    def detect(self, frame):
        """
        Run detection on frame

        Args:
            frame: RGB numpy array

        Returns:
            list: Detections with bbox, confidence, class_id, class_name
        """
        if not self.is_loaded:
            return []

        try:
            start_time = time.time()

            # Run inference (YOLOv5 handles preprocessing)
            results = self.model(frame)

            # Parse results
            detections = []
            
            # Get predictions as numpy array [x1, y1, x2, y2, conf, class]
            preds = results.xyxy[0].cpu().numpy()

            for pred in preds:
                x1, y1, x2, y2, conf, cls = pred
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class_id': int(cls),
                    'class_name': self.class_names.get(int(cls), f'class_{int(cls)}')
                }
                detections.append(detection)

            # Track timing
            self.inference_times.append(time.time() - start_time)

            return detections

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        annotated_frame = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']

            # Green color
            color = (0, 255, 0)

            # Bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"{class_name}: {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - label_h - 8), (x1 + label_w + 4, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return annotated_frame

    def get_average_inference_time(self):
        """Get average inference time in seconds"""
        if not self.inference_times:
            return 0.0
        recent = self.inference_times[-30:]
        return sum(recent) / len(recent)

    def get_fps(self):
        """Get FPS based on inference time"""
        avg_time = self.get_average_inference_time()
        return 1.0 / avg_time if avg_time > 0 else 0.0


if __name__ == "__main__":
    print("Testing YOLOv5 PyTorch detector...")
    
    detector = YOLOv5Detector("../model/best.pt")
    
    if detector.load_model():
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = detector.detect(test_frame)
        print(f"Test complete. Detections: {len(detections)}")
        print(f"Inference time: {detector.get_average_inference_time()*1000:.2f}ms")
