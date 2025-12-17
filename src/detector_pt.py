"""
YOLOv5 PyTorch Detection Module
Uses the original best.pt model for glasses detection
"""

import numpy as np
import cv2
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PyTorchDetector:
    """
    YOLOv5 PyTorch detector using torch.hub
    Uses best.pt model for glasses detection
    """

    def __init__(self, model_path, confidence_threshold=0.5, iou_threshold=0.45):
        """
        Initialize PyTorch detector

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

            print(f"Loading PyTorch model: {self.model_path.name}")
            
            import torch
            
            # Force CPU mode for Raspberry Pi
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
            self.model.to('cpu')
            
            # Get class names
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
            
            self.is_loaded = True
            print(f"Model loaded successfully")
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

            # Run inference
            results = self.model(frame)

            # Parse results
            detections = self._parse_results(results)

            # Track timing
            self.inference_times.append(time.time() - start_time)

            return detections

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

    def _parse_results(self, results):
        """Parse YOLOv5 results"""
        detections = []

        try:
            # Get predictions as numpy array [x1, y1, x2, y2, conf, class]
            if hasattr(results, 'xyxy'):
                preds = results.xyxy[0].cpu().numpy()
            elif hasattr(results, 'pandas'):
                preds = results.pandas().xyxy[0].values
            else:
                return []

            for pred in preds:
                x1, y1, x2, y2, conf, cls = pred[:6]

                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class_id': int(cls),
                    'class_name': self.class_names.get(int(cls), f"class_{int(cls)}")
                }
                detections.append(detection)

        except Exception as e:
            logger.error(f"Parse error: {e}")

        return detections

    def draw_detections(self, frame, detections):
        """Draw bounding boxes on frame"""
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            name = det['class_name']

            # Green box
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"{name}: {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return annotated

    def get_average_inference_time(self):
        """Get average inference time"""
        if not self.inference_times:
            return 0.0
        recent = self.inference_times[-30:]
        return sum(recent) / len(recent)

    def get_fps(self):
        """Get FPS"""
        avg = self.get_average_inference_time()
        return 1.0 / avg if avg > 0 else 0.0


if __name__ == "__main__":
    print("Testing PyTorch detector...")
    
    detector = PyTorchDetector("../model/best.pt")
    
    if detector.load_model():
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = detector.detect(test_frame)
        print(f"Test complete. Detections: {len(detections)}")
        print(f"Inference time: {detector.get_average_inference_time()*1000:.2f}ms")
