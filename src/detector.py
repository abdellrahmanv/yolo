"""
YOLOv8n TFLite Detection Module
Handles float16 quantized model inference for Raspberry Pi
Optimized for human detection (COCO person class)
"""

import numpy as np
import cv2
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TFLiteDetector:
    """
    TFLite float16 YOLOv8n detector
    Optimized for Raspberry Pi deployment
    Detects humans (COCO person class = index 0)

    YOLOv8 output format: [1, 84, N]  (transposed compared to v5)
      - After transpose to [N, 84]:
        columns 0-3 = x_center, y_center, width, height  (normalized 0-1)
        columns 4-83 = 80 COCO class scores  (person = class 0 = column 4)
      - NO objectness column (unlike YOLOv5)
    """

    PERSON_CLASS_ID = 0   # COCO person = class 0 → column index 4

    def __init__(self, model_path, confidence_threshold=0.40, iou_threshold=0.45):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.is_loaded = False

        self.input_shape = (224, 224)  # H, W — smaller = faster on Pi
        self.num_classes = 80
        self.class_names = {0: 'human'}

        # Performance tracking
        self.inference_times = []

    def load_model(self):
        """Load TFLite model"""
        try:
            if not self.model_path.exists():
                print(f"ERROR: Model not found: {self.model_path}")
                return False

            Interpreter = None
            try:
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                try:
                    from tflite_runtime.interpreter import Interpreter
                except ImportError:
                    try:
                        from tensorflow.lite.python.interpreter import Interpreter
                    except ImportError:
                        pass

            if Interpreter is None:
                print("ERROR: No TFLite runtime found.")
                print("  For Python 3.12+: pip install ai-edge-litert")
                print("  For Python 3.9-3.11: pip install tflite-runtime")
                return False

            self.interpreter = Interpreter(
                model_path=str(self.model_path),
                num_threads=4
            )
            self.interpreter.allocate_tensors()

            self.input_details = self.interpreter.get_input_details()[0]
            self.output_details = self.interpreter.get_output_details()[0]

            self.is_loaded = True
            in_shape = self.input_details['shape']
            out_shape = self.output_details['shape']
            print(f"Model loaded: {self.model_path.name}")
            print(f"Input: {in_shape} ({self.input_details['dtype'].__name__})")
            print(f"Output: {out_shape} ({self.output_details['dtype'].__name__})")
            return True

        except Exception as e:
            print(f"ERROR loading model: {e}")
            return False

    def preprocess(self, frame):
        """Preprocess frame for YOLOv8 inference (float32 0-1)"""
        original_h, original_w = frame.shape[:2]

        resized = cv2.resize(frame, (self.input_shape[1], self.input_shape[0]))

        # YOLOv8 float16 TFLite expects float32 input normalized to 0-1
        input_data = resized.astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        return input_data, (original_h, original_w)

    def detect(self, frame):
        """
        Run detection on frame.

        Args:
            frame: Input frame (RGB or BGR numpy array)

        Returns:
            list of dicts with 'bbox', 'confidence', 'class_id', 'class_name'
        """
        if not self.is_loaded:
            return []

        try:
            start_time = time.time()

            input_data, original_size = self.preprocess(frame)

            self.interpreter.set_tensor(self.input_details['index'], input_data)
            self.interpreter.invoke()

            output_data = self.interpreter.get_tensor(self.output_details['index'])

            # YOLOv8 output: [1, 84, N] → transpose to [N, 84]
            output = output_data[0].T

            detections = self._parse_output(output, original_size)
            detections = self._apply_nms(detections)

            self.inference_times.append(time.time() - start_time)
            if len(self.inference_times) > 30:
                self.inference_times = self.inference_times[-30:]

            return detections

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

    def _parse_output(self, output, original_size):
        """
        Parse YOLOv8 TFLite output — fully vectorized.

        output shape: [N, 84]
          cols 0-3  = x_center, y_center, width, height  (normalized 0-1)
          cols 4-83 = 80 COCO class scores
        No objectness column in YOLOv8.
        """
        original_h, original_w = original_size

        x_center = output[:, 0]
        y_center = output[:, 1]
        width    = output[:, 2]
        height   = output[:, 3]

        class_scores = output[:, 4:4 + self.num_classes]

        # Person score and best-class check
        person_score = class_scores[:, self.PERSON_CLASS_ID]
        best_class = np.argmax(class_scores, axis=1)
        is_person = (best_class == self.PERSON_CLASS_ID)

        # Filter: person is top class AND score above threshold AND valid box
        valid = (is_person
                 & (person_score >= self.confidence_threshold)
                 & (width > 0.001)
                 & (height > 0.001))

        if not np.any(valid):
            return []

        x_center = x_center[valid]
        y_center = y_center[valid]
        width    = width[valid]
        height   = height[valid]
        confidence = person_score[valid]

        # Coordinates are normalized (0-1). Scale to original frame.
        scale_x = original_w
        scale_y = original_h

        x1 = ((x_center - width / 2) * scale_x).astype(np.int32)
        y1 = ((y_center - height / 2) * scale_y).astype(np.int32)
        x2 = ((x_center + width / 2) * scale_x).astype(np.int32)
        y2 = ((y_center + height / 2) * scale_y).astype(np.int32)

        x1 = np.clip(x1, 0, original_w)
        y1 = np.clip(y1, 0, original_h)
        x2 = np.clip(x2, 0, original_w)
        y2 = np.clip(y2, 0, original_h)

        box_w = x2 - x1
        box_h = y2 - y1
        box_valid = (x2 > x1) & (y2 > y1) & (box_w >= 15) & (box_h >= 15)

        detections = []
        for i in np.where(box_valid)[0]:
            detections.append({
                'bbox': [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
                'confidence': float(confidence[i]),
                'class_id': self.PERSON_CLASS_ID,
                'class_name': 'human'
            })

        return detections

    def _apply_nms(self, detections):
        """Apply Non-Maximum Suppression"""
        if len(detections) == 0:
            return []

        boxes_xyxy = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])

        # cv2.dnn.NMSBoxes expects [x, y, w, h]
        boxes_xywh = boxes_xyxy.copy()
        boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            scores.tolist(),
            self.confidence_threshold,
            self.iou_threshold
        )

        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]

        return []

    def draw_detections(self, frame, detections, in_place=True):
        """Draw bounding boxes on frame"""
        annotated_frame = frame if in_place else frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']

            color = (0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

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
    print("Testing TFLite detector...")

    detector = TFLiteDetector("../model/yolov8n-fp16.tflite")

    if detector.load_model():
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = detector.detect(test_frame)
        print(f"Test complete. Detections: {len(detections)}")
        print(f"Inference time: {detector.get_average_inference_time()*1000:.2f}ms")
