"""
YOLOv5 TFLite Detection Module
Handles INT8 quantized model inference for Raspberry Pi
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
    TFLite INT8 quantized YOLOv5 detector
    Optimized for Raspberry Pi deployment
    Detects humans (COCO person class = index 0)
    """

    # COCO class index for person
    PERSON_CLASS_ID = 0

    def __init__(self, model_path, confidence_threshold=0.25, iou_threshold=0.45):
        """
        Initialize TFLite detector

        Args:
            model_path: Path to yolov5n-int8.tflite model file
            confidence_threshold: Minimum confidence for detections (0-1)
                                  NOTE: Lowered to 0.25 for INT8 quantized models
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.is_loaded = False

        # Model specifications
        self.input_shape = (320, 320)  # Height, Width
        self.input_dtype = np.uint8

        # Quantization parameters (will be updated from model)
        self.output_scale = 0.00586441
        self.output_zero_point = 4

        # Number of classes in COCO model
        self.num_classes = 80

        # INT8 quantization crushes objectness scores, so we gate on
        # objectness instead of multiplying it into the confidence.
        # Real detections typically have raw objectness >= 7 (~0.018)
        # while noise sits at raw=5 (~0.006), one step above zero-point.
        self.objectness_gate = 0.015

        # We only care about person (class 0)
        self.class_names = {0: 'human'}

        # Performance tracking
        self.inference_times = []

    def load_model(self):
        """Load TFLite model"""
        try:
            if not self.model_path.exists():
                print(f"ERROR: Model not found: {self.model_path}")
                return False

            # Import TFLite interpreter (supports Python 3.12+)
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

            # Create interpreter
            self.interpreter = Interpreter(
                model_path=str(self.model_path),
                num_threads=4
            )
            self.interpreter.allocate_tensors()

            # Get input/output details
            self.input_details = self.interpreter.get_input_details()[0]
            self.output_details = self.interpreter.get_output_details()[0]

            # Update quantization parameters from model
            output_quant = self.output_details.get('quantization_parameters', {})
            if 'scales' in output_quant and len(output_quant['scales']) > 0:
                self.output_scale = float(output_quant['scales'][0])
                self.output_zero_point = int(output_quant['zero_points'][0])

            self.is_loaded = True
            print(f"Model loaded: {self.model_path.name}")
            print(f"Input: {self.input_details['shape']}, Output: {self.output_details['shape']}")
            return True

        except Exception as e:
            print(f"ERROR loading model: {e}")
            return False

    def preprocess(self, frame):
        """Preprocess frame for inference"""
        original_h, original_w = frame.shape[:2]

        # Resize to model input size (320x320)
        resized = cv2.resize(frame, (self.input_shape[1], self.input_shape[0]))

        # Ensure uint8
        if resized.dtype != np.uint8:
            resized = resized.astype(np.uint8)

        # Add batch dimension [1, 320, 320, 3]
        input_data = np.expand_dims(resized, axis=0)

        return input_data, (original_h, original_w)

    def detect(self, frame):
        """
        Run object detection on frame

        Args:
            frame: Input frame (RGB numpy array)

        Returns:
            list: List of detections with bbox, confidence, class_id, class_name
        """
        if not self.is_loaded:
            return []

        try:
            start_time = time.time()

            # Preprocess
            input_data, original_size = self.preprocess(frame)

            # Run inference
            self.interpreter.set_tensor(self.input_details['index'], input_data)
            self.interpreter.invoke()

            # Get and dequantize output
            output_data = self.interpreter.get_tensor(self.output_details['index'])
            output_float = (output_data.astype(np.float32) - self.output_zero_point) * self.output_scale

            # NOTE: Do NOT clamp the whole tensor to zero — negative
            # dequantized values are valid for small bbox widths/heights.
            # Clamping would destroy bounding boxes.

            # Parse detections
            detections = self._parse_output(output_float[0], original_size)

            # Apply NMS
            detections = self._apply_nms(detections)

            # Track timing (cap at 30 entries)
            self.inference_times.append(time.time() - start_time)
            if len(self.inference_times) > 30:
                self.inference_times = self.inference_times[-30:]

            return detections

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

    def _parse_output(self, output, original_size):
        """
        Parse YOLOv5 TFLite output - VECTORIZED for speed

        YOLOv5n COCO TFLite output format: [6300, 85]
        Each row: [x_center, y_center, width, height, objectness, class_0 ... class_79]
        We only extract person detections (class index 0 = column 5).
        """
        original_h, original_w = original_size
        input_h, input_w = self.input_shape

        # Vectorized: extract bbox + objectness
        x_center = output[:, 0]
        y_center = output[:, 1]
        width = output[:, 2]
        height = output[:, 3]
        objectness = output[:, 4]

        # All 80 class scores (columns 5-84)
        all_class_scores = output[:, 5:5 + self.num_classes]

        # Only keep predictions where person (class 0) is the TOP scoring class
        best_class = np.argmax(all_class_scores, axis=1)
        is_person = (best_class == self.PERSON_CLASS_ID)

        # Person class score
        person_score = all_class_scores[:, self.PERSON_CLASS_ID]

        # Require person score to be clearly dominant:
        # person must beat the 2nd-best class by a margin.
        # This eliminates anchors where several classes score similarly
        # (ambiguous = not a real person).
        sorted_scores = np.sort(all_class_scores, axis=1)
        second_best = sorted_scores[:, -2]
        margin = person_score - second_best
        has_clear_margin = margin > 0.05

        # INT8 quantization compresses objectness into a very narrow
        # range (often max ~0.08), so the traditional
        #   confidence = objectness * class_score
        # yields values too small to be usable.
        #
        # Instead we use the person CLASS SCORE directly as the
        # confidence and keep objectness only as a hard gate
        # to suppress pure-noise anchors.
        confidence = person_score

        valid = (is_person
                 & has_clear_margin
                 & (confidence >= self.confidence_threshold)
                 & (objectness > self.objectness_gate)
                 & (width > 0.001)
                 & (height > 0.001))

        if not np.any(valid):
            return []

        # Filter to valid only
        x_center = x_center[valid]
        y_center = y_center[valid]
        width = width[valid]
        height = height[valid]
        confidence = confidence[valid]

        # Scale to original size in one step
        scale_x = original_w / input_w
        scale_y = original_h / input_h

        x1 = ((x_center - width / 2) * input_w * scale_x).astype(np.int32)
        y1 = ((y_center - height / 2) * input_h * scale_y).astype(np.int32)
        x2 = ((x_center + width / 2) * input_w * scale_x).astype(np.int32)
        y2 = ((y_center + height / 2) * input_h * scale_y).astype(np.int32)

        # Clip to frame
        x1 = np.clip(x1, 0, original_w)
        y1 = np.clip(y1, 0, original_h)
        x2 = np.clip(x2, 0, original_w)
        y2 = np.clip(y2, 0, original_h)

        # Filter invalid boxes and tiny detections
        box_w = x2 - x1
        box_h = y2 - y1
        box_valid = (x2 > x1) & (y2 > y1) & (box_w >= 15) & (box_h >= 15)

        # Build detections list
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

        # Convert [x1, y1, x2, y2] -> [x, y, w, h] for cv2.dnn.NMSBoxes
        boxes_xywh = boxes_xyxy.copy()
        boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]  # w = x2 - x1
        boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]  # h = y2 - y1

        # OpenCV NMS expects [x, y, w, h]
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
        """Draw bounding boxes on frame (in-place for speed)"""
        annotated_frame = frame if in_place else frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']

            # Green color for human detection
            color = (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Label with confidence
            label = f"{class_name}: {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Label background
            cv2.rectangle(annotated_frame, (x1, y1 - label_h - 8), (x1 + label_w + 4, y1), color, -1)
            
            # Label text
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
    
    detector = TFLiteDetector("../model/yolov5n-int8.tflite")
    
    if detector.load_model():
        # Test with dummy frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = detector.detect(test_frame)
        print(f"Test complete. Detections: {len(detections)}")
        print(f"Inference time: {detector.get_average_inference_time()*1000:.2f}ms")
