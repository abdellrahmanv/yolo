"""
YOLOv5 TFLite Detection Module
Handles INT8 quantized model inference for Raspberry Pi
Optimized for glasses detection
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
    """

    def __init__(self, model_path, confidence_threshold=0.5, iou_threshold=0.45):
        """
        Initialize TFLite detector

        Args:
            model_path: Path to best-int8.tflite model file
            confidence_threshold: Minimum confidence for detections (0-1)
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.is_loaded = False

        # Model specifications (from analysis)
        self.input_shape = (320, 320)  # Height, Width
        self.input_dtype = np.uint8
        
        # Quantization parameters
        self.input_scale = 0.003921568859368563  # ~1/255
        self.input_zero_point = 0
        self.output_scale = 0.018480218946933746
        self.output_zero_point = 3

        # Class names for glasses detection
        # Update this based on your training classes
        self.class_names = {0: 'glasses'}  # Modify if you have more classes
        
        # Performance tracking
        self.inference_times = []

        logger.info(f"Initializing TFLite detector with model: {model_path}")

    def load_model(self):
        """
        Load TFLite model

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False

            logger.info("Loading TFLite model...")

            # Import tflite runtime
            try:
                from tflite_runtime.interpreter import Interpreter
                logger.info("Using tflite-runtime (optimized)")
            except ImportError:
                try:
                    from tensorflow.lite.python.interpreter import Interpreter
                    logger.warning("Using full TensorFlow (consider using tflite-runtime)")
                except ImportError:
                    logger.error("Neither tflite-runtime nor tensorflow found!")
                    logger.error("Install with: pip install tflite-runtime")
                    return False

            # Create interpreter
            self.interpreter = Interpreter(
                model_path=str(self.model_path),
                num_threads=4  # Use 4 CPU threads on Pi
            )
            
            # Allocate tensors
            self.interpreter.allocate_tensors()

            # Get input/output details
            self.input_details = self.interpreter.get_input_details()[0]
            self.output_details = self.interpreter.get_output_details()[0]

            # Update quantization parameters from model
            input_quant = self.input_details.get('quantization_parameters', {})
            if 'scales' in input_quant and len(input_quant['scales']) > 0:
                self.input_scale = float(input_quant['scales'][0])
                self.input_zero_point = int(input_quant['zero_points'][0])

            output_quant = self.output_details.get('quantization_parameters', {})
            if 'scales' in output_quant and len(output_quant['scales']) > 0:
                self.output_scale = float(output_quant['scales'][0])
                self.output_zero_point = int(output_quant['zero_points'][0])

            # Log model info
            logger.info(f"Input shape: {self.input_details['shape']}")
            logger.info(f"Input dtype: {self.input_details['dtype']}")
            logger.info(f"Output shape: {self.output_details['shape']}")
            logger.info(f"Output dtype: {self.output_details['dtype']}")

            self.is_loaded = True
            logger.info("TFLite model loaded successfully")
            logger.info(f"Classes: {self.class_names}")

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
            return False

    def preprocess(self, frame):
        """
        Preprocess frame for TFLite inference

        Args:
            frame: Input frame (RGB numpy array, any size)

        Returns:
            numpy.ndarray: Preprocessed frame ready for inference
            tuple: Original frame dimensions (height, width)
        """
        original_h, original_w = frame.shape[:2]

        # Resize to model input size (320x320)
        resized = cv2.resize(frame, (self.input_shape[1], self.input_shape[0]))

        # Ensure RGB format and uint8 type
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
            list: List of detections, each containing:
                  - bbox: [x1, y1, x2, y2] in original frame coordinates
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
            input_data, original_size = self.preprocess(frame)

            # Set input tensor
            self.interpreter.set_tensor(self.input_details['index'], input_data)

            # Run inference
            self.interpreter.invoke()

            # Get output tensor
            output_data = self.interpreter.get_tensor(self.output_details['index'])

            # Dequantize output
            output_float = (output_data.astype(np.float32) - self.output_zero_point) * self.output_scale

            # Parse detections
            detections = self._parse_output(output_float[0], original_size)

            # Apply NMS
            detections = self._apply_nms(detections)

            # End timing
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)

            return detections

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

    def _parse_output(self, output, original_size):
        """
        Parse YOLOv5 TFLite output

        Args:
            output: Model output [6300, 6] - (x_center, y_center, w, h, conf, class_id)
            original_size: Original frame size (height, width)

        Returns:
            list: Parsed detections
        """
        detections = []
        original_h, original_w = original_size

        # Scale factors
        scale_x = original_w / self.input_shape[1]
        scale_y = original_h / self.input_shape[0]

        for pred in output:
            # YOLOv5 output format: [x_center, y_center, width, height, confidence, class_id]
            x_center, y_center, width, height, confidence, class_id = pred[:6]

            # Filter by confidence
            if confidence < self.confidence_threshold:
                continue

            # Convert from center format to corner format
            x1 = (x_center - width / 2) * scale_x
            y1 = (y_center - height / 2) * scale_y
            x2 = (x_center + width / 2) * scale_x
            y2 = (y_center + height / 2) * scale_y

            # Clip to frame boundaries
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(original_w, int(x2))
            y2 = min(original_h, int(y2))

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            class_id = int(class_id)
            detection = {
                'bbox': [x1, y1, x2, y2],
                'confidence': float(confidence),
                'class_id': class_id,
                'class_name': self.class_names.get(class_id, f"class_{class_id}")
            }
            detections.append(detection)

        return detections

    def _apply_nms(self, detections):
        """
        Apply Non-Maximum Suppression

        Args:
            detections: List of detections

        Returns:
            list: Filtered detections after NMS
        """
        if len(detections) == 0:
            return []

        # Extract boxes and scores
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])

        # Apply OpenCV NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.confidence_threshold,
            self.iou_threshold
        )

        # Filter detections
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        
        return []

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

            # Color for glasses detection (green)
            color = (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Create label
            label = f"{class_name}: {conf:.2f}"

            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), (x1 + label_w + 5, y1), color, -1)

            # Draw label text
            cv2.putText(annotated_frame, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return annotated_frame

    def get_average_inference_time(self):
        """
        Get average inference time

        Returns:
            float: Average inference time in seconds
        """
        if not self.inference_times:
            return 0.0
        # Use last 30 samples for moving average
        recent_times = self.inference_times[-30:]
        return sum(recent_times) / len(recent_times)

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
    # Test detector
    logger.info("Testing TFLite detector...")

    # Create dummy frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Initialize detector
    detector = TFLiteDetector("../model/best-int8.tflite")
    
    if detector.load_model():
        logger.info("Model loaded successfully!")
        
        # Test inference
        detections = detector.detect(test_frame)
        logger.info(f"Test inference complete. Detections: {len(detections)}")
        logger.info(f"Inference time: {detector.get_average_inference_time()*1000:.2f}ms")
    else:
        logger.error("Failed to load model")
