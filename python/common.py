"""Common utilities for Hailo-8L + ONNX hybrid inference"""

import numpy as np
import cv2
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List
from hailo_platform import VDevice, HEF, ConfigureParams, InputVStreamParams, OutputVStreamParams, InferVStreams, HailoStreamInterface, FormatType
import math

# ultralytics removed


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# ============================================================================
# Common Image Operations
# ============================================================================

def letterbox_image(img, target_size=640, color=(114, 114, 114)):
    """Resize image with aspect ratio preservation (letterbox)"""
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h))
    
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    
    padded = np.full((target_size, target_size, 3), color, dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    
    return padded, scale, pad_w, pad_h

def load_and_preprocess_image(img_path: str, target_size: int = 640, normalize: bool = False) -> Tuple[np.ndarray, Tuple[int, int], float, int, int]:
    """Load and preprocess image for inference
    
    Args:
        img_path: Path to input image
        target_size: Target size for inference (default 640x640)
        normalize: If True, normalize to [0,1]; if False, keep as uint8 [0,255]
    
    Returns:
        (input_tensor, original_size, scale, pad_w, pad_h)
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    # YOLO models expect RGB, but cv2.imread loads as BGR, so convert
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    orig_h, orig_w = img.shape[:2]
    
    padded, scale, pad_w, pad_h = letterbox_image(img, target_size)
    
    if normalize:
        padded = padded.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(padded, axis=0)
    else:
        input_tensor = np.expand_dims(padded, axis=0).astype(np.uint8)
    
    return input_tensor, (orig_h, orig_w), scale, pad_w, pad_h



# preprocess_for_onnx removed



def scale_detections_to_original(detections: List[dict], orig_h: int, orig_w: int, scale: float, pad_w: int, pad_h: int) -> List[dict]:
    """Scale detection coordinates from inference space (640x640) to original image space
    
    Args:
        detections: List of detection dicts with x, y, w, h coordinates (acting as x1, y1, x2, y2)
        orig_h, orig_w: Original image dimensions
        scale: Scale factor used in preprocessing
        pad_w, pad_h: Padding used in preprocessing
    """
    for det in detections:
        # 1. Reverse padding
        x_unpad = det['x'] - pad_w
        y_unpad = det['y'] - pad_h
        w_unpad = det['w'] - pad_w
        h_unpad = det['h'] - pad_h
        
        # 2. Reverse scaling
        x = x_unpad / scale
        y = y_unpad / scale
        w = w_unpad / scale
        h = h_unpad / scale
        
        # 3. Clip to original dimensions
        det['x'] = max(0, min(x, orig_w))
        det['y'] = max(0, min(y, orig_h))
        det['w'] = max(0, min(w, orig_w))
        det['h'] = max(0, min(h, orig_h))
        
    return detections


def format_detection_results(detections: List[dict], show_count: int = 10) -> str:
    """Format detection results for display
    
    Args:
        detections: List of detection dicts
        show_count: Maximum number to display (None for all)
    
    Returns:
        Formatted string
    """
    lines = []
    for i, det in enumerate(detections[:show_count] if show_count else detections):
        lines.append(f"  [{i+1}] {det['cls_name']} - conf={det['conf']:.2f}, x={det['x']:.0f}, y={det['y']:.0f}, w={det['w']:.0f}, h={det['h']:.0f}")
    return '\n'.join(lines)


def print_detection_summary(title: str, image_path: str, model_info: Dict, total_time_ms: float, 
                           conf_threshold: float, num_detections: int, output_path: str):
    """Print formatted detection summary
    
    Args:
        title: Summary title
        image_path: Input image path
        model_info: Dict with model paths/names
        total_time_ms: Total inference time in milliseconds
        conf_threshold: Confidence threshold used
        num_detections: Number of detections found
        output_path: Output image path
    """
    print("\n" + "="*60)
    print(title)
    print("="*60)
    print(f"Image: {image_path}")
    for key, val in model_info.items():
        print(f"Model {key}: {val}")
    print(f"Total Time: {total_time_ms:.2f}ms")
    print(f"Confidence Threshold: {conf_threshold}")
    print(f"Detections: {num_detections}")
    print(f"Output: {output_path}")


@dataclass
class InferenceStats:
    """Runtime statistics for inference"""
    preprocess_time: float
    hailo_inference_time: float
    data_mapping_time: float
    onnx_inference_time: float
    total_time: float
    hailo_output_shape: str
    final_output_shape: str


class HailoPythonInferenceEngine:
    """Encapsulates Hailo-8L backbone + Python head inference"""
    
    def __init__(self, hef_path: str):
        """Initialize the hybrid inference engine"""
        self.hef_path = hef_path
        
        # Initialize Hailo
        self.target = VDevice()
        self.hef = HEF(hef_path)
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        
        # Setup vstreams
        self.input_vstream_params = InputVStreamParams.make(self.network_group)
        self.output_vstream_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        
        # Expected shape mapping for data alignment
        # New model has 6 outputs: 3 output scales x (1 classification + 1 regression)
        # Classification: 80 classes
        # Regression: 4 coordinates
        self.shape_to_name = {
            (1, 80, 80, 80): 'cls_80',
            (1, 40, 40, 80): 'cls_40',
            (1, 20, 20, 80): 'cls_20',
            (1, 80, 80, 4): 'reg_80',
            (1, 40, 40, 4): 'reg_40',
            (1, 20, 20, 4): 'reg_20',
        }

        print(f"✓ Hailo engine initialized: {hef_path}")
        print(f"✓ Using Python head for post-processing.")
    
    @staticmethod
    def preprocess(img_path: str, width: int = 640, height: int = 640, normalize: bool = False) -> Tuple[np.ndarray, Tuple[int, int], float, int, int]:
        """Preprocess image to tensor and return original dimensions + stats"""
        return load_and_preprocess_image(img_path, width, normalize)
    
    @staticmethod
    def load_image(img_path: str) -> np.ndarray:
        """Load and read original image"""
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        return img

    
    def _run_python_head(self, dequantized_results: Dict, conf_threshold: float) -> List[dict]:
        """Run python head logic on dequantized Hailo outputs."""
        
        # 1. Map dequantized results to named tensors
        tensors = {}
        found_shapes = []
        for _, data in dequantized_results.items():
            shape = data.shape
            found_shapes.append(shape)
            if shape in self.shape_to_name:
                name = self.shape_to_name[shape]
                tensors[name] = data
        
        # Check if we have all required tensors
        required_tensors = ['cls_80', 'cls_40', 'cls_20', 'reg_80', 'reg_40', 'reg_20']
        missing = [t for t in required_tensors if t not in tensors]
        if missing:
            # Fallback for old model or error
            if 'reg' in self.shape_to_name and (1, 1, 8400, 4) in [s for s in found_shapes]:
                 print("Warning: Detected old model output structure. Please update HEF.")
            # Verify we have at least what we need
            if len(missing) > 0:
                 print(f"Error: Missing tensors from HEF: {missing}")
                 print(f"Found shapes: {found_shapes}")
                 return []

        # Prepare tensors list
        # We need pairs of (cls, reg) for each scale
        # Scales: 80x80 (stride 8), 40x40 (stride 16), 20x20 (stride 32)
        
        # Transpose to (channels, H, W) or flattened? 
        # The loops below expect specific formats.
        
        # Original code expected: 
        # cls: (80, 6400) -> (classes, anchors)
        # reg: (4, 8400) -> (coords, all_anchors) (concatenated)
        
        # Let's adapt to processing per scale
        
        STRIDES = [8, 16, 32]
        GRID_SIZES = [80, 40, 20]
        LOGIT_THRESHOLD = -math.log(1.0 / conf_threshold - 1.0)
        
        results = []
        coco_classes = DetectionPostProcessor._load_coco_classes()
        
        for scale_idx in range(len(STRIDES)):
            stride = STRIDES[scale_idx]
            grid_dim = GRID_SIZES[scale_idx]
            
            suffix = f"_{grid_dim}"
            cls_key = f"cls{suffix}"
            reg_key = f"reg{suffix}"
            
            cls_data = tensors[cls_key][0] # (H, W, 80)
            reg_data = tensors[reg_key][0] # (H, W, 4)
            
            # Reshape to (H*W, C)
            cls_flat = cls_data.reshape(-1, 80)
            reg_flat = reg_data.reshape(-1, 4)
            
            num_anchors = cls_flat.shape[0]
            
            for i in range(num_anchors):

                # Find max class score
                max_logit = -100.0
                class_id = -1
                

                # Simple loop optimization
                row_logits = cls_flat[i]
                

                for c in range(80):
                    logit = row_logits[c]
                    if logit > max_logit:
                        max_logit = logit
                        class_id = c
                
                if max_logit > LOGIT_THRESHOLD:
                    score = sigmoid(max_logit)
                    

                    row = i // grid_dim
                    col = i % grid_dim
                    
                    l = reg_flat[i, 0]
                    t = reg_flat[i, 1]
                    r = reg_flat[i, 2]
                    b = reg_flat[i, 3]
                    
                    x1 = (col + 0.5 - l) * stride
                    y1 = (row + 0.5 - t) * stride
                    x2 = (col + 0.5 + r) * stride
                    y2 = (row + 0.5 + b) * stride
                    
                    results.append({
                        'x': round(x1, 2),
                        'y': round(y1, 2),
                        'w': round(x2, 2),
                        'h': round(y2, 2),
                        'conf': round(score, 4),
                        'cls_id': class_id,
                        'cls_name': coco_classes.get(class_id, 'N/A')
                    })
        
        return results
    
    def infer(self, input_data: np.ndarray, verbose: bool = False, save_output: bool = False, conf_threshold: float = 0.5) -> Tuple[List[dict], InferenceStats]:
        """Run hybrid inference pipeline with Python head"""
        stats = InferenceStats(
            preprocess_time=0, hailo_inference_time=0, 
            data_mapping_time=0, onnx_inference_time=0, total_time=0,
            hailo_output_shape="", final_output_shape=""
        )
        
        t_start = time.perf_counter()
        
        if verbose:
            print(f"[INFERENCE] Input shape: {input_data.shape}, dtype: {input_data.dtype}")
        
        with self.network_group.activate() as active_group:
            with InferVStreams(self.network_group, self.input_vstream_params, self.output_vstream_params) as infer_pipeline:
                
                # A. Hailo Backbone Inference
                if verbose: print(f"[STAGE 1] Running Hailo backbone...")
                t_hailo = time.perf_counter()
                hailo_results = infer_pipeline.infer(input_data)
                stats.hailo_inference_time = time.perf_counter() - t_hailo
                stats.hailo_output_shape = str({k: v.shape for k, v in hailo_results.items()})
                if verbose: print(f"  ✓ Hailo inference: {stats.hailo_inference_time*1000:.2f}ms")

                # B. Python Head
                if verbose: print(f"[STAGE 2] Running Python Head...")
                t_post = time.perf_counter()
                
                detections = self._run_python_head(hailo_results, conf_threshold)
                stats.data_mapping_time = time.perf_counter() - t_post
                stats.final_output_shape = f"{len(detections)} detections"
                if verbose: print(f"  ✓ Python head: {stats.data_mapping_time*1000:.2f}ms")

        stats.total_time = time.perf_counter() - t_start
        
        if verbose:
            print(f"[SUMMARY] Pipeline timing:")
            print(f"  Hailo:   {stats.hailo_inference_time*1000:7.2f}ms")
            print(f"  PyHead:  {stats.data_mapping_time*1000:7.2f}ms")
            print(f"  Total:   {stats.total_time*1000:7.2f}ms")
        
        return detections, stats




# ONNX / Hybrid engines removed to reduce dependencies



class DetectionPostProcessor:
    """Postprocess detections and draw bboxes using YOLO COCO classes"""
    
    COCO_CLASSES = None
    
    @classmethod
    def _load_coco_classes(cls):
        """Return COCO class names (hardcoded to avoid dependencies)"""
        if cls.COCO_CLASSES is not None:
            return cls.COCO_CLASSES
        
        # Standard COCO 80 classes
        cls.COCO_CLASSES = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
            20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
            50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
            60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
            70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
        }
        
        return cls.COCO_CLASSES
    
    @staticmethod
    def postprocess(detections: np.ndarray, conf_threshold: float = 0.5) -> List[dict]:
        """Parse detections and filter by confidence
        
        Args:
            detections: Shape (num_detections, 6) - [x, y, w, h, conf, cls]
            conf_threshold: Confidence threshold
            
        Returns:
            List of dicts with keys: [x, y, w, h, conf, cls_id, cls_name]
        """
        classes = DetectionPostProcessor._load_coco_classes()
        results = []
        
        for det in detections:
            conf = det[4]
            if conf >= conf_threshold:
                cls_id = int(det[5])
                # Handle both dict (YOLO) and list formats
                cls_name = classes.get(cls_id) if isinstance(classes, dict) else classes[cls_id]
                results.append({
                    'x': float(det[0]),
                    'y': float(det[1]),
                    'w': float(det[2]),
                    'h': float(det[3]),
                    'conf': float(conf),
                    'cls_id': cls_id,
                    'cls_name': cls_name
                })
        return results
    
    @staticmethod
    def draw_bboxes(image: np.ndarray, detections: list, thickness: int = 2) -> np.ndarray:
        """Draw bounding boxes on image"""
        img = image.copy()
        h, w = img.shape[:2]
        
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, det in enumerate(detections):
            x1 = int(max(0, det['x']))
            y1 = int(max(0, det['y']))
            x2 = int(min(w, det['w']))
            y2 = int(min(h, det['h']))
            
            color = colors[i % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            label = f"{det['cls_name']} {det['conf']:.2f}"
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return img
