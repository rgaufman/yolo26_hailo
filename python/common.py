"""Common utilities for Hailo-8L + ONNX hybrid inference"""

import numpy as np
import cv2
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List
from hailo_platform import VDevice, HEF, ConfigureParams, InputVStreamParams, OutputVStreamParams, InferVStreams, HailoStreamInterface
import math

# ultralytics removed


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# ============================================================================
# Common Image Operations
# ============================================================================

def load_and_preprocess_image(img_path: str, target_size: int = 640, normalize: bool = False) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Load and preprocess image for inference
    
    Args:
        img_path: Path to input image
        target_size: Target size for inference (default 640x640)
        normalize: If True, normalize to [0,1]; if False, keep as uint8 [0,255]
    
    Returns:
        (input_tensor, original_size) where original_size is (H, W)
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    # YOLO models expect RGB, but cv2.imread loads as BGR, so convert
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    orig_h, orig_w = img.shape[:2]
    img = cv2.resize(img, (target_size, target_size))
    
    if normalize:
        img = img.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(img, axis=0)
    else:
        input_tensor = np.expand_dims(img, axis=0).astype(np.uint8)
    
    return input_tensor, (orig_h, orig_w)


def preprocess_for_onnx(img_path: str, target_size: int = 640) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Preprocess image for complete ONNX model (float32 BCHW format)
    
    Args:
        img_path: Path to input image
        target_size: Target size for inference (default 640x640)
    
    Returns:
        (input_tensor, original_size) where input_tensor is (1, 3, 640, 640) BCHW float32
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    # YOLO models expect RGB, but cv2.imread loads as BGR, so convert
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    orig_h, orig_w = img.shape[:2]
    img = cv2.resize(img, (target_size, target_size))
    
    # ONNX expects float32 [0,1] in BCHW format
    input_data = img.astype(np.float32) / 255.0
    input_data = np.transpose(np.expand_dims(input_data, axis=0), (0, 3, 1, 2))
    
    return input_data, (orig_h, orig_w)


def scale_detections_to_original(detections: List[dict], orig_h: int) -> List[dict]:
    """Scale detection coordinates from inference space (640x640) to original image space
    
    Args:
        detections: List of detection dicts with x, y, w, h coordinates
        orig_h: Original image height (width assumed 640)
    
    Returns:
        List of detections with scaled coordinates
    """
    scale_y = orig_h / 640.0
    for det in detections:
        det['y'] = det['y'] * scale_y
        det['h'] = det['h'] * scale_y
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
        self.output_vstream_params = OutputVStreamParams.make(self.network_group)
        
        # Expected shape mapping for data alignment
        self.shape_to_name = {
            (1, 80, 80, 80): 'cls_80',
            (1, 40, 40, 80): 'cls_40',
            (1, 20, 20, 80): 'cls_20',
            (1, 1, 8400, 4): 'reg'
        }

        # Extract quantization info from HEF
        self.quant_info = self._extract_quant_info()
        
        print(f"✓ Hailo engine initialized: {hef_path}")
        print(f"✓ Using Python head for post-processing.")
        print(f"✓ Quantization info: {self.quant_info}")
    
    @staticmethod
    def preprocess(img_path: str, width: int = 640, height: int = 640, normalize: bool = False) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Preprocess image to tensor and return original dimensions
        
        Args:
            img_path: Path to input image
            width: Target width (default 640)
            height: Target height (default 640)
            normalize: If True, normalize to [0,1] by dividing by 255 (for ONNX-only inference).
                      If False (default), keep as uint8 [0,255] to match HEF quantization.
                      The HEF model expects uint8 and has quantization parameters that handle normalization.
        
        Returns:
            (input_tensor, original_size) where input_tensor is (1, 640, 640, 3) and original_size is (H, W)
        """
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # YOLO models expect RGB, but cv2.imread loads as BGR, so convert
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, (width, height))
        
        if normalize:
            # For ONNX-only: normalize to [0, 1] by (x - 0) / 255 = x / 255
            img = img.astype(np.float32) / 255.0
            input_tensor = np.expand_dims(img, axis=0)
        else:
            # Default for hybrid inference: keep as uint8 [0, 255]
            # HEF quantization parameters handle the normalization
            input_tensor = np.expand_dims(img, axis=0).astype(np.uint8)
        
        return input_tensor, (orig_h, orig_w)
    
    @staticmethod
    def load_image(img_path: str) -> np.ndarray:
        """Load and read original image"""
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        return img

    def _extract_quant_info(self) -> Dict[str, Tuple[float, float]]:
        """Extract quantization scale and zero_point from HEF
        
        Returns:
            Dict mapping output names to (scale, zero_point) tuples
        """
        quant_info = {}
        
        # Get output vstream info which contains quantization parameters
        for output_name in self.hef.get_output_vstream_infos():
            quant = output_name.quant_info
            if quant is not None:
                # scale converts from int8/uint8 to float: float_val = (int_val - zero_point) * scale
                scale = quant.qp_scale
                zero_point = quant.qp_zp
                quant_info[output_name.name] = (scale, zero_point)
            else:
                # No quantization, use identity
                quant_info[output_name.name] = (1.0, 0.0)
        
        return quant_info
    
    def _dequantize_hailo_outputs(self, hailo_results: Dict, verbose: bool = False) -> Dict:
        """Dequantize Hailo outputs using quantization parameters from the HEF."""
        dequantized_results = {}
        if verbose:
            print("  [DEBUG] Dequantizing Hailo outputs...")

        for output_name, raw_data in hailo_results.items():
            data = raw_data.copy().astype(np.float32)
            
            if output_name in self.quant_info:
                scale, zero_point = self.quant_info[output_name]
                data = (data - zero_point) * scale
                if verbose:
                    print(f"    - {output_name}: Dequantized with scale={scale:.6f}, zp={zero_point}")
            else:
                if verbose:
                    print(f"    - {output_name}: No quantization info found, skipping.")
            
            dequantized_results[output_name] = data
            
        return dequantized_results

    
    def _run_python_head(self, dequantized_results: Dict, conf_threshold: float) -> List[dict]:
        """Run python head logic on dequantized Hailo outputs."""
        
        # 1. Map dequantized results to named tensors from head.py
        tensors = {}
        for _, data in dequantized_results.items():
            if data.shape in self.shape_to_name:
                name = self.shape_to_name[data.shape]
                tensors[name] = data

        cls_80 = tensors['cls_80'][0]
        cls_40 = tensors['cls_40'][0]
        cls_20 = tensors['cls_20'][0]
        reg_tensor = tensors['reg'][0, 0]

        cls_80 = cls_80.reshape(80*80, 80).transpose(1, 0)
        cls_40 = cls_40.reshape(40*40, 80).transpose(1, 0)
        cls_20 = cls_20.reshape(20*20, 80).transpose(1, 0)
        reg_tensor = reg_tensor.transpose(1, 0)
        
        cls_tensors = [cls_80, cls_40, cls_20]
        
        # 2. Configuration from head.py
        STRIDES = [8, 16, 32]
        GRID_SIZES = [80, 40, 20]
        LOGIT_THRESHOLD = -math.log(1.0 / conf_threshold - 1.0)
        
        # 3. decode_and_filter logic from head.py
        results = []
        global_offset = 0
        coco_classes = DetectionPostProcessor._load_coco_classes()

        for scale_idx in range(len(STRIDES)):
            stride = STRIDES[scale_idx]
            grid_dim = GRID_SIZES[scale_idx]
            num_anchors = grid_dim * grid_dim
            cls_data = cls_tensors[scale_idx]

            for i in range(num_anchors):
                global_idx = global_offset + i
                
                max_logit = -100.0
                class_id = -1
                for c in range(80):
                    logit = cls_data[c, i]
                    if logit > max_logit:
                        max_logit = logit
                        class_id = c
                
                if max_logit > LOGIT_THRESHOLD:
                    score = sigmoid(max_logit)
                    
                    l = reg_tensor[0, global_idx]
                    t = reg_tensor[1, global_idx]
                    r = reg_tensor[2, global_idx]
                    b = reg_tensor[3, global_idx]
                    
                    x1 = l
                    y1 = t
                    x2 = r
                    y2 = b
                    
                    results.append({
                        'x': round(x1, 2),
                        'y': round(y1, 2),
                        'w': round(x2, 2),
                        'h': round(y2, 2),
                        'conf': round(score, 4),
                        'cls_id': class_id,
                        'cls_name': coco_classes.get(class_id, 'N/A')
                    })

            global_offset += num_anchors

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

                # B. Dequantization and Python Head
                if verbose: print(f"[STAGE 2] Running Dequantization and Python Head...")
                t_post = time.perf_counter()
                dequantized_results = self._dequantize_hailo_outputs(hailo_results, verbose=verbose)
                
                detections = self._run_python_head(dequantized_results, conf_threshold)
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



class HailoOnnxInferenceEngine:
    """Encapsulates Hailo-8L backbone + ONNX Runtime head inference"""
    
    def __init__(self, hef_path: str, onnx_path: str):
        """Initialize the hybrid inference engine"""
        self.hef_path = hef_path
        self.onnx_path = onnx_path

        import onnxruntime as ort
        
        # Initialize ONNX Runtime
        self.ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.ort_inputs = [node.name for node in self.ort_session.get_inputs()]
        
        # Initialize Hailo
        self.target = VDevice()
        self.hef = HEF(hef_path)
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        
        # Setup vstreams
        self.input_vstream_params = InputVStreamParams.make(self.network_group)
        self.output_vstream_params = OutputVStreamParams.make(self.network_group)
        
        # Expected shape mapping for data alignment
        self.expected_shapes = {
            '/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0': (1, 80, 80, 80),
            '/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0': (1, 40, 40, 80),
            '/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0': (1, 20, 20, 80),
            '/model.23/Mul_2_output_0': (1, 1, 8400, 4)
        }

        # Extract quantization info from HEF
        self.quant_info = self._extract_quant_info()
        
        print(f"✓ Hailo engine initialized: {hef_path}")
        print(f"✓ ONNX engine initialized: {onnx_path}")
        print(f"✓ Quantization info: {self.quant_info}")
    
    @staticmethod
    def preprocess(img_path: str, width: int = 640, height: int = 640, normalize: bool = False) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Preprocess image to tensor and return original dimensions
        
        Args:
            img_path: Path to input image
            width: Target width (default 640)
            height: Target height (default 640)
            normalize: If True, normalize to [0,1] by dividing by 255 (for ONNX-only inference).
                      If False (default), keep as uint8 [0,255] to match HEF quantization.
                      The HEF model expects uint8 and has quantization parameters that handle normalization.
        
        Returns:
            (input_tensor, original_size) where input_tensor is (1, 640, 640, 3) and original_size is (H, W)
        """
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # YOLO models expect RGB, but cv2.imread loads as BGR, so convert
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, (width, height))
        
        if normalize:
            # For ONNX-only: normalize to [0, 1] by (x - 0) / 255 = x / 255
            img = img.astype(np.float32) / 255.0
            input_tensor = np.expand_dims(img, axis=0)
        else:
            # Default for hybrid inference: keep as uint8 [0, 255]
            # HEF quantization parameters handle the normalization
            input_tensor = np.expand_dims(img, axis=0).astype(np.uint8)
        
        return input_tensor, (orig_h, orig_w)
    
    @staticmethod
    def load_image(img_path: str) -> np.ndarray:
        """Load and read original image"""
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        return img

    def _extract_quant_info(self) -> Dict[str, Tuple[float, float]]:
        """Extract quantization scale and zero_point from HEF
        
        Returns:
            Dict mapping output names to (scale, zero_point) tuples
        """
        quant_info = {}
        
        # Get output vstream info which contains quantization parameters
        for output_name in self.hef.get_output_vstream_infos():
            quant = output_name.quant_info
            if quant is not None:
                # scale converts from int8/uint8 to float: float_val = (int_val - zero_point) * scale
                scale = quant.qp_scale
                zero_point = quant.qp_zp
                quant_info[output_name.name] = (scale, zero_point)
            else:
                # No quantization, use identity
                quant_info[output_name.name] = (1.0, 0.0)
        
        return quant_info
    
    def _dequantize_hailo_outputs(self, hailo_results: Dict, verbose: bool = False) -> Dict:
        """Dequantize Hailo outputs using quantization parameters from the HEF."""
        dequantized_results = {}
        if verbose:
            print("  [DEBUG] Dequantizing Hailo outputs...")

        for output_name, raw_data in hailo_results.items():
            data = raw_data.copy().astype(np.float32)
            
            if output_name in self.quant_info:
                scale, zero_point = self.quant_info[output_name]
                data = (data - zero_point) * scale
                if verbose:
                    print(f"    - {output_name}: Dequantized with scale={scale:.6f}, zp={zero_point}")
            else:
                if verbose:
                    print(f"    - {output_name}: No quantization info found, skipping.")
            
            dequantized_results[output_name] = data
            
        return dequantized_results

    def _map_hailo_to_onnx_inputs(self, dequantized_results: Dict, verbose: bool = False) -> Dict:
        """Map dequantized Hailo outputs to ONNX inputs with proper shape transformation."""
        shape_to_output = {}
        for output_name, data in dequantized_results.items():
            if data.shape in shape_to_output:
                raise ValueError(f"Duplicate output shape {data.shape}: {output_name} and {shape_to_output[data.shape][1]}")
            shape_to_output[data.shape] = (data, output_name)

        if verbose:
            print("  [DEBUG] Mapping dequantized outputs to ONNX inputs by shape:")
            for shape, (_, name) in shape_to_output.items():
                print(f"    {shape} -> {name}")

        ort_feed = {}
        for ort_input_name in self.ort_inputs:
            expected_shape = self.expected_shapes[ort_input_name]

            if expected_shape not in shape_to_output:
                raise ValueError(
                    f"Missing expected output shape {expected_shape} for ONNX input '{ort_input_name}'\n"
                    f"Available shapes: {list(shape_to_output.keys())}"
                )

            data, hailo_output_name = shape_to_output[expected_shape]
            
            if verbose:
                print(f"  [DEBUG] Processing ONNX input: {ort_input_name}")
                print(f"    Source: {hailo_output_name} with shape {data.shape}")

            # Shape transformation based on tensor role
            original_shape = data.shape
            
            # Detection head: (1, 1, 8400, 4) → (1, 4, 8400)
            if data.shape == (1, 1, 8400, 4):
                data = data.reshape(1, 8400, 4)
                data = np.transpose(data, (0, 2, 1))
                if verbose:
                    print(f"    Detection head reshape: {original_shape} -> (1,8400,4) -> {data.shape}")
            
            # Feature maps: NHWC → NCHW (output channels are 80)
            elif len(data.shape) == 4 and data.shape[-1] == 80:
                data = np.transpose(data, (0, 3, 1, 2))
                if verbose:
                    print(f"    Feature map NHWC->NCHW: {original_shape} -> {data.shape}")
            
            if verbose:
                print(f"    Final shape for ONNX: {data.shape}")
            
            ort_feed[ort_input_name] = data

        if verbose:
            print(f"  [DEBUG] ONNX feed dict prepared with {len(ort_feed)} inputs.")
            
        return ort_feed

    def _map_hailo_to_onnx(self, hailo_results: Dict, verbose: bool = False) -> Dict:
        """Map Hailo outputs to ONNX inputs with proper dequantization and shape transformation
        
        Args:
            hailo_results: Dict of raw Hailo output tensors
            verbose: Print detailed shape transformation info
            
        Returns:
            Dict mapping ONNX input names to properly formatted tensors
        """
        # 1. Dequantize Hailo outputs
        dequantized_results = self._dequantize_hailo_outputs(hailo_results, verbose=verbose)
        
        # 2. Map dequantized outputs to ONNX inputs
        ort_feed = self._map_hailo_to_onnx_inputs(dequantized_results, verbose=verbose)
        
        return ort_feed
    
    def infer(self, input_data: np.ndarray, verbose: bool = False, save_output: bool = False) -> Tuple[np.ndarray, InferenceStats]:
        """Run hybrid inference pipeline with optional detailed shape tracing
        
        Args:
            input_data: Input tensor shape (1, 640, 640, 3) as uint8
            verbose: Print detailed shape transformation and timing info
            save_output: If True, saves intermediate outputs to .npy files for debugging
            
        Returns:
            Tuple of (final_output, InferenceStats)
        """
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
                if verbose:
                    print(f"[STAGE 1] Running Hailo backbone...")
                t_hailo = time.perf_counter()
                hailo_results = infer_pipeline.infer(input_data)
                stats.hailo_inference_time = time.perf_counter() - t_hailo
                
                hailo_output_info = {k: v.shape for k, v in hailo_results.items()}
                stats.hailo_output_shape = str(hailo_output_info)

                if save_output:
                    for name, data in hailo_results.items():
                        np.save(f"{name.replace('/', '_')}_uint8.npy", data)
                
                if verbose:
                    print(f"  ✓ Hailo inference: {stats.hailo_inference_time*1000:.2f}ms")
                    print(f"  Outputs ({len(hailo_results)}):") 
                    for name, data in hailo_results.items():
                        print(f"    - {name}: shape={data.shape}, dtype={data.dtype}")
                
                # B. Hailo→ONNX Data Mapping and Dequantization
                if verbose:
                    print(f"[STAGE 2] Mapping Hailo outputs to ONNX inputs...")
                t_map = time.perf_counter()
                # 1. Dequantize Hailo outputs
                dequantized_results = self._dequantize_hailo_outputs(hailo_results, verbose=verbose)

                if save_output:
                    for name, data in dequantized_results.items():
                        np.save(f"{name.replace('/', '_')}_float32.npy", data)
                
                # 2. Map dequantized outputs to ONNX inputs
                ort_feed = self._map_hailo_to_onnx_inputs(dequantized_results, verbose=verbose)
                stats.data_mapping_time = time.perf_counter() - t_map
                
                if verbose:
                    print(f"  ✓ Data mapping: {stats.data_mapping_time*1000:.2f}ms")
                    print(f"  ONNX feed dict ({len(ort_feed)} inputs):")
                    for name, data in ort_feed.items():
                        print(f"    - {name}: shape={data.shape}, dtype={data.dtype}, range=[{data.min():.2f}, {data.max():.2f}]")
                
                # C. ONNX Head Inference
                if verbose:
                    print(f"[STAGE 3] Running ONNX head...")
                t_onnx = time.perf_counter()
                final_outputs = self.ort_session.run(None, ort_feed)
                stats.onnx_inference_time = time.perf_counter() - t_onnx
                stats.final_output_shape = str(final_outputs[0].shape)

                if save_output:
                    np.save("final_output.npy", final_outputs[0])
                
                if verbose:
                    print(f"  ✓ ONNX inference: {stats.onnx_inference_time*1000:.2f}ms")
                    print(f"  Output: shape={final_outputs[0].shape}, dtype={final_outputs[0].dtype}")
        
        stats.total_time = time.perf_counter() - t_start
        
        if verbose:
            print(f"[SUMMARY] Pipeline timing:")
            print(f"  Hailo:   {stats.hailo_inference_time*1000:7.2f}ms")
            print(f"  Mapping: {stats.data_mapping_time*1000:7.2f}ms")
            print(f"  ONNX:    {stats.onnx_inference_time*1000:7.2f}ms")
            print(f"  Total:   {stats.total_time*1000:7.2f}ms")
        
        return final_outputs[0], stats

class OnnxInferenceEngine:
    """ONNX-only inference engine for complete YOLO models"""
    
    def __init__(self, onnx_path: str):
        """Initialize ONNX inference engine
        
        Args:
            onnx_path: Path to complete YOLO ONNX model
        """
        self.onnx_path = onnx_path

        import onnxruntime as ort
        self.ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.ort_inputs = [node.name for node in self.ort_session.get_inputs()]
        
        print(f"✓ ONNX engine initialized: {onnx_path}")
        print(f"✓ ONNX inputs: {self.ort_inputs}")
        print(f"✓ Input shape: {self.ort_session.get_inputs()[0].shape}")
    
    def infer(self, input_data: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, float]:
        """Run inference on preprocessed input
        
        Args:
            input_data: Input tensor shape (1, 3, 640, 640) as float32 [0,1]
            verbose: Print detailed timing info
            
        Returns:
            Tuple of (detection_output, inference_time_seconds)
        """
        if verbose:
            print(f"[INFERENCE] Input shape: {input_data.shape}, dtype: {input_data.dtype}")
            print(f"  Range: [{input_data.min():.4f}, {input_data.max():.4f}]")
        
        t_start = time.perf_counter()
        outputs = self.ort_session.run(None, {self.ort_inputs[0]: input_data})
        elapsed = time.perf_counter() - t_start
        
        if verbose:
            print(f"✓ ONNX inference: {elapsed*1000:.2f}ms")
            print(f"  Output shape: {outputs[0].shape}, dtype: {outputs[0].dtype}")
        
        return outputs[0], elapsed


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
