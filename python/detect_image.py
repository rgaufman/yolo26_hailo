"""Single image detection with bbox visualization (Hybrid Hailo + ONNX)"""

import argparse
import time
from pathlib import Path
from common import (
    HailoPythonInferenceEngine, DetectionPostProcessor,
    scale_detections_to_original, format_detection_results,
    print_detection_summary
)


def detect_and_visualize(args):
    """Run detection on single image using hybrid Hailo + Python head pipeline"""
    print(f"[Loading image: {args.image}]")
    engine = HailoPythonInferenceEngine(args.hef)
    
    # Load original image for visualization
    orig_image = HailoPythonInferenceEngine.load_image(args.image)
    orig_h, orig_w = orig_image.shape[:2]
    print(f"✓ Original image size: {orig_w}x{orig_h}")
    
    # Preprocess for inference
    print("[Preprocessing...]")
    input_data, orig_size = HailoPythonInferenceEngine.preprocess(args.image, normalize=args.normalize)
    print(f"✓ Preprocessed to: {input_data.shape}, dtype={input_data.dtype}")
    
    # Run inference
    print("[Running inference...]")
    t_start = time.perf_counter()
    results, stats = engine.infer(input_data, verbose=args.verbose, save_output=args.save_output, conf_threshold=args.conf_threshold)
    total_time = time.perf_counter() - t_start
    
    print(f"✓ Inference completed in {total_time*1000:.2f}ms")
    print(f"  - Hailo: {stats.hailo_inference_time*1000:.2f}ms")
    print(f"  - Python Head: {stats.data_mapping_time*1000:.2f}ms")
    print(f"✓ Found {len(results)} detections above threshold {args.conf_threshold}")
    print(format_detection_results(results))
    
    # Scale detections to original image space
    results = scale_detections_to_original(results, orig_h)
    
    # Draw bboxes on original image
    print("[Drawing bounding boxes...]")
    output_image = DetectionPostProcessor.draw_bboxes(orig_image, results, thickness=2)
    
    # Save output image
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import cv2
    cv2.imwrite(str(output_path), output_image)
    print(f"✓ Output image saved to: {output_path}")
    
    # Print summary
    print_detection_summary(
        title="DETECTION SUMMARY (Hybrid Hailo + Python Head)",
        image_path=args.image,
        model_info={"HEF": args.hef},
        total_time_ms=total_time*1000,
        conf_threshold=args.conf_threshold,
        num_detections=len(results),
        output_path=str(output_path)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single Image Detection with Hailo-8L + Python Head")
    parser.add_argument("image", type=str, help="Input image path")
    parser.add_argument("--hef", type=str, default="../models/yolo26n.hef", help="Path to HEF model")
    parser.add_argument("--output", type=str, default="output_detected.jpg", help="Output image path")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--normalize", action="store_true", help="Normalize input to [0,1] (default: uint8 [0,255] for HEF)")
    parser.add_argument("--save-output", action="store_true", help="Save intermediate outputs as .npy files")
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        exit(1)
    
    detect_and_visualize(args)
