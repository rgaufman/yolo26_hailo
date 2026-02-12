import numpy as np
import argparse
import time
import json
from pathlib import Path
from dataclasses import asdict
from common import HailoPythonInferenceEngine, InferenceStats


def run_inference(args):
    """Run inference with statistics collection"""
    engine = HailoPythonInferenceEngine(args.hef)
    
    stats_list = []
    
    for i in range(args.iterations):
        
        # Preprocess
        t_pre = time.perf_counter()
        if args.image:
            input_data, _, _, _, _ = HailoPythonInferenceEngine.preprocess(args.image, normalize=args.normalize)
        else:
            # Generate random data: normalized float32 [0,1] or uint8 [0,255]
            if args.normalize:
                input_data = np.random.rand(1, 640, 640, 3).astype(np.float32)
            else:
                input_data = np.random.randint(0, 256, (1, 640, 640, 3), dtype=np.uint8)
        preprocess_time = time.perf_counter() - t_pre
        
        # Infer
        output, stats = engine.infer(input_data, verbose=args.verbose)
        stats.preprocess_time = preprocess_time
        
        stats_list.append(stats)
    
    # Print summary
    print("\n" + "="*60)
    print("INFERENCE SUMMARY")
    print("="*60)
    
    total_times = np.array([s.total_time for s in stats_list])
    hailo_times = np.array([s.hailo_inference_time for s in stats_list])
    python_head_times = np.array([s.data_mapping_time for s in stats_list])

    avg_total = total_times.mean()
    std_total = total_times.std()
    avg_hailo = hailo_times.mean()
    std_hailo = hailo_times.std()
    avg_python_head = python_head_times.mean()
    std_python_head = python_head_times.std()

    fps_list = 1.0 / total_times
    fps_mean = fps_list.mean()
    fps_std = fps_list.std()

    print(f"Total Iterations: {args.iterations}")
    print(f"Avg Total Time: {avg_total*1000:.2f}ms ± {std_total*1000:.2f}ms")
    print(f"  - Hailo Backbone: {avg_hailo*1000:.2f}ms ± {std_hailo*1000:.2f}ms ({avg_hailo/avg_total*100:.1f}% of mean)")
    print(f"  - Python Head: {avg_python_head*1000:.2f}ms ± {std_python_head*1000:.2f}ms ({avg_python_head/avg_total*100:.1f}% of mean)")
    print(f"Throughput: {fps_mean:.2f} FPS ± {fps_std:.2f}")
    
    # Save statistics
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump([asdict(s) for s in stats_list], f, indent=2)
        print(f"\n✓ Statistics saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Hailo-8L + Python Head Inference Engine")
    parser.add_argument("--hef", type=str, default="../models/yolo26n.hef", help="Path to HEF model")
    parser.add_argument("--image", type=str, help="Input image path (random if not provided)")
    parser.add_argument("--iterations", type=int, default=10, help="Number of inference iterations")
    parser.add_argument("--output", type=str, help="JSON file to save statistics")
    parser.add_argument("--normalize", action="store_true", help="Normalize input to [0,1] (default: uint8 [0,255] for HEF)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    run_inference(args)