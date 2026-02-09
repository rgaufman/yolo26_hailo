import argparse
import sys
from pathlib import Path
from .config import ExportConfig
from .core.pipeline import ExportPipeline

def main():
    parser = argparse.ArgumentParser(description="YOLO26 to Hailo Export Tool")
    parser.add_argument("--variant", type=str, required=True, help="YOLO variant (e.g., yolo26n)")
    parser.add_argument("--target", type=str, default="hailo8l", help="Target architecture")
    parser.add_argument("--onnx", type=str, required=True, help="Path to input ONNX model")
    parser.add_argument("--calib_dir", type=str, required=True, help="Path to calibration images directory")
    parser.add_argument("--alls", type=str, help="Optional path to .alls script")
    parser.add_argument("--tag", type=str, help="Optional tag for the experiment run")
    
    args = parser.parse_args()
    
    try:
        config = ExportConfig(
            variant=args.variant,
            target=args.target,
            onnx_path=Path(args.onnx).resolve(),
            calib_dir=Path(args.calib_dir).resolve(),
            alls_path=Path(args.alls).resolve() if args.alls else None,
            tag=args.tag
        )
        
        pipeline = ExportPipeline(config)
        pipeline.run()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
