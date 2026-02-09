from ultralytics import YOLO
import sys
import os
import shutil
import argparse
from pathlib import Path

# Add export directory to path to import config
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'export'))

try:
    import export_config
except ImportError as e:
    print(f"Warning: Could not import export_config.py: {e}")
    # Fallback or exit? Let's proceed with defaults if possible, 
    # but preferably we want the config to know the weights name.
    export_config = None

def download_and_export(variant="yolo26n", output_dir="models"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Downloading {variant} and exporting to ONNX in {output_dir}...")
    
    weights_name = f"{variant}.pt"
    if export_config:
        weights_name = export_config.VARIANTS.get(variant, {}).get('weights_name', weights_name)
    
    print(f"Using weights file: {weights_name}")
    
    try:
        # Load model (downloads automatically if not found locally)
        model = YOLO(weights_name)
        
        # Export to ONNX
        # export() returns the filename of the exported model
        print("Exporting to ONNX (opset=11, simplify=True)...")
        exported_path = model.export(format="onnx", opset=11, simplify=True)
        
        # Determine destination
        onnx_filename = f"{variant}.onnx"
        dest_path = os.path.join(output_dir, onnx_filename)
        
        # Move to destination
        if exported_path and os.path.exists(exported_path):
             # If export output is different from desired destination, move/rename it
            if os.path.abspath(exported_path) != os.path.abspath(dest_path):
                shutil.move(exported_path, dest_path)
            
            print(f"Success! Model saved to: {dest_path}")
            
            # Clean up the .pt file if it's in the root and we want it in models/ or if the user doesn't want it cluttering 
            # (But usually keeping weights is good). Let's move it to models/ if it's not there.
            local_weights = Path(weights_name)
            if local_weights.exists() and local_weights.resolve().parent != Path(output_dir).resolve():
                shutil.move(str(local_weights), os.path.join(output_dir, weights_name))
                print(f"Moved weights to: {os.path.join(output_dir, weights_name)}")
        else:
            print("Error: Export failed or file not found.")
            
    except Exception as e:
        print(f"Error during download/export: {e}")
        print("Ensure 'ultralytics' is installed: pip install ultralytics")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and export YOLO models")
    parser.add_argument("--variant", type=str, default="yolo26n", help="YOLO variant (e.g. yolo26n, yolo26s)")
    parser.add_argument("--output", type=str, default="models", help="Output directory")
    
    args = parser.parse_args()
    download_and_export(args.variant, args.output)
