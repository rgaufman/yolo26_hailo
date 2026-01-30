from ultralytics import YOLO
import sys
import os
import shutil

def download_and_export(output_dir="models"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Downloading YOLO26n and exporting to ONNX in {output_dir}...")
    
    # Load model (downloads automatically if not found locally)
    # We want it to be clean, so let's run this where we call it, but move result.
    model = YOLO("yolo26n.pt")
    
    # Export to ONNX
    # export() returns the filename of the exported model
    exported_path = model.export(format="onnx", opset=11)
    
    # Move to destination
    dest_path = os.path.join(output_dir, "yolo26n.onnx")
    
    # If exported path is not the dest path, move it
    # Note: ultralytics export usually places it next to the .pt file
    if os.path.abspath(exported_path) != os.path.abspath(dest_path):
        shutil.move(exported_path, dest_path)
        
    print(f"Success! Model saved to: {dest_path}")
    
    # Clean up the .pt file if it's in the root and we want it in models/
    if os.path.exists("yolo26n.pt") and os.path.abspath("models/yolo26n.pt") != os.path.abspath("yolo26n.pt"):
        shutil.move("yolo26n.pt", os.path.join(output_dir, "yolo26n.pt"))

if __name__ == "__main__":
    output_dir = "models"
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    download_and_export(output_dir)
