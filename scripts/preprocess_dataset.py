import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

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

def preprocess_dataset(src_dir, dst_dir, target_size=640):
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)
    
    images = list(src_path.glob("*.jpg")) + list(src_path.glob("*.jpeg")) + list(src_path.glob("*.png"))
    
    print(f"Found {len(images)} images in {src_dir}")
    print(f"Processing to {dst_dir} (Target size: {target_size}x{target_size})")
    
    for img_path in tqdm(images):
        # 1. Read Image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
            
        # 2. Convert BGR to RGB (Critical for YOLO models)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 3. Letterbox Resize
        padded, _, _, _ = letterbox_image(img, target_size=target_size)
        
        # 4. Save as .npy
        # Try to use same basename
        save_name = img_path.stem + '.npy'
        save_path = dst_path / save_name
        
        np.save(save_path, padded)

def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for Hailo calibration/quantization")
    parser.add_argument("src", help="Source directory containing images")
    parser.add_argument("dst", help="Destination directory for .npy files")
    parser.add_argument("--size", type=int, default=640, help="Target size (default 640)")
    
    args = parser.parse_args()
    
    preprocess_dataset(args.src, args.dst, args.size)

if __name__ == "__main__":
    main()
