import numpy as np
import tensorflow as tf
from hailo_sdk_client import ClientRunner
import sys
import os

# usage: python 2_quantize_har.py <input_har_path> <output_har_path> <calib_images_dir>
if len(sys.argv) != 4:
    print("Usage: python 2_quantize_har.py <input_har_path> <output_har_path> <calib_images_dir>")
    sys.exit(1)

input_har_path = sys.argv[1]
output_har_path = sys.argv[2]
calib_images_dir = sys.argv[3]
alls_script_path = 'yolo26n.alls' # Assumed to be in the same directory or adjust accordingly

if not os.path.exists(input_har_path):
    print(f"Error: Input HAR file {input_har_path} not found.")
    sys.exit(1)

if not os.path.exists(calib_images_dir):
    print(f"Error: Calibration images directory {calib_images_dir} not found.")
    sys.exit(1)

# 1. Initialize and load parsed HAR
runner = ClientRunner(hw_arch='hailo8l')
runner.load_har(input_har_path)

if os.path.exists(alls_script_path):
    print(f"Loading model script: {alls_script_path}")
    runner.load_model_script(alls_script_path)
else:
    print(f"Warning: Model script {alls_script_path} not found. Proceeding without it.")

# 2. Create TensorFlow Dataset from images
def load_and_preprocess_image(path):
    """Load and preprocess an image for Hailo calibration."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [640, 640])
    # Grounding: If you have a 'normalization' command in your .alls file,
    # keep this as [0, 255]. If not, normalize to [0, 1].
    # Most YOLO recipes in Hailo use uint8 [0, 255] inputs.
    return tf.cast(image, tf.float32), {}

# Get image paths
image_paths = []
for ext in ['*.jpg', '*.jpeg', '*.png']:
    image_paths.extend(tf.io.gfile.glob(os.path.join(calib_images_dir, '**', ext)))

# Grounding: Cap the calibration set to exactly 1024 images if available, else use all
limit = 1024
if len(image_paths) > limit:
    image_paths = image_paths[:limit]

print(f"Using {len(image_paths)} images for calibration from {calib_images_dir}.")

if len(image_paths) == 0:
    print("Error: No images found in calibration directory.")
    sys.exit(1)

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices(image_paths)
dataset = dataset.take(limit) 
dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

# 3. Optimization Call
# According to the DFC Guide, for tf.data.Dataset, we do NOT batch manually here.
# The DFC internal optimizer handles its own batching/shuffling.
runner.optimize(dataset, data_type="dataset")

# 4. Save the optimized HAR
runner.save_har(output_har_path)
print(f"Saved optimized HAR to: {output_har_path}")
