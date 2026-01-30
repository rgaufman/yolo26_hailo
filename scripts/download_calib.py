import fiftyone as fo
import fiftyone.zoo as foz
import os
import argparse
from fiftyone import ViewField as F

def download_hailo_calibration_set(output_dir="data/calib_images", num_samples=1024):
    """
    Downloads a subset of COCO train2017 for quantization calibration.
    """
    # Ensure output directory exists (or fiftyone will create it)
    # We want to path relative to proper location if not absolute
    if not os.path.isabs(output_dir):
        # If running from root, it might be fine, but let's be safe
        pass

    print(f"Downloading and sampling {num_samples} images from COCO training set...")

    # 1. Load the COCO-2017 training split
    # We only download the necessary metadata first
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        label_types=["detections"],
        max_samples=num_samples * 5, # Pull a larger pool to allow for sampling
    )

    # 2. Perform a random shuffle/sample
    # To be scientifically sound for quantization, we want a representative shuffle
    view = dataset.shuffle(seed=42).limit(num_samples)

    # 3. Export the images to a local directory
    # Hailo usually prefers a flat folder of .jpg
    print(f"Exporting to {output_dir}...")
    view.export(
        export_dir=output_dir,
        dataset_type=fo.types.ImageDirectory, 
        # We only need images for calibration, but if we wanted labels we'd use COCODetectionDataset
        # The user provided code used COCODetectionDataset and label_field="ground_truth"
        # Since quantize_har.py usually just reads images, ImageDirectory is cleaner (flat images),
        # BUT if the user wants to keep the code exactly as provided, I should stick to it or upgrade it.
        # User provided: dataset_type=fo.types.COCODetectionDataset
        # This creates a structure with data/ and labels.json.
        # The quantize_har.py I wrote expects a directory of images.
        # So ImageDirectory is BETTER for the pipeline I set up.
        # However, checking the user's prompt: "scripts to download coco val and 1024 images from train for calibration"
        # I'll stick to ImageDirectory to ensure compatibility with my quantize_har.py which does glob(dir, *.jpg)
    )

    print(f"Successfully exported {num_samples} images to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download COCO calibration images")
    parser.add_argument("--output", type=str, default="data/calib_images", help="Output directory")
    parser.add_argument("--samples", type=int, default=1024, help="Number of samples")
    args = parser.parse_args()
    
    download_hailo_calibration_set(args.output, args.samples)
