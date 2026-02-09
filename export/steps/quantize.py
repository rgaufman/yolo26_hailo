import tensorflow as tf
from hailo_sdk_client import ClientRunner
from pathlib import Path
from typing import Dict, Any, List
import shutil
import glob
from .base import Step
from ..config import ExportConfig

class QuantizeStep(Step):
    def __init__(self, config: ExportConfig):
        super().__init__("quantize", config)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.log_start()
        
        input_har_path = context['har_path']
        output_dir = self.config.output_dir / "artifacts" / "2_quantized"
        output_dir.mkdir(parents=True, exist_ok=True)
        quantized_har_path = output_dir / "model_quantized.har"
        
        calib_dir = self.config.calib_dir
        target = self.config.target
        
        # Setup .alls script
        if self.config.alls_path and self.config.alls_path.exists():
             alls_path = self.config.alls_path
             self.logger.info(f"Using provided .alls script: {alls_path}")
        else:
             # Default from config
             default_alls_name = context['variant_config'].default_alls
             # Assume it's in the export root or handled via resource loading
             # For now, look in export dir relative to this file
             # /export/steps/quantize.py -> /export/
             export_root = Path(__file__).parent.parent
             default_alls_path = export_root / default_alls_name
             if default_alls_path.exists():
                 alls_path = default_alls_path
                 self.logger.info(f"Using default .alls script: {alls_path}")
             else:
                 alls_path = None
                 self.logger.warning("No .alls script found. Proceeding without one.")

        # Copy .alls to run dir for reproducibility
        if alls_path:
            shutil.copy(alls_path, self.config.output_dir / "model_script.alls")

        self.logger.info(f"Initializing ClientRunner for {target}")
        runner = ClientRunner(hw_arch=target)
        runner.load_har(str(input_har_path))
        
        if alls_path:
            self.logger.info(f"Loading model script: {alls_path}")
            runner.load_model_script(str(alls_path))
            
        # Calibration Data Loading
        self.logger.info(f"Loading calibration images from {calib_dir}")
        images = self._load_calibration_data(calib_dir)
        
        # Optimize
        self.logger.info("Starting optimization...")
        runner.optimize(images, data_type="dataset")
        
        runner.save_har(str(quantized_har_path))
        self.logger.info(f"Saved optimized HAR to: {quantized_har_path}")
        
        context['quantized_har_path'] = quantized_har_path
        self.log_end()
        return context

    def _load_calibration_data(self, calib_dir: Path) -> Any:
        def load_and_preprocess_image(path):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [640, 640])
            return tf.cast(image, tf.float32), {}

        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(glob.glob(str(calib_dir / '**' / ext), recursive=True))
            
        limit = 1024
        if len(image_paths) > limit:
            image_paths = image_paths[:limit]
            
        if not image_paths:
            raise ValueError(f"No images found in {calib_dir}")
            
        self.logger.info(f"Using {len(image_paths)} images for calibration.")
        
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        return dataset
