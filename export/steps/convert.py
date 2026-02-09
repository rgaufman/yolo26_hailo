from hailo_sdk_client import ClientRunner
from pathlib import Path
from typing import Dict, Any
from .base import Step
from ..config import ExportConfig, YOLOVariantConfig

class OnnxToHarStep(Step):
    def __init__(self, config: ExportConfig):
        super().__init__("onnx_to_har", config)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.log_start()
        
        variant_config: YOLOVariantConfig = context['variant_config']
        onnx_path = self.config.onnx_path
        target = self.config.target
        
        output_dir = self.config.output_dir / "artifacts" / "1_parsed"
        output_dir.mkdir(parents=True, exist_ok=True)
        har_path = output_dir / "model.har"
        
        self.logger.info(f"Initializing ClientRunner for {target}")
        runner = ClientRunner(hw_arch=target)
        
        self.logger.info(f"Parsing ONNX model: {onnx_path}")
        start_node = variant_config.backbone_inputs[0] # assuming single input 'images'
        end_nodes = variant_config.end_nodes
        
        self.logger.info(f"Start node: {start_node}")
        self.logger.info(f"End nodes: {end_nodes}")
        
        runner.translate_onnx_model(
            str(onnx_path),
            variant_config.name,
            start_node_names=[start_node],
            end_node_names=end_nodes
        )
        
        runner.save_har(str(har_path))
        self.logger.info(f"Saved HAR to: {har_path}")
        
        context['har_path'] = har_path
        
        self.log_end()
        return context
