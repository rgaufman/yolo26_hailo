import os
from onnx.utils import extract_model
from pathlib import Path
from typing import Dict, Any
from .base import Step
from ..config import ExportConfig, YOLOVariantConfig

class ExtractSubgraphsStep(Step):
    def __init__(self, config: ExportConfig):
        super().__init__("extract_subgraphs", config)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.log_start()
        
        variant_config: YOLOVariantConfig = context['variant_config']
        onnx_path = self.config.onnx_path
        
        # Output paths
        output_dir = self.config.output_dir / "artifacts" / "0_subgraphs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        backbone_path = output_dir / f"{variant_config.name}_backbone.onnx"
        head_path = output_dir / f"{variant_config.name}_head.onnx"
        
        # Subgraph extraction logic
        self.logger.info(f"Extracting subgraphs from {onnx_path}")
        
        # 1. Extract Head
        self.logger.info("Extracting Head...")
        try:
            extract_model(
                str(onnx_path),
                str(head_path),
                variant_config.head_inputs,
                variant_config.head_outputs,
                check_model=True
            )
            self.logger.info(f"✓ Head saved to: {head_path}")
        except Exception as e:
            self.logger.error(f"Error extracting head: {e}")
            raise

        # 2. Extract Backbone
        self.logger.info("Extracting Backbone...")
        try:
            extract_model(
                str(onnx_path),
                str(backbone_path),
                variant_config.backbone_inputs,
                variant_config.head_inputs,
                check_model=True
            )
            self.logger.info(f"✓ Backbone saved to: {backbone_path}")
        except Exception as e:
            self.logger.error(f"Error extracting backbone: {e}")
            raise
            
        context['backbone_path'] = backbone_path
        context['head_path'] = head_path
        
        self.log_end()
        return context

if __name__ == "__main__":
    # Allow standalone run for debugging (requires mocked config)
    pass
