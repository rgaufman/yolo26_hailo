from hailo_sdk_client import ClientRunner
from pathlib import Path
from typing import Dict, Any
from .base import Step
from ..config import ExportConfig

class CompileStep(Step):
    def __init__(self, config: ExportConfig):
        super().__init__("compile", config)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.log_start()
        
        input_har_path = context['quantized_har_path']
        output_dir = self.config.output_dir / "artifacts" / "3_compiled"
        output_dir.mkdir(parents=True, exist_ok=True)
        hef_path = output_dir / "model.hef"
        target = self.config.target

        self.logger.info(f"Initializing ClientRunner for {target}")
        runner = ClientRunner(hw_arch=target)
        runner.load_har(str(input_har_path))
        
        self.logger.info("Starting compilation...")
        hef = runner.compile()
        
        with open(hef_path, 'wb') as f:
            f.write(hef)
            
        self.logger.info(f"Compilation successful: {hef_path} generated.")
        
        context['hef_path'] = hef_path
        self.log_end()
        return context
