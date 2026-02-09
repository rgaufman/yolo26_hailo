from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Literal
from pathlib import Path

class YOLOVariantConfig(BaseModel):
    name: str
    description: str
    weights_name: str
    input_shape: tuple = (640, 640)
    backbone_inputs: List[str] = Field(default_factory=lambda: ['images'])
    head_inputs: List[str]
    head_outputs: List[str] = Field(default_factory=lambda: ['output0'])
    end_nodes: List[str]
    default_alls: str

class ExportConfig(BaseModel):
    variant: str
    target: str = "hailo8l"
    onnx_path: Optional[Path] = None
    calib_dir: Path
    output_dir: Optional[Path] = None
    alls_path: Optional[Path] = None
    tag: Optional[str] = None

    @validator('target')
    def validate_target(cls, v):
        allowed = ['hailo8', 'hailo8l', 'hailo15', 'hailo10'] # Add other targets as needed
        if v not in allowed:
            # We warn but don't fail, maybe new targets exist
            pass 
        return v

# Define standard variants
VARIANTS: Dict[str, YOLOVariantConfig] = {
    'yolo26n': YOLOVariantConfig(
        name='yolo26n',
        description='YOLO26 Nano',
        weights_name='yolo26n.pt',
        head_inputs=[
            '/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0', 
            '/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0', 
            '/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0', 
            '/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0',
            '/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0',
            '/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0'
        ],
        end_nodes=[
            '/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv', 
            '/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv', 
            '/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv', 
            '/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv',
            '/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv',
            '/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv'
        ],
        default_alls='yolo26n.alls'
    ),
    'yolo26s': YOLOVariantConfig(
        name='yolo26s',
        description='YOLO26 Small',
        weights_name='yolo26s.pt',
        head_inputs=[
            '/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0', 
            '/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0', 
            '/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0', 
            '/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0', 
            '/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0', 
            '/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0'
        ],
        end_nodes=[
            '/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv', 
            '/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv', 
            '/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv', 
            '/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv', 
            '/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv', 
            '/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv'
        ],
        default_alls='yolo26s.alls'
    ),
    'yolo26m': YOLOVariantConfig(
        name='yolo26m',
        description='YOLO26 Medium',
        weights_name='yolo26m.pt',
        head_inputs=[
            '/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0', 
            '/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0', 
            '/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0', 
            '/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0', 
            '/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0', 
            '/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0'
        ],
        end_nodes=[
            '/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv', 
            '/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv', 
            '/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv', 
            '/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv', 
            '/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv', 
            '/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv'
        ],
        default_alls='yolo26m.alls'
    ),
    'yolo26l': YOLOVariantConfig(
        name='yolo26l',
        description='YOLO26 Large',
        weights_name='yolo26l.pt',
        head_inputs=[
            '/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0', 
            '/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0', 
            '/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0', 
            '/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0', 
            '/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0', 
            '/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0'
        ],
        end_nodes=[
            '/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv', 
            '/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv', 
            '/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv', 
            '/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv', 
            '/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv', 
            '/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv'
        ],
        default_alls='yolo26l.alls'
    ),
}

def get_variant_config(name: str) -> YOLOVariantConfig:
    if name not in VARIANTS:
        raise ValueError(f"Unknown variant: {name}. Available: {list(VARIANTS.keys())}")
    return VARIANTS[name]
