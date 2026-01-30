import onnx
from onnx.utils import extract_model
import sys
import os

# Usage: python 0_extract_subgraphs.py <input_onnx> <output_dir>

if len(sys.argv) < 3:
    print("Usage: python 0_extract_subgraphs.py <input_onnx> <output_dir>")
    sys.exit(1)

input_model_path = sys.argv[1]
output_dir = sys.argv[2]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_model_backbone_path = os.path.join(output_dir, 'yolo26n_backbone.onnx')
output_model_head_path = os.path.join(output_dir, 'yolo26n_head.onnx')

print(f"Extracting subgraphs from {input_model_path}...")

# Definition of split points
# These must match the node names in the graph. 
# For YOLOv8/YOLO26n (Ultralytics export), these are usually the outputs of the last C2f/Conv modules before the head.
backbone_inputs = ['images']
backbone_outputs = [
    '/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv', 
    '/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv', 
    '/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv', 
    '/model.23/Mul_2'
]

# The head inputs are the backbone outputs (typically appended with _output_0 in ONNX when they become inputs)
# BUT extract_model expects tensor names.
# Check if we should use the exact names or different ones.
# In onnx_subgraph.py provided: 
# subgraph_inputs = ['...Conv_output_0', ...]
head_inputs = [name + '_output_0' for name in backbone_outputs]
# Note: /model.23/Mul_2 might not have _output_0 if it's already an output, but usually intermediate tensors do.
# Let's trust the user's original script names which had _output_0.
head_inputs = [
    '/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0', 
    '/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0', 
    '/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0', 
    '/model.23/Mul_2_output_0'
]

# Final output of the head
head_outputs = ['output0']

# 1. Extract Head
print("Extracting Head...")
try:
    extract_model(
        input_model_path, 
        output_model_head_path, 
        head_inputs, 
        head_outputs,
        check_model=True 
    )
    print(f"✓ Head saved to: {output_model_head_path}")
except Exception as e:
    print(f"Error extracting head: {e}")
    # Fallback/Debug info: maybe names are wrong?

# 2. Extract Backbone
# Note: For Hailo Dataflow Compiler, we usually use the HAR translation process which does this internally.
# But having a standalone backbone ONNX is useful for debugging or other runners.
print("Extracting Backbone...")
try:
    extract_model(
        input_model_path, 
        output_model_backbone_path, 
        backbone_inputs, 
        head_inputs, # The outputs of the backbone are the inputs of the head
        check_model=True 
    )
    print(f"✓ Backbone saved to: {output_model_backbone_path}")
except Exception as e:
    print(f"Error extracting backbone: {e}")

print("Done.")
