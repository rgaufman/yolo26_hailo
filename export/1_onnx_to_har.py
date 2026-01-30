from hailo_sdk_client import ClientRunner
import sys
import os

# usage: python 1_onnx_to_har.py <input_onnx_path> <output_har_path>
if len(sys.argv) != 3:
    print("Usage: python 1_onnx_to_har.py <input_onnx_path> <output_har_path>")
    sys.exit(1)

onnx_path = sys.argv[1]
output_har_path = sys.argv[2]
model_name = 'yolo26n'

if not os.path.exists(onnx_path):
    print(f"Error: Input ONNX file {onnx_path} not found.")
    sys.exit(1)

# 1. Initialize the runner for Hailo-8L
runner = ClientRunner(hw_arch='hailo8l')

# 2. Parse the ONNX model
# Using start node 'images' and specified end nodes for separating backbone and head
start_node = 'images'
# Example paths from Netron for YOLOv8/YOLO26 structures
end_nodes = [
    '/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv', 
    '/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv', 
    '/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv', 
    '/model.23/Mul_2'
]

print(f"Parsing ONNX model: {onnx_path}")
print(f"Start node: {start_node}")
print(f"End nodes: {end_nodes}")

runner.translate_onnx_model(
    onnx_path, 
    model_name,
    start_node_names=[start_node],
    end_node_names=end_nodes
)

# 3. Save the Hailo Archive (HAR)
runner.save_har(output_har_path)
print(f"Saved HAR to: {output_har_path}")
