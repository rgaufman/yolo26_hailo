from hailo_sdk_client import ClientRunner
import sys
import os

# accept 2 arguments: the path to input HAR and output HEF
if len(sys.argv) != 3:
    print("Usage: python 3_compile_hef.py <input_har_path> <output_hef_path>")
    sys.exit(1)

input_har_path = sys.argv[1]
output_hef_path = sys.argv[2]
alls_script_path = 'yolo26n.alls'

if not os.path.exists(input_har_path):
    print(f"Error: Input HAR file {input_har_path} not found.")
    sys.exit(1)

# 1. Initialize and load the optimized HAR from Step 2
runner = ClientRunner(hw_arch='hailo8l')
runner.load_har(input_har_path)

if os.path.exists(alls_script_path):
    runner.load_model_script(alls_script_path)

# 2. Trigger the Compilation
# This automated process allocates hardware resources and generates microcode.
print("Starting compilation...")
hef = runner.compile()

# 3. Save the binary for your Raspberry Pi 5
with open(output_hef_path, 'wb') as f:
    f.write(hef)

print(f"Compilation successful: {output_hef_path} generated.")
