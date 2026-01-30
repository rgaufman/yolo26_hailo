#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>
#include "libnpy.hpp"
#include "postprocess.hpp"

int main() {
    // Paths to the npy files
    // Assuming we run from src/cpp, the files are in ../../output/npy/
    std::string base_path = "../../output/npy/";
    
    std::map<std::string, std::string> file_map = {
        {"cls_80", "yolo26n_conv64_float32.npy"},
        {"cls_40", "yolo26n_conv80_float32.npy"},
        {"cls_20", "yolo26n_conv94_float32.npy"},
        {"reg",    "yolo26n_ew_mult1_float32.npy"}
    };
    
    std::map<std::string, std::vector<float>> tensors;
    std::map<std::string, std::vector<unsigned long>> shapes;
    
    // Load tensors
    try {
        for (const auto& pair : file_map) {
            std::string path = base_path + pair.second;
            std::vector<unsigned long> shape;
            std::vector<float> data = npy::load_npy<float>(path, shape);
            
            tensors[pair.first] = data;
            shapes[pair.first] = shape;
            
            // Debug info
            // std::cout << "Loaded " << pair.first << " from " << path << " shape: (";
            // for (auto s : shape) std::cout << s << ", ";
            // std::cout << ")" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading .npy files: " << e.what() << std::endl;
        std::cerr << "Make sure you are running this from src/cpp and the .npy files exist in output/npy/" << std::endl;
        return 1;
    }

    // Prepare pointers for the templated function
    
    // Order matches strides 8, 16, 32 -> cls_80, cls_40, cls_20
    std::vector<const float*> cls_ptrs = {
        tensors["cls_80"].data(),
        tensors["cls_40"].data(),
        tensors["cls_20"].data()
    };
    
    const float* reg_ptr = tensors["reg"].data();

    // Run post-processing
    // Threshold 0.25 as per request
    float conf_threshold = 0.25f;
    
    // Call templated function
    // Strides: 8, 16, 32
    // Grids: 80, 40, 20
    std::vector<Detection> detections = run_postprocess(
        IntList<8, 16, 32>{},
        IntList<80, 40, 20>{},
        cls_ptrs,
        reg_ptr,
        conf_threshold
    );
    
    std::cout << "✓ Found " << detections.size() << " detections above threshold " << conf_threshold;
    
    // Check if we need to print newline here or not based on REQUEST format.
    // User request: "✓ Found 6 detections above threshold 0.25 [1] potted plant..."
    // It looks like one line?
    // "[1] potted plant - conf=0.50, x=235, y=266, w=270, h=319 [2] ..."
    // The user sample output is wrapped in the prompt but likely intended to be on one or multiple lines.
    // The previous python code printed one line per detection.
    // However, the REQUEST shows:
    // "✓ Found 6 detections above threshold 0.25 [1] potted plant - conf=0.50, x=235, y=266, w=270, h=319 [2] potted plant..."
    // It looks like it's all space separated in the example string.
    
    // But `common.py` `format_detection_results` prints:
    // "  [1] potted plant..." (with newlines).
    
    // The user explicitly asked for:
    // "expect the following results:
    // ✓ Found 6 detections above threshold 0.25 [1] potted plant - conf=0.50, x=235, y=266, w=270, h=319 [2] potted plant - conf=0.50, x=333, y=259, w=368, h=347 ..."
    
    // I will format it exactly as requested (all in one flowing text or just mimicking the sequence).
    // Actually, looking at the user prompt, it looks like a single line or maybe just a list.
    // I will iterate and print.
    
    // NOTE: The python code `format_detection_results` adds newlines.
    // The user prompt might have just pasted the output and it got wrapped.
    // But "expected results: ... [1] ... [2] ..." implies a sequence.
    // I'll print a newline after the count, then print each detection.
    
    std::cout << std::endl;
    
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        // Format: [1] potted plant - conf=0.50, x=235, y=266, w=270, h=319
        std::cout << "[" << (i + 1) << "] " << det.cls_name 
                  << " - conf=" << std::fixed << std::setprecision(2) << det.conf 
                  << ", x=" << std::setprecision(0) << det.x 
                  << ", y=" << det.y 
                  << ", w=" << det.w 
                  << ", h=" << det.h;
        
        if (i < detections.size() - 1) {
            std::cout << " "; 
        }
    }
    std::cout << std::endl;

    return 0;
}
