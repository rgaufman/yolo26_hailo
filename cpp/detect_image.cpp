/**
 * detect_image.cpp
 * 
 * Single image detection with Hailo-8L + Post-processing in C++.
 * Usage: ./detect_image <image_path> [hef_path] [conf_threshold]
 */

#include "hailo/hailort.hpp"
#include "hailo/vdevice.hpp"
#include "hailo/infer_model.hpp"
#include "hailo/vstream.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <map>

#include "postprocess.hpp"

using namespace hailort;

// Constants
constexpr int TARGET_WIDTH = 640;
constexpr int TARGET_HEIGHT = 640;

// Helper to print errors
void print_error(hailo_status status, const std::string& msg) {
    std::cerr << "Error: " << msg << " (Status: " << status << ")" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./detect_image <image_path> [hef_path] [conf_threshold]" << std::endl;
        return 1;
    }

    std::string image_path = argv[1];
    std::string hef_path = (argc > 2) ? argv[2] : "../models/yolo26n.hef";
    float conf_threshold = (argc > 3) ? std::stof(argv[3]) : 0.25f;

    std::cout << "[Loading image: " << image_path << "]" << std::endl;

    // 1. Preprocess Image
    cv::Mat orig_image = cv::imread(image_path);
    if (orig_image.empty()) {
        std::cerr << "Error: Could not read image: " << image_path << std::endl;
        return 1;
    }

    std::cout << "✓ Original image size: " << orig_image.cols << "x" << orig_image.rows << std::endl;

    cv::Mat resized_image;
    cv::resize(orig_image, resized_image, cv::Size(TARGET_WIDTH, TARGET_HEIGHT));
    
    // User requested RGB
    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);

    std::cout << "✓ Preprocessed to: (" << resized_image.rows << ", " << resized_image.cols << ", " << resized_image.channels() << ")" << std::endl;

    // 2. Setup Hailo
    auto vdevice_exp = VDevice::create();
    if (!vdevice_exp) {
        print_error(vdevice_exp.status(), "Failed to create VDevice");
        return 1;
    }
    auto vdevice = std::move(vdevice_exp.value());

    auto hef_exp = Hef::create(hef_path);
    if (!hef_exp) {
        print_error(hef_exp.status(), "Failed to load HEF");
        return 1;
    }
    auto hef = std::move(hef_exp.value());

    auto configure_params_exp = vdevice->create_configure_params(hef);
    if (!configure_params_exp) {
        print_error(configure_params_exp.status(), "Failed to create configure params");
        return 1;
    }
    auto configure_params = configure_params_exp.value();

    auto network_groups_exp = vdevice->configure(hef, configure_params);
    if (!network_groups_exp) {
        print_error(network_groups_exp.status(), "Failed to configure network group");
        return 1;
    }
    auto network_groups = std::move(network_groups_exp.value());
    auto network_group = network_groups[0];

    // Create VStreams
    // Input: UINT8, NHWC/Auto (Matches image)
    // Output: FLOAT32 (Matches requirement for automatic dequantization)
    
    // Let's create params manually to force FLOAT32 on output
    auto input_vstream_params = network_group->make_input_vstream_params(false, HAILO_FORMAT_TYPE_UINT8, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    if (!input_vstream_params) { print_error(input_vstream_params.status(), "Failed to make input params"); return 1; }
    
    auto output_vstream_params = network_group->make_output_vstream_params(false, HAILO_FORMAT_TYPE_FLOAT32, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    if (!output_vstream_params) { print_error(output_vstream_params.status(), "Failed to make output params"); return 1; }

    auto input_vstreams_exp = VStreamsBuilder::create_input_vstreams(*network_group, input_vstream_params.value());
    if (!input_vstreams_exp) { print_error(input_vstreams_exp.status(), "Failed to create input vstreams"); return 1; }
    auto input_vstreams = std::move(input_vstreams_exp.value());

    auto output_vstreams_exp = VStreamsBuilder::create_output_vstreams(*network_group, output_vstream_params.value());
    if (!output_vstreams_exp) { print_error(output_vstreams_exp.status(), "Failed to create output vstreams"); return 1; }
    auto output_vstreams = std::move(output_vstreams_exp.value());

    // 3. Inference
    std::cout << "[Running inference...]" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Send Input
    // Assuming single input
    if (input_vstreams.size() != 1) {
        std::cerr << "Error: Expected 1 input stream, got " << input_vstreams.size() << std::endl;
        return 1;
    }
    
    hailo_status status = input_vstreams[0].write(MemoryView(resized_image.data, resized_image.total() * resized_image.elemSize()));
    if (status != HAILO_SUCCESS) {
        print_error(status, "Failed to write input");
        return 1;
    }

    // Read Outputs
    // We expect cls_80, cls_40, cls_20, and reg.
    // We need to map them by name.
    
    std::map<std::string, std::vector<float>> output_data;
    
    for (auto& output_vstream : output_vstreams) {
        std::string name = output_vstream.name(); // This is the vstream name
        
        // The VStream names might be slightly different from the python dictionary keys if python remapped them.
        // In common.py: shape_to_name map was used for Python...
        // But here we might get raw names from HEF.
        // We will read the data and try to identify by shape later or hope names make sense.
        // Actually, let's collect all data.
        
        size_t framesize = output_vstream.get_frame_size();
        std::vector<float> buffer(framesize / sizeof(float)); // Since we requested FLOAT32
        
        status = output_vstream.read(MemoryView(buffer.data(), framesize));
        if (status != HAILO_SUCCESS) {
            print_error(status, "Failed to read output " + name);
            return 1;
        }
        
        output_data[name] = std::move(buffer);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    std::cout << "✓ Inference completed in " << duration.count() << "ms" << std::endl;

    // 4. Map outputs to expected structure for post-processing
    // We need to identify which is which.
    // Based on shapes (deduced from size):
    // cls_80: 80*80*80 = 512000 floats
    // cls_40: 40*40*80 = 128000 floats
    // cls_20: 20*20*80 = 32000 floats
    // reg:    8400*4   = 33600 floats
    
    std::vector<const float*> cls_ptrs(3);
    const float* reg_ptr = nullptr;
    
    for (auto& pair : output_data) {
        size_t count = pair.second.size();
        if (count == 512000) cls_ptrs[0] = pair.second.data();      // Stride 8
        else if (count == 128000) cls_ptrs[1] = pair.second.data(); // Stride 16
        else if (count == 32000) cls_ptrs[2] = pair.second.data();  // Stride 32
        else if (count == 33600) reg_ptr = pair.second.data();
        else {
            std::cout << "Warning: Unknown output tensor size: " << count << " (" << pair.first << ")" << std::endl;
        }
    }
    
    if (!cls_ptrs[0] || !cls_ptrs[1] || !cls_ptrs[2] || !reg_ptr) {
        std::cerr << "Error: Could not identify all required output tensors by size." << std::endl;
        std::cerr << "Sizes found:" << std::endl;
        for (auto& pair : output_data) std::cout << pair.first << ": " << pair.second.size() << std::endl;
        return 1;
    }

    // 5. Post Process
    std::vector<Detection> detections = run_postprocess(
        IntList<8, 16, 32>{},
        IntList<80, 40, 20>{},
        cls_ptrs,
        reg_ptr,
        conf_threshold
    );

    std::cout << "✓ Found " << detections.size() << " detections above threshold " << conf_threshold << std::endl;

    // 6. Visualize
    // Scale detections
    float scale_y = (float)orig_image.rows / TARGET_HEIGHT;
    float scale_x = (float)orig_image.cols / TARGET_WIDTH; // Actually we resized ignoring aspect ratio in step 1?
    // "cv::resize(orig_image, resized_image, cv::Size(TARGET_WIDTH, TARGET_HEIGHT));" -> Yes, stretched.
    // So separate scales are correct.
    
    for (const auto& det : detections) {
        int x1 = (int)(det.x * scale_x);
        int y1 = (int)(det.y * scale_y);
        int x2 = (int)(det.w * scale_x); // det.w is actually x2 (right)
        int y2 = (int)(det.h * scale_y); // det.h is actually y2 (bottom)
        
        cv::rectangle(orig_image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
        
        std::string label = det.cls_name + " " + std::to_string(det.conf).substr(0, 4);
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(orig_image, cv::Point(x1, y1 - textSize.height - 5), cv::Point(x1 + textSize.width, y1), cv::Scalar(0, 255, 0), -1);
        cv::putText(orig_image, label, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        
        std::cout << "  " << label << " at [" << x1 << "," << y1 << "," << x2 << "," << y2 << "]" << std::endl;
    }

    std::string output_filename = "output_detected.jpg";
    cv::imwrite(output_filename, orig_image);
    std::cout << "✓ Output image saved to: " << output_filename << std::endl;

    return 0;
}
