/**
 * run_coco_inference.cpp
 * 
 * Runs inference on the COCO validation set and generates a JSON file
 * compatible with COCO evaluation tools.
 * 
 * Usage: ./run_coco_inference <coco_images_dir> <hef_path> <output.json> [--limit N]
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
#include <fstream>
#include <iomanip>
#include <memory>
#include <fstream>

#include "postprocess.hpp"

using namespace hailort;

// Constants
constexpr int TARGET_WIDTH = 640;
constexpr int TARGET_HEIGHT = 640;

// Mapping from YOLO Model Class Index (0-79) to COCO Category ID (1-90)
// This mapping is derived from standard YOLOv8/v5 coco.yaml and COCO 2017 dataset.
// Based on: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
// And standard COCO IDs.
const int COCO_IDS[] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90
};

void print_error(hailo_status status, const std::string& msg) {
    std::cerr << "Error: " << msg << " (Status: " << status << ")" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: ./run_coco_inference <coco_val_dir> <hef_path> <output_json> [--limit N]" << std::endl;
        return 1;
    }

    std::string coco_dir = argv[1];
    std::string hef_path = argv[2];
    std::string output_json_path = argv[3];
    int limit = -1;

    if (argc > 5 && std::string(argv[4]) == "--limit") {
        limit = std::stoi(argv[5]);
    }

    // 1. Setup Hailo
    std::cout << "[Setting up Hailo Device...]" << std::endl;
    
    auto vdevice_exp = VDevice::create();
    if (!vdevice_exp) { print_error(vdevice_exp.status(), "Failed to create VDevice"); return 1; }
    auto vdevice = std::move(vdevice_exp.value());

    auto hef_exp = Hef::create(hef_path);
    if (!hef_exp) { print_error(hef_exp.status(), "Failed to load HEF"); return 1; }
    auto hef = std::move(hef_exp.value());

    auto configure_params_exp = vdevice->create_configure_params(hef);
    if (!configure_params_exp) { print_error(configure_params_exp.status(), "Failed to create configure params"); return 1; }
    auto configure_params = configure_params_exp.value();

    auto network_groups_exp = vdevice->configure(hef, configure_params);
    if (!network_groups_exp) { print_error(network_groups_exp.status(), "Failed to configure network group"); return 1; }
    auto network_groups = std::move(network_groups_exp.value());
    auto network_group = network_groups[0];

    // Create VStreams
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

    if (input_vstreams.size() != 1) {
        std::cerr << "Error: Expected 1 input stream, got " << input_vstreams.size() << std::endl;
        return 1;
    }

    // 2. Prepare Output File
    std::ofstream out_file(output_json_path);
    if (!out_file.is_open()) {
        std::cerr << "Error: Could not open output file: " << output_json_path << std::endl;
        return 1;
    }
    out_file << "[" << std::endl; // Start JSON array

    // 3. Process Images
    std::vector<std::string> images;
    cv::glob(coco_dir + "/*.jpg", images);
    
    if (images.empty()) {
        std::cerr << "Warning: No images found in " << coco_dir << std::endl;
    } else {
        std::cout << "Found " << images.size() << " images." << std::endl;
    }

    if (limit > 0 && limit < (int)images.size()) {
        images.resize(limit);
        std::cout << "Limiting to " << limit << " images." << std::endl;
    }

    bool first_detection = true;
    int processed_count = 0;
    
    // Output buffers
    std::map<std::string, std::vector<float>> output_buffers;
    for (auto& ov : output_vstreams) {
        output_buffers[ov.name()] = std::vector<float>(ov.get_frame_size() / sizeof(float));
    }

    for (const auto& img_path : images) {
        processed_count++;
        if (processed_count % 100 == 0) std::cout << "Processing " << processed_count << "/" << images.size() << std::endl;

        // Parse Image ID
        // filename is like "000000000139.jpg"
        std::string filename = img_path.substr(img_path.find_last_of("/\\") + 1);
        std::string id_str = filename.substr(0, filename.find_last_of("."));
        long long image_id = 0;
        try {
            image_id = std::stoll(id_str);
        } catch (...) {
            std::cerr << "Warning: Could not parse ID from " << filename << ", skipping." << std::endl;
            continue;
        }

        // Load & Preprocess
        cv::Mat orig_image = cv::imread(img_path);
        if (orig_image.empty()) {
            std::cerr << "Error: Could not read image: " << img_path << std::endl;
            continue;
        }

        cv::Mat resized_image;
        cv::resize(orig_image, resized_image, cv::Size(TARGET_WIDTH, TARGET_HEIGHT));
        cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);

        // Inference
        hailo_status status = input_vstreams[0].write(MemoryView(resized_image.data, resized_image.total() * resized_image.elemSize()));
        if (status != HAILO_SUCCESS) { print_error(status, "Inference write failed"); break; }
        
        for (auto& ov : output_vstreams) {
            status = ov.read(MemoryView(output_buffers[ov.name()].data(), output_buffers[ov.name()].size() * sizeof(float)));
            if (status != HAILO_SUCCESS) { print_error(status, "Inference read failed"); break; }
        }

        // Prepare Postprocess Pointers
        std::vector<const float*> cls_ptrs(3);
        const float* reg_ptr = nullptr;
        
        for (auto& pair : output_buffers) {
            size_t count = pair.second.size();
            if (count == 512000) cls_ptrs[0] = pair.second.data();      // Stride 8
            else if (count == 128000) cls_ptrs[1] = pair.second.data(); // Stride 16
            else if (count == 32000) cls_ptrs[2] = pair.second.data();  // Stride 32
            else if (count == 33600) reg_ptr = pair.second.data();
        }
        
        if (!cls_ptrs[0] || !cls_ptrs[1] || !cls_ptrs[2] || !reg_ptr) {
             std::cerr << "Error: Invalid output buffer sizes." << std::endl;
             continue;
        }

        // Postprocess
        // Use 0.001 threshold as typical for COCO mAP evaluation
        std::vector<Detection> detections = run_postprocess(
            IntList<8, 16, 32>{},
            IntList<80, 40, 20>{},
            cls_ptrs,
            reg_ptr,
            0.001f 
        );

        // Scale and Save
        float scale_x = (float)orig_image.cols / TARGET_WIDTH;
        float scale_y = (float)orig_image.rows / TARGET_HEIGHT;

        for (const auto& det : detections) {
            // Scale
            float x = det.x * scale_x;
            float y = det.y * scale_y;
            float w = det.w * scale_x;
            float h = det.h * scale_y;
            
            // COCO format: [x_min, y_min, width, height]
            // det.x, det.y are top-left?
            // postprocess.hpp:
            // "box[0] = cx - w/2; box[1] = cy - h/2; box[2] = cx + w/2; box[3] = cy + h/2;"
            // wait, run_postprocess returns x,y,w,h in what format?
            // "det.x = x1; det.y = y1; det.w = x2; det.h = y2;" based on:
            // "detections.push_back({x1, y1, x2, y2, score, class_id, COCO_CLASSES[class_id]});"
            // So detection struct x,y,w,h are actually x1, y1, x2, y2.
            // Wait, looking at struct: "float x, y, w, h;"
            // Postprocess implementation details:
            // "float x1 = (cx - w_box / 2.0f) * stride;"
            // "results.push_back({x1, y1, x2, y2, score, class_id, ...})"
            // So YES: x=x1, y=y1, w=x2, h=y2.
            
            // COCO expects [x, y, width, height].
            // So: x_coco = x1, y_coco = y1, w_coco = x2 - x1, h_coco = y2 - y1.
            
            float coco_x = x;
            float coco_y = y;
            float coco_w = w - x;
            float coco_h = h - y;
            
            if (!first_detection) out_file << "," << std::endl;
            first_detection = false;
            
            int category_id = COCO_IDS[det.cls_id]; // Map 0-79 to COCO ID

            out_file << "  {" << std::endl;
            out_file << "    \"image_id\": " << image_id << "," << std::endl;
            out_file << "    \"category_id\": " << category_id << "," << std::endl;
            out_file << "    \"bbox\": [" << coco_x << ", " << coco_y << ", " << coco_w << ", " << coco_h << "]," << std::endl;
            out_file << "    \"score\": " << std::fixed << std::setprecision(5) << det.conf << std::endl;
            out_file << "  }";
        }
    }

    out_file << std::endl << "]" << std::endl;
    out_file.close();
    
    std::cout << "Done. Saved processing results to " << output_json_path << std::endl;

    return 0;
}
