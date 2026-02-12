#ifndef POSTPROCESS_HPP
#define POSTPROCESS_HPP

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <map>

struct Detection {
    float x1, y1, x2, y2;
    float conf;
    int cls_id;
    const char* cls_name;
};

// COCO Classes
inline const std::vector<std::string> COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Identify output tensors by size
// Returns true on success, false on failure (missing tensors)
// We need to identify cls_80, cls_40, cls_20, reg_80, reg_40, reg_20
inline bool map_output_tensors(
    const std::map<std::string, std::vector<float>>& output_buffers,
    std::vector<const float*>& cls_ptrs,
    std::vector<const float*>& reg_ptrs) 
{
    cls_ptrs.resize(3, nullptr);
    reg_ptrs.resize(3, nullptr);

    for (const auto& pair : output_buffers) {
        size_t count = pair.second.size();
        
        if (count == 512000) cls_ptrs[0] = pair.second.data();      // Stride 8 cls
        else if (count == 128000) cls_ptrs[1] = pair.second.data(); // Stride 16 cls
        else if (count == 32000) cls_ptrs[2] = pair.second.data();  // Stride 32 cls
        else if (count == 25600) reg_ptrs[0] = pair.second.data();  // Stride 8 reg
        else if (count == 6400) reg_ptrs[1] = pair.second.data();   // Stride 16 reg
        else if (count == 1600) reg_ptrs[2] = pair.second.data();   // Stride 32 reg
    }
    
    // Verify all pointers found
    if (!cls_ptrs[0] || !cls_ptrs[1] || !cls_ptrs[2] || 
        !reg_ptrs[0] || !reg_ptrs[1] || !reg_ptrs[2]) {
        return false;
    }
    return true;
}

template <int... Is>
struct IntList {};

template <int... Strides, int... Grids>
std::vector<Detection> run_postprocess(
    IntList<Strides...>,
    IntList<Grids...>,
    const std::vector<const float*>& cls_tensors,
    const std::vector<const float*>& reg_tensors,
    float conf_threshold
) {
    const int strides[] = {Strides...};
    constexpr int grids[] = {Grids...};
    constexpr size_t NUM_SCALES = sizeof...(Strides);
    
    static_assert(sizeof...(Strides) == sizeof...(Grids), "Strides and Grids lists must be the same length");
    
    if (cls_tensors.size() != NUM_SCALES || reg_tensors.size() != NUM_SCALES) {
        std::cerr << "Error: Expected " << NUM_SCALES << " class and regression tensors, got " 
                  << cls_tensors.size() << " and " << reg_tensors.size() << std::endl;
        return {};
    }

    std::vector<Detection> results;
    
    // Clamp threshold
    if (conf_threshold <= 0.0f) conf_threshold = 0.001f;
    if (conf_threshold >= 1.0f) conf_threshold = 0.999f;
    
    float logit_threshold = -std::log(1.0f / conf_threshold - 1.0f);

    for (size_t s = 0; s < NUM_SCALES; ++s) {

        int grid_dim = grids[s];
        int num_anchors = grid_dim * grid_dim;
        const float* cls_data = cls_tensors[s];
        const float* reg_data = reg_tensors[s];

        for (int i = 0; i < num_anchors; ++i) {
            float max_logit = -1000.0f; 
            int class_id = -1;
            int anchor_offset = i * 80;
            
            for (int c = 0; c < 80; ++c) {
                float logit = cls_data[anchor_offset + c];
                if (logit > max_logit) {
                    max_logit = logit;
                    class_id = c;
                }
            }
            
            if (max_logit > logit_threshold) {
                float score = sigmoid(max_logit);
                
                int reg_offset = i * 4;
                float l = reg_data[reg_offset + 0];
                float t = reg_data[reg_offset + 1];
                float r = reg_data[reg_offset + 2];
                float b = reg_data[reg_offset + 3];
                
                int row = i / grid_dim;
                int col = i % grid_dim;
                
                float stride = (float)strides[s];
                
                float x1 = (col + 0.5f - l) * stride;
                float y1 = (row + 0.5f - t) * stride;
                float x2 = (col + 0.5f + r) * stride;
                float y2 = (row + 0.5f + b) * stride;
                
                Detection det;
                det.x1 = x1;
                det.y1 = y1;
                det.x2 = x2; 
                det.y2 = y2; 
                det.conf = score;
                det.cls_id = class_id;
                
                if (class_id >= 0 && (size_t)class_id < COCO_CLASSES.size()) {
                    det.cls_name = COCO_CLASSES[class_id].c_str();
                } else {
                    det.cls_name = "unknown";
                }
                
                results.push_back(det);
            }
        }
    }
    
    return results;
}

#endif // POSTPROCESS_HPP
