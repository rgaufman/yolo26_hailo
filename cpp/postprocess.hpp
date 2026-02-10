#ifndef POSTPROCESS_HPP
#define POSTPROCESS_HPP

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

struct Detection {
    float x, y, w, h;
    float conf;
    int cls_id;
    std::string cls_name;
};

// COCO Classes
static const std::vector<std::string> COCO_CLASSES = {
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

// Template Helper to carry compile-time integer lists
template <int... Is>
struct IntList {};

// Templated run_postprocess
// Strides: e.g. 8, 16, 32
// Grids: e.g. 80, 40, 20
// cls_tensors: Vector of pointers to class data for each scale. Order must match Strides/Grids.
// reg_tensor: Pointer to regression data (all scales concatenated)
template <int... Strides, int... Grids>
std::vector<Detection> run_postprocess(
    IntList<Strides...>,
    IntList<Grids...>,
    const std::vector<const float*>& cls_tensors,
    const std::vector<const float*>& reg_tensors,
    float conf_threshold
) {
    const int strides[] = {Strides...};
    (void)strides; // Suppress unused warning
    constexpr int grids[] = {Grids...};
    constexpr size_t NUM_SCALES = sizeof...(Strides);
    
    static_assert(sizeof...(Strides) == sizeof...(Grids), "Strides and Grids lists must be the same length");
    
    if (cls_tensors.size() != NUM_SCALES || reg_tensors.size() != NUM_SCALES) {
        std::cerr << "Error: Expected " << NUM_SCALES << " class and regression tensors, got " 
                  << cls_tensors.size() << " and " << reg_tensors.size() << std::endl;
        return {};
    }

    std::vector<Detection> results;
    // Calculate logit threshold once
    if (conf_threshold <= 0.0f) conf_threshold = 0.001f;
    if (conf_threshold >= 1.0f) conf_threshold = 0.999f;
    float logit_threshold = -std::log(1.0f / conf_threshold - 1.0f);

    for (size_t s = 0; s < NUM_SCALES; ++s) {

        int grid_dim = grids[s];
        int num_anchors = grid_dim * grid_dim;
        const float* cls_data = cls_tensors[s];

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
                

                const float* reg_data = reg_tensors[s];
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
                det.x = x1;
                det.y = y1;
                det.w = x2;
                det.h = y2;
                det.conf = score;
                det.cls_id = class_id;
                
                if (class_id >= 0 && (size_t)class_id < COCO_CLASSES.size()) {
                    det.cls_name = COCO_CLASSES[class_id];
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
