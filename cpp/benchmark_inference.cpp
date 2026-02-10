/**
 * benchmark_inference.cpp
 * 
 * Benchmark the Hailo-8L inference performance.
 * Measures the time taken for write + read (roundtrip) over multiple iterations.
 * Uses random input data.
 * 
 * Usage: ./benchmark_inference [hef_path] [iterations]
 */

#include "hailo/hailort.hpp"
#include "hailo/vdevice.hpp"
#include "hailo/infer_model.hpp"
#include "hailo/vstream.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <random>
#include <map>

#include "postprocess.hpp"

using namespace hailort;

// Constants matching our known model input
constexpr int INPUT_WIDTH = 640;
constexpr int INPUT_HEIGHT = 640;
constexpr int INPUT_CHANNELS = 3;
constexpr size_t INPUT_SIZE = INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS;

void print_error(hailo_status status, const std::string& msg) {
    std::cerr << "Error: " << msg << " (Status: " << status << ")" << std::endl;
}

void print_stats(const std::string& name, const std::vector<double>& timings) {
    if (timings.empty()) return;
    double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
    double mean = sum / timings.size();
    double sq_sum = std::inner_product(timings.begin(), timings.end(), timings.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / timings.size() - mean * mean);
    double min_val = *std::min_element(timings.begin(), timings.end());
    double max_val = *std::max_element(timings.begin(), timings.end());
    
    std::cout << "  " << name << ":" << std::endl;
    std::cout << "    Mean: " << mean << " ms" << std::endl;
    std::cout << "    Std : " << stdev << " ms" << std::endl;
    std::cout << "    Min : " << min_val << " ms" << std::endl;
    std::cout << "    Max : " << max_val << " ms" << std::endl;
}

int main(int argc, char** argv) {
    std::string hef_path = (argc > 1) ? argv[1] : "../models/yolo26n.hef";
    int iterations = (argc > 2) ? std::stoi(argv[2]) : 100;
    
    if (iterations <= 0) iterations = 100;

    std::cout << "[Benchmark Configuration]" << std::endl;
    std::cout << "  HEF: " << hef_path << std::endl;
    std::cout << "  Iterations: " << iterations << std::endl;

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
    // Use same config as detect_image: Input UINT8, Output FLOAT32
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

    // 2. Prepare Data
    std::cout << "[Generating Random Input Data...]" << std::endl;
    std::vector<uint8_t> input_data(INPUT_SIZE);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    for (size_t i = 0; i < INPUT_SIZE; ++i) {
        input_data[i] = static_cast<uint8_t>(dis(gen));
    }

    // Prepare output buffers map
    std::map<std::string, std::vector<float>> output_buffers;
    for (auto& ov : output_vstreams) {
        output_buffers[ov.name()] = std::vector<float>(ov.get_frame_size() / sizeof(float));
    }

    // 3. Warmup
    std::cout << "[Warming up...]" << std::endl;
    for (int i = 0; i < 10; ++i) {
        hailo_status status = input_vstreams[0].write(MemoryView(input_data.data(), input_data.size()));
        if (status != HAILO_SUCCESS) { print_error(status, "Warmup write failed"); return 1; }
        
        for (auto& ov : output_vstreams) {
            status = ov.read(MemoryView(output_buffers[ov.name()].data(), output_buffers[ov.name()].size() * sizeof(float)));
            if (status != HAILO_SUCCESS) { print_error(status, "Warmup read failed"); return 1; }
        }
    }

    // Pre-calculate pointers for post-processing to avoid overhead in loop
    std::vector<const float*> cls_ptrs(3);
    std::vector<const float*> reg_ptrs(3);
    
    // We Map once. Note: The data in the buffers changes, but the pointers to the vectors' data stay valid 
    // as long as vectors are not resized. They are fixed size here.
    if (!map_output_tensors(output_buffers, cls_ptrs, reg_ptrs)) {
         std::cerr << "Warning: Could not map output tensors correctly. benchmark might crash or fail." << std::endl;
    }

    // 4. Benchmark Loop
    std::cout << "[Running Benchmark (" << iterations << " iterations)...]" << std::endl;
    std::vector<double> inf_timings;
    std::vector<double> post_timings;
    std::vector<double> total_timings;
    
    inf_timings.reserve(iterations);
    post_timings.reserve(iterations);
    total_timings.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        // --- Inference ---
        auto start_inf = std::chrono::high_resolution_clock::now();
        
        hailo_status status = input_vstreams[0].write(MemoryView(input_data.data(), input_data.size()));
        if (status != HAILO_SUCCESS) { print_error(status, "Benchmark write failed"); return 1; }
        
        for (auto& ov : output_vstreams) {
            status = ov.read(MemoryView(output_buffers[ov.name()].data(), output_buffers[ov.name()].size() * sizeof(float)));
            if (status != HAILO_SUCCESS) { print_error(status, "Benchmark read failed"); return 1; }
        }
        
        auto end_inf = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_inf = end_inf - start_inf;
        inf_timings.push_back(duration_inf.count());

        // --- Post-process ---
        auto start_post = std::chrono::high_resolution_clock::now();
        
        // Run with a low threshold to ensure SOME processing happens, 
        // though with random data it's unpredictable how many boxes pass.
        // Let's use 0.01 threshold.
        run_postprocess(
            IntList<8, 16, 32>{},
            IntList<80, 40, 20>{},
            cls_ptrs,
            reg_ptrs,
            0.01f 
        );
        
        auto end_post = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_post = end_post - start_post;
        post_timings.push_back(duration_post.count());
        
        total_timings.push_back(duration_inf.count() + duration_post.count());
        
        if (i % 50 == 0 && i > 0) std::cout << "." << std::flush;
    }
    std::cout << std::endl;

    // 5. Statistics
    std::cout << "\n[Results]" << std::endl;
    std::cout << "  Iterations: " << iterations << std::endl;
    
    print_stats("Hailo Inference (Write+Read)", inf_timings);
    print_stats("CPU Post-processing", post_timings);
    print_stats("Total Latency", total_timings);
    
    // Calculate FPS based on Mean Total
    double mean_total = std::accumulate(total_timings.begin(), total_timings.end(), 0.0) / total_timings.size();
    std::cout << "  Estimated FPS (Serial): " << (1000.0 / mean_total) << std::endl;

    return 0;
}
