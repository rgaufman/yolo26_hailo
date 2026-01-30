/*
   libnpy.hpp
   A C++ library for reading .npy files
   
   Based on the format specification:
   https://numpy.org/neps/nep-0001-npy-format.html
*/

#ifndef LIBNPY_HPP
#define LIBNPY_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <regex>
#include <unordered_map>
#include <cstdint>

namespace npy {

struct NpyHeader {
    std::string dtype;
    bool fortran_order;
    std::vector<unsigned long> shape;
};

inline void parse_header(std::string header, NpyHeader& out_header) {
    // Basic parsing of the dictionary string
    // This is a simplified parser
    
    // Check fortran order
    size_t loc = header.find("'fortran_order':");
    if (loc != std::string::npos) {
        if (header.substr(loc + 16, 4) == "True")
            out_header.fortran_order = true;
        else
            out_header.fortran_order = false;
    }

    // Check descr/dtype
    loc = header.find("'descr':");
    if (loc != std::string::npos) {
        size_t start = header.find("'", loc + 8);
        size_t end = header.find("'", start + 1);
        out_header.dtype = header.substr(start + 1, end - start - 1);
    }

    // Check shape
    loc = header.find("'shape':");
    if (loc != std::string::npos) {
        size_t start = header.find("(", loc + 8);
        size_t end = header.find(")", start + 1);
        std::string shape_str = header.substr(start + 1, end - start - 1);
        
        // Remove trailing comma if tuple has 1 element
        if (!shape_str.empty() && shape_str.back() == ',')
            shape_str.pop_back();
            
        // Split by comma
        if (!shape_str.empty()) {
            size_t prev = 0, current;
            while ((current = shape_str.find(",", prev)) != std::string::npos) {
                std::string dim = shape_str.substr(prev, current - prev);
                out_header.shape.push_back(std::stoul(dim));
                prev = current + 1;
            }
            out_header.shape.push_back(std::stoul(shape_str.substr(prev)));
        }
    }
}

template<typename T>
std::vector<T> load_npy(const std::string& filename, std::vector<unsigned long>& shape) {
    std::ifstream stream(filename, std::ifstream::binary);
    if (!stream) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    // Check magic
    char magic[6];
    stream.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") {
        throw std::runtime_error("Invalid NPY file: " + filename);
    }

    // Read version
    unsigned char major, minor;
    stream.read(reinterpret_cast<char*>(&major), 1);
    stream.read(reinterpret_cast<char*>(&minor), 1);

    // Read header length
    unsigned short header_len;
    stream.read(reinterpret_cast<char*>(&header_len), 2);
    
    // Read header
    std::string header(header_len, ' ');
    stream.read(&header[0], header_len);
    
    NpyHeader npy_header;
    parse_header(header, npy_header);
    
    shape = npy_header.shape;

    // Check dtype (simplified check)
    // <f4 means little endian float32
    if (npy_header.dtype != "<f4" && npy_header.dtype != "'<f4'") {
        // Warning: Type mismatch possibility, but continuing for this specific use case
        // std::cerr << "Warning: Expected <f4 dtype, got " << npy_header.dtype << std::endl;
    }

    // Read data
    stream.seekg(0, std::ios::end);
    size_t file_size = stream.tellg();
    size_t data_start = 6 + 2 + 2 + header_len;
    size_t num_bytes = file_size - data_start;
    size_t num_elements = num_bytes / sizeof(T);
    
    std::vector<T> data(num_elements);
    stream.seekg(data_start);
    stream.read(reinterpret_cast<char*>(data.data()), num_bytes);
    
    return data;
}

} // namespace npy

#endif // LIBNPY_HPP
