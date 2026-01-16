#pragma once

#include "YoloV8.hpp"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <memory>
#include <iostream>

// TensorRT Logger
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};

class YoloV8TensorRT : public YoloV8 {
public:
    YoloV8TensorRT(const YoloV8Config& config);
    ~YoloV8TensorRT() override;

    std::vector<Object> detect(const cv::Mat& image) override;
    const std::vector<std::string>& getClassNames() const override;

private:
    void loadEngine(const std::string& enginePath);
    void preprocess(const cv::Mat& image, float* gpu_input, const cv::Size& size);

    YoloV8Config config;
    Logger logger;
    
    // TensorRT objects
    std::shared_ptr<nvinfer1::IRuntime> runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::shared_ptr<nvinfer1::IExecutionContext> context;

    // GPU buffers
    void* buffers[2]; // 0: input, 1: output
    float* cpu_output_buffer;
    
    // Dimensions
    nvinfer1::Dims inputDims;
    nvinfer1::Dims outputDims;
    size_t inputSize;
    size_t outputSize;
};
