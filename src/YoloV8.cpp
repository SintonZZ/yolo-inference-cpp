#include "YoloV8.hpp"
#include "backends/YoloV8OpenCV.hpp"

#ifdef ENABLE_MNN
#include "backends/YoloV8MNN.hpp"
#endif

#include <iostream>
#include "spdlog/spdlog.h"

std::unique_ptr<YoloV8> YoloV8::create(const YoloV8Config& config) {
    switch (config.backend) {
        case InferenceBackendType::OPENCV:
            return std::make_unique<YoloV8OpenCV>(config);
        case InferenceBackendType::TENSORRT:
            // return std::make_unique<YoloV8TensorRT>(config);
            spdlog::warn("TensorRT backend not implemented yet, falling back to OpenCV.");
            return std::make_unique<YoloV8OpenCV>(config);
        case InferenceBackendType::MNN:
#ifdef ENABLE_MNN
            return std::make_unique<YoloV8MNN>(config);
#else
            spdlog::error("MNN backend is not compiled. Please recompile with -DENABLE_MNN=ON.");
            spdlog::warn("Falling back to OpenCV backend.");
            return std::make_unique<YoloV8OpenCV>(config);
#endif
        default:
            return std::make_unique<YoloV8OpenCV>(config);
    }
}
