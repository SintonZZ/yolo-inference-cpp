#pragma once

#include "YoloV8.hpp"
#include <MNN/Interpreter.hpp>
#include <MNN/ImageProcess.hpp>
#include <MNN/Tensor.hpp>

class YoloV8MNN : public YoloV8 {
public:
    YoloV8MNN(const YoloV8Config& config);
    ~YoloV8MNN() override;

    std::vector<Object> detect(const cv::Mat& image) override;
    const std::vector<std::string>& getClassNames() const override;

private:
    YoloV8Config config;
    std::shared_ptr<MNN::Interpreter> net;
    MNN::Session* session = nullptr;
    MNN::Tensor* inputTensor = nullptr;

    // 预分配的缓冲区 (Memory Reuse)
    std::shared_ptr<MNN::Tensor> inputHostTensor; // 用于拷贝数据到 Device
    
    // Letterbox 预处理
    void letterbox(const cv::Mat& source, cv::Mat& dst, float& scale, std::vector<int>& pad);
    
    // 预分配的缓冲区
    cv::Mat inputImage;
    
    // 预分配的结果容器
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<int> indices;
};
