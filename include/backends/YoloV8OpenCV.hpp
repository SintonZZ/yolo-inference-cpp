#pragma once

#include "YoloV8.hpp"
#include <opencv2/dnn.hpp>

class YoloV8OpenCV : public YoloV8 {
public:
    YoloV8OpenCV(const YoloV8Config& config);
    ~YoloV8OpenCV() override = default;

    std::vector<Object> detect(const cv::Mat& image) override;
    const std::vector<std::string>& getClassNames() const override;

private:
    YoloV8Config config;
    cv::dnn::Net net;

    // Letterbox 预处理
    void letterbox(const cv::Mat& source, cv::Mat& dst, float& scale, std::vector<int>& pad);

    // 预分配的缓冲区 (Memory Reuse)
    cv::Mat inputImage;
    cv::Mat blob;
    std::vector<cv::Mat> outputs;
    cv::Mat transposeData;
    
    // 预分配的结果容器
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<int> indices;
};
