#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>

// 定义数据结构
struct Object {
    cv::Rect rect;
    int label;
    float prob;
};

// 推理后端类型
enum class InferenceBackendType {
    OPENCV,
    TENSORRT, // 预留
    MNN       // 预留
};

struct YoloV8Config {
    std::string modelPath;
    float scoreThreshold;
    float nmsThreshold;
    cv::Size modelInputSize;
    std::vector<std::string> classNames;
    
    // 指定后端类型，默认为 OpenCV
    InferenceBackendType backend = InferenceBackendType::OPENCV;
};

// 抽象基类
class YoloV8 {
public:
    virtual ~YoloV8() = default;

    // 工厂方法：根据配置创建具体的推理后端实例
    static std::unique_ptr<YoloV8> create(const YoloV8Config& config);

    // 纯虚接口
    virtual std::vector<Object> detect(const cv::Mat& image) = 0;
    virtual const std::vector<std::string>& getClassNames() const = 0;
};
