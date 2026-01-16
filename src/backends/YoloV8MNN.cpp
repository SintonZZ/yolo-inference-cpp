#include "backends/YoloV8MNN.hpp"
#include "spdlog/spdlog.h"
#include <iostream>

YoloV8MNN::YoloV8MNN(const YoloV8Config& config) : config(config) {
    // 1. 加载模型
    net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(config.modelPath.c_str()));
    if (!net) {
        spdlog::error("Error loading MNN model: {}", config.modelPath);
        throw std::runtime_error("Failed to load MNN model");
    }

    // 2. 配置 Session
    MNN::ScheduleConfig scheduleConfig;
    scheduleConfig.type = MNN_FORWARD_CPU;
    scheduleConfig.numThread = 4;
    
    // 如果需要支持其他后端（如 OpenCL/Vulkan），可以在这里配置
    // scheduleConfig.type = MNN_FORWARD_OPENCL;

    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_High;
    scheduleConfig.backendConfig = &backendConfig;

    session = net->createSession(scheduleConfig);
    inputTensor = net->getSessionInput(session, nullptr);
    
    // 3. 预分配 Host Tensor 用于输入数据拷贝
    // MNN 输入通常是 NCHW float
    // YOLOv8 输入通常是 1x3x640x640
    inputHostTensor.reset(new MNN::Tensor(inputTensor, MNN::Tensor::CAFFE)); // CAFFE layout is NCHW
}

YoloV8MNN::~YoloV8MNN() {
    if (net && session) {
        net->releaseSession(session);
    }
}

const std::vector<std::string>& YoloV8MNN::getClassNames() const {
    return config.classNames;
}

// 复用 OpenCV 的 Letterbox 逻辑
void YoloV8MNN::letterbox(const cv::Mat& source, cv::Mat& dst, float& scale, std::vector<int>& pad) {
    int col = source.cols;
    int row = source.rows;
    
    // 计算缩放比例
    scale = std::min((float)config.modelInputSize.width / col, (float)config.modelInputSize.height / row);

    int newW = round(col * scale);
    int newH = round(row * scale);

    cv::Mat resized;
    cv::resize(source, resized, cv::Size(newW, newH));

    // 计算填充
    int dw = config.modelInputSize.width - newW;
    int dh = config.modelInputSize.height - newH;

    int top = dh / 2;
    int bottom = dh - top;
    int left = dw / 2;
    int right = dw - left;

    // 填充颜色通常为灰色 114
    cv::copyMakeBorder(resized, dst, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    pad = {top, bottom, left, right};
}

std::vector<Object> YoloV8MNN::detect(const cv::Mat& image) {
    if (image.empty()) {
        spdlog::warn("Input image is empty!");
        return {};
    }

    float scale;
    std::vector<int> pad;

    // 1. 预处理 (Letterbox)
    letterbox(image, inputImage, scale, pad);

    // 2. 准备输入数据
    // OpenCV Mat (HWC, BGR) -> MNN Tensor (NCHW, RGB, float 0-1)
    
    // 将 OpenCV Mat 转换为 float 并归一化
    cv::Mat floatImage;
    inputImage.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);
    
    // BGR -> RGB
    cv::cvtColor(floatImage, floatImage, cv::COLOR_BGR2RGB);

    // 填充 inputHostTensor
    // MNN 的 Tensor::CAFFE 格式是 NCHW
    // 我们需要手动将 HWC 转为 NCHW
    int h = floatImage.rows;
    int w = floatImage.cols;
    int c = floatImage.channels();
    
    // 获取 host tensor 的数据指针
    float* destPtr = inputHostTensor->host<float>();
    
    // HWC to NCHW
    // split channels
    std::vector<cv::Mat> channels(c);
    cv::split(floatImage, channels);
    
    // Copy to MNN buffer (Planar)
    int planeSize = h * w;
    for (int i = 0; i < c; ++i) {
        memcpy(destPtr + i * planeSize, channels[i].data, planeSize * sizeof(float));
    }

    // Copy host tensor to device tensor
    inputTensor->copyFromHostTensor(inputHostTensor.get());

    // 3. 推理
    net->runSession(session);

    // 4. 获取输出
    // YOLOv8 输出: [1, 84, 8400]
    MNN::Tensor* outputTensor = net->getSessionOutput(session, nullptr);
    
    // 拷贝输出到 host
    std::shared_ptr<MNN::Tensor> outputHostTensor(new MNN::Tensor(outputTensor, MNN::Tensor::CAFFE));
    outputTensor->copyToHostTensor(outputHostTensor.get());
    
    float* data = outputHostTensor->host<float>();
    
    // 获取维度
    // shape: [1, 84, 8400]
    auto shape = outputHostTensor->shape();
    int batch = shape[0];
    int rows = shape[1]; // 84 (4 box + 80 classes)
    int cols = shape[2]; // 8400 anchors

    // 清空结果
    classIds.clear();
    confidences.clear();
    boxes.clear();
    indices.clear();

    // 5. 后处理 (解码)
    // 注意：MNN 输出布局是 NCHW (1x84x8400)，而 OpenCV DNN 输出我们之前手动转置过
    // 这里 data 布局是:
    // [cx0, cx1, ..., cx8399]
    // [cy0, cy1, ..., cy8399]
    // ...
    // [prob79_0, ..., prob79_8399]
    
    // 我们需要遍历每一列 (col) 作为单独的 anchor
    for (int i = 0; i < cols; ++i) {
        // 计算当前 anchor 的数据在 flat array 中的偏移
        // data[row * cols + i] 访问第 row 行，第 i 列
        
        // 边界框
        float cx = data[0 * cols + i];
        float cy = data[1 * cols + i];
        float w  = data[2 * cols + i];
        float h  = data[3 * cols + i];
        
        // 查找最大类别概率
        float maxScore = -1.0;
        int maxClassId = -1;
        
        for (int j = 4; j < rows; ++j) {
            float score = data[j * cols + i];
            if (score > maxScore) {
                maxScore = score;
                maxClassId = j - 4;
            }
        }

        if (maxScore > config.scoreThreshold) {
            // 还原坐标
            float left = cx - w / 2;
            float top = cy - h / 2;

            left -= pad[2];
            top -= pad[0];

            left /= scale;
            top /= scale;
            w /= scale;
            h /= scale;

            classIds.push_back(maxClassId);
            confidences.push_back(maxScore);
            boxes.push_back(cv::Rect(round(left), round(top), round(w), round(h)));
        }
    }

    // NMS
    cv::dnn::NMSBoxes(boxes, confidences, config.scoreThreshold, config.nmsThreshold, indices);

    std::vector<Object> objects;
    objects.reserve(indices.size());
    for (int idx : indices) {
        Object obj;
        obj.rect = boxes[idx];
        obj.label = classIds[idx];
        obj.prob = confidences[idx];
        objects.push_back(obj);
    }

    return objects;
}
