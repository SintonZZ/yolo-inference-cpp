#include "backends/YoloV8OpenCV.hpp"
#include "spdlog/spdlog.h"

YoloV8OpenCV::YoloV8OpenCV(const YoloV8Config& config) : config(config) {
    try {
        net = cv::dnn::readNetFromONNX(config.modelPath);
    } catch (const cv::Exception& e) {
        spdlog::error("Error loading model: {}", e.what());
        throw;
    }

    // 尝试使用 CUDA
    try {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        spdlog::info("Using CUDA backend.");
    } catch (...) {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        spdlog::warn("CUDA not available, falling back to CPU.");
    }
}

const std::vector<std::string>& YoloV8OpenCV::getClassNames() const {
    return config.classNames;
}

void YoloV8OpenCV::letterbox(const cv::Mat& source, cv::Mat& dst, float& scale, std::vector<int>& pad) {
    int col = source.cols;
    int row = source.rows;
    int maxLen = MAX(col, row);
    
    // 计算缩放比例，取短边进行缩放
    scale = std::min((float)config.modelInputSize.width / col, (float)config.modelInputSize.height / row);

    int newW = round(col * scale);
    int newH = round(row * scale);

    cv::Mat resized;
    cv::resize(source, resized, cv::Size(newW, newH));

    // 计算填充
    int dw = config.modelInputSize.width - newW;
    int dh = config.modelInputSize.height - newH;

    // 居中填充
    int top = dh / 2;
    int bottom = dh - top;
    int left = dw / 2;
    int right = dw - left;

    // 填充颜色通常为灰色 114
    cv::copyMakeBorder(resized, dst, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // 保存参数用于恢复坐标
    pad = {top, bottom, left, right};
}

std::vector<Object> YoloV8OpenCV::detect(const cv::Mat& image) {
    if (image.empty()) {
        spdlog::warn("Input image is empty!");
        return {};
    }

    float scale;
    std::vector<int> pad; // top, bottom, left, right

    // 1. 预处理
    letterbox(image, inputImage, scale, pad);

    // 2. 转换为 Blob
    // YOLOv8 训练时归一化到了 [0,1]，所以 scale 是 1/255.0
    // swapRB=true 因为 OpenCV 是 BGR，模型通常需要 RGB
    cv::dnn::blobFromImage(inputImage, blob, 1.0 / 255.0, config.modelInputSize, cv::Scalar(), true, false);

    // 3. 推理
    net.setInput(blob);
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // 4. 后处理
    // YOLOv8 输出维度通常是 [1, 84, 8400] (Batch, 4+80, Anchors)
    cv::Mat outputData = outputs[0]; 
    
    // 获取维度信息
    int dimensions = outputData.dims;
    int rows = outputData.size[1]; // 84
    int cols = outputData.size[2]; // 8400

    // 转置: 将 1x84x8400 转换为 8400x84 的矩阵以便处理
    // OpenCV 的 Mat 如果是多维的，访问比较麻烦
    // 我们将其重塑为 2D 矩阵 [84, 8400]
    outputData = outputData.reshape(1, rows);
    cv::transpose(outputData, transposeData); // 现在是 [8400, 84]

    float* data = (float*)transposeData.data;

    // 清空结果容器
    classIds.clear();
    confidences.clear();
    boxes.clear();
    indices.clear();

    // 遍历每一行 (每一个 anchor)
    for (int i = 0; i < cols; ++i) {
        float* rowPtr = data + i * rows; // 指向第 i 个检测结果的起始位置

        // 前 4 个值是边界框坐标 (cx, cy, w, h)
        float cx = rowPtr[0];
        float cy = rowPtr[1];
        float w = rowPtr[2];
        float h = rowPtr[3];

        // 后面的值是类别概率
        // 找到最大概率的类别
        float maxScore = -1.0;
        int maxClassId = -1;
        
        // 84 - 4 = 80 个类别
        for (int j = 4; j < rows; ++j) {
            if (rowPtr[j] > maxScore) {
                maxScore = rowPtr[j];
                maxClassId = j - 4;
            }
        }

        if (maxScore > config.scoreThreshold) {
            // 还原坐标到原图
            // 1. 去除 padding
            // 2. 缩放回原图
            
            // 注意：cx, cy, w, h 是在 letterbox 后的图像坐标系下的
            
            // 转换回左上角坐标
            float left = cx - w / 2;
            float top = cy - h / 2;

            // 还原 padding
            left -= pad[2]; // pad left
            top -= pad[0]; // pad top

            // 还原缩放
            left /= scale;
            top /= scale;
            w /= scale;
            h /= scale;

            classIds.push_back(maxClassId);
            confidences.push_back(maxScore);
            boxes.push_back(cv::Rect(round(left), round(top), round(w), round(h)));
        }
    }

    // NMS 非极大值抑制
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
