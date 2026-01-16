#include "YoloV8.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include <nlohmann/json.hpp>

int main(int argc, char** argv) {
    std::string configPath = "../config/config.json";
    std::string imagePath;

    if (argc >= 2) {
        imagePath = argv[1];
    }
    
    // 如果命令行提供了配置文件路径，也可以支持
    if (argc >= 3) {
        configPath = argv[2];
    }

    // 预读取配置文件以获取日志路径
    std::string logPath = "logs/app.log";
    nlohmann::json j;
    bool configLoaded = false;
    
    std::ifstream f(configPath);
    if (f.is_open()) {
        try {
            f >> j;
            configLoaded = true;
            if (j.contains("log_path")) {
                logPath = j["log_path"].get<std::string>();
            }
        } catch (const nlohmann::json::exception& e) {
            std::cerr << "Warning: Failed to parse config file for log path: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "Warning: Could not open config file at " << configPath << " for log path configuration." << std::endl;
    }

    // 初始化日志
    try {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::info);

        // 确保日志目录存在
        std::filesystem::path logDir = std::filesystem::path(logPath).parent_path();
        if (!logDir.empty() && !std::filesystem::exists(logDir)) {
            std::filesystem::create_directories(logDir);
        }

        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(logPath, false);
        file_sink->set_level(spdlog::level::debug);

        std::vector<spdlog::sink_ptr> sinks {console_sink, file_sink};
        auto logger = std::make_shared<spdlog::logger>("multi_sink", sinks.begin(), sinks.end());
        spdlog::set_default_logger(logger);
        spdlog::set_level(spdlog::level::debug);
        spdlog::flush_on(spdlog::level::info);
    } catch (const spdlog::spdlog_ex& ex) {
        std::cerr << "Log initialization failed: " << ex.what() << std::endl;
        return -1;
    }

    spdlog::info("Application started.");

    if (imagePath.empty()) {
        spdlog::error("Usage: {} <image_path> [config_path]", argv[0]);
        spdlog::error("Example: {} bus.jpg ../config/config.json", argv[0]);
        return -1;
    }

    // 读取配置文件
    if (!configLoaded) {
        spdlog::error("Error: Could not open or parse config file at {}", configPath);
        return -1;
    }

    spdlog::info("Loading config from {}...", configPath);

    YoloV8Config config;
    try {
        config.modelPath = j["model_path"].get<std::string>();
        config.scoreThreshold = j["score_threshold"].get<float>();
        config.nmsThreshold = j["nms_threshold"].get<float>();
        int width = j["input_width"].get<int>();
        int height = j["input_height"].get<int>();
        config.modelInputSize = cv::Size(width, height);

        for (const auto& item : j["class_names"]) {
            config.classNames.push_back(item.get<std::string>());
        }

        // 解析后端类型
        if (j.contains("backend")) {
            std::string backendStr = j["backend"].get<std::string>();
            if (backendStr == "MNN") {
                config.backend = InferenceBackendType::MNN;
                spdlog::info("Using Backend: MNN");
            } else if (backendStr == "TENSORRT") {
                config.backend = InferenceBackendType::TENSORRT;
                spdlog::info("Using Backend: TensorRT");
            } else {
                config.backend = InferenceBackendType::OPENCV;
                spdlog::info("Using Backend: OpenCV");
            }
        } else {
             spdlog::info("Using Default Backend: OpenCV");
        }
    } catch (const nlohmann::json::exception& e) {
        spdlog::error("JSON parsing error: {}", e.what());
        return -1;
    }

    // 检查模型路径，如果不是绝对路径且 config 位于上级目录，可能需要调整
    // 这里简单处理，假设 config.json 中的 model_path 是相对于运行目录或绝对路径
    // 如果 config.json 在上级目录 (../config.json)，而 model 在 ../yolov8n.onnx
    // 我们可能需要根据 config 文件的位置来解析 model path，或者约定必须写对路径
    // 简单起见，我们在 json 里写文件名，然后在代码里拼接，或者在 json 里写相对路径 (相对于执行路径)
    // 这里的实现假设 json 里写的是相对于执行路径的路径。
    // 为了方便用户，如果 config.json 里写的是 "yolov8n.onnx"，我们假设它和 config.json 在同一目录
    // 我们尝试加上 ../ 前缀如果文件不存在
    // {
    //     std::ifstream f(config.modelPath);
    //     if (!f.good()) {
    //         std::string tryPath = "../" + config.modelPath;
    //         std::ifstream f2(tryPath);
    //         if (f2.good()) {
    //             config.modelPath = tryPath;
    //         }
    //     }
    // }

    // 读取图片
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        spdlog::error("Error: Could not read image at {}", imagePath);
        return -1;
    }

    // 实例化 YoloV8 类 (使用工厂方法)
    spdlog::info("Loading model from {}...", config.modelPath);
    try {
        // 创建推理引擎实例
        auto yolo = YoloV8::create(config);

        // 执行推理 (Warmup)
        // spdlog::info("Warmup...");
        // yolo->detect(cv::Mat::zeros(config.modelInputSize, CV_8UC3));

        spdlog::info("Inference...");
        std::vector<Object> objects = yolo->detect(image);
        spdlog::info("Detected {} objects.", objects.size());

        // 可视化结果
        const auto& classNames = yolo->getClassNames();
        for (const auto& obj : objects) {
            cv::rectangle(image, obj.rect, cv::Scalar(0, 255, 0), 2);

            std::string labelString = "Class " + std::to_string(obj.label);
            if (obj.label >= 0 && obj.label < classNames.size()) {
                labelString = classNames[obj.label];
            }
            
            std::string label = labelString + ": " + std::to_string(obj.prob).substr(0, 4);
            
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            
            int top = std::max(obj.rect.y, labelSize.height);
            cv::rectangle(image, cv::Point(obj.rect.x, top - labelSize.height),
                        cv::Point(obj.rect.x + labelSize.width, top + baseLine), cv::Scalar(0, 255, 0), cv::FILLED);
            
            cv::putText(image, label, cv::Point(obj.rect.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }

        // 显示结果
        cv::imshow("YOLOv8 Detection", image);
        cv::waitKey(0);
    } catch (const std::exception& e) {
        spdlog::error("Exception occurred: {}", e.what());
        return -1;
    }

    return 0;
}
