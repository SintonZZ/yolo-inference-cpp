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
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <chrono>

// 线程安全的队列实现
template <typename T>
class SafeQueue {
public:
    SafeQueue(size_t maxSize = 5) : maxSize_(maxSize) {}

    void push(const T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        // 如果队列满了，丢弃最旧的帧（对于实时视频，只需最新帧）
        if (queue_.size() >= maxSize_) {
            queue_.pop_front();
        }
        queue_.push_back(item);
        cond_.notify_one();
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this] { return !queue_.empty() || stop_; });
        if (queue_.empty() && stop_) return false;
        
        item = queue_.front();
        queue_.pop_front();
        return true;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cond_.notify_all();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.clear();
    }

private:
    std::deque<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
    size_t maxSize_;
    bool stop_ = false;
};

struct InferenceResult {
    cv::Mat frame;
    std::vector<Object> objects;
    double inferenceTime;
};

// 辅助函数：绘制检测结果
void draw_results(cv::Mat& image, const std::vector<Object>& objects, const std::vector<std::string>& classNames) {
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
}

// 视频捕捉线程函数
void capture_worker(int device_id, SafeQueue<cv::Mat>& frame_queue, std::atomic<bool>& running) {
    // 优先尝试 V4L2 后端，这是 Linux 下最常用的
    cv::VideoCapture cap(device_id, cv::CAP_V4L2);
    
    if (!cap.isOpened()) {
        spdlog::warn("Cannot open webcam {} with V4L2, trying default backend...", device_id);
        cap.open(device_id, cv::CAP_ANY);
    }

    if (!cap.isOpened()) {
        spdlog::error("Cannot open webcam {}. Please check device connection and permissions.", device_id);
        running = false;
        frame_queue.stop();
        return;
    }

    // 设置 MJPG 格式以提高帧率兼容性
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    // 设置常见分辨率
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);

    spdlog::info("Webcam {} started. Backend: {}", device_id, cap.getBackendName());
    
    cv::Mat frame;
    while (running) {
        cap >> frame;
        if (frame.empty()) {
            spdlog::warn("Empty frame from webcam");
            // 不要立即退出，可能是暂时丢帧
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        // 使用 clone 确保数据独立
        frame_queue.push(frame.clone());
        
        // 简单的帧率控制，避免占用过多 CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    cap.release();
}

// 推理线程函数
void inference_worker(YoloV8* yolo, SafeQueue<cv::Mat>& frame_queue, SafeQueue<InferenceResult>& result_queue, std::atomic<bool>& running) {
    cv::Mat frame;
    while (running) {
        if (frame_queue.pop(frame)) {
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<Object> objects = yolo->detect(frame);
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

            result_queue.push({frame, objects, time_ms});
        }
    }
}

int main(int argc, char** argv) {
    std::string configPath = "../config/config.json";
    std::string inputSource;

    if (argc >= 2) {
        inputSource = argv[1];
    }
    
    if (argc >= 3) {
        configPath = argv[2];
    }

    // 初始化日志
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
    }

    try {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::info);

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

    if (inputSource.empty()) {
        spdlog::error("Usage: {} <image_path/cam_id> [config_path]", argv[0]);
        spdlog::error("Example: {} bus.jpg ../config/config.json", argv[0]);
        spdlog::error("Example: {} 0 ../config/config.json", argv[0]);
        return -1;
    }

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

    // 创建推理引擎
    spdlog::info("Loading model from {}...", config.modelPath);
    std::unique_ptr<YoloV8> yolo;
    try {
        yolo = YoloV8::create(config);
    } catch (const std::exception& e) {
        spdlog::error("Failed to create YoloV8 engine: {}", e.what());
        return -1;
    }

    // 判断输入源是否为 Webcam (数字)
    bool isWebcam = false;
    int webcamId = 0;
    try {
        webcamId = std::stoi(inputSource);
        isWebcam = true;
    } catch (...) {
        isWebcam = false;
    }

    if (isWebcam) {
        spdlog::info("Starting Webcam Inference Mode (Multi-threaded)");
        
        SafeQueue<cv::Mat> frameQueue(2); // 限制队列大小以减少延迟
        SafeQueue<InferenceResult> resultQueue(5);
        std::atomic<bool> running(true);

        // 启动线程
        std::thread captureThread(capture_worker, webcamId, std::ref(frameQueue), std::ref(running));
        std::thread inferenceThread(inference_worker, yolo.get(), std::ref(frameQueue), std::ref(resultQueue), std::ref(running));

        spdlog::info("Press ESC to exit.");

        InferenceResult result;
        while (running) {
            if (resultQueue.pop(result)) {
                // 绘制和显示
                draw_results(result.frame, result.objects, yolo->getClassNames());
                
                // 显示推理时间
                std::string timeStr = "Time: " + std::to_string(result.inferenceTime).substr(0, 5) + " ms";
                cv::putText(result.frame, timeStr, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

                cv::imshow("YOLOv8 Webcam", result.frame);
                
                int key = cv::waitKey(1);
                if (key == 27) { // ESC
                    running = false;
                    frameQueue.stop();
                    resultQueue.stop();
                }
            }
        }

        // 等待线程结束
        if (captureThread.joinable()) captureThread.join();
        if (inferenceThread.joinable()) inferenceThread.join();

    } else {
        // 单图模式
        spdlog::info("Starting Single Image Mode");
        cv::Mat image = cv::imread(inputSource);
        if (image.empty()) {
            spdlog::error("Error: Could not read image at {}", inputSource);
            return -1;
        }

        try {
            spdlog::info("Inference...");
            std::vector<Object> objects = yolo->detect(image);
            spdlog::info("Detected {} objects.", objects.size());

            draw_results(image, objects, yolo->getClassNames());

            cv::imshow("YOLOv8 Detection", image);
            cv::waitKey(0);
        } catch (const std::exception& e) {
            spdlog::error("Exception occurred: {}", e.what());
            return -1;
        }
    }

    return 0;
}
