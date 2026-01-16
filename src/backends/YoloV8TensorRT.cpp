#include "backends/YoloV8TensorRT.hpp"
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
// #include <opencv2/cudawarping.hpp>
// #include <opencv2/cudaarithm.hpp>
// #include <opencv2/cudaimgproc.hpp>
#include "spdlog/spdlog.h"

// Helper for CUDA error checking
#define CHECK_CUDA(status) \
    do { \
        if (status != 0) { \
            spdlog::error("CUDA failure: {}", status); \
            return; \
        } \
    } while (0)

YoloV8TensorRT::YoloV8TensorRT(const YoloV8Config& config) : config(config) {
    loadEngine(config.modelPath);
    
    // Allocate CPU output buffer
    cpu_output_buffer = new float[outputSize];
}

YoloV8TensorRT::~YoloV8TensorRT() {
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    delete[] cpu_output_buffer;
}

const std::vector<std::string>& YoloV8TensorRT::getClassNames() const {
    return config.classNames;
}

void YoloV8TensorRT::loadEngine(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        spdlog::error("Error reading engine file: {}", enginePath);
        return;
    }
    
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();
    
    runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engineData.data(), size));
    context = std::shared_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    
    // Setup dimensions and buffers
    // Assuming 0 is input, 1 is output. 
    // In real scenarios, should iterate binding indices and check names/io.
    
    // Input
    auto input_dims = engine->getBindingDimensions(0);
    inputSize = 1; 
    for(int i=0; i<input_dims.nbDims; i++) inputSize *= input_dims.d[i];
    
    // Output
    auto output_dims = engine->getBindingDimensions(1);
    outputSize = 1;
    for(int i=0; i<output_dims.nbDims; i++) outputSize *= output_dims.d[i];

    cudaMalloc(&buffers[0], inputSize * sizeof(float));
    cudaMalloc(&buffers[1], outputSize * sizeof(float));
    
    spdlog::info("TensorRT Engine loaded. Input Size: {}, Output Size: {}", inputSize, outputSize);
}

void YoloV8TensorRT::preprocess(const cv::Mat& image, float* gpu_input, const cv::Size& size) {
    // Basic implementation: CPU resize -> Float -> Normalize -> Upload
    // For higher performance, use cv::cuda::resize, etc.
    
    cv::Mat resized, float_img;
    cv::resize(image, resized, size);
    resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);
    
    // HWC -> CHW
    // OpenCV is HWC (BGR). YOLOv8 expects RGB or BGR depending on training, typically RGB.
    // Assuming model expects RGB.
    cv::cvtColor(float_img, float_img, cv::COLOR_BGR2RGB);
    
    // Split channels
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);
    
    // Copy to continuous buffer
    std::vector<float> input_data;
    input_data.reserve(size.width * size.height * 3);
    for(int i=0; i<3; ++i) {
        input_data.insert(input_data.end(), (float*)channels[i].data, (float*)channels[i].data + size.width * size.height);
    }
    
    cudaMemcpy(gpu_input, input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice);
}

std::vector<Object> YoloV8TensorRT::detect(const cv::Mat& image) {
    std::vector<Object> objects;
    if (!context) return objects;

    // 1. Preprocess
    preprocess(image, (float*)buffers[0], config.modelInputSize);
    
    // 2. Inference
    // enqueueV2 is async, but we will sync for simplicity
    void* bindings[] = {buffers[0], buffers[1]};
    context->executeV2(bindings);
    
    // 3. Copy Output back
    cudaMemcpy(cpu_output_buffer, buffers[1], outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 4. Post-process (Same as OpenCV/MNN logic)
    // Need to parse the raw output buffer.
    // YOLOv8 output: [Batch, 4 + Classes, Anchors] -> [1, 84, 8400]
    // Need to transpose if necessary.
    
    // Assuming output is [1, 84, 8400] (for 80 classes)
    // 4 bbox coords + 80 class probs
    int num_classes = config.classNames.size();
    int dimensions = 4 + num_classes;
    int rows = outputSize / dimensions; // e.g. 8400
    
    // The logic below is simplified and assumes specific output layout.
    // A robust implementation should check output dims.
    
    // Transpose [1, 84, 8400] -> [1, 8400, 84] to make iteration easier
    // Or just iterate carefully.
    
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    // Scale factors
    float x_factor = (float)image.cols / config.modelInputSize.width;
    float y_factor = (float)image.rows / config.modelInputSize.height;

    // Pointer to data
    float* data = cpu_output_buffer;
    
    // NOTE: TensorRT output layout might be different depending on how it was exported.
    // Standard YOLOv8 export usually gives [1, 84, 8400].
    // Let's iterate.
    
    for (int i = 0; i < rows; ++i) {
        // Accessing column i of the matrix [84, 8400]
        // data[0][i] -> x
        // data[1][i] -> y
        // data[2][i] -> w
        // data[3][i] -> h
        // data[4...][i] -> class scores
        
        // Wait, standard memory layout for [1, 84, 8400] is row-major.
        // So data[row * 8400 + col].
        // Row 0 is all x's. Row 1 is all y's.
        
        float* classes_scores = data + 4 * rows + i; // Start of class scores for this anchor
        
        // Find max class score
        float maxClassScore = -1.0;
        int classId = -1;
        
        for (int c = 0; c < num_classes; ++c) {
            // Step is 'rows' because we are moving down rows in the original [84, 8400] matrix
            float score = data[(4 + c) * rows + i];
            if (score > maxClassScore) {
                maxClassScore = score;
                classId = c;
            }
        }

        if (maxClassScore > config.scoreThreshold) {
            float x = data[0 * rows + i];
            float y = data[1 * rows + i];
            float w = data[2 * rows + i];
            float h = data[3 * rows + i];

            int left = int((x - 0.5 * w) * x_factor);
            int top = int((y - 0.5 * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);

            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(maxClassScore);
            classIds.push_back(classId);
        }
    }
    
    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, config.scoreThreshold, config.nmsThreshold, indices);

    for (int idx : indices) {
        Object obj;
        obj.rect = boxes[idx];
        obj.prob = confidences[idx];
        obj.label = classIds[idx];
        objects.push_back(obj);
    }
    
    return objects;
}
