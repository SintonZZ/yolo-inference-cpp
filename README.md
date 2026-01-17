# YOLOv8 Inference C++ (Multi-Backend)

这是一个基于 C++ 的高性能 YOLOv8 目标检测推理项目，支持多种推理后端（OpenCV DNN、MNN、TensorRT）。项目设计灵活，易于集成到工业级应用中。

## ✨ 主要特性

*   **多后端支持**: 
    *   **OpenCV DNN**: 默认后端，无额外依赖，开箱即用。
    *   **MNN**: 阿里巴巴开源的轻量级推理引擎，适合边缘设备部署 (需编译时开启)。
    *   **TensorRT**: 针对 NVIDIA GPU 的高性能推理 (需编译时开启)。
*   **高性能**: 
    *   内存复用机制，减少推理过程中的内存分配。
    *   C++17 标准编写，利用现代 C++ 特性。
*   **易用性**:
    *   通过 JSON 配置文件灵活调整参数（模型路径、阈值、后端类型）。
    *   完善的日志系统 (spdlog)，支持控制台和文件日志。
*   **工程化**:
    *   工厂模式设计，便于扩展新的推理引擎。
    *   模块化结构，清晰的代码组织。

## 📂 项目结构

```text
├── CMakeLists.txt              # CMake 构建脚本
├── 3rdparty/                   # 第三方库 (spdlog 等)
├── config/
│   └── config.json             # 配置文件 (模型路径、阈值、后端选择)
├── models/
│   └── yolov8n.onnx            # (示例) ONNX 模型文件
├── include/
│   ├── YoloV8.hpp              # 推理引擎抽象基类
│   └── backends/               # 各推理后端头文件
├── src/
│   ├── main.cpp                # 程序入口
│   ├── YoloV8.cpp              # 工厂方法实现
│   └── backends/               # 各推理后端实现
└── logs/                       # 运行日志目录
```

## 🛠️ 环境依赖

*   **OS**: Linux (Ubuntu 20.04/22.04 推荐)
*   **Compiler**: GCC/Clang (支持 C++17)
*   **CMake**: >= 3.15
*   **OpenCV**: >= 4.8.0 (推荐从源码编译以获得最佳性能)
*   **nlohmann_json**: 用于解析 JSON 配置
*   **spdlog**: 用于日志记录 (通常包含在 3rdparty 或系统安装)
*   **MNN** (可选): 如果需要 MNN 后端支持
*   **CUDA & TensorRT** (可选): 如果需要 TensorRT 后端支持

## 🚀 构建指南

### 1. 基础构建 (仅 OpenCV 后端)

默认情况下，项目仅编译 OpenCV 后端，不需要安装 MNN。

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### 2. 启用 MNN 后端支持

若要使用 MNN 推理引擎，请先安装 MNN 库，然后在编译时开启 `ENABLE_MNN` 选项。

**前提: 安装 MNN**
```bash
git clone https://github.com/alibaba/MNN.git
cd MNN
./schema/generate.sh
mkdir build && cd build
cmake -D MNN_BUILD_QUANTOOLS=OFF -D MNN_BUILD_CONVERTER=on ..
make -j4
sudo make install
```

**编译项目**
```bash
mkdir build
cd build
cmake -DENABLE_MNN=ON ..
make -j$(nproc)
```

### 3. 启用 TensorRT 后端支持

**前提: 已安装 CUDA, cuDNN 和 TensorRT**

```bash
mkdir build
cd build
cmake -DENABLE_TENSORRT=ON ..
make -j$(nproc)
```

**注意**: TensorRT 后端需要加载 `.engine` (或 `.plan`) 文件。您可以使用 `trtexec` 工具将 ONNX 转换为 Engine 文件。

```bash
# 使用 trtexec 转换 (TensorRT 自带工具)
/usr/src/tensorrt/bin/trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine
```

然后在 `config.json` 中设置 `"backend": "TENSORRT"` 和 `"model_path": "../models/yolov8n.engine"`。

## 🏃 使用方法

### 1. 准备模型

项目支持 ONNX 模型。您可以使用 `yolo` 命令行工具将官方 YOLOv8 模型导出为 ONNX。

```bash
cd models
# 安装 ultralytics
pip install ultralytics
# 导出模型 (以 yolov8n 为例)
yolo export model=yolov8n.pt format=onnx opset=12
# 生成 yolov8n.onnx
```

*如果使用 MNN 后端，建议使用 MNNConvert 工具将 ONNX 转为 `.mnn` 格式以获得更快的加载速度。*

```bash
# 使用 MNNConvert 工具 (需自行编译 MNN 获得)
./MNNConvert -f ONNX --modelFile yolov8n.onnx --MNNModel yolov8n.mnn --bizCode biz
```

### 2. 运行推理

编译完成后，可执行文件位于 `build` 目录。

```bash
# 用法: ./yolov8_app <输入源> [配置文件路径]

# 示例 1: 图片推理 (使用默认配置)
./yolov8_app ../assets/bus.jpg

# 示例 2: Webcam 实时推理 (使用默认配置)
# 0 代表 /dev/video0，程序会自动开启多线程模式
./yolov8_app 0

# 示例 3: 指定配置文件
./yolov8_app 0 ../config/config.json
```

### 3. 配置文件说明 (`config/config.json`)

```json
{
    "log_path": "../logs/app.log",      // 日志文件路径
    "backend": "OPENCV",                // 后端选择: "OPENCV", "MNN", "TENSORRT"
    "model_path": "../models/yolov8n.onnx", // 模型路径
    "score_threshold": 0.25,            // 置信度阈值
    "nms_threshold": 0.45,              // NMS 阈值
    "input_width": 640,                 // 模型输入宽度
    "input_height": 640,                // 模型输入高度
    "class_names": ["person", "bicycle", ...] // 类别名称列表
}
```

*   **切换后端**: 修改 `"backend"` 字段为 `"MNN"` 或 `"TENSORRT"` 即可切换引擎（需确保编译时已开启相应支持）。
*   **日志**: 程序运行日志会同时输出到控制台和 `log_path` 指定的文件中（追加模式）。

## 📊 性能优化

*   **内存复用**: 推理类内部维护了输入/输出缓冲区，避免了每一帧推理都进行内存申请和释放。
*   **Warmup**: 推理引擎初始化时可进行预热（代码中已预留接口），以消除首次推理的延迟。

## ✅ TODO List

*   [x] **多媒体支持**: 支持视频文件和 Webcam 实时推理输入。
*   [x] **并发优化**: 实现多线程推理管道，提高吞吐量。
*   [ ] **INT8 量化**: 完善 TensorRT INT8 校准与推理流程。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进本项目，例如添加新的后端支持或优化前处理逻辑。

