cmake_minimum_required(VERSION 3.15)
project(YoloV8Inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV 包
find_package(OpenCV REQUIRED)

# 选项：是否启用 MNN 后端
option(ENABLE_MNN "Enable MNN backend support" OFF)

if (ENABLE_MNN)
    # 查找 MNN 包
    find_path(MNN_INCLUDE_DIRS NAMES MNN/Interpreter.hpp)
    find_library(MNN_LIBS NAMES MNN)

    if (MNN_INCLUDE_DIRS AND MNN_LIBS)
        message(STATUS "Found MNN: ${MNN_INCLUDE_DIRS}")
        include_directories(${MNN_INCLUDE_DIRS})
        add_definitions(-DENABLE_MNN)
        list(APPEND BACKEND_SOURCES src/backends/YoloV8MNN.cpp)
    else()
        message(FATAL_ERROR "MNN requested (ENABLE_MNN=ON) but not found. Please install MNN or check paths.")
    endif()
else()
    message(STATUS "MNN backend disabled.")
endif()

# 选项：是否启用 TensorRT 后端
option(ENABLE_TENSORRT "Enable TensorRT backend support" OFF)

if (ENABLE_TENSORRT)
    # 查找 CUDA
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    
    # 查找 TensorRT
    # 优先查找环境变量 TRT_HOME 指定的路径
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h 
        HINTS $ENV{TRT_HOME}/include 
              /usr/include/x86_64-linux-gnu 
              /usr/local/cuda/include)
              
    find_library(TENSORRT_LIB nvinfer 
        HINTS $ENV{TRT_HOME}/lib 
              /usr/lib/x86_64-linux-gnu 
              /usr/local/cuda/lib64)
              
    find_library(TENSORRT_ONNX_PARSER_LIB nvonnxparser 
        HINTS $ENV{TRT_HOME}/lib 
              /usr/lib/x86_64-linux-gnu 
              /usr/local/cuda/lib64)

    if (TENSORRT_INCLUDE_DIR AND TENSORRT_LIB)
        message(STATUS "Found TensorRT: ${TENSORRT_INCLUDE_DIR}")
        include_directories(${TENSORRT_INCLUDE_DIR})
        add_definitions(-DENABLE_TENSORRT)
        list(APPEND BACKEND_SOURCES src/backends/YoloV8TensorRT.cpp)
    else()
        message(FATAL_ERROR "TensorRT requested (ENABLE_TENSORRT=ON) but not found.")
    endif()
else()
    message(STATUS "TensorRT backend disabled.")
endif()

# 包含头文件目录
include_directories(include)
include_directories(3rdparty)

# 打印 OpenCV 信息
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# 创建可执行文件
add_executable(yolov8_app 
    src/main.cpp 
    src/YoloV8.cpp
    src/backends/YoloV8OpenCV.cpp
    ${BACKEND_SOURCES}
)

# 链接库
target_link_libraries(yolov8_app PRIVATE ${OpenCV_LIBS})

if (ENABLE_MNN AND MNN_LIBS)
    target_link_libraries(yolov8_app PRIVATE ${MNN_LIBS})
endif()

if (ENABLE_TENSORRT AND TENSORRT_LIB)
    target_link_libraries(yolov8_app PRIVATE ${TENSORRT_LIB} ${CUDA_LIBRARIES} ${TENSORRT_ONNX_PARSER_LIB})
endif()
