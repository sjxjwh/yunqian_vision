cmake_minimum_required(VERSION 3.16)

if(NOT OpenVINO_DIR)
  set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake")
endif()

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_subdirectory(planner/tinympc)

# ---- TensorRT（Jetson 上自带；找不到就只用 OpenVINO）----
find_path(
  TRT_INCLUDE_DIR NvInfer.h
  PATHS /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu /usr/include
)
find_library(
  TRT_LIB nvinfer
  PATHS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib
)
find_library(
  CUDART_LIB cudart
  PATHS /usr/local/cuda/lib64 /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu
)
find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h PATHS /usr/local/cuda/include /usr/include)
find_library(
  TRT_PLUGIN_LIB nvinfer_plugin
  PATHS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib
)

set(AUTO_AIM_SRCS
    armor.cpp
    classifier.cpp 
    detector.cpp
    solver.cpp
    aimer.cpp
    aim_corrector.cpp
    target.cpp
    tracker.cpp
    voter.cpp
    shooter.cpp
    yolo.cpp
    yolos/yolov5.cpp
    yolos/yolov8.cpp
    yolos/yolo11.cpp
    multithread/mt_detector.cpp
    planner/planner.cpp
)

set(AUTO_AIM_TRT_LIBS "")
if(TRT_INCLUDE_DIR AND TRT_LIB AND CUDART_LIB)
  message(STATUS "TensorRT found: ${TRT_LIB} (yolo11/yolov5 可用 TensorRT 推理)")
  list(APPEND AUTO_AIM_SRCS yolos/trt_engine.cpp)
  set(AUTO_AIM_TRT_LIBS ${TRT_LIB} ${TRT_PLUGIN_LIB} ${CUDART_LIB})
else()
  message(STATUS "TensorRT not found, 只用 OpenVINO")
endif()

# ---- CUDA 预处理（Jetson: 图像 letterbox/归一化放到 GPU）----
set(AUTO_AIM_CUDA_OFF "")
find_package(CUDAToolkit QUIET)
if(AUTO_AIM_TRT_LIBS AND CUDAToolkit_FOUND AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  if(NOT CMAKE_CUDA_COMPILER)
    foreach(cand /usr/local/cuda/bin/nvcc /usr/local/cuda-12.6/bin/nvcc)
      if(EXISTS ${cand})
        set(CMAKE_CUDA_COMPILER ${cand} CACHE FILEPATH "CUDA compiler" FORCE)
        break()
      endif()
    endforeach()
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES 87 CACHE STRING "CUDA architectures" FORCE)  # Jetson Orin (sm_87)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  message(STATUS "CUDA found: ${CUDAToolkit_VERSION} (预处理将走 GPU)")
  list(APPEND AUTO_AIM_SRCS yolos/cuda_preprocess.cu)
  set(AUTO_AIM_CUDA_OFF ON)
else()
  message(STATUS "CUDA 不可用，预处理走 CPU")
endif()

add_library(auto_aim OBJECT ${AUTO_AIM_SRCS})

# 只用 io 的头文件（command、cboard 的模式枚举），不链接 io，
# 这样仿真入口不必为此编海康/迈德威视 SDK。
target_include_directories(auto_aim PUBLIC ${PROJECT_SOURCE_DIR})
if(TRT_INCLUDE_DIR)
  target_include_directories(auto_aim PUBLIC ${TRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIR})
endif()
if(AUTO_AIM_TRT_LIBS)
  target_compile_definitions(auto_aim PUBLIC SP_WITH_TENSORRT)
endif()
if(AUTO_AIM_CUDA_OFF)
  target_compile_definitions(auto_aim PUBLIC SP_WITH_CUDA_PREPROCESS)
  target_link_libraries(auto_aim ${CUDAToolkit_LIBRARIES})
endif()
target_link_libraries(auto_aim openvino::runtime tinympcstatic ${AUTO_AIM_TRT_LIBS})

# 多线程命令生成直接调用 io::CBoard::send，只服务实机入口。
# 单独成库，避免仿真入口为它去链接整个 io（海康/迈德威视 SDK）。
add_library(auto_aim_mt OBJECT multithread/commandgener.cpp)
target_link_libraries(auto_aim_mt auto_aim)