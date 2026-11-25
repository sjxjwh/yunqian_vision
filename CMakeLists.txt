cmake_minimum_required(VERSION 3.16.3)

project(sp_vision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
message(STATUS "--------------------CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}--------------------")

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
if(TARGET yaml-cpp::yaml-cpp)
  set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
else()
  set(YAML_CPP_TARGET yaml-cpp)
endif()
find_package(nlohmann_json REQUIRED)
if(NOT OpenVINO_DIR)
  set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake/")
endif()
find_package(OpenVINO REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
# 本地安装的 fmt / spdlog / yaml-cpp 不在默认搜索路径里，统一加进来。
foreach(_dep fmt spdlog yaml-cpp nlohmann_json)
  if(TARGET ${_dep}::${_dep})
    get_target_property(_inc ${_dep}::${_dep} INTERFACE_INCLUDE_DIRECTORIES)
    if(_inc)
      include_directories(${_inc})
    endif()
  endif()
endforeach()


add_subdirectory(tools)
add_subdirectory(io)

# 仿真共享内存适配。独立于 io，因为 io 链接海康/迈德威视 SDK 与 libusb，
# 仿真闭环不需要这些，NX 与 NUC 也因此能编同一份。
add_library(sim_io STATIC io/sim/talos_shm.cpp io/sim/sim_camera_model.cpp io/sim/sim_gimbal_model.cpp io/sim/sim_io.cpp)
target_link_libraries(sim_io PUBLIC ${OpenCV_LIBS} Eigen3::Eigen ${YAML_CPP_TARGET} tools)
target_include_directories(sim_io PUBLIC ${PROJECT_SOURCE_DIR})
add_subdirectory(tasks/auto_aim)
add_subdirectory(tasks/auto_buff)
add_subdirectory(tasks/omniperception)

####################src###################
add_executable(standard src/standard.cpp)
add_executable(sim_standard src/sim_standard.cpp)
add_executable(mt_standard src/mt_standard.cpp)
add_executable(standard_mpc src/standard_mpc.cpp)
add_executable(auto_aim_debug_mpc src/auto_aim_debug_mpc.cpp)
add_executable(mt_auto_aim_debug src/mt_auto_aim_debug.cpp)
add_executable(auto_buff_debug src/auto_buff_debug.cpp)
add_executable(auto_buff_debug_mpc src/auto_buff_debug_mpc.cpp)
add_executable(uav src/uav.cpp)
add_executable(uav_debug src/uav_debug.cpp)

target_link_libraries(standard ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io)
target_link_libraries(sim_standard ${OpenCV_LIBS} fmt::fmt ${YAML_CPP_TARGET} auto_aim tools sim_io)
# OpenVINO 运行时依赖 TBB。系统没装时用 OpenVINO 自带的那份。
if(NOT TBB_FOUND)
  find_library(TBB_LIBRARY tbb PATHS "${OpenVINO_DIR}/../3rdparty/tbb/lib" NO_DEFAULT_PATH)
  if(TBB_LIBRARY)
    target_link_libraries(sim_standard ${TBB_LIBRARY})
  endif()
endif()
target_link_libraries(mt_standard ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_aim_mt auto_buff tools io)
target_link_libraries(standard_mpc ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io)
target_link_libraries(auto_aim_debug_mpc ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io)
target_link_libraries(mt_auto_aim_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_aim_mt auto_buff tools io)
target_link_libraries(auto_buff_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_buff tools io ${CERES_LIBRARIES})
target_link_libraries(auto_buff_debug_mpc ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_buff tools io ${CERES_LIBRARIES})
target_link_libraries(uav ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io)
target_link_libraries(uav_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

##################calibration################
add_executable(capture calibration/capture.cpp)
add_executable(calibrate_camera calibration/calibrate_camera.cpp)
add_executable(calibrate_handeye calibration/calibrate_handeye.cpp)
add_executable(calibrate_robotworld_handeye calibration/calibrate_robotworld_handeye.cpp)
add_executable(split_video calibration/split_video.cpp)

target_link_libraries(capture ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(calibrate_camera ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(calibrate_handeye ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(calibrate_robotworld_handeye ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(split_video ${OpenCV_LIBS} fmt::fmt tools)

##################tests##################
add_executable(auto_aim_test tests/auto_aim_test.cpp)
add_executable(auto_buff_test tests/auto_buff_test.cpp)
add_executable(camera_detect_test tests/camera_detect_test.cpp)
add_executable(camera_test tests/camera_test.cpp)
add_executable(camera_thread_test tests/camera_thread_test.cpp)
add_executable(cboard_test tests/cboard_test.cpp)
add_executable(fire_test tests/fire_test.cpp)
add_executable(detector_video_test tests/detector_video_test.cpp)
add_executable(gimbal_response_test tests/gimbal_response_test.cpp)
add_executable(multi_usbcamera_test tests/multi_usbcamera_test.cpp)
add_executable(usbcamera_detect_test tests/usbcamera_detect_test.cpp)
add_executable(usbcamera_test tests/usbcamera_test.cpp)
add_executable(handeye_test tests/handeye_test.cpp)
add_executable(dm_test tests/dm_test.cpp)
add_executable(minimum_vision_system tests/minimum_vision_system.cpp)

target_link_libraries(auto_aim_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim)
target_link_libraries(auto_buff_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_buff tools io)
target_link_libraries(camera_detect_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools auto_aim io)
target_link_libraries(camera_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(camera_thread_test ${OpenCV_LIBS} fmt::fmt auto_aim tools io)
target_link_libraries(cboard_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(fire_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(detector_video_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools auto_aim)
target_link_libraries(gimbal_response_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
target_link_libraries(multi_usbcamera_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(usbcamera_detect_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim)
target_link_libraries(usbcamera_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
target_link_libraries(handeye_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim)
target_link_libraries(dm_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
target_link_libraries(minimum_vision_system ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim)

add_executable(gimbal_test tests/gimbal_test.cpp)
target_link_libraries(gimbal_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

add_executable(planner_test tests/planner_test.cpp)
target_link_libraries(planner_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

add_executable(planner_test_offline tests/planner_test_offline.cpp)
target_link_libraries(planner_test_offline ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

# 仿真姿态/世界系验收探针：只读共享内存，不进入实机路径。
add_executable(sim_pose_probe tests/sim_pose_probe.cpp)
target_link_libraries(sim_pose_probe ${OpenCV_LIBS} fmt::fmt ${YAML_CPP_TARGET} auto_aim tools sim_io)
if(TBB_LIBRARY)
  target_link_libraries(sim_pose_probe ${TBB_LIBRARY})
endif()

# 端到端云台传递函数辨识采样工具（仿真共享内存通道）。
add_executable(sim_gimbal_sysid tests/sim_gimbal_sysid.cpp)
target_link_libraries(sim_gimbal_sysid ${OpenCV_LIBS} fmt::fmt ${YAML_CPP_TARGET} tools sim_io)

# 实车云台传递函数与惯量极限辨识采样工具（串口/CAN 双通道，不依赖相机 SDK）。
add_executable(real_gimbal_sysid
  tests/real_gimbal_sysid.cpp
  io/gimbal/gimbal.cpp
  io/cboard.cpp
)
target_link_libraries(real_gimbal_sysid ${OpenCV_LIBS} fmt::fmt ${YAML_CPP_TARGET} tools serial)

# 仿真相机真实标定复现与退化模型单元测试（不依赖共享内存与相机 SDK）。
add_executable(sim_camera_model_test tests/sim_camera_model_test.cpp)
target_link_libraries(sim_camera_model_test ${OpenCV_LIBS} fmt::fmt ${YAML_CPP_TARGET} sim_io tools)

# 视觉弹道闭环（AimCorrector）与 Aimer/Planner 零偏联动单元测试（不依赖相机 SDK）。
add_executable(aim_corrector_test tests/aim_corrector_test.cpp)
target_link_libraries(aim_corrector_test ${OpenCV_LIBS} fmt::fmt ${YAML_CPP_TARGET} auto_aim tools)

# 图像平面 UVL+深度差 EKF 跟踪与动态聚焦 ROI 离线单元测试（不依赖相机 SDK）。
add_executable(uvl_tracker_test tests/uvl_tracker_test.cpp)
target_link_libraries(uvl_tracker_test ${OpenCV_LIBS} fmt::fmt ${YAML_CPP_TARGET} auto_aim tools)

# 跑打弹道、底盘速度/roll 融合与 3D 开火窗口闭环单元测试（不依赖相机 SDK）。
add_executable(shoot_on_move_test tests/shoot_on_move_test.cpp)
target_link_libraries(shoot_on_move_test ${OpenCV_LIBS} fmt::fmt ${YAML_CPP_TARGET} auto_aim sim_io tools)

# Layer 2 物理不确定度命中概率 P_phys 与 AimCorrector 散布注入 Pareto 对比测试。
add_executable(fire_prob_test tests/fire_prob_test.cpp)
target_link_libraries(fire_prob_test ${OpenCV_LIBS} fmt::fmt ${YAML_CPP_TARGET} auto_aim sim_io tools)

# 高保真同步录制与离线回放（100Hz 图像 + 1000Hz IMU）单元测试（不依赖相机 SDK 与串口）。
add_executable(sync_recorder_test tests/sync_recorder_test.cpp)
target_link_libraries(sync_recorder_test ${OpenCV_LIBS} fmt::fmt ${YAML_CPP_TARGET} tools)

# 离线同步回放自瞄入口（不依赖相机 SDK 与串口，支持 --latency-ms 离线 A/B 调参与 WebViewer）。
add_executable(replay_standard src/replay_standard.cpp)
target_link_libraries(replay_standard ${OpenCV_LIBS} fmt::fmt ${YAML_CPP_TARGET} auto_aim tools)

# 实机相机识别 + WebViewer 实时网页预览（无头 SSH 可直接看实时识别与 3D 重建）。
add_executable(camera_web_test tests/camera_web_test.cpp)
target_link_libraries(camera_web_test ${OpenCV_LIBS} fmt::fmt ${YAML_CPP_TARGET} auto_aim tools io)
if(TBB_LIBRARY)
  target_link_libraries(aim_corrector_test ${TBB_LIBRARY})
  target_link_libraries(uvl_tracker_test ${TBB_LIBRARY})
  target_link_libraries(shoot_on_move_test ${TBB_LIBRARY})
  target_link_libraries(fire_prob_test ${TBB_LIBRARY})
  target_link_libraries(replay_standard ${TBB_LIBRARY})
endif()

# 检查 ROS 环境
find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(std_msgs QUIET)
find_package(rosidl_typesupport_cpp QUIET)
find_package(sp_msgs QUIET)
# 只有在找到 ROS 环境时才会添加 sentry 和 publish_test
if(ament_cmake_FOUND AND rclcpp_FOUND AND std_msgs_FOUND AND rosidl_typesupport_cpp_FOUND AND sp_msgs_FOUND)
    # 编译 sentry 和 publish_test
    add_executable(sentry src/sentry.cpp)
    add_executable(sentry_bp src/sentry_bp.cpp)
    add_executable(sentry_debug src/sentry_debug.cpp)
    add_executable(sentry_multithread src/sentry_multithread.cpp)
    add_executable(publish_test tests/publish_test.cpp)
    add_executable(subscribe_test tests/subscribe_test.cpp)
    add_executable(topic_loop_test tests/topic_loop_test.cpp)

    target_link_libraries(sentry ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
    target_link_libraries(sentry_bp ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
    target_link_libraries(sentry_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
    target_link_libraries(sentry_multithread ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
    target_link_libraries(publish_test ${OpenCV_LIBS} fmt::fmt tools io auto_aim)
    target_link_libraries(subscribe_test ${OpenCV_LIBS} fmt::fmt tools io)
    target_link_libraries(topic_loop_test ${OpenCV_LIBS} fmt::fmt tools io)
else()
    message(STATUS "ROS2 environment not found, skipping ROS2-related code.")
endif()
