#ifndef IO__SIM__SIM_IO_HPP
#define IO__SIM__SIM_IO_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "io/command.hpp"
#include "io/sim/sim_camera_model.hpp"
#include "io/sim/sim_gimbal_model.hpp"
#include "io/sim/talos_shm.hpp"

namespace io
{

// 仿真相机。read 的契约与 io::Camera 相同：带时间戳的 BGR 图。
// 额外带出这一帧的云台姿态，供 Solver::set_R_gimbal2world 使用。
class SimCamera
{
public:
  explicit SimCamera(
    const std::string & config_path, std::vector<int> capture_cpus = {});
  ~SimCamera();

  SimCamera(const SimCamera &) = delete;
  SimCamera & operator=(const SimCamera &) = delete;

  // 阻塞到仿真给出新帧，或 exiter 要求退出。图像是 BGR。
  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp);

  struct ReadTiming
  {
    double pace_ms = 0.0;
    double wait_ms = 0.0;
    double recv_ms = 0.0;
    double model_ms = 0.0;
  };
  ReadTiming last_read_timing() const { return last_timing_; }

  // 与最近一帧图像同一时刻的云台姿态，已应用配置里的 pose_correction。
  // 仿真发布的原始四元数还乘了炮口挂载（含 90° 相机朝向），修正后才是真云台。
  Eigen::Quaterniond gimbal_quaternion() const;

  // 真实炮口在 ROS 世界系的姿态（未乘 pose_correction，即仿真发布的原始炮口/光轴朝向）。
  Eigen::Quaterniond muzzle_quaternion() const;

  // 仿真炮口相对云台的平移，单位米。
  Eigen::Vector3d muzzle_in_gimbal() const;

  // 仿真相机相对云台的平移，单位米。供配置里的外参缺省时参考。
  Eigen::Vector3d camera_in_gimbal() const;

  // 云台在 ROS 世界系的位置，单位米。视觉解算把云台当原点，真值比对时用。
  Eigen::Vector3d gimbal_in_world() const;

  // 底盘在 ROS 世界系的瞬时线速度 (m/s) 与欧拉角 [roll, pitch, yaw] (rad)
  Eigen::Vector3d chassis_vel_world() const;
  Eigen::Vector3d chassis_rpy() const;

  uint64_t frame_seq() const;

  // 最近一帧的仿真时钟，纳秒。与图像、姿态同一时刻。
  uint64_t timestamp_ns() const;

  // 云台指令写回同一块共享内存，所以云台借用相机的连接。
  talos::ShmClient & client();

private:
  talos::ShmClient client_;
  talos::Frame latest_;
  // 右乘到仿真发布的 gimbal_quat 上，用来除掉仿真多乘的那段旋转。
  // 缺省单位四元数，等价于不修正，实机路径不经过这里。
  Eigen::Quaterniond pose_correction_{Eigen::Quaterniond::Identity()};

  // 可选真实相机复现模型（默认关闭，不影响原仿真路径）
  SimCameraModel camera_model_;
  Eigen::Matrix3d R_camera2gimbal_{Eigen::Matrix3d::Identity()};
  bool has_prev_pose_ = false;
  Eigen::Quaterniond prev_gimbal_quat_{Eigen::Quaterniond::Identity()};
  uint64_t prev_sim_ts_ns_ = 0;
  std::chrono::steady_clock::time_point last_output_tp_{};
  ReadTiming last_timing_{};

  // 异步后台抓帧 + 相机退化流水线（与主线程 YOLO/Tracker 并行，只保留最新一帧不积压延迟）
  struct ReadySlot
  {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp{};
    talos::Frame meta;
    ReadTiming timing{};
    uint64_t gen = 0;
  };
  void capture_loop();
  std::atomic<bool> stop_capture_{false};
  std::atomic<bool> capture_init_done_{false};
  std::vector<int> capture_cpus_;
  std::thread capture_thread_;
  std::mutex ready_mtx_;
  std::condition_variable ready_cv_;
  ReadySlot ready_slot_;
  uint64_t consumed_gen_ = 0;
};

// 仿真云台。send 把 io::Command 换成仿真消费的度数制指令写回共享内存。
// 若配置启用 sim_gimbal_model，则先经三号步兵 LQR+LESO+DM4310 真实动力学积分再发往仿真器渲染。
class SimGimbal
{
public:
  explicit SimGimbal(talos::ShmClient & client, const std::string & config_path = "");

  // control == false 时写 distance = -1，仿真据此停止跟踪。
  // yaw/pitch 入参是弧度，pitch 抬头为负，与 io::Command 一致。
  void send(
    const Command & command, double distance_m, uint64_t timestamp_ns, double yaw_vel = 0.0,
    double yaw_acc = 0.0, double pitch_vel = 0.0, double pitch_acc = 0.0,
    std::optional<Eigen::Vector2d> meas_yp = std::nullopt);

  // 启用 sim_gimbal_model 且处于控制态时，返回 500 Hz 物理模型当前炮口姿态（无渲染帧量化阶梯）。
  std::optional<Eigen::Vector2d> model_yp() const
  {
    if (gimbal_model_.enabled() && gimbal_model_.initialized() && was_controlling_) {
      return gimbal_model_.state_yp();
    }
    return std::nullopt;
  }

private:
  talos::ShmClient & client_;
  SimGimbalModel gimbal_model_;
  bool was_controlling_ = false;
};

}  // namespace io

#endif  // IO__SIM__SIM_IO_HPP
