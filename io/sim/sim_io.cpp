#include "sim_io.hpp"

#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <vector>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

#include "tools/exiter.hpp"
#include "tools/logger.hpp"

namespace io
{

namespace
{
constexpr double RAD2DEG = 180.0 / M_PI;
}

SimCamera::SimCamera(const std::string & config_path, std::vector<int> capture_cpus)
: capture_cpus_(std::move(capture_cpus))
{
  // 仿真发布的 gimbal_quat 并不是真云台：capture.rs 里乘了 SHOT_DIRECTION 的局部
  // 旋转和一段 90° 相机旋转。这里用配置把它除掉，默认单位四元数等于不修正。
  auto yaml = YAML::LoadFile(config_path);
  if (yaml["pose_correction"]) {
    auto data = yaml["pose_correction"].as<std::vector<double>>();
    if (data.size() != 4) {
      throw std::runtime_error("pose_correction 必须是四元数 [w, x, y, z]");
    }
    pose_correction_ = Eigen::Quaterniond(data[0], data[1], data[2], data[3]);
    if (pose_correction_.norm() < 1e-9) {
      throw std::runtime_error("pose_correction 不能是零四元数");
    }
    pose_correction_.normalize();
  }
  if (yaml["R_camera2gimbal"]) {
    auto r = yaml["R_camera2gimbal"].as<std::vector<double>>();
    if (r.size() == 9) {
      R_camera2gimbal_ << r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8];
    }
  }
  camera_model_ = SimCameraModel::from_yaml(config_path);

  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (std::chrono::steady_clock::now() < deadline) {
    try {
      client_.connect();
      capture_thread_ = std::thread(&SimCamera::capture_loop, this);
      while (!capture_init_done_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
      return;
    } catch (const std::exception & e) {
      tools::logger()->warn("[Sim] {}，1 秒后重试", e.what());
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
  throw std::runtime_error("30 秒内没有连上仿真，请先启动带 talos 特性的仿真");
}

SimCamera::~SimCamera()
{
  stop_capture_.store(true, std::memory_order_release);
  ready_cv_.notify_all();
  if (capture_thread_.joinable()) capture_thread_.join();
}

void SimCamera::capture_loop()
{
#ifdef __linux__
  if (!capture_cpus_.empty()) {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int cpu : capture_cpus_) {
      if (cpu >= 0 && cpu < CPU_SETSIZE) CPU_SET(cpu, &set);
    }
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0) {
      // 在绑定好采集核后初始化 OpenCV 线程池，使其 worker 继承采集核亲和性、不与 OpenVINO 抢核
      const int n = std::max(1, std::min(static_cast<int>(capture_cpus_.size()), 12));
      cv::setNumThreads(n);
      cv::parallel_for_(cv::Range(0, n * 4), [](const cv::Range &) {});
    }
  }
#endif
  capture_init_done_.store(true, std::memory_order_release);
  talos::Frame rx_frame;
  cv::Mat proc_img;
  bool was_down = false;
  using Ms = std::chrono::duration<double, std::milli>;

  while (!stop_capture_.load(std::memory_order_acquire) && !tools::Exiter::requested()) {
    const auto t_start = std::chrono::steady_clock::now();
    const auto period = camera_model_.target_frame_period();
    if (period.count() > 0 && last_output_tp_.time_since_epoch().count() > 0) {
      // 绝对周期网格节流（phase-locked + 末段短自旋），把帧间隔抖动压到亚毫秒级
      auto target_tp = last_output_tp_ + period;
      const auto now = std::chrono::steady_clock::now();
      if (target_tp > now) {
        constexpr auto kSpinWindow = std::chrono::microseconds(400);
        if (target_tp - now > kSpinWindow) {
          std::this_thread::sleep_until(target_tp - kSpinWindow);
        }
        while (std::chrono::steady_clock::now() < target_tp &&
               !stop_capture_.load(std::memory_order_relaxed)) {
          std::this_thread::yield();
        }
      }
      const auto after = std::chrono::steady_clock::now();
      if (after > target_tp + period) {
        // 落后超过一整帧时重新对齐相位，避免连续突发
        last_output_tp_ = after;
      } else {
        last_output_tp_ = target_tp;
      }
    }
    const auto t_after_pace = std::chrono::steady_clock::now();

    auto t_recv_begin = t_after_pace;
    bool got_frame = false;
    while (!stop_capture_.load(std::memory_order_acquire) && !tools::Exiter::requested()) {
      if (!client_.connected() || !client_.alive()) {
        if (!was_down) {
          tools::logger()->warn("[Sim] 仿真无心跳，等待重连");
          was_down = true;
        }
        try {
          client_.connect();
        } catch (const std::exception &) {
          std::this_thread::sleep_for(std::chrono::milliseconds(200));
          continue;
        }
      }
      t_recv_begin = std::chrono::steady_clock::now();
      if (client_.recv(rx_frame)) {
        got_frame = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(150));
    }
    if (!got_frame) break;
    if (was_down) {
      tools::logger()->info("[Sim] 仿真已恢复");
      was_down = false;
    }

    const auto now = std::chrono::steady_clock::now();
    if (last_output_tp_.time_since_epoch().count() == 0) last_output_tp_ = now;
    std::chrono::steady_clock::time_point out_ts = now;

    if (camera_model_.enabled()) {
      Eigen::Vector3d omega_cam = Eigen::Vector3d::Zero();
      const auto q_curr = (rx_frame.gimbal_quat * pose_correction_).normalized();
      if (has_prev_pose_ && rx_frame.timestamp_ns > prev_sim_ts_ns_) {
        const double dt = static_cast<double>(rx_frame.timestamp_ns - prev_sim_ts_ns_) * 1e-9;
        if (dt > 1e-4 && dt < 0.2) {
          Eigen::Quaterniond dq = prev_gimbal_quat_.conjugate() * q_curr;
          if (dq.w() < 0.0) dq.coeffs() = -dq.coeffs();
          Eigen::AngleAxisd aa(dq);
          const Eigen::Vector3d omega_gimbal = (aa.axis() * aa.angle()) / dt;
          omega_cam = R_camera2gimbal_.transpose() * omega_gimbal;
        }
      }
      prev_gimbal_quat_ = q_curr;
      prev_sim_ts_ns_ = rx_frame.timestamp_ns;
      has_prev_pose_ = true;

      camera_model_.apply(rx_frame.bgr, proc_img, omega_cam);
      out_ts = camera_model_.adjust_timestamp(now);
    } else {
      rx_frame.bgr.copyTo(proc_img);
      out_ts = now;
    }

    const auto t_end = std::chrono::steady_clock::now();
    ReadTiming timing;
    timing.pace_ms = Ms(t_after_pace - t_start).count();
    timing.wait_ms = Ms(t_recv_begin - t_after_pace).count();
    timing.recv_ms = Ms(now - t_recv_begin).count();
    timing.model_ms = Ms(t_end - now).count();

    {
      std::lock_guard<std::mutex> lk(ready_mtx_);
      // 零拷贝交换图像缓冲（proc_img 与 ready_slot_.img 各自持有独立底层内存）
      cv::swap(proc_img, ready_slot_.img);
      ready_slot_.timestamp = out_ts;
      ready_slot_.meta.seq = rx_frame.seq;
      ready_slot_.meta.timestamp_ns = rx_frame.timestamp_ns;
      ready_slot_.meta.gimbal_quat = rx_frame.gimbal_quat;
      ready_slot_.meta.gimbal_in_world = rx_frame.gimbal_in_world;
      ready_slot_.meta.muzzle_in_gimbal = rx_frame.muzzle_in_gimbal;
      ready_slot_.meta.camera_in_gimbal = rx_frame.camera_in_gimbal;
      ready_slot_.meta.chassis_vel_world = rx_frame.chassis_vel_world;
      ready_slot_.meta.chassis_rpy = rx_frame.chassis_rpy;
      ready_slot_.meta.chassis_wz = rx_frame.chassis_wz;
      ready_slot_.meta.camera_matrix = rx_frame.camera_matrix;
      ready_slot_.meta.distort_coeffs = rx_frame.distort_coeffs;
      ready_slot_.timing = timing;
      ready_slot_.gen++;
    }
    ready_cv_.notify_one();
  }
  ready_cv_.notify_all();
}

void SimCamera::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  std::unique_lock<std::mutex> lk(ready_mtx_);
  while (!tools::Exiter::requested() && !stop_capture_.load(std::memory_order_acquire) &&
         ready_slot_.gen == consumed_gen_) {
    ready_cv_.wait_for(lk, std::chrono::milliseconds(50));
  }
  if (tools::Exiter::requested() || ready_slot_.gen == consumed_gen_) return;

  consumed_gen_ = ready_slot_.gen;
  // 把 ready_slot_.img 的所有权 swap 给调用方 img，主线程和后台线程各自持有独立底层缓冲
  cv::swap(img, ready_slot_.img);
  timestamp = ready_slot_.timestamp;
  latest_.seq = ready_slot_.meta.seq;
  latest_.timestamp_ns = ready_slot_.meta.timestamp_ns;
  latest_.gimbal_quat = ready_slot_.meta.gimbal_quat;
  latest_.gimbal_in_world = ready_slot_.meta.gimbal_in_world;
  latest_.muzzle_in_gimbal = ready_slot_.meta.muzzle_in_gimbal;
  latest_.camera_in_gimbal = ready_slot_.meta.camera_in_gimbal;
  latest_.chassis_vel_world = ready_slot_.meta.chassis_vel_world;
  latest_.chassis_rpy = ready_slot_.meta.chassis_rpy;
  latest_.chassis_wz = ready_slot_.meta.chassis_wz;
  latest_.camera_matrix = ready_slot_.meta.camera_matrix;
  latest_.distort_coeffs = ready_slot_.meta.distort_coeffs;
  last_timing_ = ready_slot_.timing;
}

Eigen::Quaterniond SimCamera::gimbal_quaternion() const
{
  // 右乘修正：仿真发布的是 q_true * q_extra，q_extra 的逆在配置里。
  return (latest_.gimbal_quat * pose_correction_).normalized();
}

Eigen::Quaterniond SimCamera::muzzle_quaternion() const { return latest_.gimbal_quat; }

Eigen::Vector3d SimCamera::muzzle_in_gimbal() const { return latest_.muzzle_in_gimbal; }

Eigen::Vector3d SimCamera::camera_in_gimbal() const { return latest_.camera_in_gimbal; }

Eigen::Vector3d SimCamera::gimbal_in_world() const { return latest_.gimbal_in_world; }

Eigen::Vector3d SimCamera::chassis_vel_world() const { return latest_.chassis_vel_world; }

Eigen::Vector3d SimCamera::chassis_rpy() const { return latest_.chassis_rpy; }

uint64_t SimCamera::frame_seq() const { return latest_.seq; }

uint64_t SimCamera::timestamp_ns() const { return latest_.timestamp_ns; }

talos::ShmClient & SimCamera::client() { return client_; }

SimGimbal::SimGimbal(talos::ShmClient & client, const std::string & config_path)
: client_(client), gimbal_model_(SimGimbalModel::from_yaml(config_path))
{
}

void SimGimbal::send(
  const Command & command, double distance_m, uint64_t timestamp_ns, double yaw_vel, double yaw_acc,
  double pitch_vel, double pitch_acc, std::optional<Eigen::Vector2d> meas_yp)
{
  talos::GimbalCmd cmd{};
  cmd.timestamp_ns = timestamp_ns;
  if (!command.control) {
    was_controlling_ = false;
    if (gimbal_model_.enabled() && meas_yp.has_value()) {
      gimbal_model_.reset((*meas_yp)[0], (*meas_yp)[1], timestamp_ns);
    }
    cmd.distance_m = talos::NO_SOLUTION_DISTANCE;
    client_.send(cmd);
    return;
  }

  double out_yaw = command.yaw;
  double out_pitch = command.pitch;
  if (gimbal_model_.enabled()) {
    if (!was_controlling_ && meas_yp.has_value()) {
      gimbal_model_.reset((*meas_yp)[0], (*meas_yp)[1], timestamp_ns);
    }
    const Eigen::Vector2d sim_yp = gimbal_model_.step(
      command.yaw, yaw_vel, yaw_acc, command.pitch, pitch_vel, pitch_acc, timestamp_ns);
    out_yaw = sim_yp[0];
    out_pitch = sim_yp[1];
  }
  was_controlling_ = true;

  // 视觉的 command.yaw/pitch 是“期望弹道方向”在真云台/世界系里的球坐标：
  //   d_ros = (cos p cos y, cos p sin y, sin p)，p 抬头为负。
  // 仿真把指令当成“炮口朝向” T 的 YXZ 欧拉角，炮口轴是 T 的 +Y；
  // 要让 T*(0,1,0) = A^T * d_ros，解出来是
  //   yaw_sim = command.yaw
  //   pitch_sim = command.pitch - 90°
  // 共享内存通道在消费前对 pitch 取反，from_solver_degrees 再减 90°：
  //   target.pitch = (-cmd.pitch_deg - 90°)
  // Bevy 中 +Y 向上，炮口轴 +Y 经 Rx(-90° + α) 后向上抬角为 +α；
  // 而视觉 command.pitch 抬头为负，故令 -cmd.pitch_deg = -command.pitch，即 cmd.pitch_deg = command.pitch。
  cmd.yaw_deg = static_cast<float>(out_yaw * RAD2DEG);
  cmd.pitch_deg = static_cast<float>(out_pitch * RAD2DEG);
  cmd.distance_m = static_cast<float>(distance_m);
  cmd.fire_advice = command.shoot ? 1 : 0;
  client_.send(cmd);
}

}  // namespace io
