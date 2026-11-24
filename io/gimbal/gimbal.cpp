#include "gimbal.hpp"

#include <algorithm>
#include <cmath>

#include "tools/crc.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace io
{
Gimbal::Gimbal(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto com_port = tools::read<std::string>(yaml, "com_port");
  const bool pitch_up_positive =
    yaml["pitch_up_positive"] ? yaml["pitch_up_positive"].as<bool>() : true;
  pitch_sign_ = pitch_up_positive ? -1.0 : 1.0;
  if (yaml["pitch_min_deg"]) {
    pitch_min_rad_ = static_cast<float>(yaml["pitch_min_deg"].as<double>() * M_PI / 180.0);
  }
  if (yaml["pitch_max_deg"]) {
    pitch_max_rad_ = static_cast<float>(yaml["pitch_max_deg"].as<double>() * M_PI / 180.0);
  }
  if (yaml["R_gimbal2imu"] && yaml["R_gimbal2imu"].size() == 9) {
    auto data = yaml["R_gimbal2imu"].as<std::vector<double>>();
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R(data.data());
    q_gimbal2imu_ = Eigen::Quaterniond(R).normalized();
    tools::logger()->info("[Gimbal] 已启用 R_gimbal2imu 四元数安装角修正 (right-multiply)");
  }
  use_euler_q_ = yaml["use_euler_q"] ? yaml["use_euler_q"].as<bool>() : false;
  if (use_euler_q_) {
    tools::logger()->info("[Gimbal] 已启用 use_euler_q：由串口 yaw/pitch 直接构造姿态四元数（忽略 IMU 四元数）");
  }
  camera_latency_ms_ =
    yaml["camera_latency_ms"] ? yaml["camera_latency_ms"].as<double>() : 0.0;
  zero_vel_ff_ = yaml["zero_vel_ff"] ? yaml["zero_vel_ff"].as<bool>() : false;
  if (camera_latency_ms_ != 0.0 || zero_vel_ff_) {
    tools::logger()->info(
      "[Gimbal] 延迟补偿 camera_latency_ms={:.1f} ms | 速度前馈清零 zero_vel_ff={}",
      camera_latency_ms_, zero_vel_ff_);
  }
  tools::logger()->info(
    "[Gimbal] 串口 pitch 约定: {} | 软限位 [{:.1f}°, {:.1f}°]",
    pitch_up_positive ? "抬头为正" : "低头为正", pitch_min_rad_ * 180.0 / M_PI,
    pitch_max_rad_ * 180.0 / M_PI);

  try {
    serial_.setPort(com_port);
    serial_.open();
  } catch (const std::exception & e) {
    tools::logger()->error("[Gimbal] Failed to open serial: {}", e.what());
    exit(1);
  }

  thread_ = std::thread(&Gimbal::read_thread, this);

  queue_.pop();
  tools::logger()->info("[Gimbal] First q received.");
}

Gimbal::~Gimbal()
{
  quit_ = true;
  if (thread_.joinable()) thread_.join();
  serial_.close();
}

GimbalMode Gimbal::mode() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return mode_;
}

GimbalState Gimbal::state() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

std::string Gimbal::str(GimbalMode mode) const
{
  switch (mode) {
    case GimbalMode::IDLE:
      return "IDLE";
    case GimbalMode::AUTO_AIM:
      return "AUTO_AIM";
    case GimbalMode::SMALL_BUFF:
      return "SMALL_BUFF";
    case GimbalMode::BIG_BUFF:
      return "BIG_BUFF";
    default:
      return "INVALID";
  }
}

Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point t)
{
  const auto t_query =
    t - std::chrono::microseconds(static_cast<int64_t>(std::llround(camera_latency_ms_ * 1000.0)));
  while (true) {
    auto [q_a, t_a] = queue_.pop();
    auto [q_b, t_b] = queue_.front();
    if (t_query <= t_a) return q_a.normalized();
    auto t_ab = tools::delta_time(t_a, t_b);
    auto t_ac = tools::delta_time(t_a, t_query);
    if (t_query <= t_b) {
      auto k = (t_ab > 1e-9) ? std::clamp(t_ac / t_ab, 0.0, 1.0) : 0.0;
      return q_a.slerp(k, q_b).normalized();
    }
  }
}

void Gimbal::send(io::VisionToGimbal VisionToGimbal)
{
  const float p_sign = static_cast<float>(pitch_sign_);
  tx_data_.mode = VisionToGimbal.mode;
  tx_data_.yaw = VisionToGimbal.yaw;
  tx_data_.yaw_vel = zero_vel_ff_ ? 0.0f : VisionToGimbal.yaw_vel;
  tx_data_.yaw_acc = zero_vel_ff_ ? 0.0f : VisionToGimbal.yaw_acc;
  tx_data_.pitch = std::clamp(VisionToGimbal.pitch * p_sign, pitch_min_rad_, pitch_max_rad_);
  tx_data_.pitch_vel = zero_vel_ff_ ? 0.0f : VisionToGimbal.pitch_vel * p_sign;
  tx_data_.pitch_acc = zero_vel_ff_ ? 0.0f : VisionToGimbal.pitch_acc * p_sign;
  tx_data_.crc16 = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));

  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

void Gimbal::send(
  bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
  float pitch_acc)
{
  const float p_sign = static_cast<float>(pitch_sign_);
  tx_data_.mode = control ? (fire ? 2 : 1) : 0;
  tx_data_.yaw = yaw;
  tx_data_.yaw_vel = zero_vel_ff_ ? 0.0f : yaw_vel;
  tx_data_.yaw_acc = zero_vel_ff_ ? 0.0f : yaw_acc;
  tx_data_.pitch = std::clamp(pitch * p_sign, pitch_min_rad_, pitch_max_rad_);
  tx_data_.pitch_vel = zero_vel_ff_ ? 0.0f : pitch_vel * p_sign;
  tx_data_.pitch_acc = zero_vel_ff_ ? 0.0f : pitch_acc * p_sign;
  tx_data_.crc16 = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));

  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

bool Gimbal::read(uint8_t * buffer, size_t size)
{
  try {
    return serial_.read(buffer, size) == size;
  } catch (const std::exception & e) {
    // tools::logger()->warn("[Gimbal] Failed to read serial: {}", e.what());
    return false;
  }
}

void Gimbal::read_thread()
{
  tools::logger()->info("[Gimbal] read_thread started.");
  int error_count = 0;

  while (!quit_) {
    if (error_count > 5000) {
      error_count = 0;
      tools::logger()->warn("[Gimbal] Too many errors, attempting to reconnect...");
      reconnect();
      continue;
    }

    if (!read(reinterpret_cast<uint8_t *>(&rx_data_), sizeof(rx_data_.head))) {
      error_count++;
      continue;
    }

    if (rx_data_.head[0] != 'S' || rx_data_.head[1] != 'P') continue;

    auto t = std::chrono::steady_clock::now();

    if (!read(
          reinterpret_cast<uint8_t *>(&rx_data_) + sizeof(rx_data_.head),
          sizeof(rx_data_) - sizeof(rx_data_.head))) {
      error_count++;
      continue;
    }

    if (!tools::check_crc16(reinterpret_cast<uint8_t *>(&rx_data_), sizeof(rx_data_))) {
      tools::logger()->debug("[Gimbal] CRC16 check failed.");
      continue;
    }

    error_count = 0;
    const double yaw_rad = static_cast<double>(rx_data_.yaw);
    const double pitch_vis = static_cast<double>(rx_data_.pitch) * pitch_sign_;
    Eigen::Quaterniond q =
      use_euler_q_
        ? Eigen::Quaterniond(
            Eigen::AngleAxisd(yaw_rad, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(pitch_vis, Eigen::Vector3d::UnitY()))
            .normalized()
        : (Eigen::Quaterniond(rx_data_.q[0], rx_data_.q[1], rx_data_.q[2], rx_data_.q[3]) *
           q_gimbal2imu_)
            .normalized();
    queue_.push({q, t});

    std::lock_guard<std::mutex> lock(mutex_);

    const float p_sign = static_cast<float>(pitch_sign_);
    state_.yaw = rx_data_.yaw;
    state_.yaw_vel = rx_data_.yaw_vel;
    state_.pitch = rx_data_.pitch * p_sign;
    state_.pitch_vel = rx_data_.pitch_vel * p_sign;
    state_.roll = use_euler_q_ ? 0.0f : static_cast<float>(tools::eulers(q, 2, 1, 0)[2]);
    state_.bullet_speed = rx_data_.bullet_speed;
    state_.bullet_count = rx_data_.bullet_count;

    switch (rx_data_.mode) {
      case 0:
        mode_ = GimbalMode::IDLE;
        break;
      case 1:
        mode_ = GimbalMode::AUTO_AIM;
        break;
      case 2:
        mode_ = GimbalMode::SMALL_BUFF;
        break;
      case 3:
        mode_ = GimbalMode::BIG_BUFF;
        break;
      default:
        mode_ = GimbalMode::IDLE;
        tools::logger()->warn("[Gimbal] Invalid mode: {}", rx_data_.mode);
        break;
    }
  }

  tools::logger()->info("[Gimbal] read_thread stopped.");
}

void Gimbal::reconnect()
{
  int max_retry_count = 10;
  for (int i = 0; i < max_retry_count && !quit_; ++i) {
    tools::logger()->warn("[Gimbal] Reconnecting serial, attempt {}/{}...", i + 1, max_retry_count);
    try {
      serial_.close();
      std::this_thread::sleep_for(std::chrono::seconds(1));
    } catch (...) {
    }

    try {
      serial_.open();  // 尝试重新打开
      queue_.clear();
      tools::logger()->info("[Gimbal] Reconnected serial successfully.");
      break;
    } catch (const std::exception & e) {
      tools::logger()->warn("[Gimbal] Reconnect failed: {}", e.what());
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
}

}  // namespace io