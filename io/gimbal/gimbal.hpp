#ifndef IO__GIMBAL_HPP
#define IO__GIMBAL_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>

#include "serial/serial.h"
#include "tools/sync_recorder.hpp"
#include "tools/thread_safe_queue.hpp"

namespace io
{
struct __attribute__((packed)) GimbalToVision
{
  uint8_t head[2] = {'S', 'P'};
  uint8_t mode;  // 0: 空闲, 1: 自瞄, 2: 小符, 3: 大符
  float q[4];    // wxyz顺序
  float yaw;
  float yaw_vel;
  float pitch;
  float pitch_vel;
  float bullet_speed;
  uint16_t bullet_count;  // 子弹累计发送次数
  uint16_t crc16;
};

static_assert(sizeof(GimbalToVision) <= 64);

struct __attribute__((packed)) VisionToGimbal
{
  uint8_t head[2] = {'S', 'P'};
  uint8_t mode;  // 0: 不控制, 1: 控制云台但不开火，2: 控制云台且开火
  float yaw;
  float yaw_vel;
  float yaw_acc;
  float pitch;
  float pitch_vel;
  float pitch_acc;
  uint16_t crc16;
};

static_assert(sizeof(VisionToGimbal) <= 64);

enum class GimbalMode
{
  IDLE,        // 空闲
  AUTO_AIM,    // 自瞄
  SMALL_BUFF,  // 小符
  BIG_BUFF     // 大符
};

struct GimbalState
{
  float yaw;
  float yaw_vel;
  float pitch;
  float pitch_vel;
  float roll = 0.0f;  // 云台在世界系下的滚转角 (rad，绕前向轴右倾为正)
  float vx = 0.0f;    // 预留：底盘世界系线速度 (m/s)，待电控协议扩展后填充
  float vy = 0.0f;
  float vz = 0.0f;
  float bullet_speed;
  uint16_t bullet_count;
};

class Gimbal
{
public:
  Gimbal(const std::string & config_path);

  ~Gimbal();

  GimbalMode mode() const;
  GimbalState state() const;
  std::string str(GimbalMode mode) const;
  Eigen::Quaterniond q(std::chrono::steady_clock::time_point t);

  void send(
    bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
    float pitch_acc);

  void send(io::VisionToGimbal VisionToGimbal);

  /// 注册高频串口/IMU 样本回调（在 read_thread 中 1000 Hz 触发，用于 SyncRecorder 同步录制）
  void set_sample_callback(std::function<void(const tools::SyncImuSample &)> cb);

private:
  serial::Serial serial_;

  std::thread thread_;
  std::atomic<bool> quit_ = false;
  mutable std::mutex mutex_;

  GimbalToVision rx_data_;
  VisionToGimbal tx_data_;

  // 电控串口 pitch 符号与限位（默认抬头为正，范围 -20° ~ +30°；视觉内部为右手系低头正/抬头负）
  double pitch_sign_ = -1.0;
  float pitch_min_rad_ = -20.0f * 3.14159265f / 180.0f;
  float pitch_max_rad_ = 30.0f * 3.14159265f / 180.0f;

  // 若下位机四元数 q 表示 imubody->world(Z朝上) 且 IMU 芯片相对云台存在固定安装旋转，
  // 右乘 q_gimbal2imu_ 将其换算为标准云台系 gimbal->world（X前 Y左 Z上）。默认为单位四元数。
  Eigen::Quaterniond q_gimbal2imu_ = Eigen::Quaterniond::Identity();

  // 当下位机 IMU 不可信、且 yaw/pitch 字段来自编码器时，改为直接用 yaw/pitch 构造四元数（roll=0），
  // 保证视觉世界系与电控编码器参考系严格一致。
  bool use_euler_q_ = false;

  // 相机曝光中点到取图时间戳的系统延迟补偿 (ms)，查询姿态时自动前移该时长
  double camera_latency_ms_ = 0.0;

  // 下发指令时是否将速度/加速度前馈清零（仅发位置，用于排查电控前馈符号或增益问题）
  bool zero_vel_ff_ = false;

  GimbalMode mode_ = GimbalMode::IDLE;
  GimbalState state_;
  tools::ThreadSafeQueue<std::tuple<Eigen::Quaterniond, std::chrono::steady_clock::time_point>>
    queue_{1000};

  mutable std::mutex callback_mtx_;
  std::function<void(const tools::SyncImuSample &)> sample_callback_;

  bool read(uint8_t * buffer, size_t size);
  void read_thread();
  void reconnect();
};

}  // namespace io

#endif  // IO__GIMBAL_HPP