#ifndef TOOLS__GIMBAL_CSV_LOGGER_HPP
#define TOOLS__GIMBAL_CSV_LOGGER_HPP

#include <Eigen/Geometry>
#include <cstdint>
#include <fstream>
#include <string>

namespace tools
{

struct GimbalLogSample
{
  double t_sec = 0.0;
  uint64_t sim_ts_ns = 0;
  uint64_t frame_seq = 0;
  bool new_meas = true;
  bool control = false;
  bool fire = false;
  double distance_m = 0.0;
  double bullet_speed = 0.0;

  Eigen::Quaterniond gimbal_q = Eigen::Quaterniond::Identity();
  double gimbal_yaw = 0.0;
  double gimbal_pitch = 0.0;
  double gimbal_yaw_vel = 0.0;
  double gimbal_pitch_vel = 0.0;

  double muzzle_yaw = 0.0;
  double muzzle_pitch = 0.0;
  double muzzle_yaw_vel = 0.0;
  double muzzle_pitch_vel = 0.0;

  double cmd_yaw = 0.0;
  double cmd_pitch = 0.0;
  double cmd_yaw_vel = 0.0;
  double cmd_yaw_acc = 0.0;
  double cmd_pitch_vel = 0.0;
  double cmd_pitch_acc = 0.0;

  double ref_yaw = 0.0;
  double ref_pitch = 0.0;
  double target_yaw = 0.0;
  double target_pitch = 0.0;
};

class GimbalCsvLogger
{
public:
  bool open(const std::string & path);
  bool is_open() const;
  void write(const GimbalLogSample & sample);

private:
  std::ofstream csv_;
  bool has_t0_ = false;
  double t0_sec_ = 0.0;
};

}  // namespace tools

#endif  // TOOLS__GIMBAL_CSV_LOGGER_HPP
