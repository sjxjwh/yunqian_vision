#ifndef AUTO_AIM__PLANNER_HPP
#define AUTO_AIM__PLANNER_HPP

#include <Eigen/Dense>
#include <atomic>
#include <list>
#include <optional>

#include "tasks/auto_aim/target.hpp"
#include "tinympc/tiny_api.hpp"

namespace auto_aim
{
constexpr double DT = 0.01;
constexpr int HALF_HORIZON = 50;
constexpr int HORIZON = HALF_HORIZON * 2;

using Trajectory = Eigen::Matrix<double, 4, HORIZON>;  // yaw, yaw_vel, pitch, pitch_vel

struct Plan
{
  bool control;
  bool fire;
  float target_yaw;
  float target_pitch;
  float yaw;
  float yaw_vel;
  float yaw_acc;
  float pitch;
  float pitch_vel;
  float pitch_acc;
  float ref_yaw = 0;    // 当前采样时刻的理想弹道 yaw（与同帧 muzzle_yaw 同相）
  float ref_pitch = 0;  // 当前采样时刻的理想弹道 pitch
};

struct SelfMotion
{
  Eigen::Vector3d vel = Eigen::Vector3d::Zero();               // 我方云台/底盘世界系线速度 (m/s)
  double roll = 0.0;                                           // 我方云台/底盘滚转角 (rad，绕前向轴，右倾为正)
  double pitch = 0.0;                                          // 我方底盘俯仰角 (rad，可选)
  Eigen::Vector3d muzzle_in_gimbal = Eigen::Vector3d::Zero();  // 炮口相对云台原点平移 (m)，零则回退 (0,0,muzzle_z_offset)
};

class Planner
{
public:
  Eigen::Vector4d debug_xyza;
  Planner(const std::string & config_path);

  Plan plan(
    Target target, double bullet_speed,
    std::optional<Eigen::Vector2d> gimbal_yp = std::nullopt,
    const SelfMotion & self_motion = SelfMotion{});
  Plan plan(
    std::optional<Target> target, double bullet_speed,
    std::optional<Eigen::Vector2d> gimbal_yp = std::nullopt, double frame_age_s = 0.0,
    const SelfMotion & self_motion = SelfMotion{});

  double air_k() const { return air_k_; }
  double muzzle_z_offset() const { return muzzle_z_offset_; }

private:
  std::atomic<double> yaw_offset_{0.0};
  std::atomic<double> pitch_offset_{0.0};
  double muzzle_z_offset_ = 0.0;
  double air_k_ = 0.0;
  double fire_thresh_;
  int shoot_offset_ = 2;
  double fire_margin_ = 0.85;
  double max_fire_angle_ = 60.0 / 57.3;
  double low_speed_delay_time_, high_speed_delay_time_, decision_speed_;
  double active_delay_time_ = 0.0;
  SelfMotion active_self_motion_{};
  bool use_yaw_plant_ = false;
  bool use_pitch_plant_ = false;

  TinySolver * yaw_solver_;
  TinySolver * pitch_solver_;

  void setup_yaw_solver(const std::string & config_path);
  void setup_pitch_solver(const std::string & config_path);

  Eigen::Vector3d muzzle_offset_in_world(double yaw, double pitch, double roll) const;
  bool check_ballistic_hit(
    const Target & hit_target, double bullet_speed, double fly_time, double muzzle_yaw,
    double muzzle_pitch) const;
  Eigen::Matrix<double, 2, 1> aim(const Target & target, double bullet_speed);
  Trajectory get_trajectory(Target target, double yaw0, double bullet_speed);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__PLANNER_HPP