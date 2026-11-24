#ifndef TOOLS__TRAJECTORY_HPP
#define TOOLS__TRAJECTORY_HPP

#include <Eigen/Dense>

namespace tools
{
// 标准 17mm 弹丸二次空气阻力系数 k1 = 0.5 * rho * Cd * A / m (m^-1)
// rho = 1.225 kg/m^3, Cd = 0.47, d = 0.017 m, m = 0.0032 kg -> 0.02043
constexpr double RM_17MM_DRAG_K = 0.02043;

struct Trajectory
{
  bool unsolvable;
  double fly_time;
  double pitch;  // 抬头为正

  // 弹道解算：k1 <= 0 时走无空气阻力解析解；k1 > 0 时走二维耦合二次阻力 RK4 迭代解算
  // v0 子弹初速度大小，单位：m/s
  // d 目标水平距离，单位：m
  // h 目标竖直高度，单位：m
  // k1 空气阻力系数 0.5 * rho * Cd * A / m，单位：m^-1（默认 0，兼容旧调用）
  Trajectory(const double v0, const double d, const double h, const double k1 = 0.0);
};

// 3D 跑打弹道解算：叠加自身底盘/炮口世界系线速度 v_self 与二次空气阻力 k1
// 弹丸世界系初速度 v_b(0) = v_self + v0 * [cos(pitch)*cos(yaw), cos(pitch)*sin(yaw), sin(pitch)]^T
struct Trajectory3D
{
  bool unsolvable = true;
  double fly_time = 0.0;
  double yaw = 0.0;    // 世界系方位角，rad（左正右负）
  double pitch = 0.0;  // 抬头为正，rad

  Trajectory3D(
    double v0, const Eigen::Vector3d & target_pos,
    const Eigen::Vector3d & v_self = Eigen::Vector3d::Zero(), double k1 = 0.0);
};

// 与 Trajectory 一致的弹道前向模型：求 t 时刻弹丸相对炮口的水平距离 r 与高度 z（pitch 抬头为正，单位 rad）
void ballistic_point(double v0, double pitch, double k1, double t, double & r, double & z);

// 与 Trajectory3D 一致的 3D 弹道前向模型：求 t 时刻弹丸相对发射炮口的 3D 位移（pitch_up_pos 抬头为正）
Eigen::Vector3d ballistic_point_3d(
  double v0, double yaw, double pitch_up_pos, const Eigen::Vector3d & v_self, double k1, double t);

}  // namespace tools

#endif  // TOOLS__TRAJECTORY_HPP