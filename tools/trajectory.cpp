#include "trajectory.hpp"

#include <algorithm>
#include <cmath>

namespace tools
{
namespace
{
constexpr double g = 9.7833;

// 二维耦合二次空气阻力弹道前向积分（RK4），返回弹丸到达水平距离 d 时的高度与飞行时间
bool simulate_to_distance(
  double v0, double pitch, double d, double k1, double & out_z, double & out_t)
{
  double x = 0.0;
  double z = 0.0;
  double vx = v0 * std::cos(pitch);
  double vz = v0 * std::sin(pitch);
  if (vx <= 1e-3) return false;

  const double dt = std::clamp(d / (v0 * 16.0), 0.002, 0.01);
  double t = 0.0;

  auto deriv = [k1](double cur_vx, double cur_vz, double & ax, double & az) {
    const double speed = std::hypot(cur_vx, cur_vz);
    ax = -k1 * speed * cur_vx;
    az = -g - k1 * speed * cur_vz;
  };

  while (t < 5.0 && vx > 0.1) {
    double ax1, az1, ax2, az2, ax3, az3, ax4, az4;
    deriv(vx, vz, ax1, az1);
    const double vx2 = vx + 0.5 * dt * ax1;
    const double vz2 = vz + 0.5 * dt * az1;
    deriv(vx2, vz2, ax2, az2);
    const double vx3 = vx + 0.5 * dt * ax2;
    const double vz3 = vz + 0.5 * dt * az2;
    deriv(vx3, vz3, ax3, az3);
    const double vx4 = vx + dt * ax3;
    const double vz4 = vz + dt * az3;
    deriv(vx4, vz4, ax4, az4);

    const double dx = (dt / 6.0) * (vx + 2.0 * vx2 + 2.0 * vx3 + vx4);
    const double dz = (dt / 6.0) * (vz + 2.0 * vz2 + 2.0 * vz3 + vz4);

    if (x + dx >= d) {
      const double alpha = (dx > 1e-9) ? std::clamp((d - x) / dx, 0.0, 1.0) : 0.0;
      out_z = z + alpha * dz;
      out_t = t + alpha * dt;
      return true;
    }

    x += dx;
    z += dz;
    vx += (dt / 6.0) * (ax1 + 2.0 * ax2 + 2.0 * ax3 + ax4);
    vz += (dt / 6.0) * (az1 + 2.0 * az2 + 2.0 * az3 + az4);
    t += dt;
  }
  return false;
}

}  // namespace

Trajectory::Trajectory(const double v0, const double d, const double h, const double k1)
{
  if (v0 <= 1e-3 || d <= 1e-6) {
    unsolvable = true;
    fly_time = 0.0;
    pitch = 0.0;
    return;
  }

  auto a = g * d * d / (2 * v0 * v0);
  auto b = -d;
  auto c = a + h;
  auto delta = b * b - 4 * a * c;

  if (delta < 0) {
    unsolvable = true;
    fly_time = 0.0;
    pitch = 0.0;
    return;
  }

  unsolvable = false;
  // 直射低弹道解析解：分子有理化 tan(pitch) = 2*c / (-b + sqrt(delta))，避免近距 2*a 极小时的数值消除
  const double sqrt_delta = std::sqrt(delta);
  const double denom = -b + sqrt_delta;
  const double tan_pitch = (denom > 1e-9) ? (2.0 * c / denom) : ((-b - sqrt_delta) / (2.0 * a));
  pitch = std::atan(tan_pitch);
  fly_time = d / (v0 * std::cos(pitch));

  if (k1 <= 0.0) return;

  // 以无阻力解析解为初值，用打靶迭代求解含空气阻力仰角与飞行时间
  double cur_pitch = pitch;
  double cur_z = 0.0;
  double cur_t = fly_time;
  bool solved = false;

  for (int iter = 0; iter < 12; ++iter) {
    if (!simulate_to_distance(v0, cur_pitch, d, k1, cur_z, cur_t)) {
      unsolvable = true;
      return;
    }
    const double err_z = h - cur_z;
    if (std::abs(err_z) < 1e-4) {
      solved = true;
      break;
    }
    cur_pitch += std::atan2(err_z, d);
    if (cur_pitch < -1.3 || cur_pitch > 1.3) {
      unsolvable = true;
      return;
    }
  }

  if (!solved) {
    unsolvable = true;
    return;
  }
  pitch = cur_pitch;
  fly_time = cur_t;
}

void ballistic_point(double v0, double pitch, double k1, double t, double & r, double & z)
{
  if (t <= 0.0 || v0 <= 1e-3) {
    r = 0.0;
    z = 0.0;
    return;
  }

  if (k1 <= 0.0) {
    r = v0 * std::cos(pitch) * t;
    z = v0 * std::sin(pitch) * t - 0.5 * g * t * t;
    return;
  }

  // 与 simulate_to_distance 同构的二维耦合二次阻力 RK4 前向积分
  double x = 0.0;
  double cur_z = 0.0;
  double vx = v0 * std::cos(pitch);
  double vz = v0 * std::sin(pitch);

  const int steps = std::max(1, static_cast<int>(std::ceil(t / 0.005)));
  const double dt = t / static_cast<double>(steps);

  auto deriv = [k1](double cur_vx, double cur_vz, double & ax, double & az) {
    const double speed = std::hypot(cur_vx, cur_vz);
    ax = -k1 * speed * cur_vx;
    az = -g - k1 * speed * cur_vz;
  };

  for (int i = 0; i < steps; ++i) {
    double ax1, az1, ax2, az2, ax3, az3, ax4, az4;
    deriv(vx, vz, ax1, az1);
    const double vx2 = vx + 0.5 * dt * ax1;
    const double vz2 = vz + 0.5 * dt * az1;
    deriv(vx2, vz2, ax2, az2);
    const double vx3 = vx + 0.5 * dt * ax2;
    const double vz3 = vz + 0.5 * dt * az2;
    deriv(vx3, vz3, ax3, az3);
    const double vx4 = vx + dt * ax3;
    const double vz4 = vz + dt * az3;
    deriv(vx4, vz4, ax4, az4);

    x += (dt / 6.0) * (vx + 2.0 * vx2 + 2.0 * vx3 + vx4);
    cur_z += (dt / 6.0) * (vz + 2.0 * vz2 + 2.0 * vz3 + vz4);
    vx += (dt / 6.0) * (ax1 + 2.0 * ax2 + 2.0 * ax3 + ax4);
    vz += (dt / 6.0) * (az1 + 2.0 * az2 + 2.0 * az3 + az4);
  }

  r = x;
  z = cur_z;
}

namespace
{
// 3D 耦合二次空气阻力前向积分，推进到水平投影 e_r · pos == target_d，返回交点位置与时间
bool simulate_3d_to_plane(
  double v0, double yaw, double pitch_up, const Eigen::Vector3d & v_self, double k1,
  const Eigen::Vector2d & e_r, double target_d, Eigen::Vector3d & out_pos, double & out_t)
{
  const double cp = std::cos(pitch_up);
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();
  Eigen::Vector3d vel =
    v_self + v0 * Eigen::Vector3d(cp * std::cos(yaw), cp * std::sin(yaw), std::sin(pitch_up));

  const double vr0 = vel.x() * e_r.x() + vel.y() * e_r.y();
  if (vr0 <= 0.1) return false;

  const double dt = std::clamp(target_d / (vr0 * 16.0), 0.002, 0.01);
  double t = 0.0;
  double proj_prev = 0.0;

  auto accel = [k1](const Eigen::Vector3d & v) -> Eigen::Vector3d {
    return Eigen::Vector3d(0.0, 0.0, -g) - k1 * v.norm() * v;
  };

  while (t < 5.0) {
    const Eigen::Vector3d a1 = accel(vel);
    const Eigen::Vector3d v2 = vel + 0.5 * dt * a1;
    const Eigen::Vector3d a2 = accel(v2);
    const Eigen::Vector3d v3 = vel + 0.5 * dt * a2;
    const Eigen::Vector3d a3 = accel(v3);
    const Eigen::Vector3d v4 = vel + dt * a3;
    const Eigen::Vector3d a4 = accel(v4);

    const Eigen::Vector3d dpos = (dt / 6.0) * (vel + 2.0 * v2 + 2.0 * v3 + v4);
    const Eigen::Vector3d next_pos = pos + dpos;
    const double proj_next = next_pos.x() * e_r.x() + next_pos.y() * e_r.y();

    if (proj_next >= target_d) {
      const double dproj = proj_next - proj_prev;
      const double alpha = (dproj > 1e-9) ? std::clamp((target_d - proj_prev) / dproj, 0.0, 1.0) : 0.0;
      out_pos = pos + alpha * dpos;
      out_t = t + alpha * dt;
      return true;
    }

    if (proj_next <= proj_prev && t > 0.05) return false;
    pos = next_pos;
    vel += (dt / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4);
    proj_prev = proj_next;
    t += dt;
  }
  return false;
}
}  // namespace

Trajectory3D::Trajectory3D(
  double v0, const Eigen::Vector3d & target_pos, const Eigen::Vector3d & v_self, double k1)
{
  const double d_horiz = std::hypot(target_pos.x(), target_pos.y());
  if (v0 <= 1e-3 || d_horiz <= 1e-6) {
    unsolvable = true;
    return;
  }

  // 静止快速路径：与 2D Trajectory 完全一致
  if (v_self.squaredNorm() < 1e-10) {
    Trajectory traj2d(v0, d_horiz, target_pos.z(), k1);
    unsolvable = traj2d.unsolvable;
    if (!unsolvable) {
      fly_time = traj2d.fly_time;
      yaw = std::atan2(target_pos.y(), target_pos.x());
      pitch = traj2d.pitch;
    }
    return;
  }

  // 1. 无阻力跑打精确解（迭代等效目标点 p_eff = target_pos - v_self * t）
  double t_est = d_horiz / v0;
  double cur_yaw = std::atan2(target_pos.y(), target_pos.x());
  double cur_pitch = 0.0;
  bool no_drag_ok = false;

  for (int iter = 0; iter < 8; ++iter) {
    const Eigen::Vector3d p_eff = target_pos - v_self * t_est;
    const double d_eff = std::hypot(p_eff.x(), p_eff.y());
    if (d_eff <= 1e-5) {
      unsolvable = true;
      return;
    }
    Trajectory traj_nd(v0, d_eff, p_eff.z(), 0.0);
    if (traj_nd.unsolvable) {
      unsolvable = true;
      return;
    }
    cur_yaw = std::atan2(p_eff.y(), p_eff.x());
    cur_pitch = traj_nd.pitch;
    if (std::abs(traj_nd.fly_time - t_est) < 1e-5) {
      t_est = traj_nd.fly_time;
      no_drag_ok = true;
      break;
    }
    t_est = traj_nd.fly_time;
    no_drag_ok = true;
  }

  if (!no_drag_ok) {
    unsolvable = true;
    return;
  }

  if (k1 <= 0.0) {
    unsolvable = false;
    fly_time = t_est;
    yaw = cur_yaw;
    pitch = cur_pitch;
    return;
  }

  // 2. 3D 耦合二次空气阻力打靶迭代
  const Eigen::Vector2d e_r(target_pos.x() / d_horiz, target_pos.y() / d_horiz);
  const Eigen::Vector2d e_l(-e_r.y(), e_r.x());
  Eigen::Vector3d hit_pos = Eigen::Vector3d::Zero();
  double cur_t = t_est;
  bool solved = false;

  for (int iter = 0; iter < 12; ++iter) {
    if (!simulate_3d_to_plane(v0, cur_yaw, cur_pitch, v_self, k1, e_r, d_horiz, hit_pos, cur_t)) {
      unsolvable = true;
      return;
    }
    const Eigen::Vector3d err = target_pos - hit_pos;
    const double err_l = err.x() * e_l.x() + err.y() * e_l.y();
    const double err_z = err.z();
    if (std::hypot(err_l, err_z) < 2e-4) {
      solved = true;
      break;
    }
    cur_yaw += std::atan2(err_l, d_horiz);
    cur_pitch += std::atan2(err_z, d_horiz);
    if (cur_pitch < -1.3 || cur_pitch > 1.3) {
      unsolvable = true;
      return;
    }
  }

  if (!solved) {
    unsolvable = true;
    return;
  }
  unsolvable = false;
  fly_time = cur_t;
  yaw = std::atan2(std::sin(cur_yaw), std::cos(cur_yaw));
  pitch = cur_pitch;
}

Eigen::Vector3d ballistic_point_3d(
  double v0, double yaw, double pitch_up_pos, const Eigen::Vector3d & v_self, double k1, double t)
{
  if (t <= 0.0 || v0 <= 1e-3) return Eigen::Vector3d::Zero();

  const double cp = std::cos(pitch_up_pos);
  Eigen::Vector3d vel =
    v_self +
    v0 * Eigen::Vector3d(cp * std::cos(yaw), cp * std::sin(yaw), std::sin(pitch_up_pos));

  if (k1 <= 0.0) {
    return vel * t + Eigen::Vector3d(0.0, 0.0, -0.5 * g * t * t);
  }

  Eigen::Vector3d pos = Eigen::Vector3d::Zero();
  const int steps = std::max(1, static_cast<int>(std::ceil(t / 0.005)));
  const double dt = t / static_cast<double>(steps);

  auto accel = [k1](const Eigen::Vector3d & v) -> Eigen::Vector3d {
    return Eigen::Vector3d(0.0, 0.0, -g) - k1 * v.norm() * v;
  };

  for (int i = 0; i < steps; ++i) {
    const Eigen::Vector3d a1 = accel(vel);
    const Eigen::Vector3d v2 = vel + 0.5 * dt * a1;
    const Eigen::Vector3d a2 = accel(v2);
    const Eigen::Vector3d v3 = vel + 0.5 * dt * a2;
    const Eigen::Vector3d a3 = accel(v3);
    const Eigen::Vector3d v4 = vel + dt * a3;
    const Eigen::Vector3d a4 = accel(v4);

    pos += (dt / 6.0) * (vel + 2.0 * v2 + 2.0 * v3 + v4);
    vel += (dt / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4);
  }
  return pos;
}

}  // namespace tools