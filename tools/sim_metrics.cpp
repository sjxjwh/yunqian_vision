#include "sim_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>

#include "tools/math_tools.hpp"

namespace tools
{
namespace
{

constexpr double GRAVITY = 9.7833;
constexpr double BULLET_RADIUS = 0.0085;  // 17mm 弹丸半径 (m)
constexpr double MAX_FLIGHT_TIME = 1.5;   // s

Eigen::Matrix3d plate_rotation(
  double ring_roll, double ring_pitch, double yaw, double mount_pitch)
{
  const double cr = std::cos(ring_roll), sr = std::sin(ring_roll);
  const double cp = std::cos(ring_pitch), sp = std::sin(ring_pitch);
  const double cy = std::cos(yaw), sy = std::sin(yaw);
  const double cm = std::cos(mount_pitch), sm = std::sin(mount_pitch);

  // 与 solver.cpp::reproject_armor 同一套旋转：Ry(pitch) * Rx(roll) * Rz(yaw) * Ry(mount_pitch)
  // clang-format off
  Eigen::Matrix3d R_roll{
    {1,  0,   0},
    {0, cr, -sr},
    {0, sr,  cr}
  };
  Eigen::Matrix3d R_pitch{
    { cp, 0, sp},
    {  0, 1,  0},
    {-sp, 0, cp}
  };
  Eigen::Matrix3d R_mount{
    {cy * cm, -sy, cy * sm},
    {sy * cm,  cy, sy * sm},
    {    -sm,   0,      cm}
  };
  // clang-format on
  return R_pitch * R_roll * R_mount;
}

// 线段 p_prev -> p_curr 与敌方机身中心竖直圆柱体的首次交点（用于阻挡弹丸穿模打中背面装甲板）
bool intersect_chassis_cylinder(
  const Eigen::Vector3d & p_prev, const Eigen::Vector3d & p_curr,
  const Eigen::Vector3d & center, double radius, double z_min, double z_max, double & out_alpha)
{
  const Eigen::Vector2d d = (p_curr - p_prev).head<2>();
  const Eigen::Vector2d f = (p_prev - center).head<2>();
  const double r_eff = radius + BULLET_RADIUS;

  // 若起点已在圆柱内且在高度范围内，视为已被机身阻挡
  if (f.squaredNorm() <= r_eff * r_eff && p_prev.z() >= z_min && p_prev.z() <= z_max) {
    out_alpha = 0.0;
    return true;
  }

  const double a = d.squaredNorm();
  if (a < 1e-12) return false;
  const double b = 2.0 * f.dot(d);
  const double c = f.squaredNorm() - r_eff * r_eff;
  const double disc = b * b - 4.0 * a * c;
  if (disc < 0.0) return false;

  const double sqrt_disc = std::sqrt(disc);
  const double t1 = (-b - sqrt_disc) / (2.0 * a);
  const double t2 = (-b + sqrt_disc) / (2.0 * a);
  const double t = (t1 >= 0.0 && t1 <= 1.0) ? t1 : ((t2 >= 0.0 && t2 <= 1.0) ? t2 : -1.0);
  if (t < 0.0) return false;

  const double z_hit = p_prev.z() + t * (p_curr.z() - p_prev.z());
  if (z_hit < z_min || z_hit > z_max) return false;
  out_alpha = t;
  return true;
}

}  // namespace

SimMetrics::SimMetrics(double fire_hz, double air_k)
: fire_interval_(fire_hz > 0.1 ? 1.0 / fire_hz : 0.1), air_k_(std::max(0.0, air_k))
{
}

bool SimMetrics::open_csv(const std::string & path)
{
  if (path.empty()) return false;
  csv_.open(path, std::ios::out | std::ios::trunc);
  if (!csv_.is_open()) return false;
  csv_ << "t_sec,control,fire,gimbal_yaw_deg,plan_yaw_deg,yaw_err_deg,"
          "gimbal_pitch_deg,plan_pitch_deg,shots,hits,resolved,hit_rate\n";
  return true;
}

void SimMetrics::reset()
{
  last_t_ = -1.0;
  last_shot_t_ = -1e9;
  shots_ = 0;
  hits_ = 0;
  resolved_ = 0;
  err_samples_ = 0;
  fire_samples_ = 0;
  yaw_err_sq_sum_ = 0.0;
  bullets_.clear();
}

bool SimMetrics::intersect_plate(
  const Eigen::Vector3d & p_prev, const Eigen::Vector3d & p_curr, const SimArmorPlate & plate,
  double & out_alpha)
{
  const Eigen::Matrix3d R =
    plate_rotation(plate.ring_roll, plate.ring_pitch, plate.yaw, plate.mount_pitch);
  const Eigen::Vector3d a = R.transpose() * (p_prev - plate.center);
  const Eigen::Vector3d b = R.transpose() * (p_curr - plate.center);

  // 装甲板局部系中 +X 指向车体内部、-X 为正面外法向，Y 为宽度，Z 为高度。
  // 仅允许正面入射（从 -X 外侧飞向 +X 内侧），拒绝从背面穿透判命中。
  const double dx = b.x() - a.x();
  if (dx <= 1e-9) return false;

  double alpha = 0.0;
  if (a.x() <= 0.0 && b.x() >= 0.0) {
    alpha = std::clamp(-a.x() / dx, 0.0, 1.0);
  } else if (a.x() <= 0.0 && b.x() >= -BULLET_RADIUS) {
    alpha = 1.0;
  } else if (a.x() >= -BULLET_RADIUS && a.x() <= BULLET_RADIUS) {
    alpha = 0.0;
  } else {
    return false;
  }

  const Eigen::Vector3d hit = a + alpha * (b - a);
  const double half_w = plate.width * 0.5 + BULLET_RADIUS;
  const double half_h = plate.height * 0.5 + BULLET_RADIUS;
  if (std::abs(hit.y()) <= half_w && std::abs(hit.z()) <= half_h) {
    out_alpha = alpha;
    return true;
  }
  return false;
}

SimMetricsSnapshot SimMetrics::update(
  double t_sec, const Eigen::Vector3d & muzzle_ypr, bool control, double plan_yaw,
  double plan_pitch, bool fire_cmd, double bullet_speed,
  const std::vector<SimArmorPlate> & armors, const Eigen::Vector3d & muzzle_origin,
  const Eigen::Vector3d & self_vel)
{
  // 仿真重启或时间回退时清空飞行中弹丸
  if (last_t_ >= 0.0 && (t_sec < last_t_ || t_sec - last_t_ > 1.0)) {
    bullets_.clear();
    recent_hits_.clear();
    last_shot_t_ = -1e9;
  }
  last_t_ = t_sec;
  // 清理超过 0.5 秒的旧命中事件
  recent_hits_.erase(
    std::remove_if(
      recent_hits_.begin(), recent_hits_.end(),
      [t_sec](const SimHitEvent & e) { return (t_sec - e.t_sec) > 0.5; }),
    recent_hits_.end());

  // 1. 跟踪误差与开火占空比统计
  double yaw_err = 0.0;
  double pitch_err = 0.0;
  if (control) {
    yaw_err = tools::limit_rad(muzzle_ypr[0] - plan_yaw);
    pitch_err = muzzle_ypr[1] - plan_pitch;
    yaw_err_sq_sum_ += yaw_err * yaw_err;
    err_samples_++;
    if (fire_cmd) fire_samples_++;
  }

  // 2. 推进现有虚拟弹丸并检测与当前时刻装甲板的碰撞（含机身遮挡，禁止穿透打背面）
  double target_max_dist = 0.0;
  Eigen::Vector3d body_center = Eigen::Vector3d::Zero();
  double body_radius = 0.0;
  const bool has_body = (armors.size() >= 2);
  for (const auto & plate : armors) {
    target_max_dist = std::max(target_max_dist, plate.center.norm());
    body_center += plate.center;
  }
  if (has_body) {
    body_center /= static_cast<double>(armors.size());
    double sum_r = 0.0;
    for (const auto & plate : armors) {
      sum_r += (plate.center - body_center).head<2>().norm();
    }
    body_radius = std::max(0.08, (sum_r / static_cast<double>(armors.size())) - 0.035);
  }

  std::vector<Bullet> remaining;
  remaining.reserve(bullets_.size());

  for (auto & b : bullets_) {
    const double dt = t_sec - b.t0;
    const double step_total = t_sec - b.last_t;
    if (dt <= 0.0 || step_total <= 0.0) {
      remaining.push_back(b);
      continue;
    }
    Eigen::Vector3d curr_pos;
    if (air_k_ <= 0.0) {
      // 惯性世界系弹丸位移减去自身云台原点位移 self_vel * step_total
      const Eigen::Vector3d dpos_world =
        b.vel * step_total + Eigen::Vector3d(0.0, 0.0, -0.5 * GRAVITY * step_total * step_total);
      b.vel.z() -= GRAVITY * step_total;
      curr_pos = b.prev_pos + dpos_world - self_vel * step_total;
      b.last_t = t_sec;
    } else {
      // 子步进 RK4 推进，与 tools::Trajectory3D 的二次空气阻力物理一致
      const int substeps = std::max(1, static_cast<int>(std::ceil(step_total / 0.005)));
      const double h = step_total / substeps;
      curr_pos = b.prev_pos;
      Eigen::Vector3d v = b.vel;
      auto accel = [this](const Eigen::Vector3d & vel) -> Eigen::Vector3d {
        return Eigen::Vector3d(0.0, 0.0, -GRAVITY) - air_k_ * vel.norm() * vel;
      };
      for (int s = 0; s < substeps; ++s) {
        const Eigen::Vector3d a1 = accel(v);
        const Eigen::Vector3d v2 = v + 0.5 * h * a1;
        const Eigen::Vector3d a2 = accel(v2);
        const Eigen::Vector3d v3 = v + 0.5 * h * a2;
        const Eigen::Vector3d a3 = accel(v3);
        const Eigen::Vector3d v4 = v + h * a3;
        const Eigen::Vector3d a4 = accel(v4);
        curr_pos += (h / 6.0) * (v + 2.0 * v2 + 2.0 * v3 + v4);
        v += (h / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4);
      }
      curr_pos -= self_vel * step_total;
      b.vel = v;
      b.last_t = t_sec;
    }

    bool hit = false;
    double best_alpha = 2.0;
    int best_armor_id = -1;
    for (int idx = 0; idx < static_cast<int>(armors.size()); ++idx) {
      double alpha = 0.0;
      if (intersect_plate(b.prev_pos, curr_pos, armors[idx], alpha) && alpha < best_alpha) {
        best_alpha = alpha;
        best_armor_id = idx;
        hit = true;
      }
    }

    // 机身遮挡检测：若在命中正面装甲板之前先撞到敌方机身，则弹丸被挡住且不算命中
    double body_alpha = 2.0;
    const bool blocked_by_body =
      has_body &&
      intersect_chassis_cylinder(
        b.prev_pos, curr_pos, body_center, body_radius, body_center.z() - 0.16,
        body_center.z() + 0.22, body_alpha);

    if (hit && (!blocked_by_body || best_alpha <= body_alpha + 1e-4)) {
      hits_++;
      resolved_++;
      const Eigen::Vector3d hit_pos = b.prev_pos + best_alpha * (curr_pos - b.prev_pos);
      recent_hits_.push_back(
        {next_hit_id_++, t_sec, hit_pos.x(), hit_pos.y(), hit_pos.z(), best_armor_id});
      continue;
    }
    if (blocked_by_body) {
      resolved_++;
      continue;
    }

    const double dist_limit = (target_max_dist > 0.2) ? (target_max_dist + 0.6) : b.max_dist;
    if (curr_pos.norm() > dist_limit || dt > MAX_FLIGHT_TIME || curr_pos.z() < -1.5) {
      resolved_++;
      continue;
    }

    b.prev_pos = curr_pos;
    remaining.push_back(b);
  }
  bullets_ = std::move(remaining);

  // 3. 按当前真实炮口朝向与位置发射新虚拟弹丸
  if (control && fire_cmd && (t_sec - last_shot_t_ >= fire_interval_ - 1e-6)) {
    const double v = (bullet_speed > 5.0 && bullet_speed < 40.0) ? bullet_speed : 23.0;
    const double cy = std::cos(muzzle_ypr[0]);
    const double sy = std::sin(muzzle_ypr[0]);
    const double cp = std::cos(muzzle_ypr[1]);
    const double sp = std::sin(muzzle_ypr[1]);

    Bullet b;
    b.t0 = t_sec;
    b.last_t = t_sec;
    b.p0 = muzzle_origin;
    // 视觉世界系 pitch 抬头为负，R_muzzle2world * (1, 0, 0)^T = (cy*cp, sy*cp, -sp)；叠加自身底盘速度 self_vel
    b.v0 = self_vel + v * Eigen::Vector3d(cy * cp, sy * cp, -sp);
    b.vel = b.v0;
    b.prev_pos = b.p0;
    b.max_dist = (target_max_dist > 0.2) ? (target_max_dist + 0.8) : 12.0;

    bullets_.push_back(b);
    shots_++;
    last_shot_t_ = t_sec;
  }

  SimMetricsSnapshot snap;
  snap.shots = shots_;
  snap.hits = hits_;
  snap.resolved = resolved_;
  snap.in_flight = static_cast<uint32_t>(bullets_.size());
  snap.hit_rate = (resolved_ > 0) ? (static_cast<double>(hits_) / resolved_) : 0.0;
  snap.fire_duty = (err_samples_ > 0) ? (static_cast<double>(fire_samples_) / err_samples_) : 0.0;
  snap.yaw_err = yaw_err;
  snap.pitch_err = pitch_err;
  snap.yaw_err_rms = (err_samples_ > 0) ? std::sqrt(yaw_err_sq_sum_ / err_samples_) : 0.0;
  snap.bullets.reserve(bullets_.size());
  for (const auto & b : bullets_) {
    snap.bullets.push_back({b.prev_pos.x(), b.prev_pos.y(), b.prev_pos.z()});
  }
  snap.recent_hits = recent_hits_;

  if (csv_.is_open()) {
    constexpr double R2D = 180.0 / M_PI;
    csv_ << std::fixed << std::setprecision(4) << t_sec << ',' << (control ? 1 : 0) << ','
         << (fire_cmd ? 1 : 0) << ',' << muzzle_ypr[0] * R2D << ',' << plan_yaw * R2D << ','
         << yaw_err * R2D << ',' << muzzle_ypr[1] * R2D << ',' << plan_pitch * R2D << ','
         << snap.shots << ',' << snap.hits << ',' << snap.resolved << ',' << snap.hit_rate << '\n';
  }

  return snap;
}

}  // namespace tools
