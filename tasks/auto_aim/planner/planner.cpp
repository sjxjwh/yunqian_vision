#include "planner.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "tools/math_tools.hpp"
#include "tools/runtime_params.hpp"
#include "tools/trajectory.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

namespace auto_aim
{
namespace
{
// 连续二阶传递函数 G(s) = wn^2 / (s^2 + 2*zeta*wn*s + wn^2) 的精确 ZOH 离散化。
// 状态 x = [y, y_dot]^T，控制 u = 位置指令；使用标度平方法计算 exp(A_c * dt)，高 wn 同样稳定。
void discretize_second_order_zoh(
  double wn, double zeta, double dt, Eigen::MatrixXd & A_d, Eigen::MatrixXd & B_d)
{
  Eigen::Matrix2d A_c;
  A_c << 0.0, 1.0, -wn * wn, -2.0 * zeta * wn;
  Eigen::Vector2d B_c(0.0, wn * wn);

  // 选缩放幂次使 ||A_c * dt / 2^s|| < 0.25，6 阶泰勒截断误差 < 1e-10
  const double norm_a = A_c.norm() * dt;
  int s = 0;
  while ((norm_a / (1 << s)) > 0.25 && s < 16) s++;
  const Eigen::Matrix2d M = A_c * (dt / static_cast<double>(1 << s));

  Eigen::Matrix2d E = Eigen::Matrix2d::Identity();
  Eigen::Matrix2d term = Eigen::Matrix2d::Identity();
  for (int k = 1; k <= 6; ++k) {
    term = (term * M) / static_cast<double>(k);
    E += term;
  }
  for (int i = 0; i < s; ++i) E = E * E;

  A_d = E;
  B_d = A_c.inverse() * (E - Eigen::Matrix2d::Identity()) * B_c;
}

// 与 solver.cpp::reproject_armor / sim_metrics.cpp 同一套装甲板旋转矩阵
Eigen::Matrix3d armor_plate_rotation(
  double ring_roll, double ring_pitch, double yaw, double mount_pitch)
{
  const double cr = std::cos(ring_roll), sr = std::sin(ring_roll);
  const double cp = std::cos(ring_pitch), sp = std::sin(ring_pitch);
  const double cy = std::cos(yaw), sy = std::sin(yaw);
  const double cm = std::cos(mount_pitch), sm = std::sin(mount_pitch);
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

bool segment_hits_cylinder(
  const Eigen::Vector3d & p0, const Eigen::Vector3d & p1, const Eigen::Vector3d & center,
  double radius, double z_min, double z_max, double & out_alpha)
{
  const Eigen::Vector2d d = (p1 - p0).head<2>();
  const Eigen::Vector2d f = (p0 - center).head<2>();
  const double a = d.squaredNorm();
  if (a < 1e-12) return false;
  const double b = 2.0 * f.dot(d);
  const double c = f.squaredNorm() - radius * radius;
  const double disc = b * b - 4.0 * a * c;
  if (disc < 0.0) return false;
  const double s = std::sqrt(disc);
  const double t1 = (-b - s) / (2.0 * a);
  const double t2 = (-b + s) / (2.0 * a);
  const double t = (t1 >= 0.0 && t1 <= 1.0) ? t1 : ((t2 >= 0.0 && t2 <= 1.0) ? t2 : -1.0);
  if (t < 0.0) return false;
  const double z_hit = p0.z() + t * (p1.z() - p0.z());
  if (z_hit < z_min || z_hit > z_max) return false;
  out_alpha = t;
  return true;
}
}  // namespace
Planner::Planner(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  yaw_offset_.store(tools::read<double>(yaml, "yaw_offset") / 57.3, std::memory_order_relaxed);
  pitch_offset_.store(tools::read<double>(yaml, "pitch_offset") / 57.3, std::memory_order_relaxed);
  if (yaml["muzzle_z_offset"]) muzzle_z_offset_ = yaml["muzzle_z_offset"].as<double>();
  if (yaml["air_k"]) air_k_ = yaml["air_k"].as<double>();
  fire_thresh_ = tools::read<double>(yaml, "fire_thresh");
  if (yaml["shoot_offset"]) shoot_offset_ = yaml["shoot_offset"].as<int>();
  if (yaml["fire_margin"]) fire_margin_ = yaml["fire_margin"].as<double>();
  if (yaml["max_fire_angle"]) max_fire_angle_ = yaml["max_fire_angle"].as<double>() / 57.3;
  decision_speed_ = tools::read<double>(yaml, "decision_speed");
  high_speed_delay_time_ = tools::read<double>(yaml, "high_speed_delay_time");
  low_speed_delay_time_ = tools::read<double>(yaml, "low_speed_delay_time");

  setup_yaw_solver(config_path);
  setup_pitch_solver(config_path);

  constexpr double R2D = 180.0 / M_PI;
  constexpr double D2R = M_PI / 180.0;
  auto & rp = tools::RuntimeParams::instance();
  rp.add(
    "planner.yaw_offset", "Planner 偏航零偏 (deg)", -15.0, 15.0,
    [this] { return yaw_offset_.load(std::memory_order_relaxed) * R2D; },
    [this](double v) { yaw_offset_.store(v * D2R, std::memory_order_relaxed); }, "yaw_offset");
  rp.add(
    "planner.pitch_offset", "Planner 俯仰零偏 (deg)", -15.0, 15.0,
    [this] { return pitch_offset_.load(std::memory_order_relaxed) * R2D; },
    [this](double v) { pitch_offset_.store(v * D2R, std::memory_order_relaxed); }, "pitch_offset");
}

Eigen::Vector3d Planner::muzzle_offset_in_world(double yaw, double pitch, double roll) const
{
  Eigen::Vector3d m_in_g = active_self_motion_.muzzle_in_gimbal;
  if (m_in_g.squaredNorm() < 1e-12) {
    m_in_g = Eigen::Vector3d(0.0, 0.0, muzzle_z_offset_);
  }
  if (m_in_g.squaredNorm() < 1e-12) return Eigen::Vector3d::Zero();

  const Eigen::Matrix3d R_g2w =
    (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
     Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
     Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
      .toRotationMatrix();
  return R_g2w * m_in_g;
}

bool Planner::check_ballistic_hit(
  const Target & hit_target, double bullet_speed, double fly_time, double muzzle_yaw,
  double muzzle_pitch) const
{
  const auto plates = hit_target.armor_xyzayp_list();
  if (plates.empty()) return false;

  const double armor_w = (hit_target.armor_type == ArmorType::big) ? 0.230 : 0.135;
  const double armor_h = (hit_target.armor_type == ArmorType::big) ? 0.127 : 0.125;
  const double mount_pitch =
    (hit_target.name == ArmorName::outpost) ? (-15.0 * M_PI / 180.0) : (15.0 * M_PI / 180.0);
  const double allow_half_w = std::max(0.0, fire_margin_ * 0.5 * armor_w);
  const double allow_half_h = std::max(0.0, fire_margin_ * 0.5 * armor_h);

  // 考虑我方滚转角 roll 带来的炮口 3D 偏移，以及我方世界系线速度 self_vel 与空气阻力 air_k_
  const Eigen::Vector3d muzzle_pos =
    muzzle_offset_in_world(muzzle_yaw, muzzle_pitch, active_self_motion_.roll);
  const double pitch_up = -muzzle_pitch;

  // 在 fly_time 附近取两段 3D 弹道微段（共覆盖 ±0.03s ≈ ±0.75m），既准确包含重力/阻力弯曲，
  // 也能正确检测换板过渡期间相邻装甲板的命中窗口。
  const double t_mid = std::max(0.005, fly_time);
  const double t_lo = std::max(0.001, t_mid - 0.030);
  const double t_hi = t_mid + 0.030;
  const Eigen::Vector3d p_lo =
    muzzle_pos +
    tools::ballistic_point_3d(
      bullet_speed, muzzle_yaw, pitch_up, active_self_motion_.vel, air_k_, t_lo);
  const Eigen::Vector3d p_mid =
    muzzle_pos +
    tools::ballistic_point_3d(
      bullet_speed, muzzle_yaw, pitch_up, active_self_motion_.vel, air_k_, t_mid);
  const Eigen::Vector3d p_hi =
    muzzle_pos +
    tools::ballistic_point_3d(
      bullet_speed, muzzle_yaw, pitch_up, active_self_motion_.vel, air_k_, t_hi);

  // 估算敌方机身遮挡圆柱（防止误判穿透机身打中背面/侧后装甲）
  Eigen::Vector3d body_center = Eigen::Vector3d::Zero();
  double body_radius = 0.0;
  const bool has_body = (plates.size() >= 2);
  if (has_body) {
    for (const auto & p : plates) body_center += p.head<3>();
    body_center /= static_cast<double>(plates.size());
    double sum_r = 0.0;
    for (const auto & p : plates) sum_r += (p.head<3>() - body_center).head<2>().norm();
    body_radius = std::max(0.08, (sum_r / static_cast<double>(plates.size())) - 0.035);
  }

  const Eigen::Vector3d segs[2][2] = {{p_lo, p_mid}, {p_mid, p_hi}};
  const double cos_max_inc = std::cos(max_fire_angle_);

  for (int s = 0; s < 2; ++s) {
    const Eigen::Vector3d & seg_a = segs[s][0];
    const Eigen::Vector3d & seg_b = segs[s][1];
    const Eigen::Vector3d seg_dir = seg_b - seg_a;
    const double seg_len = seg_dir.norm();
    if (seg_len < 1e-6) continue;

    double body_alpha = 2.0;
    const bool hit_body =
      has_body &&
      segment_hits_cylinder(
        seg_a, seg_b, body_center, body_radius, body_center.z() - 0.16, body_center.z() + 0.22,
        body_alpha);

    for (const auto & plate : plates) {
      const Eigen::Vector3d center = plate.head<3>();
      const Eigen::Matrix3d R = armor_plate_rotation(plate[4], plate[5], plate[3], mount_pitch);
      const Eigen::Vector3d a = R.transpose() * (seg_a - center);
      const Eigen::Vector3d b = R.transpose() * (seg_b - center);

      // 仅允许正面入射（装甲局部系 -X 为正面外法向，弹丸由 -X 飞向 +X）
      const double dx = b.x() - a.x();
      if (dx <= 1e-6) continue;
      if (a.x() > 0.0 || b.x() < 0.0) continue;

      // 真实 3D 入射角检查（含敌方倾角与弹道俯仰角）
      const double cos_inc = dx / seg_len;
      if (cos_inc < cos_max_inc) continue;

      const double alpha = std::clamp(-a.x() / dx, 0.0, 1.0);
      if (hit_body && alpha > body_alpha + 1e-4) continue;

      const Eigen::Vector3d hit = a + alpha * (b - a);
      if (std::abs(hit.y()) <= allow_half_w && std::abs(hit.z()) <= allow_half_h) {
        return true;
      }
    }
  }
  return false;
}

Plan Planner::plan(
  Target target, double bullet_speed, std::optional<Eigen::Vector2d> gimbal_yp,
  const SelfMotion & self_motion)
{
  active_self_motion_ = self_motion;

  // 0. Check bullet speed
  if (bullet_speed < 10 || bullet_speed > 25) {
    bullet_speed = 22;
  }

  // 1. 迭代求解 3D 跑打飞行时间（最多 6 次，收敛阈值 1ms），融合我方速度 self_vel 与 roll 炮口偏移
  double fly_time = 0.0;
  for (int iter = 0; iter < 6; ++iter) {
    Target pred = target;
    pred.predict(fly_time);
    Eigen::Vector3d xyz;
    double min_dist = 1e10;
    for (const auto & xyza : pred.armor_xyza_list()) {
      double dist = xyza.head<2>().norm();
      if (dist < min_dist) {
        min_dist = dist;
        xyz = xyza.head<3>();
      }
    }
    const double azim_est = std::atan2(xyz.y(), xyz.x());
    const double pitch_est = -std::atan2(xyz.z() - muzzle_z_offset_, std::max(0.2, min_dist));
    const Eigen::Vector3d rel_xyz =
      xyz - muzzle_offset_in_world(azim_est, pitch_est, active_self_motion_.roll);
    auto bullet_traj =
      tools::Trajectory3D(bullet_speed, rel_xyz, active_self_motion_.vel, air_k_);
    if (bullet_traj.unsolvable) {
      tools::logger()->warn("Unsolvable target {:.2f}", bullet_speed);
      active_delay_time_ = 0.0;
      return {false};
    }
    if (std::abs(bullet_traj.fly_time - fly_time) < 1e-3) {
      fly_time = bullet_traj.fly_time;
      break;
    }
    fly_time = bullet_traj.fly_time;
  }
  target.predict(fly_time);

  // 2. Get trajectory
  double yaw0;
  Trajectory traj;
  try {
    yaw0 = aim(target, bullet_speed)(0);
    traj = get_trajectory(target, yaw0, bullet_speed);
  } catch (const std::exception & e) {
    tools::logger()->warn("Unsolvable target {:.2f}", bullet_speed);
    active_delay_time_ = 0.0;
    return {false};
  }

  // 3. Solve yaw
  Eigen::VectorXd x0(2);
  x0 << traj(0, 0), traj(1, 0);
  tiny_set_x0(yaw_solver_, x0);

  double yaw_u_lo = -1e9, yaw_u_hi = 1e9;
  if (use_yaw_plant_) {
    // 动态盒约束：限制预补偿指令不超过当前锯齿幅值 + 少量前馈余量（±1.5°），封死切板尖峰
    const double margin = 0.026;
    yaw_u_lo = traj.row(0).minCoeff() - margin;
    yaw_u_hi = traj.row(0).maxCoeff() + margin;
    yaw_solver_->work->u_min.setConstant(yaw_u_lo);
    yaw_solver_->work->u_max.setConstant(yaw_u_hi);
  }
  yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON);
  tiny_solve(yaw_solver_);

  // 4. Solve pitch
  x0 << traj(2, 0), traj(3, 0);
  tiny_set_x0(pitch_solver_, x0);

  double pitch_u_lo = -1e9, pitch_u_hi = 1e9;
  if (use_pitch_plant_) {
    const double margin = 0.026;
    pitch_u_lo = traj.row(2).minCoeff() - margin;
    pitch_u_hi = traj.row(2).maxCoeff() + margin;
    pitch_solver_->work->u_min.setConstant(pitch_u_lo);
    pitch_solver_->work->u_max.setConstant(pitch_u_hi);
  }
  pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);
  tiny_solve(pitch_solver_);

  Plan plan;
  plan.control = true;

  plan.target_yaw = tools::limit_rad(traj(0, HALF_HORIZON) + yaw0);
  plan.target_pitch = traj(2, HALF_HORIZON);

  const double yaw_cmd_rel = use_yaw_plant_
                               ? std::clamp(yaw_solver_->work->u(0, HALF_HORIZON), yaw_u_lo, yaw_u_hi)
                               : yaw_solver_->work->x(0, HALF_HORIZON);
  plan.yaw = tools::limit_rad(yaw_cmd_rel + yaw0);
  plan.yaw_vel = yaw_solver_->work->x(1, HALF_HORIZON);
  plan.yaw_acc = use_yaw_plant_ ? 0.0f : static_cast<float>(yaw_solver_->work->u(0, HALF_HORIZON));

  plan.pitch = use_pitch_plant_
                 ? std::clamp(pitch_solver_->work->u(0, HALF_HORIZON), pitch_u_lo, pitch_u_hi)
                 : pitch_solver_->work->x(0, HALF_HORIZON);
  plan.pitch_vel = pitch_solver_->work->x(1, HALF_HORIZON);
  plan.pitch_acc =
    use_pitch_plant_ ? 0.0f : static_cast<float>(pitch_solver_->work->u(0, HALF_HORIZON));

  const int delay_steps =
    std::clamp(static_cast<int>(std::lround(active_delay_time_ / DT)), 0, HALF_HORIZON - 1);
  const int now_idx = HALF_HORIZON - delay_steps;
  const double now_aim_yaw = tools::limit_rad(traj(0, now_idx) + yaw0);
  const double now_aim_pitch = traj(2, now_idx);
  plan.ref_yaw = static_cast<float>(now_aim_yaw);
  plan.ref_pitch = static_cast<float>(now_aim_pitch);

  // 回退 active_delay_time_ 到“此刻开火、子弹飞抵目标时 (now + fly_time)”的真实装甲板位姿（含自身跑打位移），
  // 用 3D 弹道与所有正面装甲板直接求交，在保证 >90% 命中率的同时最大化连续开火区间（含换板过渡期）。
  Target hit_now = target;
  if (active_delay_time_ > 0.0) hit_now.predict(-active_delay_time_, active_self_motion_.vel);

  if (gimbal_yp.has_value()) {
    // 扣除静态零偏后得到炮口在世界系的真实出膛弹道方向
    const double y_off = yaw_offset_.load(std::memory_order_relaxed);
    const double p_off = pitch_offset_.load(std::memory_order_relaxed);
    const double cr = std::cos(active_self_motion_.roll);
    const double sr = std::sin(active_self_motion_.roll);
    const double shot_yaw = tools::limit_rad((*gimbal_yp)[0] - (y_off * cr - p_off * sr));
    const double shot_pitch = (*gimbal_yp)[1] + (p_off * cr + y_off * sr);
    plan.fire = check_ballistic_hit(hit_now, bullet_speed, fly_time, shot_yaw, shot_pitch);
  } else {
    // 无云台反馈时退回 MPC 规划状态检查
    const int fire_idx = std::clamp(now_idx + shoot_offset_, 0, HORIZON - 1);
    const double plan_err = std::hypot(
      traj(0, fire_idx) - yaw_solver_->work->x(0, fire_idx),
      traj(2, fire_idx) - pitch_solver_->work->x(0, fire_idx));
    const double planned_yaw = tools::limit_rad(yaw_solver_->work->x(0, now_idx) + yaw0);
    const double planned_pitch = pitch_solver_->work->x(0, now_idx);
    plan.fire = (plan_err < fire_thresh_) ||
                check_ballistic_hit(hit_now, bullet_speed, fly_time, planned_yaw, planned_pitch);
  }
  active_delay_time_ = 0.0;
  return plan;
}

Plan Planner::plan(
  std::optional<Target> target, double bullet_speed, std::optional<Eigen::Vector2d> gimbal_yp,
  double frame_age_s, const SelfMotion & self_motion)
{
  if (!target.has_value()) return {false};

  double delay_time =
    std::abs(target->ekf_x()[7]) > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;
  (void)frame_age_s;
  // target->predict(future, self_motion.vel) 将目标从图像帧时刻推至 now + delay_time，
  // 同时扣除我方在这段时间内的跑打位移，使位置相对发射时刻的云台原点。
  active_delay_time_ = delay_time;

  auto future = std::chrono::steady_clock::now() + std::chrono::microseconds(int(delay_time * 1e6));

  target->predict(future, self_motion.vel);

  return plan(*target, bullet_speed, gimbal_yp, self_motion);
}

void Planner::setup_yaw_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_yaw_acc = tools::read<double>(yaml, "max_yaw_acc");
  auto Q_yaw = tools::read<std::vector<double>>(yaml, "Q_yaw");
  auto R_yaw = tools::read<std::vector<double>>(yaml, "R_yaw");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  double u_limit = max_yaw_acc;

  if (yaml["gimbal_yaw_wn"] && yaml["gimbal_yaw_wn"].as<double>() > 0.0) {
    const double wn = yaml["gimbal_yaw_wn"].as<double>();
    const double zeta = yaml["gimbal_yaw_zeta"] ? yaml["gimbal_yaw_zeta"].as<double>() : 1.0;
    discretize_second_order_zoh(wn, zeta, DT, A, B);
    use_yaw_plant_ = true;
    u_limit = yaml["max_yaw_cmd_offset"] ? yaml["max_yaw_cmd_offset"].as<double>() : 0.6;
    // 位置指令模型与状态同量纲（rad），默认 Q=[80, 0], R=[1]（不惩罚切板虚假速度尖峰，稳态增益 98.8%）
    Q_yaw = yaml["Q_yaw_plant"] ? yaml["Q_yaw_plant"].as<std::vector<double>>()
                                : std::vector<double>{80.0, 0.0};
    R_yaw = yaml["R_yaw_plant"] ? yaml["R_yaw_plant"].as<std::vector<double>>()
                                : std::vector<double>{1.0};
    tools::logger()->info(
      "[Planner] Yaw 启用辨识传递函数模型: wn={:.2f} rad/s, zeta={:.3f}, Q=[{:.1f},{:.2f}], R={:.2f}",
      wn, zeta, Q_yaw[0], Q_yaw[1], R_yaw[0]);
  }

  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_yaw.data());
  Eigen::Matrix<double, 1, 1> R(R_yaw.data());
  tiny_setup(&yaw_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -u_limit);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, u_limit);
  tiny_set_bound_constraints(yaw_solver_, x_min, x_max, u_min, u_max);

  yaw_solver_->settings->max_iter = 10;
}

void Planner::setup_pitch_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_pitch_acc = tools::read<double>(yaml, "max_pitch_acc");
  auto Q_pitch = tools::read<std::vector<double>>(yaml, "Q_pitch");
  auto R_pitch = tools::read<std::vector<double>>(yaml, "R_pitch");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  double u_limit = max_pitch_acc;

  if (yaml["gimbal_pitch_wn"] && yaml["gimbal_pitch_wn"].as<double>() > 0.0) {
    const double wn = yaml["gimbal_pitch_wn"].as<double>();
    const double zeta = yaml["gimbal_pitch_zeta"] ? yaml["gimbal_pitch_zeta"].as<double>() : 1.0;
    discretize_second_order_zoh(wn, zeta, DT, A, B);
    use_pitch_plant_ = true;
    u_limit = yaml["max_pitch_cmd_offset"] ? yaml["max_pitch_cmd_offset"].as<double>() : 0.7;
    Q_pitch = yaml["Q_pitch_plant"] ? yaml["Q_pitch_plant"].as<std::vector<double>>()
                                    : std::vector<double>{80.0, 0.0};
    R_pitch = yaml["R_pitch_plant"] ? yaml["R_pitch_plant"].as<std::vector<double>>()
                                    : std::vector<double>{1.0};
    tools::logger()->info(
      "[Planner] Pitch 启用辨识传递函数模型: wn={:.2f} rad/s, zeta={:.3f}, Q=[{:.1f},{:.2f}], R={:.2f}",
      wn, zeta, Q_pitch[0], Q_pitch[1], R_pitch[0]);
  }

  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_pitch.data());
  Eigen::Matrix<double, 1, 1> R(R_pitch.data());
  tiny_setup(&pitch_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -u_limit);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, u_limit);
  tiny_set_bound_constraints(pitch_solver_, x_min, x_max, u_min, u_max);

  pitch_solver_->settings->max_iter = 10;
}

Eigen::Matrix<double, 2, 1> Planner::aim(const Target & target, double bullet_speed)
{
  Eigen::Vector3d xyz;
  double yaw;
  auto min_dist = 1e10;

  for (auto & xyza : target.armor_xyza_list()) {
    auto dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();
      yaw = xyza[3];
    }
  }
  debug_xyza = Eigen::Vector4d(xyz.x(), xyz.y(), xyz.z(), yaw);

  const double azim_est = std::atan2(xyz.y(), xyz.x());
  const double pitch_est = -std::atan2(xyz.z() - muzzle_z_offset_, std::max(0.2, min_dist));
  const Eigen::Vector3d rel_xyz =
    xyz - muzzle_offset_in_world(azim_est, pitch_est, active_self_motion_.roll);

  auto bullet_traj = tools::Trajectory3D(bullet_speed, rel_xyz, active_self_motion_.vel, air_k_);
  if (bullet_traj.unsolvable) throw std::runtime_error("Unsolvable bullet trajectory!");

  const double y_off = yaw_offset_.load(std::memory_order_relaxed);
  const double p_off = pitch_offset_.load(std::memory_order_relaxed);
  const double cr = std::cos(active_self_motion_.roll);
  const double sr = std::sin(active_self_motion_.roll);
  return {
    tools::limit_rad(bullet_traj.yaw + y_off * cr - p_off * sr),
    -bullet_traj.pitch - (p_off * cr + y_off * sr)};
}

Trajectory Planner::get_trajectory(Target target, double yaw0, double bullet_speed)
{
  Trajectory traj;
  Eigen::Vector4d center_xyza = debug_xyza;

  target.predict(-DT * (HALF_HORIZON + 1), active_self_motion_.vel);
  auto yaw_pitch_last = aim(target, bullet_speed);

  target.predict(DT, active_self_motion_.vel);  // [0] = -HALF_HORIZON * DT -> [HALF_HORIZON] = 0
  auto yaw_pitch = aim(target, bullet_speed);

  for (int i = 0; i < HORIZON; i++) {
    if (i == HALF_HORIZON) center_xyza = debug_xyza;
    target.predict(DT, active_self_motion_.vel);
    auto yaw_pitch_next = aim(target, bullet_speed);

    auto yaw_vel = std::clamp(
      tools::limit_rad(yaw_pitch_next(0) - yaw_pitch_last(0)) / (2 * DT), -15.0, 15.0);
    auto pitch_vel = std::clamp((yaw_pitch_next(1) - yaw_pitch_last(1)) / (2 * DT), -10.0, 10.0);

    traj.col(i) << tools::limit_rad(yaw_pitch(0) - yaw0), yaw_vel, yaw_pitch(1), pitch_vel;

    yaw_pitch_last = yaw_pitch;
    yaw_pitch = yaw_pitch_next;
  }

  debug_xyza = center_xyza;
  return traj;
}

}  // namespace auto_aim