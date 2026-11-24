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
  if (yaml["fire_prob_thresh"]) {
    fire_prob_thresh_.store(
      std::clamp(yaml["fire_prob_thresh"].as<double>(), 0.0, 0.999), std::memory_order_relaxed);
  }
  if (yaml["fire_prob_thresh_far"]) {
    fire_prob_thresh_far_.store(
      std::clamp(yaml["fire_prob_thresh_far"].as<double>(), 0.0, 0.999), std::memory_order_relaxed);
  }
  if (yaml["fire_prob_dist_near"]) {
    fire_prob_dist_near_ = std::max(0.1, yaml["fire_prob_dist_near"].as<double>());
  }
  if (yaml["fire_prob_dist_far"]) {
    fire_prob_dist_far_ =
      std::max(fire_prob_dist_near_ + 0.1, yaml["fire_prob_dist_far"].as<double>());
  }
  if (yaml["use_fire_mlp"]) {
    use_fire_mlp_.store(yaml["use_fire_mlp"].as<bool>(), std::memory_order_relaxed);
  }
  if (yaml["gun_sigma_yaw_deg"]) {
    gun_sigma_yaw_rad_.store(
      std::clamp(yaml["gun_sigma_yaw_deg"].as<double>(), 0.0, 5.0) * M_PI / 180.0,
      std::memory_order_relaxed);
  }
  if (yaml["gun_sigma_pitch_deg"]) {
    gun_sigma_pitch_rad_.store(
      std::clamp(yaml["gun_sigma_pitch_deg"].as<double>(), 0.0, 5.0) * M_PI / 180.0,
      std::memory_order_relaxed);
  }
  if (yaml["fire_delay_jitter_s"]) {
    fire_delay_jitter_s_ = std::max(0.0, yaml["fire_delay_jitter_s"].as<double>());
  }
  if (yaml["max_fire_rate_hz"]) {
    max_fire_rate_hz_.store(
      std::clamp(yaml["max_fire_rate_hz"].as<double>(), 1.0, 25.0), std::memory_order_relaxed);
  }
  if (yaml["min_fire_rate_hz"]) {
    min_fire_rate_hz_.store(
      std::clamp(
        yaml["min_fire_rate_hz"].as<double>(), 1.0,
        max_fire_rate_hz_.load(std::memory_order_relaxed)),
      std::memory_order_relaxed);
  }
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
  rp.add(
    "planner.fire_prob_thresh", "近距/固定命中概率开火阈值 (0=关闭)", 0.0, 0.99,
    [this] { return fire_prob_thresh_.load(std::memory_order_relaxed); },
    [this](double v) { fire_prob_thresh_.store(std::clamp(v, 0.0, 0.999), std::memory_order_relaxed); },
    "fire_prob_thresh");
  rp.add(
    "planner.fire_prob_thresh_far", "远距 Peek 命中概率阈值 (0=不随距离插值)", 0.0, 0.99,
    [this] { return fire_prob_thresh_far_.load(std::memory_order_relaxed); },
    [this](double v) {
      fire_prob_thresh_far_.store(std::clamp(v, 0.0, 0.999), std::memory_order_relaxed);
    },
    "fire_prob_thresh_far");
  rp.add(
    "planner.use_fire_mlp", "启用 Layer 3 残差 MLP 校准 (0/1)", 0.0, 1.0,
    [this] { return use_fire_mlp_.load(std::memory_order_relaxed) ? 1.0 : 0.0; },
    [this](double v) { use_fire_mlp_.store(v > 0.5, std::memory_order_relaxed); }, "use_fire_mlp",
    1.0);
  rp.add(
    "planner.gun_sigma_yaw", "弹道偏航散布标准差 (deg)", 0.0, 5.0,
    [this] { return gun_sigma_yaw_rad_.load(std::memory_order_relaxed) * R2D; },
    [this](double v) {
      gun_sigma_yaw_rad_.store(std::clamp(v, 0.0, 5.0) * D2R, std::memory_order_relaxed);
    },
    "gun_sigma_yaw_deg");
  rp.add(
    "planner.gun_sigma_pitch", "弹道俯仰散布标准差 (deg)", 0.0, 5.0,
    [this] { return gun_sigma_pitch_rad_.load(std::memory_order_relaxed) * R2D; },
    [this](double v) {
      gun_sigma_pitch_rad_.store(std::clamp(v, 0.0, 5.0) * D2R, std::memory_order_relaxed);
    },
    "gun_sigma_pitch_deg");
  rp.add(
    "planner.fire_mode", "火控战术模式 (0=常规, 1=爆发收割-0.15, 2=省弹控eta+0.08)", 0.0, 2.0,
    [this] { return static_cast<double>(fire_tactical_mode_.load(std::memory_order_relaxed)); },
    [this](double v) {
      fire_tactical_mode_.store(std::clamp(static_cast<int>(std::lround(v)), 0, 2),
                                std::memory_order_relaxed);
    },
    "", 1.0);
  rp.add(
    "planner.level_phase", "步兵等级阶段 (0=未知, 1=未满级发育-0.08, 2=已封顶+0.05)", 0.0, 2.0,
    [this] { return static_cast<double>(level_phase_.load(std::memory_order_relaxed)); },
    [this](double v) {
      level_phase_.store(std::clamp(static_cast<int>(std::lround(v)), 0, 2),
                         std::memory_order_relaxed);
    },
    "", 1.0);
  rp.add(
    "planner.armor_zone", "§5.2 目标加固区 (0=默认, 1=未加固x1.5 -0.08, 2=已加固x0.5 +0.12)", 0.0, 2.0,
    [this] { return static_cast<double>(armor_zone_.load(std::memory_order_relaxed)); },
    [this](double v) {
      armor_zone_.store(std::clamp(static_cast<int>(std::lround(v)), 0, 2),
                        std::memory_order_relaxed);
    },
    "", 1.0);
  rp.add(
    "planner.heat_ratio", "下位机上报枪口热量占比 [0, 1] (>0.7 非爆发时自动抬高阈值保必中)", 0.0, 1.0,
    [this] { return heat_ratio_.load(std::memory_order_relaxed); },
    [this](double v) { heat_ratio_.store(std::clamp(v, 0.0, 1.0), std::memory_order_relaxed); },
    "", 0.05);
  rp.add(
    "planner.max_fire_rate_hz", "单块装甲板防吞弹建议射频上限 (Hz)", 1.0, 25.0,
    [this] { return max_fire_rate_hz_.load(std::memory_order_relaxed); },
    [this](double v) {
      const double hi = std::clamp(v, 1.0, 25.0);
      max_fire_rate_hz_.store(hi, std::memory_order_relaxed);
      if (min_fire_rate_hz_.load(std::memory_order_relaxed) > hi) {
        min_fire_rate_hz_.store(hi, std::memory_order_relaxed);
      }
    },
    "max_fire_rate_hz", 1.0);
}

double Planner::effective_fire_thresh(double dist_m) const
{
  const double p_near = fire_prob_thresh_.load(std::memory_order_relaxed);
  if (p_near <= 1e-6) return 0.0;

  double base = p_near;
  const double p_far = fire_prob_thresh_far_.load(std::memory_order_relaxed);
  if (p_far > 1e-6) {
    const double span = std::max(0.1, fire_prob_dist_far_ - fire_prob_dist_near_);
    const double alpha = std::clamp((dist_m - fire_prob_dist_near_) / span, 0.0, 1.0);
    base = p_near + alpha * (p_far - p_near);
  }

  double delta = 0.0;
  switch (static_cast<LevelPhase>(level_phase_.load(std::memory_order_relaxed))) {
    case LevelPhase::Leveling: delta -= 0.08; break;
    case LevelPhase::Capped:   delta += 0.05; break;
    default: break;
  }
  const auto mode = static_cast<FireTacticalMode>(fire_tactical_mode_.load(std::memory_order_relaxed));
  switch (mode) {
    case FireTacticalMode::Burst: delta -= 0.15; break;
    case FireTacticalMode::Eco:   delta += 0.08; break;
    default: break;
  }
  switch (static_cast<ArmorZone>(armor_zone_.load(std::memory_order_relaxed))) {
    case ArmorZone::Unreinforced: delta -= 0.08; break;
    case ArmorZone::Reinforced:   delta += 0.12; break;
    default: break;
  }

  // 高热量保护协同：非强制爆发模式下，热量占比 > 0.70 时平滑抬高阈值（至多 +0.08），
  // 把最后宝贵热量留给高命中概率窗口；硬限幅与防超热仍由电控 1 kHz 闭环兜底。
  const double heat = heat_ratio_.load(std::memory_order_relaxed);
  if (mode != FireTacticalMode::Burst && heat > 0.70) {
    delta += 0.08 * std::clamp((heat - 0.70) / 0.30, 0.0, 1.0);
  }

  return std::clamp(base + delta, 0.45, 0.90);
}

float Planner::recommend_fire_rate_hz(double p_hit, double eff_thresh, bool plate_switched) const
{
  const double max_hz = max_fire_rate_hz_.load(std::memory_order_relaxed);
  const double min_hz = std::min(min_fire_rate_hz_.load(std::memory_order_relaxed), max_hz);
  const auto mode = static_cast<FireTacticalMode>(fire_tactical_mode_.load(std::memory_order_relaxed));

  // 强制爆发模式（操作手按键或雷达残血触发）或刚切换至新装甲板（独立检测死区）：
  // 直接输出单板防吞弹上限射频
  if (mode == FireTacticalMode::Burst || plate_switched) {
    return static_cast<float>(max_hz);
  }

  const double ref_t = (eff_thresh > 1e-6) ? eff_thresh : 0.70;
  const double span = std::max(0.05, 0.92 - ref_t);
  const double alpha = std::clamp((p_hit - ref_t) / span, 0.0, 1.0);
  double rate = min_hz + alpha * (max_hz - min_hz);

  const double heat = heat_ratio_.load(std::memory_order_relaxed);
  if (heat > 0.70) {
    const double h_alpha = std::clamp((heat - 0.70) / 0.30, 0.0, 1.0);
    rate = rate - h_alpha * (rate - min_hz);
  }
  return static_cast<float>(std::clamp(rate, min_hz, max_hz));
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
  double muzzle_pitch, FireDiagnostics * out_diag) const
{
  const auto plates = hit_target.armor_xyzayp_list();
  if (plates.empty()) return false;

  const double armor_w = (hit_target.armor_type == ArmorType::big) ? 0.230 : 0.135;
  const double armor_h = (hit_target.armor_type == ArmorType::big) ? 0.127 : 0.125;
  const double half_w = 0.5 * armor_w;
  const double half_h = 0.5 * armor_h;
  const double mount_pitch =
    (hit_target.name == ArmorName::outpost) ? (-15.0 * M_PI / 180.0) : (15.0 * M_PI / 180.0);
  const double allow_half_w = std::max(0.0, fire_margin_ * half_w);
  const double allow_half_h = std::max(0.0, fire_margin_ * half_h);

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
    double min_r = 1e9;
    for (const auto & p : plates) {
      min_r = std::min(min_r, (p.head<3>() - body_center).head<2>().norm());
    }
    body_radius = std::max(0.06, min_r - 0.055);
  }

  const Eigen::Vector3d segs[2][2] = {{p_lo, p_mid}, {p_mid, p_hi}};
  const double cos_max_inc = std::cos(max_fire_angle_);
  const double prob_thresh = fire_prob_thresh_.load(std::memory_order_relaxed);
  const bool use_prob = (prob_thresh > 1e-6);

  // Layer 2 物理不确定度参数：融合弹道实测散布 + EKF 协方差 + 发弹延迟抖动
  constexpr double BULLET_RADIUS = 0.0085;  // 17mm 弹丸半径 (m)
  constexpr double SIGMA_MIN = 0.005;       // 5 mm 最小数值保护
  const double eff_half_w = half_w + BULLET_RADIUS;
  const double eff_half_h = half_h + BULLET_RADIUS;
  const double gun_sy = gun_sigma_yaw_rad_.load(std::memory_order_relaxed);
  const double gun_sp = gun_sigma_pitch_rad_.load(std::memory_order_relaxed);

  const auto & x = hit_target.ekf_x();
  const auto & P = hit_target.ekf().P;
  const double target_w = std::abs(x[7]);
  const double target_r = std::clamp(x[8], 0.12, 0.45);
  double sig_pos_xy = 0.0, sig_pos_z = 0.0, sig_phase = 0.0, sig_r = 0.0;
  if (P.rows() >= 9 && P.cols() >= 9) {
    sig_pos_xy = std::sqrt(std::max(0.0, 0.5 * (P(0, 0) + P(2, 2))));
    sig_pos_z = std::sqrt(std::max(0.0, P(4, 4)));
    // hit_target 已经 predict 到命中时刻 (now + fly_time)，P(6,6) 已含飞行时间内的相位方差传播
    sig_phase = std::sqrt(std::max(0.0, P(6, 6)));
    sig_r = std::sqrt(std::max(0.0, P(8, 8)));
  }
  const double sig_y_delay = target_r * target_w * fire_delay_jitter_s_;
  const double sig_y_phase = target_r * sig_phase;

  auto gauss_interval_prob = [](double mu, double half_span, double sigma) -> double {
    const double inv_s2 = 1.0 / (std::sqrt(2.0) * std::max(1e-5, sigma));
    return 0.5 * (std::erf((half_span - mu) * inv_s2) - std::erf((-half_span - mu) * inv_s2));
  };

  double best_norm_r = 1e9;
  bool has_near_hit = false;
  int near_plate_id = -1;
  double near_y_norm = 0.0, near_z_norm = 0.0, near_cos_inc = 0.0, near_dist = 0.0;
  double best_p_phys = 0.0;
  double best_sig_y = SIGMA_MIN, best_sig_z = SIGMA_MIN;
  bool margin_hit = false;
  int margin_plate_id = -1;
  double margin_y_norm = 0.0, margin_z_norm = 0.0, margin_cos_inc = 0.0, margin_dist = 0.0;
  double margin_p_phys = 0.0, margin_sig_y = SIGMA_MIN, margin_sig_z = SIGMA_MIN;

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

    for (size_t pi = 0; pi < plates.size(); ++pi) {
      const auto & plate = plates[pi];
      const Eigen::Vector3d center = plate.head<3>();
      const Eigen::Matrix3d R = armor_plate_rotation(plate[4], plate[5], plate[3], mount_pitch);
      const Eigen::Vector3d a = R.transpose() * (seg_a - center);
      const Eigen::Vector3d b = R.transpose() * (seg_b - center);

      // 仅允许正面入射（装甲局部系 -X 为正面外法向，弹丸由 -X 飞向 +X）
      const double dx = b.x() - a.x();
      if (dx <= 1e-6) continue;
      if (a.x() > 0.0 || b.x() < 0.0) continue;

      const double cos_inc = dx / seg_len;
      const double alpha = std::clamp(-a.x() / dx, 0.0, 1.0);
      if (hit_body && alpha > body_alpha + 1e-4) continue;

      const Eigen::Vector3d hit = a + alpha * (b - a);
      const double y_norm = hit.y() / half_w;
      const double z_norm = hit.z() / half_h;
      const double plate_dist = (center - muzzle_pos).norm();

      // 计算该装甲板的物理不确定度命中概率 P_phys
      const double cos_safe = std::max(0.25, cos_inc);
      const double sin_inc = std::sqrt(std::max(0.0, 1.0 - cos_inc * cos_inc));
      const double sig_y_gun = (plate_dist * gun_sy) / cos_safe;
      const double sig_z_gun = plate_dist * gun_sp;
      const double sig_y_pos = sig_pos_xy / cos_safe;
      const double sig_y_rad = sig_r * (sin_inc / cos_safe);
      const double sig_y = std::sqrt(
        sig_y_gun * sig_y_gun + sig_y_pos * sig_y_pos + sig_y_phase * sig_y_phase +
        sig_y_rad * sig_y_rad + sig_y_delay * sig_y_delay + SIGMA_MIN * SIGMA_MIN);
      const double sig_z =
        std::sqrt(sig_z_gun * sig_z_gun + sig_pos_z * sig_pos_z + SIGMA_MIN * SIGMA_MIN);
      const double p_plate =
        gauss_interval_prob(hit.y(), eff_half_w, sig_y) *
        gauss_interval_prob(hit.z(), eff_half_h, sig_z);

      if (cos_inc >= cos_max_inc && p_plate > best_p_phys) {
        best_p_phys = p_plate;
        best_sig_y = sig_y;
        best_sig_z = sig_z;
        near_plate_id = static_cast<int>(pi);
        near_y_norm = y_norm;
        near_z_norm = z_norm;
        near_cos_inc = cos_inc;
        near_dist = plate_dist;
        has_near_hit = true;
      } else if (!has_near_hit && std::abs(y_norm) <= 1.25 && std::abs(z_norm) <= 1.25) {
        const double nr = std::hypot(y_norm, z_norm);
        if (nr < best_norm_r) {
          best_norm_r = nr;
          has_near_hit = true;
          near_plate_id = static_cast<int>(pi);
          near_y_norm = y_norm;
          near_z_norm = z_norm;
          near_cos_inc = cos_inc;
          near_dist = plate_dist;
          best_p_phys = p_plate;
          best_sig_y = sig_y;
          best_sig_z = sig_z;
        }
      }

      // 真实 3D 入射角检查（含敌方倾角与弹道俯仰角）
      if (cos_inc < cos_max_inc) continue;

      if (!use_prob && std::abs(hit.y()) <= allow_half_w && std::abs(hit.z()) <= allow_half_h) {
        if (!margin_hit || p_plate > margin_p_phys) {
          margin_hit = true;
          margin_plate_id = static_cast<int>(pi);
          margin_y_norm = y_norm;
          margin_z_norm = z_norm;
          margin_cos_inc = cos_inc;
          margin_dist = plate_dist;
          margin_p_phys = p_plate;
          margin_sig_y = sig_y;
          margin_sig_z = sig_z;
        }
      }
    }
  }

  auto eval_final_prob = [&](
                           double p_phys, double y_n, double z_n, double cos_i, double dist_m,
                           double sy_m, double sz_m, double & out_delta) -> double {
    out_delta = 0.0;
    if (!use_fire_mlp_.load(std::memory_order_relaxed) || !fire_mlp_.has_weights()) {
      return p_phys;
    }
    const double xy_r = std::max(1e-3, std::hypot(x[0], x[2]));
    const double v_tan = std::abs(-x[2] * x[1] + x[0] * x[3]) / xy_r;
    const double sig_pos =
      (P.rows() >= 9 && P.cols() >= 9) ? std::sqrt(std::max(0.0, P(0, 0) + P(2, 2) + P(4, 4))) : 0.0;
    const auto feat = FireResidualMLP::extract_features(
      y_n, z_n, cos_i, dist_m, fly_time, hit_target.armor_type == ArmorType::big, x[7], v_tan,
      active_self_motion_.vel.norm(), active_self_motion_.roll, sig_phase, sig_pos, sy_m, sz_m);
    out_delta = fire_mlp_.predict_delta_logit(feat);
    return FireResidualMLP::combine_prob(p_phys, out_delta);
  };

  if (use_prob) {
    double delta = 0.0;
    const double p_hit =
      has_near_hit
        ? eval_final_prob(
            best_p_phys, near_y_norm, near_z_norm, near_cos_inc, near_dist, best_sig_y, best_sig_z,
            delta)
        : 0.0;
    const double eff_thresh = has_near_hit ? effective_fire_thresh(near_dist) : prob_thresh;
    if (out_diag && has_near_hit) {
      out_diag->valid = true;
      out_diag->hit_y_norm = near_y_norm;
      out_diag->hit_z_norm = near_z_norm;
      out_diag->cos_inc = near_cos_inc;
      out_diag->dist = near_dist;
      out_diag->p_phys = best_p_phys;
      out_diag->delta_logit = delta;
      out_diag->p_hit = p_hit;
      out_diag->eff_thresh = eff_thresh;
      out_diag->hit_plate_id = near_plate_id;
      out_diag->sigma_y_m = best_sig_y;
      out_diag->sigma_z_m = best_sig_z;
    }
    return has_near_hit && (near_cos_inc >= cos_max_inc) && (p_hit >= eff_thresh);
  }

  if (margin_hit) {
    if (out_diag) {
      double delta = 0.0;
      const double p_hit = eval_final_prob(
        margin_p_phys, margin_y_norm, margin_z_norm, margin_cos_inc, margin_dist, margin_sig_y,
        margin_sig_z, delta);
      out_diag->valid = true;
      out_diag->hit_y_norm = margin_y_norm;
      out_diag->hit_z_norm = margin_z_norm;
      out_diag->cos_inc = margin_cos_inc;
      out_diag->dist = margin_dist;
      out_diag->p_phys = margin_p_phys;
      out_diag->delta_logit = delta;
      out_diag->p_hit = p_hit;
      out_diag->hit_plate_id = margin_plate_id;
      out_diag->sigma_y_m = margin_sig_y;
      out_diag->sigma_z_m = margin_sig_z;
    }
    return true;
  }
  if (out_diag && has_near_hit) {
    double delta = 0.0;
    const double p_hit = eval_final_prob(
      best_p_phys, near_y_norm, near_z_norm, near_cos_inc, near_dist, best_sig_y, best_sig_z,
      delta);
    out_diag->valid = true;
    out_diag->hit_y_norm = near_y_norm;
    out_diag->hit_z_norm = near_z_norm;
    out_diag->cos_inc = near_cos_inc;
    out_diag->dist = near_dist;
    out_diag->p_phys = best_p_phys;
    out_diag->delta_logit = delta;
    out_diag->p_hit = p_hit;
    out_diag->sigma_y_m = best_sig_y;
    out_diag->sigma_z_m = best_sig_z;
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

  {
    const auto & x = hit_now.ekf_x();
    const auto & P = hit_now.ekf().P;
    plan.diag.fly_time = fly_time;
    plan.diag.dist = std::sqrt(x[0] * x[0] + x[2] * x[2] + x[4] * x[4]);
    plan.diag.is_big_armor = (hit_now.armor_type == ArmorType::big);
    plan.diag.target_w = x[7];
    const double xy_r = std::max(1e-3, std::hypot(x[0], x[2]));
    // 目标相对视线方向的水平切向速度 (m/s)
    plan.diag.target_v_tan = std::abs(-x[2] * x[1] + x[0] * x[3]) / xy_r;
    plan.diag.self_speed = active_self_motion_.vel.norm();
    plan.diag.self_roll = active_self_motion_.roll;
    plan.diag.target_jumped = hit_now.jumped;
    if (P.rows() >= 9 && P.cols() >= 9) {
      plan.diag.ekf_sigma_phase = std::sqrt(std::max(0.0, P(6, 6)));
      plan.diag.ekf_sigma_pos =
        std::sqrt(std::max(0.0, P(0, 0) + P(2, 2) + P(4, 4)));
      plan.diag.ekf_sigma_r = std::sqrt(std::max(0.0, P(8, 8)));
    }
  }

  if (gimbal_yp.has_value()) {
    // 扣除静态零偏后得到炮口在世界系的真实出膛弹道方向
    const double y_off = yaw_offset_.load(std::memory_order_relaxed);
    const double p_off = pitch_offset_.load(std::memory_order_relaxed);
    const double cr = std::cos(active_self_motion_.roll);
    const double sr = std::sin(active_self_motion_.roll);
    const double shot_yaw = tools::limit_rad((*gimbal_yp)[0] - (y_off * cr - p_off * sr));
    const double shot_pitch = (*gimbal_yp)[1] + (p_off * cr + y_off * sr);
    plan.diag.yaw_err = tools::limit_rad((*gimbal_yp)[0] - now_aim_yaw);
    plan.diag.pitch_err = (*gimbal_yp)[1] - now_aim_pitch;
    plan.fire =
      check_ballistic_hit(hit_now, bullet_speed, fly_time, shot_yaw, shot_pitch, &plan.diag);
  } else {
    // 无云台反馈时退回 MPC 规划状态检查
    const int fire_idx = std::clamp(now_idx + shoot_offset_, 0, HORIZON - 1);
    const double plan_err = std::hypot(
      traj(0, fire_idx) - yaw_solver_->work->x(0, fire_idx),
      traj(2, fire_idx) - pitch_solver_->work->x(0, fire_idx));
    const double planned_yaw = tools::limit_rad(yaw_solver_->work->x(0, now_idx) + yaw0);
    const double planned_pitch = pitch_solver_->work->x(0, now_idx);
    plan.diag.yaw_err = tools::limit_rad(planned_yaw - now_aim_yaw);
    plan.diag.pitch_err = planned_pitch - now_aim_pitch;
    plan.fire = (plan_err < fire_thresh_) ||
                check_ballistic_hit(
                  hit_now, bullet_speed, fly_time, planned_yaw, planned_pitch, &plan.diag);
  }
  if (plan.fire) {
    const int cur_plate = plan.diag.hit_plate_id;
    const bool plate_switched =
      (cur_plate >= 0 && last_fire_plate_id_ >= 0 && cur_plate != last_fire_plate_id_);
    plan.fire_plate_id = cur_plate;
    plan.fire_rate_hz =
      recommend_fire_rate_hz(plan.diag.p_hit, plan.diag.eff_thresh, plate_switched);
    plan.diag.fire_rate_hz = plan.fire_rate_hz;
    if (cur_plate >= 0) last_fire_plate_id_ = cur_plate;
  } else {
    plan.fire_plate_id = -1;
    plan.fire_rate_hz = 0.0f;
    plan.diag.fire_rate_hz = 0.0f;
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