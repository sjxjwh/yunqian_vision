#include "io/sim/sim_gimbal_model.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace io
{

namespace
{

constexpr double RPM2RADPS = 2.0 * M_PI / 60.0;
constexpr double DEG2RAD = M_PI / 180.0;

SimGimbalAxisParams parse_axis(const YAML::Node & node, SimGimbalAxisParams def)
{
  if (!node || !node.IsMap()) return def;
  if (node["inertia_kgm2"]) def.inertia_kgm2 = node["inertia_kgm2"].as<double>();
  if (node["ctrl_inertia_kgm2"]) {
    def.ctrl_inertia_kgm2 = node["ctrl_inertia_kgm2"].as<double>();
  } else {
    def.ctrl_inertia_kgm2 = def.inertia_kgm2;
  }
  if (node["coulomb_friction_nm"]) {
    def.coulomb_friction_nm = node["coulomb_friction_nm"].as<double>();
  }
  if (node["coulomb_smooth_radps"]) {
    def.coulomb_smooth_radps = node["coulomb_smooth_radps"].as<double>();
  }
  if (node["viscous_damping"]) def.viscous_damping = node["viscous_damping"].as<double>();
  if (node["torque_lag_tau_s"]) def.torque_lag_tau_s = node["torque_lag_tau_s"].as<double>();
  if (node["torque_limit_nm"]) def.torque_limit_nm = node["torque_limit_nm"].as<double>();
  if (node["rated_torque_nm"]) def.rated_torque_nm = node["rated_torque_nm"].as<double>();
  if (node["peak_torque_nm"]) def.peak_torque_nm = node["peak_torque_nm"].as<double>();
  if (node["rated_speed_rpm"]) def.rated_speed_rpm = node["rated_speed_rpm"].as<double>();
  if (node["no_load_speed_rpm"]) def.no_load_speed_rpm = node["no_load_speed_rpm"].as<double>();
  if (node["max_vel_ref_radps"]) def.max_vel_ref_radps = node["max_vel_ref_radps"].as<double>();
  if (node["max_acc_ref_radps2"]) def.max_acc_ref_radps2 = node["max_acc_ref_radps2"].as<double>();
  if (node["lqr_gain"]) {
    auto g = node["lqr_gain"].as<std::vector<double>>();
    if (g.size() == 4) {
      def.k_pos = g[0];
      def.k_vel = g[1];
      def.k_acc = g[2];
      def.k_int = g[3];
    }
  }
  if (node["integral_limit_rad_s"]) {
    def.integral_limit_rad_s = node["integral_limit_rad_s"].as<double>();
  }
  if (node["accel_lpf_tau_s"]) def.accel_lpf_tau_s = node["accel_lpf_tau_s"].as<double>();
  if (node["leso_enable"]) def.leso_enable = node["leso_enable"].as<bool>();
  if (node["leso_bandwidth_radps"]) {
    def.leso_bandwidth_radps = node["leso_bandwidth_radps"].as<double>();
  }
  if (node["leso_max_comp_nm"]) def.leso_max_comp_nm = node["leso_max_comp_nm"].as<double>();
  if (node["enable_50hz_notch"]) def.enable_50hz_notch = node["enable_50hz_notch"].as<bool>();
  return def;
}

double friction_torque(double vel_radps, const SimGimbalAxisParams & p)
{
  const double smooth = std::max(1e-3, p.coulomb_smooth_radps);
  return p.coulomb_friction_nm * std::tanh(vel_radps / smooth) + p.viscous_damping * vel_radps;
}

// 电机转速-力矩包络：同向加速受反电动势线性衰减限制，制动方向保留峰值力矩
double apply_motor_envelope(double torque_cmd, double vel_radps, const SimGimbalAxisParams & p)
{
  const double tau_sw = std::clamp(torque_cmd, -p.torque_limit_nm, p.torque_limit_nm);
  const double w_nl = std::max(1e-3, p.no_load_speed_rpm * RPM2RADPS);
  if (tau_sw * vel_radps > 0.0) {
    const double avail = p.peak_torque_nm * std::clamp(1.0 - std::abs(vel_radps) / w_nl, 0.0, 1.0);
    return std::clamp(tau_sw, -avail, avail);
  }
  return std::clamp(tau_sw, -p.peak_torque_nm, p.peak_torque_nm);
}

}  // namespace

SimGimbalModel::SimGimbalModel(const SimGimbalModelConfig & cfg) : cfg_(cfg)
{
  if (cfg_.enabled) {
    const double wn_y = std::sqrt(cfg_.yaw.k_pos / std::max(1e-6, cfg_.yaw.inertia_kgm2));
    const double zeta_y =
      cfg_.yaw.k_vel / (2.0 * std::sqrt(std::max(1e-6, cfg_.yaw.inertia_kgm2 * cfg_.yaw.k_pos)));
    const double wn_p = std::sqrt(cfg_.pitch.k_pos / std::max(1e-6, cfg_.pitch.inertia_kgm2));
    const double zeta_p =
      cfg_.pitch.k_vel /
      (2.0 * std::sqrt(std::max(1e-6, cfg_.pitch.inertia_kgm2 * cfg_.pitch.k_pos)));
    tools::logger()->info(
      "[SimGimbalModel] 已启用三号步兵 500Hz LQI+DM4310 动力学模型(7991a95): "
      "freq={:.0f}Hz, vel_acc_ff={}, yaw(J={:.4f}, tau_lim={:.1f}Nm, wn={:.1f}rad/s, zeta={:.2f}), "
      "pitch(J={:.4f}, tau_lim={:.1f}Nm, wn={:.1f}rad/s, zeta={:.2f})",
      cfg_.control_freq_hz, cfg_.use_vel_acc_ff, cfg_.yaw.inertia_kgm2, cfg_.yaw.torque_limit_nm,
      wn_y, zeta_y, cfg_.pitch.inertia_kgm2, cfg_.pitch.torque_limit_nm, wn_p, zeta_p);
  }
}

SimGimbalModel SimGimbalModel::from_yaml(const std::string & config_path)
{
  if (config_path.empty()) return SimGimbalModel{};
  auto yaml = YAML::LoadFile(config_path);
  if (!yaml["sim_gimbal_model"]) return SimGimbalModel{};

  const auto node = yaml["sim_gimbal_model"];
  SimGimbalModelConfig cfg;
  cfg.enabled = node["enabled"] ? node["enabled"].as<bool>() : false;
  if (!cfg.enabled) return SimGimbalModel{};

  if (node["control_freq_hz"]) cfg.control_freq_hz = node["control_freq_hz"].as<double>();
  if (node["use_vel_acc_ff"]) cfg.use_vel_acc_ff = node["use_vel_acc_ff"].as<bool>();
  if (node["pitch_min_deg"]) cfg.pitch_min_rad = node["pitch_min_deg"].as<double>() * DEG2RAD;
  if (node["pitch_max_deg"]) cfg.pitch_max_rad = node["pitch_max_deg"].as<double>() * DEG2RAD;

  // Yaw 默认对齐 27-wheel-leg-wbc/gimbal @ 7991a95
  SimGimbalAxisParams yaw_def;
  yaw_def.inertia_kgm2 = 0.0146;
  yaw_def.ctrl_inertia_kgm2 = 0.0146;
  yaw_def.coulomb_friction_nm = 0.25;
  yaw_def.coulomb_smooth_radps = 0.25;
  yaw_def.viscous_damping = 0.042;
  yaw_def.torque_lag_tau_s = 0.006;
  yaw_def.torque_limit_nm = 6.0;
  yaw_def.rated_torque_nm = 3.5;
  yaw_def.peak_torque_nm = 12.5;
  yaw_def.rated_speed_rpm = 120.0;
  yaw_def.no_load_speed_rpm = 200.0;
  yaw_def.max_vel_ref_radps = 16.5;
  yaw_def.max_acc_ref_radps2 = 210.0;
  yaw_def.k_pos = 24.0;
  yaw_def.k_vel = 0.96;
  yaw_def.k_acc = 0.0;
  yaw_def.k_int = 120.0;
  yaw_def.leso_enable = false;
  yaw_def.enable_50hz_notch = true;

  // Pitch 默认对齐 27-wheel-leg-wbc/gimbal @ 7991a95
  SimGimbalAxisParams pitch_def;
  pitch_def.inertia_kgm2 = 0.0115;
  pitch_def.ctrl_inertia_kgm2 = 0.0115;
  pitch_def.coulomb_friction_nm = 0.12;
  pitch_def.coulomb_smooth_radps = 0.25;
  pitch_def.viscous_damping = 0.098;
  pitch_def.torque_lag_tau_s = 0.006;
  pitch_def.torque_limit_nm = 4.5;
  pitch_def.rated_torque_nm = 3.5;
  pitch_def.peak_torque_nm = 12.5;
  pitch_def.rated_speed_rpm = 120.0;
  pitch_def.no_load_speed_rpm = 200.0;
  pitch_def.max_vel_ref_radps = 11.0;
  pitch_def.max_acc_ref_radps2 = 135.0;
  pitch_def.k_pos = 20.0;
  pitch_def.k_vel = 0.82;
  pitch_def.k_acc = 0.0;
  pitch_def.k_int = 85.0;
  pitch_def.leso_enable = false;
  pitch_def.enable_50hz_notch = false;

  cfg.yaw = parse_axis(node["yaw"], yaw_def);
  cfg.pitch = parse_axis(node["pitch"], pitch_def);
  return SimGimbalModel(cfg);
}

void SimGimbalModel::reset(double yaw_rad, double pitch_rad, uint64_t timestamp_ns)
{
  yaw_st_ = AxisState{};
  yaw_st_.pos = tools::limit_rad(yaw_rad);
  yaw_st_.z1 = yaw_st_.pos;

  // 内部 pitch 统一按电控“抬头为正”存储，与 gimbal.c 一致
  pitch_st_ = AxisState{};
  pitch_st_.pos = std::clamp(-pitch_rad, cfg_.pitch_min_rad, cfg_.pitch_max_rad);
  pitch_st_.z1 = pitch_st_.pos;

  initialized_ = true;
  last_ts_ns_ = timestamp_ns;
  accum_s_ = 0.0;
}

void SimGimbalModel::step_axis(
  AxisState & st, const SimGimbalAxisParams & p, double ref_pos, double ref_vel, double ref_acc,
  double dt_ctrl, bool wrap_pos, double pos_min, double pos_max)
{
  const double J_real = std::max(1e-6, p.inertia_kgm2);
  const double J_ctrl = std::max(1e-6, p.ctrl_inertia_kgm2);
  const double b0 = 1.0 / J_ctrl;

  // 1. 位置误差（与 gimbal_lqr.c 符号一致：e_pos = pos - ref_pos）
  const double e_pos = wrap_pos ? tools::limit_rad(st.pos - ref_pos) : (st.pos - ref_pos);

  // 2. 低通差分加速度（对齐 GimbalLQRUpdateAcceleration）
  const double raw_acc = std::clamp((st.vel - st.prev_vel) / dt_ctrl, -500.0, 500.0);
  st.prev_vel = st.vel;
  if (p.accel_lpf_tau_s > 0.0) {
    const double alpha = dt_ctrl / (p.accel_lpf_tau_s + dt_ctrl);
    st.acc_lpf += alpha * (raw_acc - st.acc_lpf);
  } else {
    st.acc_lpf = raw_acc;
  }

  // 3. 抗饱和位置积分（对齐 7991a95 gimbal_lqr.c 修正后的符号：u=-K*e，正饱和+负误差冻结）
  const bool freeze =
    ((st.last_torque >= p.torque_limit_nm) && (e_pos < 0.0)) ||
    ((st.last_torque <= -p.torque_limit_nm) && (e_pos > 0.0));
  if (!freeze) {
    st.integral = std::clamp(
      st.integral + e_pos * dt_ctrl, -p.integral_limit_rad_s, p.integral_limit_rad_s);
  }

  // 4. 参考速度/加速度限幅 + LQI + 刚体加速度前馈
  const double v_ref =
    cfg_.use_vel_acc_ff ? std::clamp(ref_vel, -p.max_vel_ref_radps, p.max_vel_ref_radps) : 0.0;
  const double a_ref =
    cfg_.use_vel_acc_ff ? std::clamp(ref_acc, -p.max_acc_ref_radps2, p.max_acc_ref_radps2) : 0.0;
  const double u_ff = J_ctrl * a_ref;
  const double u_lqr =
    -(p.k_pos * e_pos + p.k_vel * (st.vel - v_ref) + p.k_acc * (st.acc_lpf - a_ref) +
      p.k_int * st.integral) +
    u_ff;

  // 5. 可选二阶 LESO（7991a95 默认关闭）
  double lqr_out = u_lqr;
  if (p.leso_enable) {
    const double wo = p.leso_bandwidth_radps;
    if (wo > 0.0 && wo * dt_ctrl < 0.25) {
      const double e_obs = wrap_pos ? tools::limit_rad(st.z1 - st.pos) : (st.z1 - st.pos);
      const double dz1 = st.z2 - 3.0 * wo * e_obs;
      const double dz2 = st.z3 - 3.0 * wo * wo * e_obs + b0 * st.last_torque;
      const double dz3 = -wo * wo * wo * e_obs;
      st.z1 = wrap_pos ? tools::limit_rad(st.z1 + dz1 * dt_ctrl) : (st.z1 + dz1 * dt_ctrl);
      st.z2 += dz2 * dt_ctrl;
      st.z3 += dz3 * dt_ctrl;
    }
    lqr_out += std::clamp(-st.z3 / b0, -p.leso_max_comp_nm, p.leso_max_comp_nm);
  }
  lqr_out = std::clamp(lqr_out, -p.torque_limit_nm, p.torque_limit_nm);
  st.last_torque = lqr_out;

  // 6. Yaw 50 Hz 陷波器（对齐 gimbal.c GimbalYawNotchApply，500 Hz 双线性变换系数）
  double lqr_filtered = lqr_out;
  if (p.enable_50hz_notch) {
    lqr_filtered = 0.8269340 * lqr_out - 1.2976768 * st.nx1 + 0.7770827 * st.nx2 +
                   1.2976768 * st.ny1 - 0.6040167 * st.ny2;
    st.nx2 = st.nx1;
    st.nx1 = lqr_out;
    st.ny2 = st.ny1;
    st.ny1 = lqr_filtered;
  }

  // 7. 轨迹速度库仑+粘性摩擦前馈（对齐 GimbalYawFrictionFfNm / GimbalPitchFrictionFfNm）
  const double fric_ff = friction_torque(v_ref, p);
  const double tau_cmd = std::clamp(lqr_filtered + fric_ff, -p.torque_limit_nm, p.torque_limit_nm);

  // 8. 刚体 + 6 ms 内部力矩环滞后 + 电机包络子步积分（4 子步 -> 2 kHz 物理积分）
  constexpr int SUBSTEPS = 4;
  const double h = dt_ctrl / static_cast<double>(SUBSTEPS);
  for (int i = 0; i < SUBSTEPS; ++i) {
    const double tau_target = apply_motor_envelope(tau_cmd, st.vel, p);
    if (p.torque_lag_tau_s > 0.0) {
      const double a_lag = h / (p.torque_lag_tau_s + h);
      st.actual_torque += a_lag * (tau_target - st.actual_torque);
    } else {
      st.actual_torque = tau_target;
    }
    const double tau_fric_real = friction_torque(st.vel, p);
    const double acc = (st.actual_torque - tau_fric_real) / J_real;
    st.vel += acc * h;
    st.pos += st.vel * h;
    if (wrap_pos) {
      st.pos = tools::limit_rad(st.pos);
    } else if (st.pos < pos_min) {
      st.pos = pos_min;
      if (st.vel < 0.0) st.vel = 0.0;
    } else if (st.pos > pos_max) {
      st.pos = pos_max;
      if (st.vel > 0.0) st.vel = 0.0;
    }
  }
}

Eigen::Vector2d SimGimbalModel::step(
  double cmd_yaw, double cmd_yaw_vel, double cmd_yaw_acc, double cmd_pitch, double cmd_pitch_vel,
  double cmd_pitch_acc, uint64_t timestamp_ns)
{
  if (!cfg_.enabled) return {cmd_yaw, cmd_pitch};

  if (!initialized_) {
    reset(cmd_yaw, cmd_pitch, timestamp_ns);
    return {cmd_yaw, cmd_pitch};
  }

  const double dt_ctrl = 1.0 / std::max(50.0, cfg_.control_freq_hz);
  double dt_wall = dt_ctrl;
  if (timestamp_ns > last_ts_ns_ && last_ts_ns_ > 0) {
    dt_wall = std::clamp(static_cast<double>(timestamp_ns - last_ts_ns_) * 1e-9, 0.0, 0.05);
  }
  last_ts_ns_ = timestamp_ns;

  // 视觉 pitch 抬头为负 -> 电控内部抬头为正
  const double ref_yaw = tools::limit_rad(cmd_yaw);
  const double ref_pitch_up = std::clamp(-cmd_pitch, cfg_.pitch_min_rad, cfg_.pitch_max_rad);
  double ref_pitch_vel_up = -cmd_pitch_vel;
  double ref_pitch_acc_up = -cmd_pitch_acc;
  // 触及上下软限位时截断向外前馈（对齐 gimbal.c:1123-1132）
  if ((ref_pitch_up <= cfg_.pitch_min_rad && ref_pitch_vel_up < 0.0) ||
      (ref_pitch_up >= cfg_.pitch_max_rad && ref_pitch_vel_up > 0.0)) {
    ref_pitch_vel_up = 0.0;
  }
  if ((ref_pitch_up <= cfg_.pitch_min_rad && ref_pitch_acc_up < 0.0) ||
      (ref_pitch_up >= cfg_.pitch_max_rad && ref_pitch_acc_up > 0.0)) {
    ref_pitch_acc_up = 0.0;
  }

  accum_s_ += dt_wall;
  int steps = 0;
  while (accum_s_ >= dt_ctrl && steps < 30) {
    step_axis(yaw_st_, cfg_.yaw, ref_yaw, cmd_yaw_vel, cmd_yaw_acc, dt_ctrl, true, -M_PI, M_PI);
    step_axis(
      pitch_st_, cfg_.pitch, ref_pitch_up, ref_pitch_vel_up, ref_pitch_acc_up, dt_ctrl, false,
      cfg_.pitch_min_rad, cfg_.pitch_max_rad);
    accum_s_ -= dt_ctrl;
    steps++;
  }

  // 返回视觉系弧度（pitch 抬头为负）
  return {yaw_st_.pos, -pitch_st_.pos};
}

}  // namespace io
