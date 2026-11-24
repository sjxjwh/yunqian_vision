#ifndef IO__SIM__SIM_GIMBAL_MODEL_HPP
#define IO__SIM__SIM_GIMBAL_MODEL_HPP

#include <Eigen/Core>
#include <cstdint>
#include <string>

namespace io
{

// 真实云台动力学与控制器复现模型（不修改只读仿真器源码）。
// 精确对齐 27-wheel-leg-wbc/gimbal @ 7991a95 三号步兵的 500 Hz LQI + 50 Hz 陷波器 +
// 库仑/粘性摩擦前馈 + 6 ms 力矩环滞后 + DM4310 电机转速-力矩包络模型。
struct SimGimbalAxisParams
{
  double inertia_kgm2 = 0.0146;       // 真实负载转动惯量 J (kg*m^2)
  double ctrl_inertia_kgm2 = 0.0146;  // 控制器标称惯量（用于 J*a_ref 前馈与 LESO b0）
  double coulomb_friction_nm = 0.25;  // 库仑动摩擦 (N*m，与摩擦前馈配套)
  double coulomb_smooth_radps = 0.25; // tanh 平滑速度阈值 (rad/s)
  double viscous_damping = 0.042;     // 粘性阻尼 (N*m/(rad/s))
  double torque_lag_tau_s = 0.006;    // 内部力矩环一阶滞后时间常数 (s，实测 ~6 ms)
  double torque_limit_nm = 6.0;       // 电控软件力矩限幅 (N*m，gimbal.c: yaw 6.0, pitch 4.5)
  double rated_torque_nm = 3.5;       // 电机额定扭矩 (N*m)
  double peak_torque_nm = 12.5;       // 电机峰值扭矩 (N*m)
  double rated_speed_rpm = 120.0;     // 电机额定转速 (rpm)
  double no_load_speed_rpm = 200.0;   // 电机空载转速 (rpm，决定反电动势线性限幅)

  // 参考速度/加速度限幅（对齐 yaw_reference_config / pitch_reference_config）
  double max_vel_ref_radps = 16.5;
  double max_acc_ref_radps2 = 210.0;

  // LQI 4 状态反馈增益 [k_pos, k_vel, k_acc, k_int]（正数，执行 u = -K*(x - x_ref) + u_ff）
  double k_pos = 24.0;
  double k_vel = 0.96;
  double k_acc = 0.0;
  double k_int = 120.0;

  double integral_limit_rad_s = 0.20;
  double accel_lpf_tau_s = 0.010;
  bool leso_enable = false;           // 对齐 GIMBAL_LQR_LESO_ENABLE = 0
  double leso_bandwidth_radps = 18.0;
  double leso_max_comp_nm = 2.0;
  bool enable_50hz_notch = true;      // 对齐 Yaw 50 Hz 扭振陷波器（500 Hz 双线性变换）
};

struct SimGimbalModelConfig
{
  bool enabled = false;
  double control_freq_hz = 500.0;     // 对应 GIMBAL_LQR_CONTROL_FREQ_HZ = 500 Hz
  bool use_vel_acc_ff = true;         // 对齐最新 gimbal.c 视觉模式：直接接入 yaw_vel_ff / yaw_acc_ff
  double pitch_min_rad = -20.0 * M_PI / 180.0;  // 抬头为正语义下限（低头 -20°）
  double pitch_max_rad = 30.0 * M_PI / 180.0;   // 抬头为正语义上限（抬头 +30°）
  SimGimbalAxisParams yaw;
  SimGimbalAxisParams pitch;
};

class SimGimbalModel
{
public:
  SimGimbalModel() = default;
  explicit SimGimbalModel(const SimGimbalModelConfig & cfg);

  // 从 configs/sim.yaml 读取 sim_gimbal_model 段；缺省时返回 disabled 模型。
  static SimGimbalModel from_yaml(const std::string & config_path);

  bool enabled() const { return cfg_.enabled; }
  bool initialized() const { return initialized_; }
  Eigen::Vector2d state_yp() const { return {yaw_st_.pos, -pitch_st_.pos}; }
  const SimGimbalModelConfig & config() const { return cfg_; }

  // 重置内部状态（与当前真实/仿真炮口姿态对齐，防止切换瞬间阶跃）
  // yaw/pitch 均使用视觉内部右手系弧度（yaw 左正右负，pitch 抬头为负）。
  void reset(double yaw_rad, double pitch_rad, uint64_t timestamp_ns = 0);

  // 推进一步动力学并返回模拟后的云台姿态 (yaw_rad, pitch_rad)（pitch 抬头为负）。
  // 入参 cmd_yaw/cmd_pitch 为视觉系弧度（pitch 抬头为负），cmd_*_vel/acc 同符号约定。
  Eigen::Vector2d step(
    double cmd_yaw, double cmd_yaw_vel, double cmd_yaw_acc, double cmd_pitch,
    double cmd_pitch_vel, double cmd_pitch_acc, uint64_t timestamp_ns);

private:
  struct AxisState
  {
    double pos = 0.0;
    double vel = 0.0;
    double prev_vel = 0.0;
    double acc_lpf = 0.0;
    double integral = 0.0;
    double last_torque = 0.0;
    double actual_torque = 0.0;  // 经 6 ms 力矩环一阶滞后后的实际输出力矩
    // 50 Hz 陷波器双二阶状态（对齐 GimbalBiquadState_s）
    double nx1 = 0.0, nx2 = 0.0, ny1 = 0.0, ny2 = 0.0;
    // 二阶 LESO 状态 (z1≈pos, z2≈vel, z3≈总扰动 f)
    double z1 = 0.0, z2 = 0.0, z3 = 0.0;
  };

  void step_axis(
    AxisState & st, const SimGimbalAxisParams & p, double ref_pos, double ref_vel, double ref_acc,
    double dt_ctrl, bool wrap_pos, double pos_min, double pos_max);

  SimGimbalModelConfig cfg_{};
  AxisState yaw_st_{};
  AxisState pitch_st_{};
  bool initialized_ = false;
  uint64_t last_ts_ns_ = 0;
  double accum_s_ = 0.0;
};

}  // namespace io

#endif  // IO__SIM__SIM_GIMBAL_MODEL_HPP
