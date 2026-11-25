#ifndef AUTO_AIM__PLANNER_HPP
#define AUTO_AIM__PLANNER_HPP

#include <Eigen/Dense>
#include <atomic>
#include <list>
#include <optional>

#include "tasks/auto_aim/planner/fire_mlp.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tinympc/tiny_api.hpp"

namespace auto_aim
{
constexpr double DT = 0.01;
constexpr int HALF_HORIZON = 50;
constexpr int HORIZON = HALF_HORIZON * 2;

using Trajectory = Eigen::Matrix<double, 4, HORIZON>;  // yaw, yaw_vel, pitch, pitch_vel

struct FireDiagnostics
{
  bool valid = false;           // 3D 弹道在宽松边界内与正面装甲板求交成功且未被机身遮挡
  double hit_y_norm = 0.0;      // 带符号归一化横向落点：hit.y / (0.5 * armor_w)
  double hit_z_norm = 0.0;      // 带符号归一化纵向落点：hit.z / (0.5 * armor_h)
  double cos_inc = 0.0;         // 真实 3D 入射角余弦
  double dist = 0.0;            // 炮口到目标距离 (m)
  double fly_time = 0.0;        // 弹丸飞行时间 (s)
  bool is_big_armor = false;
  double target_w = 0.0;        // 目标转速 (rad/s)
  double target_v_tan = 0.0;    // 目标相对视线切向速度 (m/s)
  double self_speed = 0.0;      // 我方线速度模长 (m/s)
  double self_roll = 0.0;       // 我方云台滚转角 (rad)
  double ekf_sigma_phase = 0.0; // EKF 预测至命中时刻的相位标准差 (rad)
  double ekf_sigma_pos = 0.0;   // EKF 旋转中心位置标准差 (m)
  double ekf_sigma_r = 0.0;     // EKF 装甲半径标准差 (m)
  bool target_jumped = false;   // 是否处于换板跳变帧
  double yaw_err = 0.0;         // rad: limit_rad(muzzle_yaw - ref_yaw)
  double pitch_err = 0.0;       // rad: muzzle_pitch - ref_pitch
  double p_phys = 0.0;          // Layer 2 物理不确定度解析命中概率 ∈ [0, 1]
  double delta_logit = 0.0;     // Layer 3 残差 MLP 输出对数几率修正 ∈ [-2, +2]
  double p_hit = 0.0;           // 最终校准命中概率 sigmoid(logit(p_phys) + delta_logit)
  double eff_thresh = 0.0;      // 本帧实际生效的动态开火概率阈值 τ ∈ [0.45, 0.90]
  int hit_plate_id = -1;        // 命中的装甲板索引（用于区分同板防吞弹与换板瞬间接发）
  float fire_rate_hz = 0.0f;    // 建议射频 (Hz)，0 表示停火
  double sigma_y_m = 0.0;       // 投影至装甲板横向的总等效标准差 (m)
  double sigma_z_m = 0.0;       // 投影至装甲板纵向的总等效标准差 (m)
};

// 2027 规则动态火控阈值偏置枚举（见 AGENTS.md §8.5）：
// τ = clamp(τ_base(d) + Δτ_level + Δτ_mode + Δτ_armor, 0.45, 0.90)
enum class FireTacticalMode : int
{
  Normal = 0,  // 常规对枪（Δτ = 0）
  Burst = 1,   // 残血收割 / 越级击杀 / 团战强杀（Δτ = -0.15）
  Eco = 2,     // 远距拉扯 / 省弹控 η（Δτ = +0.08）
};

enum class LevelPhase : int
{
  Unknown = 0,   // 未接入裁判等级（Δτ = 0）
  Leveling = 1,  // 未满级发育期，加速吃 80~120 XP/发（Δτ = -0.08）
  Capped = 2,    // 已达等级上限（5 级封顶前 / 10 级满级），收紧省弹保 η（Δτ = +0.05）
};

enum class ArmorZone : int
{
  Default = 0,       // 默认/未知加固区（×1.0，Δτ = 0）
  Unreinforced = 1,  // §5.2 未加固区（×1.5，Δτ = -0.08）
  Reinforced = 2,    // §5.2 已加固区（×0.5，Δτ = +0.12）
};

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
  float fire_rate_hz = 0.0f;  // 建议射频 (Hz)：结合单板防吞弹上限、p_hit 与强制爆发模式动态规划
  int fire_plate_id = -1;     // 当前锁定击中的装甲板索引（换板时无需等待上块板检测死区）
  FireDiagnostics diag{};
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

  void set_fire_margin(double margin) { fire_margin_ = margin; }
  void set_fire_prob_thresh(double thresh)
  {
    fire_prob_thresh_.store(std::clamp(thresh, 0.0, 0.999), std::memory_order_relaxed);
  }
  void set_fire_prob_schedule(double p_near, double p_far, double d_near = 2.0, double d_far = 4.5)
  {
    fire_prob_thresh_.store(std::clamp(p_near, 0.0, 0.999), std::memory_order_relaxed);
    fire_prob_thresh_far_.store(std::clamp(p_far, 0.0, 0.999), std::memory_order_relaxed);
    fire_prob_dist_near_ = std::max(0.1, d_near);
    fire_prob_dist_far_ = std::max(fire_prob_dist_near_ + 0.1, d_far);
  }
  double fire_prob_thresh() const { return fire_prob_thresh_.load(std::memory_order_relaxed); }
  void set_outpost_fire_prob_thresh(double thresh)
  {
    outpost_fire_prob_thresh_.store(std::clamp(thresh, 0.0, 0.95), std::memory_order_relaxed);
  }
  double outpost_fire_prob_thresh() const
  {
    return outpost_fire_prob_thresh_.load(std::memory_order_relaxed);
  }
  void set_use_fire_mlp(bool enable) { use_fire_mlp_.store(enable, std::memory_order_relaxed); }
  bool use_fire_mlp() const { return use_fire_mlp_.load(std::memory_order_relaxed); }
  void set_gun_sigma_deg(double sy_deg, double sp_deg)
  {
    constexpr double D2R = M_PI / 180.0;
    gun_sigma_yaw_rad_.store(std::clamp(sy_deg, 0.0, 5.0) * D2R, std::memory_order_relaxed);
    gun_sigma_pitch_rad_.store(std::clamp(sp_deg, 0.0, 5.0) * D2R, std::memory_order_relaxed);
  }
  void set_fire_delay_jitter(double jitter_s) { fire_delay_jitter_s_ = std::max(0.0, jitter_s); }

  void set_fire_tactical_mode(FireTacticalMode mode)
  {
    fire_tactical_mode_.store(static_cast<int>(mode), std::memory_order_relaxed);
  }
  FireTacticalMode fire_tactical_mode() const
  {
    return static_cast<FireTacticalMode>(fire_tactical_mode_.load(std::memory_order_relaxed));
  }
  void set_level_phase(LevelPhase phase)
  {
    level_phase_.store(static_cast<int>(phase), std::memory_order_relaxed);
  }
  LevelPhase level_phase() const
  {
    return static_cast<LevelPhase>(level_phase_.load(std::memory_order_relaxed));
  }
  void set_armor_zone(ArmorZone zone)
  {
    armor_zone_.store(static_cast<int>(zone), std::memory_order_relaxed);
  }
  ArmorZone armor_zone() const
  {
    return static_cast<ArmorZone>(armor_zone_.load(std::memory_order_relaxed));
  }
  void set_heat_ratio(double ratio)
  {
    heat_ratio_.store(std::clamp(ratio, 0.0, 1.0), std::memory_order_relaxed);
  }
  double heat_ratio() const { return heat_ratio_.load(std::memory_order_relaxed); }
  void set_fire_rate_bounds(double min_hz, double max_hz)
  {
    max_fire_rate_hz_.store(std::clamp(max_hz, 1.0, 25.0), std::memory_order_relaxed);
    min_fire_rate_hz_.store(
      std::clamp(min_hz, 1.0, max_fire_rate_hz_.load(std::memory_order_relaxed)),
      std::memory_order_relaxed);
  }
  double effective_fire_thresh(double dist_m, bool is_outpost = false) const;
  float recommend_fire_rate_hz(
    double p_hit, double eff_thresh, bool plate_switched, bool is_outpost = false) const;

private:
  std::atomic<double> yaw_offset_{0.0};
  std::atomic<double> pitch_offset_{0.0};
  // Layer 2 物理不确定度命中概率参数（可由 AimCorrector 实测散布在线更新）
  std::atomic<double> gun_sigma_yaw_rad_{0.25 * M_PI / 180.0};
  std::atomic<double> gun_sigma_pitch_rad_{0.25 * M_PI / 180.0};
  std::atomic<double> fire_prob_thresh_{0.0};          // 近距/固定概率开火阈值 (>0 启用，<=0 回退 fire_margin)
  std::atomic<double> fire_prob_thresh_far_{0.0};      // 远距 Peek 概率阈值 (<=0 表示固定不随距离插值)
  std::atomic<double> outpost_fire_prob_thresh_{0.45}; // 前哨站最大 DPS 概率开火阈值（不走远距 Peek 拉升）
  double fire_prob_dist_near_ = 2.0;                   // 近距爆发距离拐点 (m)
  double fire_prob_dist_far_ = 4.5;                    // 远距 Peek 距离拐点 (m)
  std::atomic<bool> use_fire_mlp_{false};          // 启用 Layer 3 残差 MLP 修正 P_phys
  std::atomic<int> fire_tactical_mode_{static_cast<int>(FireTacticalMode::Normal)};
  std::atomic<int> level_phase_{static_cast<int>(LevelPhase::Unknown)};
  std::atomic<int> armor_zone_{static_cast<int>(ArmorZone::Default)};
  std::atomic<double> heat_ratio_{0.0};            // 下位机上报枪口热量占比 [0, 1]
  std::atomic<double> min_fire_rate_hz_{6.0};      // 边缘窗口建议射频下限 (Hz)
  std::atomic<double> max_fire_rate_hz_{10.0};     // 单块装甲板防吞弹上限射频 (Hz，2027 17mm 理论上限 20Hz，工程保守 10~15Hz)
  int last_fire_plate_id_ = -1;
  FireResidualMLP fire_mlp_;
  double fire_delay_jitter_s_ = 0.005;         // 发弹延迟随机抖动标准差 (s)
  double muzzle_z_offset_ = 0.0;
  double air_k_ = 0.0;
  double fire_thresh_;
  int shoot_offset_ = 2;
  double fire_margin_ = 0.85;
  double outpost_fire_margin_ = 0.95;
  double max_fire_angle_ = 60.0 / 57.3;
  double outpost_max_fire_angle_ = 70.0 / 57.3;
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
    double muzzle_pitch, FireDiagnostics * out_diag = nullptr) const;
  Eigen::Matrix<double, 2, 1> aim(const Target & target, double bullet_speed);
  Trajectory get_trajectory(Target target, double yaw0, double bullet_speed);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__PLANNER_HPP