#ifndef TOOLS__SIM_METRICS_HPP
#define TOOLS__SIM_METRICS_HPP

#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace tools
{

// 虚拟发弹碰撞检测用的装甲板几何（视觉世界系：x 前、y 左、z 上，单位米/弧度）
struct SimArmorPlate
{
  Eigen::Vector3d center{0, 0, 0};
  double yaw = 0;
  double ring_roll = 0;
  double ring_pitch = 0;
  double mount_pitch = 15.0 * M_PI / 180.0;
  double width = 0.135;   // 小装甲 0.135 m，大装甲 0.230 m
  double height = 0.125;  // 物理装甲受击面高度：小 0.125 m，大 0.127 m
};

struct SimHitEvent
{
  uint32_t id = 0;
  double t_sec = 0.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  int armor_id = -1;
};

struct SimMetricsSnapshot
{
  uint32_t shots = 0;
  uint32_t hits = 0;
  uint32_t resolved = 0;
  uint32_t in_flight = 0;
  double hit_rate = 0.0;
  double fire_duty = 0.0;    // 控制态下允许开火的帧占比 [0, 1]
  double yaw_err = 0.0;      // rad, limit_rad(gimbal_yaw - plan_yaw)
  double pitch_err = 0.0;    // rad, gimbal_pitch - plan_pitch
  double yaw_err_rms = 0.0;  // rad
  std::vector<std::array<double, 3>> bullets;
  std::vector<SimHitEvent> recent_hits;
};

// 仿真闭环定量评估器：
// 1. 按当前云台真实朝向发射虚拟弹丸（不读仿真 GT），按抛物线推进并在到达时刻与跟踪装甲面求交；
// 2. 统计 hit_rate 与云台跟踪误差（plan_yaw vs gimbal_yaw）；
// 3. 可选导出逐帧 CSV 供离线对比（如 Aimer vs MPC Planner）。
class SimMetrics
{
public:
  explicit SimMetrics(double fire_hz = 10.0, double air_k = 0.0);

  // 打开 CSV 日志文件，写入表头；返回是否成功。
  bool open_csv(const std::string & path);

  // 每帧调用一次，返回最新统计快照。
  // muzzle_ypr 为真实炮口朝向（ZYX 欧拉角，弧度，pitch 抬头为负），muzzle_origin 为炮口在视觉世界系坐标。
  SimMetricsSnapshot update(
    double t_sec, const Eigen::Vector3d & muzzle_ypr, bool control, double plan_yaw,
    double plan_pitch, bool fire_cmd, double bullet_speed,
    const std::vector<SimArmorPlate> & armors,
    const Eigen::Vector3d & muzzle_origin = Eigen::Vector3d::Zero(),
    const Eigen::Vector3d & self_vel = Eigen::Vector3d::Zero());

  void reset();

private:
  struct Bullet
  {
    double t0 = 0.0;
    double last_t = 0.0;
    Eigen::Vector3d p0{0, 0, 0};
    Eigen::Vector3d v0{0, 0, 0};
    Eigen::Vector3d vel{0, 0, 0};
    Eigen::Vector3d prev_pos{0, 0, 0};
    double max_dist = 12.0;
  };

  double fire_interval_ = 0.1;
  double air_k_ = 0.0;
  double last_t_ = -1.0;
  double last_shot_t_ = -1e9;

  uint32_t shots_ = 0;
  uint32_t hits_ = 0;
  uint32_t resolved_ = 0;
  uint32_t next_hit_id_ = 1;
  uint64_t err_samples_ = 0;
  uint64_t fire_samples_ = 0;
  double yaw_err_sq_sum_ = 0.0;

  std::vector<Bullet> bullets_;
  std::vector<SimHitEvent> recent_hits_;
  std::ofstream csv_;

  static bool intersect_plate(
    const Eigen::Vector3d & p_prev, const Eigen::Vector3d & p_curr, const SimArmorPlate & plate,
    double & out_alpha);
};

}  // namespace tools

#endif  // TOOLS__SIM_METRICS_HPP
