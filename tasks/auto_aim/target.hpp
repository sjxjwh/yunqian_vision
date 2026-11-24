#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "armor.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{
class Solver;
}

namespace auto_aim
{

class Target
{
public:
  ArmorName name;
  ArmorType armor_type;
  ArmorPriority priority;
  bool jumped;
  int last_id;  // debug only

  Target() = default;
  Target(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig);
  Target(double x, double vyaw, double radius, double h);

  // 预测目标状态。self_vel 为我方云台在世界系下的瞬时线速度 (m/s)：
  // 传入非零 self_vel 时，位置相对自身扣除 self_vel * dt，使 EKF 状态 [vx,vy,vz] 始终保持敌方世界系真实速度，
  // 避免我方跑打加减速引发速度估计滞后或 NIS 失锁。
  void predict(
    std::chrono::steady_clock::time_point t,
    const Eigen::Vector3d & self_vel = Eigen::Vector3d::Zero());
  void predict(double dt, const Eigen::Vector3d & self_vel = Eigen::Vector3d::Zero());
  void update(const Armor & armor);
  // 用当前状态的倾角和预测 yaw 消掉 solvePnP 的 yaw 双解，再做匹配和更新；被门限拒收时返回 false。
  bool update(Armor & armor, const Solver & solver);
  // 图像平面 UVL (+ 单板深度差) 多观测联合更新与图像空间四点匹配（参考 WUST-RM/awakening）。
  // 状态量(13维)不变；返回是否至少匹配并更新了 1 块装甲板。
  bool update_uvl(std::vector<Armor *> & armors, const Solver & solver);

  // 基于当前整车状态先验生成网络聚焦 ROI（丢失越久越接近全图）。
  cv::Rect focus_roi(
    const Solver & solver, const cv::Size & img_size, std::chrono::steady_clock::time_point now,
    double lost_timeout_s = 0.25, double target_wh_ratio = 1.0,
    const Eigen::Vector3d & self_vel = Eigen::Vector3d::Zero()) const;

  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;
  // 在 armor_xyza_list() 的 [x, y, z, yaw] 后追加环 roll/pitch。只给显示，EKF 接口不变。
  std::vector<Eigen::Vector<double, 6>> armor_xyzayp_list() const;

  bool diverged() const;

  bool convergened();

  bool isinit = false;

  bool checkinit();

private:
  int armor_num_;
  int switch_count_;
  int update_count_;

  bool is_switch_, is_converged_;

  tools::ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_;
  std::chrono::steady_clock::time_point last_update_t_;

  void update_ypda(const Armor & armor, int id);  // yaw pitch distance angle
  void update_with_id(const Armor & armor, int id);

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  Eigen::Vector3d h_armor_ypr(const Eigen::VectorXd & x, int id) const;
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;

  // 装甲环平面：Rx(roll) * Ry(pitch)。roll/pitch 为 0 时退回原来的水平环。
  static Eigen::Matrix3d ring_rotation(double roll, double pitch);
  static Eigen::Vector3d armor_offset(const Eigen::VectorXd & x, int id, int armor_num);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP