#ifndef AUTO_AIM__SOLVER_HPP
#define AUTO_AIM__SOLVER_HPP

#include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <Eigen/Geometry>
#include <array>
#include <opencv2/core/eigen.hpp>

#include "armor.hpp"

namespace auto_aim
{
class Solver
{
public:
  explicit Solver(const std::string & config_path);

  Eigen::Matrix3d R_gimbal2world() const;

  void set_R_gimbal2world(const Eigen::Quaterniond & q);

  void solve(Armor & armor) const;

  std::vector<cv::Point2f> reproject_armor(
    const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name) const;

  std::vector<cv::Point2f> reproject_armor(
    const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name,
    double ring_roll, double ring_pitch) const;

  // 用当前装甲环倾角估计重投影，在低代价的 yaw 极小值里取离预测最近的那个。
  // 状态向量、观测向量、雅可比都不变；只替换进 EKF 之前的 armor_yaw。
  // predicted_yaw 无值（首帧）时保持原来的全局最小。
  void resolve_yaw(Armor & armor, double ring_roll, double ring_pitch, double predicted_yaw) const;

  double oupost_reprojection_error(Armor armor, const double & picth);

  std::vector<cv::Point2f> world2pixel(const std::vector<cv::Point3f> & worldPoints);

  // ---- 弹道闭环用：当前姿态下的相机位姿 / 像素射线（均基于最近一次 set_R_gimbal2world）----
  Eigen::Matrix3d R_camera2world() const { return R_gimbal2world_ * R_camera2gimbal_; }
  Eigen::Vector3d camera_center_in_world() const { return R_gimbal2world_ * t_camera2gimbal_; }
  /// 像素（含畸变）→ 世界系单位方向向量（起点为相机光心）
  Eigen::Vector3d pixel2world_dir(const cv::Point2f & px) const;
  /// 单点投影；在相机后方返回 false
  bool world2pixel(const Eigen::Vector3d & p_world, cv::Point2f & px) const;
  const cv::Mat & camera_matrix() const { return camera_matrix_; }
  double fx() const { return camera_matrix_.at<double>(0, 0); }
  double fy() const { return camera_matrix_.at<double>(1, 1); }

  // ---- UVL 图像平面观测与深度差辅助（参考 WUST-RM/awakening）----
  /// 将装甲板 4 角点去畸变到归一化相机平面 (x/z, y/z)，顺序同 Armor::points (LT, RT, RB, LB)
  std::array<Eigen::Vector2d, 4> undistort_armor_points(const Armor & armor) const;
  /// 纯 Eigen 计算装甲板 4 角点在相机系的 3D 坐标；全部位于相机前方时返回 true
  bool armor_corners_in_camera(
    const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name,
    double ring_roll, double ring_pitch, std::array<Eigen::Vector3d, 4> & pts_cam,
    bool * front_facing = nullptr) const;
  /// 用 IPPE 正面解计算左灯条中点与右灯条中点在相机系的深度差 (z_left - z_right)
  bool armor_lightbar_depth_diff(const Armor & armor, double & depth_diff) const;

private:
  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
  Eigen::Matrix3d R_gimbal2imubody_;
  Eigen::Matrix3d R_camera2gimbal_;
  Eigen::Vector3d t_camera2gimbal_;
  Eigen::Matrix3d R_gimbal2world_;

  void optimize_yaw(Armor & armor) const;

  double armor_reprojection_error(const Armor & armor, double yaw, const double & inclined) const;

  double armor_reprojection_error(
    const Armor & armor, double yaw, double ring_roll, double ring_pitch) const;
  double SJTU_cost(
    const std::vector<cv::Point2f> & cv_refs, const std::vector<cv::Point2f> & cv_pts,
    const double & inclined) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__SOLVER_HPP