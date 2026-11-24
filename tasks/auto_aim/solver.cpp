#include "solver.hpp"

#include <yaml-cpp/yaml.h>

#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
constexpr double LIGHTBAR_LENGTH = 56e-3;     // m
constexpr double BIG_ARMOR_WIDTH = 230e-3;    // m
constexpr double SMALL_ARMOR_WIDTH = 135e-3;  // m

const std::vector<cv::Point3f> BIG_ARMOR_POINTS{
  {0, BIG_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -BIG_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -BIG_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2},
  {0, BIG_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2}};
const std::vector<cv::Point3f> SMALL_ARMOR_POINTS{
  {0, SMALL_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -SMALL_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -SMALL_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2},
  {0, SMALL_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2}};

Solver::Solver(const std::string & config_path) : R_gimbal2world_(Eigen::Matrix3d::Identity())
{
  auto yaml = YAML::LoadFile(config_path);

  auto R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  auto R_camera2gimbal_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
  auto t_camera2gimbal_data = yaml["t_camera2gimbal"].as<std::vector<double>>();
  R_gimbal2imubody_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_gimbal2imubody_data.data());
  R_camera2gimbal_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_camera2gimbal_data.data());
  t_camera2gimbal_ = Eigen::Matrix<double, 3, 1>(t_camera2gimbal_data.data());

  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(camera_matrix_data.data());
  Eigen::Matrix<double, 1, 5> distort_coeffs(distort_coeffs_data.data());
  cv::eigen2cv(camera_matrix, camera_matrix_);
  cv::eigen2cv(distort_coeffs, distort_coeffs_);
}

Eigen::Matrix3d Solver::R_gimbal2world() const { return R_gimbal2world_; }

void Solver::set_R_gimbal2world(const Eigen::Quaterniond & q)
{
  Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
  R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

//solvePnP（获得姿态）
void Solver::solve(Armor & armor) const
{
  const auto & object_points =
    (armor.type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;

  cv::Vec3d rvec, tvec;
  cv::solvePnP(
    object_points, armor.points, camera_matrix_, distort_coeffs_, rvec, tvec, false,
    cv::SOLVEPNP_IPPE);

  Eigen::Vector3d xyz_in_camera;
  cv::cv2eigen(tvec, xyz_in_camera);
  armor.xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
  armor.xyz_in_world = R_gimbal2world_ * armor.xyz_in_gimbal;

  cv::Mat rmat;
  cv::Rodrigues(rvec, rmat);
  Eigen::Matrix3d R_armor2camera;
  cv::cv2eigen(rmat, R_armor2camera);
  Eigen::Matrix3d R_armor2gimbal = R_camera2gimbal_ * R_armor2camera;
  Eigen::Matrix3d R_armor2world = R_gimbal2world_ * R_armor2gimbal;
  armor.ypr_in_gimbal = tools::eulers(R_armor2gimbal, 2, 1, 0);
  armor.ypr_in_world = tools::eulers(R_armor2world, 2, 1, 0);
  // 灯条中点连线在装甲局部系是 y 轴。它的世界系 yaw 不依赖 solvePnP 的朝向消歧，
  // 板转到侧面时不会像 ypr_in_world[0] 那样跳 180°。
  const Eigen::Vector3d lightbar_axis = R_armor2world.col(1);
  armor.lightbar_yaw = std::atan2(lightbar_axis.y(), lightbar_axis.x());

  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);

  // 平衡不做yaw优化，因为pitch假设不成立
  auto is_balance = (armor.type == ArmorType::big) &&
                    (armor.name == ArmorName::three || armor.name == ArmorName::four ||
                     armor.name == ArmorName::five);
  if (is_balance) return;

  optimize_yaw(armor);
}

std::vector<cv::Point2f> Solver::reproject_armor(
  const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name) const
{
  return reproject_armor(xyz_in_world, yaw, type, name, 0.0, 0.0);
}

std::vector<cv::Point2f> Solver::reproject_armor(
  const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name,
  double ring_roll, double ring_pitch) const
{
  auto sin_yaw = std::sin(yaw);
  auto cos_yaw = std::cos(yaw);

  auto pitch = (name == ArmorName::outpost) ? -15.0 * CV_PI / 180.0 : 15.0 * CV_PI / 180.0;
  auto sin_pitch = std::sin(pitch);
  auto cos_pitch = std::cos(pitch);

  // 与 Target::h_armor_ypr 同一套：先环倾角，再叠板面 yaw 和固定安装角。
  // roll = pitch = 0 时退回原来的水平环。
  const double cr = std::cos(ring_roll);
  const double sr = std::sin(ring_roll);
  const double cp = std::cos(ring_pitch);
  const double sp = std::sin(ring_pitch);
  // clang-format off
  const Eigen::Matrix3d R_ring{
    { cp, sr * sp, cr * sp},
    {  0,      cr,     -sr},
    {-sp, sr * cp, cr * cp}
  };
  const Eigen::Matrix3d R_mount{
    {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
    {sin_yaw * cos_pitch,  cos_yaw, sin_yaw * sin_pitch},
    {         -sin_pitch,        0,           cos_pitch}
  };
  const Eigen::Matrix3d R_armor2world = R_ring * R_mount;
  // clang-format on

  // get R_armor2camera t_armor2camera
  const Eigen::Vector3d & t_armor2world = xyz_in_world;
  Eigen::Matrix3d R_armor2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * R_armor2world;
  Eigen::Vector3d t_armor2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * t_armor2world - t_camera2gimbal_);

  // get rvec tvec
  cv::Vec3d rvec;
  cv::Mat R_armor2camera_cv;
  cv::eigen2cv(R_armor2camera, R_armor2camera_cv);
  cv::Rodrigues(R_armor2camera_cv, rvec);
  cv::Vec3d tvec(t_armor2camera[0], t_armor2camera[1], t_armor2camera[2]);

  // reproject
  std::vector<cv::Point2f> image_points;
  const auto & object_points = (type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;
  cv::projectPoints(object_points, rvec, tvec, camera_matrix_, distort_coeffs_, image_points);
  return image_points;
}

double Solver::oupost_reprojection_error(Armor armor, const double & pitch)
{
  // solve
  const auto & object_points =
    (armor.type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;

  cv::Vec3d rvec, tvec;
  cv::solvePnP(
    object_points, armor.points, camera_matrix_, distort_coeffs_, rvec, tvec, false,
    cv::SOLVEPNP_IPPE);

  Eigen::Vector3d xyz_in_camera;
  cv::cv2eigen(tvec, xyz_in_camera);
  armor.xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
  armor.xyz_in_world = R_gimbal2world_ * armor.xyz_in_gimbal;

  cv::Mat rmat;
  cv::Rodrigues(rvec, rmat);
  Eigen::Matrix3d R_armor2camera;
  cv::cv2eigen(rmat, R_armor2camera);
  Eigen::Matrix3d R_armor2gimbal = R_camera2gimbal_ * R_armor2camera;
  Eigen::Matrix3d R_armor2world = R_gimbal2world_ * R_armor2gimbal;
  armor.ypr_in_gimbal = tools::eulers(R_armor2gimbal, 2, 1, 0);
  armor.ypr_in_world = tools::eulers(R_armor2world, 2, 1, 0);

  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);

  auto yaw = armor.ypr_in_world[0];
  auto xyz_in_world = armor.xyz_in_world;

  auto sin_yaw = std::sin(yaw);
  auto cos_yaw = std::cos(yaw);

  auto sin_pitch = std::sin(pitch);
  auto cos_pitch = std::cos(pitch);

  // clang-format off
  const Eigen::Matrix3d _R_armor2world {
    {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
    {sin_yaw * cos_pitch,  cos_yaw, sin_yaw * sin_pitch},
    {         -sin_pitch,        0,           cos_pitch}
  };
  // clang-format on

  // get R_armor2camera t_armor2camera
  const Eigen::Vector3d & t_armor2world = xyz_in_world;
  Eigen::Matrix3d _R_armor2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * _R_armor2world;
  Eigen::Vector3d t_armor2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * t_armor2world - t_camera2gimbal_);

  // get rvec tvec
  cv::Vec3d _rvec;
  cv::Mat R_armor2camera_cv;
  cv::eigen2cv(_R_armor2camera, R_armor2camera_cv);
  cv::Rodrigues(R_armor2camera_cv, _rvec);
  cv::Vec3d _tvec(t_armor2camera[0], t_armor2camera[1], t_armor2camera[2]);

  // reproject
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(object_points, _rvec, _tvec, camera_matrix_, distort_coeffs_, image_points);

  auto error = 0.0;
  for (int i = 0; i < 4; i++) error += cv::norm(armor.points[i] - image_points[i]);
  return error;
}

void Solver::optimize_yaw(Armor & armor) const
{
  Eigen::Vector3d gimbal_ypr = tools::eulers(R_gimbal2world_, 2, 1, 0);

  constexpr double SEARCH_RANGE = 140;  // degree
  auto yaw0 = tools::limit_rad(gimbal_ypr[0] - SEARCH_RANGE / 2 * CV_PI / 180.0);

  auto min_error = 1e10;
  auto best_yaw = armor.ypr_in_world[0];

  for (int i = 0; i < SEARCH_RANGE; i++) {
    double yaw = tools::limit_rad(yaw0 + i * CV_PI / 180.0);
    auto error = armor_reprojection_error(armor, yaw, (i - SEARCH_RANGE / 2) * CV_PI / 180.0);

    if (error < min_error) {
      min_error = error;
      best_yaw = yaw;
    }
  }

  armor.yaw_raw = armor.ypr_in_world[0];
  armor.ypr_in_world[0] = best_yaw;
}

double Solver::SJTU_cost(
  const std::vector<cv::Point2f> & cv_refs, const std::vector<cv::Point2f> & cv_pts,
  const double & inclined) const
{
  std::size_t size = cv_refs.size();
  std::vector<Eigen::Vector2d> refs;
  std::vector<Eigen::Vector2d> pts;
  for (std::size_t i = 0u; i < size; ++i) {
    refs.emplace_back(cv_refs[i].x, cv_refs[i].y);
    pts.emplace_back(cv_pts[i].x, cv_pts[i].y);
  }
  double cost = 0.;
  for (std::size_t i = 0u; i < size; ++i) {
    std::size_t p = (i + 1u) % size;
    // i - p 构成线段。过程：先移动起点，再补长度，再旋转
    Eigen::Vector2d ref_d = refs[p] - refs[i];  // 标准
    Eigen::Vector2d pt_d = pts[p] - pts[i];
    // 长度差代价 + 起点差代价(1 / 2)（0 度左右应该抛弃)
    double pixel_dis =  // dis 是指方差平面内到原点的距离
      (0.5 * ((refs[i] - pts[i]).norm() + (refs[p] - pts[p]).norm()) +
       std::fabs(ref_d.norm() - pt_d.norm())) /
      ref_d.norm();
    double angular_dis = ref_d.norm() * tools::get_abs_angle(ref_d, pt_d) / ref_d.norm();
    // 平方可能是为了配合 sin 和 cos
    // 弧度差代价（0 度左右占比应该大）
    double cost_i =
      tools::square(pixel_dis * std::sin(inclined)) +
      tools::square(angular_dis * std::cos(inclined)) * 2.0;  // DETECTOR_ERROR_PIXEL_BY_SLOPE
    // 重投影像素误差越大，越相信斜率
    cost += std::sqrt(cost_i);
  }
  return cost;
}

double Solver::armor_reprojection_error(
  const Armor & armor, double yaw, const double & inclined) const
{
  auto image_points = reproject_armor(armor.xyz_in_world, yaw, armor.type, armor.name);
  auto error = 0.0;
  for (int i = 0; i < 4; i++) error += cv::norm(armor.points[i] - image_points[i]);
  // auto error = SJTU_cost(image_points, armor.points, inclined);

  return error;
}

double Solver::armor_reprojection_error(
  const Armor & armor, double yaw, double ring_roll, double ring_pitch) const
{
  auto image_points =
    reproject_armor(armor.xyz_in_world, yaw, armor.type, armor.name, ring_roll, ring_pitch);
  auto error = 0.0;
  for (int i = 0; i < 4; i++) error += cv::norm(armor.points[i] - image_points[i]);
  return error;
}

void Solver::resolve_yaw(
  Armor & armor, double ring_roll, double ring_pitch, double predicted_yaw) const
{
  // 首帧还没有预测。保持原来的水平环、固定安装角搜索，避免用未收敛的倾角改掉初始化。
  if (!std::isfinite(predicted_yaw)) {
    optimize_yaw(armor);
    return;
  }

  Eigen::Vector3d gimbal_ypr = tools::eulers(R_gimbal2world_, 2, 1, 0);
  constexpr double kSearchRangeDeg = 140;
  auto yaw0 = tools::limit_rad(gimbal_ypr[0] - kSearchRangeDeg / 2 * CV_PI / 180.0);

  struct Sample
  {
    double yaw;
    double error;
  };
  std::vector<Sample> samples;
  samples.reserve(kSearchRangeDeg);
  double min_error = 1e10;
  for (int i = 0; i < kSearchRangeDeg; i++) {
    double yaw = tools::limit_rad(yaw0 + i * CV_PI / 180.0);
    double error = armor_reprojection_error(armor, yaw, ring_roll, ring_pitch);
    samples.push_back({yaw, error});
    min_error = std::min(min_error, error);
  }

  // 两个极小值在斜面上深度可以差到几十像素，8 px 的坑会把另一个解丢掉。
  // 这里保留所有局部极小，但离全局最小超过 40 px 的不参与，避免平地的次极小抢相位。
  constexpr double kBasinPx = 40.0;
  double best_yaw = armor.ypr_in_world[0];
  double best_gap = 1e10;
  for (int i = 1; i + 1 < static_cast<int>(samples.size()); i++) {
    const auto & sample = samples[i];
    if (sample.error > min_error + kBasinPx) continue;
    if (sample.error > samples[i - 1].error || sample.error > samples[i + 1].error) continue;
    double gap = std::abs(tools::limit_rad(sample.yaw - predicted_yaw));
    if (gap < best_gap) {
      best_gap = gap;
      best_yaw = sample.yaw;
    }
  }
  // 预测已经偏离所有低代价极小值时，不再跟着预测走。否则静止目标的观测会被
  // 一个错误的角速度拽着转，w 越积越大。0.5 rad 小于两个极小值的间距，也小于换板的 90°。
  if (best_gap > 0.5) {
    for (const auto & sample : samples) {
      if (sample.error == min_error) {
        best_yaw = sample.yaw;
        break;
      }
    }
  }

  armor.yaw_raw = armor.ypr_in_world[0];
  armor.ypr_in_world[0] = best_yaw;
}

// 世界坐标到像素坐标的转换
std::vector<cv::Point2f> Solver::world2pixel(const std::vector<cv::Point3f> & worldPoints)
{
  Eigen::Matrix3d R_world2camera = R_camera2gimbal_.transpose() * R_gimbal2world_.transpose();
  Eigen::Vector3d t_world2camera = -R_camera2gimbal_.transpose() * t_camera2gimbal_;

  cv::Mat rvec;
  cv::Mat tvec;
  cv::eigen2cv(R_world2camera, rvec);
  cv::eigen2cv(t_world2camera, tvec);

  std::vector<cv::Point3f> valid_world_points;
  for (const auto & world_point : worldPoints) {
    Eigen::Vector3d world_point_eigen(world_point.x, world_point.y, world_point.z);
    Eigen::Vector3d camera_point = R_world2camera * world_point_eigen + t_world2camera;

    if (camera_point.z() > 0) {
      valid_world_points.push_back(world_point);
    }
  }
  // 如果没有有效点，返回空vector
  if (valid_world_points.empty()) {
    return std::vector<cv::Point2f>();
  }
  std::vector<cv::Point2f> pixelPoints;
  cv::projectPoints(valid_world_points, rvec, tvec, camera_matrix_, distort_coeffs_, pixelPoints);
  return pixelPoints;
}

Eigen::Vector3d Solver::pixel2world_dir(const cv::Point2f & px) const
{
  std::vector<cv::Point2f> in{px}, out;
  cv::undistortPoints(in, out, camera_matrix_, distort_coeffs_);
  Eigen::Vector3d dir_cam(out[0].x, out[0].y, 1.0);
  return (R_camera2world() * dir_cam).normalized();
}

bool Solver::world2pixel(const Eigen::Vector3d & p_world, cv::Point2f & px) const
{
  const Eigen::Matrix3d R_world2camera = R_camera2world().transpose();
  const Eigen::Vector3d p_cam = R_world2camera * (p_world - camera_center_in_world());
  if (p_cam.z() <= 1e-3) return false;
  std::vector<cv::Point3f> obj{cv::Point3f(p_cam.x(), p_cam.y(), p_cam.z())};
  std::vector<cv::Point2f> img;
  cv::projectPoints(obj, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), camera_matrix_, distort_coeffs_, img);
  px = img[0];
  return true;
}

std::array<Eigen::Vector2d, 4> Solver::undistort_armor_points(const Armor & armor) const
{
  std::array<Eigen::Vector2d, 4> res{};
  if (armor.points.size() != 4) return res;
  std::vector<cv::Point2f> und;
  cv::undistortPoints(armor.points, und, camera_matrix_, distort_coeffs_);
  for (int i = 0; i < 4; ++i) {
    res[i] = Eigen::Vector2d(und[i].x, und[i].y);
  }
  return res;
}

bool Solver::armor_corners_in_camera(
  const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name,
  double ring_roll, double ring_pitch, std::array<Eigen::Vector3d, 4> & pts_cam,
  bool * front_facing) const
{
  const double sin_yaw = std::sin(yaw);
  const double cos_yaw = std::cos(yaw);
  const double pitch = (name == ArmorName::outpost) ? -15.0 * CV_PI / 180.0 : 15.0 * CV_PI / 180.0;
  const double sin_pitch = std::sin(pitch);
  const double cos_pitch = std::cos(pitch);

  const double cr = std::cos(ring_roll);
  const double sr = std::sin(ring_roll);
  const double cp = std::cos(ring_pitch);
  const double sp = std::sin(ring_pitch);
  // clang-format off
  const Eigen::Matrix3d R_ring{
    { cp, sr * sp, cr * sp},
    {  0,      cr,     -sr},
    {-sp, sr * cp, cr * cp}
  };
  const Eigen::Matrix3d R_mount{
    {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
    {sin_yaw * cos_pitch,  cos_yaw, sin_yaw * sin_pitch},
    {         -sin_pitch,        0,           cos_pitch}
  };
  // clang-format on
  const Eigen::Matrix3d R_armor2world = R_ring * R_mount;
  const Eigen::Matrix3d R_world2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose();
  const Eigen::Matrix3d R_armor2camera = R_world2camera * R_armor2world;
  const Eigen::Vector3d t_armor2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * xyz_in_world - t_camera2gimbal_);

  if (front_facing) {
    // 装甲板正面法向为局部 -X，朝向相机等价于 (-col0)·(-t) > 0
    *front_facing = (R_armor2camera.col(0).dot(t_armor2camera) > 0.0);
  }

  const auto & object_points = (type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;
  for (int i = 0; i < 4; ++i) {
    const Eigen::Vector3d p_local(object_points[i].x, object_points[i].y, object_points[i].z);
    pts_cam[i] = R_armor2camera * p_local + t_armor2camera;
    if (pts_cam[i].z() <= 1e-3) return false;
  }
  return true;
}

bool Solver::armor_lightbar_depth_diff(const Armor & armor, double & depth_diff) const
{
  if (armor.points.size() != 4) return false;
  // 复用带 15° 先验的 optimize_yaw 消除近正面 IPPE 镜像翻转，避免向 DiffMeasure 注入反号深度差
  Armor tmp = armor;
  solve(tmp);
  const double pitch =
    (tmp.name == ArmorName::outpost) ? -15.0 * CV_PI / 180.0 : 15.0 * CV_PI / 180.0;
  const double yaw = tmp.ypr_in_world[0];
  const double sy = std::sin(yaw);
  const double cy = std::cos(yaw);
  const double sp = std::sin(pitch);
  const double cp = std::cos(pitch);
  // clang-format off
  const Eigen::Matrix3d R_armor2world{
    {cy * cp, -sy, cy * sp},
    {sy * cp,  cy, sy * sp},
    {    -sp,   0,      cp}
  };
  // clang-format on
  const Eigen::Matrix3d R_armor2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * R_armor2world;
  const double width = (tmp.type == ArmorType::big) ? BIG_ARMOR_WIDTH : SMALL_ARMOR_WIDTH;
  // 左灯条中心 (0, +W/2, 0)，右灯条中心 (0, -W/2, 0)，深度差 z_L - z_R = R_cam(2,1) * W
  depth_diff = R_armor2camera(2, 1) * width;
  return std::isfinite(depth_diff);
}

}  // namespace auto_aim