#include "target.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>

#include "solver.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig)
: name(armor.name),
  armor_type(armor.type),
  jumped(false),
  last_id(0),
  update_count_(0),
  armor_num_(armor_num),
  t_(t),
  last_update_t_(t),
  is_switch_(false),
  is_converged_(false),
  switch_count_(0)
{
  auto r = radius;
  priority = armor.priority;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  // 装甲环平面倾角初始置 0（水平先验），由后续多帧观测在线估计。
  // 单帧 IPPE PnP 的 pitch/roll 存在平面镜像翻转与大噪声，直接作为初值会在第一帧就破坏 UVL 重投影。
  const double ring_roll = 0.0;
  const double ring_pitch = 0.0;

  // 旋转中心的坐标。环已倾斜时，半径不再只落在水平面。
  const Eigen::Vector3d radial = ring_rotation(ring_roll, ring_pitch) * Eigen::Vector3d{std::cos(ypr[0]), std::sin(ypr[0]), 0};
  auto center_x = xyz[0] + r * radial[0];
  auto center_y = xyz[1] + r * radial[1];
  auto center_z = xyz[2] + r * radial[2];

  // x vx y vy z vz a w r l h roll pitch
  // a: 参考装甲板 yaw
  // w: yaw 角速度
  // l: r2 - r1
  // h: z2 - z1（环平面内的高度差，倾斜后再旋到世界系）
  // roll/pitch: 装甲环平面相对水平面的倾角，常量
  Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, 0, ring_roll, ring_pitch}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    c[11] = tools::limit_rad(c[11]);
    c[12] = tools::limit_rad(c[12]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
  isinit = true;
}

Target::Target(double x, double vyaw, double radius, double h) : armor_num_(4)
{
  Eigen::VectorXd x0{{x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h, 0, 0}};
  Eigen::VectorXd P0_dig{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    c[11] = tools::limit_rad(c[11]);
    c[12] = tools::limit_rad(c[12]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
}

void Target::predict(std::chrono::steady_clock::time_point t, const Eigen::Vector3d & self_vel)
{
  auto dt = tools::delta_time(t, t_);
  predict(dt, self_vel);
  t_ = t;
}

void Target::predict(double dt, const Eigen::Vector3d & self_vel)
{
  // 状态转移矩阵
  // clang-format off
  Eigen::MatrixXd F{
    {1, dt,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  1, dt,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  1, dt,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  1, dt,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1}
  };
  // clang-format on

  // Piecewise White Noise Model
  // https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
  double v1, v1_z, v2;
  if (name == ArmorName::outpost) {
    v1 = 10;    // 前哨站水平加速度方差
    v1_z = 1;   // 前哨站竖直加速度方差
    v2 = 0.1;   // 前哨站角加速度方差
  } else {
    v1 = 8;     // 水平加速度方差：抑制云台快速跟随时图像-姿态微小错位被吸成假平动速度
    v1_z = 1;   // 竖直加速度方差：避免相邻板高度差 h 被 z/vz 吃掉
    v2 = 40;    // 角加速度方差（抑制换板时 w 剧烈甩动）
  }
  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;
  // 预测过程噪声偏差的方差
  // clang-format off
  // 倾角与高度差是常量，给很小的过程噪声，避免 P 在早期塌成 0 后学不动。
  const double v_tilt = 1e-4;
  const double v_h = 1e-3;
  Eigen::MatrixXd Q{
    {a * v1, b * v1,      0,      0,        0,        0,      0,      0, 0, 0,       0,       0,       0},
    {b * v1, c * v1,      0,      0,        0,        0,      0,      0, 0, 0,       0,       0,       0},
    {     0,      0, a * v1, b * v1,        0,        0,      0,      0, 0, 0,       0,       0,       0},
    {     0,      0, b * v1, c * v1,        0,        0,      0,      0, 0, 0,       0,       0,       0},
    {     0,      0,      0,      0, a * v1_z, b * v1_z,      0,      0, 0, 0,       0,       0,       0},
    {     0,      0,      0,      0, b * v1_z, c * v1_z,      0,      0, 0, 0,       0,       0,       0},
    {     0,      0,      0,      0,        0,        0, a * v2, b * v2, 0, 0,       0,       0,       0},
    {     0,      0,      0,      0,        0,        0, b * v2, c * v2, 0, 0,       0,       0,       0},
    {     0,      0,      0,      0,        0,        0,      0,      0, 0, 0,       0,       0,       0},
    {     0,      0,      0,      0,        0,        0,      0,      0, 0, 0,       0,       0,       0},
    {     0,      0,      0,      0,        0,        0,      0,      0, 0, 0, c * v_h,       0,       0},
    {     0,      0,      0,      0,        0,        0,      0,      0, 0, 0,       0, c * v_tilt,    0},
    {     0,      0,      0,      0,        0,        0,      0,      0, 0, 0,       0,       0, c * v_tilt}
  };
  // clang-format on

  // 防止夹角求和出现异常值；同时扣除自身在世界系中的位移 self_vel * dt，
  // 保持位置相对当前云台原点、速度 [vx,vy,vz] 为目标在惯性世界系的真实速度。
  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[0] -= self_vel.x() * dt;
    x_prior[2] -= self_vel.y() * dt;
    x_prior[4] -= self_vel.z() * dt;
    x_prior[6] = tools::limit_rad(x_prior[6]);
    x_prior[11] = tools::limit_rad(x_prior[11]);
    x_prior[12] = tools::limit_rad(x_prior[12]);
    return x_prior;
  };

  // 前哨站转速特判
  if (this->convergened() && this->name == ArmorName::outpost && std::abs(this->ekf_.x[7]) > 2)
    this->ekf_.x[7] = this->ekf_.x[7] > 0 ? 2.51 : -2.51;

  ekf_.predict(F, Q, f);
}

bool Target::update(Armor & armor, const Solver & solver)
{
  // 每块预测板各消歧一次，再和检测的位置一起打分。
  // 先用上一帧的板消歧再按朝向匹配，会把刚转入画面的新板写进旧板，相位每圈少一块。
  const std::vector<Eigen::Vector4d> xyza_list = armor_xyza_list();
  const Eigen::Vector3d raw_yaw = armor.ypr_in_world;

  int id = -1;
  double best_pos = 1e10;
  double best_yaw = armor.ypr_in_world[0];
  for (int i = 0; i < armor_num_; i++) {
    armor.ypr_in_world = raw_yaw;
    solver.resolve_yaw(armor, ekf_.x[11], ekf_.x[12], xyza_list[i][3]);
    const double pos = (armor.xyz_in_world - xyza_list[i].head(3)).norm();
    const double gap = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza_list[i][3]));
    // 0.15 rad 约等于相邻板在画面边缘的朝向差。位置仍是主项，避免侧面板仅靠 yaw 选中。
    const double score = pos + 0.15 * gap;
    if (score < best_pos) {
      best_pos = score;
      best_yaw = armor.ypr_in_world[0];
      id = i;
    }
  }

  if (id < 0 || best_pos > 0.6) return false;
  armor.ypr_in_world = raw_yaw;
  armor.ypr_in_world[0] = best_yaw;
  update_with_id(armor, id);
  return true;
}

void Target::update(const Armor & armor)
{
  // 装甲板匹配
  int id;
  auto min_angle_error = 1e10;
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

  std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
  for (int i = 0; i < armor_num_; i++) {
    xyza_i_list.push_back({xyza_list[i], i});
  }

  std::sort(
    xyza_i_list.begin(), xyza_i_list.end(),
    [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
      Eigen::Vector3d ypd1 = tools::xyz2ypd(a.first.head(3));
      Eigen::Vector3d ypd2 = tools::xyz2ypd(b.first.head(3));
      return ypd1[2] < ypd2[2];
    });

  // 取前3个distance最小的装甲板。
  for (int i = 0; i < 3 && i < static_cast<int>(xyza_i_list.size()); i++) {
    const auto & xyza = xyza_i_list[i].first;
    Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
    auto angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
                       std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));

    if (std::abs(angle_error) < std::abs(min_angle_error)) {
      id = xyza_i_list[i].second;
      min_angle_error = angle_error;
    }
  }

  // 没有朝向吻合的候选时不更新。退回“最近的一块”会把侧面板的观测写进当前板，半径会被拉负。
  if (min_angle_error > 1e9) return;

  update_with_id(armor, id);
}

void Target::update_with_id(const Armor & armor, int id)
{
  if (id != 0) jumped = true;

  if (id != last_id) {
    is_switch_ = true;
  } else {
    is_switch_ = false;
  }

  if (is_switch_) switch_count_++;

  last_id = id;
  update_count_++;
  last_update_t_ = t_;

  update_ypda(armor, id);
}

namespace
{
inline Eigen::Vector4d points_to_uvl(const Eigen::Vector2d & top, const Eigen::Vector2d & bottom)
{
  const Eigen::Vector2d delta = top - bottom;
  const Eigen::Vector2d center = 0.5 * (top + bottom);
  return Eigen::Vector4d(
    std::atan2(delta.x(), delta.y()), center.x(), center.y(), delta.norm());
}
}  // namespace

bool Target::update_uvl(std::vector<Armor *> & armors, const Solver & solver)
{
  if (armors.empty()) return false;

  // 1. 筛选正面朝向相机的候选板；若全被判背面则退回所有位于相机前方的板
  const double roll = ekf_.x[11];
  const double pitch = ekf_.x[12];
  std::vector<int> front_ids;
  std::vector<int> in_front_ids;
  for (int i = 0; i < armor_num_; ++i) {
    const double yaw = tools::limit_rad(ekf_.x[6] + i * 2.0 * CV_PI / armor_num_);
    const Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    std::array<Eigen::Vector3d, 4> pts_cam;
    bool front_facing = false;
    if (!solver.armor_corners_in_camera(
          xyz, yaw, armor_type, name, roll, pitch, pts_cam, &front_facing)) {
      continue;
    }
    in_front_ids.push_back(i);
    if (front_facing) front_ids.push_back(i);
  }
  const auto & candidates = !front_ids.empty() ? front_ids : in_front_ids;
  if (candidates.empty()) return false;

  // 2. 图像空间四点代价匹配（中心误差 + 四边角度误差 + 周长比例误差）
  std::vector<std::vector<cv::Point2f>> pred_quads;
  pred_quads.reserve(candidates.size());
  for (int id : candidates) {
    const double yaw = tools::limit_rad(ekf_.x[6] + id * 2.0 * CV_PI / armor_num_);
    const Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, id);
    pred_quads.push_back(solver.reproject_armor(xyz, yaw, armor_type, name, roll, pitch));
  }

  constexpr double kWCenter = 5.0;
  constexpr double kWAngle = 10.0;
  constexpr double kWSide = 1.0;
  const double gate = (update_count_ < 10) ? 1000.0 : 300.0;

  std::vector<std::tuple<double, size_t, size_t>> edges;
  for (size_t j = 0; j < armors.size(); ++j) {
    const auto & meas = armors[j]->points;
    if (meas.size() != 4) continue;
    cv::Point2f cm(0.f, 0.f);
    double perim_m = 0.0;
    for (int k = 0; k < 4; ++k) {
      cm += meas[k];
      perim_m += cv::norm(meas[k] - meas[(k + 1) % 4]);
    }
    cm *= 0.25f;

    for (size_t ci = 0; ci < candidates.size(); ++ci) {
      const auto & pred = pred_quads[ci];
      if (pred.size() != 4) continue;
      cv::Point2f cp(0.f, 0.f);
      double perim_p = 0.0;
      double angle_err = 0.0;
      for (int k = 0; k < 4; ++k) {
        cp += pred[k];
        const cv::Point2f dp = pred[(k + 1) % 4] - pred[k];
        const cv::Point2f dm = meas[(k + 1) % 4] - meas[k];
        perim_p += std::hypot(dp.x, dp.y);
        angle_err +=
          std::abs(tools::limit_rad(std::atan2(dp.y, dp.x) - std::atan2(dm.y, dm.x)));
      }
      cp *= 0.25f;
      if (perim_p < 1e-3) continue;
      const double center_err = cv::norm(cp - cm);
      const double side_err = std::abs(perim_p - perim_m) / perim_p;
      const double cost = kWCenter * center_err + kWAngle * angle_err + kWSide * side_err;
      if (std::isfinite(cost) && cost < gate) {
        edges.emplace_back(cost, j, ci);
      }
    }
  }

  if (edges.empty()) return false;
  std::sort(
    edges.begin(), edges.end(),
    [](const auto & a, const auto & b) { return std::get<0>(a) < std::get<0>(b); });

  std::vector<bool> used_obs(armors.size(), false);
  std::vector<bool> used_cand(candidates.size(), false);
  std::vector<std::pair<int, Armor *>> matched;
  for (const auto & [cost, obs_i, cand_i] : edges) {
    if (used_obs[obs_i] || used_cand[cand_i]) continue;
    used_obs[obs_i] = true;
    used_cand[cand_i] = true;
    matched.emplace_back(candidates[cand_i], armors[obs_i]);
  }
  if (matched.empty()) return false;

  // 3. 构造 UVL (+ 单板深度差) 多观测联合更新向量 z 和噪声方差 R
  const double fx = std::max(1.0, solver.fx());
  const double fy = std::max(1.0, solver.fy());
  const double f_avg = 0.5 * (fx + fy);

  bool use_diff = false;
  double depth_diff_obs = 0.0;
  if (matched.size() == 1 && solver.armor_lightbar_depth_diff(*matched[0].second, depth_diff_obs)) {
    use_diff = true;
  }

  const int n_lights = static_cast<int>(matched.size()) * 2;
  const int z_dim = n_lights * 4 + (use_diff ? 1 : 0);
  Eigen::VectorXd z(z_dim);
  Eigen::VectorXd R_dig(z_dim);

  int row = 0;
  for (const auto & [id, armor] : matched) {
    const auto und = solver.undistort_armor_points(*armor);
    auto fill_light = [&](int top_i, int bot_i) {
      const Eigen::Vector4d uvl = points_to_uvl(und[top_i], und[bot_i]);
      const double len_px = std::max(4.0, cv::norm(armor->points[top_i] - armor->points[bot_i]));
      const double sigma_px = std::max(1.5, 0.20 * len_px);
      const double sx = sigma_px / fx;
      const double sy = sigma_px / fy;
      // 实车灯条端点长度受曝光/阈值影响、角度受小像素高度与外参滚转残差影响，
      // 对齐 awakening 实车配置（omni/leg.yaml: length_ratio=0.5, angle=0.5）避免误触发 NIS 失锁。
      const double sl = std::max(2.5, 0.45 * len_px) / f_avg;
      const double sa = 0.45;
      z.segment<4>(row) = uvl;
      R_dig[row + 0] = 0.5 * sa * sa;
      R_dig[row + 1] = 0.5 * sx * sx;
      R_dig[row + 2] = 0.5 * sy * sy;
      R_dig[row + 3] = 0.5 * sl * sl;
      row += 4;
    };
    fill_light(0, 3);  // 左灯条: LT -> LB
    fill_light(1, 2);  // 右灯条: RT -> RB
  }
  if (use_diff) {
    constexpr double kSigmaDiff = 0.15;
    z[row] = depth_diff_obs;
    R_dig[row] = 0.5 * kSigmaDiff * kSigmaDiff;
  }

  auto h = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd z_pred = Eigen::VectorXd::Zero(z_dim);
    int r = 0;
    for (const auto & [id, armor] : matched) {
      const double yaw = tools::limit_rad(x[6] + id * 2.0 * CV_PI / armor_num_);
      const Eigen::Vector3d xyz = h_armor_xyz(x, id);
      std::array<Eigen::Vector3d, 4> pts_cam;
      solver.armor_corners_in_camera(xyz, yaw, armor_type, name, x[11], x[12], pts_cam);
      auto to_norm = [](const Eigen::Vector3d & p) {
        const double inv_z = 1.0 / std::max(1e-3, p.z());
        return Eigen::Vector2d(p.x() * inv_z, p.y() * inv_z);
      };
      const Eigen::Vector2d lt = to_norm(pts_cam[0]);
      const Eigen::Vector2d rt = to_norm(pts_cam[1]);
      const Eigen::Vector2d rb = to_norm(pts_cam[2]);
      const Eigen::Vector2d lb = to_norm(pts_cam[3]);
      z_pred.segment<4>(r) = points_to_uvl(lt, lb);
      r += 4;
      z_pred.segment<4>(r) = points_to_uvl(rt, rb);
      r += 4;
      if (use_diff) {
        const double z_l = 0.5 * (pts_cam[0].z() + pts_cam[3].z());
        const double z_r = 0.5 * (pts_cam[1].z() + pts_cam[2].z());
        z_pred[r] = z_l - z_r;
      }
    }
    return z_pred;
  };

  auto z_subtract = [n_lights](const Eigen::VectorXd & a, const Eigen::VectorXd & b) {
    Eigen::VectorXd c = a - b;
    for (int i = 0; i < n_lights; ++i) {
      c[i * 4] = tools::limit_rad(c[i * 4]);
    }
    return c;
  };

  // 中心差分雅可比 H (z_dim x state_dim)
  constexpr double kStep = 1e-6;
  const int x_dim = static_cast<int>(ekf_.x.size());
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(z_dim, x_dim);
  for (int col = 0; col < x_dim; ++col) {
    Eigen::VectorXd up = ekf_.x;
    Eigen::VectorXd down = ekf_.x;
    up[col] += kStep;
    down[col] -= kStep;
    H.col(col) = z_subtract(h(up), h(down)) / (2.0 * kStep);
  }

  Eigen::MatrixXd R = R_dig.asDiagonal();
  ekf_.update(z, H, R, h, z_subtract);

  // 更新跟踪状态标记（取本帧代价最低的主板 id 作为 last_id）
  const int primary_id = matched.front().first;
  for (const auto & [id, _] : matched) {
    if (id != 0) jumped = true;
  }
  if (primary_id != last_id) {
    is_switch_ = true;
    switch_count_++;
  } else {
    is_switch_ = false;
  }
  last_id = primary_id;
  update_count_++;
  last_update_t_ = t_;
  return true;
}

cv::Rect Target::focus_roi(
  const Solver & solver, const cv::Size & img_size, std::chrono::steady_clock::time_point now,
  double lost_timeout_s, double target_wh_ratio, const Eigen::Vector3d & self_vel) const
{
  const cv::Rect full_rect(0, 0, img_size.width, img_size.height);
  if (img_size.width <= 0 || img_size.height <= 0) return full_rect;

  // 临时前向推演到 now，不改动滤波器内部状态
  Eigen::VectorXd x_pred = ekf_.x;
  const double dt_pred = std::clamp(tools::delta_time(now, t_), 0.0, 0.25);
  x_pred[0] += (x_pred[1] - self_vel.x()) * dt_pred;
  x_pred[2] += (x_pred[3] - self_vel.y()) * dt_pred;
  x_pred[4] += (x_pred[5] - self_vel.z()) * dt_pred;
  x_pred[6] = tools::limit_rad(x_pred[6] + x_pred[7] * dt_pred);

  std::vector<cv::Point2f> proj_pts;
  proj_pts.reserve(armor_num_ * 4);
  for (int i = 0; i < armor_num_; ++i) {
    const double yaw = tools::limit_rad(x_pred[6] + i * 2.0 * CV_PI / armor_num_);
    const Eigen::Vector3d xyz = h_armor_xyz(x_pred, i);
    std::array<Eigen::Vector3d, 4> pts_cam;
    if (!solver.armor_corners_in_camera(
          xyz, yaw, armor_type, name, x_pred[11], x_pred[12], pts_cam)) {
      continue;
    }
    auto px = solver.reproject_armor(xyz, yaw, armor_type, name, x_pred[11], x_pred[12]);
    proj_pts.insert(proj_pts.end(), px.begin(), px.end());
  }
  if (proj_pts.empty()) return full_rect;

  cv::Rect rect = cv::boundingRect(proj_pts);
  const double expand = (name == ArmorName::base) ? 3.0 : 1.6;
  const double cx = rect.x + rect.width * 0.5;
  const double cy = rect.y + rect.height * 0.5;
  double w = std::max(rect.width * expand, 256.0);
  double h = std::max(rect.height * expand, 256.0);

  const double ratio = (std::isfinite(target_wh_ratio) && target_wh_ratio > 0.0)
                         ? target_wh_ratio
                         : 1.0;
  if (w / h < ratio) {
    w = h * ratio;
  } else {
    h = w / ratio;
  }

  const double lost_dt =
    std::clamp(tools::delta_time(now, last_update_t_), 0.0, std::max(1e-3, lost_timeout_s));
  const double alpha = lost_dt / std::max(1e-3, lost_timeout_s);
  const double base_side = std::max(w, h);
  const double max_side = static_cast<double>(std::max(img_size.width, img_size.height));
  const double side = base_side + (max_side - base_side) * alpha;

  if (side >= max_side * 0.95) return full_rect;

  cv::Rect roi(
    static_cast<int>(std::round(cx - side * 0.5)),
    static_cast<int>(std::round(cy - side * 0.5)),
    static_cast<int>(std::round(side)), static_cast<int>(std::round(side)));
  roi &= full_rect;
  if (roi.width < 128 || roi.height < 128) return full_rect;
  return roi;
}

void Target::update_ypda(const Armor & armor, int id)
{
  //观测jacobi
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
  // Eigen::VectorXd R_dig{{4e-3, 4e-3, 1, 9e-2}};
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  // 后两维是板面 pitch/roll。大倾角时 PnP 的法向噪声明显大于 yaw，给 0.05 rad²，
  // 大约 13° 的 1σ，避免单帧法向把半径和高度差拉飞。
  // ypd[0]/ypd[1]（图像方位/俯仰角）像素精度远高于 4e-3 rad²，统一收紧到 4e-4 rad²（1σ≈1.15°），
  // 既让相位与高度差 h 快速收敛，又避免过紧触发 NIS 误判。
  Eigen::VectorXd R_dig{
    {4e-4, 4e-4, log(std::abs(delta_angle) + 1) + 1,
     log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2, 0.05, 0.05}};

  //测量过程噪声偏差的方差
  Eigen::MatrixXd R = R_dig.asDiagonal();

  // 定义非线性转换函数h: x -> z
  // z = [yaw, pitch, distance, armor_yaw, armor_pitch, armor_roll]
  // 板面 yaw 继续更新旋转相位。pitch/roll 用来估计装甲环倾角。
  auto h = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    Eigen::Vector3d ypr = h_armor_ypr(x, id);
    return (Eigen::VectorXd(6) << ypd[0], ypd[1], ypd[2], ypr[0], ypr[1], ypr[2]).finished();
  };

  // 防止夹角求差出现异常值
  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    c[4] = tools::limit_rad(c[4]);
    c[5] = tools::limit_rad(c[5]);
    return c;
  };

  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0], ypr[1], ypr[2]}};

  ekf_.update(z, H, R, h, z_subtract);
}

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> _armor_xyza_list;

  for (int i = 0; i < armor_num_; i++) {
    auto angle = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
    Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    _armor_xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return _armor_xyza_list;
}

std::vector<Eigen::Vector<double, 6>> Target::armor_xyzayp_list() const
{
  std::vector<Eigen::Vector<double, 6>> poses;
  for (const auto & xyza : armor_xyza_list()) {
    poses.push_back({xyza[0], xyza[1], xyza[2], xyza[3], ekf_.x[11], ekf_.x[12]});
  }
  return poses;
}

bool Target::diverged() const
{
  auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
  auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;

  if (r_ok && l_ok) return false;

  tools::logger()->debug("[Target] r={:.3f}, l={:.3f}", ekf_.x[8], ekf_.x[9]);
  return true;
}

bool Target::convergened()
{
  if (this->name != ArmorName::outpost && update_count_ > 3 && !this->diverged()) {
    is_converged_ = true;
  }

  //前哨站特殊判断
  if (this->name == ArmorName::outpost && update_count_ > 10 && !this->diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}

Eigen::Matrix3d Target::ring_rotation(double roll, double pitch)
{
  const double cr = std::cos(roll);
  const double sr = std::sin(roll);
  const double cp = std::cos(pitch);
  const double sp = std::sin(pitch);
  // R = Rx(roll) * Ry(pitch)。水平环先绕 y 再绕 x，得到装甲环平面。
  // clang-format off
  return Eigen::Matrix3d{
    { cp, sr * sp, cr * sp},
    {  0,      cr,     -sr},
    {-sp, sr * cp, cr * cp}
  };
  // clang-format on
}

Eigen::Vector3d Target::armor_offset(const Eigen::VectorXd & x, int id, int armor_num)
{
  const double angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num);
  // 四板用对角两块做第二组半径/高度；平衡两板的另一块用 l。
  const bool use_l_h = armor_num == 2 ? id == 1 : armor_num == 4 && (id == 1 || id == 3);
  const double radius = use_l_h ? x[8] + x[9] : x[8];
  const double height = use_l_h ? x[10] : 0.0;
  // 环平面内：半径沿板朝向的反方向，高度差沿环的竖直方向。
  return ring_rotation(x[11], x[12]) * Eigen::Vector3d{-radius * std::cos(angle), -radius * std::sin(angle), height};
}

// 计算出装甲板中心的坐标（考虑长短轴和装甲环倾角）
Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  // 前哨站/基地三块板没有第二组半径和高度，也不估计倾角。
  // 平衡两块板和普通四块板都走倾斜环。1 号英雄是四块板。
  if (armor_num_ == 3) {
    const double angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    return {x[0] - x[8] * std::cos(angle), x[2] - x[8] * std::sin(angle), x[4]};
  }
  return Eigen::Vector3d{x[0], x[2], x[4]} + armor_offset(x, id, armor_num_);
}

Eigen::Vector3d Target::h_armor_ypr(const Eigen::VectorXd & x, int id) const
{
  const double yaw = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  if (armor_num_ == 3) return {yaw, 0, 0};

  const double mount_pitch = (name == ArmorName::outpost) ? -15.0 * CV_PI / 180.0 : 15.0 * CV_PI / 180.0;
  const double yaw_s = std::sin(yaw);
  const double yaw_c = std::cos(yaw);
  const double mount_s = std::sin(mount_pitch);
  const double mount_c = std::cos(mount_pitch);
  // 与 solver::reproject_armor 同一套 ZYX：先环的 yaw，再叠固定安装角。
  // clang-format off
  const Eigen::Matrix3d R_mount{
    {yaw_c * mount_c, -yaw_s, yaw_c * mount_s},
    {yaw_s * mount_c,  yaw_c, yaw_s * mount_s},
    {       -mount_s,      0,        mount_c}
  };
  // clang-format on
  return tools::eulers(ring_rotation(x[11], x[12]) * R_mount, 2, 1, 0);
}

Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  // 位置对倾角、法向对 yaw/倾角都是旋转复合，解析式容易和观测函数对不上。
  // 这里对 h() 做中心差分，和非线性观测用同一份几何。
  Eigen::VectorXd x_local = x;
  if (x_local.size() < 13) {
    x_local.conservativeResize(13);
    x_local.tail(13 - x.size()).setZero();
  }

  auto h = [&](const Eigen::VectorXd & state) {
    Eigen::Vector3d xyz = h_armor_xyz(state, id);
    Eigen::Vector3d ypd = tools::xyz2ypd(xyz);
    Eigen::Vector3d ypr = h_armor_ypr(state, id);
    Eigen::Matrix<double, 6, 1> z;
    z << ypd[0], ypd[1], ypd[2], ypr[0], ypr[1], ypr[2];
    return z;
  };
  auto subtract = [](const Eigen::Matrix<double, 6, 1> & a, const Eigen::Matrix<double, 6, 1> & b) {
    Eigen::Matrix<double, 6, 1> c = a - b;
    for (int i : {0, 1, 3, 4, 5}) c[i] = tools::limit_rad(c[i]);
    return c;
  };

  constexpr double kStep = 1e-6;
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(6, x_local.size());
  for (int col = 0; col < x_local.size(); ++col) {
    Eigen::VectorXd up = x_local;
    Eigen::VectorXd down = x_local;
    up[col] += kStep;
    down[col] -= kStep;
    H.col(col) = subtract(h(up), h(down)) / (2 * kStep);
  }
  return H;
}

bool Target::checkinit() { return isinit; }

}  // namespace auto_aim
