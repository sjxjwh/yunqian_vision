#include <fmt/core.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <list>
#include <random>
#include <vector>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tools/math_tools.hpp"

using namespace auto_aim;

namespace
{
struct SimCar
{
  Eigen::Vector3d center{3.5, 0.2, -0.1};
  Eigen::Vector3d vel{0.0, 0.0, 0.0};
  double yaw0 = 0.0;
  double w = 4.5;
  double r1 = 0.25;
  double r2 = 0.28;
  double h = 0.09;
  double ring_roll = 0.0;
  double ring_pitch = 0.0;
};

Eigen::Matrix3d ring_rot(double roll, double pitch)
{
  const double cr = std::cos(roll), sr = std::sin(roll);
  const double cp = std::cos(pitch), sp = std::sin(pitch);
  Eigen::Matrix3d R;
  R << cp, sr * sp, cr * sp, 0, cr, -sr, -sp, sr * cp, cr * cp;
  return R;
}

std::list<Armor> synthesize_frame(
  const SimCar & car, double t_sec, const Solver & solver, std::mt19937 & rng, double noise_px)
{
  std::normal_distribution<float> dist(0.0f, static_cast<float>(noise_px));
  const Eigen::Vector3d c = car.center + car.vel * t_sec;
  const Eigen::Matrix3d R_ring = ring_rot(car.ring_roll, car.ring_pitch);

  std::list<Armor> armors;
  for (int i = 0; i < 4; ++i) {
    const double yaw = tools::limit_rad(car.yaw0 + car.w * t_sec + i * CV_PI / 2.0);
    const bool odd = (i == 1 || i == 3);
    const double r = odd ? car.r2 : car.r1;
    const double z_off = odd ? car.h : 0.0;
    const Eigen::Vector3d xyz =
      c + R_ring * Eigen::Vector3d(-r * std::cos(yaw), -r * std::sin(yaw), z_off);

    // 仅保留朝向相机的板（板面法向与视线夹角 < 62°）
    std::array<Eigen::Vector3d, 4> pts_cam;
    bool front = false;
    if (!solver.armor_corners_in_camera(
          xyz, yaw, ArmorType::small, ArmorName::four, car.ring_roll, car.ring_pitch, pts_cam,
          &front)) {
      continue;
    }
    if (!front) continue;
    const Eigen::Vector3d n_world = R_ring * Eigen::Vector3d(-std::cos(yaw), -std::sin(yaw), 0.0);
    const Eigen::Vector3d to_cam = (-xyz).normalized();
    if (n_world.dot(to_cam) < std::cos(62.0 * CV_PI / 180.0)) continue;

    auto px = solver.reproject_armor(
      xyz, yaw, ArmorType::small, ArmorName::four, car.ring_roll, car.ring_pitch);
    if (px.size() != 4) continue;

    std::vector<cv::Point2f> noisy_pts(4);
    cv::Point2f center(0.f, 0.f);
    for (int k = 0; k < 4; ++k) {
      noisy_pts[k] = cv::Point2f(px[k].x + dist(rng), px[k].y + dist(rng));
      center += noisy_pts[k];
    }
    Armor a(0, 4, 0.95f, cv::boundingRect(noisy_pts), noisy_pts);
    a.color = Color::red;
    a.name = ArmorName::four;
    a.type = ArmorType::small;
    a.priority = ArmorPriority::forth;
    a.center = center * 0.25f;
    a.center_norm = cv::Point2f(a.center.x / 1440.f, a.center.y / 1080.f);
    armors.push_back(a);
  }
  return armors;
}

bool run_scenario(
  const std::string & title, const SimCar & car, const std::string & cfg, double max_w_err,
  double max_h_err, double max_med_px)
{
  Solver solver(cfg);
  solver.set_R_gimbal2world(Eigen::Quaterniond::Identity());
  Tracker tracker(cfg, solver);

  std::mt19937 rng(42);
  const auto t0 = std::chrono::steady_clock::now();
  const int total_frames = 300;  // 3.0s @ 100Hz
  int tracked_after_warmup = 0;
  std::vector<double> w_errs, h_errs, px_errs;

  for (int f = 0; f < total_frames; ++f) {
    const double t_sec = f * 0.01;
    const auto tp = t0 + std::chrono::milliseconds(f * 10);
    auto armors = synthesize_frame(car, t_sec, solver, rng, 1.0);
    auto armors_copy = armors;
    auto targets = tracker.track(armors_copy, tp);

    if (f >= 120) {
      if (targets.empty()) {
        fmt::print("[FAIL] {}: 第 {} 帧失锁！\n", title, f);
        return false;
      }
      tracked_after_warmup++;
      const auto & tgt = targets.front();
      const auto x = tgt.ekf_x();
      w_errs.push_back(std::abs(x[7] - car.w));
      // h 可能取 +h 或 -h（取决于初始化先看到高板还是低板）
      h_errs.push_back(std::min(std::abs(x[10] - car.h), std::abs(x[10] + car.h)));

      if (!armors.empty()) {
        double best_err = 1e9;
        for (const auto & ap : tgt.armor_xyzayp_list()) {
          auto proj = solver.reproject_armor(
            ap.head(3), ap[3], tgt.armor_type, tgt.name, ap[4], ap[5]);
          cv::Point2f cp(0.f, 0.f);
          for (const auto & p : proj) cp += p;
          cp *= 0.25f;
          best_err = std::min(best_err, cv::norm(cp - armors.front().center));
        }
        px_errs.push_back(best_err);
      }

      // 检验动态聚焦 ROI：必须缩小且完整覆盖当前可见板
      if (f == 200 && !armors.empty()) {
        const cv::Size img_sz(1440, 1080);
        const cv::Rect roi = tracker.focus_roi(img_sz, tp);
        const double area_ratio =
          static_cast<double>(roi.area()) / static_cast<double>(img_sz.area());
        bool covers_all = true;
        for (const auto & a : armors) {
          for (const auto & pt : a.points) {
            if (!roi.contains(cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)))) {
              covers_all = false;
            }
          }
        }
        if (!covers_all || area_ratio >= 0.65) {
          fmt::print(
            "[FAIL] {}: focus_roi 异常 roi={}x{} (ratio={:.2f}) covers_all={}\n", title,
            roi.width, roi.height, area_ratio, covers_all);
          return false;
        }
        // 丢帧超时后应退回全图
        const cv::Rect lost_roi =
          tracker.focus_roi(img_sz, tp + std::chrono::milliseconds(300));
        if (lost_roi.width != img_sz.width || lost_roi.height != img_sz.height) {
          fmt::print(
            "[FAIL] {}: 丢帧超时后 focus_roi 未退回全图 ({}x{})\n", title, lost_roi.width,
            lost_roi.height);
          return false;
        }
      }
    }
  }

  auto median = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  const double med_w = median(w_errs);
  const double med_h = median(h_errs);
  const double med_px = median(px_errs);

  fmt::print(
    "[{}] tracked={}/180 | med|w_err|={:.3f} rad/s | med|h_err|={:.4f} m | med_px_err={:.2f} px\n",
    title, tracked_after_warmup, med_w, med_h, med_px);

  if (med_w > max_w_err || med_h > max_h_err || med_px > max_med_px) {
    fmt::print(
      "[FAIL] {} 超出阈值 (lim w<={:.2f}, h<={:.3f}, px<={:.1f})\n", title, max_w_err, max_h_err,
      max_med_px);
    return false;
  }
  return true;
}
}  // namespace

int main(int argc, char ** argv)
{
  const std::string cfg = (argc > 1) ? argv[1] : "configs/standard3.yaml";

  // 1. 深度差符号与精度校验
  {
    Solver solver(cfg);
    solver.set_R_gimbal2world(Eigen::Quaterniond::Identity());
    const Eigen::Vector3d xyz(3.0, 0.0, 0.0);
    const double yaw = 0.5;  // 左偏：左灯条更远还是更近由几何决定，与 armor_corners_in_camera 一致
    std::array<Eigen::Vector3d, 4> pts_cam;
    if (!solver.armor_corners_in_camera(xyz, yaw, ArmorType::small, ArmorName::four, 0, 0, pts_cam)) {
      std::cerr << "[FAIL] armor_corners_in_camera failed\n";
      return 1;
    }
    const double gt_diff =
      0.5 * (pts_cam[0].z() + pts_cam[3].z()) - 0.5 * (pts_cam[1].z() + pts_cam[2].z());
    auto pts = solver.reproject_armor(xyz, yaw, ArmorType::small, ArmorName::four, 0, 0);
    Armor a(0, 4, 0.95f, cv::boundingRect(pts), pts);
    a.type = ArmorType::small;
    a.name = ArmorName::four;
    a.points = pts;
    double ippe_diff = 0.0;
    if (!solver.armor_lightbar_depth_diff(a, ippe_diff) || std::abs(ippe_diff - gt_diff) > 0.015) {
      fmt::print("[FAIL] depth_diff mismatch: gt={:.4f} ippe={:.4f}\n", gt_diff, ippe_diff);
      return 1;
    }
    fmt::print("[DepthDiff] PASS gt={:+.4f}m ippe={:+.4f}m\n", gt_diff, ippe_diff);
  }

  // 2. 平地大高低差高速旋转整车
  SimCar flat_big_h;
  flat_big_h.r1 = 0.25;
  flat_big_h.r2 = 0.28;
  flat_big_h.h = 0.09;
  flat_big_h.w = 4.5;
  if (!run_scenario("FlatBigHeightDiff", flat_big_h, cfg, 0.35, 0.02, 3.5)) return 1;

  // 3. 斜面旋转整车（KI-001 回归）
  SimCar slope_car;
  slope_car.r1 = 0.25;
  slope_car.r2 = 0.25;
  slope_car.h = 0.0;
  slope_car.w = 3.5;
  slope_car.ring_roll = 5.0 * CV_PI / 180.0;
  slope_car.ring_pitch = -8.0 * CV_PI / 180.0;
  if (!run_scenario("SlopeSpinCar", slope_car, cfg, 0.50, 0.03, 4.5)) return 1;

  fmt::print("ALL UVL + FOCUS_ROI TESTS PASSED!\n");
  return 0;
}
