#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/imgproc.hpp>

#include "tasks/auto_aim/aim_corrector.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tools/runtime_params.hpp"
#include "tools/trajectory.hpp"

int main()
{
  constexpr double D2R = M_PI / 180.0;
  constexpr double R2D = 180.0 / M_PI;

  // 1. 验证 ballistic_point 与 Trajectory（无阻力 & 二维耦合二次阻力）的一致性
  for (double k1 : {0.0, tools::RM_17MM_DRAG_K}) {
    const double v0 = 25.0;
    const double d = 4.5;
    const double h = 0.35;
    tools::Trajectory traj(v0, d, h, k1);
    if (traj.unsolvable) {
      std::cerr << "[FAIL] Trajectory unsolvable for k1=" << k1 << "\n";
      return 1;
    }
    double r = 0.0, z = 0.0;
    tools::ballistic_point(v0, traj.pitch, k1, traj.fly_time, r, z);
    if (std::abs(r - d) > 2e-3 || std::abs(z - h) > 2e-3) {
      std::cerr << "[FAIL] ballistic_point mismatch for k1=" << k1 << ": r=" << r << " (exp " << d
                << "), z=" << z << " (exp " << h << ")\n";
      return 1;
    }
  }
  std::cout << "[PASS] ballistic_point matches Trajectory (k1=0 & k1=RM_17MM_DRAG_K)\n";

  // 2. 合成带已知系统误差的弹丸视频流，验证 AimCorrector 检测 + 配对 + 误差估计 + Aimer/Planner 零偏同步写入
  const std::string cfg = "configs/sim.yaml";
  auto_aim::Solver solver(cfg);
  auto_aim::Aimer aimer(cfg);
  auto_aim::Planner planner(cfg);
  auto_aim::AimCorrector corrector(cfg);

  const double v0 = 25.0;
  const double k1 = tools::RM_17MM_DRAG_K;
  const double z0 = corrector.muzzle_z_offset();
  const double true_dyaw_deg = 1.25;
  const double true_dpitch_deg = -0.85;
  const double true_dyaw = true_dyaw_deg * D2R;
  const double true_dpitch = true_dpitch_deg * D2R;

  const auto t_base = auto_aim::AimCorrector::Clock::now();
  auto t_now = t_base;

  // 发射 6 发，每发模拟 0.35s 飞行 + 0.50s 结束结算，同时云台做小幅旋转以验证单应帧差补偿
  const int n_shots = 6;
  const double dt = 0.02;  // 50 Hz
  const cv::Size img_sz(1440, 1080);

  for (int shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    // configs/sim.yaml 中相机光轴相对真云台抬头 +25°（与仿真炮口一致）
    const double base_yaw = 0.05 * shot_idx;
    const double base_pitch = 25.0 * D2R;  // 炮口抬头为正（绕 y 轴是 -base_pitch）
    const Eigen::AngleAxisd Rz_fire(base_yaw, Eigen::Vector3d::UnitZ());
    const Eigen::AngleAxisd Ry_fire(-base_pitch, Eigen::Vector3d::UnitY());
    const Eigen::Matrix3d R_fire = (Rz_fire * Ry_fire).toRotationMatrix();

    const auto t_fire = t_now;
    corrector.add_shot(t_fire, R_fire, v0, k1);

    // 每发跑 45 帧（0.90 s > max_flight 0.8 s），保证自动 finalize
    for (int f = 0; f < 45; ++f) {
      const double t_flight = f * dt;
      t_now = t_fire + std::chrono::microseconds(static_cast<int64_t>(t_flight * 1e6));

      // 真云台轻微扫动（相机光轴位于云台 +25° 处，故弹丸落在画面中央附近），测试 IMU 旋转单应帧差补偿
      const double cam_yaw = base_yaw + 0.02 * std::sin(2.0 * M_PI * f / 30.0);
      const double cam_pitch = 0.01 * std::cos(2.0 * M_PI * f / 30.0);
      Eigen::Quaterniond q_cam(
        Eigen::AngleAxisd(cam_yaw, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(-cam_pitch, Eigen::Vector3d::UnitY()));
      solver.set_R_gimbal2world(q_cam);

      // 静态灰底背景 + 若干静止绿点（验证帧差能滤掉静止绿物）
      cv::Mat frame(img_sz, CV_8UC3, cv::Scalar(40, 40, 40));

      // 在真实偏置弹道位置渲染荧光绿弹丸（t_flight 在 0.04~0.35s 内）
      if (t_flight >= 0.04 && t_flight <= 0.35) {
        double r = 0.0, z = 0.0;
        tools::ballistic_point(v0, base_pitch + true_dpitch, k1, t_flight, r, z);
        const Eigen::Vector3d P_real(
          r * std::cos(base_yaw + true_dyaw), r * std::sin(base_yaw + true_dyaw), z0 + z);
        cv::Point2f px;
        if (solver.world2pixel(P_real, px)) {
          cv::circle(frame, px, 6, cv::Scalar(20, 240, 20), -1);
        }
      }

      corrector.update(frame, t_now, solver);
    }
    t_now += std::chrono::milliseconds(50);
  }

  const auto st = corrector.status();
  const bool ok = st["est"]["ok"].get<bool>();
  const int n_matched = st["est"]["n"].get<int>();
  const double ey = st["est"]["ey"].get<double>();
  const double ep = st["est"]["ep"].get<double>();
  std::cout << "[INFO] matched=" << n_matched << " ok=" << ok << " ey=" << ey << " deg (true "
            << true_dyaw_deg << "), ep=" << ep << " deg (true " << true_dpitch_deg << ")\n";

  if (!ok || n_matched < 5) {
    std::cerr << "[FAIL] AimCorrector matched insufficient shots: " << st.dump() << "\n";
    return 1;
  }
  if (std::abs(ey - true_dyaw_deg) > 0.15 || std::abs(ep - true_dpitch_deg) > 0.15) {
    std::cerr << "[FAIL] Estimation error too large: ey=" << ey << " ep=" << ep << "\n";
    return 1;
  }

  std::string msg;
  if (!corrector.apply_suggestion(msg)) {
    std::cerr << "[FAIL] apply_suggestion failed: " << msg << "\n";
    return 1;
  }
  std::cout << "[INFO] " << msg << "\n";

  auto & rp = tools::RuntimeParams::instance();
  const double ay = rp.get("aimer.yaw_offset").value_or(999);
  const double ap = rp.get("aimer.pitch_offset").value_or(999);
  const double py = rp.get("planner.yaw_offset").value_or(999);
  const double pp = rp.get("planner.pitch_offset").value_or(999);

  if (
    std::abs(ay + true_dyaw_deg) > 0.15 || std::abs(ap + true_dpitch_deg) > 0.15 ||
    std::abs(py + true_dyaw_deg) > 0.15 || std::abs(pp + true_dpitch_deg) > 0.15) {
    std::cerr << "[FAIL] Offsets not synced to Aimer/Planner: aimer=(" << ay << "," << ap
              << ") planner=(" << py << "," << pp << ")\n";
    return 1;
  }

  std::cout << "[PASS] AimCorrector end-to-end test passed (Aimer & Planner offsets synced)\n";
  return 0;
}
