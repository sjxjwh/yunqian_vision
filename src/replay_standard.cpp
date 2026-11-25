#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/sync_recorder.hpp"
#include "tools/web_viewer.hpp"

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明 }"
  "{@record-dir    |                        | SyncRecorder 录制目录路径（含 video.avi/frames.csv/imu.csv） }"
  "{config c       | configs/standard3.yaml | yaml 配置文件路径 }"
  "{latency-ms     | -9999.0                | 覆盖 camera_latency_ms（默认读 YAML） }"
  "{use-raw-q      | false                  | 离线改用串口原始 raw_q * R_gimbal2imu }"
  "{use-euler-q    | false                  | 离线改用串口 yaw/pitch 欧拉角构造姿态 }"
  "{force-aim      | true                   | 忽略录制 mode，强制全程运行自瞄链路 }"
  "{speed          | 0.0                    | 回放倍速（0=全速离线评估，1.0=原速） }"
  "{loop           | false                  | 循环回放（配合 --web 调试用） }"
  "{web            | false                  | 开启 WebViewer 离线回放可视化 }"
  "{port p         | 8090                   | WebViewer 监听端口 }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help") || cli.get<std::string>("@record-dir").empty()) {
    cli.printMessage();
    return 0;
  }

  const auto record_dir = cli.get<std::string>("@record-dir");
  const auto config_path = cli.get<std::string>("config");
  const double latency_override = cli.get<double>("latency-ms");
  const bool use_raw_q = cli.get<bool>("use-raw-q");
  const bool use_euler_q = cli.get<bool>("use-euler-q");
  const bool force_aim = cli.get<bool>("force-aim");
  const double speed = cli.get<double>("speed");
  const bool loop = cli.get<bool>("loop");
  const bool enable_web = cli.get<bool>("web");
  const auto web_port = static_cast<uint16_t>(cli.get<int>("port"));

  tools::SyncPlayer player(record_dir);
  if (!player.ok()) {
    tools::logger()->error("[ReplayStandard] 加载录制目录失败: {}", record_dir);
    return 1;
  }

  double latency_ms = 0.0;
  Eigen::Quaterniond q_gimbal2imu = Eigen::Quaterniond::Identity();
  std::array<double, 3> cam_in_gimbal{0.06, 0.0, 0.095};
  try {
    auto y = YAML::LoadFile(config_path);
    if (y["camera_latency_ms"]) latency_ms = y["camera_latency_ms"].as<double>();
    if (y["R_gimbal2imu"] && y["R_gimbal2imu"].size() == 9) {
      auto d = y["R_gimbal2imu"].as<std::vector<double>>();
      Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R(d.data());
      q_gimbal2imu = Eigen::Quaterniond(R).normalized();
    }
    if (y["t_camera2gimbal"] && y["t_camera2gimbal"].size() == 3) {
      cam_in_gimbal = {
        y["t_camera2gimbal"][0].as<double>(), y["t_camera2gimbal"][1].as<double>(),
        y["t_camera2gimbal"][2].as<double>()};
    }
  } catch (const std::exception & e) {
    tools::logger()->warn("[ReplayStandard] 读取 YAML 附加字段失败: {}", e.what());
  }

  if (latency_override > -9000.0) latency_ms = latency_override;
  player.set_latency_ms(latency_ms);
  if (use_raw_q) player.set_use_raw_q(true, q_gimbal2imu);
  if (use_euler_q) player.set_use_euler_q(true);

  const auto f_dt = player.frame_dt_stats();
  const auto i_dt = player.imu_dt_stats();
  tools::logger()->info(
    "[ReplayStandard] 开始离线回放: dir={} | config={} | frames={} | imu={} | latency={:.2f}ms",
    record_dir, config_path, player.frame_count(), player.imu_count(), latency_ms);
  if (f_dt.count > 0) {
    tools::logger()->info(
      "[ReplayStandard] 实采图像 dt: mean={:.2f}ms std={:.2f}ms P50={:.2f}ms P95={:.2f}ms "
      "P99={:.2f}ms max={:.2f}ms",
      f_dt.mean_ms, f_dt.std_ms, f_dt.p50_ms, f_dt.p95_ms, f_dt.p99_ms, f_dt.max_ms);
  }
  if (i_dt.count > 0) {
    tools::logger()->info(
      "[ReplayStandard] 实采 IMU dt: mean={:.2f}ms std={:.2f}ms P50={:.2f}ms P99={:.2f}ms "
      "max={:.2f}ms",
      i_dt.mean_ms, i_dt.std_ms, i_dt.p50_ms, i_dt.p99_ms, i_dt.max_ms);
  }

  tools::Exiter exiter;
  std::unique_ptr<tools::WebViewer> viewer;
  if (enable_web) viewer = std::make_unique<tools::WebViewer>(web_port, "web");

  auto_aim::YOLO yolo(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);

  cv::Mat img;
  std::chrono::steady_clock::time_point t{};
  std::chrono::steady_clock::time_point prev_t{};
  bool has_prev_t = false;

  uint64_t total_frames = 0;
  uint64_t det_frames = 0;
  uint64_t tracked_frames = 0;
  uint64_t fire_frames = 0;
  std::vector<double> reproj_errs_px;

  while (!exiter.exit()) {
    size_t frame_idx = 0;
    if (!player.read_frame(img, t, &frame_idx)) {
      if (loop) {
        player.reset();
        has_prev_t = false;
        continue;
      }
      break;
    }

    if (speed > 1e-6 && has_prev_t) {
      const double dt_s = tools::delta_time(t, prev_t);
      if (dt_s > 0.0 && dt_s < 1.0) {
        const auto wait_us = static_cast<int64_t>((dt_s / speed) * 1e6);
        if (wait_us > 0) std::this_thread::sleep_for(std::chrono::microseconds(wait_us));
      }
    }
    prev_t = t;
    has_prev_t = true;

    const auto q = player.q(t);
    const auto gs = player.state_at(t);
    solver.set_R_gimbal2world(q);

    const bool run_aim = force_aim || (gs.mode == 1);
    std::list<auto_aim::Armor> armors;
    std::list<auto_aim::Target> targets;
    auto_aim::Plan plan{};

    if (run_aim) {
      const Eigen::Vector3d self_vel(gs.vx, gs.vy, gs.vz);
      const cv::Rect focus_roi = tracker.focus_roi(img.size(), t, self_vel);
      armors = yolo.detect(img, focus_roi);
      if (!armors.empty()) det_frames++;
      targets = tracker.track(armors, t, true, self_vel);

      std::optional<auto_aim::Target> target_opt =
        targets.empty() ? std::nullopt : std::optional<auto_aim::Target>(targets.front());
      if (target_opt.has_value()) {
        tracked_frames++;
        auto_aim::SelfMotion self_motion;
        self_motion.vel = self_vel;
        self_motion.roll = static_cast<double>(gs.roll);
        const double bspeed = (gs.bullet_speed > 5.0f) ? gs.bullet_speed : 23.0;
        plan = planner.plan(target_opt, bspeed, Eigen::Vector2d(gs.yaw, gs.pitch), 0.0, self_motion);
        if (plan.fire) fire_frames++;

        // 统计当前帧检测角点与 EKF 重投影四角点的最小像素误差
        if (!armors.empty()) {
          const auto & tgt = target_opt.value();
          const auto xyza_list = tgt.armor_xyza_list();
          const auto & ekf_x = tgt.ekf_x();
          double best_err = std::numeric_limits<double>::max();
          for (const auto & a : armors) {
            if (a.points.size() != 4) continue;
            for (const auto & xyza : xyza_list) {
              const auto proj = solver.reproject_armor(
                xyza.head<3>(), xyza[3], tgt.armor_type, tgt.name, ekf_x[11], ekf_x[12]);
              if (proj.size() != 4) continue;
              double err = 0.0;
              for (int k = 0; k < 4; ++k) err += cv::norm(proj[k] - a.points[k]);
              err *= 0.25;
              if (err < best_err) best_err = err;
            }
          }
          if (best_err < 200.0) reproj_errs_px.push_back(best_err);
        }
      }
    }

    total_frames++;

    if (viewer) {
      tools::ViewerFrame vf;
      vf.stamp_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count());
      const auto ypr = tools::eulers(q, 2, 1, 0);
      vf.gimbal_yaw = ypr[0];
      vf.gimbal_pitch = ypr[1];
      vf.muzzle_yaw = gs.yaw;
      vf.muzzle_pitch = gs.pitch;
      vf.camera_in_gimbal = cam_in_gimbal;
      vf.control = plan.control;
      vf.shoot = plan.fire;
      vf.command_yaw = plan.yaw;
      vf.command_pitch = plan.pitch;
      vf.ref_yaw = plan.yaw;
      vf.ref_pitch = plan.pitch;

      for (const auto & a : armors) {
        if (a.points.size() == 4) {
          vf.detections.push_back(
            {a.points[0].x, a.points[0].y, a.points[1].x, a.points[1].y, a.points[2].x,
             a.points[2].y, a.points[3].x, a.points[3].y});
        }
      }
      if (!targets.empty()) {
        const auto & tgt = targets.front();
        const auto & x = tgt.ekf_x();
        vf.tracked = true;
        vf.target_name = auto_aim::ARMOR_NAMES[tgt.name];
        vf.target_center = {x[0], x[2], x[4]};
        vf.target_vel = {x[1], x[3], x[5]};
        vf.target_w = x[7];
        vf.target_is_big = (tgt.armor_type == auto_aim::ArmorType::big);
        vf.current_armor = tgt.last_id;
        for (const auto & xyza : tgt.armor_xyza_list()) {
          vf.armors.push_back({xyza[0], xyza[1], xyza[2], xyza[3], x[11], x[12]});
          const auto proj = solver.reproject_armor(
            xyza.head<3>(), xyza[3], tgt.armor_type, tgt.name, x[11], x[12]);
          if (proj.size() == 4) {
            vf.armor_pixels.push_back(
              {proj[0].x, proj[0].y, proj[1].x, proj[1].y, proj[2].x, proj[2].y, proj[3].x,
               proj[3].y});
          }
        }
      }
      viewer->publish(img, vf);
    }
  }

  double med_reproj = 0.0;
  double mean_reproj = 0.0;
  if (!reproj_errs_px.empty()) {
    for (double v : reproj_errs_px) mean_reproj += v;
    mean_reproj /= static_cast<double>(reproj_errs_px.size());
    std::sort(reproj_errs_px.begin(), reproj_errs_px.end());
    med_reproj = reproj_errs_px[reproj_errs_px.size() / 2];
  }

  const double det_ratio = total_frames > 0 ? 100.0 * det_frames / total_frames : 0.0;
  const double trk_ratio = total_frames > 0 ? 100.0 * tracked_frames / total_frames : 0.0;
  const double fire_ratio = tracked_frames > 0 ? 100.0 * fire_frames / tracked_frames : 0.0;
  tools::logger()->info(
    "[ReplayStandard] 回放汇总: frames={} | det={:.1f}% | tracked={:.1f}% | fire_duty={:.1f}% | "
    "reproj_med={:.2f}px reproj_mean={:.2f}px (n={})",
    total_frames, det_ratio, trk_ratio, fire_ratio, med_reproj, mean_reproj, reproj_errs_px.size());
  return 0;
}
