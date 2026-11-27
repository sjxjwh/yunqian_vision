#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/dm_imu/dm_imu.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/multithread/commandgener.hpp"
#include "tasks/auto_aim/multithread/mt_detector.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tasks/auto_buff/buff_type.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/sync_recorder.hpp"
#include "tools/web_viewer.hpp"

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明 }"
  "{no-fire        | false                  | 禁止发弹（强制 fire=false，仅跟随目标） }"
  "{force-aim      | false                  | 忽略电控 mode，强制进入自瞄跟随模式 }"
  "{web            | false                  | 开启 WebViewer 实时画面与跟踪曲线 }"
  "{port p         | 8090                   | WebViewer 监听端口 }"
  "{record         | false                  | 开启传统视频录制 }"
  "{sync-record    |                        | 开启同步录制(图像+1000Hz IMU)，填 auto 或目录路径 }"
  "{@config-path   | configs/standard3.yaml | yaml配置文件路径 }";

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  const auto config_path = cli.get<std::string>("@config-path");
  const bool no_fire = cli.get<bool>("no-fire");
  const bool force_aim = cli.get<bool>("force-aim");
  const bool enable_web = cli.get<bool>("web");
  const auto web_port = static_cast<uint16_t>(cli.get<int>("port"));
  const bool enable_record = cli.get<bool>("record");
  const auto sync_record_dir = cli.get<std::string>("sync-record");

  tools::Exiter exiter;
  tools::Plotter plotter;
  std::unique_ptr<tools::Recorder> recorder;
  if (enable_record) recorder = std::make_unique<tools::Recorder>();

  std::unique_ptr<tools::SyncRecorder> sync_recorder;
  if (!sync_record_dir.empty()) {
    tools::SyncRecorder::Options opts;
    opts.output_dir = sync_record_dir;
    opts.config_path = config_path;
    opts.fps_hint = 100.0;
    sync_recorder = std::make_unique<tools::SyncRecorder>(opts);
  }

  std::unique_ptr<tools::WebViewer> viewer;
  if (enable_web) viewer = std::make_unique<tools::WebViewer>(web_port, "web");

  std::array<double, 3> cam_in_gimbal{0.06, 0.0, 0.095};
  try {
    auto y = YAML::LoadFile(config_path);
    if (y["t_camera2gimbal"] && y["t_camera2gimbal"].size() == 3) {
      cam_in_gimbal = {
        y["t_camera2gimbal"][0].as<double>(), y["t_camera2gimbal"][1].as<double>(),
        y["t_camera2gimbal"][2].as<double>()};
    }
  } catch (const std::exception &) {
  }

  tools::logger()->info(
    "[StandardMPC] 启动: config={} | no_fire={} | force_aim={} | web={}:{} | record={}",
    config_path, no_fire, force_aim, enable_web, web_port, enable_record);

  io::Gimbal gimbal(config_path);
  if (sync_recorder) {
    gimbal.set_sample_callback(
      [&sync_recorder](const tools::SyncImuSample & s) { sync_recorder->record_imu(s); });
  }
  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  auto_buff::Buff_Detector buff_detector(config_path);
  auto_buff::Solver buff_solver(config_path);
  auto_buff::SmallTarget buff_small_target;
  auto_buff::BigTarget buff_big_target;
  auto_buff::Aimer buff_aimer(config_path);

  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  const auto t_start = std::chrono::steady_clock::now();

  std::atomic<bool> quit = false;
  std::atomic<io::GimbalMode> mode{force_aim ? io::GimbalMode::AUTO_AIM : io::GimbalMode::IDLE};
  auto last_mode{io::GimbalMode::IDLE};

  std::mutex plan_mtx;
  auto_aim::Plan latest_plan{};

  auto plan_thread = std::thread([&]() {
    auto next_tick = std::chrono::steady_clock::now();
    while (!quit) {
      const auto cur_mode = mode.load();
      const bool run_auto_aim =
        (cur_mode == io::GimbalMode::AUTO_AIM || cur_mode == io::GimbalMode::IDLE);
      if (!target_queue.empty() && run_auto_aim) {
        const auto now_tick = std::chrono::steady_clock::now();
        if (now_tick > next_tick + 20ms) next_tick = now_tick;
        next_tick += 10ms;

        auto target = target_queue.front();
        auto gs = gimbal.state();
        auto_aim::SelfMotion self_motion;
        self_motion.vel = Eigen::Vector3d(gs.vx, gs.vy, gs.vz);
        self_motion.roll = static_cast<double>(gs.roll);
        auto plan =
          planner.plan(target, gs.bullet_speed, Eigen::Vector2d(gs.yaw, gs.pitch), 0.0, self_motion);
        if (no_fire || cur_mode != io::GimbalMode::AUTO_AIM) plan.fire = false;

        {
          std::lock_guard<std::mutex> lk(plan_mtx);
          latest_plan = plan;
        }

        // IDLE 时仅本地识别/预测与 WebViewer 预览，不下发控制指令（电控不跟随）
        if (cur_mode == io::GimbalMode::AUTO_AIM) {
          gimbal.send(
            plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch,
            plan.pitch_vel, plan.pitch_acc);
        } else {
          gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
        }

        std::this_thread::sleep_until(next_tick);
      } else {
        std::this_thread::sleep_for(50ms);
        next_tick = std::chrono::steady_clock::now();
      }
    }
  });

  auto last_report = t_start;
  int frames = 0;
  double cur_fps = 0.0;

  while (!exiter.exit()) {
    mode = force_aim ? io::GimbalMode::AUTO_AIM : gimbal.mode();

    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", gimbal.str(mode));
      last_mode = mode.load();
    }

    camera.read(img, t);
    if (img.empty()) continue;

    auto q = gimbal.q(t);
    auto gs = gimbal.state();
    if (recorder) recorder->record(img, q, t);
    if (sync_recorder) sync_recorder->record_frame(img, t);
    solver.set_R_gimbal2world(q);

    std::list<auto_aim::Armor> armors_vis;
    std::list<auto_aim::Target> targets_vis;
    cv::Rect focus_roi(0, 0, img.cols, img.rows);

    /// 自瞄（IDLE 下也持续识别与 EKF 预测，仅在 plan_thread 中拦截控制输出）
    if (mode.load() == io::GimbalMode::AUTO_AIM || mode.load() == io::GimbalMode::IDLE) {
      const Eigen::Vector3d self_vel(gs.vx, gs.vy, gs.vz);
      focus_roi = tracker.focus_roi(img.size(), t, self_vel);
      auto armors = yolo.detect(img, focus_roi);
      if (viewer) armors_vis = armors;
      auto targets = tracker.track(armors, t, true, self_vel);
      if (viewer) targets_vis = targets;
      if (!targets.empty())
        target_queue.push(targets.front());
      else
        target_queue.push(std::nullopt);
    }

    /// 打符
    else if (mode.load() == io::GimbalMode::SMALL_BUFF || mode.load() == io::GimbalMode::BIG_BUFF) {
      buff_solver.set_R_gimbal2world(q);

      auto power_runes = buff_detector.detect(img);

      buff_solver.solve(power_runes);

      auto_aim::Plan buff_plan;
      if (mode.load() == io::GimbalMode::SMALL_BUFF) {
        buff_small_target.get_target(power_runes, t);
        auto target_copy = buff_small_target;
        buff_plan = buff_aimer.mpc_aim(target_copy, t, gs, true);
      } else if (mode.load() == io::GimbalMode::BIG_BUFF) {
        buff_big_target.get_target(power_runes, t);
        auto target_copy = buff_big_target;
        buff_plan = buff_aimer.mpc_aim(target_copy, t, gs, true);
      }
      if (no_fire) buff_plan.fire = false;
      gimbal.send(
        buff_plan.control, buff_plan.fire, buff_plan.yaw, buff_plan.yaw_vel, buff_plan.yaw_acc,
        buff_plan.pitch, buff_plan.pitch_vel, buff_plan.pitch_acc);

    } else {
      target_queue.push(std::nullopt);
      gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
    }

    ++frames;
    const auto now = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double>(now - last_report).count();
    if (dt >= 1.0) {
      cur_fps = frames / dt;
      auto_aim::Plan lp{};
      {
        std::lock_guard<std::mutex> lk(plan_mtx);
        lp = latest_plan;
      }
      std::string trk_info;
      if (!targets_vis.empty()) {
        const auto & tg = targets_vis.front();
        const auto ex = tg.ekf_x();
        trk_info = fmt::format(
          " | w={:+.2f} r={:.3f} l={:+.3f} nis={:.1f}", ex[7], ex[8], ex[9], tg.ekf().last_nis);
      }
      tools::logger()->info(
        "[StandardMPC] {:.1f} fps | mode={} | det={} trk={} | gimbal=({:+.1f}°,{:+.1f}°) | "
        "plan=({:+.1f}°,{:+.1f}°) ctrl={} fire={}{}",
        cur_fps, gimbal.str(mode), armors_vis.size(), targets_vis.size(),
        gs.yaw * 180.0 / M_PI, -gs.pitch * 180.0 / M_PI, lp.yaw * 180.0 / M_PI,
        -lp.pitch * 180.0 / M_PI, lp.control, lp.fire, trk_info);
      frames = 0;
      last_report = now;
    }

    if (viewer) {
      tools::ViewerFrame view;
      view.stamp_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t - t_start).count());
      view.camera_in_gimbal = cam_in_gimbal;
      view.gimbal_yaw = gs.yaw;
      view.gimbal_pitch = gs.pitch;
      view.muzzle_yaw = gs.yaw;
      view.muzzle_pitch = gs.pitch;

      auto_aim::Plan lp{};
      {
        std::lock_guard<std::mutex> lk(plan_mtx);
        lp = latest_plan;
      }
      view.command_yaw = lp.yaw;
      view.command_pitch = lp.pitch;
      view.ref_yaw = lp.ref_yaw;
      view.ref_pitch = lp.ref_pitch;
      view.control = lp.control;
      view.shoot = lp.fire;

      cv::Mat vis = img.clone();
      if (focus_roi.width < img.cols || focus_roi.height < img.rows) {
        cv::rectangle(vis, focus_roi, cv::Scalar(220, 220, 40), 1, cv::LINE_AA);
      }
      for (const auto & armor : armors_vis) {
        if (armor.points.size() != 4) continue;
        std::array<double, 8> pts{};
        for (int i = 0; i < 4; ++i) {
          pts[i * 2] = armor.points[i].x;
          pts[i * 2 + 1] = armor.points[i].y;
        }
        view.detections.push_back(pts);
      }

      if (!targets_vis.empty()) {
        auto & target = targets_vis.front();
        const auto ekf_x = target.ekf_x();
        view.tracked = true;
        view.target_name = auto_aim::ARMOR_NAMES[target.name];
        view.target_is_big = (target.armor_type == auto_aim::ArmorType::big);
        view.target_center = {ekf_x[0], ekf_x[2], ekf_x[4]};
        view.target_vel = {ekf_x[1], ekf_x[3], ekf_x[5]};
        view.target_w = ekf_x[7];
        view.current_armor = target.last_id;
        for (const auto & a : target.armor_xyzayp_list()) {
          view.armors.push_back({a[0], a[1], a[2], a[3], a[4], a[5]});
          auto projected = solver.reproject_armor(
            a.head(3), a[3], target.armor_type, target.name, a[4], a[5]);
          std::array<double, 8> pixels{};
          for (int i = 0; i < 4 && i < static_cast<int>(projected.size()); ++i) {
            pixels[i * 2] = projected[i].x;
            pixels[i * 2 + 1] = projected[i].y;
          }
          view.armor_pixels.push_back(pixels);
        }
      }

      cv::putText(
        vis,
        fmt::format(
          "{:.1f} FPS | {} | det:{} | Y:{:+.1f} P:{:+.1f}", cur_fps, gimbal.str(mode),
          armors_vis.size(), gs.yaw * 180.0 / M_PI, -gs.pitch * 180.0 / M_PI),
        cv::Point(18, vis.rows - 18), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(160, 230, 180), 2,
        cv::LINE_AA);
      viewer->publish(vis, view);
    }
  }

  quit = true;
  if (plan_thread.joinable()) plan_thread.join();
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

  return 0;
}