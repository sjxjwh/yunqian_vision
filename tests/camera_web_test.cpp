#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Geometry>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/web_viewer.hpp"

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明 }"
  "{port p         | 8090                   | WebViewer 监听端口 }"
  "{@config-path   | configs/standard3.yaml | yaml 配置文件路径 }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  const auto config_path = cli.get<std::string>(0);
  const auto port = static_cast<uint16_t>(cli.get<int>("port"));

  tools::Exiter exiter;
  io::Camera camera(config_path);
  auto_aim::YOLO yolo(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);
  tools::WebViewer viewer(port, "web");

  // 无损原始帧抓拍状态：通过 HTTP /api/snap?n=1&prefix=shot 触发，保存未经叠加与压缩的 PNG
  std::mutex snap_mtx;
  int snap_remaining = 0;
  int snap_saved_total = 0;
  std::string snap_dir = "calib_captures";
  std::string snap_prefix = "shot";
  int snap_batch_idx = 0;
  viewer.set_api_handler([&](const std::string & path) -> std::string {
    if (path.rfind("api/snap", 0) == 0) {
      int n = 1;
      std::string prefix = "shot";
      const auto qpos = path.find('?');
      if (qpos != std::string::npos) {
        const std::string query = path.substr(qpos + 1);
        auto get_param = [&](const std::string & key) -> std::string {
          const std::string needle = key + "=";
          const auto p = query.find(needle);
          if (p == std::string::npos) return {};
          const auto v0 = p + needle.size();
          const auto v1 = query.find('&', v0);
          return query.substr(v0, v1 == std::string::npos ? std::string::npos : v1 - v0);
        };
        if (auto sn = get_param("n"); !sn.empty()) {
          n = std::clamp(std::stoi(sn), 1, 100);
        }
        if (auto sp = get_param("prefix"); !sp.empty()) {
          prefix = sp;
        }
      }
      std::filesystem::create_directories(snap_dir);
      std::lock_guard<std::mutex> lk(snap_mtx);
      snap_prefix = prefix;
      snap_batch_idx = 0;
      snap_remaining = n;
      return fmt::format(
        "{{\"ok\":true,\"queued\":{},\"dir\":\"{}\",\"prefix\":\"{}\"}}", n, snap_dir, prefix);
    }
    if (path == "api/snap_status") {
      std::lock_guard<std::mutex> lk(snap_mtx);
      return fmt::format(
        "{{\"ok\":true,\"remaining\":{},\"saved_total\":{}}}", snap_remaining, snap_saved_total);
    }
    return "{\"ok\":false,\"msg\":\"unknown api\"}";
  });

  // 从 yaml 读取相机外参供 3D 重建显示
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

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  const auto t_start = std::chrono::steady_clock::now();
  auto last_report = t_start;
  int frames = 0;
  double cur_fps = 0.0;

  while (!exiter.exit()) {
    camera.read(img, timestamp);
    if (img.empty()) continue;

    {
      std::string out_path;
      {
        std::lock_guard<std::mutex> lk(snap_mtx);
        if (snap_remaining > 0) {
          out_path = fmt::format("{}/{}_{:04d}.png", snap_dir, snap_prefix, ++snap_batch_idx);
          --snap_remaining;
        }
      }
      if (!out_path.empty()) {
        cv::imwrite(out_path, img, {cv::IMWRITE_PNG_COMPRESSION, 1});
        std::lock_guard<std::mutex> lk(snap_mtx);
        ++snap_saved_total;
        tools::logger()->info("[CameraWeb] 无损保存 {}", out_path);
      }
    }

    // 无下位机时云台姿态取单位四元数
    solver.set_R_gimbal2world(Eigen::Quaterniond::Identity());

    const cv::Rect focus_roi = tracker.focus_roi(img.size(), timestamp);
    auto armors = yolo.detect(img, focus_roi);
    // 拷贝一份用于跟踪，保留全量 YOLO 检出（含所有颜色）以便画标注观察识别效果
    auto armors_for_track = armors;
    auto targets = tracker.track(armors_for_track, timestamp);

    tools::ViewerFrame view;
    view.stamp_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(timestamp - t_start).count());
    view.camera_in_gimbal = cam_in_gimbal;

    cv::Mat vis = img.clone();
    if (focus_roi.width < img.cols || focus_roi.height < img.rows) {
      cv::rectangle(vis, focus_roi, cv::Scalar(220, 220, 40), 1, cv::LINE_AA);
    }
    for (const auto & armor : armors) {
      if (armor.points.size() != 4) continue;
      std::array<double, 8> pts{};
      for (int i = 0; i < 4; ++i) {
        pts[i * 2] = armor.points[i].x;
        pts[i * 2 + 1] = armor.points[i].y;
      }
      view.detections.push_back(pts);

      const cv::Scalar col = (armor.color == auto_aim::Color::red)
                               ? cv::Scalar(80, 90, 255)
                               : (armor.color == auto_aim::Color::blue ? cv::Scalar(255, 170, 60)
                                                                       : cv::Scalar(180, 180, 180));
      const std::string tag = fmt::format(
        "{} {} {:.2f}", auto_aim::COLORS[armor.color], auto_aim::ARMOR_NAMES[armor.name],
        armor.confidence);
      cv::putText(
        vis, tag, armor.points[0] + cv::Point2f(0, -6), cv::FONT_HERSHEY_SIMPLEX, 0.55, col, 2,
        cv::LINE_AA);
    }

    if (!targets.empty()) {
      auto & target = targets.front();
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

      // 无下位机时用标称 22 m/s 跑 MPC 预测，直观验证提前量与规划角
      auto plan = planner.plan(std::optional<auto_aim::Target>(target), 22.0);
      view.control = plan.control;
      view.shoot = plan.fire;
      view.command_yaw = plan.yaw;
      view.command_pitch = plan.pitch;
      view.ref_yaw = plan.target_yaw;
      view.ref_pitch = plan.target_pitch;
      if (plan.control) {
        const double dist = std::hypot(ekf_x[0], ekf_x[2]);
        const double cp = std::cos(plan.pitch);
        const Eigen::Vector3d aim_pt(
          dist * cp * std::cos(plan.yaw), dist * cp * std::sin(plan.yaw),
          -dist * std::sin(plan.pitch));
        view.aim_valid = true;
        view.aim_x = aim_pt.x();
        view.aim_y = aim_pt.y();
        view.aim_z = aim_pt.z();
        cv::Point2f uv;
        if (solver.world2pixel(aim_pt, uv)) {
          cv::drawMarker(
            vis, uv, cv::Scalar(0, 255, 255), cv::MARKER_CROSS, 22, 2, cv::LINE_AA);
        }
      }
    }

    frames++;
    const auto now = std::chrono::steady_clock::now();
    if (now - last_report >= std::chrono::seconds(1)) {
      const double sec = std::chrono::duration<double>(now - last_report).count();
      cur_fps = frames / sec;
      tools::logger()->info(
        "[CameraWeb] {:.1f} fps | 检出 {} | 跟踪 {}", cur_fps, armors.size(), targets.size());
      frames = 0;
      last_report = now;
    }

    cv::putText(
      vis, fmt::format("{:.1f} FPS | det:{}", cur_fps, armors.size()), cv::Point(18, vis.rows - 18),
      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(160, 230, 180), 2, cv::LINE_AA);

    viewer.publish(vis, view);
  }

  return 0;
}
