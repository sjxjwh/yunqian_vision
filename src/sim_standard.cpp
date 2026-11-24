#include <yaml-cpp/yaml.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>
#ifdef __linux__
#include <fstream>
#include <pthread.h>
#include <sched.h>
#endif

#include "io/sim/sim_io.hpp"
#include "tasks/auto_aim/aim_corrector.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/gimbal_csv_logger.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/runtime_params.hpp"
#include "tools/sim_metrics.hpp"
#include "tools/web_viewer.hpp"

// 仿真闭环入口。循环与 src/standard.cpp 相同，只把相机和下位机换成仿真共享内存。
// 不链接海康/迈德威视 SDK，也不依赖 ROS2，所以 NUC 与 Orin NX 编的是同一份代码。

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{mode m         |                        | 控制模式：aimer 或 planner（留空按 yaml use_planner）}"
  "{log l          |                        | 可选：导出逐帧评估指标 CSV 路径}"
  "{gimbal-log g   |                        | 可选：导出逐帧当前云台/下发指令 CSV 路径（实机串口符号约定）}"
  "{fire-hz f      | 20.0                   | 虚拟发弹评估射频 (Hz)，与仿真 cooldown=0.05s 对齐}"
  "{corrector      | false                  | 启用视觉弹道闭环预览（在画面叠加虚拟弹道/ROI/检测配对）}"
  "{profile p      | false                  | 每秒输出视觉/控制分项耗时与帧率抖动(avg/p95/max/std) }"
  "{@config-path   | configs/sim.yaml       | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  auto sim_yaml = YAML::LoadFile(config_path);
  auto bullet_speed = sim_yaml["bullet_speed"].as<double>();
  if (bullet_speed < 14) {
    tools::logger()->warn(
      "[Sim] bullet_speed {:.1f} 低于 14，aimer 会把它改成 23，请在配置里给一个真实弹速", bullet_speed);
  }
  bool use_planner = sim_yaml["use_planner"] ? sim_yaml["use_planner"].as<bool>() : false;
  auto mode_arg = cli.get<std::string>("mode");
  if (mode_arg == "planner") use_planner = true;
  else if (mode_arg == "aimer") use_planner = false;
  tools::logger()->info("[Sim] 控制模式: {}", use_planner ? "MPC Planner" : "Aimer + Shooter");

  const double air_k = sim_yaml["air_k"] ? sim_yaml["air_k"].as<double>() : 0.0;
  const bool use_corrector =
    cli.get<bool>("corrector") ||
    (sim_yaml["use_corrector"] && sim_yaml["use_corrector"].as<bool>());
  const bool enable_profile = cli.get<bool>("profile");
  const double fire_hz = cli.get<double>("fire-hz");
  const auto fire_period =
    std::chrono::microseconds(static_cast<int64_t>(1e6 / std::max(1.0, fire_hz)));
  auto sim_tp = [](uint64_t ns) {
    return auto_aim::AimCorrector::Clock::time_point(std::chrono::nanoseconds(ns));
  };
  tools::Exiter exiter;

  const bool use_trt =
#ifdef SP_WITH_TENSORRT
    sim_yaml["trt_engine_path"] && !sim_yaml["trt_engine_path"].as<std::string>().empty();
#else
    false;
#endif

  // 若进程分配了充足的 CPU 核，自动做三级核隔离以消除线程抢占：
  //   OpenVINO CPU 模式：P 核全给推理，E 核给后台相机退化，独占 1 核给 100 Hz MPC 规划
  //   TensorRT GPU 模式：GPU 跑推理，2 个 P 核给主线程前/后处理，其余 P 核给相机退化，独占 1 核给 MPC 规划
  std::vector<int> infer_cpus;
  std::vector<int> capture_cpus;
  int plan_cpu = -1;
#ifdef __linux__
  {
    cpu_set_t avail;
    CPU_ZERO(&avail);
    if (sched_getaffinity(0, sizeof(avail), &avail) == 0) {
      std::vector<int> cpus;
      for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &avail)) cpus.push_back(c);
      }
      if (cpus.size() >= 8) {
        std::vector<std::pair<int, long>> cpu_freq;
        long max_f = 0, min_f = std::numeric_limits<long>::max();
        for (int c : cpus) {
          long f = 0;
          std::ifstream ifs(
            "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/cpufreq/cpuinfo_max_freq");
          if (ifs >> f && f > 0) {
            max_f = std::max(max_f, f);
            min_f = std::min(min_f, f);
          }
          cpu_freq.push_back({c, f});
        }
        if (max_f > 0 && min_f < max_f * 85 / 100) {
          const long thresh = (max_f + min_f) / 2;
          std::vector<int> p_cpus, e_cpus;
          for (const auto & [c, f] : cpu_freq) {
            if (f >= thresh) p_cpus.push_back(c);
            else e_cpus.push_back(c);
          }
          if (use_trt && p_cpus.size() >= 6) {
            // TensorRT 模式下主线程轻量，把主力 P 核给 SimCameraModel 压到 ~3.5 ms
            infer_cpus.assign(p_cpus.begin(), p_cpus.begin() + 2);
            capture_cpus.assign(p_cpus.begin() + 2, p_cpus.end());
            plan_cpu = !e_cpus.empty() ? e_cpus.back() : p_cpus.back();
            if (e_cpus.empty()) capture_cpus.pop_back();
          } else if (p_cpus.size() >= 4 && e_cpus.size() >= 4) {
            infer_cpus = p_cpus;
            plan_cpu = e_cpus.back();
            capture_cpus.assign(e_cpus.begin(), e_cpus.end() - 1);
          }
        }
        if (infer_cpus.empty()) {
          plan_cpu = cpus.back();
          const size_t n_infer = use_trt ? 2 : std::min<size_t>(8, cpus.size() / 2);
          infer_cpus.assign(cpus.begin(), cpus.begin() + n_infer);
          capture_cpus.assign(cpus.begin() + n_infer, cpus.end() - 1);
        }
        tools::logger()->info(
          "[Sim] 启用核隔离({}): infer=[{}..{}]({}核) capture=[{}..{}]({}核) plan=[{}]",
          use_trt ? "TRT-GPU" : "OV-CPU", infer_cpus.front(), infer_cpus.back(), infer_cpus.size(),
          capture_cpus.front(), capture_cpus.back(), capture_cpus.size(), plan_cpu);
      }
    }
  }
#endif

  io::SimCamera camera(config_path, capture_cpus);
  io::SimGimbal gimbal(camera.client(), config_path);
  tools::WebViewer viewer;
  tools::SimMetrics metrics(fire_hz, air_k);
  tools::SimMetrics gt_metrics(fire_hz, air_k);
  std::atomic<bool> reset_metrics_req{false};
  const uint8_t enemy_team =
    (sim_yaml["enemy_color"] && sim_yaml["enemy_color"].as<std::string>() == "red") ? 0 : 1;
  Eigen::Vector2d last_gt_dxy = Eigen::Vector2d::Zero();
  auto build_gt_plates = [&](
                           const std::vector<io::talos::GroundTruthRobot> & gt_robots,
                           const Eigen::Vector3d & odom_pos,
                           const std::vector<tools::SimArmorPlate> & ekf_plates) {
    // 计算当前 EKF 跟踪目标的 XY 中心；仅取与当前跟踪目标对应的那一台敌方机器人，
    // 避免 SimMetrics 将多台敌方机器人的装甲板混算成一个超大机身遮挡圆柱。
    Eigen::Vector2d ekf_xy = Eigen::Vector2d::Zero();
    const bool has_ekf = !ekf_plates.empty();
    if (has_ekf) {
      for (const auto & p : ekf_plates) ekf_xy += p.center.head<2>();
      ekf_xy /= static_cast<double>(ekf_plates.size());
    }

    const io::talos::GroundTruthRobot * best = nullptr;
    double best_d = 1e9;
    for (const auto & r : gt_robots) {
      if (r.team != enemy_team) continue;
      const Eigen::Vector2d rel_xy = (r.pos_world - odom_pos).head<2>();
      const double d = has_ekf ? (rel_xy - ekf_xy).norm() : rel_xy.norm();
      if (d < best_d) {
        best_d = d;
        best = &r;
      }
    }

    std::vector<tools::SimArmorPlate> out;
    if (!best) return out;
    const auto & r = *best;
    const Eigen::Vector3d rel_center = r.pos_world - odom_pos;
    // 若与当前 EKF 目标匹配且仿真未发布子底盘旋转角，将 EKF 装甲环刚性平移到仿真真值 XY 中心，
    // 保留真实深度/横向偏差惩罚，同时对齐真实旋转相位与装甲高度。
    if (has_ekf && std::abs(r.vyaw) < 1e-3 && best_d < 0.6) {
      const Eigen::Vector2d dxy = rel_center.head<2>() - ekf_xy;
      last_gt_dxy = dxy;
      out.reserve(ekf_plates.size());
      for (auto p : ekf_plates) {
        p.center.x() += dxy.x();
        p.center.y() += dxy.y();
        out.push_back(p);
      }
      return out;
    }

    const bool is_big = (r.armor_label == 1);
    const double radius = is_big ? 0.255 : 0.243;
    const double plate_w = is_big ? 0.230 : 0.135;
    const double plate_h = is_big ? 0.127 : 0.125;
    const double mount_pitch = r.is_outpost ? (-15.0 * M_PI / 180.0) : (15.0 * M_PI / 180.0);
    const int n_plates = r.is_outpost ? 3 : 4;
    out.reserve(n_plates);
    for (int i = 0; i < n_plates; ++i) {
      const double theta_out = tools::limit_rad(r.yaw + i * (2.0 * M_PI / n_plates));
      tools::SimArmorPlate p;
      p.center =
        rel_center + Eigen::Vector3d(radius * std::cos(theta_out), radius * std::sin(theta_out), 0.0);
      p.yaw = tools::limit_rad(theta_out + M_PI);
      p.mount_pitch = mount_pitch;
      p.width = plate_w;
      p.height = plate_h;
      out.push_back(p);
    }
    return out;
  };
  auto log_path = cli.get<std::string>("log");
  if (log_path == "true") {
    tools::logger()->warn("[Sim] 检测到 --log 未带 '='，请使用 --log=path.csv 指定路径");
    log_path.clear();
  }
  if (!log_path.empty()) {
    if (metrics.open_csv(log_path)) {
      tools::logger()->info("[Sim] 评估指标写入 {}", log_path);
    } else {
      tools::logger()->warn("[Sim] 无法打开指标 CSV：{}", log_path);
    }
  }
  tools::GimbalCsvLogger gimbal_log;
  auto gimbal_log_path = cli.get<std::string>("gimbal-log");
  if (gimbal_log_path == "true") {
    tools::logger()->warn(
      "[Sim] 检测到 --gimbal-log 未带 '='，请使用 --gimbal-log=path.csv 指定路径");
    gimbal_log_path.clear();
  }
  if (!gimbal_log_path.empty()) {
    if (gimbal_log.open(gimbal_log_path)) {
      tools::logger()->info("[Sim] 云台/下发指令写入 {}", gimbal_log_path);
    } else {
      tools::logger()->warn("[Sim] 无法打开云台 CSV：{}", gimbal_log_path);
    }
  }

#ifdef __linux__
  if (!infer_cpus.empty()) {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int cpu : infer_cpus) CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  }
#endif
  auto_aim::YOLO detector(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);
  std::unique_ptr<auto_aim::Planner> planner;
  if (use_planner) planner = std::make_unique<auto_aim::Planner>(config_path);
  std::unique_ptr<auto_aim::AimCorrector> corrector;
  if (use_corrector) {
    corrector = std::make_unique<auto_aim::AimCorrector>(config_path);
    tools::logger()->info("[Sim] 已启用 AimCorrector 弹道闭环预览");
  }
  viewer.set_api_handler([&](const std::string & path) -> std::string {
    if (path == "api/metrics/reset") {
      reset_metrics_req.store(true, std::memory_order_release);
      tools::logger()->info("[Sim] 已清零累计弹丸发射量与命中率统计");
      return nlohmann::json{{"ok", true}, {"msg", "已清除累计弹丸统计"}}.dump();
    }
    if (corrector) {
      if (path == "api/corrector/apply") {
        std::string msg;
        const bool ok = corrector->apply_suggestion(msg);
        return nlohmann::json{{"ok", ok}, {"msg", msg}}.dump();
      }
      if (path == "api/corrector/clear") {
        corrector->clear();
        return nlohmann::json{{"ok", true}, {"msg", "已清空弹道闭环样本"}}.dump();
      }
      if (path.rfind("api/corrector/auto", 0) == 0) {
        const bool on = (path.find("v=1") != std::string::npos);
        tools::RuntimeParams::instance().set("corrector.auto_apply", on ? 1.0 : 0.0);
        return nlohmann::json{
          {"ok", true}, {"auto_apply", on}, {"msg", on ? "自动闭环已开启" : "自动闭环已关闭"}}
          .dump();
      }
    }
    return nlohmann::json{{"ok", false}, {"msg", "unknown api"}}.dump();
  });
  auto last_corr_shot = std::chrono::steady_clock::time_point{};
  struct PendingShot
  {
    auto_aim::AimCorrector::Clock::time_point t_fire;
    double stop_range;
  };
  std::mutex pending_shots_mtx;
  std::vector<PendingShot> pending_shots;

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  auto last_report = std::chrono::steady_clock::now();
  std::atomic<int> vis_frames{0};
  std::atomic<int> plan_frames{0};
  struct ProfileWindow
  {
    std::mutex mtx;
    std::vector<double> pace_ms, wait_ms, recv_ms, model_ms, detect_ms, track_ms, vis_dt_ms;
    std::vector<double> plan_dt_ms, plan_exec_ms;

    static std::string fmt_stat(const std::vector<double> & v)
    {
      if (v.empty()) return "-";
      auto s = v;
      std::sort(s.begin(), s.end());
      double sum = 0.0;
      for (double x : s) sum += x;
      const double mean = sum / static_cast<double>(s.size());
      double var = 0.0;
      for (double x : s) var += (x - mean) * (x - mean);
      const double sd = std::sqrt(var / static_cast<double>(s.size()));
      const double p95 = s[std::min(s.size() - 1, static_cast<size_t>(s.size() * 0.95))];
      const double mx = s.back();
      char buf[96];
      std::snprintf(buf, sizeof(buf), "%.2f/%.2f/%.2f(sd=%.2f)", mean, p95, mx, sd);
      return std::string(buf);
    }

    void dump_and_clear()
    {
      std::lock_guard<std::mutex> lk(mtx);
      tools::logger()->info(
        "[Profile] vis_dt={}ms | pace={} wait={} recv={} model={} detect={} track={} | "
        "plan_dt={}ms plan_exec={}ms",
        fmt_stat(vis_dt_ms), fmt_stat(pace_ms), fmt_stat(wait_ms), fmt_stat(recv_ms),
        fmt_stat(model_ms), fmt_stat(detect_ms), fmt_stat(track_ms), fmt_stat(plan_dt_ms),
        fmt_stat(plan_exec_ms));
      pace_ms.clear();
      wait_ms.clear();
      recv_ms.clear();
      model_ms.clear();
      detect_ms.clear();
      track_ms.clear();
      vis_dt_ms.clear();
      plan_dt_ms.clear();
      plan_exec_ms.clear();
    }
  };
  ProfileWindow prof;
  Eigen::Vector3d last_gimbal_ypr{0, 0, 0};
  Eigen::Vector3d last_muzzle_ypr{0, 0, 0};
  uint64_t last_rate_ts_ns = 0;

  // 100 Hz 独立 MPC 规划与发指令线程（与 src/standard_mpc.cpp 同构，消除视觉帧保持滞后）
  struct SharedVisionState
  {
    std::mutex mtx;
    bool ready = false;
    uint64_t frame_gen = 0;
    uint64_t frame_seq = 0;
    uint64_t sim_ts_ns = 0;
    std::chrono::steady_clock::time_point wall_ts{};
    Eigen::Vector3d gimbal_ypr{0, 0, 0};
    Eigen::Vector3d muzzle_ypr{0, 0, 0};
    Eigen::Quaterniond gimbal_q = Eigen::Quaterniond::Identity();
    Eigen::Quaterniond muzzle_q = Eigen::Quaterniond::Identity();
    double gimbal_yaw_vel = 0.0;
    double gimbal_pitch_vel = 0.0;
    double muzzle_yaw_vel = 0.0;
    double muzzle_pitch_vel = 0.0;
    Eigen::Matrix3d R_muzzle2world = Eigen::Matrix3d::Identity();
    Eigen::Vector3d muzzle_origin{0, 0, 0};
    Eigen::Vector3d muzzle_in_gimbal{0, 0, 0};
    Eigen::Vector3d self_vel{0, 0, 0};
    Eigen::Vector3d chassis_rpy{0, 0, 0};
    std::optional<auto_aim::Target> target;
    tools::ViewerFrame view_base;
    cv::Mat img;
    size_t armor_count = 0;
  };

  SharedVisionState shared;
  std::atomic<bool> quit{false};
  std::thread plan_thread;

  if (use_planner) {
    plan_thread = std::thread([&]() {
#ifdef __linux__
      if (plan_cpu >= 0 && plan_cpu < CPU_SETSIZE) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(plan_cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
      }
#endif
      uint64_t last_pub_gen = 0;
      uint64_t last_logged_gen = 0;
      uint64_t last_est_sim_ts_ns = 0;
      uint64_t last_live_seq = 0;
      auto next_tick = std::chrono::steady_clock::now();
      auto last_plan_report = next_tick;
      auto last_plan_tick = std::chrono::steady_clock::time_point{};

      while (!quit.load(std::memory_order_relaxed)) {
        const auto tick_start = std::chrono::steady_clock::now();
        if (enable_profile && last_plan_tick.time_since_epoch().count() > 0) {
          const double dt_ms =
            std::chrono::duration<double, std::milli>(tick_start - last_plan_tick).count();
          std::lock_guard<std::mutex> lk(prof.mtx);
          prof.plan_dt_ms.push_back(dt_ms);
        }
        last_plan_tick = tick_start;
        next_tick += std::chrono::milliseconds(10);  // 100 Hz

        bool has_data = false;
        uint64_t gen = 0;
        uint64_t frame_seq = 0;
        uint64_t sim_ts_ns = 0;
        std::chrono::steady_clock::time_point wall_ts;
        Eigen::Vector3d gimbal_ypr;
        Eigen::Vector3d muzzle_ypr;
        Eigen::Quaterniond gimbal_q = Eigen::Quaterniond::Identity();
        Eigen::Quaterniond muzzle_q = Eigen::Quaterniond::Identity();
        double gimbal_yaw_vel = 0.0;
        double gimbal_pitch_vel = 0.0;
        double muzzle_yaw_vel = 0.0;
        double muzzle_pitch_vel = 0.0;
        Eigen::Matrix3d R_muzzle2world = Eigen::Matrix3d::Identity();
        Eigen::Vector3d muzzle_origin;
        Eigen::Vector3d muzzle_in_gimbal{0, 0, 0};
        Eigen::Vector3d self_vel{0, 0, 0};
        Eigen::Vector3d chassis_rpy{0, 0, 0};
        std::optional<auto_aim::Target> opt_target;
        tools::ViewerFrame view;
        cv::Mat pub_img;
        size_t armor_count = 0;

        {
          std::lock_guard<std::mutex> lk(shared.mtx);
          if (shared.ready) {
            has_data = true;
            gen = shared.frame_gen;
            frame_seq = shared.frame_seq;
            sim_ts_ns = shared.sim_ts_ns;
            wall_ts = shared.wall_ts;
            gimbal_ypr = shared.gimbal_ypr;
            muzzle_ypr = shared.muzzle_ypr;
            gimbal_q = shared.gimbal_q;
            muzzle_q = shared.muzzle_q;
            gimbal_yaw_vel = shared.gimbal_yaw_vel;
            gimbal_pitch_vel = shared.gimbal_pitch_vel;
            muzzle_yaw_vel = shared.muzzle_yaw_vel;
            muzzle_pitch_vel = shared.muzzle_pitch_vel;
            R_muzzle2world = shared.R_muzzle2world;
            muzzle_origin = shared.muzzle_origin;
            muzzle_in_gimbal = shared.muzzle_in_gimbal;
            self_vel = shared.self_vel;
            chassis_rpy = shared.chassis_rpy;
            opt_target = shared.target;
            view = shared.view_base;
            armor_count = shared.armor_count;
            if (gen != last_pub_gen) {
              pub_img = shared.img;
              last_pub_gen = gen;
            }
          }
        }

        if (has_data) {
          const auto now = std::chrono::steady_clock::now();
          // 100 Hz 控制线程直接从共享内存窥视最新已提交的炮口姿态（跳过 write_idx 防止脏读，不外推避免拐点过冲）
          Eigen::Quaterniond live_muzzle_q;
          uint64_t live_seq = 0;
          if (camera.client().peek_gimbal_quat(live_muzzle_q, &live_seq)) {
            if (live_seq >= last_live_seq) {
              muzzle_q = live_muzzle_q;
              R_muzzle2world = live_muzzle_q.toRotationMatrix();
              muzzle_ypr = tools::eulers(R_muzzle2world, 2, 1, 0);
              last_live_seq = live_seq;
            }
          }
          const double frame_age_s = std::clamp(tools::delta_time(now, wall_ts), 0.0, 0.25);
          const uint64_t raw_est_ts_ns =
            sim_ts_ns + static_cast<uint64_t>(frame_age_s * 1e9);
          const uint64_t est_sim_ts_ns =
            (raw_est_ts_ns > last_est_sim_ts_ns) ? raw_est_ts_ns : (last_est_sim_ts_ns + 1000000ULL);
          last_est_sim_ts_ns = est_sim_ts_ns;

          const bool sim_following = camera.client().peek_sim_following();
          // 仿真开启自瞄跟随时（sim_following=true），直接使用仿真器真实炮口 muzzle_ypr 做开火判据与曲线展示，
          // 消除 SimGimbalModel 虚拟姿态与仿真真实炮口之间的相位/开火 gap；未开启跟随时回退 500 Hz 虚拟模型预览。
          Eigen::Vector2d fire_yp(muzzle_ypr[0], muzzle_ypr[1]);
          if (!sim_following) {
            if (auto myp = gimbal.model_yp(); myp.has_value()) {
              fire_yp = *myp;
            }
          }
          auto_aim::SelfMotion self_motion;
          self_motion.vel = self_vel;
          self_motion.roll = gimbal_ypr[2];
          self_motion.pitch = chassis_rpy[1];
          self_motion.muzzle_in_gimbal = muzzle_in_gimbal;
          auto plan = planner->plan(opt_target, bullet_speed, fire_yp, frame_age_s, self_motion);

          io::Command command{plan.control, plan.fire, plan.yaw, plan.pitch};
          double distance = 0.0;
          double target_bearing = 0.0;
          std::vector<tools::SimArmorPlate> sim_plates;
          if (opt_target.has_value()) {
            auto pred_tgt = *opt_target;
            pred_tgt.predict(now, self_vel);
            auto state = pred_tgt.ekf_x();
            distance = std::hypot(state[0], state[2]);
            target_bearing = std::atan2(state[2], state[0]) * 180.0 / M_PI;

            const bool is_big = (pred_tgt.armor_type == auto_aim::ArmorType::big);
            const double plate_w = is_big ? 0.230 : 0.135;
            const double plate_h = is_big ? 0.127 : 0.125;
            const double mount_pitch =
              (pred_tgt.name == auto_aim::ArmorName::outpost) ? (-15.0 * M_PI / 180.0)
                                                              : (15.0 * M_PI / 180.0);
            for (const auto & a : pred_tgt.armor_xyzayp_list()) {
              tools::SimArmorPlate p;
              p.center = a.head<3>();
              p.yaw = a[3];
              p.ring_roll = a[4];
              p.ring_pitch = a[5];
              p.mount_pitch = mount_pitch;
              p.width = plate_w;
              p.height = plate_h;
              sim_plates.push_back(p);
            }
          }

          const uint64_t wall_now_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
          gimbal.send(
            command, distance, wall_now_ns, plan.yaw_vel, plan.yaw_acc, plan.pitch_vel,
            plan.pitch_acc, Eigen::Vector2d(muzzle_ypr[0], muzzle_ypr[1]));

          if (gimbal_log.is_open()) {
            const bool new_meas = (gen != last_logged_gen);
            last_logged_gen = gen;
            tools::GimbalLogSample sample;
            sample.t_sec = static_cast<double>(est_sim_ts_ns) * 1e-9;
            sample.sim_ts_ns = est_sim_ts_ns;
            sample.frame_seq = frame_seq;
            sample.new_meas = new_meas;
            sample.control = command.control;
            sample.fire = command.shoot;
            sample.distance_m = distance;
            sample.bullet_speed = bullet_speed;
            // 仿真中受控轴是炮口系 muzzle（水平为 0°，与 tx_yaw/tx_pitch 严格同零点）
            sample.gimbal_q = muzzle_q;
            sample.gimbal_yaw = muzzle_ypr[0];
            sample.gimbal_pitch = muzzle_ypr[1];
            sample.gimbal_yaw_vel = muzzle_yaw_vel;
            sample.gimbal_pitch_vel = muzzle_pitch_vel;
            sample.muzzle_yaw = muzzle_ypr[0];
            sample.muzzle_pitch = muzzle_ypr[1];
            sample.muzzle_yaw_vel = muzzle_yaw_vel;
            sample.muzzle_pitch_vel = muzzle_pitch_vel;
            sample.cmd_yaw = plan.yaw;
            sample.cmd_pitch = plan.pitch;
            sample.cmd_yaw_vel = plan.yaw_vel;
            sample.cmd_yaw_acc = plan.yaw_acc;
            sample.cmd_pitch_vel = plan.pitch_vel;
            sample.cmd_pitch_acc = plan.pitch_acc;
            sample.ref_yaw = plan.ref_yaw;
            sample.ref_pitch = plan.ref_pitch;
            sample.target_yaw = plan.target_yaw;
            sample.target_pitch = plan.target_pitch;
            gimbal_log.write(sample);
          }

          if (corrector && command.shoot) {
            const auto sim_now = sim_tp(est_sim_ts_ns);
            if (sim_now < last_corr_shot || (sim_now - last_corr_shot >= fire_period)) {
              const auto t_fire = sim_now + std::chrono::microseconds(
                                              static_cast<int64_t>(corrector->fire_delay() * 1e6));
              const double stop_r = (distance > 0.8) ? std::max(0.6, distance - 0.15) : 25.0;
              std::lock_guard<std::mutex> lk(pending_shots_mtx);
              pending_shots.push_back({t_fire, stop_r});
              last_corr_shot = sim_now;
            }
          }

          const double t_sec = static_cast<double>(est_sim_ts_ns) * 1e-9;
          if (reset_metrics_req.exchange(false, std::memory_order_acq_rel)) {
            metrics.reset();
            gt_metrics.reset();
          }
          Eigen::Vector3d eval_ypr = muzzle_ypr;
          if (!sim_following) {
            if (auto myp = gimbal.model_yp(); myp.has_value()) {
              eval_ypr[0] = (*myp)[0];
              eval_ypr[1] = (*myp)[1];
              eval_ypr[2] = 0.0;
            }
          }
          const double recon_ref_yaw = sim_following ? plan.ref_yaw : plan.target_yaw;
          const double recon_ref_pitch = sim_following ? plan.ref_pitch : plan.target_pitch;
          auto msnap = metrics.update(
            t_sec, eval_ypr, command.control, recon_ref_yaw, recon_ref_pitch, command.shoot,
            bullet_speed, sim_plates, muzzle_origin, self_vel);

          std::vector<io::talos::GroundTruthRobot> gt_robots;
          Eigen::Vector3d live_odom = camera.gimbal_in_world();
          camera.client().peek_odom_pos(live_odom);
          const bool has_gt = camera.client().peek_ground_truth(gt_robots);
          tools::SimMetricsSnapshot gsnap{};
          if (has_gt) {
            auto gt_plates = build_gt_plates(gt_robots, live_odom, sim_plates);
            gsnap = gt_metrics.update(
              t_sec, muzzle_ypr, command.control && sim_following, plan.ref_yaw, plan.ref_pitch,
              command.shoot && sim_following, bullet_speed, gt_plates, muzzle_origin, self_vel);
          }

          view.stamp_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
          view.muzzle_yaw = eval_ypr[0];
          view.muzzle_pitch = eval_ypr[1];
          view.command_yaw = command.yaw;
          view.command_pitch = command.pitch;
          view.ref_yaw = recon_ref_yaw;
          view.ref_pitch = recon_ref_pitch;
          view.control = command.control;
          view.shoot = command.shoot;
          view.aim_valid = plan.control;
          if (plan.control) {
            view.aim_x = planner->debug_xyza[0];
            view.aim_y = planner->debug_xyza[1];
            view.aim_z = planner->debug_xyza[2];
          }
          view.shots = msnap.shots;
          view.hits = msnap.hits;
          view.resolved_shots = msnap.resolved;
          view.hit_rate = msnap.hit_rate;
          view.fire_duty = msnap.fire_duty;
          view.yaw_err_rms = msnap.yaw_err_rms;
          view.has_sim_metrics = has_gt;
          if (has_gt) {
            view.sim_shots = gsnap.shots;
            view.sim_hits = gsnap.hits;
            view.sim_resolved_shots = gsnap.resolved;
            view.sim_hit_rate = gsnap.hit_rate;
          }
          view.bullets = msnap.bullets;
          view.hit_events.clear();
          view.hit_events.reserve(msnap.recent_hits.size());
          for (const auto & h : msnap.recent_hits) {
            view.hit_events.push_back({h.id, h.x, h.y, h.z, h.armor_id});
          }
          if (corrector) view.corrector_json = corrector->status().dump();
          viewer.publish(pub_img, view);
          plan_frames++;

          if (now - last_plan_report >= std::chrono::seconds(1)) {
            int vf = vis_frames.exchange(0);
            int pf = plan_frames.exchange(0);
            tools::logger()->info(
              "[Sim] 视觉 {} FPS / 规划 {} Hz，检出装甲 {}，跟踪目标 {}，control={} "
              "cmd_yaw={:.1f}° cmd_pitch={:.1f}° muzzle_yaw={:.1f}° muzzle_pitch={:.1f}° "
              "dist={:.3f}m bearing={:.1f}° w={:.2f} r={:.3f} id={} "
              "recon_hit={:.1f}%({}/{}) sim_hit={:.1f}%({}/{}) "
              "fire_duty={:.1f}% yaw_rms={:.2f}° gt_dxy=({:+.3f},{:+.3f})m v_self={:.2f}m/s roll={:.1f}°",
              vf, pf, armor_count, opt_target.has_value() ? 1 : 0, command.control,
              command.yaw * 180 / M_PI, command.pitch * 180 / M_PI, muzzle_ypr[0] * 180 / M_PI,
              muzzle_ypr[1] * 180 / M_PI, distance, target_bearing,
              opt_target.has_value() ? opt_target->ekf_x()[7] : 0.0,
              opt_target.has_value() ? opt_target->ekf_x()[8] : 0.0,
              opt_target.has_value() ? opt_target->last_id : -1, msnap.hit_rate * 100.0, msnap.hits,
              msnap.resolved, gsnap.hit_rate * 100.0, gsnap.hits, gsnap.resolved,
              msnap.fire_duty * 100.0, msnap.yaw_err_rms * 180 / M_PI, last_gt_dxy.x(),
              last_gt_dxy.y(), self_vel.norm(), gimbal_ypr[2] * 180 / M_PI);
            last_plan_report = now;
            if (enable_profile) prof.dump_and_clear();
          }
        }

        const auto now_after = std::chrono::steady_clock::now();
        if (enable_profile) {
          const double exec_ms =
            std::chrono::duration<double, std::milli>(now_after - tick_start).count();
          std::lock_guard<std::mutex> lk(prof.mtx);
          prof.plan_exec_ms.push_back(exec_ms);
        }
        if (next_tick > now_after) {
          // 混合定时：提前 500us 结束 sleep_until，末段短自旋消除 Linux CFS 唤醒抖动
          constexpr auto kSpinWindow = std::chrono::microseconds(500);
          if (next_tick - now_after > kSpinWindow) {
            std::this_thread::sleep_until(next_tick - kSpinWindow);
          }
          while (std::chrono::steady_clock::now() < next_tick) {
            std::this_thread::yield();
          }
        } else {
          next_tick = now_after;
        }
      }
    });
  }

  auto last_vis_tp = std::chrono::steady_clock::time_point{};
  while (!exiter.exit()) {
    camera.read(img, timestamp);
    if (exiter.exit()) break;
    const auto t_after_read = std::chrono::steady_clock::now();
    auto q = camera.gimbal_quaternion();

    solver.set_R_gimbal2world(q);
    const auto img_sim_tp = sim_tp(camera.timestamp_ns());
    Eigen::Vector3d gimbal_ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
    Eigen::Matrix3d R_muzzle2world = camera.muzzle_quaternion().toRotationMatrix();
    Eigen::Vector3d muzzle_ypr = tools::eulers(R_muzzle2world, 2, 1, 0);
    Eigen::Vector3d muzzle_origin = solver.R_gimbal2world() * camera.muzzle_in_gimbal();

    double gimbal_yaw_vel = 0.0;
    double gimbal_pitch_vel = 0.0;
    double muzzle_yaw_vel = 0.0;
    double muzzle_pitch_vel = 0.0;
    if (last_rate_ts_ns != 0 && camera.timestamp_ns() > last_rate_ts_ns) {
      const double dt = static_cast<double>(camera.timestamp_ns() - last_rate_ts_ns) * 1e-9;
      if (dt > 1e-4 && dt < 0.2) {
        gimbal_yaw_vel = tools::limit_rad(gimbal_ypr[0] - last_gimbal_ypr[0]) / dt;
        gimbal_pitch_vel = (gimbal_ypr[1] - last_gimbal_ypr[1]) / dt;
        muzzle_yaw_vel = tools::limit_rad(muzzle_ypr[0] - last_muzzle_ypr[0]) / dt;
        muzzle_pitch_vel = (muzzle_ypr[1] - last_muzzle_ypr[1]) / dt;
      }
    }
    last_gimbal_ypr = gimbal_ypr;
    last_muzzle_ypr = muzzle_ypr;
    last_rate_ts_ns = camera.timestamp_ns();

    if (corrector) {
      std::vector<PendingShot> due;
      {
        std::lock_guard<std::mutex> lk(pending_shots_mtx);
        for (auto it = pending_shots.begin(); it != pending_shots.end();) {
          if (img_sim_tp >= it->t_fire) {
            due.push_back(*it);
            it = pending_shots.erase(it);
          } else {
            ++it;
          }
        }
      }
      for (const auto & ps : due) {
        // 用真实出膛时刻的炮口姿态与出膛坐标生成虚拟弹丸
        corrector->add_shot(ps.t_fire, R_muzzle2world, bullet_speed, air_k, muzzle_origin, ps.stop_range);
      }
      corrector->update(img, img_sim_tp, solver);
    }

    const auto t_det0 = std::chrono::steady_clock::now();
    const Eigen::Vector3d self_vel = camera.chassis_vel_world();
    const Eigen::Vector3d chassis_rpy = camera.chassis_rpy();
    const cv::Rect focus_roi = tracker.focus_roi(img.size(), timestamp, self_vel);
    auto armors = detector.detect(img, focus_roi);
    const auto t_det1 = std::chrono::steady_clock::now();
    auto targets = tracker.track(armors, timestamp, true, self_vel);
    const auto t_trk1 = std::chrono::steady_clock::now();

    if (enable_profile) {
      using Ms = std::chrono::duration<double, std::milli>;
      const auto rt = camera.last_read_timing();
      std::lock_guard<std::mutex> lk(prof.mtx);
      if (last_vis_tp.time_since_epoch().count() > 0) {
        prof.vis_dt_ms.push_back(Ms(t_after_read - last_vis_tp).count());
      }
      last_vis_tp = t_after_read;
      prof.pace_ms.push_back(rt.pace_ms);
      prof.wait_ms.push_back(rt.wait_ms);
      prof.recv_ms.push_back(rt.recv_ms);
      prof.model_ms.push_back(rt.model_ms);
      prof.detect_ms.push_back(Ms(t_det1 - t_det0).count());
      prof.track_ms.push_back(Ms(t_trk1 - t_det1).count());
    }

    const bool need_view = viewer.has_clients();
    cv::Mat pub_frame;
    if (need_view) {
      const bool has_focus_roi = (focus_roi.width < img.cols || focus_roi.height < img.rows);
      pub_frame = img.clone();
      if (has_focus_roi) {
        cv::rectangle(pub_frame, focus_roi, cv::Scalar(220, 220, 40), 1, cv::LINE_AA);
      }
      if (corrector) {
        const Eigen::Vector3d dir = R_muzzle2world.col(0);
        const double m_yaw = std::atan2(dir.y(), dir.x());
        const double m_pitch = std::atan2(dir.z(), std::hypot(dir.x(), dir.y()));
        auto_aim::AimCorrector::draw_ballistic(
          pub_frame, solver, m_yaw, m_pitch, bullet_speed, air_k, 0.8, {0, 220, 0}, 1,
          corrector->muzzle_z_offset());
        corrector->draw(pub_frame, solver, img_sim_tp);
      }
    }

    tools::ViewerFrame view;
    if (need_view) {
      view.stamp_ns = camera.timestamp_ns();
      view.gimbal_yaw = gimbal_ypr[0];
      view.gimbal_pitch = gimbal_ypr[1];
      view.muzzle_yaw = muzzle_ypr[0];
      view.muzzle_pitch = muzzle_ypr[1];
      {
        const Eigen::Vector3d self_w = camera.gimbal_in_world();
        view.self_pos = {self_w.x(), self_w.y(), self_w.z()};
        const Eigen::Vector3d cam_g = camera.camera_in_gimbal();
        if (cam_g.norm() > 1e-4) view.camera_in_gimbal = {cam_g.x(), cam_g.y(), cam_g.z()};
        const Eigen::Vector3d muz_g = camera.muzzle_in_gimbal();
        if (muz_g.norm() > 1e-4) view.muzzle_in_gimbal = {muz_g.x(), muz_g.y(), muz_g.z()};
      }
      for (const auto & armor : armors) {
        if (armor.points.size() != 4) continue;
        std::array<double, 8> points{};
        for (int i = 0; i < 4; ++i) {
          points[i * 2] = armor.points[i].x;
          points[i * 2 + 1] = armor.points[i].y;
        }
        view.detections.push_back(points);
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
        for (auto & a : target.armor_xyzayp_list()) {
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
    }

    if (use_planner) {
      {
        std::lock_guard<std::mutex> lk(shared.mtx);
        shared.ready = true;
        shared.frame_gen++;
        shared.frame_seq = camera.frame_seq();
        shared.sim_ts_ns = camera.timestamp_ns();
        shared.wall_ts = timestamp;
        shared.gimbal_ypr = gimbal_ypr;
        shared.muzzle_ypr = muzzle_ypr;
        shared.gimbal_q = q;
        shared.muzzle_q = camera.muzzle_quaternion();
        shared.gimbal_yaw_vel = gimbal_yaw_vel;
        shared.gimbal_pitch_vel = gimbal_pitch_vel;
        shared.muzzle_yaw_vel = muzzle_yaw_vel;
        shared.muzzle_pitch_vel = muzzle_pitch_vel;
        shared.R_muzzle2world = R_muzzle2world;
        shared.muzzle_origin = muzzle_origin;
        shared.muzzle_in_gimbal = camera.muzzle_in_gimbal();
        shared.self_vel = self_vel;
        shared.chassis_rpy = chassis_rpy;
        shared.target = targets.empty() ? std::nullopt : std::optional<auto_aim::Target>(targets.front());
        shared.view_base = std::move(view);
        shared.img = pub_frame;
        shared.armor_count = armors.size();
      }
      vis_frames++;
      continue;
    }

    // Aimer + Shooter 单线程路径
    io::Command command = aimer.aim(targets, timestamp, bullet_speed);
    command.shoot = shooter.shoot(command, aimer, targets, muzzle_ypr);

    double distance = 0;
    double target_bearing = 0;
    std::vector<tools::SimArmorPlate> sim_plates;
    if (!targets.empty()) {
      const auto & target = targets.front();
      auto state = target.ekf_x();
      distance = std::hypot(state[0], state[2]);
      target_bearing = std::atan2(state[2], state[0]) * 180 / M_PI;
      const bool is_big = (target.armor_type == auto_aim::ArmorType::big);
      const double plate_w = is_big ? 0.230 : 0.135;
      const double plate_h = is_big ? 0.127 : 0.125;
      const double mount_pitch =
        (target.name == auto_aim::ArmorName::outpost) ? (-15.0 * M_PI / 180.0)
                                                      : (15.0 * M_PI / 180.0);
      for (const auto & a : target.armor_xyzayp_list()) {
        tools::SimArmorPlate p;
        p.center = a.head<3>();
        p.yaw = a[3];
        p.ring_roll = a[4];
        p.ring_pitch = a[5];
        p.mount_pitch = mount_pitch;
        p.width = plate_w;
        p.height = plate_h;
        sim_plates.push_back(p);
      }
    }
    gimbal.send(command, distance, camera.timestamp_ns());

    if (gimbal_log.is_open()) {
      tools::GimbalLogSample sample;
      sample.t_sec = static_cast<double>(camera.timestamp_ns()) * 1e-9;
      sample.sim_ts_ns = camera.timestamp_ns();
      sample.frame_seq = camera.frame_seq();
      sample.new_meas = true;
      sample.control = command.control;
      sample.fire = command.shoot;
      sample.distance_m = distance;
      sample.bullet_speed = bullet_speed;
      sample.gimbal_q = camera.muzzle_quaternion();
      sample.gimbal_yaw = muzzle_ypr[0];
      sample.gimbal_pitch = muzzle_ypr[1];
      sample.gimbal_yaw_vel = muzzle_yaw_vel;
      sample.gimbal_pitch_vel = muzzle_pitch_vel;
      sample.muzzle_yaw = muzzle_ypr[0];
      sample.muzzle_pitch = muzzle_ypr[1];
      sample.muzzle_yaw_vel = muzzle_yaw_vel;
      sample.muzzle_pitch_vel = muzzle_pitch_vel;
      sample.cmd_yaw = command.yaw;
      sample.cmd_pitch = command.pitch;
      sample.ref_yaw = command.yaw;
      sample.ref_pitch = command.pitch;
      sample.target_yaw = command.yaw;
      sample.target_pitch = command.pitch;
      gimbal_log.write(sample);
    }

    if (corrector && command.shoot) {
      if (img_sim_tp < last_corr_shot || (img_sim_tp - last_corr_shot >= fire_period)) {
        const auto t_fire = img_sim_tp + std::chrono::microseconds(
                                           static_cast<int64_t>(corrector->fire_delay() * 1e6));
        const double stop_r = (distance > 0.8) ? std::max(0.6, distance - 0.15) : 25.0;
        corrector->add_shot(t_fire, R_muzzle2world, bullet_speed, air_k, muzzle_origin, stop_r);
        last_corr_shot = img_sim_tp;
      }
    }

    const double t_sec = static_cast<double>(camera.timestamp_ns()) * 1e-9;
    if (reset_metrics_req.exchange(false, std::memory_order_acq_rel)) {
      metrics.reset();
      gt_metrics.reset();
    }
    auto msnap = metrics.update(
      t_sec, muzzle_ypr, command.control, command.yaw, command.pitch, command.shoot, bullet_speed,
      sim_plates, muzzle_origin, self_vel);

    std::vector<io::talos::GroundTruthRobot> gt_robots;
    Eigen::Vector3d live_odom = camera.gimbal_in_world();
    camera.client().peek_odom_pos(live_odom);
    const bool has_gt = camera.client().peek_ground_truth(gt_robots);
    const bool sim_following = camera.client().peek_sim_following();
    tools::SimMetricsSnapshot gsnap{};
    if (has_gt) {
      auto gt_plates = build_gt_plates(gt_robots, live_odom, sim_plates);
      gsnap = gt_metrics.update(
        t_sec, muzzle_ypr, command.control && sim_following, command.yaw, command.pitch,
        command.shoot && sim_following, bullet_speed, gt_plates, muzzle_origin, self_vel);
    }

    view.command_yaw = command.yaw;
    view.command_pitch = command.pitch;
    view.ref_yaw = command.yaw;
    view.ref_pitch = command.pitch;
    view.control = command.control;
    view.shoot = command.shoot;
    view.aim_valid = aimer.debug_aim_point.valid;
    if (view.aim_valid) {
      view.aim_x = aimer.debug_aim_point.xyza[0];
      view.aim_y = aimer.debug_aim_point.xyza[1];
      view.aim_z = aimer.debug_aim_point.xyza[2];
    }
    view.shots = msnap.shots;
    view.hits = msnap.hits;
    view.resolved_shots = msnap.resolved;
    view.hit_rate = msnap.hit_rate;
    view.fire_duty = msnap.fire_duty;
    view.yaw_err_rms = msnap.yaw_err_rms;
    view.has_sim_metrics = has_gt;
    if (has_gt) {
      view.sim_shots = gsnap.shots;
      view.sim_hits = gsnap.hits;
      view.sim_resolved_shots = gsnap.resolved;
      view.sim_hit_rate = gsnap.hit_rate;
    }
    view.bullets = msnap.bullets;
    view.hit_events.clear();
    view.hit_events.reserve(msnap.recent_hits.size());
    for (const auto & h : msnap.recent_hits) {
      view.hit_events.push_back({h.id, h.x, h.y, h.z, h.armor_id});
    }
    if (corrector) view.corrector_json = corrector->status().dump();
    viewer.publish(pub_frame, view);
    vis_frames++;

    auto now = std::chrono::steady_clock::now();
    if (now - last_report > std::chrono::seconds(1)) {
      int vf = vis_frames.exchange(0);
      tools::logger()->info(
        "[Sim] {} 帧/秒，检出装甲 {}，跟踪目标 {}，control={} cmd_yaw={:.1f}° cmd_pitch={:.1f}° "
        "muzzle_yaw={:.1f}° muzzle_pitch={:.1f}° dist={:.3f}m bearing={:.1f}° "
        "w={:.2f} r={:.3f} id={} hit_rate={:.1f}%({}/{}) yaw_rms={:.2f}°",
        vf, armors.size(), targets.size(), command.control, command.yaw * 180 / M_PI,
        command.pitch * 180 / M_PI, muzzle_ypr[0] * 180 / M_PI, muzzle_ypr[1] * 180 / M_PI,
        distance, target_bearing, targets.empty() ? 0.0 : targets.front().ekf_x()[7],
        targets.empty() ? 0.0 : targets.front().ekf_x()[8],
        targets.empty() ? -1 : targets.front().last_id, msnap.hit_rate * 100.0, msnap.hits,
        msnap.resolved, msnap.yaw_err_rms * 180 / M_PI);
      last_report = now;
    }
  }

  quit = true;
  if (plan_thread.joinable()) plan_thread.join();
  return 0;
}
