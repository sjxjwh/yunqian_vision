#include <fmt/core.h>

#include <Eigen/Geometry>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "tools/sync_recorder.hpp"

namespace
{
double quat_angle_err_rad(const Eigen::Quaterniond & a, const Eigen::Quaterniond & b)
{
  const double dot = std::clamp(std::abs(a.normalized().dot(b.normalized())), 0.0, 1.0);
  return 2.0 * std::acos(dot);
}
}  // namespace

int main()
{
  const std::string test_dir = "/tmp/opencode/sync_recorder_test_run";
  std::error_code ec;
  std::filesystem::remove_all(test_dir, ec);

  const auto t0 = std::chrono::steady_clock::now();
  const double yaw_rate = 2.0;    // rad/s
  const double pitch_rate = 0.5;  // rad/s
  const Eigen::Quaterniond q_g2i(
    Eigen::AngleAxisd(M_PI * 0.5, Eigen::Vector3d::UnitX()));  // 模拟 +90° 滚转安装角

  // 1. 合成 60 帧 ~100Hz 图像（含故意注入的 1 个 18ms 抖动脉冲）+ 650 包 1000Hz IMU
  {
    tools::SyncRecorder::Options opts;
    opts.output_dir = test_dir;
    opts.config_path = "configs/standard3.yaml";
    opts.fps_hint = 100.0;
    tools::SyncRecorder rec(opts);

    for (int i = 0; i < 650; ++i) {
      const double t_s = i * 0.001;
      const double yaw = yaw_rate * t_s;
      const double pitch = pitch_rate * t_s;
      const Eigen::Quaterniond q_true =
        (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()))
          .normalized();

      tools::SyncImuSample s;
      s.t = t0 + std::chrono::milliseconds(i);
      s.mode = 1;
      s.q = q_true;
      s.raw_q = (q_true * q_g2i.conjugate()).normalized();
      s.yaw = static_cast<float>(yaw);
      s.yaw_vel = static_cast<float>(yaw_rate);
      s.pitch = static_cast<float>(pitch);
      s.pitch_vel = static_cast<float>(pitch_rate);
      s.roll = 0.0f;
      s.bullet_speed = 23.5f;
      s.bullet_count = static_cast<uint16_t>(i / 100);
      s.vx = 1.2f;
      s.vy = -0.3f;
      s.vz = 0.0f;
      rec.record_imu(s);
    }

    int64_t cur_ms = 20;  // 从 20ms 开始，确保 latency 回退落在 IMU 覆盖区间内
    for (int f = 0; f < 60; ++f) {
      cv::Mat frame(120, 160, CV_8UC3, cv::Scalar(f * 3 % 255, 80, 160));
      cv::putText(
        frame, fmt::format("F{:02d}", f), {10, 60}, cv::FONT_HERSHEY_SIMPLEX, 1.0,
        cv::Scalar(255, 255, 255), 2);
      const auto tf = t0 + std::chrono::milliseconds(cur_ms);
      if (!rec.record_frame(frame, tf)) {
        fmt::print("[FAIL] record_frame dropped frame {}\n", f);
        return 1;
      }
      cur_ms += (f == 30) ? 18 : 10;  // 在第 30 帧注入一个 18ms 抖动
    }

    rec.stop();
    if (rec.written_frames() != 60 || rec.dropped_frames() != 0 || rec.written_imu() != 650) {
      fmt::print(
        "[FAIL] count mismatch: frames={} drop={} imu={}\n", rec.written_frames(),
        rec.dropped_frames(), rec.written_imu());
      return 1;
    }
  }

  // 2. 用 SyncPlayer 读回并校验时间戳、dt 抖动统计、SLERP 插值、延迟覆盖与 raw_q 还原
  tools::SyncPlayer player(test_dir);
  if (!player.ok() || player.frame_count() != 60 || player.imu_count() != 650) {
    fmt::print(
      "[FAIL] SyncPlayer load failed: ok={} frames={} imu={}\n", player.ok(), player.frame_count(),
      player.imu_count());
    return 1;
  }

  const auto f_st = player.frame_dt_stats();
  const auto i_st = player.imu_dt_stats();
  if (
    std::abs(f_st.p50_ms - 10.0) > 0.05 || std::abs(f_st.max_ms - 18.0) > 0.05 ||
    std::abs(i_st.mean_ms - 1.0) > 0.02) {
    fmt::print(
      "[FAIL] dt stats mismatch: frame p50={:.3f} max={:.3f} | imu mean={:.3f}\n", f_st.p50_ms,
      f_st.max_ms, i_st.mean_ms);
    return 1;
  }

  double max_err_0ms = 0.0;
  double max_err_8ms = 0.0;
  double max_err_raw = 0.0;
  cv::Mat img;
  std::chrono::steady_clock::time_point tf{};
  size_t idx = 0;
  int64_t expected_ms = 20;
  while (player.read_frame(img, tf, &idx)) {
    if (img.cols != 160 || img.rows != 120) {
      fmt::print("[FAIL] frame size mismatch at {}\n", idx);
      return 1;
    }

    // 0 ms 延迟查询
    const double t_s = expected_ms * 0.001;
    const Eigen::Quaterniond q_gt_0 =
      (Eigen::AngleAxisd(yaw_rate * t_s, Eigen::Vector3d::UnitZ()) *
       Eigen::AngleAxisd(pitch_rate * t_s, Eigen::Vector3d::UnitY()))
        .normalized();
    max_err_0ms = std::max(max_err_0ms, quat_angle_err_rad(player.q(tf, 0.0), q_gt_0));

    // 8.5 ms 非整数毫秒延迟查询（验证 SLERP 亚毫秒插值）
    const double t_s_8 = (expected_ms - 8.5) * 0.001;
    const Eigen::Quaterniond q_gt_8 =
      (Eigen::AngleAxisd(yaw_rate * t_s_8, Eigen::Vector3d::UnitZ()) *
       Eigen::AngleAxisd(pitch_rate * t_s_8, Eigen::Vector3d::UnitY()))
        .normalized();
    max_err_8ms = std::max(max_err_8ms, quat_angle_err_rad(player.q(tf, 8.5), q_gt_8));

    // 校验 state_at 底盘速度与弹速还原
    const auto st = player.state_at(tf, 0.0);
    if (std::abs(st.bullet_speed - 23.5f) > 1e-2f || std::abs(st.vx - 1.2f) > 1e-2f) {
      fmt::print("[FAIL] state_at mismatch at frame {}\n", idx);
      return 1;
    }

    expected_ms += (idx == 30) ? 18 : 10;
  }

  // 验证离线切换 raw_q * q_gimbal2imu 还原
  player.reset();
  player.set_use_raw_q(true, q_g2i);
  expected_ms = 20;
  while (player.read_frame(img, tf, &idx)) {
    const double t_s = (expected_ms - 5.0) * 0.001;
    const Eigen::Quaterniond q_gt =
      (Eigen::AngleAxisd(yaw_rate * t_s, Eigen::Vector3d::UnitZ()) *
       Eigen::AngleAxisd(pitch_rate * t_s, Eigen::Vector3d::UnitY()))
        .normalized();
    max_err_raw = std::max(max_err_raw, quat_angle_err_rad(player.q(tf, 5.0), q_gt));
    expected_ms += (idx == 30) ? 18 : 10;
  }

  std::filesystem::remove_all(test_dir, ec);

  fmt::print(
    "[PASS] sync_recorder_test: frames=60 imu=650 | frame_dt P50={:.2f}ms max={:.2f}ms | "
    "err_0ms={:.2e}rad err_8.5ms={:.2e}rad err_raw={:.2e}rad\n",
    f_st.p50_ms, f_st.max_ms, max_err_0ms, max_err_8ms, max_err_raw);

  if (max_err_0ms > 1e-3 || max_err_8ms > 1e-3 || max_err_raw > 1e-3) {
    fmt::print("[FAIL] quaternion interpolation error exceeded 1e-3 rad\n");
    return 1;
  }
  return 0;
}
