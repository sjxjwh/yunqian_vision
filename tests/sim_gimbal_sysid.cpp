// 端到端云台传递函数辨识采样工具（仿真共享内存通道）。
// 向云台注入安全幅值的阶跃 + 对数扫频（Chirp）指令，记录 (t, cmd_yaw, muzzle_yaw, cmd_pitch, muzzle_pitch)
// 到 CSV，配合 tools/fit_gimbal_tf.py 拟合纯延迟与一/二阶传递函数。
// 注意：与 sim_standard 共享同一块共享内存，运行前需先停止 sim_standard。

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "io/sim/sim_io.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace
{

constexpr double DEG2RAD = M_PI / 180.0;

// 对数扫频正弦：t in [0, T], 频率从 f0 指数扫到 f1
double log_chirp(double t, double T, double f0, double f1)
{
  const double k = std::log(f1 / f0) / T;
  const double phase = 2.0 * M_PI * f0 * (std::exp(k * t) - 1.0) / k;
  return std::sin(phase);
}

}  // namespace

int main(int argc, char * argv[])
{
  const std::string keys =
    "{help h usage ? |                              | 输出命令行参数说明}"
    "{out o          | /tmp/opencode/gimbal_sysid.csv | 输出 CSV 路径}"
    "{yaw-amp        | 8.0                          | Yaw 扫频幅值 (deg)}"
    "{pitch-amp      | 5.0                          | Pitch 扫频幅值 (deg)}"
    "{@config-path   | configs/sim.yaml             | yaml 配置文件路径}";

  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  const auto config_path = cli.get<std::string>(0);
  const auto out_path = cli.get<std::string>("out");
  const double yaw_amp = cli.get<double>("yaw-amp") * DEG2RAD;
  const double pitch_amp = cli.get<double>("pitch-amp") * DEG2RAD;

  std::ofstream csv(out_path, std::ios::out | std::ios::trunc);
  if (!csv.is_open()) {
    std::cerr << "无法打开输出 CSV: " << out_path << "\n";
    return 1;
  }
  csv << "t_sim_s,t_wall_s,phase,cmd_yaw_rad,muzzle_yaw_rad,cmd_pitch_rad,muzzle_pitch_rad\n";

  tools::Exiter exiter;
  io::SimCamera camera(config_path);
  io::SimGimbal gimbal(camera.client(), config_path);

  cv::Mat img;
  std::chrono::steady_clock::time_point ts;

  // 读两帧清掉旧缓存，再记录基准云台角，激励围绕 (yaw0, pitch0) 展开
  camera.read(img, ts);
  camera.read(img, ts);
  if (exiter.exit()) return 0;

  const Eigen::Vector3d ypr0 = tools::eulers(camera.muzzle_quaternion().toRotationMatrix(), 2, 1, 0);
  const double yaw0 = ypr0[0];
  const double pitch0 = std::clamp(ypr0[1], -0.25, 0.25);
  const uint64_t sim_t0_ns = camera.timestamp_ns();
  const auto wall_t0 = std::chrono::steady_clock::now();

  tools::logger()->info(
    "[SysID] 开始端到端云台辨识采样，基准 yaw={:.1f}° pitch={:.1f}°，输出 {}",
    yaw0 * 180.0 / M_PI, pitch0 * 180.0 / M_PI, out_path);

  constexpr double TOTAL_DURATION = 18.8;  // s
  int frames = 0;

  while (!exiter.exit()) {
    camera.read(img, ts);
    if (exiter.exit()) break;

    const double t_sim = static_cast<double>(camera.timestamp_ns() - sim_t0_ns) * 1e-9;
    const double t_wall = tools::delta_time(std::chrono::steady_clock::now(), wall_t0);
    if (t_sim > TOTAL_DURATION) break;

    const Eigen::Vector3d mz =
      tools::eulers(camera.muzzle_quaternion().toRotationMatrix(), 2, 1, 0);
    const double rel_mz_yaw = tools::limit_rad(mz[0] - yaw0);
    const double rel_mz_pitch = mz[1] - pitch0;

    double dyaw = 0.0;
    double dpitch = 0.0;
    const char * phase = "hold";

    if (t_sim < 0.8) {
      phase = "hold";
    } else if (t_sim < 1.8) {
      phase = "yaw_step_pos";
      dyaw = yaw_amp;
    } else if (t_sim < 2.8) {
      phase = "yaw_step_zero";
      dyaw = 0.0;
    } else if (t_sim < 10.8) {
      phase = "yaw_chirp";
      dyaw = yaw_amp * log_chirp(t_sim - 2.8, 8.0, 0.5, 8.0);
    } else if (t_sim < 11.8) {
      phase = "pitch_step_up";
      dpitch = -pitch_amp;
    } else if (t_sim < 12.8) {
      phase = "pitch_step_zero";
      dpitch = 0.0;
    } else {
      phase = "pitch_chirp";
      dpitch = pitch_amp * log_chirp(t_sim - 12.8, 6.0, 0.5, 6.0);
    }

    io::Command cmd;
    cmd.control = true;
    cmd.shoot = false;
    cmd.yaw = tools::limit_rad(yaw0 + dyaw);
    cmd.pitch = pitch0 + dpitch;
    gimbal.send(cmd, 2.0, camera.timestamp_ns(), 0.0, 0.0, 0.0, 0.0, Eigen::Vector2d(mz[0], mz[1]));

    csv << std::fixed << std::setprecision(6) << t_sim << ',' << t_wall << ',' << phase << ','
        << dyaw << ',' << rel_mz_yaw << ',' << dpitch << ',' << rel_mz_pitch << '\n';
    frames++;
  }

  // 结束时释放云台控制
  gimbal.send({false, false, 0, 0}, -1.0, camera.timestamp_ns());
  csv.flush();
  tools::logger()->info("[SysID] 采样完成，共 {} 帧，已写入 {}", frames, out_path);
  return 0;
}
