// 实车云台端到端传递函数与惯量极限辨识采样工具。
// 支持串口 (io::Gimbal) 与 CAN (io::CBoard) 双通道，不依赖相机 SDK，可在 NUC / Orin NX / 本地 PC 直接编译。
// 安全特性：
//   1. 强制输入并校验 Yaw/Pitch 机械限位角度，自动预留安全余量并硬 clamp 所有输出指令；
//   2. 永不开火 (fire=false)；
//   3. 串口模式下默认等待遥控器进入 AUTO_AIM 才开始激励，中途退出 AUTO_AIM 或按 Ctrl+C 立即释放云台控制；
//   4. 扫频首尾加平滑窗，避免机械冲击。

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include <opencv2/opencv.hpp>

#include "io/cboard.hpp"
#include "io/command.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace
{

constexpr double DEG2RAD = M_PI / 180.0;
constexpr double RAD2DEG = 180.0 / M_PI;

// 对数扫频正弦及其解析一阶/二阶导数（给电控串级前馈下发一致的 pos / vel / acc）
void log_chirp_pva(
  double t, double T, double f0, double f1, double & p, double & v, double & a)
{
  const double k = std::log(f1 / f0) / T;
  const double w = 2.0 * M_PI * f0 * std::exp(k * t);  // 瞬时角频率 rad/s
  const double dw = k * w;
  const double phase = (w - 2.0 * M_PI * f0) / k;
  const double s = std::sin(phase);
  const double c = std::cos(phase);
  p = s;
  v = w * c;
  a = dw * c - w * w * s;
}

// 首尾各 fade_s 秒余弦缓启停窗及其一阶导数
void taper_window_pv(double t, double T, double & w, double & dw, double fade_s = 0.25)
{
  if (t < fade_s) {
    w = 0.5 * (1.0 - std::cos(M_PI * t / fade_s));
    dw = 0.5 * (M_PI / fade_s) * std::sin(M_PI * t / fade_s);
  } else if (t > T - fade_s) {
    const double u = T - t;
    w = 0.5 * (1.0 - std::cos(M_PI * u / fade_s));
    dw = -0.5 * (M_PI / fade_s) * std::sin(M_PI * u / fade_s);
  } else {
    w = 1.0;
    dw = 0.0;
  }
}

}  // namespace

int main(int argc, char * argv[])
{
  const std::string keys =
    "{help h usage ?   |                        | 输出命令行参数说明}"
    "{bus b            | gimbal                 | 通信通道：gimbal (串口) 或 cboard (CAN)}"
    "{yaw-min          | -30.0                  | Yaw 机械/安全下限 (deg，360°云台可设 -180)}"
    "{yaw-max          | 30.0                   | Yaw 机械/安全上限 (deg，360°云台可设 180)}"
    "{pitch-min        | -15.0                  | Pitch 机械/安全下限 (deg，必填安全值)}"
    "{pitch-max        | 15.0                   | Pitch 机械/安全上限 (deg，必填安全值)}"
    "{margin           | 2.0                    | 距上下限额外保留的安全余量 (deg)}"
    "{yaw-amp          | 12.0                   | 期望 Yaw 激励幅值 (deg，自动按限位裁剪)}"
    "{pitch-amp        | 6.0                    | 期望 Pitch 激励幅值 (deg，自动按限位裁剪)}"
    "{rate r           | 100.0                  | 采样与控制频率 (Hz)}"
    "{f-max            | 8.0                    | Yaw 扫频最高频率 (Hz)}"
    "{with-ff          | true                   | 扫频时同步下发解析速度/加速度前馈给电控}"
    "{skip-mode-wait   | false                  | 跳过等待遥控器切入 AUTO_AIM（仅台架测试用）}"
    "{out o            | gimbal_sysid_real.csv  | 输出 CSV 路径}"
    "{@config-path     | configs/standard3.yaml | yaml 配置文件路径}";

  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  const auto config_path = cli.get<std::string>(0);
  const auto bus = cli.get<std::string>("bus");
  const auto out_path = cli.get<std::string>("out");
  const double margin_deg = std::max(0.5, cli.get<double>("margin"));
  const double yaw_min_deg = cli.get<double>("yaw-min");
  const double yaw_max_deg = cli.get<double>("yaw-max");
  const double pitch_min_deg = cli.get<double>("pitch-min");
  const double pitch_max_deg = cli.get<double>("pitch-max");
  const double rate_hz = std::clamp(cli.get<double>("rate"), 20.0, 500.0);
  const double f_max = std::clamp(cli.get<double>("f-max"), 2.0, 15.0);
  const bool with_ff = cli.get<bool>("with-ff");
  const bool skip_mode_wait = cli.get<bool>("skip-mode-wait");

  if (bus != "gimbal" && bus != "cboard") {
    tools::logger()->error("[RealSysID] 不支持的 --bus={}，仅支持 gimbal 或 cboard", bus);
    return 1;
  }

  // 安全限位合法性校验
  const double safe_yaw_min = (yaw_min_deg + margin_deg) * DEG2RAD;
  const double safe_yaw_max = (yaw_max_deg - margin_deg) * DEG2RAD;
  const double safe_pitch_min = (pitch_min_deg + margin_deg) * DEG2RAD;
  const double safe_pitch_max = (pitch_max_deg - margin_deg) * DEG2RAD;

  if (safe_yaw_max <= safe_yaw_min + 2.0 * DEG2RAD) {
    tools::logger()->error(
      "[RealSysID] Yaw 限位过窄: [{:.1f}°, {:.1f}°] 扣除余量 {:.1f}° 后不足以激励",
      yaw_min_deg, yaw_max_deg, margin_deg);
    return 1;
  }
  if (safe_pitch_max <= safe_pitch_min + 2.0 * DEG2RAD) {
    tools::logger()->error(
      "[RealSysID] Pitch 限位过窄: [{:.1f}°, {:.1f}°] 扣除余量 {:.1f}° 后不足以激励",
      pitch_min_deg, pitch_max_deg, margin_deg);
    return 1;
  }

  const bool yaw_unlimited = (yaw_max_deg - yaw_min_deg >= 350.0);
  const double yaw_amp = std::min(
    std::abs(cli.get<double>("yaw-amp")) * DEG2RAD, 0.45 * (safe_yaw_max - safe_yaw_min));
  const double pitch_amp = std::min(
    std::abs(cli.get<double>("pitch-amp")) * DEG2RAD, 0.45 * (safe_pitch_max - safe_pitch_min));

  std::ofstream csv(out_path, std::ios::out | std::ios::trunc);
  if (!csv.is_open()) {
    tools::logger()->error("[RealSysID] 无法打开输出 CSV: {}", out_path);
    return 1;
  }
  csv << "t_sim_s,t_wall_s,phase,cmd_yaw_rad,muzzle_yaw_rad,cmd_pitch_rad,muzzle_pitch_rad,"
         "cmd_yaw_vel,cmd_yaw_acc,cmd_pitch_vel,cmd_pitch_acc,"
         "yaw_vel_meas,pitch_vel_meas,bullet_speed\n";

  tools::Exiter exiter;
  std::unique_ptr<io::Gimbal> gimbal;
  std::unique_ptr<io::CBoard> cboard;

  if (bus == "gimbal") {
    gimbal = std::make_unique<io::Gimbal>(config_path);
  } else {
    cboard = std::make_unique<io::CBoard>(config_path);
  }

  auto read_state = [&](double & yaw, double & pitch, double & yv, double & pv, double & bs) {
    if (gimbal) {
      auto st = gimbal->state();
      yaw = st.yaw;
      pitch = st.pitch;
      yv = st.yaw_vel;
      pv = st.pitch_vel;
      bs = st.bullet_speed;
    } else {
      auto q = cboard->imu_at(std::chrono::steady_clock::now());
      auto ypr = tools::eulers(q, 2, 1, 0);
      yaw = ypr[0];
      pitch = ypr[1];
      yv = 0.0;
      pv = 0.0;
      bs = cboard->bullet_speed;
    }
  };

  auto send_cmd = [&](
                    bool control, double yaw, double yv, double ya, double pitch, double pv,
                    double pa) {
    if (gimbal) {
      gimbal->send(
        control, false, static_cast<float>(yaw), static_cast<float>(yv), static_cast<float>(ya),
        static_cast<float>(pitch), static_cast<float>(pv), static_cast<float>(pa));
    } else {
      io::Command cmd{control, false, yaw, pitch};
      cboard->send(cmd);
    }
  };

  // 等待串口/CAN 就绪与遥控器切入 AUTO_AIM（防止未就绪时突然甩头）
  if (gimbal && !skip_mode_wait) {
    tools::logger()->info("[RealSysID] 请将遥控器切入 AUTO_AIM 模式以开始辨识（Ctrl+C 随时急停）...");
    while (!exiter.exit() && gimbal->mode() != io::GimbalMode::AUTO_AIM) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (exiter.exit()) return 0;
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  double cur_yaw = 0, cur_pitch = 0, cur_yv = 0, cur_pv = 0, cur_bs = 0;
  read_state(cur_yaw, cur_pitch, cur_yv, cur_pv, cur_bs);

  // 360° 无限位云台围绕当前朝向摆动；有限位云台以安全限位中点为中心
  const double yaw0 = yaw_unlimited ? cur_yaw : 0.5 * (safe_yaw_min + safe_yaw_max);
  const double pitch0 = 0.5 * (safe_pitch_min + safe_pitch_max);

  tools::logger()->info(
    "[RealSysID] 参数确认 | bus={} | rate={:.0f}Hz | "
    "yaw限位=[{:.1f}°, {:.1f}°] 中心={:.1f}° 幅值={:.1f}° | "
    "pitch限位=[{:.1f}°, {:.1f}°] 中心={:.1f}° 幅值={:.1f}° | 输出={}",
    bus, rate_hz, yaw_min_deg, yaw_max_deg, yaw0 * RAD2DEG, yaw_amp * RAD2DEG,
    pitch_min_deg, pitch_max_deg, pitch0 * RAD2DEG, pitch_amp * RAD2DEG, out_path);

  const auto dt_us = std::chrono::microseconds(static_cast<int64_t>(1e6 / rate_hz));
  const auto wall_t0 = std::chrono::steady_clock::now();
  auto next_tick = wall_t0;

  constexpr double TOTAL_DURATION = 22.0;  // s
  int frames = 0;

  while (!exiter.exit()) {
    if (gimbal && !skip_mode_wait && gimbal->mode() != io::GimbalMode::AUTO_AIM) {
      tools::logger()->warn("[RealSysID] 检测到退出 AUTO_AIM，立即中止并释放云台！");
      break;
    }

    const auto now = std::chrono::steady_clock::now();
    const double t = tools::delta_time(now, wall_t0);
    if (t > TOTAL_DURATION) break;

    read_state(cur_yaw, cur_pitch, cur_yv, cur_pv, cur_bs);
    const double rel_yaw = tools::limit_rad(cur_yaw - yaw0);
    const double rel_pitch = cur_pitch - pitch0;

    double dyaw = 0.0, dyaw_vel = 0.0, dyaw_acc = 0.0;
    double dpitch = 0.0, dpitch_vel = 0.0, dpitch_acc = 0.0;
    const char * phase = "hold";

    if (t < 1.0) {
      phase = "hold";
    } else if (t < 2.0) {
      phase = "yaw_step_pos";
      dyaw = yaw_amp;
    } else if (t < 3.0) {
      phase = "yaw_step_neg";
      dyaw = -yaw_amp;
    } else if (t < 4.0) {
      phase = "yaw_step_zero";
      dyaw = 0.0;
    } else if (t < 12.0) {
      phase = "yaw_chirp";
      const double u = t - 4.0;
      double cp, cv, ca, w, dw;
      log_chirp_pva(u, 8.0, 0.5, f_max, cp, cv, ca);
      taper_window_pv(u, 8.0, w, dw);
      dyaw = yaw_amp * w * cp;
      if (with_ff) {
        dyaw_vel = yaw_amp * (dw * cp + w * cv);
        dyaw_acc = yaw_amp * (2.0 * dw * cv + w * ca);
      }
    } else if (t < 13.0) {
      phase = "pitch_step_up";
      dpitch = -pitch_amp;
    } else if (t < 14.0) {
      phase = "pitch_step_down";
      dpitch = pitch_amp;
    } else if (t < 15.0) {
      phase = "pitch_step_zero";
      dpitch = 0.0;
    } else if (t < 21.0) {
      phase = "pitch_chirp";
      const double u = t - 15.0;
      const double fp_max = std::min(f_max, 6.0);
      double cp, cv, ca, w, dw;
      log_chirp_pva(u, 6.0, 0.5, fp_max, cp, cv, ca);
      taper_window_pv(u, 6.0, w, dw);
      dpitch = pitch_amp * w * cp;
      if (with_ff) {
        dpitch_vel = pitch_amp * (dw * cp + w * cv);
        dpitch_acc = pitch_amp * (2.0 * dw * cv + w * ca);
      }
    } else {
      phase = "return_zero";
    }

    // 硬限位保护（双重保险）：若触及限位则前馈归零
    double target_yaw = tools::limit_rad(yaw0 + dyaw);
    if (!yaw_unlimited) {
      const double clamped_y = std::clamp(target_yaw, safe_yaw_min, safe_yaw_max);
      if (clamped_y != target_yaw) {
        dyaw_vel = 0.0;
        dyaw_acc = 0.0;
      }
      target_yaw = clamped_y;
      dyaw = tools::limit_rad(target_yaw - yaw0);
    }
    const double raw_pitch = pitch0 + dpitch;
    const double target_pitch = std::clamp(raw_pitch, safe_pitch_min, safe_pitch_max);
    if (target_pitch != raw_pitch) {
      dpitch_vel = 0.0;
      dpitch_acc = 0.0;
    }
    dpitch = target_pitch - pitch0;

    send_cmd(true, target_yaw, dyaw_vel, dyaw_acc, target_pitch, dpitch_vel, dpitch_acc);

    csv << std::fixed << std::setprecision(6) << t << ',' << t << ',' << phase << ',' << dyaw
        << ',' << rel_yaw << ',' << dpitch << ',' << rel_pitch << ',' << dyaw_vel << ','
        << dyaw_acc << ',' << dpitch_vel << ',' << dpitch_acc << ',' << cur_yv << ',' << cur_pv
        << ',' << cur_bs << '\n';
    frames++;

    next_tick += dt_us;
    const auto after = std::chrono::steady_clock::now();
    if (next_tick > after) {
      std::this_thread::sleep_until(next_tick);
    } else {
      next_tick = after;
    }
  }

  // 安全退出：释放云台控制，永不开火
  send_cmd(false, yaw0, 0.0, 0.0, pitch0, 0.0, 0.0);
  csv.flush();
  tools::logger()->info("[RealSysID] 辨识采样结束，共 {} 帧，已写入 {}", frames, out_path);
  return 0;
}
