#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include "io/sim/sim_gimbal_model.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tools/math_tools.hpp"
#include "tools/sim_metrics.hpp"
#include "tools/trajectory.hpp"

namespace
{

struct ScenarioResult
{
  double hit_rate = 0.0;
  double fire_duty = 0.0;
  uint32_t shots = 0;
  uint32_t hits = 0;
  uint32_t resolved = 0;
  double yaw_rms_deg = 0.0;
};

ScenarioResult run_closed_loop(
  const std::string & config_path, double dist_m, double target_w,
  const Eigen::Vector3d & v_self_true, double roll_true_rad, bool pass_self_vel, bool pass_roll,
  double duration_s = 6.0)
{
  auto_aim::Planner planner(config_path);
  // 跑打补偿单元测试固定使用几何 fire_margin，避免受 YAML 远距 Peek 阈值影响断言
  planner.set_fire_prob_thresh(0.0);
  auto gimbal_model = io::SimGimbalModel::from_yaml(config_path);
  const double bullet_speed = 25.0;
  const double air_k = planner.air_k();
  tools::SimMetrics metrics(20.0, air_k);

  const Eigen::Vector3d muzzle_in_gimbal(0.06, 0.0, 0.12);
  auto_aim::Target target(dist_m, target_w, 0.22, 0.03);
  target.name = auto_aim::ArmorName::three;
  target.armor_type = auto_aim::ArmorType::small;

  const double dt = 0.01;  // 100 Hz
  const int steps = static_cast<int>(duration_s / dt);
  gimbal_model.reset(0.0, 0.0, 0ULL);

  tools::SimMetricsSnapshot last_snap{};
  for (int k = 0; k < steps; ++k) {
    const double t_sec = k * dt;
    const uint64_t ts_ns = static_cast<uint64_t>(t_sec * 1e9);

    if (k > 0) {
      // 真实世界：敌方旋转且位置相对我方移动（我方以 v_self_true 跑打）
      target.predict(dt, v_self_true);
      // 单元测试无视觉 update()，清空持续累加的过程噪声协方差 P，避免 6s 纯预测导致 P 发散到数十米
      const_cast<tools::ExtendedKalmanFilter &>(target.ekf()).P.setZero();
    }

    const Eigen::Vector2d cur_yp = gimbal_model.state_yp();
    auto_aim::SelfMotion sm;
    sm.vel = pass_self_vel ? v_self_true : Eigen::Vector3d::Zero();
    sm.roll = pass_roll ? roll_true_rad : 0.0;
    sm.muzzle_in_gimbal = muzzle_in_gimbal;

    auto plan = planner.plan(target, bullet_speed, cur_yp, sm);
    const Eigen::Vector2d next_yp = gimbal_model.step(
      plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel, plan.pitch_acc, ts_ns);

    // 构造当前帧真实装甲板列表
    std::vector<tools::SimArmorPlate> plates;
    for (const auto & a : target.armor_xyzayp_list()) {
      tools::SimArmorPlate p;
      p.center = a.head<3>();
      p.yaw = a[3];
      p.ring_roll = a[4];
      p.ring_pitch = a[5];
      p.mount_pitch = 15.0 * M_PI / 180.0;
      p.width = 0.135;
      p.height = 0.125;
      plates.push_back(p);
    }

    // 真实炮口位置受 roll_true_rad 影响
    const Eigen::Matrix3d R_g2w =
      (Eigen::AngleAxisd(cur_yp[0], Eigen::Vector3d::UnitZ()) *
       Eigen::AngleAxisd(cur_yp[1], Eigen::Vector3d::UnitY()) *
       Eigen::AngleAxisd(roll_true_rad, Eigen::Vector3d::UnitX()))
        .toRotationMatrix();
    const Eigen::Vector3d muzzle_origin = R_g2w * muzzle_in_gimbal;

    // 暖机 0.4s 后开始统计（等待云台收敛）
    const bool fire_cmd = (t_sec >= 0.4) && plan.fire;
    last_snap = metrics.update(
      t_sec, Eigen::Vector3d(cur_yp[0], cur_yp[1], roll_true_rad), t_sec >= 0.4 && plan.control,
      plan.ref_yaw, plan.ref_pitch, fire_cmd, bullet_speed, plates, muzzle_origin, v_self_true);
    (void)next_yp;
  }

  ScenarioResult res;
  res.hit_rate = last_snap.hit_rate;
  res.fire_duty = last_snap.fire_duty;
  res.shots = last_snap.shots;
  res.hits = last_snap.hits;
  res.resolved = last_snap.resolved;
  res.yaw_rms_deg = last_snap.yaw_err_rms * 180.0 / M_PI;
  return res;
}

}  // namespace

int main()
{
  int failures = 0;

  // 1. 验证 Trajectory3D 与 ballistic_point_3d 闭环一致性（无阻力 & 17mm 二次阻力）
  for (double k1 : {0.0, tools::RM_17MM_DRAG_K}) {
    const Eigen::Vector3d target_pos(3.2, -0.8, 0.35);
    const Eigen::Vector3d v_self(-0.7, 2.6, 0.15);
    const double v0 = 24.0;
    tools::Trajectory3D traj(v0, target_pos, v_self, k1);
    if (traj.unsolvable) {
      std::cerr << "[FAIL] Trajectory3D unsolvable for k1=" << k1 << "\n";
      return 1;
    }
    const Eigen::Vector3d hit =
      tools::ballistic_point_3d(v0, traj.yaw, traj.pitch, v_self, k1, traj.fly_time);
    const double err = (hit - target_pos).norm();
    std::printf(
      "[Test 1] Trajectory3D (k1=%.5f): t=%.4fs yaw=%.2f° pitch=%.2f° err=%.4fmm\n", k1,
      traj.fly_time, traj.yaw * 180.0 / M_PI, traj.pitch * 180.0 / M_PI, err * 1000.0);
    if (err > 2e-3) {
      std::cerr << "[FAIL] Trajectory3D error too large: " << err << " m\n";
      failures++;
    }
  }

  const std::string cfg = "configs/sim.yaml";

  // 2. 静止打小陀螺：命中率 > 90%，开火占空比高
  auto r_static = run_closed_loop(cfg, 2.0, 6.0, Eigen::Vector3d::Zero(), 0.0, true, true);
  std::printf(
    "[Test 2] 静止小陀螺(w=6): hit_rate=%.1f%% (%u/%u) fire_duty=%.1f%% yaw_rms=%.2f°\n",
    r_static.hit_rate * 100.0, r_static.hits, r_static.resolved, r_static.fire_duty * 100.0,
    r_static.yaw_rms_deg);
  if (r_static.hit_rate < 0.90 || r_static.fire_duty < 0.55 || r_static.resolved < 30) {
    std::cerr << "[FAIL] Static spin target did not meet >90% hit rate & high fire duty\n";
    failures++;
  }

  // 3. 高速横移跑打（2.5 m/s）：对比开/关底盘速度融合
  const Eigen::Vector3d v_strafe(-0.3, 2.5, 0.0);
  auto r_move_off = run_closed_loop(cfg, 2.5, 5.0, v_strafe, 0.0, false, true);
  auto r_move_on = run_closed_loop(cfg, 2.5, 5.0, v_strafe, 0.0, true, true);
  std::printf(
    "[Test 3] 横移跑打(2.5m/s) 关闭补偿: hit_rate=%.1f%% (%u/%u) fire_duty=%.1f%%\n",
    r_move_off.hit_rate * 100.0, r_move_off.hits, r_move_off.resolved,
    r_move_off.fire_duty * 100.0);
  std::printf(
    "[Test 3] 横移跑打(2.5m/s) 开启补偿: hit_rate=%.1f%% (%u/%u) fire_duty=%.1f%%\n",
    r_move_on.hit_rate * 100.0, r_move_on.hits, r_move_on.resolved, r_move_on.fire_duty * 100.0);
  if (r_move_on.hit_rate < 0.90 || r_move_on.hit_rate <= r_move_off.hit_rate + 0.15) {
    std::cerr << "[FAIL] Shoot-on-move velocity compensation did not meet >90% hit rate\n";
    failures++;
  }

  // 4. 机身大滚转角（roll = 18°）补偿
  const double roll_rad = 18.0 * M_PI / 180.0;
  auto r_roll_off = run_closed_loop(cfg, 2.0, 5.0, Eigen::Vector3d::Zero(), roll_rad, true, false);
  auto r_roll_on = run_closed_loop(cfg, 2.0, 5.0, Eigen::Vector3d::Zero(), roll_rad, true, true);
  std::printf(
    "[Test 4] 车身滚转(roll=18°) 关闭补偿: hit_rate=%.1f%% (%u/%u) fire_duty=%.1f%%\n",
    r_roll_off.hit_rate * 100.0, r_roll_off.hits, r_roll_off.resolved,
    r_roll_off.fire_duty * 100.0);
  std::printf(
    "[Test 4] 车身滚转(roll=18°) 开启补偿: hit_rate=%.1f%% (%u/%u) fire_duty=%.1f%%\n",
    r_roll_on.hit_rate * 100.0, r_roll_on.hits, r_roll_on.resolved, r_roll_on.fire_duty * 100.0);
  if (r_roll_on.hit_rate < 0.90) {
    std::cerr << "[FAIL] Roll compensation did not meet >90% hit rate\n";
    failures++;
  }

  // 5. 联合工况：斜向跑打 + 滚转 + 高速小陀螺 (w=7 rad/s)
  const Eigen::Vector3d v_combo(0.6, -2.0, 0.0);
  auto r_combo =
    run_closed_loop(cfg, 2.2, 7.0, v_combo, -12.0 * M_PI / 180.0, true, true);
  std::printf(
    "[Test 5] 联合跑打+滚转+小陀螺(w=7): hit_rate=%.1f%% (%u/%u) fire_duty=%.1f%%\n",
    r_combo.hit_rate * 100.0, r_combo.hits, r_combo.resolved, r_combo.fire_duty * 100.0);
  if (r_combo.hit_rate < 0.90 || r_combo.fire_duty < 0.45) {
    std::cerr << "[FAIL] Combined shoot-on-move + roll + spin failed\n";
    failures++;
  }

  if (failures == 0) {
    std::printf("ALL SHOOT-ON-MOVE & FIRE WINDOW TESTS PASSED!\n");
    return 0;
  }
  return 1;
}
