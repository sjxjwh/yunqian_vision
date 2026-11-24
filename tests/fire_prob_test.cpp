#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "io/sim/sim_gimbal_model.hpp"
#include "tasks/auto_aim/planner/fire_mlp.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tools/math_tools.hpp"
#include "tools/runtime_params.hpp"
#include "tools/sim_metrics.hpp"

namespace
{

struct Scenario
{
  std::string name;
  double dist_m;
  double target_w;
  Eigen::Vector3d v_self;
  double self_roll_rad;
  bool is_big_armor;
  double gun_sigma_deg;
  double ekf_pos_sigma_m;
  double ekf_phase_sigma_rad;
  double delay_jitter_s;
};

struct EvalStats
{
  uint32_t shots = 0;
  uint32_t hits = 0;
  uint32_t resolved = 0;
  uint64_t fire_frames = 0;
  uint64_t ctrl_frames = 0;
  double sum_p_fired = 0.0;

  double hit_rate() const { return resolved > 0 ? static_cast<double>(hits) / resolved : 0.0; }
  double fire_duty() const
  {
    return ctrl_frames > 0 ? static_cast<double>(fire_frames) / ctrl_frames : 0.0;
  }
  double mean_p() const { return fire_frames > 0 ? sum_p_fired / fire_frames : 0.0; }
};

struct TrainSample
{
  auto_aim::FireResidualMLP::VecIn x;
  float p_phys = 0.5f;
  float logit_phys = 0.0f;
  float label = 0.0f;
};

// 在精确 6DoF 真值 + 真实物理误差（枪管散布、EKF 估计误差、发弹延迟抖动）下闭环评估；
// 若 out_samples 非空，则把每发已结算弹丸的无偏特征与真值标签写入 out_samples。
EvalStats eval_scenario(
  const std::string & config_path, const Scenario & sc, double fire_margin,
  double fire_prob_thresh, bool use_mlp, double known_gun_sigma_deg, uint32_t seed,
  double duration_s = 12.0, std::vector<TrainSample> * out_samples = nullptr,
  double fire_prob_thresh_far = 0.0,
  auto_aim::LevelPhase level_phase = auto_aim::LevelPhase::Unknown,
  auto_aim::FireTacticalMode tact_mode = auto_aim::FireTacticalMode::Normal)
{
  auto_aim::Planner planner(config_path);
  planner.set_fire_margin(fire_margin);
  planner.set_fire_prob_schedule(fire_prob_thresh, fire_prob_thresh_far, 3.0, 5.0);
  planner.set_use_fire_mlp(use_mlp);
  planner.set_level_phase(level_phase);
  planner.set_fire_tactical_mode(tact_mode);
  planner.set_gun_sigma_deg(known_gun_sigma_deg, known_gun_sigma_deg);
  planner.set_fire_delay_jitter(sc.delay_jitter_s);

  auto gimbal_model = io::SimGimbalModel::from_yaml(config_path);
  const double bullet_speed = 25.0;
  const double air_k = planner.air_k();
  tools::SimMetrics gt_metrics(20.0, air_k);

  const Eigen::Vector3d muzzle_in_gimbal(0.06, 0.0, 0.12);
  auto_aim::Target gt_target(sc.dist_m, sc.target_w, sc.is_big_armor ? 0.26 : 0.22, 0.03);
  gt_target.name = sc.is_big_armor ? auto_aim::ArmorName::one : auto_aim::ArmorName::three;
  gt_target.armor_type = sc.is_big_armor ? auto_aim::ArmorType::big : auto_aim::ArmorType::small;

  std::mt19937 rng(seed);
  std::normal_distribution<double> n01(0.0, 1.0);
  constexpr double D2R = M_PI / 180.0;

  // 慢变 EKF 估计误差（AR(1)，时间常数 ~0.15s，模拟真实视觉跟踪中相关的相位/位置误差）
  double err_x = 0.0, err_y = 0.0, err_z = 0.0, err_phi = 0.0;
  const double dt = 0.01;
  const double ar_a = std::exp(-dt / 0.15);
  const double ar_b = std::sqrt(1.0 - ar_a * ar_a);

  const int steps = static_cast<int>(duration_s / dt);
  gimbal_model.reset(0.0, 0.0, 0ULL);

  std::unordered_map<uint64_t, TrainSample> pending;
  EvalStats st;

  for (int k = 0; k < steps; ++k) {
    const double t_sec = k * dt;
    const uint64_t ts_ns = static_cast<uint64_t>(t_sec * 1e9);
    // 跑打采用周期 4s 的往复横移（模拟实战左右拉扯/蛇皮机动），保持在设定作战距离附近，
    // 避免 15s 单向匀速狂奔跑出 30m 场地外。
    const double strafe_cos = std::cos(2.0 * M_PI * t_sec / 4.0);
    const Eigen::Vector3d cur_v_self = sc.v_self * strafe_cos;

    if (k > 0) {
      gt_target.predict(dt, cur_v_self);
      err_x = ar_a * err_x + ar_b * sc.ekf_pos_sigma_m * n01(rng);
      err_y = ar_a * err_y + ar_b * sc.ekf_pos_sigma_m * n01(rng);
      err_z = ar_a * err_z + ar_b * (0.5 * sc.ekf_pos_sigma_m) * n01(rng);
      err_phi = ar_a * err_phi + ar_b * sc.ekf_phase_sigma_rad * n01(rng);
    }

    auto_aim::Target est_target = gt_target;
    auto & ekf = const_cast<tools::ExtendedKalmanFilter &>(est_target.ekf());
    ekf.x[0] += err_x;
    ekf.x[2] += err_y;
    ekf.x[4] += err_z;
    ekf.x[6] = tools::limit_rad(ekf.x[6] + err_phi);
    ekf.P.setZero();
    ekf.P(0, 0) = sc.ekf_pos_sigma_m * sc.ekf_pos_sigma_m;
    ekf.P(2, 2) = sc.ekf_pos_sigma_m * sc.ekf_pos_sigma_m;
    ekf.P(4, 4) = (0.5 * sc.ekf_pos_sigma_m) * (0.5 * sc.ekf_pos_sigma_m);
    ekf.P(6, 6) = sc.ekf_phase_sigma_rad * sc.ekf_phase_sigma_rad;
    ekf.P(7, 7) = 0.04;
    ekf.P(8, 8) = 0.008 * 0.008;

    const Eigen::Vector2d cur_yp = gimbal_model.state_yp();
    auto_aim::SelfMotion sm;
    sm.vel = cur_v_self;
    sm.roll = sc.self_roll_rad;
    sm.muzzle_in_gimbal = muzzle_in_gimbal;

    auto plan = planner.plan(est_target, bullet_speed, cur_yp, sm);
    gimbal_model.step(
      plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel, plan.pitch_acc, ts_ns);

    std::vector<tools::SimArmorPlate> gt_plates;
    for (const auto & a : gt_target.armor_xyzayp_list()) {
      tools::SimArmorPlate p;
      p.center = a.head<3>();
      p.yaw = a[3];
      p.ring_roll = a[4];
      p.ring_pitch = a[5];
      p.mount_pitch = 15.0 * M_PI / 180.0;
      p.width = sc.is_big_armor ? 0.230 : 0.135;
      p.height = sc.is_big_armor ? 0.127 : 0.125;
      gt_plates.push_back(p);
    }

    const double dyaw_gun = sc.gun_sigma_deg * D2R * n01(rng);
    const double dpitch_gun = sc.gun_sigma_deg * D2R * n01(rng);
    const double dt_jit = sc.delay_jitter_s * n01(rng);
    const double dyaw_jit =
      ((sc.is_big_armor ? 0.26 : 0.22) * sc.target_w * dt_jit) / std::max(0.5, sc.dist_m);

    const double shot_yaw = tools::limit_rad(cur_yp[0] + dyaw_gun + dyaw_jit);
    const double shot_pitch = cur_yp[1] + dpitch_gun;

    const Eigen::Matrix3d R_g2w =
      (Eigen::AngleAxisd(cur_yp[0], Eigen::Vector3d::UnitZ()) *
       Eigen::AngleAxisd(cur_yp[1], Eigen::Vector3d::UnitY()) *
       Eigen::AngleAxisd(sc.self_roll_rad, Eigen::Vector3d::UnitX()))
        .toRotationMatrix();
    const Eigen::Vector3d muzzle_origin = R_g2w * muzzle_in_gimbal;

    const bool active = (t_sec >= 0.4) && plan.control;
    const bool fire_cmd = active && plan.fire;
    if (active) {
      st.ctrl_frames++;
      if (fire_cmd) {
        st.fire_frames++;
        st.sum_p_fired += use_mlp ? plan.diag.p_hit : plan.diag.p_phys;
      }
    }

    uint64_t fired_id = 0;
    auto snap = gt_metrics.update(
      t_sec, Eigen::Vector3d(shot_yaw, shot_pitch, sc.self_roll_rad), active, plan.ref_yaw,
      plan.ref_pitch, fire_cmd, bullet_speed, gt_plates, muzzle_origin, cur_v_self, &fired_id);

    if (out_samples) {
      if (fired_id > 0 && plan.diag.valid) {
        const auto & d = plan.diag;
        TrainSample ts;
        ts.x = auto_aim::FireResidualMLP::extract_features(
          d.hit_y_norm, d.hit_z_norm, d.cos_inc, d.dist, d.fly_time, d.is_big_armor, d.target_w,
          d.target_v_tan, d.self_speed, d.self_roll, d.ekf_sigma_phase, d.ekf_sigma_pos,
          d.sigma_y_m, d.sigma_z_m);
        const double pc = std::clamp(d.p_phys, 1e-4, 1.0 - 1e-4);
        ts.p_phys = static_cast<float>(pc);
        ts.logit_phys = static_cast<float>(std::log(pc / (1.0 - pc)));
        pending[fired_id] = ts;
      }
      for (const auto & r : snap.resolved_shots) {
        auto it = pending.find(r.shot_id);
        if (it != pending.end()) {
          it->second.label = (r.outcome == 1) ? 1.0f : 0.0f;
          out_samples->push_back(it->second);
          pending.erase(it);
        }
      }
    }

    st.shots = snap.shots;
    st.hits = snap.hits;
    st.resolved = snap.resolved;
  }
  return st;
}

EvalStats sum_stats(const std::vector<EvalStats> & list)
{
  EvalStats total;
  for (const auto & s : list) {
    total.shots += s.shots;
    total.hits += s.hits;
    total.resolved += s.resolved;
    total.fire_frames += s.fire_frames;
    total.ctrl_frames += s.ctrl_frames;
    total.sum_p_fired += s.sum_p_fired;
  }
  return total;
}

// 纯 C++ Eigen3 残差 MLP 训练器：14 -> 32 (SiLU) -> 16 (SiLU) -> 1 (2.0 * tanh)
struct MlpParams
{
  static constexpr int IN = auto_aim::FireResidualMLP::IN_DIM;
  static constexpr int H1 = auto_aim::FireResidualMLP::H1_DIM;
  static constexpr int H2 = auto_aim::FireResidualMLP::H2_DIM;
  static constexpr float MAX_D = auto_aim::FireResidualMLP::MAX_DELTA_LOGIT;

  Eigen::Matrix<float, H1, IN, Eigen::RowMajor> W1;
  Eigen::Matrix<float, H1, 1> B1;
  Eigen::Matrix<float, H2, H1, Eigen::RowMajor> W2;
  Eigen::Matrix<float, H2, 1> B2;
  Eigen::Matrix<float, H2, 1> W3;
  float B3 = 0.0f;

  void set_zero()
  {
    W1.setZero();
    B1.setZero();
    W2.setZero();
    B2.setZero();
    W3.setZero();
    B3 = 0.0f;
  }
};

struct CalibMetrics
{
  double bce = 0.0;
  double brier = 0.0;
  double ece = 0.0;
  double acc = 0.0;
};

inline float sigmoid_f(float z)
{
  z = std::clamp(z, -20.0f, 20.0f);
  return 1.0f / (1.0f + std::exp(-z));
}

CalibMetrics evaluate_dataset(const std::vector<TrainSample> & data, const MlpParams * p)
{
  CalibMetrics m;
  if (data.empty()) return m;
  constexpr int BINS = 10;
  int bin_cnt[BINS] = {};
  double bin_p[BINS] = {};
  double bin_y[BINS] = {};

  for (const auto & s : data) {
    float delta = 0.0f;
    if (p) {
      Eigen::Matrix<float, MlpParams::H1, 1> h1 = p->W1 * s.x + p->B1;
      for (int i = 0; i < MlpParams::H1; ++i) h1[i] = h1[i] * sigmoid_f(h1[i]);
      Eigen::Matrix<float, MlpParams::H2, 1> h2 = p->W2 * h1 + p->B2;
      for (int i = 0; i < MlpParams::H2; ++i) h2[i] = h2[i] * sigmoid_f(h2[i]);
      delta = MlpParams::MAX_D * std::tanh(p->W3.dot(h2) + p->B3);
    }
    const double prob = std::clamp<double>(sigmoid_f(s.logit_phys + delta), 1e-6, 1.0 - 1e-6);
    const double y = s.label;
    m.bce += -(y * std::log(prob) + (1.0 - y) * std::log(1.0 - prob));
    m.brier += (prob - y) * (prob - y);
    if ((prob >= 0.5) == (y >= 0.5)) m.acc += 1.0;
    const int b = std::clamp(static_cast<int>(prob * BINS), 0, BINS - 1);
    bin_cnt[b]++;
    bin_p[b] += prob;
    bin_y[b] += y;
  }
  const double n = static_cast<double>(data.size());
  m.bce /= n;
  m.brier /= n;
  m.acc /= n;
  for (int b = 0; b < BINS; ++b) {
    if (bin_cnt[b] > 0) {
      m.ece += (std::abs(bin_p[b] - bin_y[b])) / n;
    }
  }
  return m;
}

void write_weights_header(const std::string & path, const MlpParams & best)
{
  std::ofstream ofs(path, std::ios::out | std::ios::trunc);
  ofs << "#ifndef AUTO_AIM__FIRE_MLP_WEIGHTS_HPP\n"
      << "#define AUTO_AIM__FIRE_MLP_WEIGHTS_HPP\n\n"
      << "#include <array>\n\n"
      << "namespace auto_aim::fire_mlp_default\n"
      << "{\n"
      << "// 由 `./build/fire_prob_test --train` 自动生成的 Layer 3 残差 MLP 权重。\n"
      << "inline constexpr bool HAS_TRAINED_WEIGHTS = true;\n"
      << "inline constexpr int IN_DIM = " << MlpParams::IN << ";\n"
      << "inline constexpr int H1_DIM = " << MlpParams::H1 << ";\n"
      << "inline constexpr int H2_DIM = " << MlpParams::H2 << ";\n"
      << "inline constexpr float MAX_DELTA_LOGIT = " << std::fixed << std::setprecision(1)
      << MlpParams::MAX_D << "f;\n\n";

  auto dump_arr = [&](const char * name, const float * ptr, int len) {
    ofs << "inline constexpr std::array<float, " << len << "> " << name << " = {\n";
    ofs << std::defaultfloat << std::showpoint << std::setprecision(7);
    for (int i = 0; i < len; ++i) {
      if (i % 8 == 0) ofs << "  ";
      ofs << ptr[i] << "f";
      if (i + 1 < len) ofs << ", ";
      if (i % 8 == 7 || i + 1 == len) ofs << "\n";
    }
    ofs << "};\n\n";
  };

  dump_arr("W1", best.W1.data(), MlpParams::H1 * MlpParams::IN);
  dump_arr("B1", best.B1.data(), MlpParams::H1);
  dump_arr("W2", best.W2.data(), MlpParams::H2 * MlpParams::H1);
  dump_arr("B2", best.B2.data(), MlpParams::H2);
  dump_arr("W3", best.W3.data(), MlpParams::H2);
  ofs << std::setprecision(7) << "inline constexpr float B3 = " << best.B3 << "f;\n"
      << "}  // namespace auto_aim::fire_mlp_default\n\n"
      << "#endif  // AUTO_AIM__FIRE_MLP_WEIGHTS_HPP\n";
}

int run_training(const std::string & cfg)
{
  std::printf("================ 关卡 3：Domain Randomization 无偏采集 + 残差 MLP 训练 ================\n");
  std::mt19937 sc_rng(202701);
  std::uniform_real_distribution<double> u_dist(1.4, 5.2);
  std::uniform_real_distribution<double> u_w(-9.0, 9.0);
  std::uniform_real_distribution<double> u_vself(-2.5, 2.5);
  std::uniform_real_distribution<double> u_roll(-0.30, 0.30);
  std::uniform_real_distribution<double> u_gun(0.05, 0.85);
  std::uniform_real_distribution<double> u_pos(0.01, 0.045);
  std::uniform_real_distribution<double> u_phi(0.02, 0.11);
  std::uniform_real_distribution<double> u_jit(0.0, 0.010);
  std::bernoulli_distribution u_big(0.25);

  std::vector<TrainSample> train_data, val_data;
  const int TOTAL_SCENARIOS = 100;
  const int TRAIN_SCENARIOS = 80;

  for (int s = 0; s < TOTAL_SCENARIOS; ++s) {
    Scenario sc;
    sc.name = "dr_" + std::to_string(s);
    sc.dist_m = u_dist(sc_rng);
    sc.target_w = u_w(sc_rng);
    // 50% 概率静止、50% 概率随机 2D 跑打
    if (s % 2 == 0) {
      sc.v_self = Eigen::Vector3d::Zero();
      sc.self_roll_rad = 0.0;
    } else {
      sc.v_self = Eigen::Vector3d(u_vself(sc_rng) * 0.5, u_vself(sc_rng), 0.0);
      sc.self_roll_rad = u_roll(sc_rng);
    }
    sc.is_big_armor = u_big(sc_rng);
    sc.gun_sigma_deg = u_gun(sc_rng);
    sc.ekf_pos_sigma_m = u_pos(sc_rng);
    sc.ekf_phase_sigma_rad = u_phi(sc_rng);
    sc.delay_jitter_s = u_jit(sc_rng);

    auto & dst = (s < TRAIN_SCENARIOS) ? train_data : val_data;
    // 无偏宽边界采集（margin = 1.25），覆盖正中到边缘外侧的完整过渡带
    eval_scenario(cfg, sc, 1.25, 0.0, false, sc.gun_sigma_deg, 10000 + s, 10.0, &dst);
  }

  auto hit_ratio = [](const std::vector<TrainSample> & ds) {
    double s = 0;
    for (const auto & d : ds) s += d.label;
    return ds.empty() ? 0.0 : s / ds.size();
  };
  std::printf(
    "采集完成: train=%zu (hit=%.1f%%), val=%zu (hit=%.1f%%)\n", train_data.size(),
    hit_ratio(train_data) * 100.0, val_data.size(), hit_ratio(val_data) * 100.0);

  auto base_val = evaluate_dataset(val_data, nullptr);
  std::printf(
    "[Val 纯 P_phys 基线] BCE=%.4f  Brier=%.4f  ECE=%.4f  Acc=%.2f%%\n", base_val.bce,
    base_val.brier, base_val.ece, base_val.acc * 100.0);

  // 初始化残差网络：He 初始化隐藏层，输出层小权重（使初始 Δ_logit ≈ 0，从纯 P_phys 平滑起步）
  std::mt19937 w_rng(42);
  MlpParams w, m_adam, v_adam, grad, best_w;
  m_adam.set_zero();
  v_adam.set_zero();
  {
    std::normal_distribution<float> n1(0.0f, std::sqrt(2.0f / MlpParams::IN));
    std::normal_distribution<float> n2(0.0f, std::sqrt(2.0f / MlpParams::H1));
    std::normal_distribution<float> n3(0.0f, 0.02f);
    for (int i = 0; i < w.W1.size(); ++i) w.W1.data()[i] = n1(w_rng);
    w.B1.setZero();
    for (int i = 0; i < w.W2.size(); ++i) w.W2.data()[i] = n2(w_rng);
    w.B2.setZero();
    for (int i = 0; i < w.W3.size(); ++i) w.W3.data()[i] = n3(w_rng);
    w.B3 = 0.0f;
  }

  std::vector<size_t> idx(train_data.size());
  std::iota(idx.begin(), idx.end(), 0);

  const int EPOCHS = 120;
  const int BATCH = 128;
  const float LR0 = 3e-3f;
  const float WD = 3e-4f;
  const float BETA1 = 0.9f, BETA2 = 0.999f, EPS = 1e-8f;

  double best_val_bce = base_val.bce;
  best_w = w;
  int step_t = 0;

  for (int ep = 1; ep <= EPOCHS; ++ep) {
    std::shuffle(idx.begin(), idx.end(), w_rng);
    const float lr = LR0 * 0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * ep / EPOCHS));

    for (size_t start = 0; start < idx.size(); start += BATCH) {
      const size_t end = std::min(idx.size(), start + BATCH);
      const float inv_b = 1.0f / static_cast<float>(end - start);
      grad.set_zero();

      for (size_t k = start; k < end; ++k) {
        const auto & s = train_data[idx[k]];
        // Forward
        Eigen::Matrix<float, MlpParams::H1, 1> z1 = w.W1 * s.x + w.B1;
        Eigen::Matrix<float, MlpParams::H1, 1> sig1, h1;
        for (int i = 0; i < MlpParams::H1; ++i) {
          sig1[i] = sigmoid_f(z1[i]);
          h1[i] = z1[i] * sig1[i];
        }
        Eigen::Matrix<float, MlpParams::H2, 1> z2 = w.W2 * h1 + w.B2;
        Eigen::Matrix<float, MlpParams::H2, 1> sig2, h2;
        for (int i = 0; i < MlpParams::H2; ++i) {
          sig2[i] = sigmoid_f(z2[i]);
          h2[i] = z2[i] * sig2[i];
        }
        const float raw = w.W3.dot(h2) + w.B3;
        const float th = std::tanh(raw);
        const float delta = MlpParams::MAX_D * th;
        const float prob = sigmoid_f(s.logit_phys + delta);

        // Backward: dL/d(logit) = prob - y
        const float d_delta = (prob - s.label) * inv_b;
        const float d_raw = d_delta * MlpParams::MAX_D * (1.0f - th * th);

        grad.W3 += d_raw * h2;
        grad.B3 += d_raw;

        Eigen::Matrix<float, MlpParams::H2, 1> dh2 = d_raw * w.W3;
        Eigen::Matrix<float, MlpParams::H2, 1> dz2;
        for (int i = 0; i < MlpParams::H2; ++i) {
          // d(SiLU(z))/dz = sig(z) * (1 + z * (1 - sig(z)))
          dz2[i] = dh2[i] * sig2[i] * (1.0f + z2[i] * (1.0f - sig2[i]));
        }
        grad.W2.noalias() += dz2 * h1.transpose();
        grad.B2 += dz2;

        Eigen::Matrix<float, MlpParams::H1, 1> dh1 = w.W2.transpose() * dz2;
        Eigen::Matrix<float, MlpParams::H1, 1> dz1;
        for (int i = 0; i < MlpParams::H1; ++i) {
          dz1[i] = dh1[i] * sig1[i] * (1.0f + z1[i] * (1.0f - sig1[i]));
        }
        grad.W1.noalias() += dz1 * s.x.transpose();
        grad.B1 += dz1;
      }

      // AdamW update
      step_t++;
      const float bc1 = 1.0f - std::pow(BETA1, step_t);
      const float bc2 = 1.0f - std::pow(BETA2, step_t);
      auto adam_arr = [&](float * param, float * m, float * v, const float * g, int n, bool decay) {
        for (int i = 0; i < n; ++i) {
          m[i] = BETA1 * m[i] + (1.0f - BETA1) * g[i];
          v[i] = BETA2 * v[i] + (1.0f - BETA2) * g[i] * g[i];
          const float mh = m[i] / bc1;
          const float vh = v[i] / bc2;
          if (decay) param[i] *= (1.0f - lr * WD);
          param[i] -= lr * mh / (std::sqrt(vh) + EPS);
        }
      };
      adam_arr(w.W1.data(), m_adam.W1.data(), v_adam.W1.data(), grad.W1.data(), w.W1.size(), true);
      adam_arr(w.B1.data(), m_adam.B1.data(), v_adam.B1.data(), grad.B1.data(), w.B1.size(), false);
      adam_arr(w.W2.data(), m_adam.W2.data(), v_adam.W2.data(), grad.W2.data(), w.W2.size(), true);
      adam_arr(w.B2.data(), m_adam.B2.data(), v_adam.B2.data(), grad.B2.data(), w.B2.size(), false);
      adam_arr(w.W3.data(), m_adam.W3.data(), v_adam.W3.data(), grad.W3.data(), w.W3.size(), true);
      adam_arr(&w.B3, &m_adam.B3, &v_adam.B3, &grad.B3, 1, false);
    }

    auto vm = evaluate_dataset(val_data, &w);
    if (vm.bce < best_val_bce) {
      best_val_bce = vm.bce;
      best_w = w;
    }
    if (ep % 20 == 0 || ep == 1) {
      std::printf(
        "  epoch %3d/%d | val BCE=%.4f Brier=%.4f ECE=%.4f Acc=%.2f%%\n", ep, EPOCHS, vm.bce,
        vm.brier, vm.ece, vm.acc * 100.0);
    }
  }

  auto best_vm = evaluate_dataset(val_data, &best_w);
  std::printf(
    "[Val 最优 P_phys+MLP] BCE=%.4f (%.4f)  Brier=%.4f (%.4f)  ECE=%.4f (%.4f)  Acc=%.2f%%\n",
    best_vm.bce, best_vm.bce - base_val.bce, best_vm.brier, best_vm.brier - base_val.brier,
    best_vm.ece, best_vm.ece - base_val.ece, best_vm.acc * 100.0);

  const std::string out_hdr = "tasks/auto_aim/planner/fire_mlp_weights.hpp";
  write_weights_header(out_hdr, best_w);
  std::printf("权重已导出至 %s（请重新编译以生效）\n", out_hdr.c_str());
  return 0;
}

struct PolicySpec
{
  std::string name;
  double fire_margin;
  double prob_near;
  double prob_far;
  bool use_mlp;
  auto_aim::LevelPhase level_phase;
  auto_aim::FireTacticalMode tact_mode;
};

int run_roi_eval(const std::string & cfg)
{
  constexpr double D2R = M_PI / 180.0;
  const double DUR = 15.0;  // 每工况 15 秒 (~200~300 发，低方差)
  // 按机械组指标：常规发射散布 σ=0.18°~0.22°（5m 静止 3σ <= 67.5mm 单板内）；工况 8 保留机械松动恶化压测
  const std::vector<Scenario> scenarios = {
    {"1.超近贴脸(1.5m,w=6)", 1.5, 6.0, {0, 0, 0}, 0.0, false, 0.18, 0.015, 0.04, 0.004},
    {"2.近距大装甲(2.0m,w=5)", 2.0, 5.0, {0, 0, 0}, 0.0, true, 0.20, 0.020, 0.05, 0.004},
    {"3.中近小陀螺(2.5m,w=7)", 2.5, 7.0, {0, 0, 0}, 0.0, false, 0.20, 0.020, 0.05, 0.005},
    {"4.中距跑打滚转(3.2m,v=2)", 3.2, 7.0, {0.3, 2.0, 0}, 12 * D2R, false, 0.22, 0.025, 0.06, 0.005},
    {"5.中距直线机动(3.5m,w=0)", 3.5, 0.0, {0.2, 1.5, 0}, 5 * D2R, false, 0.20, 0.025, 0.04, 0.004},
    {"6.远距小陀螺(4.5m,w=6)", 4.5, 6.0, {0, 0, 0}, 0.0, false, 0.20, 0.035, 0.08, 0.005},
    {"7.超远大装甲(5.5m,w=4)", 5.5, 4.0, {0, 0, 0}, 0.0, true, 0.20, 0.040, 0.08, 0.005},
    {"8.散布恶化(3.2m,sig=0.65°)", 3.2, 6.5, {0, 0.8, 0}, 6 * D2R, false, 0.65, 0.025, 0.06, 0.005},
  };

  const std::vector<PolicySpec> policies = {
    {"margin=0.85(原默认)", 0.85, 0.0, 0.0, false, auto_aim::LevelPhase::Unknown, auto_aim::FireTacticalMode::Normal},
    {"Burst爆发收割(-0.15)", 0.85, 0.70, 0.82, true, auto_aim::LevelPhase::Unknown, auto_aim::FireTacticalMode::Burst},
    {"Leveling发育期(-0.08)", 0.85, 0.70, 0.82, true, auto_aim::LevelPhase::Leveling, auto_aim::FireTacticalMode::Normal},
    {"Normal常规(70->82) ", 0.85, 0.70, 0.82, true, auto_aim::LevelPhase::Unknown, auto_aim::FireTacticalMode::Normal},
    {"Capped满级控eta(+0.05)", 0.85, 0.70, 0.82, true, auto_aim::LevelPhase::Capped, auto_aim::FireTacticalMode::Normal},
  };

  auto net_roi = [&](const EvalStats & s, double c_miss, double dur) {
    const double misses = static_cast<double>(s.resolved - s.hits);
    return (static_cast<double>(s.hits) - c_miss * misses) / dur;
  };

  std::vector<EvalStats> totals(policies.size());
  std::printf("================ 8 工况全维度对比（机械精度 σ=0.20° + 动态阈值各档位，每工况 %.0fs）================\n", DUR);

  for (size_t si = 0; si < scenarios.size(); ++si) {
    const auto & sc = scenarios[si];
    std::printf("\n--- 工况 %s ---\n", sc.name.c_str());
    std::printf(
      "  %-22s |  hit%%  ( h / r ) | duty%% |  DPS  | XP/s | ROI(团战0.5) | ROI(常规2.0) | ROI(省弹4.0)\n",
      "策略");
    for (size_t pi = 0; pi < policies.size(); ++pi) {
      const auto & pol = policies[pi];
      auto st = eval_scenario(
        cfg, sc, pol.fire_margin, pol.prob_near, pol.use_mlp, sc.gun_sigma_deg, 500 + si, DUR,
        nullptr, pol.prob_far, pol.level_phase, pol.tact_mode);
      totals[pi].shots += st.shots;
      totals[pi].hits += st.hits;
      totals[pi].resolved += st.resolved;
      totals[pi].fire_frames += st.fire_frames;
      totals[pi].ctrl_frames += st.ctrl_frames;
      totals[pi].sum_p_fired += st.sum_p_fired;

      const double dps = 20.0 * st.hits / (DUR - 0.4);
      // 步兵默认装甲经验速率：80 XP/命中 + 1 XP/发射
      const double xp_per_s = (80.0 * st.hits + 1.0 * st.resolved) / (DUR - 0.4);
      std::printf(
        "  %-22s | %5.1f%% (%3u/%3u) | %5.1f%% | %5.1f | %4.0f |    %+6.2f    |    %+6.2f    |    %+6.2f\n",
        pol.name.c_str(), st.hit_rate() * 100.0, st.hits, st.resolved, st.fire_duty() * 100.0,
        dps, xp_per_s, net_roi(st, 0.5, DUR - 0.4), net_roi(st, 2.0, DUR - 0.4),
        net_roi(st, 4.0, DUR - 0.4));
    }
  }

  const double total_dur = scenarios.size() * (DUR - 0.4);
  std::printf("\n================ 8 工况汇总（总计 %.1fs）================\n", total_dur);
  std::printf(
    "  %-22s |  hit%%  (  h /  r ) | duty%% |  DPS  | XP/s | ROI(团战0.5) | ROI(常规2.0) | ROI(省弹4.0)\n",
    "策略");
  for (size_t pi = 0; pi < policies.size(); ++pi) {
    const auto & st = totals[pi];
    const double dps = 20.0 * st.hits / total_dur;
    const double xp_per_s = (80.0 * st.hits + 1.0 * st.resolved) / total_dur;
    std::printf(
      "  %-22s | %5.1f%% (%4u/%4u) | %5.1f%% | %5.1f | %4.0f |    %+6.2f    |    %+6.2f    |    %+6.2f\n",
      policies[pi].name.c_str(), st.hit_rate() * 100.0, st.hits, st.resolved,
      st.fire_duty() * 100.0, dps, xp_per_s, net_roi(st, 0.5, total_dur),
      net_roi(st, 2.0, total_dur), net_roi(st, 4.0, total_dur));
  }
  return 0;
}

}  // namespace

int main(int argc, char ** argv)
{
  const std::string cfg = "configs/sim.yaml";
  if (argc > 1 && std::strcmp(argv[1], "--train") == 0) {
    return run_training(cfg);
  }
  if (argc > 1 && std::strcmp(argv[1], "--eval-roi") == 0) {
    return run_roi_eval(cfg);
  }

  const std::vector<Scenario> scenarios = {
    {"近距稳态(1.8m, sig=0.15°)", 1.8, 6.0, Eigen::Vector3d::Zero(), 0.0, false, 0.15, 0.015, 0.04, 0.004},
    {"中距跑打(3.2m, sig=0.35°)", 3.2, 7.0, Eigen::Vector3d(0.0, 1.5, 0.0), 0.15, false, 0.35, 0.025, 0.06, 0.005},
    {"远距大散(4.5m, sig=0.60°)", 4.5, 6.0, Eigen::Vector3d::Zero(), 0.0, false, 0.60, 0.035, 0.08, 0.006},
  };

  std::printf("================ 关卡 2 & 3：固定 margin vs 纯 P_phys vs P_phys + 残差 MLP ================\n");
  std::printf("[Baseline: 固定 fire_margin]\n");
  for (double m : {0.50, 0.65, 0.75, 0.85, 0.95}) {
    std::vector<EvalStats> res;
    for (size_t i = 0; i < scenarios.size(); ++i) {
      res.push_back(eval_scenario(cfg, scenarios[i], m, 0.0, false, scenarios[i].gun_sigma_deg, 100 + i));
    }
    auto tot = sum_stats(res);
    std::printf(
      "  margin=%.2f -> total: hit=%5.1f%% (%3u/%3u) duty=%5.1f%% | 近=%5.1f%%(d=%4.1f%%) 中=%5.1f%%(d=%4.1f%%) 远=%5.1f%%(d=%4.1f%%)\n",
      m, tot.hit_rate() * 100.0, tot.hits, tot.resolved, tot.fire_duty() * 100.0,
      res[0].hit_rate() * 100.0, res[0].fire_duty() * 100.0,
      res[1].hit_rate() * 100.0, res[1].fire_duty() * 100.0,
      res[2].hit_rate() * 100.0, res[2].fire_duty() * 100.0);
  }

  std::printf("\n[Layer 2: 纯物理不确定度 P_phys]\n");
  for (double pth : {0.50, 0.60, 0.70, 0.80, 0.88}) {
    std::vector<EvalStats> res;
    for (size_t i = 0; i < scenarios.size(); ++i) {
      res.push_back(eval_scenario(cfg, scenarios[i], 0.85, pth, false, scenarios[i].gun_sigma_deg, 100 + i));
    }
    auto tot = sum_stats(res);
    std::printf(
      "  p_th  =%.2f -> total: hit=%5.1f%% (%3u/%3u) duty=%5.1f%% (mean_p=%.1f%%) | 近=%5.1f%%(d=%4.1f%%) 中=%5.1f%%(d=%4.1f%%) 远=%5.1f%%(d=%4.1f%%)\n",
      pth, tot.hit_rate() * 100.0, tot.hits, tot.resolved, tot.fire_duty() * 100.0,
      tot.mean_p() * 100.0,
      res[0].hit_rate() * 100.0, res[0].fire_duty() * 100.0,
      res[1].hit_rate() * 100.0, res[1].fire_duty() * 100.0,
      res[2].hit_rate() * 100.0, res[2].fire_duty() * 100.0);
  }

  if (auto_aim::fire_mlp_default::HAS_TRAINED_WEIGHTS) {
    std::printf("\n[Layer 3: P_phys + 轻量残差 MLP (14->32->16->1)]\n");
    for (double pth : {0.50, 0.60, 0.70, 0.80, 0.88}) {
      std::vector<EvalStats> res;
      for (size_t i = 0; i < scenarios.size(); ++i) {
        res.push_back(eval_scenario(cfg, scenarios[i], 0.85, pth, true, scenarios[i].gun_sigma_deg, 100 + i));
      }
      auto tot = sum_stats(res);
      std::printf(
        "  p_th  =%.2f -> total: hit=%5.1f%% (%3u/%3u) duty=%5.1f%% (mean_p=%.1f%%) | 近=%5.1f%%(d=%4.1f%%) 中=%5.1f%%(d=%4.1f%%) 远=%5.1f%%(d=%4.1f%%)\n",
        pth, tot.hit_rate() * 100.0, tot.hits, tot.resolved, tot.fire_duty() * 100.0,
        tot.mean_p() * 100.0,
        res[0].hit_rate() * 100.0, res[0].fire_duty() * 100.0,
        res[1].hit_rate() * 100.0, res[1].fire_duty() * 100.0,
        res[2].hit_rate() * 100.0, res[2].fire_duty() * 100.0);
    }
  }

  // 2. 验证纯 MLP 前向耗时（100,000 次均值）
  auto_aim::FireResidualMLP mlp;
  auto feat = auto_aim::FireResidualMLP::extract_features(
    0.35, 0.12, 0.88, 3.2, 0.13, false, 7.0, 1.5, 1.5, 0.1, 0.06, 0.03, 0.04, 0.02);
  volatile double sink = 0.0;
  const int M = 100000;
  const auto mt0 = std::chrono::steady_clock::now();
  for (int i = 0; i < M; ++i) sink += mlp.predict_delta_logit(feat);
  const auto mt1 = std::chrono::steady_clock::now();
  const double ns_per_mlp =
    std::chrono::duration_cast<std::chrono::nanoseconds>(mt1 - mt0).count() / static_cast<double>(M);

  // 3. 验证 RuntimeParams 在线散布注入（模拟实车散布恶化）
  std::printf("\n[Real-Recon 模拟：中远距真实散布 0.70°，对比无散布感知 vs AimCorrector 在线注入]\n");
  Scenario bad_gun{"中远距恶化(3.0m, true_sig=0.70°)", 3.0, 6.0, Eigen::Vector3d::Zero(), 0.0, false, 0.70, 0.020, 0.05, 0.005};
  const bool test_mlp = auto_aim::fire_mlp_default::HAS_TRAINED_WEIGHTS;
  auto st_blind = eval_scenario(cfg, bad_gun, 0.85, 0.72, test_mlp, 0.15, 777, 25.0);
  auto st_adapt = eval_scenario(cfg, bad_gun, 0.85, 0.72, test_mlp, 0.70, 777, 25.0);
  std::printf(
    "  未感知散布(先验 0.15°): hit=%5.1f%% (%3u/%3u) duty=%5.1f%%\n",
    st_blind.hit_rate() * 100.0, st_blind.hits, st_blind.resolved, st_blind.fire_duty() * 100.0);
  std::printf(
    "  注入实测散布(  0.70°): hit=%5.1f%% (%3u/%3u) duty=%5.1f%%\n",
    st_adapt.hit_rate() * 100.0, st_adapt.hits, st_adapt.resolved, st_adapt.fire_duty() * 100.0);

  auto_aim::Planner bench_planner(cfg);
  auto & rp = tools::RuntimeParams::instance();
  rp.set("planner.gun_sigma_yaw", 0.55);
  rp.set("planner.gun_sigma_pitch", 0.45);
  rp.set("planner.fire_prob_thresh", 0.70);
  rp.set("planner.fire_prob_thresh_far", 0.82);
  rp.set("planner.use_fire_mlp", 1.0);

  // 验证 2027 规则动态阈值公式 τ = clamp(τ_base(d) + Δτ_level + Δτ_mode + Δτ_armor, 0.45, 0.90)
  rp.set("planner.level_phase", 0.0);
  rp.set("planner.fire_mode", 0.0);
  rp.set("planner.armor_zone", 0.0);
  const double tau_near = bench_planner.effective_fire_thresh(2.0);  // 0.70
  const double tau_far = bench_planner.effective_fire_thresh(5.0);   // 0.82
  rp.set("planner.level_phase", 1.0);  // Leveling: -0.08
  const double tau_lvl = bench_planner.effective_fire_thresh(2.0);   // 0.62
  rp.set("planner.fire_mode", 1.0);    // Burst: -0.15
  const double tau_burst = bench_planner.effective_fire_thresh(2.0); // 0.47
  rp.set("planner.level_phase", 2.0);  // Capped: +0.05
  rp.set("planner.fire_mode", 2.0);    // Eco: +0.08
  rp.set("planner.armor_zone", 2.0);   // Reinforced: +0.12
  const double tau_cap = bench_planner.effective_fire_thresh(5.0);   // clamp(1.07) -> 0.90

  // 验证高热量软抬升阈值与单板防吞弹射频建议
  rp.set("planner.level_phase", 0.0);
  rp.set("planner.fire_mode", 0.0);
  rp.set("planner.armor_zone", 0.0);
  rp.set("planner.heat_ratio", 0.85);  // +0.04
  const double tau_heat = bench_planner.effective_fire_thresh(2.0);  // 0.74
  const float rate_edge = bench_planner.recommend_fire_rate_hz(0.74, 0.74, false);
  const float rate_switch = bench_planner.recommend_fire_rate_hz(0.74, 0.74, true);
  rp.set("planner.fire_mode", 1.0);    // ForceBurst: 忽略高热量降频与阈值抬升
  const double tau_burst_heat = bench_planner.effective_fire_thresh(2.0);  // 0.55
  const float rate_burst = bench_planner.recommend_fire_rate_hz(0.60, 0.55, false);

  const bool rp_ok =
    std::abs(rp.get("planner.gun_sigma_yaw").value_or(0.0) - 0.55) < 1e-4 &&
    std::abs(rp.get("planner.gun_sigma_pitch").value_or(0.0) - 0.45) < 1e-4 &&
    std::abs(bench_planner.fire_prob_thresh() - 0.70) < 1e-4 && bench_planner.use_fire_mlp() &&
    std::abs(tau_near - 0.70) < 1e-4 && std::abs(tau_far - 0.82) < 1e-4 &&
    std::abs(tau_lvl - 0.62) < 1e-4 && std::abs(tau_burst - 0.47) < 1e-4 &&
    std::abs(tau_cap - 0.90) < 1e-4 && std::abs(tau_heat - 0.74) < 1e-4 &&
    std::abs(tau_burst_heat - 0.55) < 1e-4 && std::abs(rate_edge - 6.0f) < 1e-3f &&
    std::abs(rate_switch - 10.0f) < 1e-3f && std::abs(rate_burst - 10.0f) < 1e-3f;

  std::printf(
    "\n[性能与接口] FireResidualMLP 单次推理 = %.0f ns (%.3f us), trained=%d, "
    "tau(near=%.2f far=%.2f lvl=%.2f burst=%.2f heat=%.2f cap=%.2f), "
    "rate(edge=%.1f switch=%.1f burst=%.1f Hz), RuntimeParams=%s\n",
    ns_per_mlp, ns_per_mlp / 1000.0, auto_aim::fire_mlp_default::HAS_TRAINED_WEIGHTS ? 1 : 0,
    tau_near, tau_far, tau_lvl, tau_burst, tau_heat, tau_cap, rate_edge, rate_switch, rate_burst,
    rp_ok ? "OK" : "FAIL");
  (void)sink;

  if (!rp_ok) {
    std::cerr << "[FAIL] RuntimeParams check failed!\n";
    return 1;
  }
  std::printf("ALL FIRE_PROB LAYER 2+3 TESTS PASSED!\n");
  return 0;
}
