#ifndef AUTO_AIM__FIRE_MLP_HPP
#define AUTO_AIM__FIRE_MLP_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

#include "fire_mlp_weights.hpp"

namespace auto_aim
{

// Layer 3 轻量残差 MLP：14 -> 32 (SiLU) -> 16 (SiLU) -> 1 (2.0 * tanh)
// 纯静态 Eigen 矩阵运算，无堆分配、零外部推理框架依赖，Orin NX / NUC 单次推理 < 2 us。
class FireResidualMLP
{
public:
  static constexpr int IN_DIM = fire_mlp_default::IN_DIM;
  static constexpr int H1_DIM = fire_mlp_default::H1_DIM;
  static constexpr int H2_DIM = fire_mlp_default::H2_DIM;
  static constexpr float MAX_DELTA_LOGIT = fire_mlp_default::MAX_DELTA_LOGIT;

  using VecIn = Eigen::Matrix<float, IN_DIM, 1>;
  using VecH1 = Eigen::Matrix<float, H1_DIM, 1>;
  using VecH2 = Eigen::Matrix<float, H2_DIM, 1>;
  using MatW1 = Eigen::Matrix<float, H1_DIM, IN_DIM, Eigen::RowMajor>;
  using MatW2 = Eigen::Matrix<float, H2_DIM, H1_DIM, Eigen::RowMajor>;

  FireResidualMLP() { load_default(); }

  void load_default()
  {
    has_weights_ = fire_mlp_default::HAS_TRAINED_WEIGHTS;
    w1_ = Eigen::Map<const MatW1>(fire_mlp_default::W1.data());
    b1_ = Eigen::Map<const VecH1>(fire_mlp_default::B1.data());
    w2_ = Eigen::Map<const MatW2>(fire_mlp_default::W2.data());
    b2_ = Eigen::Map<const VecH2>(fire_mlp_default::B2.data());
    w3_ = Eigen::Map<const VecH2>(fire_mlp_default::W3.data());
    b3_ = fire_mlp_default::B3;
  }

  bool has_weights() const { return has_weights_; }

  static VecIn extract_features(
    double hit_y_norm, double hit_z_norm, double cos_inc, double dist, double fly_time,
    bool is_big_armor, double target_w, double target_v_tan, double self_speed, double self_roll,
    double ekf_sigma_phase, double ekf_sigma_pos, double sigma_y_m, double sigma_z_m)
  {
    VecIn x;
    x[0] = static_cast<float>(std::clamp(std::abs(hit_y_norm), 0.0, 2.0));
    x[1] = static_cast<float>(std::clamp(std::abs(hit_z_norm), 0.0, 2.0));
    x[2] = static_cast<float>(std::clamp(cos_inc, 0.0, 1.0));
    x[3] = static_cast<float>(std::clamp(dist / 5.0, 0.0, 2.0));
    x[4] = static_cast<float>(std::clamp(fly_time / 0.25, 0.0, 2.0));
    x[5] = is_big_armor ? 1.0f : 0.0f;
    x[6] = static_cast<float>(std::clamp(std::abs(target_w) / 10.0, 0.0, 2.0));
    x[7] = static_cast<float>(std::clamp(std::abs(target_v_tan) / 4.0, 0.0, 2.0));
    x[8] = static_cast<float>(std::clamp(self_speed / 4.0, 0.0, 2.0));
    x[9] = static_cast<float>(std::clamp(std::abs(self_roll) / 0.4, 0.0, 2.0));
    x[10] = static_cast<float>(std::clamp(ekf_sigma_phase / 0.2, 0.0, 2.5));
    x[11] = static_cast<float>(std::clamp(ekf_sigma_pos / 0.1, 0.0, 2.5));
    x[12] = static_cast<float>(std::clamp(sigma_y_m / 0.1, 0.0, 2.5));
    x[13] = static_cast<float>(std::clamp(sigma_z_m / 0.1, 0.0, 2.5));
    return x;
  }

  static inline float silu(float v) { return v / (1.0f + std::exp(-v)); }

  double predict_delta_logit(const VecIn & x) const
  {
    if (!has_weights_) return 0.0;
    VecH1 h1 = w1_ * x + b1_;
    for (int i = 0; i < H1_DIM; ++i) h1[i] = silu(h1[i]);
    VecH2 h2 = w2_ * h1 + b2_;
    for (int i = 0; i < H2_DIM; ++i) h2[i] = silu(h2[i]);
    const float raw = w3_.dot(h2) + b3_;
    return static_cast<double>(MAX_DELTA_LOGIT * std::tanh(raw));
  }

  static double combine_prob(double p_phys, double delta_logit)
  {
    const double p_clamped = std::clamp(p_phys, 1e-4, 1.0 - 1e-4);
    const double logit = std::log(p_clamped / (1.0 - p_clamped)) + delta_logit;
    return 1.0 / (1.0 + std::exp(-std::clamp(logit, -20.0, 20.0)));
  }

private:
  bool has_weights_ = false;
  MatW1 w1_ = MatW1::Zero();
  VecH1 b1_ = VecH1::Zero();
  MatW2 w2_ = MatW2::Zero();
  VecH2 b2_ = VecH2::Zero();
  VecH2 w3_ = VecH2::Zero();
  float b3_ = 0.0f;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__FIRE_MLP_HPP
