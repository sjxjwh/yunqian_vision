#include "io/sim/sim_camera_model.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>

#include "tools/logger.hpp"

namespace io
{

namespace
{

constexpr double DEG2RAD = M_PI / 180.0;

cv::Mat read_mat_3x3(const YAML::Node & node, const char * name)
{
  auto data = node.as<std::vector<double>>();
  if (data.size() != 9) {
    throw std::runtime_error(std::string(name) + " 必须是 9 元素 3x3 矩阵");
  }
  cv::Mat m(3, 3, CV_64F);
  for (int i = 0; i < 9; ++i) m.at<double>(i / 3, i % 3) = data[i];
  return m;
}

cv::Mat read_distort(const YAML::Node & node, const char * name)
{
  auto data = node.as<std::vector<double>>();
  if (data.size() < 4 || data.size() > 8) {
    throw std::runtime_error(std::string(name) + " 长度必须在 4~8 之间");
  }
  cv::Mat d(1, static_cast<int>(data.size()), CV_64F);
  for (size_t i = 0; i < data.size(); ++i) d.at<double>(0, static_cast<int>(i)) = data[i];
  return d;
}

}  // namespace

Eigen::Matrix3d SimCameraModel::euler_deg_to_rot(const Eigen::Vector3d & rpy_deg)
{
  const double r = rpy_deg.x() * DEG2RAD;
  const double p = rpy_deg.y() * DEG2RAD;
  const double y = rpy_deg.z() * DEG2RAD;
  return (Eigen::AngleAxisd(y, Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(p, Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(r, Eigen::Vector3d::UnitX()))
    .toRotationMatrix();
}

SimCameraModel::SimCameraModel(const Config & cfg) : cfg_(cfg)
{
  const uint32_t seed = cfg_.seed != 0 ? cfg_.seed : std::random_device{}();
  cv_rng_ = cv::RNG(seed);
  rng_.seed(seed);
  init_photometric_lut();
}

SimCameraModel SimCameraModel::from_yaml(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  if (!yaml["sim_camera_model"]) return SimCameraModel{};

  const auto node = yaml["sim_camera_model"];
  Config cfg;
  cfg.enabled = node["enabled"] ? node["enabled"].as<bool>() : false;
  if (!cfg.enabled) return SimCameraModel{};

  cfg.enable_geometry = node["enable_geometry"] ? node["enable_geometry"].as<bool>() : true;
  cfg.output_width = node["output_width"] ? node["output_width"].as<int>() : 0;
  cfg.output_height = node["output_height"] ? node["output_height"].as<int>() : 0;

  if (node["sim_camera_matrix"]) {
    cfg.sim_camera_matrix = read_mat_3x3(node["sim_camera_matrix"], "sim_camera_matrix");
  } else if (yaml["camera_matrix"]) {
    cfg.sim_camera_matrix = read_mat_3x3(yaml["camera_matrix"], "camera_matrix");
  }

  if (node["true_camera_matrix"]) {
    cfg.true_camera_matrix = read_mat_3x3(node["true_camera_matrix"], "true_camera_matrix");
  } else if (yaml["camera_matrix"]) {
    cfg.true_camera_matrix = read_mat_3x3(yaml["camera_matrix"], "camera_matrix");
  }

  if (node["true_distort_coeffs"]) {
    cfg.true_distort_coeffs = read_distort(node["true_distort_coeffs"], "true_distort_coeffs");
  } else if (yaml["distort_coeffs"]) {
    cfg.true_distort_coeffs = read_distort(yaml["distort_coeffs"], "distort_coeffs");
  } else {
    cfg.true_distort_coeffs = cv::Mat::zeros(1, 5, CV_64F);
  }

  if (node["R_perturb_deg"]) {
    auto v = node["R_perturb_deg"].as<std::vector<double>>();
    if (v.size() != 3) {
      throw std::runtime_error("sim_camera_model.R_perturb_deg 必须是 3 元素 [roll, pitch, yaw]");
    }
    cfg.R_perturb_deg = Eigen::Vector3d(v[0], v[1], v[2]);
  }

  cfg.enable_motion_blur =
    node["enable_motion_blur"] ? node["enable_motion_blur"].as<bool>() : false;
  cfg.exposure_ms = node["exposure_ms"] ? node["exposure_ms"].as<double>() : 4.0;
  cfg.max_motion_blur_px =
    node["max_motion_blur_px"] ? node["max_motion_blur_px"].as<double>() : 35.0;

  cfg.gaussian_blur_sigma =
    node["gaussian_blur_sigma"] ? node["gaussian_blur_sigma"].as<double>() : 0.0;
  cfg.vignette_strength =
    node["vignette_strength"] ? node["vignette_strength"].as<double>() : 0.0;

  cfg.brightness_scale =
    node["brightness_scale"] ? node["brightness_scale"].as<double>() : 1.0;
  cfg.gamma = node["gamma"] ? node["gamma"].as<double>() : 1.0;
  cfg.black_level_offset =
    node["black_level_offset"] ? node["black_level_offset"].as<double>() : 0.0;

  cfg.read_noise_std = node["read_noise_std"] ? node["read_noise_std"].as<double>() : 0.0;
  cfg.shot_noise_k = node["shot_noise_k"] ? node["shot_noise_k"].as<double>() : 0.0;

  cfg.target_fps = node["target_fps"] ? node["target_fps"].as<double>() : 0.0;
  cfg.timestamp_offset_ms =
    node["timestamp_offset_ms"] ? node["timestamp_offset_ms"].as<double>() : 0.0;
  cfg.timestamp_jitter_ms =
    node["timestamp_jitter_ms"] ? node["timestamp_jitter_ms"].as<double>() : 0.0;

  cfg.seed = node["seed"] ? node["seed"].as<uint32_t>() : 0u;

  tools::logger()->info(
    "[SimCameraModel] 已启用真实相机复现: geom={}, motion_blur={}({:.1f}ms), "
    "blur_sigma={:.2f}, vignette={:.2f}, gain={:.2f}, gamma={:.2f}, "
    "noise=(read:{:.2f}, shot_k:{:.3f}), fps={:.1f}",
    cfg.enable_geometry, cfg.enable_motion_blur, cfg.exposure_ms, cfg.gaussian_blur_sigma,
    cfg.vignette_strength, cfg.brightness_scale, cfg.gamma, cfg.read_noise_std, cfg.shot_noise_k,
    cfg.target_fps);

  return SimCameraModel(cfg);
}

void SimCameraModel::init_photometric_lut()
{
  use_photometric_lut_ =
    (std::abs(cfg_.brightness_scale - 1.0) > 1e-4) || (std::abs(cfg_.gamma - 1.0) > 1e-4) ||
    (std::abs(cfg_.black_level_offset) > 1e-4);

  const double gamma = std::max(0.05, cfg_.gamma);
  for (int i = 0; i < 256; ++i) {
    if (use_photometric_lut_) {
      double v = (static_cast<double>(i) / 255.0) * cfg_.brightness_scale;
      v = std::clamp(v, 0.0, 1.0);
      double out = 255.0 * std::pow(v, gamma) + cfg_.black_level_offset;
      photometric_lut_[i] = cv::saturate_cast<uint8_t>(std::lround(out));
    } else {
      photometric_lut_[i] = static_cast<uint8_t>(i);
    }
  }

  has_noise_ = (cfg_.read_noise_std > 1e-4) || (cfg_.shot_noise_k > 1e-6);
  const double read_var = cfg_.read_noise_std * cfg_.read_noise_std;
  for (int i = 0; i < 256; ++i) {
    const double sigma = std::sqrt(std::max(0.0, read_var + cfg_.shot_noise_k * i));
    noise_std_q8_[i] = static_cast<int32_t>(std::lround(sigma * 256.0));
  }
}

void SimCameraModel::ensure_maps(const cv::Size & src_size)
{
  if (cached_src_size_ == src_size) return;
  cached_src_size_ = src_size;

  const int out_w = cfg_.output_width > 0 ? cfg_.output_width : src_size.width;
  const int out_h = cfg_.output_height > 0 ? cfg_.output_height : src_size.height;
  out_size_ = cv::Size(out_w, out_h);

  // 1. 预计算几何逆映射表：输出(实机畸变像素) -> 输入(仿真理想针孔像素)
  if (cfg_.enable_geometry && !cfg_.sim_camera_matrix.empty() && !cfg_.true_camera_matrix.empty()) {
    cv::Mat dst_grid(out_h, out_w, CV_64FC2);
    for (int y = 0; y < out_h; ++y) {
      auto * row = dst_grid.ptr<cv::Vec2d>(y);
      for (int x = 0; x < out_w; ++x) {
        row[x] = cv::Vec2d(static_cast<double>(x), static_cast<double>(y));
      }
    }

    Eigen::Matrix3d R_eig = euler_deg_to_rot(cfg_.R_perturb_deg);
    cv::Mat R_cv(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) R_cv.at<double>(r, c) = R_eig(r, c);
    }

    cv::Mat src_grid;
    cv::undistortPoints(
      dst_grid.reshape(2, out_h * out_w), src_grid, cfg_.true_camera_matrix,
      cfg_.true_distort_coeffs, R_cv, cfg_.sim_camera_matrix);
    src_grid = src_grid.reshape(2, out_h);

    cv::Mat map_x(out_h, out_w, CV_32FC1);
    cv::Mat map_y(out_h, out_w, CV_32FC1);
    size_t valid_count = 0;
    const double max_x = static_cast<double>(src_size.width - 1);
    const double max_y = static_cast<double>(src_size.height - 1);

    for (int y = 0; y < out_h; ++y) {
      const auto * srow = src_grid.ptr<cv::Vec2d>(y);
      auto * mx = map_x.ptr<float>(y);
      auto * my = map_y.ptr<float>(y);
      for (int x = 0; x < out_w; ++x) {
        const double sx = srow[x][0];
        const double sy = srow[x][1];
        mx[x] = static_cast<float>(sx);
        my[x] = static_cast<float>(sy);
        if (sx >= 0.0 && sx <= max_x && sy >= 0.0 && sy <= max_y) ++valid_count;
      }
    }
    valid_pixel_ratio_ =
      static_cast<double>(valid_count) / static_cast<double>(out_w * out_h);
    if (valid_pixel_ratio_ < 0.99) {
      tools::logger()->warn(
        "[SimCameraModel] 实机视角超出仿真视野，有效像素占比 {:.1f}%（边缘将出现黑边）",
        valid_pixel_ratio_ * 100.0);
    }
    cv::convertMaps(map_x, map_y, map1_, map2_, CV_16SC2);
  } else {
    map1_.release();
    map2_.release();
    valid_pixel_ratio_ = 1.0;
  }

  // 2. 预计算单通道暗角掩膜（中心为实机光心 cx, cy）
  if (cfg_.vignette_strength > 1e-4) {
    const double cx = !cfg_.true_camera_matrix.empty()
                        ? cfg_.true_camera_matrix.at<double>(0, 2)
                        : 0.5 * out_w;
    const double cy = !cfg_.true_camera_matrix.empty()
                        ? cfg_.true_camera_matrix.at<double>(1, 2)
                        : 0.5 * out_h;
    const double rx = std::max(cx, static_cast<double>(out_w) - cx);
    const double ry = std::max(cy, static_cast<double>(out_h) - cy);
    const double inv_r2 = 1.0 / std::max(1.0, rx * rx + ry * ry);
    const double k = std::clamp(cfg_.vignette_strength, 0.0, 0.98);

    vignette_u8_.create(out_h, out_w, CV_8UC1);
    for (int y = 0; y < out_h; ++y) {
      auto * row = vignette_u8_.ptr<uint8_t>(y);
      const double dy = static_cast<double>(y) - cy;
      for (int x = 0; x < out_w; ++x) {
        const double dx = static_cast<double>(x) - cx;
        const double gain = std::clamp(1.0 - k * (dx * dx + dy * dy) * inv_r2, 0.0, 1.0);
        row[x] = cv::saturate_cast<uint8_t>(std::lround(gain * 255.0));
      }
    }
  } else {
    vignette_u8_.release();
  }
}

void SimCameraModel::apply_motion_blur(cv::Mat & img, const Eigen::Vector3d & omega_cam_rad_s)
{
  if (!cfg_.enable_motion_blur || cfg_.exposure_ms <= 0.0) return;

  const double fx =
    !cfg_.true_camera_matrix.empty() ? cfg_.true_camera_matrix.at<double>(0, 0) : 1300.0;
  const double fy =
    !cfg_.true_camera_matrix.empty() ? cfg_.true_camera_matrix.at<double>(1, 1) : 1300.0;
  const double exp_s = cfg_.exposure_ms * 1e-3;

  // 光学系 (+X 右, +Y 下, +Z 前): 相机绕 Y 轴正转(右转)时静止景物向左(-u)移动；
  // 绕 X 轴正转(抬头)时静止景物向下(+v)移动。
  double du = -fx * omega_cam_rad_s.y() * exp_s;
  double dv = +fy * omega_cam_rad_s.x() * exp_s;
  const double len = std::hypot(du, dv);
  if (len < 0.75) return;

  if (len > cfg_.max_motion_blur_px && cfg_.max_motion_blur_px > 0.0) {
    const double s = cfg_.max_motion_blur_px / len;
    du *= s;
    dv *= s;
  }

  const int taps = std::clamp(static_cast<int>(std::ceil(std::hypot(du, dv))) | 1, 3, 9);
  int off_x[9];
  int off_y[9];
  int min_ox = 0, max_ox = 0;
  for (int k = 0; k < taps; ++k) {
    const double t = -0.5 + static_cast<double>(k) / static_cast<double>(taps - 1);
    off_x[k] = static_cast<int>(std::lround(t * du));
    off_y[k] = static_cast<int>(std::lround(t * dv));
    min_ox = std::min(min_ox, off_x[k]);
    max_ox = std::max(max_ox, off_x[k]);
  }

  motion_tmp_buf_.create(img.size(), CV_8UC3);
  const int W = img.cols;
  const int H = img.rows;
  const int inv_taps_q16 = (65536 + taps / 2) / taps;
  // 安全无越界区：对所有 k，0 <= x - off_x[k] < W
  const int x_safe_lo = std::clamp(max_ox, 0, W);
  const int x_safe_hi = std::clamp(W + min_ox, x_safe_lo, W);
  const int safe_bytes = (x_safe_hi - x_safe_lo) * 3;

  cv::parallel_for_(cv::Range(0, H), [&](const cv::Range & range) {
    const uint8_t * rows[9];
    const uint8_t * safe_ptrs[9];
    for (int y = range.start; y < range.end; ++y) {
      auto * dst = motion_tmp_buf_.ptr<uint8_t>(y);
      for (int k = 0; k < taps; ++k) {
        const int sy = std::clamp(y - off_y[k], 0, H - 1);
        rows[k] = img.ptr<uint8_t>(sy);
        safe_ptrs[k] = rows[k] + (x_safe_lo - off_x[k]) * 3;
      }

      // 左边界
      for (int x = 0; x < x_safe_lo; ++x) {
        int b = 0, g = 0, r = 0;
        for (int k = 0; k < taps; ++k) {
          const int sx = std::clamp(x - off_x[k], 0, W - 1);
          const auto * p = rows[k] + sx * 3;
          b += p[0];
          g += p[1];
          r += p[2];
        }
        dst[x * 3 + 0] = static_cast<uint8_t>((b * inv_taps_q16 + 32768) >> 16);
        dst[x * 3 + 1] = static_cast<uint8_t>((g * inv_taps_q16 + 32768) >> 16);
        dst[x * 3 + 2] = static_cast<uint8_t>((r * inv_taps_q16 + 32768) >> 16);
      }

      // 中间无越界段：按连续字节累加，便于编译器自动向量化
      auto * dst_mid = dst + x_safe_lo * 3;
      for (int i = 0; i < safe_bytes; ++i) {
        int acc = 0;
        for (int k = 0; k < taps; ++k) acc += safe_ptrs[k][i];
        dst_mid[i] = static_cast<uint8_t>((acc * inv_taps_q16 + 32768) >> 16);
      }

      // 右边界
      for (int x = x_safe_hi; x < W; ++x) {
        int b = 0, g = 0, r = 0;
        for (int k = 0; k < taps; ++k) {
          const int sx = std::clamp(x - off_x[k], 0, W - 1);
          const auto * p = rows[k] + sx * 3;
          b += p[0];
          g += p[1];
          r += p[2];
        }
        dst[x * 3 + 0] = static_cast<uint8_t>((b * inv_taps_q16 + 32768) >> 16);
        dst[x * 3 + 1] = static_cast<uint8_t>((g * inv_taps_q16 + 32768) >> 16);
        dst[x * 3 + 2] = static_cast<uint8_t>((r * inv_taps_q16 + 32768) >> 16);
      }
    }
  });
  cv::swap(img, motion_tmp_buf_);
}

void SimCameraModel::apply_fused_postprocess(cv::Mat & img)
{
  const bool has_vig = !vignette_u8_.empty();
  const bool has_lut = use_photometric_lut_;
  if (!has_vig && !has_lut && !has_noise_) return;

  constexpr int kPad = 128;
  int ox = 0, oy = 0;
  if (has_noise_) {
    const int pool_w = img.cols + kPad;
    const int pool_h = img.rows + kPad;
    if (noise_pool_q5_.cols < pool_w || noise_pool_q5_.rows < pool_h) {
      cv::Mat tmp(pool_h, pool_w, CV_32FC3);
      cv_rng_.fill(tmp, cv::RNG::NORMAL, 0.0f, 32.0f);
      tmp.convertTo(noise_pool_q5_, CV_8SC3);
    }
    std::uniform_int_distribution<int> dist_x(0, kPad);
    std::uniform_int_distribution<int> dist_y(0, kPad);
    ox = dist_x(rng_);
    oy = dist_y(rng_);
  }

  const int W = img.cols;
  const int H = img.rows;
  cv::parallel_for_(cv::Range(0, H), [&](const cv::Range & range) {
    for (int y = range.start; y < range.end; ++y) {
      auto * row = img.ptr<uint8_t>(y);
      const uint8_t * vig = has_vig ? vignette_u8_.ptr<uint8_t>(y) : nullptr;
      const int8_t * n_row =
        has_noise_ ? (noise_pool_q5_.ptr<int8_t>(y + oy) + ox * 3) : nullptr;
      for (int x = 0; x < W; ++x) {
        const int v_gain = vig ? static_cast<int>(vig[x]) : 255;
        for (int c = 0; c < 3; ++c) {
          int v = row[x * 3 + c];
          if (vig) v = ((v * v_gain + 128) * 257) >> 16;
          if (has_lut) v = photometric_lut_[v];
          if (n_row) {
            // noise_q5 * std_q8 >> 13 近似 N(0, sigma)
            const int delta = (static_cast<int>(n_row[x * 3 + c]) * noise_std_q8_[v] + 4096) >> 13;
            v = std::clamp(v + delta, 0, 255);
          }
          row[x * 3 + c] = static_cast<uint8_t>(v);
        }
      }
    }
  });
}

void SimCameraModel::apply(
  const cv::Mat & src_bgr, cv::Mat & dst_bgr, const Eigen::Vector3d & omega_cam_rad_s)
{
  if (!cfg_.enabled || src_bgr.empty()) {
    dst_bgr = src_bgr;
    return;
  }

  ensure_maps(src_bgr.size());

  // 1. 几何重映射（内参 + 畸变 + 安装偏角）
  // 当 dst_bgr 与 src_bgr 不同缓冲时直接写入 dst_bgr，便于调用方做零拷贝双缓冲交换
  if (!map1_.empty()) {
    if (&dst_bgr != &src_bgr) {
      cv::remap(src_bgr, dst_bgr, map1_, map2_, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    } else {
      cv::remap(src_bgr, remap_buf_, map1_, map2_, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
      dst_bgr = remap_buf_;
    }
  } else {
    if (&dst_bgr != &src_bgr) {
      src_bgr.copyTo(dst_bgr);
    } else {
      src_bgr.copyTo(remap_buf_);
      dst_bgr = remap_buf_;
    }
  }

  // 2. 光学链路：运动模糊 -> 镜头 MTF 模糊 -> 单遍融合(暗角 + 曝光/伽马 + 泊松-高斯噪声)
  apply_motion_blur(dst_bgr, omega_cam_rad_s);

  if (cfg_.gaussian_blur_sigma > 0.05) {
    const int k = std::clamp(
      (static_cast<int>(std::ceil(cfg_.gaussian_blur_sigma * 2.2)) * 2 + 1), 3, 7);
    cv::GaussianBlur(dst_bgr, dst_bgr, cv::Size(k, k), cfg_.gaussian_blur_sigma);
  }

  apply_fused_postprocess(dst_bgr);
}

std::chrono::steady_clock::time_point SimCameraModel::adjust_timestamp(
  std::chrono::steady_clock::time_point ts)
{
  if (!cfg_.enabled) return ts;
  double delta_ms = cfg_.timestamp_offset_ms;
  if (cfg_.timestamp_jitter_ms > 1e-4) {
    std::normal_distribution<double> dist(0.0, cfg_.timestamp_jitter_ms);
    delta_ms += dist(rng_);
  }
  if (std::abs(delta_ms) < 1e-6) return ts;
  const auto delta_us = static_cast<int64_t>(std::lround(delta_ms * 1000.0));
  return ts + std::chrono::microseconds(delta_us);
}

std::chrono::microseconds SimCameraModel::target_frame_period() const
{
  if (!cfg_.enabled || cfg_.target_fps <= 0.0) return std::chrono::microseconds{0};
  return std::chrono::microseconds(static_cast<int64_t>(std::llround(1e6 / cfg_.target_fps)));
}

}  // namespace io
