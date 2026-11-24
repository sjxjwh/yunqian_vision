#ifndef IO__SIM__SIM_CAMERA_MODEL_HPP
#define IO__SIM__SIM_CAMERA_MODEL_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <chrono>
#include <cstdint>
#include <opencv2/core.hpp>
#include <random>
#include <string>

namespace io
{

// 仿真相机退化与标定复现模型：
// 在不修改只读仿真器的前提下，把仿真理想针孔图重映射到实机标定内参/畸变/安装偏角，
// 并按开关叠加运动模糊、镜头 MTF 模糊、暗角、曝光/伽马/过曝、泊松-高斯传感器噪声及帧率/时序抖动。
class SimCameraModel
{
public:
  struct Config
  {
    bool enabled = false;

    // --- 几何成像（仿真理想针孔 -> 实机标定相机） ---
    bool enable_geometry = true;
    int output_width = 0;   // 0 = 沿用输入图宽
    int output_height = 0;  // 0 = 沿用输入图高
    cv::Mat sim_camera_matrix;    // 3x3 CV_64F，仿真原始针孔内参
    cv::Mat true_camera_matrix;   // 3x3 CV_64F，复现目标（实机真值）内参
    cv::Mat true_distort_coeffs;  // 1x5 CV_64F，复现目标（实机真值）畸变 [k1,k2,p1,p2,k3]
    // 相机光学系安装旋转扰动 [roll, pitch, yaw] (deg)，按 Rz(yaw)*Ry(pitch)*Rx(roll)
    // 定义为 R_real2sim（真实相机光轴相对仿真理想相机的旋转）。
    Eigen::Vector3d R_perturb_deg{0.0, 0.0, 0.0};

    // --- 运动模糊 ---
    bool enable_motion_blur = false;
    double exposure_ms = 4.0;          // 实机曝光时间 (ms)
    double max_motion_blur_px = 35.0;  // 防止异常跳帧导致过大模糊核

    // --- 镜头 MTF / 离焦模糊 ---
    double gaussian_blur_sigma = 0.0;  // px，0 = 关闭

    // --- 暗角 ---
    double vignette_strength = 0.0;  // [0, 1)，边角亮度衰减比例

    // --- 光度 / 曝光 / 伽马 ---
    double brightness_scale = 1.0;    // 线性亮度增益，>1 会触发高亮饱和截断
    double gamma = 1.0;               // out = 255 * (in/255)^gamma，<1 提亮暗部，>1 压暗
    double black_level_offset = 0.0;  // 暗电流黑电平偏移 (DN)

    // --- 传感器噪声（泊松-高斯：Var(I) = read_noise_std^2 + shot_noise_k * I） ---
    double read_noise_std = 0.0;  // 读出噪声标准差 (DN)
    double shot_noise_k = 0.0;    // 散粒噪声系数 (DN)

    // --- 时序与帧率 ---
    double target_fps = 0.0;           // 0 = 不限帧率；如 100.0 对齐海康相机
    double timestamp_offset_ms = 0.0;  // 固定时间同步偏差 (ms)
    double timestamp_jitter_ms = 0.0;  // 高斯时间戳抖动标准差 (ms)

    uint32_t seed = 0;  // 0 = 随机种子，非 0 = 可复现噪声/抖动
  };

  SimCameraModel() = default;
  explicit SimCameraModel(const Config & cfg);

  // 从 configs/sim.yaml 读取 sim_camera_model 段；缺省时返回 disabled 模型。
  static SimCameraModel from_yaml(const std::string & config_path);

  bool enabled() const { return cfg_.enabled; }
  const Config & config() const { return cfg_; }
  double valid_pixel_ratio() const { return valid_pixel_ratio_; }

  // 对单帧仿真 BGR 图执行几何重映射 + 光学/传感器退化。
  // omega_cam_rad_s: 当前帧相机光学系下的角速度 (rad/s)，用于计算运动模糊位移。
  void apply(
    const cv::Mat & src_bgr, cv::Mat & dst_bgr,
    const Eigen::Vector3d & omega_cam_rad_s = Eigen::Vector3d::Zero());

  // 按配置注入时间同步偏差与随机抖动。
  std::chrono::steady_clock::time_point adjust_timestamp(
    std::chrono::steady_clock::time_point ts);

  // 目标帧间隔；未启用限帧时返回 0。
  std::chrono::microseconds target_frame_period() const;

  // 工具函数：把 [roll, pitch, yaw] (deg) 转成 3x3 旋转矩阵 Rz*Ry*Rx。
  static Eigen::Matrix3d euler_deg_to_rot(const Eigen::Vector3d & rpy_deg);

private:
  void init_photometric_lut();
  void ensure_maps(const cv::Size & src_size);
  void apply_motion_blur(cv::Mat & img, const Eigen::Vector3d & omega_cam_rad_s);
  void apply_fused_postprocess(cv::Mat & img);

  Config cfg_{};
  cv::Size cached_src_size_{0, 0};
  cv::Size out_size_{0, 0};

  // 预计算几何 remap 表（CV_16SC2 + CV_16UC1 定点表加速）
  cv::Mat map1_;
  cv::Mat map2_;
  double valid_pixel_ratio_ = 1.0;

  // 预计算单通道暗角掩膜（CV_8UC1）、光度 LUT 与定点数噪声查找表
  cv::Mat vignette_u8_;
  std::array<uint8_t, 256> photometric_lut_{};
  bool use_photometric_lut_ = false;
  std::array<int32_t, 256> noise_std_q8_{};  // sqrt(read^2 + shot_k * I) * 256
  bool has_noise_ = false;

  // 复用缓冲区，避免逐帧内存分配
  cv::Mat remap_buf_;
  cv::Mat motion_tmp_buf_;
  cv::Mat noise_pool_q5_;  // CV_8SC3, N(0,1)*32 预生成标准正态噪声池

  cv::RNG cv_rng_;
  std::mt19937 rng_;
};

}  // namespace io

#endif  // IO__SIM__SIM_CAMERA_MODEL_HPP
