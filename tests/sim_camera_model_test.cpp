#include <fmt/core.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "io/sim/sim_camera_model.hpp"

namespace
{

cv::Mat make_k(double fx, double fy, double cx, double cy)
{
  return (cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
}

cv::Mat make_d(double k1, double k2, double p1, double p2, double k3)
{
  return (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, k3);
}

// 在图像上画一个亚像素高斯亮斑，便于用质心法验证重映射亚像素精度
void draw_subpixel_blob(cv::Mat & img, double cx, double cy, double sigma = 2.0)
{
  const int r = static_cast<int>(std::ceil(4.0 * sigma));
  const int x0 = std::max(0, static_cast<int>(std::floor(cx)) - r);
  const int x1 = std::min(img.cols - 1, static_cast<int>(std::ceil(cx)) + r);
  const int y0 = std::max(0, static_cast<int>(std::floor(cy)) - r);
  const int y1 = std::min(img.rows - 1, static_cast<int>(std::ceil(cy)) + r);
  const double inv2s2 = 0.5 / (sigma * sigma);
  for (int y = y0; y <= y1; ++y) {
    auto * row = img.ptr<cv::Vec3b>(y);
    for (int x = x0; x <= x1; ++x) {
      const double d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
      const uint8_t v = cv::saturate_cast<uint8_t>(std::lround(255.0 * std::exp(-d2 * inv2s2)));
      row[x] = cv::Vec3b(std::max(row[x][0], v), std::max(row[x][1], v), std::max(row[x][2], v));
    }
  }
}

cv::Point2d find_local_centroid(const cv::Mat & img, double est_x, double est_y, int win = 10)
{
  const int x0 = std::max(0, static_cast<int>(std::lround(est_x)) - win);
  const int x1 = std::min(img.cols - 1, static_cast<int>(std::lround(est_x)) + win);
  const int y0 = std::max(0, static_cast<int>(std::lround(est_y)) - win);
  const int y1 = std::min(img.rows - 1, static_cast<int>(std::lround(est_y)) + win);
  double sw = 0.0, sx = 0.0, sy = 0.0;
  for (int y = y0; y <= y1; ++y) {
    const auto * row = img.ptr<cv::Vec3b>(y);
    for (int x = x0; x <= x1; ++x) {
      const double w = row[x][0];
      if (w > 20.0) {
        sw += w;
        sx += w * x;
        sy += w * y;
      }
    }
  }
  return sw > 0.0 ? cv::Point2d(sx / sw, sy / sw) : cv::Point2d(-1.0, -1.0);
}

bool test_geometry_reprojection(const Eigen::Vector3d & R_perturb_deg, const char * tag)
{
  const int W = 1440, H = 1080;
  const cv::Mat K_sim = make_k(1304.673, 1304.673, 720.0, 540.0);
  const cv::Mat K_real = make_k(1826.66492172, 1829.56362176, 728.81993781, 572.91422500);
  const cv::Mat D_real =
    make_d(-0.0463416026, 0.0444278789, -0.0000845216, 0.00193664628, -0.112060289);

  io::SimCameraModel::Config cfg;
  cfg.enabled = true;
  cfg.enable_geometry = true;
  cfg.sim_camera_matrix = K_sim;
  cfg.true_camera_matrix = K_real;
  cfg.true_distort_coeffs = D_real;
  cfg.R_perturb_deg = R_perturb_deg;

  io::SimCameraModel model(cfg);

  // 在 Z=2.5m 平面生成 3D 点阵
  std::vector<cv::Point3f> pts_3d;
  for (int iy = -3; iy <= 3; ++iy) {
    for (int ix = -4; ix <= 4; ++ix) {
      pts_3d.emplace_back(ix * 0.16f, iy * 0.14f, 2.5f);
    }
  }

  // 1) 在理想仿真相机上投影并绘制
  std::vector<cv::Point2f> pts_sim;
  cv::projectPoints(
    pts_3d, cv::Mat::zeros(3, 1, CV_64F), cv::Mat::zeros(3, 1, CV_64F), K_sim,
    cv::Mat::zeros(1, 5, CV_64F), pts_sim);

  cv::Mat sim_img = cv::Mat::zeros(H, W, CV_8UC3);
  for (const auto & p : pts_sim) draw_subpixel_blob(sim_img, p.x, p.y, 2.5);

  // 2) 通过 SimCameraModel 重映射到实机视角
  cv::Mat real_img;
  model.apply(sim_img, real_img);

  // 3) 用 OpenCV projectPoints 计算真实相机理论投影位置
  // p_real = R_real2sim^T * p_sim
  const Eigen::Matrix3d R_sim2real = io::SimCameraModel::euler_deg_to_rot(R_perturb_deg).transpose();
  cv::Mat R_cv(3, 3, CV_64F);
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) R_cv.at<double>(r, c) = R_sim2real(r, c);
  cv::Mat rvec;
  cv::Rodrigues(R_cv, rvec);

  std::vector<cv::Point2f> pts_real_gt;
  cv::projectPoints(pts_3d, rvec, cv::Mat::zeros(3, 1, CV_64F), K_real, D_real, pts_real_gt);

  double max_err = 0.0, sum_err = 0.0;
  int checked = 0;
  for (const auto & gt : pts_real_gt) {
    if (gt.x < 25 || gt.x > W - 25 || gt.y < 25 || gt.y > H - 25) continue;
    const auto c = find_local_centroid(real_img, gt.x, gt.y, 12);
    const double err = std::hypot(c.x - gt.x, c.y - gt.y);
    max_err = std::max(max_err, err);
    sum_err += err;
    ++checked;
  }
  const double mean_err = sum_err / std::max(1, checked);

  fmt::print(
    "[{}] checked={} valid_ratio={:.4f} mean_err={:.3f}px max_err={:.3f}px\n", tag, checked,
    model.valid_pixel_ratio(), mean_err, max_err);
  return checked >= 20 && model.valid_pixel_ratio() > 0.999 && mean_err < 0.25 && max_err < 0.6;
}

bool test_sensor_degradations()
{
  const int W = 640, H = 480;
  const cv::Mat K = make_k(1000.0, 1000.0, 320.0, 240.0);

  // 1) 运动模糊：竖直白线在绕 Y 轴旋转时应水平展宽
  {
    io::SimCameraModel::Config cfg;
    cfg.enabled = true;
    cfg.enable_geometry = false;
    cfg.true_camera_matrix = K;
    cfg.enable_motion_blur = true;
    cfg.exposure_ms = 5.0;  // 5ms * 1000px * 2 rad/s = 10 px
    io::SimCameraModel model(cfg);

    cv::Mat src = cv::Mat::zeros(H, W, CV_8UC3);
    src.col(W / 2).setTo(cv::Scalar(255, 255, 255));
    cv::Mat dst;
    model.apply(src, dst, Eigen::Vector3d(0.0, 2.0, 0.0));

    int lit_cols = 0;
    for (int x = W / 2 - 20; x <= W / 2 + 20; ++x) {
      if (dst.at<cv::Vec3b>(H / 2, x)[0] > 10) ++lit_cols;
    }
    fmt::print("[motion_blur] expected~11px, lit_cols={}\n", lit_cols);
    if (lit_cols < 8 || lit_cols > 15) return false;
  }

  // 2) 暗角 + 光度 LUT
  {
    io::SimCameraModel::Config cfg;
    cfg.enabled = true;
    cfg.enable_geometry = false;
    cfg.true_camera_matrix = K;
    cfg.vignette_strength = 0.25;
    cfg.brightness_scale = 1.5;
    cfg.gamma = 1.0;
    io::SimCameraModel model(cfg);

    cv::Mat src(H, W, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat dst;
    model.apply(src, dst);
    const double center_val = dst.at<cv::Vec3b>(H / 2, W / 2)[0];
    const double corner_val = dst.at<cv::Vec3b>(0, 0)[0];
    fmt::print("[vignette+gain] center={:.1f} (exp ~150), corner={:.1f} (exp ~112.5)\n", center_val, corner_val);
    if (std::abs(center_val - 150.0) > 2.0 || std::abs(corner_val - 112.5) > 3.0) return false;
  }

  // 3) 泊松-高斯噪声统计校验
  {
    io::SimCameraModel::Config cfg;
    cfg.enabled = true;
    cfg.enable_geometry = false;
    cfg.true_camera_matrix = K;
    cfg.read_noise_std = 2.0;
    cfg.shot_noise_k = 0.05;  // at I=100: std = sqrt(4 + 5) = 3.0
    cfg.seed = 42;
    io::SimCameraModel model(cfg);

    cv::Mat src(H, W, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat dst;
    model.apply(src, dst);
    cv::Scalar mean, stddev;
    cv::meanStdDev(dst, mean, stddev);
    fmt::print("[poisson_gaussian_noise] mean={:.2f}, std={:.3f} (exp 3.00)\n", mean[0], stddev[0]);
    if (std::abs(mean[0] - 100.0) > 0.5 || std::abs(stddev[0] - 3.0) > 0.25) return false;
  }

  // 4) 时序偏移与抖动
  {
    io::SimCameraModel::Config cfg;
    cfg.enabled = true;
    cfg.timestamp_offset_ms = -2.5;
    cfg.timestamp_jitter_ms = 0.5;
    cfg.seed = 123;
    io::SimCameraModel model(cfg);

    const auto base = std::chrono::steady_clock::now();
    double sum = 0.0, sum2 = 0.0;
    const int N = 2000;
    for (int i = 0; i < N; ++i) {
      const auto t = model.adjust_timestamp(base);
      const double d_ms =
        std::chrono::duration<double, std::milli>(t - base).count();
      sum += d_ms;
      sum2 += d_ms * d_ms;
    }
    const double mean_ms = sum / N;
    const double std_ms = std::sqrt(std::max(0.0, sum2 / N - mean_ms * mean_ms));
    fmt::print("[timing] mean={:.3f}ms (exp -2.5), std={:.3f}ms (exp 0.5)\n", mean_ms, std_ms);
    if (std::abs(mean_ms + 2.5) > 0.1 || std::abs(std_ms - 0.5) > 0.1) return false;
  }

  return true;
}

bool test_runtime_benchmark()
{
  const int W = 1440, H = 1080;
  io::SimCameraModel::Config cfg;
  cfg.enabled = true;
  cfg.enable_geometry = true;
  cfg.sim_camera_matrix = make_k(1304.673, 1304.673, 720.0, 540.0);
  cfg.true_camera_matrix = make_k(1826.665, 1829.564, 728.820, 572.914);
  cfg.true_distort_coeffs = make_d(-0.0463, 0.0444, -8.45e-5, 1.94e-3, -0.112);
  cfg.enable_motion_blur = true;
  cfg.exposure_ms = 4.0;
  cfg.gaussian_blur_sigma = 0.6;
  cfg.vignette_strength = 0.12;
  cfg.brightness_scale = 1.15;
  cfg.gamma = 0.95;
  cfg.read_noise_std = 1.5;
  cfg.shot_noise_k = 0.04;
  cfg.seed = 7;

  io::SimCameraModel model(cfg);
  cv::Mat src(H, W, CV_8UC3, cv::Scalar(80, 120, 160));
  cv::Mat dst;
  model.apply(src, dst, Eigen::Vector3d(0.0, 1.5, 0.0));  // warmup

  const int iters = 30;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    model.apply(src, dst, Eigen::Vector3d(0.0, 1.5, 0.0));
  }
  const double ms =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() /
    iters;
  fmt::print("[benchmark 1440x1080 full pipeline] {:.2f} ms/frame\n", ms);
  return ms < 50.0;
}

}  // namespace

int main()
{
  bool ok = true;
  ok &= test_geometry_reprojection(Eigen::Vector3d(0.0, 0.0, 0.0), "geom_standard3");
  ok &= test_geometry_reprojection(Eigen::Vector3d(1.5, -2.0, 0.8), "geom_R_perturb");
  ok &= test_sensor_degradations();
  ok &= test_runtime_benchmark();

  if (!ok) {
    std::cerr << "sim_camera_model_test FAILED\n";
    return 1;
  }
  std::cout << "sim_camera_model_test PASSED\n";
  return 0;
}
