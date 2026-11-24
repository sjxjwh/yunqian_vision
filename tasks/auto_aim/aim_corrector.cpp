#include "aim_corrector.hpp"

#include <fmt/format.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/runtime_params.hpp"
#include "tools/trajectory.hpp"

namespace auto_aim
{
namespace
{
constexpr double R2D = 180.0 / M_PI;

double median(std::vector<double> v)
{
  if (v.empty()) return 0;
  const size_t m = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + m, v.end());
  double hi = v[m];
  if (v.size() % 2) return hi;
  return 0.5 * (hi + *std::max_element(v.begin(), v.begin() + m));
}

// 1.4826 * MAD ≈ 正态分布的 σ
double robust_sigma(const std::vector<double> & v, double med)
{
  if (v.size() < 2) return 0;
  std::vector<double> d;
  d.reserve(v.size());
  for (double x : v) d.push_back(std::abs(x - med));
  return 1.4826 * median(d);
}

double yaml_or(const YAML::Node & y, const char * k, double def)
{
  return y[k] ? y[k].as<double>() : def;
}

double seconds(AimCorrector::Clock::duration d) { return std::chrono::duration<double>(d).count(); }
}  // namespace

// ============================== BulletDetector ==============================

BulletDetector::BulletDetector()
{
  auto & rp = tools::RuntimeParams::instance();
  rp.add(
    "bullet.h_lo", "弹丸 H 下限 (OpenCV 0~180)", 0, 180, [this] { return h_lo; },
    [this](double v) { h_lo = v; }, "bullet_h_lo", 1);
  rp.add(
    "bullet.h_hi", "弹丸 H 上限", 0, 180, [this] { return h_hi; },
    [this](double v) { h_hi = v; }, "bullet_h_hi", 1);
  rp.add(
    "bullet.s_min", "弹丸 S 下限", 0, 255, [this] { return s_min; },
    [this](double v) { s_min = v; }, "bullet_s_min", 1);
  rp.add(
    "bullet.v_min", "弹丸 V 下限", 0, 255, [this] { return v_min; },
    [this](double v) { v_min = v; }, "bullet_v_min", 1);
  rp.add(
    "bullet.diff", "帧差阈值（越小越敏感）", 0, 255, [this] { return diff_thresh; },
    [this](double v) { diff_thresh = v; }, "bullet_diff", 1);
  rp.add(
    "bullet.min_area", "最小面积 (px)", 1, 500, [this] { return min_area; },
    [this](double v) { min_area = v; }, "bullet_min_area", 1);
  rp.add(
    "bullet.max_area", "最大面积 (px)", 10, 20000, [this] { return max_area; },
    [this](double v) { max_area = v; }, "bullet_max_area", 1);
  rp.add(
    "bullet.min_fill", "最小圆度（面积/外接圆）", 0, 1, [this] { return min_fill; },
    [this](double v) { min_fill = v; }, "bullet_min_fill");
}

std::vector<BulletBlob> BulletDetector::feed(
  const cv::Mat & bgr, const Eigen::Matrix3d & R_c2w, const cv::Mat & K,
  const std::vector<cv::Rect> & rois, const std::vector<BulletBlob> & keep)
{
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<BulletBlob> out;

  if (has_prev_ && !rois.empty() && prev_.size() == bgr.size()) {
    Eigen::Matrix3d Kd;
    cv::cv2eigen(K, Kd);
    // 上一帧像素 → 当前帧像素（纯旋转单应；相机平移相对背景距离可忽略）
    const Eigen::Matrix3d H = Kd * (R_c2w.transpose() * prev_R_) * Kd.inverse();
    const cv::Rect full(0, 0, bgr.cols, bgr.rows);
    const auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});

    for (const auto & roi0 : rois) {
      const cv::Rect roi = roi0 & full;
      if (roi.width < 4 || roi.height < 4) continue;

      Eigen::Matrix3d T;
      T << 1, 0, -roi.x, 0, 1, -roi.y, 0, 0, 1;
      // 只把上一帧里会被采样到的那一小块喂给 warpPerspective（外扩 3 px 保双线性邻域一致）
      const Eigen::Matrix3d M = T * H;  // 上一帧像素 → ROI 像素
      const Eigen::Matrix3d Mi = M.inverse();
      double x0 = 1e18, y0 = 1e18, x1 = -1e18, y1 = -1e18;
      for (const auto & c : {Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(roi.width, 0, 1),
                             Eigen::Vector3d(0, roi.height, 1),
                             Eigen::Vector3d(roi.width, roi.height, 1)}) {
        const Eigen::Vector3d q = Mi * c;
        x0 = std::min(x0, q.x() / q.z()), x1 = std::max(x1, q.x() / q.z());
        y0 = std::min(y0, q.y() / q.z()), y1 = std::max(y1, q.y() / q.z());
      }
      cv::Rect src_box(int(std::floor(x0)) - 3, int(std::floor(y0)) - 3, 0, 0);
      src_box.width = int(std::ceil(x1)) + 4 - src_box.x;
      src_box.height = int(std::ceil(y1)) + 4 - src_box.y;
      src_box &= full;
      Eigen::Matrix3d S;
      S << 1, 0, src_box.x, 0, 1, src_box.y, 0, 0, 1;
      cv::Mat Hcv;
      cv::Mat warped;
      if (src_box.width > 0 && src_box.height > 0) {
        cv::eigen2cv(Eigen::Matrix3d(M * S), Hcv);
        cv::warpPerspective(
          prev_(src_box), warped, Hcv, roi.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
      } else {
        cv::eigen2cv(M, Hcv);
        cv::warpPerspective(prev_, warped, Hcv, roi.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
      }

      const cv::Mat cur = bgr(roi);
      cv::Mat diff, ch[3], dmax;
      cv::absdiff(cur, warped, diff);
      cv::split(diff, ch);
      cv::max(ch[0], ch[1], dmax);
      cv::max(dmax, ch[2], dmax);
      cv::Mat mask = dmax > diff_thresh;
      for (const auto & b : keep) {
        const Eigen::Vector3d p = T * H * Eigen::Vector3d(b.center.x, b.center.y, 1);
        if (std::abs(p.z()) < 1e-9) continue;
        cv::circle(
          mask, cv::Point(int(p.x() / p.z()), int(p.y() / p.z())), int(b.radius + 3), 255, -1);
      }

      cv::Mat hsv, color;
      cv::cvtColor(cur, hsv, cv::COLOR_BGR2HSV);
      cv::inRange(hsv, cv::Scalar(h_lo, s_min, v_min), cv::Scalar(h_hi, 255, 255), color);
      mask &= color;
      cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
      for (const auto & c : contours) {
        const cv::Rect bb = cv::boundingRect(c);
        const cv::Mat sub = mask(bb);
        const double area = cv::countNonZero(sub);
        if (area < min_area || area > max_area) continue;
        cv::Point2f cc;
        float rr;
        cv::minEnclosingCircle(c, cc, rr);
        rr = std::max(rr, 0.5f);
        if (area / (M_PI * rr * rr) < min_fill) continue;
        const auto m = cv::moments(sub, true);
        if (m.m00 <= 0) continue;
        const cv::Point2f center(
          float(m.m10 / m.m00 + bb.x + roi.x), float(m.m01 / m.m00 + bb.y + roi.y));
        // ROI 有重叠时去重
        bool dup = false;
        for (const auto & o : out)
          if (cv::norm(o.center - center) < 3) dup = true;
        if (!dup) out.push_back({center, rr});
      }
    }
  }

  // 只留引用不拷贝；若驱动复用同一块缓冲区则自动退回拷贝模式
  if (!copy_mode_ && has_prev_ && bgr.data == prev_.data) {
    copy_mode_ = true;
    tools::logger()->warn(
      "[AimCorrector] 相机复用图像缓冲区，帧差改为拷贝模式（每帧多 ~0.5 ms）");
  }
  if (copy_mode_)
    bgr.copyTo(prev_);
  else
    prev_ = bgr;
  prev_R_ = R_c2w;
  has_prev_ = true;
  last_ms_ =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return out;
}

// ============================== AimCorrector ==============================

AimCorrector::AimCorrector(const std::string & config_path)
{
  auto y = YAML::LoadFile(config_path);
  fire_delay_ = yaml_or(y, "fire_delay", fire_delay_);
  muzzle_z_offset_ = yaml_or(y, "muzzle_z_offset", muzzle_z_offset_);
  gate_px_ = yaml_or(y, "corrector_gate_px", gate_px_);
  min_range_ = yaml_or(y, "corrector_min_range", min_range_);
  max_flight_ = yaml_or(y, "corrector_max_flight", max_flight_);
  min_samples_ = yaml_or(y, "corrector_min_samples", min_samples_);
  window_ = yaml_or(y, "corrector_window", window_);
  max_step_ = yaml_or(y, "corrector_max_step", max_step_);
  min_shots_ = yaml_or(y, "corrector_min_shots", min_shots_);
  sync_sigma_ = yaml_or(y, "corrector_sync_sigma", sync_sigma_);
  det_.h_lo = yaml_or(y, "bullet_h_lo", det_.h_lo);
  det_.h_hi = yaml_or(y, "bullet_h_hi", det_.h_hi);
  det_.s_min = yaml_or(y, "bullet_s_min", det_.s_min);
  det_.v_min = yaml_or(y, "bullet_v_min", det_.v_min);
  det_.diff_thresh = yaml_or(y, "bullet_diff", det_.diff_thresh);
  det_.min_area = yaml_or(y, "bullet_min_area", det_.min_area);
  det_.max_area = yaml_or(y, "bullet_max_area", det_.max_area);
  det_.min_fill = yaml_or(y, "bullet_min_fill", det_.min_fill);

  auto & rp = tools::RuntimeParams::instance();
  auto reg = [&](const char * k, const char * d, double lo, double hi, double & ref,
                 const char * yk, double step = 0) {
    rp.add(k, d, lo, hi, [&ref] { return ref; }, [&ref](double v) { ref = v; }, yk, step);
  };
  reg("corrector.fire_delay", "发射反馈→出膛延迟 (s，可为负)", -0.2, 0.3, fire_delay_, "fire_delay");
  reg("corrector.gate_px", "配对门限 (px)", 10, 400, gate_px_, "corrector_gate_px", 1);
  reg("corrector.min_range", "最近采样距离 (m)", 0, 5, min_range_, "corrector_min_range");
  reg("corrector.max_flight", "虚拟弹丸最长飞行 (s)", 0.1, 2, max_flight_, "corrector_max_flight");
  reg("corrector.min_samples", "每发最少观测帧数", 1, 20, min_samples_, "corrector_min_samples", 1);
  reg("corrector.window", "估计窗口（最近几发）", 1, 50, window_, "corrector_window", 1);
  reg("corrector.min_shots", "给出建议的最少发数", 1, 30, min_shots_, "corrector_min_shots", 1);
  reg("corrector.auto_apply", "自动写入 offset (0/1)", 0, 1, auto_apply_, "", 1);
  reg("corrector.sync_sigma", "自动同步实测散布至 Planner (0/1)", 0, 1, sync_sigma_, "corrector_sync_sigma", 1);
  reg("corrector.max_step", "自动写入每发最大改动 (度)", 0, 2, max_step_, "corrector_max_step");
  reg("corrector.detect_all", "全图检测弹丸（调 HSV 用，较耗 CPU）(0/1)", 0, 1, detect_all_, "", 1);
}

Eigen::Vector3d AimCorrector::sim_pos(const Shot & s, double t)
{
  double r, z;
  tools::ballistic_point(s.v0, s.pitch, s.k, t, r, z);
  return s.p0 + Eigen::Vector3d(r * std::cos(s.yaw), r * std::sin(s.yaw), z);
}

void AimCorrector::add_shot(
  Clock::time_point t_fire, const Eigen::Matrix3d & R_gimbal2world, double v0, double k,
  const Eigen::Vector3d & origin, double stop_range)
{
  // 云台 x 轴在世界系中的方向 = 枪管方向（与 Aimer/Planner 模型一致）
  const Eigen::Vector3d dir = R_gimbal2world.col(0);
  Shot s;
  s.id = 0;
  s.t_fire = t_fire;
  s.yaw = std::atan2(dir.y(), dir.x());
  s.pitch = std::atan2(dir.z(), std::hypot(dir.x(), dir.y()));
  s.v0 = v0;
  s.k = k;
  s.p0 = std::isnan(origin.z()) ? Eigen::Vector3d(0.0, 0.0, muzzle_z_offset_) : origin;
  s.stop_range = std::clamp(stop_range, 0.5, 25.0);
  std::lock_guard<std::mutex> lk(mtx_);
  s.id = next_id_++;
  ++shots_total_;
  shots_.push_back(std::move(s));
  while (shots_.size() > 20) shots_.pop_front();
}

bool AimCorrector::active() const
{
  std::lock_guard<std::mutex> lk(mtx_);
  return !shots_.empty() || detect_all_ > 0.5;
}

void AimCorrector::update(const cv::Mat & bgr, Clock::time_point t_img, const Solver & solver)
{
  std::lock_guard<std::mutex> lk(mtx_);
  img_size_ = bgr.size();

  if (shots_.empty() && detect_all_ < 0.5) {
    det_.reset();
    matched_prev_.clear();
    last_blobs_.clear();
    last_rois_.clear();
    return;
  }

  const cv::Rect full(0, 0, bgr.cols, bgr.rows);
  const Eigen::Vector3d cam_c = solver.camera_center_in_world();

  // ---- 1. 推进虚拟弹丸，生成 ROI ----
  struct Live
  {
    Shot * s;
    Eigen::Vector3d P;
    cv::Point2f px;
  };
  for (auto it = shots_.begin(); it != shots_.end();) {
    const double t = seconds(t_img - it->t_fire);
    const Eigen::Vector3d P = sim_pos(*it, std::max(0.0, t));
    const double flown = (P - it->p0).head<2>().norm();
    if (t > max_flight_ || P.z() < -2.0 || flown > it->stop_range) {
      finalize(*it);
      it = shots_.erase(it);
    } else {
      ++it;
    }
  }
  std::vector<Live> live;
  std::vector<cv::Rect> rois;
  for (auto & s : shots_) {
    const double t = seconds(t_img - s.t_fire);
    const Eigen::Vector3d P = sim_pos(s, std::max(0.0, t));
    s.has_match = false;
    cv::Point2f px;
    if (t >= 0 && (P - s.p0).norm() >= min_range_ && solver.world2pixel(P, px) && full.contains(px)) {
      live.push_back({&s, P, px});
      const int g = int(gate_px_);
      rois.emplace_back(int(px.x) - g, int(px.y) - g, 2 * g, 2 * g);
    }
  }
  if (detect_all_ > 0.5) rois = {full};
  last_rois_ = rois;

  // ---- 2. 检测 ----
  last_blobs_ =
    det_.feed(bgr, solver.R_camera2world(), solver.camera_matrix(), rois, matched_prev_);
  matched_prev_.clear();

  // ---- 3. 配对（每个检测只用一次，按距离贪心）----
  std::vector<bool> used(last_blobs_.size(), false);
  for (auto & l : live) {
    int best = -1;
    double best_d = gate_px_;
    for (size_t i = 0; i < last_blobs_.size(); ++i) {
      if (used[i]) continue;
      const double d = cv::norm(last_blobs_[i].center - l.px);
      if (d < best_d) best_d = d, best = int(i);
    }
    if (best < 0) continue;
    used[best] = true;
    matched_prev_.push_back(last_blobs_[best]);

    // 观测弹丸放到与虚拟弹丸相同的相机距离上，比较“从出膛点 p0 看过去”的角度
    const Eigen::Vector3d dir = solver.pixel2world_dir(last_blobs_[best].center);
    const Eigen::Vector3d O = cam_c + (l.P - cam_c).norm() * dir;
    const Eigen::Vector3d dO = O - l.s->p0;
    const Eigen::Vector3d dP = l.P - l.s->p0;
    const double ey = tools::limit_rad(std::atan2(dO.y(), dO.x()) - std::atan2(dP.y(), dP.x()));
    const double ep =
      std::atan2(dO.z(), dO.head<2>().norm()) - std::atan2(dP.z(), dP.head<2>().norm());
    l.s->errs.emplace_back(ey, ep);
    l.s->obs_px.push_back(last_blobs_[best].center);
    l.s->max_range = std::max(l.s->max_range, dP.norm());
    l.s->has_match = true;
    l.s->match_obs = last_blobs_[best].center;
    l.s->match_sim = l.px;
  }
}

void AimCorrector::finalize(const Shot & s)
{
  if ((int)s.errs.size() < std::max(1, int(min_samples_))) return;
  std::vector<double> ys, ps;
  for (const auto & e : s.errs) ys.push_back(e[0]), ps.push_back(e[1]);
  results_.push_back({s.id, median(ys) * R2D, median(ps) * R2D, int(s.errs.size()), s.max_range});
  while (results_.size() > 50) results_.pop_front();
  ++shots_matched_;
  maybe_auto_apply();
  maybe_sync_sigma();
}

bool AimCorrector::estimate(double & ey, double & ep, double & sy, double & sp, int & n) const
{
  const size_t w = std::max<size_t>(1, size_t(window_));
  std::vector<double> ys, ps;
  for (auto it = results_.rbegin(); it != results_.rend() && ys.size() < w; ++it)
    ys.push_back(it->ey), ps.push_back(it->ep);
  n = int(ys.size());
  if (ys.empty()) return false;
  ey = median(ys);
  ep = median(ps);
  sy = robust_sigma(ys, ey);
  sp = robust_sigma(ps, ep);
  return n >= int(min_shots_);
}

void AimCorrector::maybe_auto_apply()
{
  if (auto_apply_ < 0.5) return;
  double ey, ep, sy, sp;
  int n;
  if (!estimate(ey, ep, sy, sp, n)) return;
  auto & rp = tools::RuntimeParams::instance();
  auto step = [&](const char * key, double target) {
    const auto cur = rp.get(key);
    if (!cur) return;
    const double d = std::clamp(target - *cur, -max_step_, max_step_);
    rp.set(key, *cur + d);
  };
  step("aimer.yaw_offset", -ey);
  step("aimer.pitch_offset", -ep);
  step("planner.yaw_offset", -ey);
  step("planner.pitch_offset", -ep);
}

void AimCorrector::maybe_sync_sigma()
{
  if (sync_sigma_ < 0.5) return;
  double ey, ep, sy, sp;
  int n;
  if (!estimate(ey, ep, sy, sp, n)) return;
  auto & rp = tools::RuntimeParams::instance();
  auto ema_update = [&](const char * key, double meas_deg) {
    const auto cur = rp.get(key);
    if (!cur) return;
    const double clamped = std::clamp(meas_deg, 0.05, 3.0);
    // 一阶指数滑动平均，既能响应实车机械/弹速散布变化，又避免小样本单发跳变
    constexpr double ALPHA = 0.30;
    rp.set(key, (1.0 - ALPHA) * (*cur) + ALPHA * clamped);
  };
  ema_update("planner.gun_sigma_yaw", sy);
  ema_update("planner.gun_sigma_pitch", sp);
}

bool AimCorrector::apply_suggestion(std::string & msg)
{
  std::lock_guard<std::mutex> lk(mtx_);
  double ey, ep, sy, sp;
  int n;
  if (!estimate(ey, ep, sy, sp, n)) {
    msg = fmt::format("样本不够：{} / {} 发", n, int(min_shots_));
    return false;
  }
  auto & rp = tools::RuntimeParams::instance();
  const bool set_aimer = rp.set("aimer.yaw_offset", -ey) | rp.set("aimer.pitch_offset", -ep);
  const bool set_planner = rp.set("planner.yaw_offset", -ey) | rp.set("planner.pitch_offset", -ep);
  const double cur_y = rp.get("planner.yaw_offset")
                         .value_or(rp.get("aimer.yaw_offset").value_or(-ey));
  const double cur_p = rp.get("planner.pitch_offset")
                         .value_or(rp.get("aimer.pitch_offset").value_or(-ep));
  msg = fmt::format(
    "已写入 yaw_offset={:.2f}° pitch_offset={:.2f}°（{} 发，aimer={} planner={}）", cur_y, cur_p,
    n, set_aimer, set_planner);
  return true;
}

void AimCorrector::clear()
{
  std::lock_guard<std::mutex> lk(mtx_);
  shots_.clear();
  results_.clear();
  last_blobs_.clear();
  matched_prev_.clear();
  shots_total_ = shots_matched_ = 0;
  det_.reset();
}

nlohmann::json AimCorrector::status() const
{
  std::lock_guard<std::mutex> lk(mtx_);
  nlohmann::json res = nlohmann::json::array();
  for (const auto & r : results_)
    res.push_back({{"id", r.id}, {"ey", r.ey}, {"ep", r.ep}, {"n", r.n}, {"range", r.range}});
  double ey = 0, ep = 0, sy = 0, sp = 0;
  int n = 0;
  const bool ok = estimate(ey, ep, sy, sp, n);
  auto & rp = tools::RuntimeParams::instance();
  const double cur_y = rp.get("planner.yaw_offset")
                         .value_or(rp.get("aimer.yaw_offset").value_or(0.0));
  const double cur_p = rp.get("planner.pitch_offset")
                         .value_or(rp.get("aimer.pitch_offset").value_or(0.0));
  return {
    {"live", shots_.size()},
    {"shots", shots_total_},
    {"matched", shots_matched_},
    {"blobs", last_blobs_.size()},
    {"det_ms", det_.last_ms()},
    {"results", res},
    {"est", {{"ok", ok}, {"n", n}, {"ey", ey}, {"ep", ep}, {"sy", sy}, {"sp", sp}}},
    {"suggest", {{"yaw_offset", -ey}, {"pitch_offset", -ep}}},
    {"current", {{"yaw_offset", cur_y}, {"pitch_offset", cur_p}}},
    {"auto_apply", auto_apply_ > 0.5},
    {"min_shots", int(min_shots_)}};
}

void AimCorrector::draw_ballistic(
  cv::Mat & img, const Solver & solver, double yaw, double pitch, double v0, double k, double t_end,
  const cv::Scalar & color, int thickness, double muzzle_z_offset)
{
  std::vector<cv::Point> pts;
  const int n = 40;
  for (int i = 1; i <= n; ++i) {
    double r, z;
    tools::ballistic_point(v0, pitch, k, t_end * i / n, r, z);
    cv::Point2f px;
    if (r < 0.3) continue;  // 枪口附近在相机视野外 / 视差大
    if (solver.world2pixel(
          Eigen::Vector3d(r * std::cos(yaw), r * std::sin(yaw), muzzle_z_offset + z), px))
      pts.emplace_back(px);
  }
  if (pts.size() >= 2) cv::polylines(img, pts, false, color, thickness, cv::LINE_AA);
}

void AimCorrector::draw(cv::Mat & img, const Solver & solver, Clock::time_point t_img) const
{
  std::lock_guard<std::mutex> lk(mtx_);
  for (const auto & r : last_rois_) cv::rectangle(img, r, {90, 90, 90}, 1);
  for (const auto & s : shots_) {
    const double t = seconds(t_img - s.t_fire);
    draw_ballistic(
      img, solver, s.yaw, s.pitch, s.v0, s.k, max_flight_, {255, 255, 0}, 1, s.p0.z());  // 青
    cv::Point2f px;
    if (t >= 0 && solver.world2pixel(sim_pos(s, t), px)) {
      cv::circle(img, px, 5, {0, 255, 255}, 2, cv::LINE_AA);  // 黄：虚拟弹丸当前位置
      cv::putText(
        img, std::to_string(s.id), px + cv::Point2f(7, -7), cv::FONT_HERSHEY_SIMPLEX, 0.5,
        {0, 255, 255}, 1, cv::LINE_AA);
    }
    for (const auto & o : s.obs_px) cv::circle(img, o, 3, {255, 0, 255}, -1, cv::LINE_AA);
    if (s.has_match) cv::line(img, s.match_sim, s.match_obs, {255, 255, 255}, 1, cv::LINE_AA);
  }
  for (const auto & b : last_blobs_)
    cv::circle(img, b.center, std::max(4.0f, b.radius + 3), {255, 0, 255}, 1, cv::LINE_AA);
}

}  // namespace auto_aim
