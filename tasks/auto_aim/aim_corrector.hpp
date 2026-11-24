#ifndef AUTO_AIM__AIM_CORRECTOR_HPP
#define AUTO_AIM__AIM_CORRECTOR_HPP

// 弹道闭环：思路来自 SJTU rm.cv.fans 的 AimCorrector，移植自 origin/vision 并对接本仓库
// Aimer 与 Planner 的 RuntimeParams 零偏接口及二维耦合空气阻力弹道模型。
//
//   1. 每个发射事件（下位机反馈或仿真触发）生成一颗“虚拟弹丸”：
//        出射方向 = 发射时刻 IMU 测得的云台姿态（不是指令），弹速 = 实测弹速，阻力 = air_k
//      所以虚拟弹丸代表“按弹道模型，这一发本该怎么飞”
//   2. 每帧在虚拟弹丸投影附近的小 ROI 里找真实弹丸：
//        上一帧按 IMU 旋转做单应补偿 → 帧差 ∩ 荧光绿 HSV → 形态学 → 轮廓
//   3. 配对后，把观测弹丸放到与虚拟弹丸相同的相机距离上，算两者相对云台原点的 yaw/pitch 差 E
//      （云台原点看过去的角度对深度误差不敏感，不需要靠弹丸半径估深度）
//   4. 每发取中位数，最近 N 发再取中位数；建议补偿 = -E（绝对值，与当前 offset 无关，
//      因为虚拟弹丸用的是实测姿态，已经包含了当前 offset 和云台跟随误差）
//
// E 吸收的是：枪管安装误差、相机外参误差、弹速/阻力模型误差（在当前距离上）。
// 云台跟随误差不在 E 里（由电控与 MPC 规划器处理）。
#include <Eigen/Dense>
#include <chrono>
#include <deque>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "solver.hpp"

namespace auto_aim
{
struct BulletBlob
{
  cv::Point2f center;
  float radius;
};

/// 荧光弹丸检测（只在给定 ROI 内做，带 IMU 旋转补偿的帧差）
class BulletDetector
{
public:
  BulletDetector();

  /// 处理一帧；rois 为空时只缓存这一帧（给下一帧做差）
  /// keep：上一帧已配对的弹丸（上一帧像素坐标）。弹丸远离相机时相邻帧几乎重叠，帧差会把它吃掉，
  ///       所以这些位置免帧差（同 rm.cv.fans 的 lst_fr_bullets；只对已配对的做，防止静止绿物“粘住”）
  std::vector<BulletBlob> feed(
    const cv::Mat & bgr, const Eigen::Matrix3d & R_camera2world, const cv::Mat & K,
    const std::vector<cv::Rect> & rois, const std::vector<BulletBlob> & keep = {});
  void reset() { has_prev_ = false; }
  double last_ms() const { return last_ms_; }

  // 参数（运行期可调）
  double h_lo = 25, h_hi = 90, s_min = 40, v_min = 60;  // OpenCV HSV，H 0~180
  double diff_thresh = 30;                              // 帧差阈值（三通道最大差）
  double min_area = 4, max_area = 3000;                 // 像素
  double min_fill = 0.3;                                // 面积 / 外接圆面积

private:
  cv::Mat prev_;
  Eigen::Matrix3d prev_R_;
  bool has_prev_ = false;
  bool copy_mode_ = false;
  double last_ms_ = 0;
};

class AimCorrector
{
public:
  using Clock = std::chrono::steady_clock;

  explicit AimCorrector(const std::string & config_path);

  /// 新的发射事件。t_fire 已含 fire_delay；R 为发射时刻的 R_gimbal2world；
  /// origin 为出膛点世界坐标（若 z 为 NaN 则退回 {0, 0, muzzle_z_offset_}）；
  /// stop_range 为最大有效采样距离（建议传入目标距离 - 0.15m，避免匹配撞击反弹后的弹丸）
  void add_shot(
    Clock::time_point t_fire, const Eigen::Matrix3d & R_gimbal2world, double v0, double k,
    const Eigen::Vector3d & origin =
      Eigen::Vector3d(0.0, 0.0, std::numeric_limits<double>::quiet_NaN()),
    double stop_range = 25.0);

  /// 每帧调用；solver 需已 set_R_gimbal2world(图像时刻姿态)
  /// 注意：会保留 bgr 的引用到下一帧做帧差，调用方之后不能再往这块图像上画东西
  ///       （要画标注请画在 clone 上）。可以和 YOLO 在不同线程并行调用
  void update(const cv::Mat & bgr, Clock::time_point t_img, const Solver & solver);

  /// 在（全分辨率）图上画虚拟弹道 / 检测 / 配对
  void draw(cv::Mat & img, const Solver & solver, Clock::time_point t_img) const;

  /// 画一条弹道（从炮口出发），预览“按模型这一发会怎么飞”
  static void draw_ballistic(
    cv::Mat & img, const Solver & solver, double yaw, double pitch, double v0, double k,
    double t_end, const cv::Scalar & color, int thickness = 1, double muzzle_z_offset = 0.0);

  nlohmann::json status() const;
  void clear();
  /// 把建议值同步写进已注册的 aimer / planner yaw_offset & pitch_offset；样本不够返回 false
  bool apply_suggestion(std::string & msg);

  double fire_delay() const { return fire_delay_; }
  double muzzle_z_offset() const { return muzzle_z_offset_; }
  bool active() const;

private:
  struct Shot
  {
    int id;
    Clock::time_point t_fire;
    double yaw, pitch, v0, k;
    Eigen::Vector3d p0{0.0, 0.0, 0.0};
    double stop_range = 25.0;
    std::vector<Eigen::Vector2d> errs;  // rad: (yaw, pitch)
    std::vector<cv::Point2f> obs_px;
    double max_range = 0;
    bool has_match = false;
    cv::Point2f match_obs, match_sim;
  };
  struct Result
  {
    int id;
    double ey, ep;  // deg，观测 - 模型
    int n;
    double range;
  };

  mutable std::mutex mtx_;
  BulletDetector det_;
  std::deque<Shot> shots_;
  std::deque<Result> results_;
  std::vector<BulletBlob> last_blobs_;
  std::vector<BulletBlob> matched_prev_;  // 上一帧配对上的检测，给帧差免检
  std::vector<cv::Rect> last_rois_;
  int next_id_ = 1;
  long shots_total_ = 0, shots_matched_ = 0;
  cv::Size img_size_{0, 0};

  // 参数
  double fire_delay_ = 0;       // s，发射反馈时刻 → 弹丸出膛
  double muzzle_z_offset_ = 0;  // m，炮口相对云台原点的高度差
  double gate_px_ = 80;         // 配对门限（像素，全分辨率）
  double min_range_ = 0.8;      // m，太近的样本视差大，不用
  double max_flight_ = 0.8;     // s
  double min_samples_ = 2;      // 每发至少几个观测才算数
  double window_ = 10;          // 估计用最近几发
  double auto_apply_ = 0;       // 1 = 自动写 offset
  double max_step_ = 0.3;       // 自动写入时每发最多改多少度
  double min_shots_ = 5;        // 至少几发才给建议 / 自动写入
  double detect_all_ = 0;       // 1 = 全图检测（调 HSV 用，较耗时）

  static Eigen::Vector3d sim_pos(const Shot & s, double t);
  void finalize(const Shot & s);
  bool estimate(double & ey, double & ep, double & sy, double & sp, int & n) const;
  void maybe_auto_apply();
};

}  // namespace auto_aim

#endif  // AUTO_AIM__AIM_CORRECTOR_HPP
