#ifndef TOOLS__SYNC_RECORDER_HPP
#define TOOLS__SYNC_RECORDER_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

namespace tools
{

/// 高频（如 1000 Hz）下位机 IMU + 云台串口状态样本
struct SyncImuSample
{
  std::chrono::steady_clock::time_point t{};
  int64_t t_ns = 0;  // 相对录制起点 t0 的纳秒时间戳（录制/回放时自动填充）
  uint8_t mode = 0;  // 0: IDLE, 1: AUTO_AIM, 2: SMALL_BUFF, 3: BIG_BUFF
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();      // 处理后的 gimbal->world 四元数
  Eigen::Quaterniond raw_q = Eigen::Quaterniond::Identity();  // 串口原始 IMU 四元数
  float yaw = 0.0f;
  float yaw_vel = 0.0f;
  float pitch = 0.0f;
  float pitch_vel = 0.0f;
  float roll = 0.0f;
  float bullet_speed = 0.0f;
  uint16_t bullet_count = 0;
  float vx = 0.0f;
  float vy = 0.0f;
  float vz = 0.0f;
};

/// 帧间隔或采样间隔抖动统计（直接用于评估是否需要硬件触发 1B）
struct SyncDtStats
{
  size_t count = 0;  // 间隔样本数 (N - 1)
  double mean_ms = 0.0;
  double std_ms = 0.0;
  double p50_ms = 0.0;
  double p95_ms = 0.0;
  double p99_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
};

/// 根据单调递增的纳秒时间戳序列计算间隔抖动统计
SyncDtStats compute_dt_stats(const std::vector<int64_t> & timestamps_ns);

/// 生成默认录制目录路径：records/sync_YYYY-MM-DD_HH-MM-SS
std::string make_default_sync_record_dir(const std::string & base_dir = "records");

/// 非阻塞同步录制器：后台线程异步写 MJPEG 视频 + frames.csv + 1000 Hz imu.csv + meta.json
class SyncRecorder
{
public:
  struct Options
  {
    std::string output_dir;       // 为空或 "auto" 时自动生成 records/sync_<timestamp>
    std::string config_path;      // 可选：记录本次录制使用的 YAML 路径
    double fps_hint = 100.0;      // 视频容器写入帧率提示（精确时间戳以 frames.csv 为准）
    size_t max_frame_queue = 64;  // 有界图像队列深度，满时丢弃新帧绝不阻塞主线程
    size_t max_imu_queue = 4096;  // 有界 IMU 队列深度
  };

  explicit SyncRecorder(const Options & opts);
  ~SyncRecorder();

  SyncRecorder(const SyncRecorder &) = delete;
  SyncRecorder & operator=(const SyncRecorder &) = delete;

  /// 非阻塞推入一帧图像与单调时间戳；队列满时返回 false 并累加 dropped_frames
  bool record_frame(const cv::Mat & img, std::chrono::steady_clock::time_point t);

  /// 非阻塞推入一包高频 IMU/串口样本（可在 Gimbal::read_thread 中 1000 Hz 调用）
  bool record_imu(const SyncImuSample & sample);

  /// 排空后台队列、关闭文件、写出 meta.json 并打印抖动统计报告（幂等）
  void stop();

  const std::string & output_dir() const { return output_dir_; }
  uint64_t written_frames() const { return written_frames_.load(std::memory_order_relaxed); }
  uint64_t dropped_frames() const { return dropped_frames_.load(std::memory_order_relaxed); }
  uint64_t written_imu() const { return written_imu_.load(std::memory_order_relaxed); }
  uint64_t dropped_imu() const { return dropped_imu_.load(std::memory_order_relaxed); }

private:
  struct FrameTask
  {
    cv::Mat img;
    int64_t t_ns = 0;
  };

  int64_t to_rel_ns(std::chrono::steady_clock::time_point t);
  void writer_loop();
  void write_meta_json();

  std::string output_dir_;
  std::string config_path_;
  double fps_hint_ = 100.0;
  size_t max_frame_queue_ = 64;
  size_t max_imu_queue_ = 4096;

  std::mutex t0_mtx_;
  bool has_t0_ = false;
  std::chrono::steady_clock::time_point t0_{};

  std::mutex queue_mtx_;
  std::condition_variable queue_cv_;
  std::deque<FrameTask> frame_queue_;
  std::deque<SyncImuSample> imu_queue_;

  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> stopped_{false};
  std::thread writer_thread_;

  int video_width_ = 0;
  int video_height_ = 0;
  std::vector<int64_t> written_frame_ts_ns_;
  std::vector<int64_t> written_imu_ts_ns_;

  std::atomic<uint64_t> written_frames_{0};
  std::atomic<uint64_t> dropped_frames_{0};
  std::atomic<uint64_t> written_imu_{0};
  std::atomic<uint64_t> dropped_imu_{0};
};

/// 离线同步回放器：读取 SyncRecorder 目录，提供与 io::Camera / io::Gimbal 一致的同步接口
class SyncPlayer
{
public:
  explicit SyncPlayer(const std::string & record_dir);

  bool ok() const { return loaded_; }
  const std::string & record_dir() const { return record_dir_; }
  size_t frame_count() const { return frame_ts_ns_.size(); }
  size_t imu_count() const { return imu_samples_.size(); }
  size_t current_frame_index() const { return next_frame_idx_; }

  /// 设置姿态查询时的系统延迟补偿 (ms)，用于离线 A/B 对比不同 camera_latency_ms
  void set_latency_ms(double latency_ms) { latency_ms_ = latency_ms; }
  double latency_ms() const { return latency_ms_; }

  /// 切换离线姿态构造模式：默认使用录制时的 q；也可切换为 raw_q * q_gimbal2imu 或 euler(yaw,pitch)
  void set_use_raw_q(
    bool enable, const Eigen::Quaterniond & q_gimbal2imu = Eigen::Quaterniond::Identity());
  void set_use_euler_q(bool enable);

  /// 复位到第 0 帧
  void reset();

  /// 顺序读取下一帧图像与单调时间戳；到达末尾返回 false
  bool read_frame(
    cv::Mat & img, std::chrono::steady_clock::time_point & t, size_t * out_frame_idx = nullptr);

  /// 在 t - latency_ms 时刻对高频 IMU 轨迹做二分查找 + SLERP 插值
  Eigen::Quaterniond q(std::chrono::steady_clock::time_point t) const;
  Eigen::Quaterniond q(std::chrono::steady_clock::time_point t, double latency_ms_override) const;

  /// 返回 t - latency_ms 时刻最近的完整串口/IMU 状态样本
  SyncImuSample state_at(std::chrono::steady_clock::time_point t) const;
  SyncImuSample state_at(std::chrono::steady_clock::time_point t, double latency_ms_override) const;

  SyncDtStats frame_dt_stats() const { return compute_dt_stats(frame_ts_ns_); }
  SyncDtStats imu_dt_stats() const { return compute_dt_stats(imu_ts_ns_); }

private:
  Eigen::Quaterniond extract_q(const SyncImuSample & s) const;
  int64_t to_rel_ns(std::chrono::steady_clock::time_point t) const;

  std::string record_dir_;
  bool loaded_ = false;
  cv::VideoCapture cap_;
  std::chrono::steady_clock::time_point base_tp_{};

  std::vector<int64_t> frame_ts_ns_;
  std::vector<int64_t> imu_ts_ns_;
  std::vector<SyncImuSample> imu_samples_;
  size_t next_frame_idx_ = 0;

  double latency_ms_ = 0.0;
  bool use_raw_q_ = false;
  bool use_euler_q_ = false;
  Eigen::Quaterniond q_gimbal2imu_ = Eigen::Quaterniond::Identity();
};

}  // namespace tools

#endif  // TOOLS__SYNC_RECORDER_HPP
