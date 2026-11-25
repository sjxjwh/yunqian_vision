#include "sync_recorder.hpp"

#include <fmt/chrono.h>
#include <fmt/core.h>

#include <algorithm>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

#include "logger.hpp"

namespace tools
{

namespace
{
double percentile_sorted(const std::vector<double> & sorted, double p)
{
  if (sorted.empty()) return 0.0;
  const double idx = std::clamp(p, 0.0, 1.0) * static_cast<double>(sorted.size() - 1);
  const auto lo = static_cast<size_t>(std::floor(idx));
  const auto hi = static_cast<size_t>(std::ceil(idx));
  const double frac = idx - static_cast<double>(lo);
  return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

std::string now_iso_string()
{
  auto now = std::chrono::system_clock::now();
  std::time_t tt = std::chrono::system_clock::to_time_t(now);
  std::tm tm_buf{};
  localtime_r(&tt, &tm_buf);
  return fmt::format("{:%Y-%m-%d_%H-%M-%S}", tm_buf);
}
}  // namespace

SyncDtStats compute_dt_stats(const std::vector<int64_t> & timestamps_ns)
{
  SyncDtStats st;
  if (timestamps_ns.size() < 2) return st;

  std::vector<double> dts;
  dts.reserve(timestamps_ns.size() - 1);
  double sum = 0.0;
  for (size_t i = 1; i < timestamps_ns.size(); ++i) {
    const double dt_ms = static_cast<double>(timestamps_ns[i] - timestamps_ns[i - 1]) * 1e-6;
    if (dt_ms >= 0.0) {
      dts.push_back(dt_ms);
      sum += dt_ms;
    }
  }
  if (dts.empty()) return st;

  st.count = dts.size();
  st.mean_ms = sum / static_cast<double>(dts.size());
  double var_sum = 0.0;
  for (double v : dts) {
    const double d = v - st.mean_ms;
    var_sum += d * d;
  }
  st.std_ms = std::sqrt(var_sum / static_cast<double>(dts.size()));

  std::sort(dts.begin(), dts.end());
  st.min_ms = dts.front();
  st.max_ms = dts.back();
  st.p50_ms = percentile_sorted(dts, 0.50);
  st.p95_ms = percentile_sorted(dts, 0.95);
  st.p99_ms = percentile_sorted(dts, 0.99);
  return st;
}

std::string make_default_sync_record_dir(const std::string & base_dir)
{
  return fmt::format("{}/sync_{}", base_dir, now_iso_string());
}

SyncRecorder::SyncRecorder(const Options & opts)
: output_dir_(
    (opts.output_dir.empty() || opts.output_dir == "auto") ? make_default_sync_record_dir()
                                                           : opts.output_dir),
  config_path_(opts.config_path),
  fps_hint_(opts.fps_hint > 1.0 ? opts.fps_hint : 100.0),
  max_frame_queue_(std::max<size_t>(opts.max_frame_queue, 4)),
  max_imu_queue_(std::max<size_t>(opts.max_imu_queue, 64))
{
  std::filesystem::create_directories(output_dir_);
  writer_thread_ = std::thread(&SyncRecorder::writer_loop, this);
  logger()->info(
    "[SyncRecorder] 启动异步同步录制: dir={} | fps_hint={:.1f}", output_dir_, fps_hint_);
}

SyncRecorder::~SyncRecorder() { stop(); }

int64_t SyncRecorder::to_rel_ns(std::chrono::steady_clock::time_point t)
{
  std::lock_guard<std::mutex> lk(t0_mtx_);
  if (!has_t0_) {
    t0_ = t;
    has_t0_ = true;
    return 0;
  }
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t - t0_).count();
}

bool SyncRecorder::record_frame(const cv::Mat & img, std::chrono::steady_clock::time_point t)
{
  if (img.empty() || stop_requested_.load(std::memory_order_relaxed)) return false;

  FrameTask task;
  task.img = img.clone();
  task.t_ns = to_rel_ns(t);

  {
    std::lock_guard<std::mutex> lk(queue_mtx_);
    if (frame_queue_.size() >= max_frame_queue_) {
      dropped_frames_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    frame_queue_.push_back(std::move(task));
  }
  queue_cv_.notify_one();
  return true;
}

bool SyncRecorder::record_imu(const SyncImuSample & sample)
{
  if (stop_requested_.load(std::memory_order_relaxed)) return false;

  SyncImuSample s = sample;
  if (s.t.time_since_epoch().count() != 0) {
    s.t_ns = to_rel_ns(s.t);
  }

  {
    std::lock_guard<std::mutex> lk(queue_mtx_);
    if (imu_queue_.size() >= max_imu_queue_) {
      dropped_imu_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    imu_queue_.push_back(s);
  }
  queue_cv_.notify_one();
  return true;
}

void SyncRecorder::stop()
{
  if (stopped_.exchange(true)) return;
  stop_requested_.store(true);
  queue_cv_.notify_all();
  if (writer_thread_.joinable()) writer_thread_.join();
  write_meta_json();

  const auto f_st = compute_dt_stats(written_frame_ts_ns_);
  const auto i_st = compute_dt_stats(written_imu_ts_ns_);
  logger()->info(
    "[SyncRecorder] 录制结束: dir={} | frames={} (drop={}) | imu={} (drop={})", output_dir_,
    written_frames_.load(), dropped_frames_.load(), written_imu_.load(), dropped_imu_.load());
  if (f_st.count > 0) {
    logger()->info(
      "[SyncRecorder] 图像 dt: mean={:.2f}ms std={:.2f}ms P50={:.2f}ms P95={:.2f}ms P99={:.2f}ms "
      "max={:.2f}ms",
      f_st.mean_ms, f_st.std_ms, f_st.p50_ms, f_st.p95_ms, f_st.p99_ms, f_st.max_ms);
  }
  if (i_st.count > 0) {
    logger()->info(
      "[SyncRecorder] IMU  dt: mean={:.2f}ms std={:.2f}ms P50={:.2f}ms P99={:.2f}ms max={:.2f}ms",
      i_st.mean_ms, i_st.std_ms, i_st.p50_ms, i_st.p99_ms, i_st.max_ms);
  }
}

void SyncRecorder::writer_loop()
{
  std::ofstream frames_csv(output_dir_ + "/frames.csv");
  frames_csv << "frame_idx,t_ns\n";

  std::ofstream imu_csv(output_dir_ + "/imu.csv");
  imu_csv << "t_ns,mode,qw,qx,qy,qz,raw_qw,raw_qx,raw_qy,raw_qz,"
             "yaw,yaw_vel,pitch,pitch_vel,roll,bullet_speed,bullet_count,vx,vy,vz\n";

  cv::VideoWriter writer;
  uint64_t frame_idx = 0;

  std::deque<FrameTask> local_frames;
  std::deque<SyncImuSample> local_imu;

  while (true) {
    {
      std::unique_lock<std::mutex> lk(queue_mtx_);
      queue_cv_.wait(lk, [&]() {
        return stop_requested_.load(std::memory_order_relaxed) || !frame_queue_.empty() ||
               !imu_queue_.empty();
      });
      local_frames.swap(frame_queue_);
      local_imu.swap(imu_queue_);
      if (local_frames.empty() && local_imu.empty() && stop_requested_.load()) {
        break;
      }
    }

    for (const auto & s : local_imu) {
      imu_csv << fmt::format(
        "{},{},{:.8f},{:.8f},{:.8f},{:.8f},{:.8f},{:.8f},{:.8f},{:.8f},"
        "{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.3f},{},{:.4f},{:.4f},{:.4f}\n",
        s.t_ns, static_cast<int>(s.mode), s.q.w(), s.q.x(), s.q.y(), s.q.z(), s.raw_q.w(),
        s.raw_q.x(), s.raw_q.y(), s.raw_q.z(), s.yaw, s.yaw_vel, s.pitch, s.pitch_vel, s.roll,
        s.bullet_speed, s.bullet_count, s.vx, s.vy, s.vz);
      written_imu_ts_ns_.push_back(s.t_ns);
      written_imu_.fetch_add(1, std::memory_order_relaxed);
    }
    local_imu.clear();

    for (auto & f : local_frames) {
      if (!writer.isOpened()) {
        video_width_ = f.img.cols;
        video_height_ = f.img.rows;
        const int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        writer.open(output_dir_ + "/video.avi", fourcc, fps_hint_, f.img.size(), true);
        if (!writer.isOpened()) {
          logger()->error("[SyncRecorder] 无法打开视频写入器: {}/video.avi", output_dir_);
        }
      }
      if (writer.isOpened()) {
        writer.write(f.img);
        frames_csv << fmt::format("{},{}\n", frame_idx, f.t_ns);
        written_frame_ts_ns_.push_back(f.t_ns);
        frame_idx++;
        written_frames_.fetch_add(1, std::memory_order_relaxed);
      }
    }
    local_frames.clear();
  }

  if (writer.isOpened()) writer.release();
  frames_csv.flush();
  imu_csv.flush();
}

void SyncRecorder::write_meta_json()
{
  const auto f_st = compute_dt_stats(written_frame_ts_ns_);
  const auto i_st = compute_dt_stats(written_imu_ts_ns_);

  nlohmann::json j;
  j["version"] = 1;
  j["created_at"] = now_iso_string();
  j["config_path"] = config_path_;
  j["video_file"] = "video.avi";
  j["video_codec"] = "MJPG";
  j["width"] = video_width_;
  j["height"] = video_height_;
  j["fps_hint"] = fps_hint_;
  j["frames"] = written_frames_.load();
  j["dropped_frames"] = dropped_frames_.load();
  j["imu_samples"] = written_imu_.load();
  j["dropped_imu"] = dropped_imu_.load();
  j["frame_dt_ms"] = {
    {"count", f_st.count}, {"mean", f_st.mean_ms}, {"std", f_st.std_ms}, {"p50", f_st.p50_ms},
    {"p95", f_st.p95_ms},  {"p99", f_st.p99_ms},   {"min", f_st.min_ms}, {"max", f_st.max_ms}};
  j["imu_dt_ms"] = {
    {"count", i_st.count}, {"mean", i_st.mean_ms}, {"std", i_st.std_ms}, {"p50", i_st.p50_ms},
    {"p95", i_st.p95_ms},  {"p99", i_st.p99_ms},   {"min", i_st.min_ms}, {"max", i_st.max_ms}};

  std::ofstream meta_out(output_dir_ + "/meta.json");
  meta_out << j.dump(2) << "\n";
}

// ============================= SyncPlayer =============================

SyncPlayer::SyncPlayer(const std::string & record_dir)
: record_dir_(record_dir), base_tp_(std::chrono::steady_clock::now())
{
  const std::string video_path = record_dir_ + "/video.avi";
  const std::string frames_path = record_dir_ + "/frames.csv";
  const std::string imu_path = record_dir_ + "/imu.csv";

  if (!cap_.open(video_path)) {
    logger()->error("[SyncPlayer] 无法打开视频文件: {}", video_path);
    return;
  }

  // 加载 frames.csv
  {
    std::ifstream fin(frames_path);
    if (!fin.is_open()) {
      logger()->error("[SyncPlayer] 无法打开 frames.csv: {}", frames_path);
      return;
    }
    std::string line;
    std::getline(fin, line);  // header
    while (std::getline(fin, line)) {
      if (line.empty()) continue;
      const auto comma = line.find(',');
      if (comma == std::string::npos) continue;
      const int64_t t_ns = std::stoll(line.substr(comma + 1));
      frame_ts_ns_.push_back(t_ns);
    }
  }

  // 加载 imu.csv（可选，但通常存在）
  {
    std::ifstream fin(imu_path);
    if (fin.is_open()) {
      std::string line;
      std::getline(fin, line);  // header
      while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string tok;
        std::vector<std::string> cols;
        cols.reserve(20);
        while (std::getline(ss, tok, ',')) cols.push_back(tok);
        if (cols.size() < 20) continue;

        SyncImuSample s;
        s.t_ns = std::stoll(cols[0]);
        s.t = base_tp_ + std::chrono::nanoseconds(s.t_ns);
        s.mode = static_cast<uint8_t>(std::stoi(cols[1]));
        s.q = Eigen::Quaterniond(
                std::stod(cols[2]), std::stod(cols[3]), std::stod(cols[4]), std::stod(cols[5]))
                .normalized();
        s.raw_q = Eigen::Quaterniond(
                    std::stod(cols[6]), std::stod(cols[7]), std::stod(cols[8]), std::stod(cols[9]))
                    .normalized();
        s.yaw = std::stof(cols[10]);
        s.yaw_vel = std::stof(cols[11]);
        s.pitch = std::stof(cols[12]);
        s.pitch_vel = std::stof(cols[13]);
        s.roll = std::stof(cols[14]);
        s.bullet_speed = std::stof(cols[15]);
        s.bullet_count = static_cast<uint16_t>(std::stoi(cols[16]));
        s.vx = std::stof(cols[17]);
        s.vy = std::stof(cols[18]);
        s.vz = std::stof(cols[19]);

        imu_ts_ns_.push_back(s.t_ns);
        imu_samples_.push_back(s);
      }
    }
  }

  loaded_ = !frame_ts_ns_.empty();
  if (loaded_) {
    logger()->info(
      "[SyncPlayer] 加载成功: dir={} | frames={} | imu={}", record_dir_, frame_ts_ns_.size(),
      imu_samples_.size());
  }
}

void SyncPlayer::set_use_raw_q(bool enable, const Eigen::Quaterniond & q_gimbal2imu)
{
  use_raw_q_ = enable;
  q_gimbal2imu_ = q_gimbal2imu.normalized();
}

void SyncPlayer::set_use_euler_q(bool enable) { use_euler_q_ = enable; }

void SyncPlayer::reset()
{
  next_frame_idx_ = 0;
  if (cap_.isOpened()) {
    cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
  }
}

bool SyncPlayer::read_frame(
  cv::Mat & img, std::chrono::steady_clock::time_point & t, size_t * out_frame_idx)
{
  if (!loaded_ || next_frame_idx_ >= frame_ts_ns_.size()) return false;
  if (!cap_.read(img) || img.empty()) return false;

  t = base_tp_ + std::chrono::nanoseconds(frame_ts_ns_[next_frame_idx_]);
  if (out_frame_idx) *out_frame_idx = next_frame_idx_;
  next_frame_idx_++;
  return true;
}

int64_t SyncPlayer::to_rel_ns(std::chrono::steady_clock::time_point t) const
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t - base_tp_).count();
}

Eigen::Quaterniond SyncPlayer::extract_q(const SyncImuSample & s) const
{
  if (use_euler_q_) {
    return Eigen::Quaterniond(
             Eigen::AngleAxisd(static_cast<double>(s.yaw), Eigen::Vector3d::UnitZ()) *
             Eigen::AngleAxisd(static_cast<double>(s.pitch), Eigen::Vector3d::UnitY()))
      .normalized();
  }
  if (use_raw_q_) {
    return (s.raw_q * q_gimbal2imu_).normalized();
  }
  return s.q.normalized();
}

Eigen::Quaterniond SyncPlayer::q(std::chrono::steady_clock::time_point t) const
{
  return q(t, latency_ms_);
}

Eigen::Quaterniond SyncPlayer::q(
  std::chrono::steady_clock::time_point t, double latency_ms_override) const
{
  if (imu_samples_.empty()) return Eigen::Quaterniond::Identity();

  const int64_t query_ns = to_rel_ns(t) - static_cast<int64_t>(std::llround(latency_ms_override * 1e6));
  auto it = std::lower_bound(imu_ts_ns_.begin(), imu_ts_ns_.end(), query_ns);
  if (it == imu_ts_ns_.begin()) return extract_q(imu_samples_.front());
  if (it == imu_ts_ns_.end()) return extract_q(imu_samples_.back());

  const size_t idx_b = static_cast<size_t>(std::distance(imu_ts_ns_.begin(), it));
  const size_t idx_a = idx_b - 1;
  const int64_t ta = imu_ts_ns_[idx_a];
  const int64_t tb = imu_ts_ns_[idx_b];
  const double denom = static_cast<double>(tb - ta);
  const double k = (denom > 1.0) ? std::clamp(static_cast<double>(query_ns - ta) / denom, 0.0, 1.0) : 0.0;
  return extract_q(imu_samples_[idx_a]).slerp(k, extract_q(imu_samples_[idx_b])).normalized();
}

SyncImuSample SyncPlayer::state_at(std::chrono::steady_clock::time_point t) const
{
  return state_at(t, latency_ms_);
}

SyncImuSample SyncPlayer::state_at(
  std::chrono::steady_clock::time_point t, double latency_ms_override) const
{
  if (imu_samples_.empty()) return {};

  const int64_t query_ns = to_rel_ns(t) - static_cast<int64_t>(std::llround(latency_ms_override * 1e6));
  auto it = std::lower_bound(imu_ts_ns_.begin(), imu_ts_ns_.end(), query_ns);
  if (it == imu_ts_ns_.begin()) return imu_samples_.front();
  if (it == imu_ts_ns_.end()) return imu_samples_.back();

  const size_t idx_b = static_cast<size_t>(std::distance(imu_ts_ns_.begin(), it));
  const size_t idx_a = idx_b - 1;
  const int64_t da = std::llabs(query_ns - imu_ts_ns_[idx_a]);
  const int64_t db = std::llabs(imu_ts_ns_[idx_b] - query_ns);
  return (da <= db) ? imu_samples_[idx_a] : imu_samples_[idx_b];
}

}  // namespace tools
