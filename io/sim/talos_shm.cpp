#include "talos_shm.hpp"

#include <atomic>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <opencv2/imgproc.hpp>

#include "tools/logger.hpp"

namespace io
{
namespace talos
{

namespace
{

// 仿真侧每个结构都标了 align，C++ 默认对齐不同，必须逐个指定。
struct alignas(32) ImageMetaRaw
{
  uint64_t seq;
  uint64_t timestamp_ns;
  uint32_t width;
  uint32_t height;
  uint8_t buffer_id;
  uint8_t format;
  uint8_t pad[6];
};

struct alignas(64) PoseMetaRaw
{
  uint64_t frame_seq;
  float position[3];
  float quaternion[4];
  uint64_t timestamp_ns;
  uint8_t pad[16];
};

struct alignas(32) GimbalCmdRaw
{
  uint64_t timestamp_ns;
  float yaw_deg;
  float pitch_deg;
  float distance_m;
  uint8_t fire_advice;
  uint8_t pad[11];
};

struct alignas(64) CameraInfoRaw
{
  uint64_t timestamp_ns;
  double fx;
  double fy;
  double cx;
  double cy;
  double distortion[5];
  uint32_t width;
  uint32_t height;
  uint8_t pad[24];
};

struct alignas(64) ChassisObservationRaw
{
  uint64_t frame_seq;
  uint64_t timestamp_ns;
  float dt_s;
  float v_body[2];
  float wz_radps;
  float wheel_linear_mps[4];
  float wheel_angular_radps[4];
  float a_body[2];
  float alpha_z_radps2;
  float rpy_rad[3];
  float gyro_xyz_radps[3];
  float accel_xyz_mps2[3];
  uint8_t pad[16];
};

constexpr std::size_t GROUND_TRUTH_MAX_TARGETS = 16;
constexpr std::size_t GROUND_TRUTH_MAX_RUNES = 4;

struct alignas(32) GroundTruthTargetRaw
{
  uint64_t frame_seq;
  uint64_t timestamp_ns;
  uint8_t team;
  uint8_t armor_label;
  uint8_t is_outpost;
  uint8_t pad1;
  float position[3];
  float vyaw;
  float yaw;
  uint8_t pad[24];
};

struct alignas(64) GroundTruthRuneRaw
{
  uint64_t frame_seq;
  uint64_t timestamp_ns;
  uint8_t team;
  uint8_t rune_mode;
  uint8_t mechanism_state;
  uint8_t pad1;
  float r_center_odom[3];
  float radius;
  float current_angle;
  float v_roll;
  int32_t direction;
  float sin_amplitude;
  float sin_omega;
  float sin_phase;
  float sin_offset;
  float relative_time;
  int32_t blade_id;
  uint8_t target_activations[5];
  uint8_t pad[20];
};

struct alignas(64) GroundTruthBatchRaw
{
  uint64_t frame_seq;
  uint64_t timestamp_ns;
  uint32_t target_count;
  uint32_t rune_count;
  GroundTruthTargetRaw targets[GROUND_TRUTH_MAX_TARGETS];
  GroundTruthRuneRaw runes[GROUND_TRUTH_MAX_RUNES];
  uint8_t pad[64];
};

// 三缓冲的控制字。state 是原子量，生产者 swap、消费者 CAS，其余两个索引由各自独占。
struct alignas(64) TripleBufferRaw
{
  std::atomic<uint8_t> state;
  uint8_t write_idx;
  uint8_t read_idx;
  uint8_t pad[61];
};

struct alignas(64) ImageTripleBufferRaw
{
  TripleBufferRaw ctrl;
  ImageMetaRaw slots[3];
};

struct alignas(64) PoseTripleBufferRaw
{
  TripleBufferRaw ctrl;
  PoseMetaRaw slots[3];
};

struct alignas(64) GimbalTripleBufferRaw
{
  TripleBufferRaw ctrl;
  GimbalCmdRaw slots[3];
};

struct alignas(64) ShmHeaderRaw
{
  uint32_t magic;
  uint32_t version;
  uint64_t created_ns;
  uint64_t heartbeat_ns;
  uint32_t image_width;
  uint32_t image_height;
  uint8_t pad[32];
};
static_assert(offsetof(ShmHeaderRaw, heartbeat_ns) == 16, "heartbeat offset");

// 只映射到本侧会读写的部分。后面还有真值与运行态，不碰，所以不要求整块大小相等，
// 只要求文件不小于这里声明的范围。
struct alignas(64) MetaPrefix
{
  ShmHeaderRaw header;
  ImageTripleBufferRaw image;
  PoseTripleBufferRaw poses[5];
  GimbalTripleBufferRaw gimbal_cmd;
  CameraInfoRaw camera_info;
  ChassisObservationRaw chassis_observation;
};

// 与 talos-ipc/src/layout.rs 的 offset_of 断言一一对应。错一位就拒绝编译。
static_assert(sizeof(ImageMetaRaw) == 32, "ImageMeta");
static_assert(sizeof(PoseMetaRaw) == 64, "PoseMeta");
static_assert(sizeof(GimbalCmdRaw) == 32, "GimbalCmd");
static_assert(sizeof(CameraInfoRaw) == 128, "CameraInfo");
static_assert(sizeof(ChassisObservationRaw) == 128, "ChassisObservation");
static_assert(sizeof(ImageTripleBufferRaw) == 192, "ImageTripleBuffer");
static_assert(sizeof(PoseTripleBufferRaw) == 256, "PoseTripleBuffer");
static_assert(sizeof(GimbalTripleBufferRaw) == 192, "GimbalTripleBuffer");
static_assert(sizeof(ShmHeaderRaw) == 64, "ShmHeader");
static_assert(offsetof(MetaPrefix, camera_info) == 1728, "camera_info offset");
static_assert(offsetof(MetaPrefix, chassis_observation) == 1856, "chassis_observation offset");
struct alignas(64) RuntimeStateRaw
{
  uint64_t timestamp_ns;
  uint8_t following;
  uint8_t pad[55];
};

struct alignas(64) MetaWithGt
{
  MetaPrefix prefix;
  GroundTruthBatchRaw ground_truth;
  RuntimeStateRaw runtime_state;
};

static_assert(sizeof(GroundTruthTargetRaw) == 64, "GroundTruthTarget");
static_assert(sizeof(GroundTruthRuneRaw) == 128, "GroundTruthRune");
static_assert(sizeof(GroundTruthBatchRaw) == 1664, "GroundTruthBatch");
static_assert(sizeof(RuntimeStateRaw) == 64, "RuntimeState");
static_assert(offsetof(MetaWithGt, ground_truth) == 1984, "ground_truth offset");
static_assert(offsetof(MetaWithGt, runtime_state) == 3648, "runtime_state offset");
static_assert(sizeof(MetaWithGt) == 3712, "ShmMetaRegion total size");
static_assert(std::atomic<uint8_t>::is_always_lock_free, "triple buffer state must be lock-free");

uint64_t now_ns()
{
  return static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch())
      .count());
}

// 消费一次三缓冲。没有新数据返回空。协议与仿真侧 TripleBufferConsumer::borrow 相同：
// 读到 FLAG_NEW 后用 CAS 把 state 换成自己上次的槽号，失败再试一次，并同步写回 ctrl.read_idx
// 以保证进程重启重连时三缓冲 {write_idx, state&3, read_idx} 互异不塌缩。
template <typename Slot>
const Slot * consume(TripleBufferRaw & ctrl, Slot (&slots)[3], uint8_t & read_idx)
{
  uint8_t expected = ctrl.state.load(std::memory_order_acquire);
  if ((expected & FLAG_NEW) == 0) return nullptr;

  uint8_t ready = expected & INDEX_MASK;
  uint8_t desired = read_idx;

  if (ctrl.state.compare_exchange_weak(
        expected, desired, std::memory_order_acq_rel, std::memory_order_acquire)) {
    read_idx = ready;
    ctrl.read_idx = ready;
    return &slots[ready];
  }

  if ((expected & FLAG_NEW) == 0) return nullptr;
  ready = expected & INDEX_MASK;
  desired = read_idx;
  if (!ctrl.state.compare_exchange_weak(
        expected, desired, std::memory_order_acq_rel, std::memory_order_relaxed)) {
    return nullptr;
  }
  read_idx = ready;
  ctrl.read_idx = ready;
  return &slots[ready];
}

// 重连时恢复消费者独占的 read_idx：确保与生产者的 write_idx 及 state 当前槽号互异，
// 若历史进程异常退出导致三缓冲塌缩也能原地自愈。
uint8_t recover_read_idx(TripleBufferRaw & ctrl)
{
  const uint8_t s = ctrl.state.load(std::memory_order_acquire) & INDEX_MASK;
  const uint8_t w = ctrl.write_idx & INDEX_MASK;
  uint8_t r = ctrl.read_idx;
  if (r <= 2 && r != s && r != w) return r;
  for (uint8_t cand = 0; cand < 3; ++cand) {
    if (cand != s && cand != w) {
      ctrl.read_idx = cand;
      return cand;
    }
  }
  r = static_cast<uint8_t>((w + 1) % 3);
  ctrl.read_idx = r;
  return r;
}

}  // namespace

struct ShmClient::Mapping
{
  int fd = -1;
  void * addr = nullptr;
  std::size_t size = 0;

  ~Mapping()
  {
    if (addr && addr != MAP_FAILED) munmap(addr, size);
    if (fd >= 0) ::close(fd);
  }
};

ShmClient::ShmClient() = default;

ShmClient::~ShmClient() { close(); }

void ShmClient::close()
{
  delete meta_;
  delete pool_;
  meta_ = nullptr;
  pool_ = nullptr;
}

bool ShmClient::connected() const { return meta_ != nullptr && pool_ != nullptr; }

void ShmClient::connect()
{
  close();

  auto open_one = [](const char * name, std::size_t min_size) {
    auto path = std::string("/tmp/") + name;
    auto mapping = std::make_unique<Mapping>();
    mapping->fd = ::open(path.c_str(), O_RDWR);
    if (mapping->fd < 0) {
      throw std::runtime_error("打不开仿真共享内存 " + path + "，仿真是否已启动");
    }
    struct stat st;
    if (fstat(mapping->fd, &st) != 0 || static_cast<std::size_t>(st.st_size) < min_size) {
      throw std::runtime_error(path + " 尺寸与 talos-ipc 不符");
    }
    mapping->size = static_cast<std::size_t>(st.st_size);
    mapping->addr = mmap(nullptr, mapping->size, PROT_READ | PROT_WRITE, MAP_SHARED, mapping->fd, 0);
    if (mapping->addr == MAP_FAILED) {
      throw std::runtime_error("映射 " + path + " 失败");
    }
    return mapping;
  };

  auto meta = open_one(SHM_NAME_META, sizeof(MetaPrefix));
  auto pool = open_one(SHM_NAME_IMAGE_POOL, IMAGE_POOL_SIZE);

  auto header = static_cast<MetaPrefix *>(meta->addr)->header;
  if (header.magic != SHM_MAGIC || header.version != SHM_VERSION) {
    throw std::runtime_error("仿真共享内存版本不匹配，请确认仿真与本代码基于同一版 talos-ipc");
  }
  if (header.image_width != IMAGE_WIDTH || header.image_height != IMAGE_HEIGHT) {
    throw std::runtime_error("仿真图像分辨率不是 1440x1080，本适配只支持仿真默认分辨率");
  }

  meta_ = meta.release();
  pool_ = pool.release();
  auto & m = *static_cast<MetaPrefix *>(meta_->addr);
  // 重连时若上次进程在消费完 image 但尚未消费 poses 时被杀，残留的 FLAG_NEW 会让仿真握手卡死；
  // 同时恢复三缓冲不变量 {write_idx, state&3, read_idx} 互异，先清掉残留标志，立刻唤醒仿真继续出图。
  image_read_idx_ = recover_read_idx(m.image.ctrl);
  consume(m.image.ctrl, m.image.slots, image_read_idx_);
  for (int i = 0; i < 5; ++i) {
    pose_read_idx_[i] = recover_read_idx(m.poses[i].ctrl);
    consume(m.poses[i].ctrl, m.poses[i].slots, pose_read_idx_[i]);
  }
}

bool ShmClient::alive(std::chrono::milliseconds max_age) const
{
  if (!connected()) return false;
  auto heartbeat = static_cast<const MetaPrefix *>(meta_->addr)->header.heartbeat_ns;
  auto now = now_ns();
  if (heartbeat == 0 || now < heartbeat) return false;
  return now - heartbeat < static_cast<uint64_t>(max_age.count()) * 1000000ULL;
}

bool ShmClient::recv(Frame & frame)
{
  if (!connected()) return false;
  auto & meta = *static_cast<MetaPrefix *>(meta_->addr);

  auto image = consume(meta.image.ctrl, meta.image.slots, image_read_idx_);
  if (image == nullptr) return false;
  if (image->width != IMAGE_WIDTH || image->height != IMAGE_HEIGHT || image->buffer_id > 2) {
    return false;
  }

  // 图像槽在元数据提交前已经写完，这里读到的是稳定的一帧。仿真写的是 rgb8。
  auto src = static_cast<const uint8_t *>(pool_->addr) + image->buffer_id * IMAGE_SIZE;
  cv::Mat rgb(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, const_cast<uint8_t *>(src));
  cv::cvtColor(rgb, frame.bgr, cv::COLOR_RGB2BGR);

  frame.seq = image->seq;
  frame.timestamp_ns = image->timestamp_ns;

  // 姿态与图像同一 frame_seq 提交。消费掉是为了让仿真的握手知道这帧已被取走，
  // 否则仿真会停在上一帧不再发布。
  for (int i = 0; i < 5; ++i) {
    consume(meta.poses[i].ctrl, meta.poses[i].slots, pose_read_idx_[i]);
  }
  auto & gimbal = meta.poses[static_cast<int>(PoseIndex::gimbal)].slots[pose_read_idx_[0]];
  frame.gimbal_quat = Eigen::Quaterniond(
                        gimbal.quaternion[0], gimbal.quaternion[1], gimbal.quaternion[2],
                        gimbal.quaternion[3])
                        .normalized();

  auto & odom = meta.poses[static_cast<int>(PoseIndex::odom)].slots[pose_read_idx_[1]];
  frame.gimbal_in_world = Eigen::Vector3d(odom.position[0], odom.position[1], odom.position[2]);

  auto & muzzle = meta.poses[static_cast<int>(PoseIndex::muzzle)].slots[pose_read_idx_[2]];
  frame.muzzle_in_gimbal =
    Eigen::Vector3d(muzzle.position[0], muzzle.position[1], muzzle.position[2]);

  auto & camera = meta.poses[static_cast<int>(PoseIndex::camera)].slots[pose_read_idx_[3]];
  frame.camera_in_gimbal =
    Eigen::Vector3d(camera.position[0], camera.position[1], camera.position[2]);

  const auto & obs = meta.chassis_observation;
  const double c_roll = static_cast<double>(obs.rpy_rad[0]);
  const double c_pitch = static_cast<double>(obs.rpy_rad[1]);
  const double c_yaw = static_cast<double>(obs.rpy_rad[2]);
  frame.chassis_rpy = Eigen::Vector3d(c_roll, c_pitch, c_yaw);
  frame.chassis_wz = static_cast<double>(obs.wz_radps);
  // 将底盘本体速度 [vx_body, vy_body, 0] 经底盘姿态 Rz(yaw)*Ry(pitch)*Rx(roll) 旋转至 ROS 世界系
  const Eigen::Matrix3d R_chassis2world =
    (Eigen::AngleAxisd(c_yaw, Eigen::Vector3d::UnitZ()) *
     Eigen::AngleAxisd(c_pitch, Eigen::Vector3d::UnitY()) *
     Eigen::AngleAxisd(c_roll, Eigen::Vector3d::UnitX()))
      .toRotationMatrix();
  frame.chassis_vel_world =
    R_chassis2world *
    Eigen::Vector3d(static_cast<double>(obs.v_body[0]), static_cast<double>(obs.v_body[1]), 0.0);

  auto & info = meta.camera_info;
  frame.camera_matrix = cv::Matx33d(info.fx, 0, info.cx, 0, info.fy, info.cy, 0, 0, 1);
  for (int i = 0; i < 5; ++i) frame.distort_coeffs(i) = info.distortion[i];
  return true;
}

bool ShmClient::peek_gimbal_quat(Eigen::Quaterniond & q, uint64_t * out_seq) const
{
  if (!connected()) return false;
  auto & buf = static_cast<MetaPrefix *>(meta_->addr)->poses[static_cast<int>(PoseIndex::gimbal)];
  // 只读非 write_idx 的两个已提交槽位（ready 与 read_idx），前后校验 state 与 write_idx 一致，
  // 彻底避免读到仿真生产者正在半写中的脏槽位导致偶发姿态跳变/倒退。
  for (int attempt = 0; attempt < 6; ++attempt) {
    const uint8_t s1 = buf.ctrl.state.load(std::memory_order_acquire);
    const uint8_t w1 = buf.ctrl.write_idx;
    std::atomic_thread_fence(std::memory_order_acquire);
    uint64_t best_seq = 0;
    double bw = 1.0, bx = 0.0, by = 0.0, bz = 0.0;
    bool found = false;
    for (uint8_t i = 0; i < 3; ++i) {
      if (i == w1) continue;
      const auto & slot = buf.slots[i];
      const uint64_t seq1 = slot.frame_seq;
      std::atomic_thread_fence(std::memory_order_acquire);
      const double w = slot.quaternion[0];
      const double x = slot.quaternion[1];
      const double y = slot.quaternion[2];
      const double z = slot.quaternion[3];
      std::atomic_thread_fence(std::memory_order_acquire);
      const uint64_t seq2 = slot.frame_seq;
      if (seq1 == seq2 && seq1 > 0 && seq1 >= best_seq) {
        best_seq = seq1;
        bw = w;
        bx = x;
        by = y;
        bz = z;
        found = true;
      }
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    const uint8_t w2 = buf.ctrl.write_idx;
    const uint8_t s2 = buf.ctrl.state.load(std::memory_order_acquire);
    if (s1 == s2 && w1 == w2 && found) {
      const double n2 = bw * bw + bx * bx + by * by + bz * bz;
      if (n2 < 1e-6) return false;
      q = Eigen::Quaterniond(bw, bx, by, bz).normalized();
      if (out_seq) *out_seq = best_seq;
      return true;
    }
  }
  return false;
}

bool ShmClient::peek_odom_pos(Eigen::Vector3d & pos) const
{
  if (!connected()) return false;
  auto & buf = static_cast<MetaPrefix *>(meta_->addr)->poses[static_cast<int>(PoseIndex::odom)];
  for (int attempt = 0; attempt < 6; ++attempt) {
    const uint8_t s1 = buf.ctrl.state.load(std::memory_order_acquire);
    const uint8_t w1 = buf.ctrl.write_idx;
    std::atomic_thread_fence(std::memory_order_acquire);
    uint64_t best_seq = 0;
    double px = 0.0, py = 0.0, pz = 0.0;
    bool found = false;
    for (uint8_t i = 0; i < 3; ++i) {
      if (i == w1) continue;
      const auto & slot = buf.slots[i];
      const uint64_t seq1 = slot.frame_seq;
      std::atomic_thread_fence(std::memory_order_acquire);
      const double x = slot.position[0];
      const double y = slot.position[1];
      const double z = slot.position[2];
      std::atomic_thread_fence(std::memory_order_acquire);
      const uint64_t seq2 = slot.frame_seq;
      if (seq1 == seq2 && seq1 > 0 && seq1 >= best_seq) {
        best_seq = seq1;
        px = x;
        py = y;
        pz = z;
        found = true;
      }
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    const uint8_t w2 = buf.ctrl.write_idx;
    const uint8_t s2 = buf.ctrl.state.load(std::memory_order_acquire);
    if (s1 == s2 && w1 == w2 && found) {
      pos = Eigen::Vector3d(px, py, pz);
      return true;
    }
  }
  return false;
}

bool ShmClient::peek_ground_truth(std::vector<GroundTruthRobot> & out) const
{
  out.clear();
  if (!connected() || meta_->size < sizeof(MetaWithGt)) return false;
  const auto & gt = static_cast<const MetaWithGt *>(meta_->addr)->ground_truth;
  const uint32_t count =
    std::min<uint32_t>(gt.target_count, static_cast<uint32_t>(GROUND_TRUTH_MAX_TARGETS));
  if (count == 0) return false;
  out.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    const auto & src = gt.targets[i];
    GroundTruthRobot r;
    r.team = src.team;
    r.armor_label = src.armor_label;
    r.is_outpost = (src.is_outpost != 0);
    r.pos_world = Eigen::Vector3d(src.position[0], src.position[1], src.position[2]);
    r.yaw = static_cast<double>(src.yaw);
    r.vyaw = static_cast<double>(src.vyaw);
    out.push_back(r);
  }
  return true;
}

bool ShmClient::peek_sim_following() const
{
  if (!connected() || meta_->size < sizeof(MetaWithGt)) return false;
  return static_cast<const MetaWithGt *>(meta_->addr)->runtime_state.following != 0;
}

void ShmClient::send(const GimbalCmd & cmd)
{
  if (!connected()) return;
  auto & buf = static_cast<MetaPrefix *>(meta_->addr)->gimbal_cmd;

  // 本侧是该缓冲的唯一生产者，write_idx 由本侧独占，与仿真侧的消费者不冲突。
  auto & slot = buf.slots[buf.ctrl.write_idx];
  slot.timestamp_ns = cmd.timestamp_ns;
  slot.yaw_deg = cmd.yaw_deg;
  slot.pitch_deg = cmd.pitch_deg;
  slot.distance_m = cmd.distance_m;
  slot.fire_advice = cmd.fire_advice;

  uint8_t old = buf.ctrl.state.exchange(buf.ctrl.write_idx | FLAG_NEW, std::memory_order_acq_rel);
  buf.ctrl.write_idx = old & INDEX_MASK;
}

}  // namespace talos
}  // namespace io
