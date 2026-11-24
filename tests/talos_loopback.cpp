// 按仿真 talos-ipc 的布局起一个最小生产者，验证 io/sim 的读写。
// 不依赖 OpenCV / Eigen / spdlog，只验证共享内存契约本身。
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

namespace
{
constexpr uint32_t MAGIC = 0x54414C05;
constexpr uint32_t VERSION = 2;
constexpr uint32_t WIDTH = 1440;
constexpr uint32_t HEIGHT = 1080;
constexpr size_t IMAGE_SIZE = (size_t)WIDTH * HEIGHT * 3;
constexpr uint8_t FLAG_NEW = 0x80;
constexpr uint8_t INDEX_MASK = 0x03;

struct alignas(32) ImageMeta
{
  uint64_t seq, timestamp_ns;
  uint32_t width, height;
  uint8_t buffer_id, format, pad[6];
};
struct alignas(64) PoseMeta
{
  uint64_t frame_seq;
  float position[3];
  float quaternion[4];
  uint64_t timestamp_ns;
  uint8_t pad[16];
};
struct alignas(32) GimbalCmd
{
  uint64_t timestamp_ns;
  float yaw_deg, pitch_deg, distance_m;
  uint8_t fire_advice, pad[11];
};
struct alignas(64) CameraInfo
{
  uint64_t timestamp_ns;
  double fx, fy, cx, cy, distortion[5];
  uint32_t width, height;
  uint8_t pad[24];
};
struct alignas(64) Ctrl
{
  std::atomic<uint8_t> state;
  uint8_t write_idx, read_idx, pad[61];
};
struct alignas(64) ImageBuf
{
  Ctrl ctrl;
  ImageMeta slots[3];
};
struct alignas(64) PoseBuf
{
  Ctrl ctrl;
  PoseMeta slots[3];
};
struct alignas(64) GimbalBuf
{
  Ctrl ctrl;
  GimbalCmd slots[3];
};
struct alignas(64) Header
{
  uint32_t magic, version;
  uint64_t created_ns, heartbeat_ns;
  uint32_t image_width, image_height;
  uint8_t pad[32];
};
struct Meta
{
  Header header;
  ImageBuf image;
  PoseBuf poses[5];
  GimbalBuf gimbal_cmd;
  CameraInfo camera_info;
};

static_assert(sizeof(Meta) == 1856, "meta size");
static_assert(offsetof(Meta, camera_info) == 1728, "camera_info offset");

void publish(Ctrl & ctrl)
{
  uint8_t old = ctrl.state.exchange(ctrl.write_idx | FLAG_NEW, std::memory_order_acq_rel);
  ctrl.write_idx = old & INDEX_MASK;
}

template <typename Slot>
const Slot * consume(Ctrl & ctrl, Slot (&slots)[3], uint8_t & read_idx)
{
  uint8_t expected = ctrl.state.load(std::memory_order_acquire);
  if ((expected & FLAG_NEW) == 0) return nullptr;
  uint8_t ready = expected & INDEX_MASK;
  if (ctrl.state.compare_exchange_strong(
        expected, read_idx, std::memory_order_acq_rel, std::memory_order_acquire)) {
    read_idx = ready;
    return &slots[ready];
  }
  return nullptr;
}

int fail(const std::string & msg)
{
  std::cerr << "FAIL: " << msg << "\n";
  return 1;
}
}  // namespace

int main()
{
  auto now = [] {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
  };

  int meta_fd = ::open("/tmp/talos_ipc_meta", O_RDWR | O_CREAT | O_TRUNC, 0666);
  int pool_fd = ::open("/tmp/talos_ipc_image_pool", O_RDWR | O_CREAT | O_TRUNC, 0666);
  if (meta_fd < 0 || pool_fd < 0) return fail("open");
  if (ftruncate(meta_fd, sizeof(Meta)) || ftruncate(pool_fd, IMAGE_SIZE * 3)) return fail("truncate");
  auto meta = (Meta *)mmap(nullptr, sizeof(Meta), PROT_READ | PROT_WRITE, MAP_SHARED, meta_fd, 0);
  auto pool =
    (uint8_t *)mmap(nullptr, IMAGE_SIZE * 3, PROT_READ | PROT_WRITE, MAP_SHARED, pool_fd, 0);
  std::memset(meta, 0, sizeof(Meta));

  meta->header.magic = MAGIC;
  meta->header.version = VERSION;
  meta->header.image_width = WIDTH;
  meta->header.image_height = HEIGHT;
  meta->header.heartbeat_ns = now();
  meta->image.ctrl.state.store(1, std::memory_order_relaxed);
  for (auto & pose : meta->poses) pose.ctrl.state.store(1, std::memory_order_relaxed);
  meta->gimbal_cmd.ctrl.state.store(1, std::memory_order_relaxed);

  meta->camera_info.fx = 1304.673;
  meta->camera_info.fy = 1304.673;
  meta->camera_info.cx = 720;
  meta->camera_info.cy = 540;
  meta->camera_info.width = WIDTH;
  meta->camera_info.height = HEIGHT;

  for (size_t i = 0; i < IMAGE_SIZE; i += 3) pool[i] = 255;

  auto & img = meta->image.slots[meta->image.ctrl.write_idx];
  img.seq = 7;
  img.timestamp_ns = 1000000000;
  img.width = WIDTH;
  img.height = HEIGHT;
  img.buffer_id = 0;
  publish(meta->image.ctrl);

  auto & gimbal = meta->poses[0].slots[meta->poses[0].ctrl.write_idx];
  gimbal.quaternion[0] = 1;
  gimbal.frame_seq = 7;
  publish(meta->poses[0].ctrl);

  std::cout << "producer ready, meta=" << sizeof(Meta) << "\n";

  // 持续发帧并维持心跳。只发一帧的话，消费者读完就再也等不到新帧，
  // 心跳一过期就判定掉线，回环测不出连续跟踪。
  uint8_t read_idx = 2;
  uint64_t seq = 7;
  int received = 0;
  const GimbalCmd * first = nullptr;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(12);
  while (std::chrono::steady_clock::now() < deadline) {
    meta->header.heartbeat_ns = now();
    seq++;
    auto & slot = meta->image.slots[meta->image.ctrl.write_idx];
    slot.seq = seq;
    slot.timestamp_ns = 1000000000 + seq * 10000000;
    slot.width = WIDTH;
    slot.height = HEIGHT;
    slot.buffer_id = 0;
    publish(meta->image.ctrl);

    if (auto cmd = consume(meta->gimbal_cmd.ctrl, meta->gimbal_cmd.slots, read_idx)) {
      received++;
      if (first == nullptr) first = cmd;
    }
    usleep(10000);
  }
  if (received == 0) return fail("no gimbal command received");
  std::cout << "frames=" << received << "\n";

  auto cmd = first;
  std::cout << "recv yaw=" << cmd->yaw_deg << " pitch=" << cmd->pitch_deg
            << " dist=" << cmd->distance_m << " fire=" << (int)cmd->fire_advice << "\n";

  bool ok = true;
  auto expect = [&](const char * name, double got, double want) {
    if (std::abs(got - want) > 1e-3) {
      std::cerr << "FAIL " << name << ": got " << got << " want " << want << "\n";
      ok = false;
    }
  };
  // 回环图里没有装甲板，自瞄给出 control=false。此时只要求 distance=-1，
  // 仿真据此停止云台跟踪，yaw/pitch 不使用。
  expect("yaw", cmd->yaw_deg, 0);
  expect("distance", cmd->distance_m, -1);

  munmap(meta, sizeof(Meta));
  munmap(pool, IMAGE_SIZE * 3);
  close(meta_fd);
  close(pool_fd);
  return ok ? 0 : 1;
}
