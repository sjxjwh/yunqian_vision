// 仿真姿态/世界系探针：读共享内存，跑检测+解算+跟踪，打印
//   - 跟踪目标的 EKF 中心在世界系的 distance / bearing
//   - EKF 装甲板重投影回图像后的四角平均像素误差
//   - 同一帧仿真的真值（GroundTruthBatch）相对云台的距离 / bearing
// 只用于验收，不进入 standard / sim_standard 的运行路径。

#include <fcntl.h>
#include <yaml-cpp/yaml.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "io/sim/sim_io.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"

namespace
{

constexpr std::size_t GT_OFFSET = 1984;         // layout.rs: ShmMetaRegion.ground_truth
constexpr std::size_t GT_TARGETS_OFFSET = 32;   // targets 按 32 字节对齐
constexpr std::size_t GT_TARGET_SIZE = 64;
constexpr std::size_t GT_MAX_TARGETS = 16;

struct GtTarget
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
static_assert(sizeof(GtTarget) == GT_TARGET_SIZE, "GroundTruthTarget layout");

struct GtSnapshot
{
  uint64_t frame_seq = 0;
  std::vector<GtTarget> targets;
};

// 直接 pread 真值区。仿真侧是覆盖写、带撕裂风险，读两遍取一致的那份。
GtSnapshot read_gt(int fd)
{
  GtSnapshot best;
  for (int attempt = 0; attempt < 3; ++attempt) {
    uint8_t buf[GT_TARGETS_OFFSET + GT_MAX_TARGETS * GT_TARGET_SIZE];
    auto n = pread(fd, buf, sizeof(buf), GT_OFFSET);
    if (n != static_cast<ssize_t>(sizeof(buf))) return best;

    uint64_t seq;
    uint32_t target_count;
    std::memcpy(&seq, buf, sizeof(seq));
    std::memcpy(&target_count, buf + 16, sizeof(target_count));
    if (target_count > GT_MAX_TARGETS) return best;

    GtSnapshot snapshot;
    snapshot.frame_seq = seq;
    snapshot.targets.resize(target_count);
    std::memcpy(
      snapshot.targets.data(), buf + GT_TARGETS_OFFSET, target_count * GT_TARGET_SIZE);
    return snapshot;
  }
  return best;
}

double deg(double rad) { return rad * 180.0 / M_PI; }

double distance_xy(const Eigen::Vector3d & v) { return std::hypot(v.x(), v.y()); }

double bearing_xy(const Eigen::Vector3d & v) { return deg(std::atan2(v.y(), v.x())); }

// EKF 的每块装甲板都重投影一次，取和当前检测到的装甲板最接近的那块。
// 四角平均像素误差，单位 px。同时带回这块装甲板的世界坐标，便于和真值比。
struct ReprojectionMatch
{
  double error = 1e9;
  Eigen::Vector4d xyza = Eigen::Vector4d::Zero();
  const auto_aim::Armor * armor = nullptr;
};

ReprojectionMatch best_reprojection(
  const auto_aim::Solver & solver, const auto_aim::Target & target,
  const std::list<auto_aim::Armor> & armors)
{
  ReprojectionMatch best;
  // 带上 EKF 的环 roll/pitch。只比 yaw 会把倾斜的板当成重投影误差。
  auto xyz_list = target.armor_xyzayp_list();
  for (const auto & xyza : xyz_list) {
    auto projected = solver.reproject_armor(
      xyza.head(3), xyza[3], target.armor_type, target.name, xyza[4], xyza[5]);
    if (projected.size() != 4) continue;
    for (const auto & armor : armors) {
      if (armor.points.size() != 4) continue;
      double error = 0;
      for (int i = 0; i < 4; ++i) error += cv::norm(projected[i] - armor.points[i]);
      error /= 4.0;
      if (error < best.error) {
        best.error = error;
        best.xyza = xyza.head(4);
        best.armor = &armor;
      }
    }
  }
  return best;
}

}  // namespace

int main(int argc, char * argv[])
{
  const std::string keys =
    "{help h usage ? |                        | 输出命令行参数说明}"
    "{@config-path   | configs/sim.yaml       | yaml 配置路径}"
    "{frames f       | 600                    | 最多处理多少帧}"
    "{target t       |                        | 只保留指定装甲名（one/three），空为全部}";

  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }
  int frames = cli.get<int>("frames");
  auto target_filter = cli.get<std::string>("target");

  tools::Exiter exiter;
  io::SimCamera camera(config_path);
  auto_aim::YOLO detector(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);

  int gt_fd = ::open("/tmp/talos_ipc_meta", O_RDONLY);
  if (gt_fd < 0) {
    std::cerr << "打不开 /tmp/talos_ipc_meta，仿真是否已启动\n";
    return 1;
  }

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;

  for (int frame = 0; frame < frames && !exiter.exit(); ++frame) {
    camera.read(img, timestamp);
    solver.set_R_gimbal2world(camera.gimbal_quaternion());

    auto armors = detector.detect(img);
    if (!target_filter.empty()) {
      armors.remove_if([&](const auto_aim::Armor & armor) {
        return auto_aim::ARMOR_NAMES[armor.name] != target_filter;
      });
    }
    auto targets = tracker.track(armors, timestamp, true);
    auto gt = read_gt(gt_fd);

    if (targets.empty()) {
      std::cout << "[probe] frame=" << frame << " no target, detections=" << armors.size()
                << "\n";
      continue;
    }

    auto & target = targets.front();
    auto state = target.ekf_x();
    Eigen::Vector3d center(state[0], state[2], state[4]);
    auto match = best_reprojection(solver, target, armors);
    Eigen::Vector3d armor_xyz = match.xyza.head(3);
    auto gimbal = camera.gimbal_in_world();
    double pnp_range = match.xyza.head(3).norm();

    std::cout << "[probe] frame=" << frame << " name=" << auto_aim::ARMOR_NAMES[target.name]
              << " converged=" << target.convergened() << " ekf_dist=" << distance_xy(center)
              << " ekf_bearing=" << bearing_xy(center) << " armor_dist=" << distance_xy(armor_xyz)
              << " armor_bearing=" << bearing_xy(armor_xyz) << " reproj_px=" << match.error
              << " pnp_range=" << pnp_range << " r=" << state[8] << " w=" << state[7]
              << " l=" << state[9] << " h=" << state[10]
              << " roll=" << (state.size() > 11 ? state[11] : 0)
              << " pitch=" << (state.size() > 12 ? state[12] : 0)
              << " jumped=" << target.jumped
              << " last_id=" << target.last_id
              << " det_yaw=" << (match.armor ? match.armor->ypr_in_world[0] : 0.0)
              << " gimbal=(" << gimbal.x() << "," << gimbal.y() << "," << gimbal.z() << ")";

    for (const auto & t : gt.targets) {
      if (t.team != 1) continue;  // 仿真里敌方是蓝队
      Eigen::Vector3d pos(t.position[0], t.position[1], t.position[2]);
      Eigen::Vector3d rel = pos - gimbal;
      std::cout << " gt[label=" << int(t.armor_label) << "] dist=" << distance_xy(rel)
                << " bearing=" << bearing_xy(rel) << " yaw=" << t.yaw << " vyaw=" << t.vyaw;
    }
    std::cout << "\n";
  }

  ::close(gt_fd);
  return 0;
}
