#ifndef IO__SIM__TALOS_SHM_HPP
#define IO__SIM__TALOS_SHM_HPP

#include <Eigen/Geometry>
#include <chrono>
#include <cstdint>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace io
{
namespace talos
{

// 与 bevy_robomaster_simulator/crates/talos-ipc/src/layout.rs 逐字节对齐。
// 仿真是生产者，本侧只读图像/位姿/内参，只写云台指令。改字段前先对那边的
// size_of 与 offset_of 断言，两边不一致时宁可拒绝连接也不要读错。

inline constexpr uint32_t SHM_MAGIC = 0x54414C05;
inline constexpr uint32_t SHM_VERSION = 2;

inline constexpr uint32_t IMAGE_WIDTH = 1440;
inline constexpr uint32_t IMAGE_HEIGHT = 1080;
inline constexpr uint32_t IMAGE_CHANNELS = 3;
inline constexpr std::size_t IMAGE_SIZE =
  static_cast<std::size_t>(IMAGE_WIDTH) * IMAGE_HEIGHT * IMAGE_CHANNELS;
inline constexpr std::size_t IMAGE_POOL_SIZE = IMAGE_SIZE * 3;

inline constexpr const char * SHM_NAME_META = "talos_ipc_meta";
inline constexpr const char * SHM_NAME_IMAGE_POOL = "talos_ipc_image_pool";

inline constexpr uint8_t FLAG_NEW = 0x80;
inline constexpr uint8_t INDEX_MASK = 0x03;

// 无解哨兵。仿真收到 distance_m == -1 会撤掉云台跟踪，PID 停止输出。
inline constexpr float NO_SOLUTION_DISTANCE = -1.f;

enum class PoseIndex : int
{
  gimbal = 0,
  odom = 1,
  muzzle = 2,
  camera = 3,
  chassis_observation = 4,
};

struct ImageMeta
{
  uint64_t seq;
  uint64_t timestamp_ns;
  uint32_t width;
  uint32_t height;
  uint8_t buffer_id;
  uint8_t format;
  uint8_t pad[6];
};

struct PoseMeta
{
  uint64_t frame_seq;
  float position[3];
  float quaternion[4];  // w, x, y, z，已经过仿真侧的 ROS 对齐
  uint64_t timestamp_ns;
  uint8_t pad[16];
};

struct GimbalCmd
{
  uint64_t timestamp_ns;
  float yaw_deg;
  float pitch_deg;  // 从竖直轴量起的度数，见仿真 from_solver_degrees
  float distance_m;
  uint8_t fire_advice;
  uint8_t pad[11];
};

struct CameraInfo
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

// 一帧仿真输入。图像是 BGR，时间戳来自仿真时钟而不是本机墙钟。
struct Frame
{
  cv::Mat bgr;
  uint64_t seq = 0;
  uint64_t timestamp_ns = 0;
  Eigen::Quaterniond gimbal_quat{1, 0, 0, 0};
  // 仿真里的云台位置（Odom 通道），ROS 世界系。视觉把云台当作原点，
  // 和真值比对时要用它把机器人位置换成相对云台的向量。
  Eigen::Vector3d gimbal_in_world{0, 0, 0};
  Eigen::Vector3d muzzle_in_gimbal{0, 0, 0};
  Eigen::Vector3d camera_in_gimbal{0, 0, 0};
  // 底盘观测（来自 ShmMetaRegion::chassis_observation）
  Eigen::Vector3d chassis_vel_world{0, 0, 0};  // 底盘在 ROS 世界系下的线速度 (m/s)
  Eigen::Vector3d chassis_rpy{0, 0, 0};        // 底盘欧拉角 [roll, pitch, yaw] (rad)
  double chassis_wz = 0.0;                     // 底盘偏航角速度 (rad/s)
  cv::Matx33d camera_matrix = cv::Matx33d::eye();
  cv::Matx<double, 1, 5> distort_coeffs = cv::Matx<double, 1, 5>::zeros();
};

// 仿真真值目标（仅用于评估统计，严禁输入给 EKF/Tracker/Planner）
struct GroundTruthRobot
{
  uint8_t team = 0;         // 0=Red, 1=Blue
  uint8_t armor_label = 0;  // 1=Hero, 3/4/5=Infantry
  bool is_outpost = false;
  Eigen::Vector3d pos_world{0, 0, 0};  // ROS 世界系机器人位置 (m)
  double yaw = 0.0;                    // ROS 世界系偏航角 (rad)
  double vyaw = 0.0;                   // 偏航角速度 (rad/s)
};

// 仿真共享内存的消费者。仿真未启动时 connect 失败，调用方自行重试。
class ShmClient
{
public:
  ShmClient();
  ~ShmClient();

  ShmClient(const ShmClient &) = delete;
  ShmClient & operator=(const ShmClient &) = delete;

  // 打开 /tmp 下的两块共享内存并校验 magic、version、尺寸。失败抛 std::runtime_error。
  void connect();

  bool connected() const;

  // 仿真心跳超过该时长视为掉线。仿真每帧更新 heartbeat_ns。
  bool alive(std::chrono::milliseconds max_age = std::chrono::milliseconds(500)) const;

  // 取最新一帧。没有新帧返回 false，不阻塞。
  bool recv(Frame & frame);

  // 非阻塞窥视最新已提交的炮口/云台姿态（不消费帧，供 100 Hz 控制线程零延迟读取）。
  bool peek_gimbal_quat(Eigen::Quaterniond & q, uint64_t * out_seq = nullptr) const;

  // 非阻塞窥视最新已提交的我方云台世界系位置（Odom 通道）。
  bool peek_odom_pos(Eigen::Vector3d & pos) const;

  // 非阻塞窥视仿真真值目标列表（仅用于离线/网页实际命中率评估，不参与自瞄解算）。
  bool peek_ground_truth(std::vector<GroundTruthRobot> & out) const;

  // 非阻塞窥视仿真是否开启了自瞄跟随/发弹订阅（ShmMetaRegion::runtime_state.following）。
  bool peek_sim_following() const;

  // 写云台指令。仿真侧只有一个消费者，本侧是唯一生产者。
  void send(const GimbalCmd & cmd);

private:
  struct Mapping;

  Mapping * meta_ = nullptr;
  Mapping * pool_ = nullptr;
  uint8_t image_read_idx_ = 2;
  uint8_t pose_read_idx_[5] = {2, 2, 2, 2, 2};

  void close();
};

}  // namespace talos
}  // namespace io

#endif  // IO__SIM__TALOS_SHM_HPP
