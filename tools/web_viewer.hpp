#ifndef TOOLS__WEB_VIEWER_HPP
#define TOOLS__WEB_VIEWER_HPP

#include <opencv2/core.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace tools
{

// 一帧重建所需的状态。坐标是视觉世界系：x 向前、y 向左、z 向上，单位米。
struct ViewerFrame
{
  uint64_t stamp_ns = 0;

  // 云台朝向，ZYX 欧拉角，弧度（用于相机投影）。
  double gimbal_yaw = 0;
  double gimbal_pitch = 0;
  // 真实炮口朝向，ZYX 欧拉角，弧度（用于 3D 炮管与曲线）。
  double muzzle_yaw = 0;
  double muzzle_pitch = 0;

  // 跟踪目标。tracked 为 false 时右侧只画本车。
  bool tracked = false;
  std::string target_name;
  // 每块装甲板：x y z yaw，后两维是装甲环 roll/pitch。旧的四维数据 roll/pitch 为 0。
  std::vector<std::array<double, 6>> armors;
  // 和 armors 一一对应的四角像素坐标（x0 y0 ... x3 y3），由 C++ 的
  // Solver::reproject_armor 算好，左图直接画，避免网页端重复推相机。
  std::vector<std::array<double, 8>> armor_pixels;
  // 当前正在被观测/更新的装甲板在 armors 里的下标。-1 表示未知。
  int current_armor = -1;
  // 本帧检测到的装甲板四角（图像像素，x0 y0 x1 y1 ... x3 y3）。
  // 左图直接画这些像素点，和 EKF 的预测框区分开。
  std::vector<std::array<double, 8>> detections;
  // 瞄准点。valid 为 false 时不画。
  bool aim_valid = false;
  double aim_x = 0;
  double aim_y = 0;
  double aim_z = 0;

  double command_yaw = 0;
  double command_pitch = 0;
  // 本帧时刻的目标理想弹道角（与 muzzle_yaw 同相，用于曲线对比与跟踪误差统计）
  double ref_yaw = 0;
  double ref_pitch = 0;
  bool control = false;
  bool shoot = false;

  // 闭环定量评估指标（可选，未启用时保持 0）
  uint32_t shots = 0;
  uint32_t hits = 0;
  uint32_t resolved_shots = 0;
  double hit_rate = 0.0;
  double fire_duty = 0.0;
  double yaw_err_rms = 0.0;  // rad
  // 仿真实际命中率（仅仿真评估用：真实炮口朝向 × 仿真真值目标）
  bool has_sim_metrics = false;
  uint32_t sim_shots = 0;
  uint32_t sim_hits = 0;
  uint32_t sim_resolved_shots = 0;
  double sim_hit_rate = 0.0;
  // 飞行中的虚拟弹丸世界坐标 [x, y, z]
  std::vector<std::array<double, 3>> bullets;

  // 我方云台在真实世界系的位置（用于 3D 场景地面滚动与我方运动轨迹；实机可接底盘里程计）
  std::array<double, 3> self_pos{0.0, 0.0, 0.0};
  // EKF 估计的敌方旋转中心、线速度与角速度（用于 3D 骨架、速度箭头与轨迹尾迹）
  std::array<double, 3> target_center{0.0, 0.0, 0.0};
  std::array<double, 3> target_vel{0.0, 0.0, 0.0};
  double target_w = 0.0;
  bool target_is_big = false;

  // 相机与枪口相对云台原点的机械坐标（来自 URDF/YAML/仿真，供 3D 重建真实装配）
  std::array<double, 3> camera_in_gimbal{0.06, 0.0, 0.095};
  std::array<double, 3> muzzle_in_gimbal{0.15, 0.0, 0.097};

  // 最近命中事件（用于 3D/2D 命中冲击波与装甲板受击闪光）
  struct HitEvent
  {
    uint32_t id = 0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    int armor_id = -1;
  };
  std::vector<HitEvent> hit_events;

  // 视觉弹道闭环（AimCorrector::status().dump()），为空表示未启用
  std::string corrector_json;
};

// 本地网页：左图是相机画面，右图是用视觉估计重建的 3D 场景。
// 浏览器打开 http://127.0.0.1:<port>。页面与 three.js 从 web/ 目录提供，不依赖外网。
class WebViewer
{
public:
  using ApiHandler = std::function<std::string(const std::string & path)>;

  explicit WebViewer(uint16_t port = 8090, std::string web_root = "web");
  ~WebViewer();

  WebViewer(const WebViewer &) = delete;
  WebViewer & operator=(const WebViewer &) = delete;

  // 注册 /api/... 轻量 HTTP JSON 回调（如网页端点击应用/清空弹道闭环零偏）
  void set_api_handler(ApiHandler handler);

  // 当前是否有浏览器 WebSocket 客户端正在订阅
  bool has_clients() const;

  // 推一帧。相机图会被压成 JPEG。调用方可在任意线程调用。
  void publish(const cv::Mat & bgr, const ViewerFrame & frame);

private:
  struct Impl;
  Impl * impl_;
};

}  // namespace tools

#endif  // TOOLS__WEB_VIEWER_HPP
