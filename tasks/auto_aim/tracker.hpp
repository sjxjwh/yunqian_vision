#ifndef AUTO_AIM__TRACKER_HPP
#define AUTO_AIM__TRACKER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <list>
#include <string>

#include "armor.hpp"
#include "solver.hpp"
#include "target.hpp"
#include "tasks/omniperception/perceptron.hpp"
#include "tools/thread_safe_queue.hpp"

namespace auto_aim
{
class Tracker
{
public:
  Tracker(const std::string & config_path, Solver & solver);

  std::string state() const;

  std::list<Target> track(
    std::list<Armor> & armors, std::chrono::steady_clock::time_point t,
    bool use_enemy_color = true, const Eigen::Vector3d & self_vel = Eigen::Vector3d::Zero());

  std::tuple<omniperception::DetectionResult, std::list<Target>> track(
    const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
    std::chrono::steady_clock::time_point t, bool use_enemy_color = true);

  // 基于当前整车跟踪状态返回网络聚焦 ROI（未收敛/丢失时返回全图）
  cv::Rect focus_roi(
    const cv::Size & img_size, std::chrono::steady_clock::time_point now,
    const Eigen::Vector3d & self_vel = Eigen::Vector3d::Zero()) const;

private:
  Solver & solver_;
  Color enemy_color_;
  bool use_uvl_ekf_ = true;
  bool use_focus_roi_ = true;
  int min_detect_count_;
  int max_temp_lost_count_;
  int detect_count_;
  int temp_lost_count_;
  int outpost_max_temp_lost_count_;
  int normal_temp_lost_count_;
  std::string state_, pre_state_;
  Target target_;
  std::chrono::steady_clock::time_point last_timestamp_;
  ArmorPriority omni_target_priority_;

  void state_machine(bool found);

  bool set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);

  bool update_target(
    std::list<Armor> & armors, std::chrono::steady_clock::time_point t,
    const Eigen::Vector3d & self_vel = Eigen::Vector3d::Zero());
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TRACKER_HPP