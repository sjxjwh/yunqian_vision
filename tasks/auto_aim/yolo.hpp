#ifndef AUTO_AIM__YOLO_HPP
#define AUTO_AIM__YOLO_HPP

#include <list>

#include <opencv2/opencv.hpp>

#include "armor.hpp"

namespace auto_aim
{
class YOLOBase
{
public:
  virtual std::list<Armor> detect(const cv::Mat & img, int frame_count) = 0;

  virtual std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) = 0;

  /// 运行时调整：送去过滤前的最低置信度
  virtual void set_min_confidence(double) {}
  /// 运行时调整：识别平滑帧数（0/1 = 关闭）
  virtual void set_vote_frames(int) {}
};

class YOLO
{
public:
  YOLO(const std::string & config_path, bool debug = true);

  std::list<Armor> detect(const cv::Mat & img, int frame_count = -1);

  /// 在指定 ROI 内推理并自动将关键点/检测框还原至原图坐标系（兼容 OpenVINO / TensorRT）
  std::list<Armor> detect(const cv::Mat & img, const cv::Rect & roi, int frame_count = -1);

  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

  void set_min_confidence(double c) { yolo_->set_min_confidence(c); }
  void set_vote_frames(int n) { yolo_->set_vote_frames(n); }

private:
  std::unique_ptr<YOLOBase> yolo_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLO_HPP