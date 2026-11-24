#include "yolo.hpp"

#include <yaml-cpp/yaml.h>

#include "yolos/yolo11.hpp"
#include "yolos/yolov5.hpp"
#include "yolos/yolov8.hpp"

namespace auto_aim
{
YOLO::YOLO(const std::string & config_path, bool debug)
{
  auto yaml = YAML::LoadFile(config_path);
  auto yolo_name = yaml["yolo_name"].as<std::string>();

  if (yolo_name == "yolov8") {
    yolo_ = std::make_unique<YOLOV8>(config_path, debug);
  }

  else if (yolo_name == "yolo11") {
    yolo_ = std::make_unique<YOLO11>(config_path, debug);
  }

  else if (yolo_name == "yolov5") {
    yolo_ = std::make_unique<YOLOV5>(config_path, debug);
  }

  else {
    throw std::runtime_error("Unknown yolo name: " + yolo_name + "!");
  }
}

std::list<Armor> YOLO::detect(const cv::Mat & img, int frame_count)
{
  return yolo_->detect(img, frame_count);
}

std::list<Armor> YOLO::detect(const cv::Mat & img, const cv::Rect & roi, int frame_count)
{
  if (img.empty()) return {};
  const cv::Rect full_rect(0, 0, img.cols, img.rows);
  const cv::Rect safe_roi = roi & full_rect;
  if (
    safe_roi.width <= 0 || safe_roi.height <= 0 ||
    (safe_roi.x == 0 && safe_roi.y == 0 && safe_roi.width == img.cols &&
     safe_roi.height == img.rows)) {
    return yolo_->detect(img, frame_count);
  }

  cv::Mat crop = img(safe_roi).clone();
  auto armors = yolo_->detect(crop, frame_count);
  const cv::Point2f offset(static_cast<float>(safe_roi.x), static_cast<float>(safe_roi.y));
  for (auto & armor : armors) {
    for (auto & pt : armor.points) pt += offset;
    armor.center += offset;
    armor.center_norm = cv::Point2f(armor.center.x / img.cols, armor.center.y / img.rows);
    armor.box.x += safe_roi.x;
    armor.box.y += safe_roi.y;
    armor.left.center += offset;
    armor.left.top += offset;
    armor.left.bottom += offset;
    for (auto & pt : armor.left.points) pt += offset;
    armor.right.center += offset;
    armor.right.top += offset;
    armor.right.bottom += offset;
    for (auto & pt : armor.right.points) pt += offset;
  }
  return armors;
}

std::list<Armor> YOLO::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return yolo_->postprocess(scale, output, bgr_img, frame_count);
}

}  // namespace auto_aim