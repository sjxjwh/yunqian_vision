#ifndef AUTO_AIM__YOLO11_HPP
#define AUTO_AIM__YOLO11_HPP

#include <chrono>
#include <list>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"

#ifdef SP_WITH_TENSORRT
#include <memory>

#include "tasks/auto_aim/yolos/trt_engine.hpp"
#endif

namespace auto_aim
{
class YOLO11 : public YOLOBase
{
public:
  YOLO11(const std::string & config_path, bool debug);

  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count) override;

  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
  std::string device_, model_path_;
  std::string save_path_, debug_path_;
  bool debug_, use_roi_;

  const int class_num_ = 38;
  const float nms_threshold_ = 0.3;
  const float score_threshold_ = 0.7;
  double min_confidence_, binary_threshold_;

  ov::Core core_;
  ov::CompiledModel compiled_model_;
  ov::InferRequest infer_request_;

#ifdef SP_WITH_TENSORRT
  // Jetson 上用 TensorRT 推理（配置了 trt_engine_path 时生效），否则走 OpenVINO
  std::unique_ptr<TrtEngine> trt_;
  std::vector<float> trt_input_;
#endif

  // 预处理是否走 GPU（编译带 CUDA 时有效）、是否打印分段耗时
  bool use_cuda_ = true;
  bool profile_ = false;
  int profile_count_ = 0;
  double profile_pre_ = 0, profile_infer_ = 0, profile_parse_ = 0;
  void profile_report(
    std::chrono::steady_clock::time_point t0, std::chrono::steady_clock::time_point t1,
    std::chrono::steady_clock::time_point t2, std::chrono::steady_clock::time_point t3);

  cv::Rect roi_;
  cv::Point2f offset_;
  cv::Mat tmp_img_;

  Detector detector_;

  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;

  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  std::list<Armor> parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

  void save(const Armor & armor) const;
  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;
  void sort_keypoints(std::vector<cv::Point2f> & keypoints);
};

}  // namespace auto_aim

#endif  //AUTO_AIM__YOLO11_HPP