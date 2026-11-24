#ifndef IO__CAMERA_HPP
#define IO__CAMERA_HPP

#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

namespace io
{
class CameraBase
{
public:
  virtual ~CameraBase() = default;
  virtual void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) = 0;
  // 运行时可调的相机参数（默认空实现，供网页/调试界面实时调）
  virtual void set_exposure_ms(double) {}
  virtual void set_gain(double) {}
};

class Camera
{
public:
  Camera(const std::string & config_path);
  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp);
  void set_exposure_ms(double ms) { camera_->set_exposure_ms(ms); }
  void set_gain(double g) { camera_->set_gain(g); }

private:
  std::unique_ptr<CameraBase> camera_;
};

}  // namespace io

#endif  // IO__CAMERA_HPP