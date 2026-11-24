#include "hikrobot.hpp"

#include <libusb-1.0/libusb.h>

#include "tools/logger.hpp"
#include "tools/runtime_params.hpp"

using namespace std::chrono_literals;

namespace io
{
HikRobot::HikRobot(
  double exposure_ms, double gain, const std::string & vid_pid, double frame_rate)
: exposure_us_(exposure_ms * 1e3),
  gain_(gain),
  frame_rate_(frame_rate),
  queue_(1),
  daemon_quit_(false),
  vid_(-1),
  pid_(-1)
{
  set_vid_pid(vid_pid);
  if (libusb_init(NULL)) tools::logger()->warn("Unable to init libusb!");

  // 注册运行期可调参数（网页 / 调试工具可改）
  auto & rp = tools::RuntimeParams::instance();
  rp.add("camera.exposure_ms", "曝光 (ms)", 0.1, 30.0, [this] { return exposure_us_ / 1e3; },
         [this](double v) { set_exposure_ms(v); }, "exposure_ms");
  rp.add("camera.gain", "增益", 0.0, 20.0, [this] { return gain_; },
         [this](double v) { set_gain(v); }, "gain");
  rp.add("camera.frame_rate", "相机出图帧率上限", 10.0, 250.0, [this] { return frame_rate_; },
         [this](double v) { set_frame_rate(v); }, "frame_rate");

  daemon_thread_ = std::thread{[this] {
    tools::logger()->info("HikRobot's daemon thread started.");

    capture_start();

    while (!daemon_quit_) {
      std::this_thread::sleep_for(100ms);

      if (capturing_) continue;

      capture_stop();
      reset_usb();
      capture_start();
    }

    capture_stop();

    tools::logger()->info("HikRobot's daemon thread stopped.");
  }};
}

HikRobot::~HikRobot()
{
  daemon_quit_ = true;
  if (daemon_thread_.joinable()) daemon_thread_.join();
  tools::logger()->info("HikRobot destructed.");
}

void HikRobot::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  CameraData data;
  queue_.pop(data);

  img = data.img;
  timestamp = data.timestamp;
}

void HikRobot::set_exposure_ms(double ms)
{
  exposure_us_ = ms * 1e3;
  if (!handle_) return;
  auto ret = MV_CC_SetFloatValue(handle_, "ExposureTime", exposure_us_);
  if (ret != MV_OK) tools::logger()->warn("设置曝光失败: {:#x}", ret);
}

void HikRobot::set_gain(double g)
{
  gain_ = g;
  if (!handle_) return;
  auto ret = MV_CC_SetFloatValue(handle_, "Gain", gain_);
  if (ret != MV_OK) tools::logger()->warn("设置增益失败: {:#x}", ret);
}

void HikRobot::set_frame_rate(double fps)
{
  frame_rate_ = fps;
  if (!handle_) return;
  auto ret = MV_CC_SetFrameRate(handle_, static_cast<float>(frame_rate_));
  if (ret != MV_OK) tools::logger()->warn("设置帧率失败: {:#x}", ret);
}

void HikRobot::capture_start()
{
  capturing_ = false;
  capture_quit_ = false;

  unsigned int ret;

  MV_CC_DEVICE_INFO_LIST device_list;
  ret = MV_CC_EnumDevices(MV_USB_DEVICE, &device_list);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_EnumDevices failed: {:#x}", ret);
    return;
  }

  if (device_list.nDeviceNum == 0) {
    tools::logger()->warn("Not found camera!");
    return;
  }

  ret = MV_CC_CreateHandle(&handle_, device_list.pDeviceInfo[0]);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_CreateHandle failed: {:#x}", ret);
    return;
  }

  ret = MV_CC_OpenDevice(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_OpenDevice failed: {:#x}", ret);
    return;
  }

  set_enum_value("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS);
  set_enum_value("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
  set_enum_value("GainAuto", MV_GAIN_MODE_OFF);
  set_float_value("ExposureTime", exposure_us_);
  set_float_value("Gain", gain_);
  MV_CC_SetFrameRate(handle_, static_cast<float>(frame_rate_));  // 出图帧率上限（yaml: frame_rate）

  ret = MV_CC_StartGrabbing(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_StartGrabbing failed: {:#x}", ret);
    return;
  }

  capture_thread_ = std::thread{[this] {
    tools::logger()->info("HikRobot's capture thread started.");

    capturing_ = true;

    MV_FRAME_OUT raw;
    MV_CC_PIXEL_CONVERT_PARAM cvt_param;

    while (!capture_quit_) {
      std::this_thread::sleep_for(1ms);

      unsigned int ret;
      unsigned int nMsec = 100;

      ret = MV_CC_GetImageBuffer(handle_, &raw, nMsec);
      if (ret != MV_OK) {
        tools::logger()->warn("MV_CC_GetImageBuffer failed: {:#x}", ret);
        break;
      }

      auto timestamp = std::chrono::steady_clock::now();
      cv::Mat img(cv::Size(raw.stFrameInfo.nWidth, raw.stFrameInfo.nHeight), CV_8U, raw.pBufAddr);

      cvt_param.nWidth = raw.stFrameInfo.nWidth;
      cvt_param.nHeight = raw.stFrameInfo.nHeight;

      cvt_param.pSrcData = raw.pBufAddr;
      cvt_param.nSrcDataLen = raw.stFrameInfo.nFrameLen;
      cvt_param.enSrcPixelType = raw.stFrameInfo.enPixelType;

      cvt_param.pDstBuffer = img.data;
      cvt_param.nDstBufferSize = img.total() * img.elemSize();
      cvt_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed;

      // ret = MV_CC_ConvertPixelType(handle_, &cvt_param);
      const auto & frame_info = raw.stFrameInfo;
      auto pixel_type = frame_info.enPixelType;
      cv::Mat dst_image;
      const static std::unordered_map<MvGvspPixelType, cv::ColorConversionCodes> type_map = {
        {PixelType_Gvsp_BayerGR8, cv::COLOR_BayerGR2RGB},
        {PixelType_Gvsp_BayerRG8, cv::COLOR_BayerRG2RGB},
        {PixelType_Gvsp_BayerGB8, cv::COLOR_BayerGB2RGB},
        {PixelType_Gvsp_BayerBG8, cv::COLOR_BayerBG2RGB}};
      cv::cvtColor(img, dst_image, type_map.at(pixel_type));
      img = dst_image;

      queue_.push({img, timestamp});

      ret = MV_CC_FreeImageBuffer(handle_, &raw);
      if (ret != MV_OK) {
        tools::logger()->warn("MV_CC_FreeImageBuffer failed: {:#x}", ret);
        break;
      }
    }

    capturing_ = false;
    tools::logger()->info("HikRobot's capture thread stopped.");
  }};
}

void HikRobot::capture_stop()
{
  capture_quit_ = true;
  if (capture_thread_.joinable()) capture_thread_.join();

  unsigned int ret;

  ret = MV_CC_StopGrabbing(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_StopGrabbing failed: {:#x}", ret);
    return;
  }

  ret = MV_CC_CloseDevice(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_CloseDevice failed: {:#x}", ret);
    return;
  }

  ret = MV_CC_DestroyHandle(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_DestroyHandle failed: {:#x}", ret);
    return;
  }
}

void HikRobot::set_float_value(const std::string & name, double value)
{
  unsigned int ret;

  ret = MV_CC_SetFloatValue(handle_, name.c_str(), value);

  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_SetFloatValue(\"{}\", {}) failed: {:#x}", name, value, ret);
    return;
  }
}

void HikRobot::set_enum_value(const std::string & name, unsigned int value)
{
  unsigned int ret;

  ret = MV_CC_SetEnumValue(handle_, name.c_str(), value);

  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_SetEnumValue(\"{}\", {}) failed: {:#x}", name, value, ret);
    return;
  }
}

void HikRobot::set_vid_pid(const std::string & vid_pid)
{
  auto index = vid_pid.find(':');
  if (index == std::string::npos) {
    tools::logger()->warn("Invalid vid_pid: \"{}\"", vid_pid);
    return;
  }

  auto vid_str = vid_pid.substr(0, index);
  auto pid_str = vid_pid.substr(index + 1);

  try {
    vid_ = std::stoi(vid_str, 0, 16);
    pid_ = std::stoi(pid_str, 0, 16);
  } catch (const std::exception &) {
    tools::logger()->warn("Invalid vid_pid: \"{}\"", vid_pid);
  }
}

void HikRobot::reset_usb() const
{
  if (vid_ == -1 || pid_ == -1) return;

  // https://github.com/ralight/usb-reset/blob/master/usb-reset.c
  auto handle = libusb_open_device_with_vid_pid(NULL, vid_, pid_);
  if (!handle) {
    tools::logger()->warn("Unable to open usb!");
    return;
  }

  if (libusb_reset_device(handle))
    tools::logger()->warn("Unable to reset usb!");
  else
    tools::logger()->info("Reset usb successfully :)");

  libusb_close(handle);
}

}  // namespace io