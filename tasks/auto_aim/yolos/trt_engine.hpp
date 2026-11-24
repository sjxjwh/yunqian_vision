#ifndef AUTO_AIM__TRT_ENGINE_HPP
#define AUTO_AIM__TRT_ENGINE_HPP

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <opencv2/opencv.hpp>

namespace auto_aim
{
/// 极简 TensorRT 推理封装：单输入，1~2 个输出，全部 float32（NMS 版还有一个 int32 的 num_dets）。
class TrtEngine
{
public:
  explicit TrtEngine(const std::string & engine_path);
  ~TrtEngine();

  TrtEngine(const TrtEngine &) = delete;
  TrtEngine & operator=(const TrtEngine &) = delete;

  /// 输入数据按 NCHW 连续存放，out 会被 resize 成输出尺寸（单输出引擎用）
  void infer(const float * input, size_t input_elems, std::vector<float> & out);

  /// 输入已经由 GPU 前处理直接写进设备缓冲区时用这个，省掉一次 H2D
  void infer_from_device(std::vector<float> & out);

  /// 多输出引擎：一次推理把所有输出拷回主机
  void infer_from_device_all();

  /// 设备端输入缓冲区指针（大小 = 3*640*640 个 float）
  float * input_device_ptr() const;

  /// 内部 CUDA stream（GPU 前处理可复用，避免额外同步）
  cudaStream_t stream() const;

  // ---- 输出描述 ----
  int nb_outputs() const;
  const std::vector<int> & output_shape(int i = 0) const;
  bool output_is_int32(int i) const;
  const std::vector<float> & output_float(int i);
  const std::vector<int32_t> & output_int(int i);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// BGR u8 (HxWx3) -> RGB float32 NCHW，归一化到 [0,1]，写入 dst（大小 3*H*W）
void bgr_to_rgb_chw(const cv::Mat & bgr, float * dst);

}  // namespace auto_aim

#endif  // AUTO_AIM__TRT_ENGINE_HPP
