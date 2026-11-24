#ifndef AUTO_AIM__CUDA_PREPROCESS_HPP
#define AUTO_AIM__CUDA_PREPROCESS_HPP

#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>

namespace auto_aim::cuda_ops
{
/// CUDA 是否可用（编译进来了且设备可用）
bool available();

/// 检测预处理：3 通道 u8 图（任意尺寸）-> letterbox 到 dw×dh -> RGB f32 CHW（/255），
/// 结果直接写进 device 指针 d_dst（大小 3*dw*dh 个 float）。
/// 几何与 cpu 版一致：scale = min(dw/cols, dh/rows)，缩放后贴左上角，其余补 0。
void letterbox_to_chw(const cv::Mat & src, float * d_dst, int dw, int dh, cudaStream_t stream = 0);

/// 相机去马赛克：BayerRG8(u8, HxW) -> 3 通道 u8，通道顺序与 cv::COLOR_BayerRG2RGB 一致
void bayer_rg_to_rgb(const cv::Mat & bayer, cv::Mat & rgb);

}  // namespace auto_aim::cuda_ops

#endif  // AUTO_AIM__CUDA_PREPROCESS_HPP
