#include "cuda_preprocess.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace auto_aim::cuda_ops
{
namespace
{
// ---------------- 设备缓冲区（进程内复用） ----------------
struct Buffers
{
  unsigned char * d_u8 = nullptr;
  size_t u8_cap = 0;
  float * d_f32 = nullptr;
  size_t f32_cap = 0;
  unsigned char * d_out_u8 = nullptr;
  size_t out_cap = 0;
  unsigned char * h_pinned = nullptr;   // page-locked 暂存，避免 pageable 拷贝把 stream 卡住
  size_t pinned_cap = 0;
};

Buffers & buffers()
{
  static Buffers b;
  return b;
}

void ensure_u8(size_t bytes)
{
  auto & b = buffers();
  if (b.u8_cap >= bytes) return;
  if (b.d_u8) cudaFree(b.d_u8);
  if (cudaMalloc(&b.d_u8, bytes) != cudaSuccess)
    throw std::runtime_error("cudaMalloc(u8) 失败");
  b.u8_cap = bytes;
}

void ensure_pinned(size_t bytes)
{
  auto & b = buffers();
  if (b.pinned_cap >= bytes) return;
  if (b.h_pinned) cudaFreeHost(b.h_pinned);
  if (cudaHostAlloc((void **)&b.h_pinned, bytes, cudaHostAllocDefault) != cudaSuccess)
    throw std::runtime_error("cudaHostAlloc 失败");
  b.pinned_cap = bytes;
}

void ensure_out_u8(size_t bytes)
{
  auto & b = buffers();
  if (b.out_cap >= bytes) return;
  if (b.d_out_u8) cudaFree(b.d_out_u8);
  if (cudaMalloc(&b.d_out_u8, bytes) != cudaSuccess)
    throw std::runtime_error("cudaMalloc(out) 失败");
  b.out_cap = bytes;
}

// ---------------- 1. letterbox + RGB/CHW/归一化 ----------------
__device__ __forceinline__ unsigned char clamp_uc(int v)
{
  return (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

// src: 3 通道 u8，行跨度 src_stride 字节；dst: 3 个 dw*dh 平面
__global__ void letterbox_chw_kernel(
  const unsigned char * __restrict__ src, int sw, int sh, int src_stride, int ow, int oh,
  float * __restrict__ dst, int dw, int dh)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dw || y >= dh) return;

  const int plane = dw * dh;
  float r = 0.f, g = 0.f, b = 0.f;

  if (x < ow && y < oh) {
    // 与 cv::resize(INTER_LINEAR) 一致的映射关系
    float fx = (float)sw / (float)ow;
    float fy = (float)sh / (float)oh;
    float sx = (x + 0.5f) * fx - 0.5f;
    float sy = (y + 0.5f) * fy - 0.5f;

    int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
    float ax = sx - x0, ay = sy - y0;
    int x1 = x0 + 1, y1 = y0 + 1;

    x0 = max(0, min(sw - 1, x0));
    x1 = max(0, min(sw - 1, x1));
    y0 = max(0, min(sh - 1, y0));
    y1 = max(0, min(sh - 1, y1));

    const unsigned char * p00 = src + (size_t)y0 * src_stride + x0 * 3;
    const unsigned char * p01 = src + (size_t)y0 * src_stride + x1 * 3;
    const unsigned char * p10 = src + (size_t)y1 * src_stride + x0 * 3;
    const unsigned char * p11 = src + (size_t)y1 * src_stride + x1 * 3;

    float v[3];
    for (int c = 0; c < 3; ++c) {
      float top = p00[c] * (1.f - ax) + p01[c] * ax;
      float bot = p10[c] * (1.f - ax) + p11[c] * ax;
      v[c] = (top * (1.f - ay) + bot * ay) / 255.f;
    }
    // 相机输出的 3 通道图在本仓库里按 BGR 语义喂给网络，模型吃 RGB => 交换 0 和 2
    b = v[0];
    g = v[1];
    r = v[2];
  }

  const size_t idx = (size_t)y * dw + x;
  dst[idx] = r;
  dst[plane + idx] = g;
  dst[2 * plane + idx] = b;
}

// ---------------- 2. Bayer RG8 -> RGB8（双线性去马赛克） ----------------
__device__ __forceinline__ int at(
  const unsigned char * src, int stride, int w, int h, int x, int y)
{
  x = max(0, min(w - 1, x));
  y = max(0, min(h - 1, y));
  return src[(size_t)y * stride + x];
}

__global__ void bayer_rg_kernel(
  const unsigned char * __restrict__ src, int w, int h, int stride,
  unsigned char * __restrict__ dst, int dst_stride)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;

  const int p = at(src, stride, w, h, x, y);
  int R, G, B;
  const bool even_y = (y % 2) == 0;
  const bool even_x = (x % 2) == 0;

  const int l = at(src, stride, w, h, x - 1, y);
  const int r = at(src, stride, w, h, x + 1, y);
  const int u = at(src, stride, w, h, x, y - 1);
  const int d = at(src, stride, w, h, x, y + 1);
  const int ul = at(src, stride, w, h, x - 1, y - 1);
  const int ur = at(src, stride, w, h, x + 1, y - 1);
  const int dl = at(src, stride, w, h, x - 1, y + 1);
  const int dr = at(src, stride, w, h, x + 1, y + 1);

  const int cross = (l + r + u + d + 2) >> 2;
  const int diag = (ul + ur + dl + dr + 2) >> 2;
  const int hor = (l + r + 1) >> 1;
  const int ver = (u + d + 1) >> 1;

  if (even_y && even_x) {  // R 像素
    R = p;
    G = cross;
    B = diag;
  } else if (!even_y && !even_x) {  // B 像素
    R = diag;
    G = cross;
    B = p;
  } else if (even_y) {  // G 像素（左右为 R）
    R = hor;
    G = p;
    B = ver;
  } else {  // G 像素（上下为 R）
    R = ver;
    G = p;
    B = hor;
  }

  unsigned char * out = dst + (size_t)y * dst_stride + (size_t)x * 3;
  out[0] = clamp_uc(R);
  out[1] = clamp_uc(G);
  out[2] = clamp_uc(B);
}
}  // namespace

bool available()
{
  static int count = -1;
  if (count < 0) {
    if (cudaFree(0) != cudaSuccess) {
      count = 0;
    } else {
      int devices = 0;
      cudaGetDeviceCount(&devices);
      count = devices;
    }
  }
  return count > 0;
}

void letterbox_to_chw(const cv::Mat & src, float * d_dst, int dw, int dh, cudaStream_t stream)
{
  CV_Assert(src.type() == CV_8UC3);
  const int sw = src.cols, sh = src.rows;

  const size_t row_bytes = (size_t)sw * 3;
  ensure_u8(row_bytes * sh);
  ensure_pinned(row_bytes * sh);
  auto & b = buffers();

  // 拷到 page-locked 内存（CPU memcpy 很快），再异步上传，避免 pageable 拷贝阻塞 stream
  if ((size_t)src.step == row_bytes) {
    memcpy(b.h_pinned, src.data, row_bytes * sh);
  } else {
    for (int y = 0; y < sh; ++y)
      memcpy(b.h_pinned + y * row_bytes, src.data + (size_t)y * src.step, row_bytes);
  }
  cudaMemcpy2DAsync(
    b.d_u8, row_bytes, b.h_pinned, row_bytes, row_bytes, sh, cudaMemcpyHostToDevice, stream);

  const float scale = std::min((float)dw / sw, (float)dh / sh);
  const int ow = (int)(sw * scale), oh = (int)(sh * scale);

  dim3 block(16, 16);
  dim3 grid((dw + block.x - 1) / block.x, (dh + block.y - 1) / block.y);
  letterbox_chw_kernel<<<grid, block, 0, stream>>>(
    b.d_u8, sw, sh, sw * 3, ow, oh, d_dst, dw, dh);
}

void bayer_rg_to_rgb(const cv::Mat & bayer, cv::Mat & rgb)
{
  CV_Assert(bayer.type() == CV_8UC1);
  const int w = bayer.cols, h = bayer.rows;

  ensure_u8((size_t)bayer.step * h);
  ensure_out_u8((size_t)w * 3 * h);
  auto & b = buffers();

  cudaMemcpy2DAsync(
    b.d_u8, (size_t)w, bayer.data, (size_t)bayer.step, (size_t)w, h, cudaMemcpyHostToDevice, 0);

  dim3 block(32, 8);
  dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
  bayer_rg_kernel<<<grid, block>>>(b.d_u8, w, h, w, b.d_out_u8, w * 3);

  rgb.create(h, w, CV_8UC3);
  cudaMemcpy2DAsync(
    rgb.data, (size_t)rgb.step, b.d_out_u8, (size_t)w * 3, (size_t)w * 3, h,
    cudaMemcpyDeviceToHost, 0);
  cudaStreamSynchronize(0);
}

}  // namespace auto_aim::cuda_ops
