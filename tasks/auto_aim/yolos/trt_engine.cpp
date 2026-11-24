#include "trt_engine.hpp"

#include <cuda_runtime_api.h>
#include <NvInferPlugin.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "tools/logger.hpp"

namespace auto_aim
{
namespace
{
class TrtLogger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * msg) noexcept override
  {
    if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR)
      tools::logger()->error("[TRT] {}", msg);
    else if (severity == Severity::kWARNING)
      tools::logger()->warn("[TRT] {}", msg);
  }
};

std::vector<char> read_file(const std::string & path)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("无法打开 TensorRT engine: " + path);
  f.seekg(0, std::ios::end);
  auto size = static_cast<size_t>(f.tellg());
  f.seekg(0, std::ios::beg);
  std::vector<char> buf(size);
  f.read(buf.data(), static_cast<std::streamsize>(size));
  return buf;
}
}  // namespace

struct TrtEngine::Impl
{
  struct Out
  {
    std::string name;
    std::vector<int> shape;
    size_t elems = 0;
    bool is_int32 = false;
    void * d_ptr = nullptr;
    std::vector<float> host_f;
    std::vector<int32_t> host_i;
  };

  TrtLogger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  cudaStream_t stream = nullptr;

  // CUDA Graph：把 enqueueV3 录成图，之后每帧一次 cudaGraphLaunch，省掉逐层 launch 开销。
  // 同样的 kernel、同样的地址，结果逐位一致；捕获失败自动退回 enqueueV3。
  // 环境变量 SP_TRT_CUDA_GRAPH=0 可关闭。
  bool graph_enabled = true;
  int graph_state = 0;  // 0 未尝试 1 可用 -1 失败
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;

  void enqueue()
  {
    if (graph_enabled && graph_state == 0) {
      // 第一次先正常跑一遍（TRT 要求捕获前至少 enqueue 过一次，做完内部初始化）
      if (!context->enqueueV3(stream)) throw std::runtime_error("enqueueV3 失败");
      cudaStreamSynchronize(stream);
      graph_state = -1;
      if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
        const bool ok = context->enqueueV3(stream);
        const auto end = cudaStreamEndCapture(stream, &graph);
        if (ok && end == cudaSuccess && graph &&
            cudaGraphInstantiate(&graph_exec, graph, 0) == cudaSuccess)
          graph_state = 1;
      }
      cudaGetLastError();  // 清掉捕获失败留下的错误码
      tools::logger()->info("[TRT] CUDA Graph {}", graph_state == 1 ? "已启用" : "捕获失败，使用 enqueueV3");
      return;  // 这一帧已经算过了
    }
    if (graph_state == 1) {
      if (cudaGraphLaunch(graph_exec, stream) != cudaSuccess) throw std::runtime_error("cudaGraphLaunch 失败");
      return;
    }
    if (!context->enqueueV3(stream)) throw std::runtime_error("enqueueV3 失败");
  }

  std::string input_name;
  size_t input_elems = 0;
  void * d_input = nullptr;
  std::vector<Out> outs;

  Out & out(int i)
  {
    if (i < 0 || i >= static_cast<int>(outs.size())) throw std::runtime_error("输出下标越界");
    return outs[static_cast<size_t>(i)];
  }
};

TrtEngine::TrtEngine(const std::string & engine_path) : impl_(std::make_unique<Impl>())
{
  auto blob = read_file(engine_path);

  // 注册官方插件（EfficientNMS_TRT 等），否则反序列化带插件的 engine 会失败
  if (!initLibNvInferPlugins(&impl_->logger, ""))
    tools::logger()->warn("[TRT] initLibNvInferPlugins 失败");

  impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
  if (!impl_->runtime) throw std::runtime_error("createInferRuntime 失败");

  impl_->engine.reset(impl_->runtime->deserializeCudaEngine(blob.data(), blob.size()));
  if (!impl_->engine) throw std::runtime_error("deserializeCudaEngine 失败: " + engine_path);

  impl_->context.reset(impl_->engine->createExecutionContext());
  if (!impl_->context) throw std::runtime_error("createExecutionContext 失败");

  for (int i = 0; i < impl_->engine->getNbIOTensors(); ++i) {
    const char * name = impl_->engine->getIOTensorName(i);
    auto mode = impl_->engine->getTensorIOMode(name);
    auto dims = impl_->engine->getTensorShape(name);
    auto dtype = impl_->engine->getTensorDataType(name);

    int64_t n = 1;
    std::vector<int> shape;
    for (int d = 0; d < dims.nbDims; ++d) {
      n *= dims.d[d];
      shape.push_back(static_cast<int>(dims.d[d]));
    }

    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      impl_->input_name = name;
      impl_->input_elems = static_cast<size_t>(n);
      if (cudaMalloc(&impl_->d_input, impl_->input_elems * sizeof(float)) != cudaSuccess)
        throw std::runtime_error("cudaMalloc(input) 失败");
      impl_->context->setTensorAddress(impl_->input_name.c_str(), impl_->d_input);
    } else {
      Impl::Out o;
      o.name = name;
      o.shape = shape;
      o.elems = static_cast<size_t>(n);
      o.is_int32 = (dtype == nvinfer1::DataType::kINT32);
      const size_t bytes = o.elems * (o.is_int32 ? sizeof(int32_t) : sizeof(float));
      if (cudaMalloc(&o.d_ptr, bytes) != cudaSuccess) throw std::runtime_error("cudaMalloc(out) 失败");
      impl_->context->setTensorAddress(o.name.c_str(), o.d_ptr);
      impl_->outs.push_back(std::move(o));
    }
  }
  if (impl_->input_name.empty() || impl_->outs.empty())
    throw std::runtime_error("engine 缺少输入或输出张量");

  if (cudaStreamCreate(&impl_->stream) != cudaSuccess)
    throw std::runtime_error("cudaStreamCreate 失败");
  if (const char * e = std::getenv("SP_TRT_CUDA_GRAPH")) impl_->graph_enabled = std::string(e) != "0";

  std::string desc;
  for (const auto & o : impl_->outs) {
    desc += " " + o.name + "(";
    for (size_t k = 0; k < o.shape.size(); ++k) desc += (k ? "x" : "") + std::to_string(o.shape[k]);
    desc += o.is_int32 ? ",int32)" : ",f32)";
  }
  tools::logger()->info(
    "[TRT] engine loaded: {} in={} ({} elems) out:{}", engine_path, impl_->input_name,
    impl_->input_elems, desc);
}

TrtEngine::~TrtEngine()
{
  if (!impl_) return;
  if (impl_->graph_exec) cudaGraphExecDestroy(impl_->graph_exec);
  if (impl_->graph) cudaGraphDestroy(impl_->graph);
  for (auto & o : impl_->outs)
    if (o.d_ptr) cudaFree(o.d_ptr);
  if (impl_->d_input) cudaFree(impl_->d_input);
  if (impl_->stream) cudaStreamDestroy(impl_->stream);
}

float * TrtEngine::input_device_ptr() const { return static_cast<float *>(impl_->d_input); }

cudaStream_t TrtEngine::stream() const { return impl_->stream; }

int TrtEngine::nb_outputs() const { return static_cast<int>(impl_->outs.size()); }

const std::vector<int> & TrtEngine::output_shape(int i) const { return impl_->out(i).shape; }

bool TrtEngine::output_is_int32(int i) const { return impl_->out(i).is_int32; }

const std::vector<float> & TrtEngine::output_float(int i)
{
  auto & o = impl_->out(i);
  if (o.is_int32) throw std::runtime_error("该输出是 int32");
  return o.host_f;
}

const std::vector<int32_t> & TrtEngine::output_int(int i)
{
  auto & o = impl_->out(i);
  if (!o.is_int32) throw std::runtime_error("该输出不是 int32");
  return o.host_i;
}

void TrtEngine::infer_from_device_all()
{
  impl_->enqueue();

  for (auto & o : impl_->outs) {
    if (o.is_int32) {
      o.host_i.resize(o.elems);
      cudaMemcpyAsync(
        o.host_i.data(), o.d_ptr, o.elems * sizeof(int32_t), cudaMemcpyDeviceToHost,
        impl_->stream);
    } else {
      o.host_f.resize(o.elems);
      cudaMemcpyAsync(
        o.host_f.data(), o.d_ptr, o.elems * sizeof(float), cudaMemcpyDeviceToHost, impl_->stream);
    }
  }
  cudaStreamSynchronize(impl_->stream);
}

void TrtEngine::infer_from_device(std::vector<float> & out)
{
  impl_->enqueue();

  auto & o = impl_->out(0);
  out.resize(o.elems);
  cudaMemcpyAsync(
    out.data(), o.d_ptr, o.elems * sizeof(float), cudaMemcpyDeviceToHost, impl_->stream);
  cudaStreamSynchronize(impl_->stream);
}

void TrtEngine::infer(const float * input, size_t input_elems, std::vector<float> & out)
{
  if (input_elems != impl_->input_elems) throw std::runtime_error("输入尺寸与 engine 不匹配");

  cudaMemcpyAsync(
    impl_->d_input, input, input_elems * sizeof(float), cudaMemcpyHostToDevice, impl_->stream);
  infer_from_device(out);
}

void bgr_to_rgb_chw(const cv::Mat & bgr, float * dst)
{
  const int h = bgr.rows, w = bgr.cols;
  CV_Assert(bgr.type() == CV_8UC3);
  const size_t plane = static_cast<size_t>(h) * w;

  cv::Mat planes[3];
  cv::split(bgr, planes);  // B, G, R

  for (int c = 0; c < 3; ++c) {
    cv::Mat src = planes[2 - c];  // 0->R, 1->G, 2->B
    cv::Mat o(h, w, CV_32F, dst + c * plane);
    src.convertTo(o, CV_32F, 1.0 / 255.0);
  }
}

}  // namespace auto_aim
