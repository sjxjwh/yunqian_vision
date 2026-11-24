#include "yolov5.hpp"

#include "tasks/auto_aim/armor_smoother.hpp"
#include "tools/runtime_params.hpp"

#include <fmt/chrono.h>

#include <cmath>
#include <yaml-cpp/yaml.h>

#include <filesystem>

#ifdef SP_WITH_CUDA_PREPROCESS
#include "cuda_preprocess.hpp"
#endif
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{
YOLOV5::YOLOV5(const std::string & config_path, bool debug)
: debug_(debug), detector_(config_path, false)
{
  auto yaml = YAML::LoadFile(config_path);

  model_path_ = yaml["yolov5_model_path"].as<std::string>();
  device_ = yaml["device"].as<std::string>();
  binary_threshold_ = yaml["threshold"].as<double>();
  min_confidence_ = yaml["min_confidence"].as<double>();
  int x = 0, y = 0, width = 0, height = 0;
  x = yaml["roi"]["x"].as<int>();
  y = yaml["roi"]["y"].as<int>();
  width = yaml["roi"]["width"].as<int>();
  height = yaml["roi"]["height"].as<int>();
  use_roi_ = yaml["use_roi"].as<bool>();
  use_traditional_ = yaml["use_traditional"].as<bool>();
  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(x, y);

  save_path_ = "imgs";
  std::filesystem::create_directory(save_path_);

  logit_threshold_ = std::log(score_threshold_ / (1.0f - score_threshold_));

  const int vote_frames = yaml["vote_frames"] ? yaml["vote_frames"].as<int>() : 0;

  // 运行期可调参数
  auto & rp = tools::RuntimeParams::instance();
  rp.add("yolo.min_confidence", "最小置信度(后过滤)", 0.1, 0.99, [this] { return min_confidence_; },
         [this](double v) { min_confidence_ = v; }, "min_confidence");
  rp.add("yolo.vote_frames", "识别平滑帧数(0=关)", 0, 15,
         [this] { return double(smoother_.frames()); },
         [this](double v) { smoother_.set_frames(int(v)); }, "vote_frames");
  rp.add("detector.threshold", "传统识别二值化阈值", 0, 255, [this] { return binary_threshold_; },
         [this](double v) { binary_threshold_ = v; }, "threshold");
  smoother_ = ArmorSmoother(vote_frames);
  if (vote_frames > 1) tools::logger()->info("识别平滑: 最近 {} 帧多数票（color/name）", vote_frames);

  use_cuda_ = yaml["use_cuda_preprocess"] ? yaml["use_cuda_preprocess"].as<bool>() : true;
  profile_ = yaml["profile"] ? yaml["profile"].as<bool>() : false;
#ifdef SP_WITH_CUDA_PREPROCESS
  tools::logger()->info(
    "preprocess: cuda={} (编译已启用, 设备可用={}), profile={}", use_cuda_,
    auto_aim::cuda_ops::available(), profile_);
#else
  tools::logger()->info("preprocess: cpu (未编译 CUDA), profile={}", profile_);
#endif

  // 可选：用 TensorRT engine 推理（Jetson 上更快）
  auto trt_path = yaml["trt_engine_path"] ? yaml["trt_engine_path"].as<std::string>() : "";
  if (!trt_path.empty()) {
#ifdef SP_WITH_TENSORRT
    if (!std::filesystem::exists(trt_path)) {
      tools::logger()->warn("trt_engine_path 不存在: {}，回退 OpenVINO", trt_path);
    } else {
      trt_ = std::make_unique<TrtEngine>(trt_path);
      tools::logger()->info("YOLOV5 使用 TensorRT 推理: {}", trt_path);
    }
#else
    tools::logger()->warn("编译时未启用 TensorRT，忽略 trt_engine_path，回退 OpenVINO");
#endif
  }

#ifdef SP_WITH_TENSORRT
  if (trt_) return;  // TensorRT 路径不需要 OpenVINO
#endif

  auto model = core_.read_model(model_path_);
  ov::preprocess::PrePostProcessor ppp(model);
  auto & input = ppp.input();

  input.tensor()
    .set_element_type(ov::element::u8)
    .set_shape({1, 640, 640, 3})
    .set_layout("NHWC")
    .set_color_format(ov::preprocess::ColorFormat::BGR);

  input.model().set_layout("NCHW");

  input.preprocess()
    .convert_element_type(ov::element::f32)
    .convert_color(ov::preprocess::ColorFormat::RGB)
    .scale(255.0);

  model = ppp.build();
  ov::AnyMap ov_props;
  ov_props[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
  if (device_ == "CPU") {
    ov_props[ov::hint::scheduling_core_type.name()] = ov::hint::SchedulingCoreType::PCORE_ONLY;
    if (yaml["ov_num_threads"]) {
      ov_props[ov::inference_num_threads.name()] = yaml["ov_num_threads"].as<int>();
    }
  }
  try {
    compiled_model_ = core_.compile_model(model, device_, ov_props);
  } catch (const std::exception &) {
    compiled_model_ = core_.compile_model(
      model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
  }
  infer_request_ = compiled_model_.create_infer_request();
}

std::list<Armor> YOLOV5::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::list<Armor>();
  }

  cv::Mat bgr_img;
  if (use_roi_) {
    if (roi_.width == -1) {  // -1 表示该维度不裁切
      roi_.width = raw_img.cols;
    }
    if (roi_.height == -1) {  // -1 表示该维度不裁切
      roi_.height = raw_img.rows;
    }
    bgr_img = raw_img(roi_);
  } else {
    bgr_img = raw_img;
  }

  auto x_scale = static_cast<double>(640) / bgr_img.rows;
  auto y_scale = static_cast<double>(640) / bgr_img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(bgr_img.rows * scale);
  auto w = static_cast<int>(bgr_img.cols * scale);


#if defined(SP_WITH_TENSORRT) && defined(SP_WITH_CUDA_PREPROCESS)
  if (trt_ && use_cuda_ && cuda_ops::available()) {
    const auto t0 = std::chrono::steady_clock::now();
    cuda_ops::letterbox_to_chw(bgr_img, trt_->input_device_ptr(), 640, 640, trt_->stream());

    // 引擎里已经接了官方 EfficientNMS_TRT：直接吐 [num_dets, rows[22]]
    if (trt_->nb_outputs() >= 2) {
      if (profile_) cudaStreamSynchronize(trt_->stream());
      const auto t1 = std::chrono::steady_clock::now();
      trt_->infer_from_device_all();
      const auto t2 = std::chrono::steady_clock::now();

      const auto & num = trt_->output_int(0);
      const int n = num.empty() ? 0 : static_cast<int>(num[0]);
      const int cols = trt_->output_shape(1).back();  // [1,1,N,22] -> 22
      auto armors = parse_nms(scale, trt_->output_float(1), n, cols, raw_img, frame_count);

      const auto t3 = std::chrono::steady_clock::now();
      if (profile_) profile_report(t0, t1, t2, t3);
      return armors;
    }

    if (profile_) cudaStreamSynchronize(trt_->stream());  // 仅测量用：把预处理和推理分开计时
    std::vector<float> out;
    const auto t1 = std::chrono::steady_clock::now();
    trt_->infer_from_device(out);
    const auto t2 = std::chrono::steady_clock::now();

    auto & shape = trt_->output_shape();
    cv::Mat output(shape[1], shape[2], CV_32F, out.data());
    auto armors = parse(scale, output, raw_img, frame_count);
    const auto t3 = std::chrono::steady_clock::now();

    if (profile_) profile_report(t0, t1, t2, t3);
    return armors;
  }
#endif

  // CPU letterbox：只有 CPU-TensorRT / OpenVINO 路径用得到（CUDA 路径在 GPU 上做，这里以前白算一遍）
  auto input = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(bgr_img, input(roi), {w, h});

#ifdef SP_WITH_TENSORRT
  if (trt_) {
    // BGR u8 -> RGB float32 NCHW / 255（直接写进复用缓冲区，省掉 blobFromImage 的额外拷贝）
    if (trt_input_.empty()) trt_input_.resize(static_cast<size_t>(3) * 640 * 640);
    bgr_to_rgb_chw(input, trt_input_.data());

    std::vector<float> out;
    trt_->infer(trt_input_.data(), trt_input_.size(), out);

    auto & shape = trt_->output_shape();
    cv::Mat output(shape[1], shape[2], CV_32F, out.data());
    return parse(scale, output, raw_img, frame_count);
  }
#endif

  ov::Tensor input_tensor(ov::element::u8, {1, 640, 640, 3}, input.data);

  // infer
  infer_request_.set_input_tensor(input_tensor);
  infer_request_.infer();

  // postprocess
  auto output_tensor = infer_request_.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());

  return parse(scale, output, raw_img, frame_count);
}

std::list<Armor> YOLOV5::parse(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  // for each row: 4 个关键点(8) + objectness(1) + color(4) + num(9) = 22
  std::vector<int> color_ids, num_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> armors_key_points;
  boxes.reserve(16);
  confidences.reserve(16);
  armors_key_points.reserve(16);
  color_ids.reserve(16);
  num_ids.reserve(16);

  const float inv_scale = 1.0f / (float)scale;
  const float * rows = output.ptr<float>(0);
  const int cols = output.cols;
  int passed = 0;
  for (int r = 0; r < output.rows; r++) {
    const float * row = rows + (size_t)r * cols;

    // 直接用 sigmoid 的比较形式，省掉 exp：sigmoid(x) >= thr  <=>  x >= ln(thr/(1-thr))
    if (row[8] < logit_threshold_) continue;
    ++passed;

    // 颜色(9..12)、类别(13..21) 的 argmax（裸指针，不用 cv::Mat 行切片）
    int color_id = 0, class_id = 0;
    float best_color = row[9], best_class = row[13];
    for (int c = 1; c < 4; c++)
      if (row[9 + c] > best_color) { best_color = row[9 + c]; color_id = c; }
    for (int c = 1; c < 9; c++)
      if (row[13 + c] > best_class) { best_class = row[13 + c]; class_id = c; }

    std::vector<cv::Point2f> armor_key_points;
    armor_key_points.reserve(4);
    armor_key_points.push_back({row[0] * inv_scale, row[1] * inv_scale});
    armor_key_points.push_back({row[6] * inv_scale, row[7] * inv_scale});
    armor_key_points.push_back({row[4] * inv_scale, row[5] * inv_scale});
    armor_key_points.push_back({row[2] * inv_scale, row[3] * inv_scale});

    float min_x = armor_key_points[0].x, max_x = min_x;
    float min_y = armor_key_points[0].y, max_y = min_y;
    for (int i = 1; i < 4; i++) {
      const auto & p = armor_key_points[i];
      if (p.x < min_x) min_x = p.x;
      if (p.x > max_x) max_x = p.x;
      if (p.y < min_y) min_y = p.y;
      if (p.y > max_y) max_y = p.y;
    }
    cv::Rect rect(min_x, min_y, max_x - min_x, max_y - min_y);

    color_ids.emplace_back(color_id);
    num_ids.emplace_back(class_id);
    boxes.emplace_back(rect);
    confidences.emplace_back(1.0f / (1.0f + std::exp(-row[8])));
    armors_key_points.emplace_back(armor_key_points);
  }

  if (profile_) {
    static int cnt = 0;
    if ((cnt++ % 100) == 0)
      tools::logger()->info("[parse] rows={} passed={}", output.rows, passed);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::list<Armor> armors;
  for (const auto & i : indices) {
    if (use_roi_) {
      armors.emplace_back(
        color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i], offset_);
    } else {
      armors.emplace_back(color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i]);
    }
  }

  smoother_.apply(armors);  // 时间平滑（与 NMS 路径一致）

  tmp_img_ = bgr_img;
  for (auto it = armors.begin(); it != armors.end();) {
    if (!check_name(*it)) {
      it = armors.erase(it);
      continue;
    }

    if (!check_type(*it)) {
      it = armors.erase(it);
      continue;
    }
    // 使用传统方法二次矫正角点
    if (use_traditional_) detector_.detect(*it, bgr_img);

    it->center_norm = get_center_norm(bgr_img, it->center);
    ++it;
  }

  if (debug_) draw_detections(bgr_img, armors, frame_count);

  return armors;
}

bool YOLOV5::check_name(const Armor & armor) const
{
  auto name_ok = armor.name != ArmorName::not_armor;
  auto confidence_ok = armor.confidence > min_confidence_;

  // 保存不确定的图案，用于神经网络的迭代
  // if (name_ok && !confidence_ok) save(armor);

  return name_ok && confidence_ok;
}

bool YOLOV5::check_type(const Armor & armor) const
{
  auto name_ok = (armor.type == ArmorType::small)
                   ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
                   : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
                      armor.name != ArmorName::outpost);

  // 保存异常的图案，用于神经网络的迭代
  // if (!name_ok) save(armor);

  return name_ok;
}

cv::Point2f YOLOV5::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  auto h = bgr_img.rows;
  auto w = bgr_img.cols;
  return {center.x / w, center.y / h};
}

void YOLOV5::draw_detections(
  const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  auto detection = img.clone();
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  for (const auto & armor : armors) {
    auto info = fmt::format(
      "{:.2f} {} {} {}", armor.confidence, COLORS[armor.color], ARMOR_NAMES[armor.name],
      ARMOR_TYPES[armor.type]);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    tools::draw_text(detection, info, armor.center, {0, 255, 0});
  }

  if (use_roi_) {
    cv::Scalar green(0, 255, 0);
    cv::rectangle(detection, roi_, green, 2);
  }
  cv::resize(detection, detection, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
  cv::imshow("detection", detection);
}

void YOLOV5::save(const Armor & armor) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, armor.name, file_name);
  cv::imwrite(img_path, tmp_img_);
}

double YOLOV5::sigmoid(double x)
{
  if (x > 0)
    return 1.0 / (1.0 + exp(-x));
  else
    return exp(x) / (1.0 + exp(x));
}

std::list<Armor> YOLOV5::parse_nms(
  double scale, const std::vector<float> & rows, int n, int cols, const cv::Mat & bgr_img,
  int frame_count)
{
  std::list<Armor> armors;
  const float inv_scale = 1.0f / static_cast<float>(scale);
  if (profile_) {
    static int cnt = 0;
    if ((cnt++ % 100) == 0)
      tools::logger()->info("[nms] num_dets={} cols={}", n, cols);
  }

  for (int i = 0; i < n; ++i) {
    const float * row = rows.data() + static_cast<size_t>(i) * cols;

    const float score = 1.0f / (1.0f + std::exp(-row[8]));
    if (score < score_threshold_) continue;

    int color_id = 0, class_id = 0;
    float best_color = row[9], best_class = row[13];
    for (int c = 1; c < 4; c++)
      if (row[9 + c] > best_color) {
        best_color = row[9 + c];
        color_id = c;
      }
    for (int c = 1; c < 9; c++)
      if (row[13 + c] > best_class) {
        best_class = row[13 + c];
        class_id = c;
      }

    std::vector<cv::Point2f> kp;
    kp.reserve(4);
    kp.push_back({row[0] * inv_scale, row[1] * inv_scale});
    kp.push_back({row[6] * inv_scale, row[7] * inv_scale});
    kp.push_back({row[4] * inv_scale, row[5] * inv_scale});
    kp.push_back({row[2] * inv_scale, row[3] * inv_scale});

    float min_x = kp[0].x, max_x = kp[0].x, min_y = kp[0].y, max_y = kp[0].y;
    for (int k = 1; k < 4; k++) {
      if (kp[k].x < min_x) min_x = kp[k].x;
      if (kp[k].x > max_x) max_x = kp[k].x;
      if (kp[k].y < min_y) min_y = kp[k].y;
      if (kp[k].y > max_y) max_y = kp[k].y;
    }

    cv::Rect rect(min_x, min_y, max_x - min_x, max_y - min_y);
    if (use_roi_) {
      armors.emplace_back(color_id, class_id, score, rect, kp, offset_);
    } else {
      armors.emplace_back(color_id, class_id, score, rect, kp);
    }
  }

  smoother_.apply(armors);  // 时间平滑：抑制单帧的 蓝/熄灭、号码 抖动

  tmp_img_ = bgr_img;
  for (auto it = armors.begin(); it != armors.end();) {
    if (!check_name(*it) || !check_type(*it)) {
      it = armors.erase(it);
      continue;
    }
    if (use_traditional_) detector_.detect(*it, bgr_img);
    it->center_norm = get_center_norm(bgr_img, it->center);
    ++it;
  }

  if (debug_) draw_detections(bgr_img, armors, frame_count);
  return armors;
}

void YOLOV5::profile_report(
  std::chrono::steady_clock::time_point t0, std::chrono::steady_clock::time_point t1,
  std::chrono::steady_clock::time_point t2, std::chrono::steady_clock::time_point t3)
{
  auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  profile_pre_ += ms(t0, t1);
  profile_infer_ += ms(t1, t2);
  profile_parse_ += ms(t2, t3);
  if (++profile_count_ % 50 != 0) return;
  tools::logger()->info(
    "[profile] preprocess {:.2f} ms | infer {:.2f} ms | parse {:.2f} ms | total {:.2f} ms",
    profile_pre_ / profile_count_, profile_infer_ / profile_count_,
    profile_parse_ / profile_count_,
    (profile_pre_ + profile_infer_ + profile_parse_) / profile_count_);
  profile_pre_ = profile_infer_ = profile_parse_ = 0;
  profile_count_ = 0;
}

std::list<Armor> YOLOV5::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, output, bgr_img, frame_count);
}

}  // namespace auto_aim