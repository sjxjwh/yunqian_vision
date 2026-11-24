#include "yolo11.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>

#ifdef SP_WITH_CUDA_PREPROCESS
#include "cuda_preprocess.hpp"
#endif
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{
YOLO11::YOLO11(const std::string & config_path, bool debug)
: debug_(debug), detector_(config_path, false)
{
  auto yaml = YAML::LoadFile(config_path);

  model_path_ = yaml["yolo11_model_path"].as<std::string>();
  device_ = yaml["device"].as<std::string>();
  binary_threshold_ = yaml["threshold"].as<double>();
  min_confidence_ = yaml["min_confidence"].as<double>();
  int x = 0, y = 0, width = 0, height = 0;
  x = yaml["roi"]["x"].as<int>();
  y = yaml["roi"]["y"].as<int>();
  width = yaml["roi"]["width"].as<int>();
  height = yaml["roi"]["height"].as<int>();
  use_roi_ = yaml["use_roi"].as<bool>();
  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(x, y);

  save_path_ = "imgs";
  std::filesystem::create_directory(save_path_);

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
      tools::logger()->info("YOLO11 使用 TensorRT 推理: {}", trt_path);
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

std::list<Armor> YOLO11::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::list<Armor>();
  }

  cv::Mat bgr_img;
  tmp_img_ = raw_img;
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
    auto armors = parse(scale, output, raw_img, frame_count);
    return armors;
  }
#endif

  ov::Tensor input_tensor(ov::element::u8, {1, 640, 640, 3}, input.data);

  /// infer
  infer_request_.set_input_tensor(input_tensor);
  infer_request_.infer();

  // postprocess
  auto output_tensor = infer_request_.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());

  return parse(scale, output, raw_img, frame_count);
}

std::list<Armor> YOLO11::parse(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{  // for each row: xywh + classess
  cv::transpose(output, output);

  std::vector<int> ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> armors_key_points;
  for (int r = 0; r < output.rows; r++) {
    auto xywh = output.row(r).colRange(0, 4);
    auto scores = output.row(r).colRange(4, 4 + class_num_);
    auto one_key_points = output.row(r).colRange(4 + class_num_, 50);

    std::vector<cv::Point2f> armor_key_points;

    double score;
    cv::Point max_point;
    cv::minMaxLoc(scores, nullptr, &score, nullptr, &max_point);

    if (score < score_threshold_) continue;

    auto x = xywh.at<float>(0);
    auto y = xywh.at<float>(1);
    auto w = xywh.at<float>(2);
    auto h = xywh.at<float>(3);
    auto left = static_cast<int>((x - 0.5 * w) / scale);
    auto top = static_cast<int>((y - 0.5 * h) / scale);
    auto width = static_cast<int>(w / scale);
    auto height = static_cast<int>(h / scale);

    for (int i = 0; i < 4; i++) {
      float x = one_key_points.at<float>(0, i * 2 + 0) / scale;
      float y = one_key_points.at<float>(0, i * 2 + 1) / scale;
      cv::Point2f kp = {x, y};
      armor_key_points.push_back(kp);
    }
    ids.emplace_back(max_point.x);
    confidences.emplace_back(score);
    boxes.emplace_back(left, top, width, height);
    armors_key_points.emplace_back(armor_key_points);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::list<Armor> armors;
  for (const auto & i : indices) {
    sort_keypoints(armors_key_points[i]);
    if (use_roi_) {
      armors.emplace_back(ids[i], confidences[i], boxes[i], armors_key_points[i], offset_);
    } else {
      armors.emplace_back(ids[i], confidences[i], boxes[i], armors_key_points[i]);
    }
  }

  for (auto it = armors.begin(); it != armors.end();) {
    if (!check_name(*it)) {
      it = armors.erase(it);
      continue;
    }

    if (!check_type(*it)) {
      it = armors.erase(it);
      continue;
    }

    it->center_norm = get_center_norm(bgr_img, it->center);
    ++it;
  }

  if (debug_) draw_detections(bgr_img, armors, frame_count);

  return armors;
}

bool YOLO11::check_name(const Armor & armor) const
{
  auto name_ok = armor.name != ArmorName::not_armor;
  auto confidence_ok = armor.confidence > min_confidence_;

  // 保存不确定的图案，用于神经网络的迭代
  // if (name_ok && !confidence_ok) save(armor);

  return name_ok && confidence_ok;
}

bool YOLO11::check_type(const Armor & armor) const
{
  auto name_ok = (armor.type == ArmorType::small)
                   ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
                   : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
                      armor.name != ArmorName::outpost);

  // 保存异常的图案，用于神经网络的迭代
  // if (!name_ok) save(armor);

  return name_ok;
}

cv::Point2f YOLO11::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  auto h = bgr_img.rows;
  auto w = bgr_img.cols;
  return {center.x / w, center.y / h};
}

void YOLO11::sort_keypoints(std::vector<cv::Point2f> & keypoints)
{
  if (keypoints.size() != 4) {
    std::cout << "beyond 4!!" << std::endl;
    return;
  }

  std::sort(keypoints.begin(), keypoints.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
    return a.y < b.y;
  });

  std::vector<cv::Point2f> top_points = {keypoints[0], keypoints[1]};
  std::vector<cv::Point2f> bottom_points = {keypoints[2], keypoints[3]};

  std::sort(top_points.begin(), top_points.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
    return a.x < b.x;
  });

  std::sort(
    bottom_points.begin(), bottom_points.end(),
    [](const cv::Point2f & a, const cv::Point2f & b) { return a.x < b.x; });

  keypoints[0] = top_points[0];     // top-left
  keypoints[1] = top_points[1];     // top-right
  keypoints[2] = bottom_points[1];  // bottom-right
  keypoints[3] = bottom_points[0];  // bottom-left
}

void YOLO11::draw_detections(
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

void YOLO11::save(const Armor & armor) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, armor.name, file_name);
  cv::imwrite(img_path, tmp_img_);
}

void YOLO11::profile_report(
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

std::list<Armor> YOLO11::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, output, bgr_img, frame_count);
}

}  // namespace auto_aim