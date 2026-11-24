#ifndef AUTO_AIM__ARMOR_SMOOTHER_HPP
#define AUTO_AIM__ARMOR_SMOOTHER_HPP

#include <algorithm>
#include <deque>
#include <list>
#include <map>
#include <utility>
#include <vector>

#include "armor.hpp"

namespace auto_aim
{
/// 逐目标的时间平滑：用 IoU 把本帧检测和上一帧的目标关联起来，
/// 用最近 N 帧的多数票决定 color / name，抑制单帧误分类
/// （实测主要问题是 蓝 ↔ 熄灭 抖动，以及偶尔 sentry ↔ 其他号码）。
class ArmorSmoother
{
public:
  explicit ArmorSmoother(int vote_frames = 5, int max_miss = 10)
  : vote_frames_(vote_frames), max_miss_(max_miss)
  {
  }

  bool enabled() const { return vote_frames_ > 1; }

  int frames() const { return vote_frames_; }

  void set_frames(int n)
  {
    vote_frames_ = n;
    if (n <= 1) slots_.clear();
  }

  void apply(std::list<Armor> & armors)
  {
    if (!enabled()) return;
    ++tick_;

    for (auto & a : armors) {
      Slot * s = match(a);
      if (!s) {
        slots_.emplace_back();
        s = &slots_.back();
      }
      s->box = a.box;
      s->last_seen = tick_;
      s->hist.emplace_back(a.color, a.name);
      while (static_cast<int>(s->hist.size()) > vote_frames_) s->hist.pop_front();

      const int n = static_cast<int>(s->hist.size());
      auto bc = majority_color(s->hist);
      if (bc.second * 2 > n) a.color = bc.first;
      auto bn = majority_name(s->hist);
      if (bn.second * 2 > n) a.name = bn.first;
    }

    // 长时间没出现的目标清掉
    for (auto it = slots_.begin(); it != slots_.end();) {
      if (tick_ - it->last_seen > max_miss_)
        it = slots_.erase(it);
      else
        ++it;
    }
  }

private:
  struct Slot
  {
    cv::Rect box;
    long last_seen = -1000;
    std::deque<std::pair<Color, ArmorName>> hist;
  };

  static double iou(const cv::Rect & a, const cv::Rect & b)
  {
    const int x0 = std::max(a.x, b.x), y0 = std::max(a.y, b.y);
    const int x1 = std::min(a.x + a.width, b.x + b.width);
    const int y1 = std::min(a.y + a.height, b.y + b.height);
    if (x1 <= x0 || y1 <= y0) return 0.0;
    const double inter = double(x1 - x0) * (y1 - y0);
    return inter / (double(a.width) * a.height + double(b.width) * b.height - inter);
  }

  Slot * match(const Armor & a)
  {
    Slot * best = nullptr;
    double best_iou = 0.4;  // 关联阈值，太小会串目标，太大跟不上抖动
    for (auto & s : slots_) {
      if (tick_ - s.last_seen > max_miss_) continue;
      const double v = iou(a.box, s.box);
      if (v > best_iou) {
        best_iou = v;
        best = &s;
      }
    }
    return best;
  }

  static std::pair<Color, int> majority_color(
    const std::deque<std::pair<Color, ArmorName>> & h)
  {
    std::map<int, int> cnt;
    for (const auto & [c, n] : h) cnt[static_cast<int>(c)]++;
    int best = 0, best_n = 0;
    for (const auto & [k, v] : cnt)
      if (v > best_n) {
        best_n = v;
        best = k;
      }
    return {static_cast<Color>(best), best_n};
  }

  static std::pair<ArmorName, int> majority_name(
    const std::deque<std::pair<Color, ArmorName>> & h)
  {
    std::map<int, int> cnt;
    for (const auto & [c, n] : h) cnt[static_cast<int>(n)]++;
    int best = 0, best_n = 0;
    for (const auto & [k, v] : cnt)
      if (v > best_n) {
        best_n = v;
        best = k;
      }
    return {static_cast<ArmorName>(best), best_n};
  }

  int vote_frames_;
  int max_miss_;
  long tick_ = 0;
  std::list<Slot> slots_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__ARMOR_SMOOTHER_HPP
