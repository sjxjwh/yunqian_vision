#ifndef TOOLS__RUNTIME_PARAMS_HPP
#define TOOLS__RUNTIME_PARAMS_HPP

#include <yaml-cpp/yaml.h>

#include <functional>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace tools
{
/// 运行期可调参数总表：各模块在构造时把自己的可调量注册进来，
/// 弹道闭环 / 网页 / 调试工具通过键值对读写，改完立即生效（不用重启）
class RuntimeParams
{
public:
  struct Item
  {
    std::string key;
    std::string desc;
    double lo = 0, hi = 1;
    std::function<double()> get;
    std::function<void(double)> set;
    std::string yaml_key;  // 同名 yaml 键，reload 时从这里取值刷新（空 = 不参与）
    double step = 0;       // 网页滑条步长（0 = 前端按量程自动取）
  };

  static RuntimeParams & instance()
  {
    static RuntimeParams rp;
    return rp;
  }

  void add(
    const std::string & key, const std::string & desc, double lo, double hi,
    std::function<double()> get, std::function<void(double)> set,
    const std::string & yaml_key = "", double step = 0)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    items_[key] = Item{key, desc, lo, hi, std::move(get), std::move(set), yaml_key, step};
  }

  /// 给已注册参数指定步长（整数量用 1），未注册的键忽略
  void set_step(const std::string & key, double step)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    auto it = items_.find(key);
    if (it != items_.end()) it->second.step = step;
  }

  /// 重新读 yaml，把有同名 yaml 键的参数刷一遍
  int reload_from_yaml(const std::string & path)
  {
    YAML::Node yaml;
    try {
      yaml = YAML::LoadFile(path);
    } catch (const std::exception &) {
      return -1;
    }
    std::vector<std::pair<std::function<void(double)>, double>> todo;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      for (const auto & [k, it] : items_) {
        if (it.yaml_key.empty()) continue;
        auto node = yaml[it.yaml_key];
        if (!node || !node.IsScalar()) continue;
        try {
          todo.emplace_back(it.set, node.as<double>());
        } catch (const std::exception &) {
        }
      }
    }
    for (auto & [set, v] : todo) set(v);
    return static_cast<int>(todo.size());
  }

  bool set(const std::string & key, double v)
  {
    Item item;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      auto it = items_.find(key);
      if (it == items_.end()) return false;
      item = it->second;
    }
    if (v < item.lo) v = item.lo;
    if (v > item.hi) v = item.hi;
    item.set(v);
    return true;
  }

  /// 读当前值（键不存在返回空）
  std::optional<double> get(const std::string & key) const
  {
    std::lock_guard<std::mutex> lk(mtx_);
    auto it = items_.find(key);
    if (it == items_.end()) return std::nullopt;
    return it->second.get();
  }

  nlohmann::json to_json() const
  {
    nlohmann::json arr = nlohmann::json::array();
    std::lock_guard<std::mutex> lk(mtx_);
    for (const auto & [k, it] : items_)
      arr.push_back(
        {{"key", it.key},
         {"desc", it.desc},
         {"min", it.lo},
         {"max", it.hi},
         {"value", it.get()},
         {"step", it.step},
         {"yaml", it.yaml_key}});
    return arr;
  }

  bool known(const std::string & key) const
  {
    std::lock_guard<std::mutex> lk(mtx_);
    return items_.count(key) > 0;
  }

private:
  mutable std::mutex mtx_;
  std::map<std::string, Item> items_;
};

}  // namespace tools

#endif  // TOOLS__RUNTIME_PARAMS_HPP
